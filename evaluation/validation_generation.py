"""Generate over the validation split once, and read it three ways (arch v2.1 §8.2, §11.2, §5.3).

Checkpoint selection, calibration and the promotion gate all need the same
thing: the model's own generations over validation documents, scored against
their golden labels. Three separate generators would mean three definitions of
"correct", and a selector that scores by one definition picks a checkpoint the
gate, scoring by another, rejects. So there is one generator here and three
readers:

* :func:`score_generations` — the gate's metrics, through
  :func:`evaluation.run_eval.build_report`. Checkpoint selection reads its
  ``field_normalized_match``.
* :func:`calibration_samples` — ``(features, correct)`` per field, split into the
  calibration and threshold halves the dataset build stamped on each row.
* :func:`vllm_scorer` in :mod:`evaluation.checkpoint_eval` — the first reader
  applied per checkpoint, with the checkpoint as a decoder LoRA.
* :func:`gate_mirror` and :func:`error_tallies` — the finalists' pass read as
  the gate reads a release, and its errors counted by field, line, group and mode.

The selected finalist's generations are saved (:func:`write_generations`) and
calibration restores them onto their rows (:func:`restored_generation`) rather
than generating the split again.

The prompt is the row's own ``messages`` with the assistant turn removed, so the
model sees exactly what training rendered for that document (arch §7).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ValidationGeneration:
    """One validation row, generated and parsed."""

    row: dict[str, Any]
    golden: dict[str, Any]
    extraction: dict[str, Any] | None = None
    logprobs_by_path: dict[str, list[float]] = field(default_factory=dict)
    #: The same fields' spans (``inference_core.span_map.FieldSpan``, mapped
    #: ones, by values-view path): beside the logprobs, whether the printed
    #: text was written apart from the value and how close the runner-up token
    #: came - what calibration learns ``r_written`` and ``near_tie`` from
    #: (``calibration.features.build_document_features``). Empty where only
    #: logprobs are known; the readers then fall back to those.
    spans: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    #: ``"setup"`` when nothing usable came back (the row could not be prepared,
    #: the request failed); ``"output"`` when the model answered with something
    #: that is not a JSON object - a wrong answer, scored as one.
    failure_kind: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """What :func:`evaluation.run_eval.subset_of` reads."""
        images = sum(
            1
            for message in self.row.get("messages", [])
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image"
        )
        return {
            "source_id": self.row.get("source_id", ""),
            "doc_type": self.row.get("doc_type", "unknown"),
            # Both select the schema the output is validated against. Without the
            # form every ACORD row scored as unselectable; without the line every
            # policy was judged against the canonical fallback.
            "acord_form": self.row.get("acord_form"),
            "lob": self.row.get("lob"),
            # A policy row is one window: its output is judged against the slice
            # it was asked for, not the whole schema it is a part of.
            "sections": self.row.get("sections"),
            "modality_mode": self.row.get("modality_mode", "ocr_plus_image"),
            # On every row of a corpus built since the data mix needed it; a
            # row of an older corpus counts as not scanned, which only affects
            # which eval subset a document is also counted in.
            "is_scanned": bool(self.row.get("is_scanned", False)),
            "page_count": images or 1,
            # The text the prompt carried (None for image-only), so the report
            # can tell an invented value from a misread one (hallucination_rate),
            # and each page's own, so a value is looked for on the pages it cites.
            "ocr_text": self.page_text,
            "ocr_pages": self.ocr_pages,
            # The report's error analysis by carrier, and by whether training
            # saw the carrier (both on every row of a current corpus).
            "carrier": self.row.get("carrier"),
            "carrier_seen_in_train": self.row.get("carrier_seen_in_train"),
        }

    def span_view(self) -> dict[str, Any]:
        """The spans the readers take: the span objects where kept, else the logprobs."""
        return self.spans or self.logprobs_by_path

    @property
    def page_text(self) -> str | None:
        """The OCR text the prompt carried, for the OCR-agreement feature: the
        pages' text only, as serving's ``shown_ocr_text`` reads it. A fast-format
        row ends its user turn with the window's request block (:data:`_REQUEST_BLOCK`),
        which is prompt, not page."""
        from calibration.features import shown_ocr_text

        if self.row.get("modality_mode") == "image_only":
            return None
        return shown_ocr_text([
            part.get("text", "")
            for message in self.row.get("messages", [])
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "text"
            and not (part.get("text") or "").startswith(_REQUEST_BLOCK)
        ])


    @property
    def ocr_pages(self) -> dict[int, str] | None:
        """Each page's OCR text the prompt carried, by its page number (read
        from its ``<page N of M>`` marker); ``None`` for image-only."""
        from inference_core.input_builder import EMPTY_PAGE_TEXT

        if self.row.get("modality_mode") == "image_only":
            return None
        pages: dict[int, str] = {}
        for message in self.row.get("messages", []):
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "text":
                    marker = _PAGE_MARKER.match(part.get("text") or "")
                    if marker:
                        text = part["text"][marker.end():].strip()
                        pages[int(marker.group(1))] = "" if text == EMPTY_PAGE_TEXT else text
        return pages or None


#: A page's marker at the head of its text block (inference_core.input_builder.PAGE_MARKER).
_PAGE_MARKER = re.compile(r"<page (\d+) of \d+>\s*")

#: How the fast format's window request opens: the user turn's last text part,
#: after the pages (common.prompts, the fast format profile).
_REQUEST_BLOCK = "\n\n# Request: "


def _compact(row: Mapping[str, Any]) -> bool:
    """Whether ``row`` was written in the fast profile's compact answer form."""
    from common.canonical import COMPACT_ANSWER_FORMAT

    return row.get("answer_format") == COMPACT_ANSWER_FORMAT


def read_rows(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def split_prompt(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The prompt messages and the golden label from one corpus row, the label in
    the full model form (a compact target expanded, ``common.canonical.expand_answer``)."""
    from common.canonical import expand_answer

    messages = list(row.get("messages") or [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            f"row {row.get('source_id')!r} has no trailing assistant turn, so it carries no "
            "golden label to score against"
        )
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return messages[:-1], expand_answer(json.loads(content))


#: Rows per ``generate_batch`` call when the serving config names none. vLLM
#: schedules a call's requests together (continuous batching); one row at a time
#: left the GPU waiting on each answer, ~50 s a row - hours per checkpoint on the
#: smoke set, days on the full corpus. Bounded so a call's opened page images
#: stay a modest amount of memory.
BATCH_ROWS = 32


def configured_batch_rows() -> int:
    """Rows per call: ``generation.validation_batch_rows`` in
    configs/inference/vllm_serving.yaml, else :data:`BATCH_ROWS`."""
    from common.config import serving_config

    value = (serving_config().get("generation") or {}).get("validation_batch_rows")
    return int(value) if value else BATCH_ROWS


def _answer_length(row: dict[str, Any]) -> int:
    """Characters of the row's assistant target - how long its answer should run."""
    messages = row.get("messages") or []
    content = messages[-1].get("content") if messages else ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return len(content or "")


def generate_validation(
    rows: Iterable[dict[str, Any]],
    model: Any,
    *,
    adapter: str | None = None,
    constrain: bool | None = None,
    batch_rows: int | None = None,
    stats: dict[str, Any] | None = None,
    order: str = "planned",
    shard: tuple[int, int] | None = None,
) -> list[ValidationGeneration]:
    """Generate every row once. A row that fails is recorded, never dropped.

    ``shard`` ``(k, n)``: only the calls whose index in the plan is ``k`` modulo
    ``n`` are made - one GPU's share of a pass split across ``n`` (checkpoint
    selection, calibration). The plan is the whole pass's, so each call holds
    exactly the rows, in the order, it holds unsharded: the ``n`` shards
    together are the unsharded pass, call for call. Only the rows of this
    shard's calls come back, with every row that failed setup (each shard
    records those, identically).

    ``constrain`` defaults to what serving does. Calibration must read the
    distribution serving will produce; a calibrator fitted on unconstrained
    generations describes a model nobody serves.

    ``batch_rows`` defaults to :func:`configured_batch_rows`. ``stats``, when
    given, is filled with what the pass cost: rows, calls, seconds and the
    slowest call - the numbers a stage reports, so a run's time is measured.

    ``order`` is the order the rows go to the engine: ``planned`` (calls of
    ``batch_rows`` rows in row order, each longest answer first) or
    ``reversed`` - that order backwards, in calls of the same size, so the
    engine batches the rows differently (run_to_run_agreement's second pass).
    The results come back in row order either way.
    """
    if order not in ("planned", "reversed"):
        raise ValueError(f"order is {order!r}; expected 'planned' or 'reversed'")
    if shard is not None and not (shard[1] >= 1 and 0 <= shard[0] < shard[1]):
        raise ValueError(f"shard is {shard!r}; expected (k, n) with 0 <= k < n")
    import time

    from common.schemas import resolved_schema, with_page_bounds
    from inference_core.input_builder import page_total, shown_pages
    from inference_core.model_runner import generate, generate_batch

    if constrain is None:
        constrain = bool(getattr(model.config, "structured_outputs", False))

    from common.config import answer_cap

    out: list[ValidationGeneration] = []
    pending: list[tuple[ValidationGeneration, list[dict[str, Any]], dict[str, Any] | None, int]] = []
    for row in rows:
        entry = ValidationGeneration(row=row, golden={})
        out.append(entry)
        try:
            # Inside the per-row guard: a row with no assistant turn, or an ACORD
            # row with no form (no schema to select), used to raise out of the
            # loop and abort every other row's scoring with it.
            messages, entry.golden = split_prompt(row)
            # Page references bounded to the pages the row shows (with_page_bounds),
            # as serving bounds them.
            schema = (
                with_page_bounds(
                    resolved_schema(
                        row["doc_type"], row.get("acord_form"), row.get("lob"), row.get("sections")
                    ),
                    page_total(messages),
                    pages=shown_pages(messages),
                )
                if constrain else None
            )
            if schema is not None and _compact(row):
                # Decoded as it was trained: the fast format's adapter never
                # learned the full envelope the grammar would otherwise force.
                from common.canonical import compact_schema

                schema = compact_schema(schema)
        except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
            _record_failure(entry, exc, "setup")
            continue
        # The row's own answer budget: a longer answer is a loop, not an answer.
        pending.append((entry, messages, schema, answer_cap(row.get("task"), row.get("doc_type")),
                        _answer_length(row)))

    def _retry_in_halves(part: list[Any]) -> list[Any]:
        """Results for ``part``, a call that failed together, retried half by half."""
        if len(part) == 1:
            _entry, messages, schema, cap, _n = part[0]
            try:
                return [generate(model, messages, adapter=adapter, json_schema=schema, max_new_tokens=cap)]
            except Exception as exc:  # noqa: BLE001 - the one bad row is recorded, not raised
                return [exc]
        out: list[Any] = []
        middle = len(part) // 2
        for half in (part[:middle], part[middle:]):
            if len(half) == 1:
                out += _retry_in_halves(half)
                continue
            got = generate_batch(model, [(m, s, cap) for _e, m, s, cap, _n in half], adapter=adapter)
            out += _retry_in_halves(half) if all(isinstance(r, Exception) for r in got) else got
        return out

    size = max(1, batch_rows if batch_rows is not None else configured_batch_rows())
    # Longest expected answer first, within the call only: a call ends when its
    # longest answer does, so started first it runs beside the short ones
    # instead of after them. The chunks are the same for every checkpoint and
    # each result goes back to its own row.
    planned = [item for first in range(0, len(pending), size)
               for item in sorted(pending[first:first + size], key=lambda item: -item[4])]
    if order == "reversed":
        planned.reverse()
    chunks = [planned[first:first + size] for first in range(0, len(planned), size)]
    if shard is not None:
        # Another shard's calls: their rows are left out of what comes back.
        others = {id(item[0]) for index, chunk in enumerate(chunks) if index % shard[1] != shard[0]
                  for item in chunk}
        chunks = [chunk for index, chunk in enumerate(chunks) if index % shard[1] == shard[0]]
        out = [entry for entry in out if id(entry) not in others]
    mine = sum(len(chunk) for chunk in chunks)
    started, calls, slowest, done = time.monotonic(), 0, 0.0, 0
    for chunk in chunks:
        began = time.monotonic()
        results = generate_batch(model, [(m, s, cap) for _e, m, s, cap, _n in chunk], adapter=adapter)
        if len(chunk) > 1 and all(isinstance(r, Exception) for r in results):
            # A whole call fails together (one unreadable page image refuses the
            # batch): retried in halves until the bad row is alone - the only
            # loss - in a few batched rounds, not one call per row (at ~50 s a
            # row, 96 rows one by one cost over an hour).
            results = _retry_in_halves(chunk)
        for (entry, _messages, _schema, _cap, _n), result in zip(chunk, results, strict=True):
            _finish(entry, result)
        calls += 1
        done += len(chunk)
        slowest = max(slowest, time.monotonic() - began)
        log.info("validation generation: %d/%d row(s)%s%s", done, mine,
                 f" ({adapter})" if adapter else "", f" [shard {shard[0] + 1}/{shard[1]}]" if shard else "")
    if stats is not None:
        stats.update(rows=len(out), generated=mine, batch_rows=size, calls=calls,
                     seconds=round(time.monotonic() - started, 1), slowest_call_seconds=round(slowest, 1))
        if shard is not None:
            stats["shard"] = f"{shard[0] + 1}/{shard[1]}"
    return out


def _record_failure(entry: ValidationGeneration, exc: BaseException, kind: str,
                    note: str = "", evidence: str = "") -> None:
    # Scored as an empty extraction, not skipped: a model that cannot produce
    # JSON for a document has got every field on it wrong, and leaving it out
    # would score the model on the documents it managed.
    entry.error = f"{note}{type(exc).__name__}: {exc}{evidence}"
    entry.failure_kind = kind
    entry.extraction = {}
    log.warning("validation generation failed for %s: %s", entry.row.get("source_id"), entry.error)


def _finish(entry: ValidationGeneration, result: Any) -> None:
    """Parse one generation into ``entry``, or record why it failed."""
    from common.canonical import collapse_spans, expand_answer, with_output_dates
    from inference_core.span_map import map_compact_field_spans, map_field_spans

    if isinstance(result, BaseException):
        _record_failure(entry, result, "setup")
        return
    try:
        extraction = json.loads(result.text)
        if not isinstance(extraction, dict):
            raise TypeError(f"expected a JSON object, got {type(extraction).__name__}")
        # Keyed and formatted exactly as serving does it (serving.pipeline), or
        # the calibrators are fitted on paths and values serving never looks up.
        # A compact answer (the fast profile) is mapped and expanded to the
        # full form, as serving expands it before anything reads it.
        if _compact(entry.row):
            spans = collapse_spans(map_compact_field_spans(result.text, result.tokens, result.token_logprobs))
            extraction = with_output_dates(expand_answer(extraction))
        else:
            spans = collapse_spans(map_field_spans(result.text, result.tokens, result.token_logprobs))
            extraction = with_output_dates(extraction)
    except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
        truncated = bool(getattr(result, "truncated", lambda: False)())
        text = getattr(result, "text", "") or ""
        # How generation ended and what it ended on: the parse error alone cannot
        # tell a model looping to the limit from a decode that stopped mid-value.
        evidence = (f" [finish={getattr(result, 'finish_reason', None)}, "
                    f"{len(getattr(result, 'tokens', []) or [])} tokens, {len(text)} chars, "
                    f"ends {text[-60:]!r}]")
        _record_failure(entry, exc, "output",
                        "stopped at the token limit before the JSON closed - " if truncated else "",
                        evidence)
        return
    entry.extraction = extraction
    # The span objects kept whole (r_written, the top-2 margin), the logprobs beside them.
    entry.spans = {path: span for path, span in spans.items() if span.mapped}
    entry.logprobs_by_path = {path: span.token_logprobs for path, span in entry.spans.items()}


def score_generations(
    generations: Sequence[ValidationGeneration], *, model_version: str = "validation",
    repeat: Sequence[ValidationGeneration] | None = None,
) -> dict[str, Any]:
    """The gate's metrics over these generations — the one definition of correct.

    Window by window, as each row was asked; and, where policies were read in
    windows, the same metrics over whole documents (``document_*``): each
    document's windows merged as serving merges them (:func:`merged_documents`).
    A window's score cannot see what only the merge does - a row read twice, a
    value two windows disagree on, a link that survives only when one window
    shows both rows - and the document is what ships.

    ``repeat`` is the same rows generated a second time in another order
    (``generate_validation(order="reversed")``): each answer is compared with
    its second reading, as the gate compares them - ``run_to_run_agreement``,
    and ``document_run_to_run_agreement`` over whole documents. It may be the
    rows of some modes only - the gate's repeat mode, image_only - each
    document whole in them: the readings with no second one are scored as
    before and compared with nothing.

    An image-only row carries no page text: its values are checked for
    invention on its document's page text borrowed from the same document's
    ocr_plus_image rows among ``generations`` (``run_eval.build_report``,
    decision 2026-10-26). What that could not cover is counted beside the rate
    (:func:`_invention_coverage`), never read as nothing invented.
    """
    from evaluation.run_eval import REPEAT_ANSWER, build_report

    rows = [(g.golden, g.extraction or {}, g.metadata) for g in generations]
    partial = repeat is not None and len(repeat) != len(generations)
    if repeat is not None and not partial:
        rows = _with_repeat(rows, [(g.golden, g.extraction or {}, g.metadata) for g in repeat])
    elif partial:
        paired = _paired_repeat(generations, repeat)
        rows = [(gold, answer, {**metadata, REPEAT_ANSWER: again.extraction or {}} if again is not None else metadata)
                for (gold, answer, metadata), again in zip(rows, paired, strict=True)]
    report = build_report(model_version, rows)
    metrics = {**report.gate_metrics(), **_invention_coverage(report)}
    if any(_is_window(g) for g in generations):
        documents = merged_documents(generations)
        if repeat is not None and not partial:
            documents = _with_repeat(documents, merged_documents(repeat))
        elif partial:
            documents = _with_document_repeat(documents, merged_documents(
                [again for again in paired if again is not None]))
        whole_report = build_report(model_version, documents)
        whole = {**whole_report.gate_metrics(), **_invention_coverage(whole_report)}
        metrics.update({f"document_{name}": value for name, value in whole.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return metrics


def _invention_coverage(report: Any) -> dict[str, int]:
    """The counts beside hallucination_rate (``EvalReport.invention_coverage``),
    as ``hallucination_<count>``: readings checked on borrowed page text, the
    pages they cite with none, and readings left unscored. Only those above
    zero: a pass every reading of which had its own text reports as before."""
    return {f"hallucination_{name}": count for name, count in report.invention_coverage().items() if count}


def _with_repeat(first: list[tuple[Any, Any, dict[str, Any]]],
                 second: list[tuple[Any, Any, dict[str, Any]]]) -> list[tuple[Any, Any, dict[str, Any]]]:
    """``first``'s triples, each carrying ``second``'s answer to the same
    reading (``run_eval.REPEAT_ANSWER``). Both passes generate the same rows and
    return them in row order, so they pair by position - checked, never assumed."""
    from evaluation.run_eval import REPEAT_ANSWER

    if len(first) != len(second):
        raise ValueError(f"the repeat pass read {len(second)} reading(s), the first {len(first)}")
    out = []
    for (gold, answer, metadata), (_gold, again, other) in zip(first, second, strict=True):
        reading = (metadata.get("source_id"), metadata.get("modality_mode"), metadata.get("sections"))
        if reading != (other.get("source_id"), other.get("modality_mode"), other.get("sections")):
            raise ValueError(f"the repeat pass is out of step with the first at {reading}")
        out.append((gold, answer, {**metadata, REPEAT_ANSWER: again}))
    return out


def _paired_repeat(generations: Sequence[ValidationGeneration],
                   repeat: Sequence[ValidationGeneration]) -> list[ValidationGeneration | None]:
    """``repeat``'s reading of each of ``generations``, or None for one not
    read twice: a second pass over some modes only (the gate's repeat mode).
    Paired by row - the same row, else the same reading (:func:`reading_key`) -
    never by position; a second reading of nothing in the first is refused."""
    by_row = {id(g.row): index for index, g in enumerate(generations)}
    by_key: dict[str, int] = {}
    for index, g in enumerate(generations):
        by_key.setdefault(reading_key(g.row), index)
    paired: list[ValidationGeneration | None] = [None] * len(generations)
    for again in repeat:
        index = by_row.get(id(again.row), by_key.get(reading_key(again.row)))
        if index is None or paired[index] is not None:
            raise ValueError(f"the repeat pass is out of step with the first at {reading_key(again.row)}")
        paired[index] = again
    return paired


def _with_document_repeat(first: list[tuple[Any, Any, dict[str, Any]]],
                          second: list[tuple[Any, Any, dict[str, Any]]]) -> list[tuple[Any, Any, dict[str, Any]]]:
    """``first``'s documents, those ``second`` read again carrying its answer
    (``run_eval.REPEAT_ANSWER``), paired by document and mode; the rest as they are."""
    from evaluation.run_eval import REPEAT_ANSWER

    def key(metadata: dict[str, Any]) -> tuple[Any, Any, Any]:
        return metadata.get("source_id"), metadata.get("modality_mode"), metadata.get("sections")

    again = {}
    for _gold, answer, metadata in second:
        if key(metadata) in again:
            raise ValueError(f"the repeat pass read {key(metadata)} twice")
        again[key(metadata)] = answer
    known = {key(metadata) for _g, _a, metadata in first}
    if set(again) - known:
        raise ValueError(f"the repeat pass is out of step with the first at {sorted(set(again) - known)[:1]}")
    return [(gold, answer, {**metadata, REPEAT_ANSWER: again[key(metadata)]} if key(metadata) in again else metadata)
            for gold, answer, metadata in first]


def reading_key(row: Mapping[str, Any]) -> str:
    """One validation reading, as text: its document, mode, window group and
    window (index, half and pages). What pairs a row with its saved answer
    (:func:`restored_generation`) and with its second reading.

    The half (``window_half``: "0", "1", "1.0" ...) is part of it: a window
    whose answer overruns its reservation is built as serving's halves, each
    keeping the window's index (review #38), and without it two halves were
    one reading. A row read whole has none ("")."""
    pages = row.get("window_pages")
    return json.dumps([str(row.get("source_id") or ""), str(row.get("modality_mode") or "ocr_plus_image"),
                       row.get("sections"), row.get("window_index"), str(row.get("window_half") or ""),
                       [int(p) for p in pages] if pages else None], separators=(",", ":"))


def _half_order(row: Mapping[str, Any]) -> tuple[int, ...]:
    """Where a window's half goes among its window's halves - "0" < "1.0" <
    "1.1", page order, as serving puts them back; () for a row read whole."""
    half = str(row.get("window_half") or "")
    try:
        return tuple(int(part) for part in half.split(".")) if half else ()
    except ValueError:
        return ()


def row_digest(row: Mapping[str, Any]) -> str:
    """A short hash of the row's target (its assistant turn): a saved answer is
    restored only onto the row it answered, never onto a rebuilt split's."""
    import hashlib

    messages = row.get("messages") or []
    content = messages[-1].get("content") if messages and messages[-1].get("role") == "assistant" else ""
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()[:16]


def generation_settings(config: Any, *, constrain: bool | None = None, batch_rows: int | None = None) -> dict[str, Any]:
    """What decides a generation besides the rows and the adapter: the runner's
    output fingerprint (backend, sampling, seed, answer limit, image cap,
    grammar, logprob mode - ``RunnerConfig.fingerprint``), whether it was
    constrained, and the rows per call. Two passes with equal settings and
    adapter generate alike; saved answers are reused only then (calibration)."""
    if constrain is None:
        constrain = bool(getattr(config, "structured_outputs", False))
    fingerprint = getattr(config, "fingerprint", None)
    return {
        "runner": fingerprint() if callable(fingerprint) else None,
        "constrain": bool(constrain),
        "logprobs": bool(getattr(config, "logprobs", True)),
        "batch_rows": int(batch_rows or configured_batch_rows()),
    }


def generation_record(generation: ValidationGeneration) -> dict[str, Any]:
    """One generated row as saved: its reading, its row's digest, and what came
    back - the logprobs by path, and of each span what calibration reads besides
    them (``span_details``: the path it was written at, its top-2 margin)."""
    return {"key": reading_key(generation.row), "digest": row_digest(generation.row),
            "extraction": generation.extraction, "logprobs_by_path": generation.logprobs_by_path,
            "span_details": {path: [getattr(span, "field_path", path), getattr(span, "top2_margin", None)]
                             for path, span in generation.spans.items()},
            "error": generation.error, "failure_kind": generation.failure_kind}


def restored_generation(row: dict[str, Any], record: Mapping[str, Any]) -> ValidationGeneration:
    """A saved generation back on its row: what :func:`generate_validation`
    returned for it, the golden label read off the row again, and its spans
    rebuilt from their logprobs and details - all calibration reads of them."""
    from inference_core.span_map import FieldSpan

    try:
        golden = split_prompt(row)[1]
    except Exception:  # noqa: BLE001 - as generate_validation: a row with no label scores as none
        golden = {}
    logprobs = {path: list(values) for path, values in (record.get("logprobs_by_path") or {}).items()}
    spans = {path: FieldSpan(field_path=str(written), value=None, char_start=-1, char_end=-1,
                             token_logprobs=logprobs.get(path, []), mapped=True,
                             top2_margin=float(margin) if isinstance(margin, (int, float)) else None)
             for path, (written, margin) in (record.get("span_details") or {}).items()}
    return ValidationGeneration(row=row, golden=golden, extraction=record.get("extraction"),
                                logprobs_by_path=logprobs, spans=spans,
                                error=record.get("error"), failure_kind=record.get("failure_kind"))


def write_generations(path: Any, header: Mapping[str, Any], generations: Sequence[ValidationGeneration]) -> None:
    """``generations`` as JSON lines after ``header`` (who generated them, how,
    on what): the form calibration reads them back in (:func:`read_generations`)."""
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"header": dict(header)}) + "\n")
        for generation in generations:
            handle.write(json.dumps(generation_record(generation)) + "\n")
    partial.replace(target)


def read_generations(text: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``(header, records)`` from :func:`write_generations`'s text."""
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not lines or "header" not in lines[0]:
        raise ValueError("saved generations carry no header")
    return dict(lines[0]["header"]), lines[1:]


def _is_window(generation: ValidationGeneration) -> bool:
    row = generation.row
    return row.get("doc_type") == "policy" and bool(row.get("sections")) and bool(row.get("window_pages"))


def merged_documents(
    generations: Sequence[ValidationGeneration],
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """``(gold, answer, metadata)`` per document and reading mode: a policy's
    window rows merged into one document by the serving merge - its window
    targets into the gold, its generations into the answer - and every other row
    as it is.

    The gold is the merged targets, not the full label: what windowing loses
    (the oracle's ceiling) is lost from both, and the score reads the model
    alone. A window that failed to generate merges as an empty answer: its
    values count as missed, as they do window by window.
    """
    out = [(g.golden, g.extraction or {}, g.metadata) for g in generations if not _is_window(g)]
    out += [(d.gold, d.answer, d.metadata) for d in merged_windows(generations)]
    return out


@dataclass
class MergedDocument:
    """One document's windows merged as serving merges them."""

    gold: dict[str, Any]
    answer: dict[str, Any]
    #: The answer's token logprobs by values-view path, carried through the merge.
    spans: dict[str, Any]
    metadata: dict[str, Any]
    members: list[ValidationGeneration]


def merged_windows(generations: Sequence[ValidationGeneration]) -> list[MergedDocument]:
    """Each policy document read in windows, per reading mode, merged."""
    from common.schema_sections import group_names
    from serving.policy_merge import PolicyWindow, merge_policy_windows

    documents: dict[tuple[Any, Any], list[ValidationGeneration]] = {}
    for generation in generations:
        if _is_window(generation):
            key = (generation.row.get("source_id"), generation.row.get("modality_mode"))
            documents.setdefault(key, []).append(generation)

    out: list[MergedDocument] = []
    for members in documents.values():
        lob = members[0].row.get("lob")
        rank = {name: index for index, name in enumerate(group_names(lob))}
        # A window built as halves (review #38): its halves in page order, in its place.
        members.sort(key=lambda g: (rank.get(g.row["sections"], len(rank)), int(g.row.get("window_index") or 0),
                                    _half_order(g.row)))

        def window(g: ValidationGeneration, extraction: Any, spans: Any) -> Any:
            # ``leads`` as serving keeps it (PolicyWindowPlan.leads): the window
            # its group's declarations lead merges first in its group. Without
            # it, a split declarations window was ordered by its first page here
            # and by its lead in serving, so selection judged another merge.
            return PolicyWindow(group=g.row["sections"], pages=[int(p) for p in g.row["window_pages"]],
                                extraction=extraction or {}, spans=spans or {},
                                leads=bool(g.row.get("window_leads")))

        gold = merge_policy_windows([window(g, g.golden, None) for g in members], lob=lob)
        answer = merge_policy_windows(
            [window(g, g.extraction, g.span_view()) for g in members], lob=lob)
        pages: dict[int, str] = {}
        for generation in members:
            pages.update(generation.ocr_pages or {})
        metadata = {
            **members[0].metadata,
            "sections": None,
            "page_count": len({int(p) for g in members for p in g.row["window_pages"]}),
            "ocr_pages": pages or None,
            "ocr_text": "\n\n".join(pages[number] for number in sorted(pages)) if pages else None,
        }
        out.append(MergedDocument(gold.extraction, answer.extraction, dict(answer.spans), metadata, members))
    return out


def calibration_samples(
    generations: Sequence[ValidationGeneration],
) -> dict[str, list[tuple[Any, bool]]]:
    """``(features, correct)`` per field, by validation half (arch v2.1 §8.2).

    Rows with no ``val_half`` are left out rather than guessed into a half:
    fitting a calibrator and choosing its threshold on the same rows prices risk
    the calibrator has already been pulled toward.
    """
    from calibration.features import build_document_features
    from common.canonical import values_view, without_bare_values
    from common.schemas import is_common_model
    from evaluation.metrics.field_accuracy import (
        aligned_for_scoring,
        flatten_scalars,
        rows_aligned_to,
        values_agree,
    )
    from serving.conventions import conform

    halves: dict[str, list[tuple[Any, bool]]] = {"calibration": [], "threshold": []}
    unassigned = 0

    def add(half: str, golden: Any, answer: Any, spans: Any, *, common_model: bool, page_text: Any,
            page_texts: Any) -> None:
        # The label's rows in the order the model wrote its own, paired by
        # identifier: compared by position, one omitted or reordered row made
        # every later row's correct values "wrong", and the calibrator learned
        # to distrust them. A common-model row pairs as its field match and
        # auto-accept pair it (aligned_for_scoring): paired on the code alone,
        # a coverage with a wrong code faced nothing, and its correctly read
        # name, limits and premium were fitted as wrong while the gate counted
        # them right. Answer row i keeps index i; label rows nothing paired go
        # after the answer's, where no feature path reaches them.
        gold_values = values_view(golden)
        got_values = values_view(answer or {})
        expected = flatten_scalars(
            aligned_for_scoring(gold_values, got_values)[0] if common_model
            else rows_aligned_to(gold_values, got_values)
        )
        # Fitted as serving calibrates (serving.pipeline._feature_calibrated): on
        # a common-model line, typed as its schema declares and without its ids,
        # codes and links, which serving never calibrates. Otherwise a calibrator
        # is fitted on fields, and under types, it is never asked about.
        for features in build_document_features(
            extraction=without_bare_values(answer or {}) if common_model else (answer or {}),
            spans=spans, page_text=page_text, common_model=common_model, page_texts=page_texts,
        ):
            correct = values_agree(expected.get(features.field_path), features.value, features.field_path)
            halves[half].append((features, bool(correct)))

    # A policy read in windows is fitted as it is served: merged, then held to
    # the label conventions (serving.conventions) - the answer serving asks the
    # calibrators about, not one window of it.
    for document in merged_windows(generations):
        half = document.members[0].row.get("val_half")
        if half not in halves:
            unassigned += len(document.members)
            continue
        if all(g.error for g in document.members):
            continue
        conformed = conform(document.answer, document.spans, document.metadata.get("ocr_pages"),
                            native=not document.metadata.get("is_scanned"))
        add(half, document.gold, conformed.extraction, conformed.spans, common_model=True,
            page_text=document.metadata.get("ocr_text"), page_texts=document.metadata.get("ocr_pages"))

    for generation in generations:
        if _is_window(generation):
            continue
        half = generation.row.get("val_half")
        if half not in halves:
            unassigned += 1
            continue
        if generation.error:
            # No generation means no features. The failure is already counted in
            # the scored metrics; it has no token evidence to calibrate on.
            continue
        common_model = generation.row.get("doc_type") == "policy" and is_common_model(
            "policy", None, generation.row.get("lob"))
        add(half, generation.golden, generation.extraction, generation.span_view(),
            common_model=common_model, page_text=generation.page_text, page_texts=generation.ocr_pages)

    if unassigned:
        log.warning(
            "%d validation row(s) carry no val_half and were left out of calibration. "
            "Rebuild the corpus: the dataset build stamps every val row with its half.",
            unassigned,
        )
    return halves


#: The finalists' field match as the gate measures it (:func:`gate_mirror`).
GATE_MIRROR_ACCURACY = "gate_mirror_accuracy"


def gate_mirror(generations: Sequence[ValidationGeneration], *, model_version: str = "validation") -> dict[str, float]:
    """The pass scored as the golden gate scores a release, as far as
    validation can: ``gate_mirror_accuracy`` and what it is over.

    * image_only readings only - the mode production reads (fields first) and
      the gate's run-to-run mode;
    * whole documents, the windows merged as serving merges them, then held to
      the label conventions as serving holds a common-model answer;
    * held to the page text AS SERVED: production checks an image_only answer on
      MinerU's text once it arrives, and the gate on the document's text
      (golden_eval.borrowed_printed_check). Here the text is borrowed from the
      same document's ocr_plus_image rows, and the answer goes through serving's
      own rule (``serving.printed_values.hold_to_printed``, in the configured
      mode - strict): a value no page prints is taken out before it is scored. A
      page it cites with no text is checked as blank: a value there is kept;
    * real documents only: a synthetic twin counts toward nothing the gate reads.

    The gold is the merged window targets, not the frozen full label: what
    windowing loses is lost from both, so this reads somewhat above the gate,
    by an offset the early warning measures cycle by cycle.

    A document with no ocr_plus_image text to borrow is left out and counted
    (``gate_mirror_unscored_documents``). ``{}`` when no real image_only
    document could be scored: the caller falls back to its own measure.
    """
    import copy

    from common.lob import merge_line
    from common.schemas import is_common_model
    from evaluation.run_eval import (
        BORROWED_TEXT,
        IMAGE_ONLY_MODE,
        PRINTED_CHECK,
        PRINTED_TEXT,
        TEXT_DONOR_MODE,
        UNREAD_PAGES,
        build_report,
        cited_pages,
        page_has_text,
    )
    from serving.conventions import conform
    from serving.printed_values import hold_to_printed

    documents = merged_windows(generations)
    donors: dict[Any, dict[int, str]] = {}
    for document in documents:
        if document.metadata.get("modality_mode") == TEXT_DONOR_MODE and document.metadata.get("ocr_pages"):
            donors.setdefault(document.metadata.get("source_id"), {}).update(document.metadata["ocr_pages"])
    triples: list[tuple[Any, Any, dict[str, Any]]] = []
    synthetic = unscored = 0
    for document in documents:
        if document.metadata.get("modality_mode") != IMAGE_ONLY_MODE:
            continue
        if any(member.row.get("synthetic") for member in document.members):
            synthetic += 1
            continue
        texts = {int(page): str(text or "") for page, text in (donors.get(document.metadata.get("source_id")) or {}).items()}
        if not any(page_has_text(text) for text in texts.values()):
            unscored += 1
            continue
        lob = document.members[0].row.get("lob")
        answer, spans = copy.deepcopy(document.answer), dict(document.spans)
        if is_common_model("policy", None, lob):
            # As serving holds an image_only answer before the text arrives: no page text.
            conformed = conform(answer, spans, None, native=not document.metadata.get("is_scanned"))
            answer, spans = conformed.extraction, conformed.spans
        unread = sorted(page for page in cited_pages(answer) if not page_has_text(texts.get(page)))
        texts.update({page: "" for page in unread if page not in texts})
        held = hold_to_printed(answer, spans, texts, lob=merge_line(lob))
        if held.record is None:
            unscored += 1
            continue
        triples.append((document.gold, held.extraction, {
            **document.metadata, PRINTED_CHECK: held.record, PRINTED_TEXT: BORROWED_TEXT, UNREAD_PAGES: unread}))
    counts = {"gate_mirror_unscored_documents": float(unscored), "gate_mirror_synthetic_documents": float(synthetic)}
    if not triples:
        return {}
    metrics = build_report(model_version, triples).gate_metrics()
    accuracy = metrics.get("field_normalized_match")
    if not isinstance(accuracy, (int, float)):
        return {}
    out = {GATE_MIRROR_ACCURACY: float(accuracy), "gate_mirror_documents": float(len(triples)), **counts}
    if isinstance(metrics.get("hallucination_rate"), (int, float)):
        out["gate_mirror_hallucination_rate"] = float(metrics["hallucination_rate"])
    return out


#: An error class (``training.vit_gate.classify_error``) as the tallies count it.
_TALLY_KIND = {"omission": "missed", "invented": "invented"}


def error_tallies(generations: Sequence[ValidationGeneration]) -> dict[str, dict[str, dict[str, int]]]:
    """Every value the pass scored, and its errors: ``values``, ``wrong``,
    ``missed`` and ``invented``, by field (``vehicles[].vin``), line, window
    group and reading mode.

    Over whole documents (:func:`merged_documents`), each value scored as
    ``run_eval.score_subset`` scores it - the same view, field match's own
    scorer, and its error class (``classify_error``): left empty is missed, a
    value where the label holds none is invented, any other failure is wrong.
    ``values`` counts every scoring, right or wrong, as the report's own
    per-field tally does. The group is the window group whose sections hold
    the field's top-level key (``other`` for none).
    """
    from common.lob import merge_line
    from common.schema_sections import group_names, sections_for
    from evaluation.metrics.common_model import core_for_scoring
    from evaluation.metrics.field_accuracy import score_fields
    from evaluation.run_eval import _is_common_model_document, _line_name, _scored_view
    from training.vit_gate import classify_error

    tallies: dict[str, dict[str, dict[str, int]]] = {"by_field": {}, "by_line": {}, "by_group": {}, "by_mode": {}}
    groups_by_line: dict[str, dict[str, str]] = {}

    def group_of(path: str, lob: Any) -> str:
        line = json.dumps(lob, sort_keys=True, default=str)
        if line not in groups_by_line:
            groups_by_line[line] = {section: group for group in group_names(lob) for section in sections_for(group, lob)}
        top = re.match(r"[^.\[]+", path)
        return groups_by_line[line].get(top.group(0) if top else "", "other")

    for expected, got, metadata in merged_documents(generations):
        doc_type = metadata.get("doc_type", "unknown")
        common_model = _is_common_model_document(doc_type, metadata)
        selectors = (doc_type, metadata.get("acord_form"), metadata.get("lob"))
        line = merge_line(metadata.get("lob")) if common_model else metadata.get("lob")
        expected = _scored_view(expected, selectors, line if common_model else None, common_model)
        got = _scored_view(got, selectors, line if common_model else None, common_model)
        accuracy = (score_fields(*core_for_scoring(expected, got, line), skip_lists=False)
                    if common_model else score_fields(expected, got))
        mode = str(metadata.get("modality_mode") or "ocr_plus_image")
        for result in accuracy.results:
            path = re.sub(r"\[[^\]]*\]", "[]", result.field_path)
            kind = None if result.correct else _TALLY_KIND.get(
                classify_error(result.expected, result.got, all_expected=expected), "wrong")
            for table, name in (("by_field", path), ("by_line", _line_name(line)),
                                ("by_group", group_of(path, metadata.get("lob"))), ("by_mode", mode)):
                tally = tallies[table].setdefault(name, {"values": 0, "wrong": 0, "missed": 0, "invented": 0})
                tally["values"] += 1
                if kind:
                    tally[kind] += 1
    return {table: dict(sorted(body.items())) for table, body in tallies.items()}


#: Above this share of rows with nothing usable back, a pass over validation is
#: not a measurement. Scoring the failures as empty answers turned a broken
#: setup — unreadable images, a model that would not load — into a checkpoint
#: "chosen" at 0.0 and a calibrator fitted on nothing, with no error anywhere.
MAX_GENERATION_FAILURE_RATE = 0.10

#: Rows the model answered with unusable JSON are wrong answers, not a broken
#: pass, and are scored as such: a checkpoint that loops on 14% of documents has
#: to lose to one that does not, not be dropped from the comparison. Counting
#: them against the 10% above refused every checkpoint of the smoke run. Only
#: when most rows are unusable is there too little left to rank on.
MAX_UNUSABLE_OUTPUT_RATE = 0.50


class ValidationGenerationError(RuntimeError):
    """Raised when too many validation rows failed to generate to trust a score."""


def assert_generations_usable(
    generations: list[ValidationGeneration], *, what: str,
    max_failure_rate: float = MAX_GENERATION_FAILURE_RATE,
) -> None:
    """Refuse a pass that measured nothing.

    More than ``max_failure_rate`` of rows with nothing back is a broken setup.
    Rows the model answered with unusable JSON are wrong answers and stay in the
    score, until they are most of the pass (:data:`MAX_UNUSABLE_OUTPUT_RATE`).
    """
    if not generations:
        raise ValidationGenerationError(f"{what}: no validation rows were generated")
    failed = [g for g in generations if g.error and g.failure_kind != "output"]
    unusable = [g for g in generations if g.error and g.failure_kind == "output"]
    if unusable:
        log.info("%s: %d of %d row(s) answered with unusable JSON, scored as wrong answers",
                 what, len(unusable), len(generations))
    if len(unusable) / len(generations) > MAX_UNUSABLE_OUTPUT_RATE:
        sample = sorted({g.error for g in unusable})[:3]
        raise ValidationGenerationError(
            f"{what}: {len(unusable)} of {len(generations)} validation rows came back as "
            f"unusable JSON (e.g. {sample}). With most answers unreadable there is too little "
            "left to rank or calibrate on."
        )
    if len(failed) / len(generations) > max_failure_rate:
        sample = sorted({g.error for g in failed})[:3]
        raise ValidationGenerationError(
            f"{what}: {len(failed)} of {len(generations)} validation rows failed to generate "
            f"(e.g. {sample}). That is a broken pass, not a score — scoring the failures as "
            "empty answers would pick or fit on nothing."
        )
