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

The prompt is the row's own ``messages`` with the assistant turn removed, so the
model sees exactly what training rendered for that document (arch §7).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
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
        }

    @property
    def page_text(self) -> str | None:
        """The OCR text the prompt carried, for the OCR-agreement feature."""
        from calibration.features import shown_ocr_text

        if self.row.get("modality_mode") == "image_only":
            return None
        return shown_ocr_text([
            part.get("text", "")
            for message in self.row.get("messages", [])
            if message.get("role") == "user" and isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "text"
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


def read_rows(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def split_prompt(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The prompt messages and the golden label from one corpus row."""
    messages = list(row.get("messages") or [])
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(
            f"row {row.get('source_id')!r} has no trailing assistant turn, so it carries no "
            "golden label to score against"
        )
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return messages[:-1], json.loads(content)


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
) -> list[ValidationGeneration]:
    """Generate every row once. A row that fails is recorded, never dropped.

    ``constrain`` defaults to what serving does. Calibration must read the
    distribution serving will produce; a calibrator fitted on unconstrained
    generations describes a model nobody serves.

    ``batch_rows`` defaults to :func:`configured_batch_rows`. ``stats``, when
    given, is filled with what the pass cost: rows, calls, seconds and the
    slowest call - the numbers a stage reports, so a run's time is measured.
    """
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
        except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
            _record_failure(entry, exc, "setup")
            continue
        # The row's own answer budget: a longer answer is a loop, not an answer.
        pending.append((entry, messages, schema, answer_cap(row.get("task"), row.get("doc_type")),
                        _answer_length(row)))

    size = max(1, batch_rows if batch_rows is not None else configured_batch_rows())
    started, calls, slowest = time.monotonic(), 0, 0.0
    for first in range(0, len(pending), size):
        # Longest expected answer first, within the call only: a call ends when
        # its longest answer does, so started first it runs beside the short ones
        # instead of after them. The chunks are the same for every checkpoint and
        # each result goes back to its own row.
        chunk = sorted(pending[first:first + size], key=lambda item: -item[4])
        began = time.monotonic()
        results = generate_batch(model, [(m, s, cap) for _e, m, s, cap, _n in chunk], adapter=adapter)
        if len(chunk) > 1 and all(isinstance(r, Exception) for r in results):
            # A whole call fails together (one unreadable page image refuses the
            # batch): retry its rows alone, so the bad one is the only loss.
            results = []
            for _entry, messages, schema, cap, _n in chunk:
                try:
                    results.append(generate(model, messages, adapter=adapter, json_schema=schema,
                                            max_new_tokens=cap))
                except Exception as exc:  # noqa: BLE001
                    results.append(exc)
        for (entry, _messages, _schema, _cap, _n), result in zip(chunk, results, strict=True):
            _finish(entry, result)
        calls += 1
        slowest = max(slowest, time.monotonic() - began)
        log.info("validation generation: %d/%d row(s)%s", min(first + size, len(pending)),
                 len(pending), f" ({adapter})" if adapter else "")
    if stats is not None:
        stats.update(rows=len(out), generated=len(pending), batch_rows=size, calls=calls,
                     seconds=round(time.monotonic() - started, 1), slowest_call_seconds=round(slowest, 1))
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
    from common.canonical import collapse_spans, with_output_dates
    from inference_core.span_map import map_field_spans

    if isinstance(result, BaseException):
        _record_failure(entry, result, "setup")
        return
    try:
        extraction = json.loads(result.text)
        if not isinstance(extraction, dict):
            raise TypeError(f"expected a JSON object, got {type(extraction).__name__}")
        # Keyed and formatted exactly as serving does it (serving.pipeline), or
        # the calibrators are fitted on paths and values serving never looks up.
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
    entry.logprobs_by_path = {
        path: span.token_logprobs for path, span in spans.items() if span.mapped
    }


def score_generations(
    generations: Sequence[ValidationGeneration], *, model_version: str = "validation"
) -> dict[str, Any]:
    """The gate's metrics over these generations — the one definition of correct.

    Window by window, as each row was asked; and, where policies were read in
    windows, the same metrics over whole documents (``document_*``): each
    document's windows merged as serving merges them (:func:`merged_documents`).
    A window's score cannot see what only the merge does - a row read twice, a
    value two windows disagree on, a link that survives only when one window
    shows both rows - and the document is what ships.
    """
    from evaluation.run_eval import build_report

    report = build_report(
        model_version,
        [(g.golden, g.extraction or {}, g.metadata) for g in generations],
    )
    metrics = dict(report.gate_metrics())
    if any(_is_window(g) for g in generations):
        whole = build_report(model_version, merged_documents(generations)).gate_metrics()
        metrics.update({f"document_{name}": value for name, value in whole.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)})
    return metrics


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
        members.sort(key=lambda g: (rank.get(g.row["sections"], len(rank)), int(g.row.get("window_index") or 0)))

        def window(g: ValidationGeneration, extraction: Any, spans: Any) -> Any:
            return PolicyWindow(group=g.row["sections"], pages=[int(p) for p in g.row["window_pages"]],
                                extraction=extraction or {}, spans=spans or {})

        gold = merge_policy_windows([window(g, g.golden, None) for g in members], lob=lob)
        answer = merge_policy_windows(
            [window(g, g.extraction, g.logprobs_by_path) for g in members], lob=lob)
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
        add(half, generation.golden, generation.extraction, generation.logprobs_by_path,
            common_model=common_model, page_text=generation.page_text, page_texts=generation.ocr_pages)

    if unassigned:
        log.warning(
            "%d validation row(s) carry no val_half and were left out of calibration. "
            "Rebuild the corpus: the dataset build stamps every val row with its half.",
            unassigned,
        )
    return halves


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
