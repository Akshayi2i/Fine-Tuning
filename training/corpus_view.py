"""The adapter's view of one corpus (arch v2.1 §8.2).

The adapter trains on the corpus's **single-line policy rows** (a package and
a policy with no line go to the base model at serving, so they do not train),
written beside the corpus under ``train/scope/policy/`` with the data mix,
render rotation and line balance applied. The split and the group assignment are the corpus's own, so
two versions trained on one corpus stay comparable.

Filtering rather than re-splitting also keeps the epoch structure: every epoch
file holds each train document exactly once, in that epoch's sampled modality
regime, which is what makes "3 epochs" mean three passes.

The configured shares are applied here: per line, its synthetic
fraction and scanned share decide which synthetic documents train, the same
ones in every epoch (:mod:`training.data_mix`, Fideon SPEC_09 amendment items 6
and 7). Every real document trains. With ``data_mix.twins_per_seed_per_epoch``
set, each epoch trains that many twins of every seed instead, the next ones the
next epoch (:func:`training.data_mix.plan_twin_rotation`); null trains every
twin in every epoch.

**Rehearsal** (``continued_training.rehearsal_old_share``, null by default): a
run that continues the promoted adapter trains, in each epoch, every NEW
document and a rotated, stratified sample of the OLD ones - those its chain of
runs trained on (:func:`plan_rehearsal`). Fresh runs train every document.

**Hard examples** (``hard_examples.top_fields``, null by default): the train
windows whose targets hold one of the fields the parent run (or the promoted
release's run) got most wrong on validation are repeated, each repeat from
another epoch file (:func:`hard_fields`, :func:`target_fields`).

**Render rotation** (``render_rotation: true`` in the training config): a
twin delivered as a digital and a scanned PDF with one gold trains in ONE of
them per epoch, the other the next (:func:`plan_render_rotation`). Both renders
carry the same answer, so training both in one epoch repeats it; rotated, every
twin still trains every epoch and both renders over the run, in about half the
documents. Half of a seed's twins start digital and half scanned, so each epoch
stays about half and half. Real documents and twins with one render train every
epoch, and a line that rotation would take under the line-balance floor keeps
both renders - distinct inputs before repeated rows. Validation and test keep
every render.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.adapter import ADAPTER

log = logging.getLogger(__name__)

#: Materialized epoch files, matching the corpus build (arch v2.1 §6.1).
EPOCH_FILES = 4


class CorpusViewError(RuntimeError):
    """Raised when the adapter's view cannot be built from the corpus."""


@dataclass
class CorpusView:
    """Where a training run reads from, and what it found."""

    adapter: str = ADAPTER.name
    epoch_files: list[str] = field(default_factory=list)
    val_path: str = ""
    rows_by_epoch: dict[int, int] = field(default_factory=dict)
    val_rows: int = 0
    test_rows: int = 0
    dropped_rows: int = 0
    #: Line of business -> times its rows appear in each epoch file (line_balance).
    line_repeats: dict[str, int] = field(default_factory=dict)
    #: The configured shares, and per line what its train documents
    #: came to (training.data_mix). Train documents the shares left out.
    mix_settings: dict = field(default_factory=dict)
    data_mix: dict[str, dict] = field(default_factory=dict)
    rested_documents: int = 0
    #: split -> line -> documents, real, synthetic and rows (train: one epoch,
    #: after the shares and render rotation, before line balance repeats anything).
    examples_by_line: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    #: Render rotation as applied: on or off, and per line its documents, the
    #: documents of one epoch and the twins rotated (empty when off).
    render_rotation: dict = field(default_factory=dict)
    #: Twin rotation as applied (data_mix.twins_per_seed_per_epoch; empty when off).
    twin_rotation: dict = field(default_factory=dict)
    #: Rehearsal of a continued run: its new and old documents, and the old ones
    #: each epoch trains (continued_training.rehearsal_old_share; empty when off).
    rehearsal: dict = field(default_factory=dict)
    #: Hard examples as applied: the fields and the rows repeated per epoch
    #: (hard_examples; empty when off).
    hard_examples: dict = field(default_factory=dict)
    #: How line balance counted a line and filled its repeats.
    line_balance: dict = field(default_factory=dict)

    @property
    def train_rows(self) -> int:
        return sum(self.rows_by_epoch.values())

    def describe(self) -> str:
        return (
            f"{self.adapter}: {self.train_rows} train row(s) across {len(self.epoch_files)} epoch "
            f"file(s), {self.val_rows} val row(s), {self.dropped_rows} row(s) of other types"
        )


def _filter(text: str) -> tuple[str, int, int]:
    """Keep the rows the adapter trains on: policies of ONE known line.

    A package (several lines) and a policy with no line are read against
    ``_fallback.json``, a shape the adapter is not taught, and serving sends
    them to the base model (``ServedRelease.covers_lob``); training on them
    would teach inputs the adapter is never asked. Returns ``(jsonl, kept, dropped)``.
    """
    rows, dropped = _filter_rows(text)
    kept = [line for line, _ in rows]
    return "\n".join(kept) + ("\n" if kept else ""), len(kept), dropped


def _filter_rows(text: str) -> tuple[list[tuple[str, dict]], int]:
    """:func:`_filter`'s rows, each as ``(its JSONL line, the row)``, and the rows dropped."""
    from common.lob import known_lines, single_line

    lines = known_lines()
    kept: list[tuple[str, dict]] = []
    dropped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CorpusViewError(f"corpus row is not JSON: {exc}") from exc
        if row.get("doc_type") in ADAPTER.doc_types and single_line(row.get("lob")) in lines:
            kept.append((line, row))
        else:
            dropped += 1
    return kept, dropped


def _line_of(row: dict) -> str:
    """A row's line by its canonical schema name (``workers_comp`` is ``wc``,
    ``classic_auto`` is ``personal_auto``), as the data mix's per-line settings
    and the release's lines name it; a package's lines joined with ","."""
    from common.lob import lob_lines

    return ",".join(sorted(lob_lines(row.get("lob")))) or "unknown"


def _balance_settings() -> dict[str, int]:
    from common.config import training_config

    settings = training_config(ADAPTER.training_config).get("line_balance") or {}
    return {"min_documents": int(settings.get("min_documents") or 0),
            "max_repeat": int(settings.get("max_repeat") or 1)}


#: What line balance counts a line's size in (``line_balance.count``).
BALANCE_COUNTS = ("documents", "families", "real")


def _balance_options() -> dict[str, Any]:
    """``line_balance.count`` and ``line_balance.distinct_repeats``: today's
    behaviour (documents, identical repeats) when they are absent."""
    from common.config import training_config

    settings = training_config(ADAPTER.training_config).get("line_balance") or {}
    count = settings.get("count") or "documents"
    if count not in BALANCE_COUNTS:
        raise CorpusViewError(f"line_balance.count is {count!r}; expected one of {list(BALANCE_COUNTS)}")
    return {"count": count, "distinct_repeats": bool(settings.get("distinct_repeats"))}


def _unit(row: dict, count: str) -> Any:
    """What one row counts as toward its line's size; None counts for nothing."""
    if count == "families":                       # a seed and its twins are one
        return str(row.get("group_id") or row.get("source_id"))
    if count == "real":
        return None if row.get("synthetic") else str(row.get("source_id"))
    return row.get("source_id")


def line_repeats(rows: list[dict], *, min_documents: int, max_repeat: int,
                 target: int | None = None, count: str = "documents") -> dict[str, int]:
    """How many times each line's rows go into an epoch file (``line_balance``).

    Counted in DOCUMENTS, not rows: a line of long policies has many rows per
    document and is not small for it. ``target`` is the documents a line is
    raised toward when the caller has fixed it (render rotation); by default it
    is ``min_documents``, capped at the largest line.

    ``count`` is what a line's size is counted in: ``documents`` (every render
    of every twin is one; the default), ``families`` (a seed and all its twins
    are one) or ``real`` (synthetic documents count for nothing). Twins are
    near-copies of their seed, so a line of five seeds with twenty twins each is
    small however many documents it holds. A line with nothing to count is
    repeated ``max_repeat`` times.
    """
    if count not in BALANCE_COUNTS:
        raise CorpusViewError(f"line balance cannot count in {count!r}; expected one of {list(BALANCE_COUNTS)}")
    documents: dict[str, set] = {}
    for row in rows:
        units = documents.setdefault(_line_of(row), set())
        unit = _unit(row, count)
        if unit is not None:
            units.add(unit)
    if min_documents <= 0 or max_repeat <= 1 or not documents:
        return {line: 1 for line in documents}
    # Toward the largest line, never past it: when every line is small (a smoke
    # corpus), repeating all of them alike would multiply the run's length and
    # change nothing about the balance.
    if target is None:
        target = min(min_documents, max(len(ids) for ids in documents.values()))
    return {
        line: 1 if len(ids) >= target else max_repeat if not ids else min(max_repeat, -(-target // len(ids)))
        for line, ids in documents.items()
    }


def trained_lobs(view: CorpusView) -> list[object]:
    """Each line the view's train documents are of, as ``assert_one_output_shape``
    reads it. The view holds single-line policies only (:func:`_filter`); a
    package's lines or None would be read as ``_fallback.json``."""
    out: list[object] = []
    for key in (view.examples_by_line.get("train") or {}):
        out.append(None if key == "unknown" else key.split(",") if "," in key else key)
    return out


@dataclass
class RenderRotation:
    """Which documents rest in which epoch, and what each line came to."""

    #: epoch -> source_ids that do not train in it (their twin's other render does).
    rested: dict[int, set[str]] = field(default_factory=dict)
    #: The documents a line is raised toward by line balance (0: balance is off).
    target: int = 0
    #: line -> documents, documents_per_epoch, twins_rotated, both_renders.
    lines: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"enabled": True, "balance_target": self.target, "lines": self.lines}


def _rotation_enabled() -> bool:
    from common.config import training_config

    return bool(training_config(ADAPTER.training_config).get("render_rotation"))


def plan_render_rotation(rows: list[dict], *, seed: int, epochs: int = EPOCH_FILES,
                         min_documents: int = 0) -> RenderRotation:
    """One render per twin per epoch, from one epoch's train rows.

    A twin is a synthetic document's (line, family, twin_index); the documents
    sharing one are its renders, which share one gold. Each epoch trains one of
    them, the next epoch the next. A family's twins are ordered by a seeded
    hash and start on alternate renders, so every epoch holds about as many of
    each render. A line whose documents per epoch would fall under the balance
    target - ``min_documents``, capped at the largest line after rotation, as
    line balance counts it - is not rotated: both renders are distinct inputs,
    where line balance would repeat identical rows.
    """
    import hashlib

    docs: dict[str, dict] = {}
    for row in rows:
        docs.setdefault(str(row.get("source_id")), row)
    by_line: dict[str, set[str]] = {}
    twins: dict[tuple[str, str, int], list[str]] = {}
    for source_id, row in docs.items():
        line = _line_of(row)
        by_line.setdefault(line, set()).add(source_id)
        if row.get("synthetic") and row.get("twin_index") is not None:
            key = (line, str(row.get("group_id") or source_id), int(row["twin_index"]))
            twins.setdefault(key, []).append(source_id)
    rotating = {key: ids for key, ids in twins.items() if len(ids) > 1}
    per_epoch = {line: len(ids) - sum(len(r) - 1 for (ln, _, _), r in rotating.items() if ln == line)
                 for line, ids in by_line.items()}
    plan = RenderRotation()
    if min_documents > 0 and per_epoch:
        plan.target = min(min_documents, max(per_epoch.values()))
    small = {line for line, n in per_epoch.items() if n < plan.target}

    def order(source_id: str) -> tuple:
        row = docs[source_id]                    # digital first, then scanned
        return (bool(row.get("is_scanned")), str(row.get("render_mode") or ""), source_id)

    def position(key: tuple[str, str, int]) -> str:
        return hashlib.sha256(f"{seed}:render_rotation:{key[1]}:{key[2]}".encode()).hexdigest()

    families: dict[tuple[str, str], list[tuple[str, str, int]]] = {}
    for key in rotating:
        if key[0] not in small:
            families.setdefault(key[:2], []).append(key)
    for keys in families.values():
        for index, key in enumerate(sorted(keys, key=lambda k: (position(k), k[2]))):
            renders = sorted(rotating[key], key=order)
            for epoch in range(1, epochs + 1):
                chosen = renders[(index + epoch - 1) % len(renders)]
                plan.rested.setdefault(epoch, set()).update(r for r in renders if r != chosen)
    for line, ids in sorted(by_line.items()):
        rotated = line not in small
        twins_of_line = sum(1 for key in rotating if key[0] == line)
        plan.lines[line] = {
            "documents": len(ids),
            "documents_per_epoch": per_epoch[line] if rotated else len(ids),
            "twins_rotated": twins_of_line if rotated else 0,
            "both_renders": not rotated and twins_of_line > 0,
        }
    return plan


def _rows_of(body: str) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def examples_by_line(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Per line: documents, how many real and synthetic, and rows."""
    documents: dict[str, dict[str, bool]] = {}
    row_counts: dict[str, int] = {}
    for row in rows:
        line = _line_of(row)
        documents.setdefault(line, {})[str(row.get("source_id"))] = bool(row.get("synthetic"))
        row_counts[line] = row_counts.get(line, 0) + 1
    return {
        line: {"documents": len(ids), "real": sum(1 for s in ids.values() if not s),
               "synthetic": sum(1 for s in ids.values() if s), "rows": row_counts[line]}
        for line, ids in sorted(documents.items())
    }


def _balance(body: str, repeats: dict[str, int]) -> tuple[str, int]:
    """``body`` with each row written as many times as its line's repeat count."""
    out: list[str] = []
    for line in body.splitlines():
        if line.strip():
            out.extend([line] * repeats.get(_line_of(json.loads(line)), 1))
    return "\n".join(out) + ("\n" if out else ""), len(out)


def _hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:corpus_view:{value}".encode()).digest()[:8], "big")


# --------------------------------------------------------------------------
# Rehearsal: a continued run's new documents, and a rotated sample of its old ones
# --------------------------------------------------------------------------


@dataclass
class Rehearsal:
    """Which old documents rest in which epoch, and the record of it."""

    #: epoch -> old source_ids that do not train in it.
    rested: dict[int, set[str]] = field(default_factory=dict)
    record: dict = field(default_factory=dict)


def _rehearsal_settings() -> tuple[float | None, bool]:
    """``continued_training.rehearsal_old_share`` and ``rehearsal_weight_by_line_errors``."""
    from common.config import training_config

    block = training_config(ADAPTER.training_config).get("continued_training") or {}
    share = block.get("rehearsal_old_share")
    if share is not None and (isinstance(share, bool) or not isinstance(share, (int, float))
                              or not 0 < share <= 1):
        raise CorpusViewError(f"continued_training.rehearsal_old_share must be a share above 0 and at "
                              f"most 1, or null; got {share!r}")
    return (None if share is None else float(share)), bool(block.get("rehearsal_weight_by_line_errors"))


def _interleaved(carriers: dict[str, list[str]], seed: int) -> list[str]:
    """One line's old documents in an order that spreads every carrier evenly:
    a carrier's documents, ranked by a seeded hash, sit at even steps over the
    whole order. Any run of ``k`` consecutive documents then holds about
    ``k / n`` of each carrier's - which is what makes a rotated window a
    stratified sample."""
    keyed: list[tuple[float, int, str]] = []
    for carrier, ids in carriers.items():
        ranked = sorted(ids, key=lambda s: (_hash(f"rehearsal:{s}", seed), s))
        offset = _hash(f"rehearsal_carrier:{carrier}", seed) / float(1 << 64)
        for index, source_id in enumerate(ranked):
            keyed.append(((index + offset) / len(ranked), _hash(f"rehearsal_carrier:{carrier}", seed), source_id))
    return [source_id for *_, source_id in sorted(keyed)]


def line_error_rates(errors: dict | None) -> dict[str, float]:
    """Per line, its validation error rate in a validation-errors record (``by_line``)."""
    out: dict[str, float] = {}
    for line, counts in ((errors or {}).get("by_line") or {}).items():
        if isinstance(counts, dict):
            rate, values = error_rate(counts)
            if values > 0:
                out[str(line)] = rate
    return out


def _line_shares(share: float, old_by_line: dict[str, int], rates: dict[str, float] | None) -> dict[str, float]:
    """Each line's share of its old documents per epoch: ``share``, or - weighted
    by validation errors - ``share`` times its error rate over the old documents'
    mean rate, at most 1. A line the errors do not name keeps ``share``."""
    known = {line: rates[line] for line in old_by_line if rates and line in rates}
    documents = sum(old_by_line[line] for line in known)
    mean = sum(old_by_line[line] * rate for line, rate in known.items()) / documents if documents else 0.0
    return {line: min(1.0, share * known[line] / mean) if line in known and mean > 0 else share
            for line in old_by_line}


def plan_rehearsal(rows: list[dict], old: set[str], share: float, *, seed: int, epochs: int = EPOCH_FILES,
                   line_rates: dict[str, float] | None = None) -> Rehearsal:
    """Every new document in every epoch; of the old ones, a rotated share per epoch.

    ``old`` are the documents the parent's chain of runs trained on. Per line,
    ``round(share * old documents)`` of them train in each epoch (at least one):
    stratified by line exactly and by carrier closely (:func:`_interleaved`),
    and rotated - epoch ``e`` takes the window after epoch ``e - 1``'s - so
    every old document trains within ``ceil(1 / share)`` epochs. Seeded, so a
    rebuild takes the same ones. ``line_rates`` (validation error rate per line)
    weight the share toward the lines the parent got wrong (:func:`_line_shares`).
    """
    docs: dict[str, dict] = {}
    for row in rows:
        docs.setdefault(str(row.get("source_id")), row)
    old_by_line: dict[str, dict[str, list[str]]] = {}
    new_by_line: dict[str, int] = {}
    new_ids: list[str] = []
    for source_id, row in docs.items():
        line = _line_of(row)
        if source_id in old:
            old_by_line.setdefault(line, {}).setdefault(str(row.get("carrier") or "unknown"), []).append(source_id)
        else:
            new_ids.append(source_id)
            new_by_line[line] = new_by_line.get(line, 0) + 1
    counts = {line: sum(len(ids) for ids in carriers.values()) for line, carriers in old_by_line.items()}
    shares = _line_shares(share, counts, line_rates)
    plan = Rehearsal(rested={e: set() for e in range(1, epochs + 1)})
    rehearsed: dict[int, list[str]] = {e: [] for e in range(1, epochs + 1)}
    lines: dict[str, dict] = {}
    for line, carriers in sorted(old_by_line.items()):
        ordered = _interleaved(carriers, seed)
        n = len(ordered)
        k = min(n, max(1, math.floor(shares[line] * n + 0.5)))
        for epoch in range(1, epochs + 1):
            start = (epoch - 1) * k
            chosen = {ordered[(start + i) % n] for i in range(k)}
            plan.rested[epoch].update(source_id for source_id in ordered if source_id not in chosen)
            rehearsed[epoch].extend(chosen)
        lines[line] = {"new": new_by_line.get(line, 0), "old": n, "carriers": len(carriers),
                       "share": round(shares[line], 4), "old_per_epoch": k,
                       "epochs_to_see_every_old_document": -(-n // k),
                       "error_rate": None if line_rates is None or line not in line_rates
                       else round(line_rates[line], 4)}
    for line, n in new_by_line.items():
        lines.setdefault(line, {"new": n, "old": 0, "carriers": 0, "share": share, "old_per_epoch": 0,
                                "epochs_to_see_every_old_document": 0, "error_rate": None})
    plan.record = {
        "enabled": True, "old_share": share, "weighted_by_line_errors": line_rates is not None,
        "new_documents": len(new_ids), "old_documents": sum(counts.values()),
        "lines": dict(sorted(lines.items())),
        "new": sorted(new_ids),
        "rehearsed_by_epoch": {e: sorted(ids) for e, ids in rehearsed.items()},
    }
    return plan


def chain_trained_documents(client: BlobClient, run_id: str,
                            tenant_id: str | None = None) -> tuple[set[str], str | None]:
    """The documents ``run_id`` and the runs it continued trained on, as
    ``orchestration.pipeline_dag.lineage_training_documents`` reads them (the
    same set the split pins to train): ``(ids, None)``, or ``(ids, why not all)``."""
    from types import SimpleNamespace

    from orchestration.pipeline_dag import lineage_training_documents

    return lineage_training_documents(SimpleNamespace(client=client, tenant_id=tenant_id), run_id)


def promoted_run(client: BlobClient, tenant_id: str | None, out_version: str | None) -> str | None:
    """The run of this tenant's promoted release other than ``out_version``'s,
    as ``orchestration.pipeline_dag.promoted_release`` reads it; None if none."""
    from types import SimpleNamespace

    from orchestration.pipeline_dag import promoted_release

    promoted = promoted_release(SimpleNamespace(client=client, tenant_id=tenant_id, release_id=out_version))
    return ADAPTER.run_id(promoted[1]) if promoted else None


# --------------------------------------------------------------------------
# Hard examples: the fields the last model got most wrong on validation
# --------------------------------------------------------------------------


def _hard_settings() -> dict[str, int] | None:
    """``hard_examples``: ``top_fields`` (None: off), ``max_repeat`` and ``min_values``."""
    from common.config import training_config

    block = training_config(ADAPTER.training_config).get("hard_examples") or {}
    if block.get("top_fields") is None:
        return None
    settings = {"top_fields": block.get("top_fields"), "max_repeat": block.get("max_repeat", 2),
                "min_values": block.get("min_values", 30)}
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise CorpusViewError(f"hard_examples.{key} must be a whole number of at least 1; got {value!r}")
    return settings


def error_rate(counts: dict) -> tuple[float, int]:
    """``(errors / values, values)`` of one validation-errors tally: wrong,
    missed and invented values over the values the gold states."""
    values = int(counts.get("values") or 0)
    errors = sum(int(counts.get(key) or 0) for key in ("wrong", "missed", "invented"))
    return (errors / values if values else 0.0), values


def field_key(path: str) -> str:
    """A field path with its list positions as ``[]`` (``vehicles[0].vin`` -> ``vehicles[].vin``)."""
    return re.sub(r"\[[^\]]*\]", "[]", str(path))


def hard_fields(errors: dict, top_fields: int, *, min_values: int = 30) -> list[dict]:
    """The ``top_fields`` fields of a validation-errors record with the highest
    error rate, among those with at least ``min_values`` gold values (fewer is
    noise) and at least one error; ties go to the field with more values."""
    ranked = []
    for path, counts in ((errors or {}).get("by_field") or {}).items():
        if not isinstance(counts, dict):
            continue
        rate, values = error_rate(counts)
        if values >= min_values and rate > 0:
            ranked.append((-rate, -values, field_key(path), rate, values))
    ranked.sort()
    return [{"field": path, "error_rate": round(rate, 4), "values": values}
            for _, _, path, rate, values in ranked[:top_fields]]


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def target_fields(row: dict) -> set[str]:
    """The fields a row's target states a value for, as validation errors name
    them: the answer in the full model form (a compact answer expanded),
    flattened as scoring flattens it, list positions as ``[]``."""
    messages = row.get("messages") or []
    if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
        return set()
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    try:
        target = json.loads(content or "")
    except (TypeError, ValueError):
        return set()
    if not isinstance(target, dict):
        return set()
    from common.canonical import expand_answer
    from evaluation.metrics.field_accuracy import flatten_scalars

    return {field_key(path) for path, value in flatten_scalars(expand_answer(target)).items() if not _empty(value)}


def holds_any(fields: set[str], wanted: set[str]) -> bool:
    """Whether a row's fields hold one of ``wanted``: the same path, or one ending
    in the other at a ``.`` - a record and a target may root a path differently."""
    for path in fields:
        if path in wanted or any(path.endswith(f".{w}") or w.endswith(f".{path}") for w in wanted):
            return True
    return False


def load_validation_errors(client: BlobClient, run_id: str,
                           tenant_id: str | None = None) -> tuple[dict | None, str | None]:
    """The validation errors of ``run_id``'s winning checkpoint, or ``(None, why not)``.

    Written by checkpoint selection at ``paths.validation_errors_key``; a build
    without that key, or a run that has no record (older than it), has none.
    """
    key_of = getattr(paths, "validation_errors_key", None)
    if key_of is None:
        return None, "this build has no artifact_registry.paths.validation_errors_key"
    key = key_of(run_id, tenant_id=tenant_id)
    if not client.exists(key):
        return None, f"{run_id} has no validation errors at {key}"
    try:
        errors = client.read_json(key)
    except ValueError as exc:  # not JSON
        return None, f"{run_id}'s validation errors at {key} cannot be read ({type(exc).__name__})"
    if not isinstance(errors, dict) or errors.get("version") != 1:
        return None, f"{run_id}'s validation errors at {key} are not version 1"
    return errors, None


# --------------------------------------------------------------------------
# Writing the view
# --------------------------------------------------------------------------


def _epoch_lines(epoch: int, files: dict[int, list[tuple[str, dict]]], docs: set[str],
                 repeats: dict[str, int], *, distinct: bool, hard: set[str] | None = None,
                 hard_extra: int = 0, holds: Any = None) -> tuple[list[str], int]:
    """One epoch file's lines, and how many of them are hard-example repeats.

    The documents ``docs`` train once from this epoch's file. A line repeated
    ``r`` times by line balance writes its rows ``r`` times - identical, or with
    ``distinct``, the repeats from the NEXT epoch files (each its own mode draw
    and windows; the fourth file, which a three-epoch run never reads alone,
    included). A hard example's ``hard_extra`` repeats come from the epoch files
    after those: the rows there of the same documents whose targets hold a hard
    field. An offset past the last file wraps back to this one.
    """
    def source(offset: int) -> list[tuple[str, dict]]:
        return files[(epoch - 1 + offset) % EPOCH_FILES + 1]

    def trains(row: dict) -> bool:
        return str(row.get("source_id")) in docs

    out: list[str] = []
    own = [(text, row) for text, row in files[epoch] if trains(row)]
    if distinct:
        out.extend(text for text, _ in own)
        most = max((repeats.get(_line_of(row), 1) for _, row in own), default=1)
        for offset in range(1, most):
            out.extend(text for text, row in source(offset)
                       if trains(row) and repeats.get(_line_of(row), 1) > offset)
    else:
        for text, row in own:
            out.extend([text] * repeats.get(_line_of(row), 1))
    extra = 0
    if hard and hard_extra > 0:
        most = max(repeats.values(), default=1) if distinct else 1
        for offset in range(1, most + hard_extra):
            for text, row in source(offset):
                if not trains(row):
                    continue
                first = repeats.get(_line_of(row), 1) if distinct else 1
                if first <= offset < first + hard_extra and holds(row):
                    out.append(text)
                    extra += 1
    return out, extra


def materialize(
    corpus_version: str,
    client: BlobClient,
    *,
    tenant_id: str | None = None,
    parent_run_id: str | None = None,
    out_version: str | None = None,
) -> CorpusView:
    """Write the adapter's epoch and validation files, and report them.

    ``parent_run_id`` is the run a continued cycle continues (None: the run
    starts from the base): rehearsal reads which documents its chain trained
    on, and hard examples its validation errors. ``out_version`` is this
    cycle's version, so that a fresh run reading the promoted release's errors
    never takes its own release for it.
    """
    from common.config import training_config
    from training.data_mix import mix_settings, plan_twin_rotation, select_documents

    view = CorpusView()
    files: dict[int, list[tuple[str, dict]]] = {}
    for epoch in range(1, EPOCH_FILES + 1):
        source = paths.corpus_epoch_file(corpus_version, epoch, tenant_id)
        if not client.exists(source):
            raise CorpusViewError(
                f"corpus {corpus_version} has no {source}. The view filters the corpus that "
                "was built; it does not build one of its own."
            )
        files[epoch], dropped = _filter_rows(client.read_text(source))
        view.dropped_rows += dropped

    # One choice of documents and one set of repeat counts for every epoch,
    # from the documents of the first: every epoch file holds every training
    # document once - or, with render rotation, every twin once, in the render
    # that epoch gives it; with twin rotation or rehearsal, that epoch's twins
    # and old documents.
    first = [row for _, row in files[1]]
    settings = mix_settings()
    seed = int(training_config(ADAPTER.training_config).get("seed", 42))
    keep, mix = select_documents(first, settings, seed=seed, line_of=_line_of)
    view.mix_settings = settings.as_dict()
    view.data_mix = {line: m.as_dict() for line, m in mix.items()}
    kept_rows = [row for row in first if str(row.get("source_id")) in keep]
    view.rested_documents = len({str(r.get("source_id")) for r in first}) - len(keep)
    if settings.steers:
        log.info("data mix: %s", {line: (m.real, m.synthetic_kept, m.synthetic_available)
                                  for line, m in mix.items()})
    balance = _balance_settings()
    options = _balance_options()
    rested: dict[int, set[str]] = {e: set() for e in range(1, EPOCH_FILES + 1)}

    errors_cache: dict[str, tuple[dict | None, str | None]] = {}

    def errors_of(run_id: str) -> tuple[dict | None, str | None]:
        if run_id not in errors_cache:
            errors_cache[run_id] = load_validation_errors(client, run_id, tenant_id)
        return errors_cache[run_id]

    share, weighted = _rehearsal_settings()
    if share is not None and parent_run_id is None:
        view.rehearsal = {"enabled": False, "old_share": share,
                          "reason": "the run continues no recorded run (it starts from the base): "
                                    "every document trains every epoch"}
    elif share is not None:
        old, unread = chain_trained_documents(client, parent_run_id, tenant_id)
        if unread:
            # continuation() refuses to continue on an unread chain; a run here
            # anyway trains everything rather than guess which documents are old.
            view.rehearsal = {"enabled": False, "old_share": share,
                              "reason": f"what {parent_run_id}'s chain trained on cannot be read ({unread})"}
            log.warning("rehearsal is off for this run: %s", view.rehearsal["reason"])
        else:
            rates, note = None, None
            if weighted:
                errors, note = errors_of(parent_run_id)
                rates = line_error_rates(errors) if errors else None
            plan = plan_rehearsal(kept_rows, old, share, seed=seed, line_rates=rates)
            plan.record.update({"parent_run_id": parent_run_id, "errors_note": note})
            view.rehearsal = plan.record
            for epoch, ids in plan.rested.items():
                rested[epoch] |= ids
            log.info("rehearsal: %d new document(s) every epoch, %d old one(s) at a share of %s: %s",
                     plan.record["new_documents"], plan.record["old_documents"], share,
                     {line: (r["new"], r["old"], r["old_per_epoch"]) for line, r in plan.record["lines"].items()})

    if settings.twins_per_seed_per_epoch:
        twins = plan_twin_rotation(kept_rows, settings.twins_per_seed_per_epoch, seed=seed,
                                   epochs=EPOCH_FILES, line_of=_line_of)
        view.twin_rotation = twins.as_dict()
        for epoch, ids in twins.rested.items():
            rested[epoch] |= ids
        log.info("twin rotation: %d twin(s) of every seed per epoch: %s", twins.per_seed,
                 {line: (r["twins"], r["twins_per_epoch"]) for line, r in twins.lines.items()})

    rotation: RenderRotation | None = None
    if _rotation_enabled():
        # After the shares (they choose documents, rotation chooses a render
        # per epoch) and before balance, which counts one epoch.
        rotation = plan_render_rotation(
            kept_rows, seed=seed,
            min_documents=balance["min_documents"] if balance["max_repeat"] > 1 else 0)
        view.render_rotation = rotation.as_dict()
        for epoch, ids in rotation.rested.items():
            rested[epoch] |= ids
        log.info("render rotation: documents per epoch %s", {
            line: (r["documents"], r["documents_per_epoch"]) for line, r in rotation.lines.items()})

    epoch_rows = [row for row in kept_rows if str(row.get("source_id")) not in rested[1]]
    view.examples_by_line["train"] = examples_by_line(epoch_rows)
    # Render rotation fixes the documents a line is raised toward before a
    # small line keeps both renders. Counted in families or real documents, or
    # with twin rotation or rehearsal thinning the epoch, the target is counted
    # on the epoch's own rows instead.
    fixed = (rotation is not None and options["count"] == "documents"
             and not view.twin_rotation and not view.rehearsal.get("enabled"))
    view.line_repeats = line_repeats(epoch_rows, **balance, count=options["count"],
                                     target=(rotation.target or None) if fixed else None)
    view.line_balance = {**balance, **options}
    for line, record in view.data_mix.items():
        record["repeats"] = view.line_repeats.get(line, 1)
    boosted = {line: n for line, n in view.line_repeats.items() if n > 1}
    if boosted:
        log.info("line balance: repeating %s in every epoch file%s", boosted,
                 ", each repeat from another epoch file" if options["distinct_repeats"] else "")

    hard = _hard_settings()
    wanted: set[str] | None = None
    if hard is not None:
        try:
            errors_run = parent_run_id or promoted_run(client, tenant_id, out_version)
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            errors_run, reason = None, f"the promoted release cannot be read ({type(exc).__name__}: {exc})"
        else:
            reason = "no release is promoted and the run continues none" if errors_run is None else None
        errors = None
        if errors_run is not None:
            errors, reason = errors_of(errors_run)
        fields = hard_fields(errors, hard["top_fields"], min_values=hard["min_values"]) if errors else []
        if errors is not None and not fields:
            reason = f"no field of {errors_run}'s validation errors has {hard['min_values']}+ values and an error"
        view.hard_examples = {"enabled": bool(fields), **hard, "errors_run": errors_run, "fields": fields}
        if fields:
            wanted = {entry["field"] for entry in fields}
            view.hard_examples["rows_repeated_by_epoch"] = {}
            log.info("hard examples: repeating the windows holding %s", sorted(wanted))
        else:
            view.hard_examples["reason"] = reason
            log.warning("hard examples are configured but not applied: %s", reason)

    holds_cache: dict[int, bool] = {}

    def holds(row: dict) -> bool:
        if id(row) not in holds_cache:
            holds_cache[id(row)] = holds_any(target_fields(row), wanted or set())
        return holds_cache[id(row)]

    for epoch in range(1, EPOCH_FILES + 1):
        docs = keep - rested[epoch]
        if options["distinct_repeats"] or wanted:
            out, extra = _epoch_lines(epoch, files, docs, view.line_repeats,
                                      distinct=options["distinct_repeats"], hard=wanted,
                                      hard_extra=hard["max_repeat"] - 1 if hard else 0, holds=holds)
            body, kept = "\n".join(out) + ("\n" if out else ""), len(out)
            if wanted:
                view.hard_examples["rows_repeated_by_epoch"][epoch] = extra
        else:
            own = [text for text, row in files[epoch] if str(row.get("source_id")) in docs]
            body, kept = _balance("\n".join(own), view.line_repeats)
        target = paths.corpus_scope_epoch_file(corpus_version, epoch, ADAPTER.name, tenant_id)
        client.write_text(target, body)
        view.epoch_files.append(target)
        view.rows_by_epoch[epoch] = kept

    if not view.train_rows:
        raise CorpusViewError(
            f"the adapter trains {list(ADAPTER.doc_types)}, and corpus {corpus_version} holds "
            "no training rows of them. Training on nothing would produce an adapter that "
            "reports success and learned nothing."
        )

    source = paths.corpus_eval_split(corpus_version, "val", tenant_id)
    if not client.exists(source):
        raise CorpusViewError(f"corpus {corpus_version} has no validation split at {source}")
    body, kept, _ = _filter(client.read_text(source))
    view.val_path = paths.corpus_scope_eval_split(corpus_version, "val", ADAPTER.name, tenant_id)
    client.write_text(view.val_path, body)
    view.val_rows = kept
    view.examples_by_line["val"] = examples_by_line(_rows_of(body))

    test = paths.corpus_eval_split(corpus_version, "test", tenant_id)
    if client.exists(test):
        test_body, view.test_rows, _ = _filter(client.read_text(test))
        view.examples_by_line["test"] = examples_by_line(_rows_of(test_body))

    if not view.val_rows:
        # Not fatal here — it is the checkpoint selector and the gate that need
        # validation rows, and they say so themselves. But a run that trains with
        # no validation has no early stopping and no checkpoint selection, and an
        # operator should learn that before the GPU bill, not after.
        log.warning(
            "the adapter has no validation rows in corpus %s: early stopping and checkpoint "
            "selection will have nothing to read.", corpus_version,

        )

    log.info("corpus view %s", view.describe())
    return view
