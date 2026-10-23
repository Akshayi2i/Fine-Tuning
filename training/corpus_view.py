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
and 7). Every real document trains.

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

import json
import logging
from dataclasses import dataclass, field

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
    from common.lob import known_lines, single_line

    lines = known_lines()
    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CorpusViewError(f"corpus row is not JSON: {exc}") from exc
        if row.get("doc_type") in ADAPTER.doc_types and single_line(row.get("lob")) in lines:
            kept.append(line)
        else:
            dropped += 1
    return "\n".join(kept) + ("\n" if kept else ""), len(kept), dropped


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


def line_repeats(rows: list[dict], *, min_documents: int, max_repeat: int,
                 target: int | None = None) -> dict[str, int]:
    """How many times each line's rows go into an epoch file (``line_balance``).

    Counted in DOCUMENTS, not rows: a line of long policies has many rows per
    document and is not small for it. ``target`` is the documents a line is
    raised toward when the caller has fixed it (render rotation); by default it
    is ``min_documents``, capped at the largest line.
    """
    documents: dict[str, set] = {}
    for row in rows:
        documents.setdefault(_line_of(row), set()).add(row.get("source_id"))
    if min_documents <= 0 or max_repeat <= 1 or not documents:
        return {line: 1 for line in documents}
    # Toward the largest line, never past it: when every line is small (a smoke
    # corpus), repeating all of them alike would multiply the run's length and
    # change nothing about the balance.
    if target is None:
        target = min(min_documents, max(len(ids) for ids in documents.values()))
    return {
        line: 1 if len(ids) >= target else min(max_repeat, -(-target // len(ids)))
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


def _keep(body: str, keep: set[str]) -> tuple[str, int]:
    """``body`` with only the rows of the documents in ``keep``."""
    out = [line for line in body.splitlines()
           if line.strip() and str(json.loads(line).get("source_id")) in keep]
    return "\n".join(out) + ("\n" if out else ""), len(out)


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


def materialize(
    corpus_version: str,
    client: BlobClient,
    *,
    tenant_id: str | None = None,
) -> CorpusView:
    """Write the adapter's epoch and validation files, and report them."""
    view = CorpusView()
    rotation: RenderRotation | None = None
    for epoch in range(1, EPOCH_FILES + 1):
        source = paths.corpus_epoch_file(corpus_version, epoch, tenant_id)
        if not client.exists(source):
            raise CorpusViewError(
                f"corpus {corpus_version} has no {source}. The view filters the corpus that "
                "was built; it does not build one of its own."
            )
        body, kept, dropped = _filter(client.read_text(source))
        if epoch == 1:
            # One choice of documents and one set of repeat counts for every
            # epoch, from the documents of the first: every epoch file holds
            # every training document once - or, with render rotation, every
            # twin once, in the render that epoch gives it.
            from common.config import training_config
            from training.data_mix import mix_settings, select_documents

            settings = mix_settings()
            seed = int(training_config(ADAPTER.training_config).get("seed", 42))
            keep, mix = select_documents(_rows_of(body), settings, seed=seed, line_of=_line_of)
            view.mix_settings = settings.as_dict()
            view.data_mix = {line: m.as_dict() for line, m in mix.items()}
            kept_rows = [row for row in _rows_of(body) if str(row.get("source_id")) in keep]
            view.rested_documents = len({str(r.get("source_id")) for r in _rows_of(body)}) - len(keep)
            if settings.steers:
                log.info("data mix: %s", {line: (m.real, m.synthetic_kept, m.synthetic_available)
                                          for line, m in mix.items()})
            balance = _balance_settings()
            if _rotation_enabled():
                # After the shares (they choose documents, rotation chooses a
                # render per epoch) and before balance, which counts one epoch.
                rotation = plan_render_rotation(
                    kept_rows, seed=seed,
                    min_documents=balance["min_documents"] if balance["max_repeat"] > 1 else 0)
                view.render_rotation = rotation.as_dict()
                log.info("render rotation: documents per epoch %s", {
                    line: (r["documents"], r["documents_per_epoch"]) for line, r in rotation.lines.items()})
            first = rotation.rested.get(1, set()) if rotation else set()
            epoch_rows = [row for row in kept_rows if str(row.get("source_id")) not in first]
            view.examples_by_line["train"] = examples_by_line(epoch_rows)
            view.line_repeats = line_repeats(epoch_rows, **balance,
                                             target=(rotation.target or None) if rotation else None)
            for line, record in view.data_mix.items():
                record["repeats"] = view.line_repeats.get(line, 1)
            boosted = {line: n for line, n in view.line_repeats.items() if n > 1}
            if boosted:
                log.info("line balance: repeating %s in every epoch file", boosted)
        rested = rotation.rested.get(epoch, set()) if rotation else set()
        body, kept = _keep(body, keep - rested)
        body, kept = _balance(body, view.line_repeats)
        target = paths.corpus_scope_epoch_file(corpus_version, epoch, ADAPTER.name, tenant_id)
        client.write_text(target, body)
        view.epoch_files.append(target)
        view.rows_by_epoch[epoch] = kept
        view.dropped_rows += dropped

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
