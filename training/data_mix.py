"""Per-line synthetic fraction and scanned share of the adapter's training set
(Fideon SPEC_09 amendment items 6 and 7).

The corpus is built once and the adapter reads a view of it
(:mod:`training.corpus_view`), so the shares are applied where the view is
written: it keeps, line by line, every real document and the synthetic ones
that bring the line to its configured shares. Verified real golds always
train. Validation and test are never sampled.

Configured in the training config::

    data_mix:
      synthetic_fraction: 0.8   # synthetic share of a line's train documents
      scanned_share: null       # scanned share of a line's train documents
      lines:                    # per line, over the defaults above
        homeowners: {scanned_share: 0.4}

A per-line ``null`` (or a share left out) takes the default; a default of
``null`` (or absent) leaves that share unsteered. With neither share set, every
document trains, as before these settings existed. Counted in documents, as
``line_balance`` counts them: a long policy has many rows and is one example.

The input-mode mix (50% OCR text and images, 20% corrupted OCR and images, 30%
images only) is a separate setting: :data:`common.constants.MODALITY_MIX` is
the global default, and the training config may override it with
``modality_mix:``. Modes are drawn when the corpus is built, so the corpus
records the mix it was drawn with and a run that wants another is refused
(:func:`assert_corpus_mix`).
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from common.constants import MODALITY_MIX, MODALITY_MODES

log = logging.getLogger(__name__)

SHARES = ("synthetic_fraction", "scanned_share")


class DataMixError(RuntimeError):
    """Raised on a data mix that is malformed or cannot be applied."""


@dataclass(frozen=True)
class MixSettings:
    """The configured shares: the defaults and the per-line values."""

    synthetic_fraction: float | None = None
    scanned_share: float | None = None
    lines: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def for_line(self, line: str) -> tuple[float | None, float | None]:
        entry = self.lines.get(line, {})
        return (entry.get("synthetic_fraction", self.synthetic_fraction),
                entry.get("scanned_share", self.scanned_share))

    @property
    def steers(self) -> bool:
        return any(v is not None for v in (self.synthetic_fraction, self.scanned_share)) or any(
            v is not None for entry in self.lines.values() for v in entry.values())

    @property
    def needs_scan_flag(self) -> bool:
        return self.scanned_share is not None or any(
            entry.get("scanned_share") is not None for entry in self.lines.values())

    def as_dict(self) -> dict[str, Any]:
        return {"synthetic_fraction": self.synthetic_fraction, "scanned_share": self.scanned_share,
                "lines": {line: dict(entry) for line, entry in sorted(self.lines.items())}}


def _share(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise DataMixError(f"{where} must be a share between 0 and 1, or null; got {value!r}")
    return float(value)


def parse_settings(raw: dict[str, Any] | None, *, lines: frozenset[str] = frozenset()) -> MixSettings:
    """``data_mix`` of a training config. ``lines``, when given, are the only
    lines a per-line entry may name: a misspelt line would otherwise fall back
    to the defaults without a word."""
    raw = raw or {}
    unknown = set(raw) - {*SHARES, "lines"}
    if unknown:
        raise DataMixError(f"data_mix has unknown key(s) {sorted(unknown)}; expected {[*SHARES, 'lines']}")
    per_line: dict[str, dict[str, float | None]] = {}
    for line, entry in (raw.get("lines") or {}).items():
        if lines and not set(str(line).split(",")) <= lines:
            raise DataMixError(f"data_mix names line {line!r}, which is no known line "
                               f"({sorted(lines)})")
        entry = entry or {}
        if set(entry) - set(SHARES):
            raise DataMixError(f"data_mix.lines.{line} has unknown key(s) {sorted(set(entry) - set(SHARES))}")
        # A null is "not set for this line": the default applies. And a
        # package's lines in sorted order, as the view joins them (corpus_view._line_of).
        key = ",".join(sorted(part.strip() for part in str(line).split(",")))
        per_line[key] = {k: share for k, v in entry.items()
                         if (share := _share(v, f"data_mix.lines.{line}.{k}")) is not None}
    return MixSettings(_share(raw.get("synthetic_fraction"), "data_mix.synthetic_fraction"),
                       _share(raw.get("scanned_share"), "data_mix.scanned_share"), per_line)


def mix_settings() -> MixSettings:
    from common.adapter import ADAPTER
    from common.config import training_config
    from common.lob import known_lines

    return parse_settings(training_config(ADAPTER.training_config).get("data_mix"),
                          lines=known_lines())


def configured_modality_mix() -> dict[str, float]:
    """The input-mode mix the adapter trains on: the config's ``modality_mix`` or the global default."""
    from common.adapter import ADAPTER
    from common.config import training_config

    mix = training_config(ADAPTER.training_config).get("modality_mix")
    if not mix:
        return dict(MODALITY_MIX)
    if set(mix) - set(MODALITY_MODES):
        raise DataMixError(f"modality_mix names unknown mode(s) {sorted(set(mix) - set(MODALITY_MODES))}; "
                           f"the modes are {list(MODALITY_MODES)}")
    full = {mode: float(mix.get(mode, 0.0)) for mode in MODALITY_MODES}
    if abs(sum(full.values()) - 1.0) > 1e-6:
        raise DataMixError(f"modality_mix sums to {sum(full.values())}, not 1.0: {full}")
    return full


def assert_corpus_mix(corpus_manifest: dict[str, Any]) -> None:
    """Refuse a run that wants another input-mode mix than the corpus was drawn with.

    A corpus built before the target was recorded is not checked: it was drawn
    with the global default, which is all a run could have configured then.
    """
    recorded = corpus_manifest.get("modality_mix_target")
    if not recorded:
        return
    wanted = configured_modality_mix()
    if any(abs(float(recorded.get(m, 0.0)) - wanted[m]) > 1e-6 for m in MODALITY_MODES):
        raise DataMixError(
            f"the training config asks for input-mode mix {wanted}, but corpus "
            f"{corpus_manifest.get('corpus_version', '?')} was drawn with {recorded}. Modes are drawn "
            "when the corpus is built: rebuild it so the rows carry the mix the run records."
        )



# --------------------------------------------------------------------------
# Choosing the documents
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Doc:
    source_id: str
    family: str
    synthetic: bool
    scanned: bool | None


@dataclass
class LineMix:
    """One line's train documents, and the shares configured and reached."""

    synthetic_fraction: float | None = None
    scanned_share: float | None = None
    real: int = 0
    synthetic_available: int = 0
    synthetic_kept: int = 0
    #: Scanned documents among those kept; None when the rows do not say.
    scanned_kept: int | None = None
    note: str | None = None

    @property
    def documents(self) -> int:
        return self.real + self.synthetic_kept

    def as_dict(self) -> dict[str, Any]:
        n = self.documents
        return {
            "synthetic_fraction": self.synthetic_fraction,
            "scanned_share": self.scanned_share,
            "real": self.real,
            "synthetic_available": self.synthetic_available,
            "synthetic_kept": self.synthetic_kept,
            "scanned_kept": self.scanned_kept,
            "documents": n,
            "realised_synthetic_fraction": round(self.synthetic_kept / n, 4) if n else None,
            "realised_scanned_share": (round(self.scanned_kept / n, 4)
                                       if n and self.scanned_kept is not None else None),
            "note": self.note,
        }


def _hash(value: str, seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:data_mix:{value}".encode()).digest()[:8], "big")


def _round_robin(docs: list[_Doc], seed: int) -> list[_Doc]:
    """``docs`` in the order they are taken: one twin of every seed, then a
    second of every seed, and so on - a budget spreads over the seeds instead of
    spending itself on the first few. Seeded, so a rebuild takes the same ones."""
    by_family: dict[str, list[_Doc]] = defaultdict(list)
    for doc in docs:
        by_family[doc.family].append(doc)
    for twins in by_family.values():
        twins.sort(key=lambda d: (_hash(d.source_id, seed), d.source_id))
    families = sorted(by_family, key=lambda f: (_hash(f, seed), f))
    ordered: list[_Doc] = []
    depth = 0
    while len(ordered) < len(docs):
        ordered.extend(by_family[f][depth] for f in families if depth < len(by_family[f]))
        depth += 1
    return ordered


def _nearest(x: float) -> int:
    return math.floor(x + 0.5)


def synthetic_budget(fraction: float | None, real: int, available: int) -> int:
    """How many synthetic documents bring a line with ``real`` real ones to ``fraction``.

    ``k / (real + k) = fraction`` gives ``k = fraction * real / (1 - fraction)``,
    to the nearest document - within half a document of the configured share.
    Every one when the fraction is unset or 1, or the line has no real document
    (there is nothing to take a ratio against).
    """
    if fraction is None or fraction >= 1 or real == 0:
        return available
    return min(available, _nearest(fraction * real / (1 - fraction)))


def _scanned_split(budget: int, real: int, real_scanned: int, scanned_pool: int, digital_pool: int,
                   share: float) -> tuple[int, int, str | None]:
    """``(scanned, digital, note)`` synthetic documents to take for ``share``.

    The scanned share decides what the model is shown, so it comes first: when
    one pool runs short, fewer synthetic documents are taken rather than the
    share missed. When no number of synthetic documents reaches it - the real
    ones alone are further off than the pools can correct - the whole budget is
    taken as close to the share as the pools allow, and the note says so.
    """
    for total in range(budget, -1, -1):
        scanned = _nearest(share * (real + total)) - real_scanned
        digital = total - scanned
        if 0 <= scanned <= scanned_pool and 0 <= digital <= digital_pool:
            note = None if total == budget else (
                f"{budget - total} synthetic document(s) fewer than synthetic_fraction asks, "
                "to keep the scanned share")
            return scanned, digital, note
    scanned = min(max(_nearest(share * (real + budget)) - real_scanned, budget - digital_pool, 0),
                  scanned_pool, budget)
    return scanned, budget - scanned, (
        "scanned_share is out of reach: the real documents and the synthetic pools cannot make it")


def plan_line(docs: list[_Doc], synthetic_fraction: float | None, scanned_share: float | None,
              *, seed: int) -> tuple[list[str], LineMix]:
    """The source ids of one line that train, and what they come to."""
    real = [d for d in docs if not d.synthetic]
    synthetic = [d for d in docs if d.synthetic]
    known = all(d.scanned is not None for d in docs)
    mix = LineMix(synthetic_fraction, scanned_share, real=len(real), synthetic_available=len(synthetic))
    budget = synthetic_budget(synthetic_fraction, len(real), len(synthetic))
    if synthetic_fraction is not None and synthetic_fraction < 1 and not real and synthetic:
        mix.note = "no real document in the line: synthetic_fraction cannot apply, every synthetic one kept"
    if scanned_share is None:
        chosen = _round_robin(synthetic, seed)[:budget]
    else:
        scanned_pool = _round_robin([d for d in synthetic if d.scanned], seed)
        digital_pool = _round_robin([d for d in synthetic if not d.scanned], seed)
        n_scanned, n_digital, note = _scanned_split(
            budget, len(real), sum(1 for d in real if d.scanned),
            len(scanned_pool), len(digital_pool), scanned_share)
        chosen = scanned_pool[:n_scanned] + digital_pool[:n_digital]
        mix.note = note or mix.note
    mix.synthetic_kept = len(chosen)
    if known:
        mix.scanned_kept = sum(1 for d in (*real, *chosen) if d.scanned)
    return [d.source_id for d in (*real, *chosen)], mix


def select_documents(rows: list[dict[str, Any]], settings: MixSettings, *, seed: int,
                     line_of: Any) -> tuple[set[str], dict[str, LineMix]]:
    """The train documents a scope keeps, and each line's record.

    ``rows`` are one epoch's train rows of the scope; ``line_of`` reads a row's
    line as the view counts it. Every document is kept when nothing is steered;
    the record is made either way, for the run manifest and the model card.
    """
    docs: dict[str, _Doc] = {}
    lines: dict[str, list[_Doc]] = defaultdict(list)
    for row in rows:
        source_id = str(row.get("source_id"))
        if source_id in docs:
            continue
        scanned = row.get("is_scanned")
        doc = _Doc(source_id, str(row.get("group_id") or source_id), bool(row.get("synthetic")),
                   None if scanned is None else bool(scanned))
        docs[source_id] = doc
        lines[line_of(row)].append(doc)

    if settings.needs_scan_flag and any(d.scanned is None for d in docs.values()):
        raise DataMixError(
            "scanned_share is configured, but this corpus's rows do not record whether their "
            "document is scanned (is_scanned). Rebuild the corpus: guessing would sample the "
            "wrong documents.")

    keep: set[str] = set()
    record: dict[str, LineMix] = {}
    for line, line_docs in sorted(lines.items()):
        fraction, share = settings.for_line(line)
        kept, record[line] = plan_line(line_docs, fraction, share, seed=seed)
        keep.update(kept)
        if record[line].note:
            log.warning("data mix, %s: %s", line, record[line].note)
    return keep, record
