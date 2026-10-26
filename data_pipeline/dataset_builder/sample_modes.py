"""Per-document, per-epoch modality sampling (arch v2.1 §6.1).

**What v1 did and why it was wrong.** Each source document was expanded into
three rows — one per modality regime — and all three went into the corpus. That
had two consequences nobody intended:

* The realised mix was **33/33/33**, not the 50/20/30 the architecture specifies.
  A downstream sampler tried to correct it by discarding rows, which threw away
  labelled data to fix a shape problem.
* Every document was seen **three times per epoch**, so a "3 epoch" run was nine
  passes over the corpus. The epoch count in the manifest described something
  other than what ran, and the early-stopping patience was counted in the wrong
  unit.

**What happens instead.** Each source document appears **once per epoch**, and
its mode for that epoch is drawn from a seeded generator against the target mix.
Over three or four epochs a document is seen in two or three different regimes,
which is the modality-dropout behaviour the design wanted — without inflating the
epoch count or discarding anything.

The corpus materializes four epoch files regardless of how many a run uses
(§8.1), because the §11a epoch sweep tests up to four passes and a sweep that
regenerates its own data is not comparing what it thinks it is.

A three-epoch run reads the first three as its passes; the others are not
wasted. With ``line_balance.distinct_repeats`` or ``hard_examples`` on, the
adapter's view (``training.corpus_view``) fills a repeated document's extra
copies in one epoch from the NEXT epoch files - the fourth included - so each
copy has its own independently drawn mode and windows instead of being the
same row again.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from common.constants import MODALITY_MIX, MODALITY_MODES

log = logging.getLogger(__name__)

#: Always materialized, whatever a given run uses (arch v2.1 §6.1). A
#: three-epoch run trains on the first three; the view's distinct repeats draw
#: on all four (see the module docstring).
EPOCH_FILES = 4

#: Below this many draws the mix check only warns. A regime's realised share has
#: standard deviation sqrt(p(1-p)/n), at most 0.5/sqrt(n); a three-sigma swing
#: stays inside a 5-point tolerance only from n = 900. At pilot volume (30 train
#: documents x 4 epochs = 120 draws) sigma is ~4.6 points, so an enforced check
#: would fail correct corpora routinely and teach everyone to ignore it.
MIN_DRAWS_FOR_MIX_CHECK = 900


class ModeSamplingError(RuntimeError):
    """Raised when a mode draw cannot be made or is not usable."""


@dataclass
class ModeAssignment:
    """Which regime each document is shown in, per epoch."""

    #: (source_id, epoch) -> mode
    draws: dict[tuple[str, int], str] = field(default_factory=dict)
    seed: int = 42
    epochs: int = EPOCH_FILES

    def mode_for(self, source_id: str, epoch: int) -> str:
        try:
            return self.draws[(source_id, epoch)]
        except KeyError:
            raise ModeSamplingError(
                f"no mode drawn for {source_id!r} in epoch {epoch}. Every document appears "
                "exactly once per epoch (arch v2.1 §6.1)."
            ) from None

    def realised_mix(self, epoch: int | None = None) -> dict[str, float]:
        """The share each regime actually got, over one epoch or all of them."""
        counts = Counter(
            mode for (_, e), mode in self.draws.items()
            if epoch is None or e == epoch
        )
        total = sum(counts.values())
        return {m: (counts.get(m, 0) / total if total else 0.0) for m in MODALITY_MODES}

    def modes_seen_by(self, source_id: str) -> list[str]:
        """Which regimes one document is shown in across the whole run.

        The point of per-epoch sampling: a document seen only ever with clean OCR
        teaches nothing about arbitration.
        """
        return [
            self.draws[(source_id, e)]
            for e in range(1, self.epochs + 1)
            if (source_id, e) in self.draws
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "realised_mix_overall": {k: round(v, 4) for k, v in self.realised_mix().items()},
            "realised_mix_by_epoch": {
                e: {k: round(v, 4) for k, v in self.realised_mix(e).items()}
                for e in range(1, self.epochs + 1)
            },
        }


def _draw(source_id: str, epoch: int, seed: int) -> float:
    """A deterministic value in [0, 1) for one document in one epoch.

    Seeded on all three so the corpus is reproducible from the manifest, and so
    two epochs draw independently — hashing on the document alone would give it
    the same mode every epoch, which is the v1 behaviour with extra steps.
    """
    digest = hashlib.sha256(f"{seed}:{epoch}:{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def sample_modes(
    source_ids: list[str],
    *,
    seed: int = 42,
    epochs: int = EPOCH_FILES,
    mix: dict[str, float] | None = None,
) -> ModeAssignment:
    """Draw one modality regime per document per epoch.

    Drawn against cumulative thresholds rather than shuffled, so adding documents
    later does not change the mode any existing document was shown in.

    The mix is approached statistically, not enforced exactly. Forcing exact
    per-epoch counts would mean the draw for one document depends on the draws
    for all the others, and a corpus rebuilt after one new document arrives would
    reassign modes across the board.
    """
    target = mix or MODALITY_MIX
    total = sum(target.values())
    if abs(total - 1.0) > 1e-6:
        raise ModeSamplingError(
            f"modality mix sums to {total}, not 1.0: {target}. A mix that does not sum to one "
            "silently starves whichever regime falls off the end of the thresholds."
        )

    # Cumulative edges, in the canonical mode order so the thresholds are stable.
    edges: list[tuple[float, str]] = []
    running = 0.0
    for mode in MODALITY_MODES:
        running += target.get(mode, 0.0)
        edges.append((running, mode))

    assignment = ModeAssignment(seed=seed, epochs=epochs)
    for source_id in sorted(set(source_ids)):
        for epoch in range(1, epochs + 1):
            position = _draw(source_id, epoch, seed)
            mode = next((m for edge, m in edges if position < edge), MODALITY_MODES[-1])
            assignment.draws[(source_id, epoch)] = mode

    realised = assignment.realised_mix()
    drift = {m: round(realised[m] - target.get(m, 0.0), 3) for m in MODALITY_MODES}
    log.info(
        "sampled %d documents x %d epochs; realised mix %s (drift from target %s)",
        len(set(source_ids)), epochs,
        {k: round(v, 3) for k, v in realised.items()}, drift,
    )
    return assignment


def assert_mix_is_close(
    assignment: ModeAssignment,
    *,
    tolerance: float = 0.05,
    mix: dict[str, float] | None = None,
    minimum_draws: int = MIN_DRAWS_FOR_MIX_CHECK,
) -> None:
    """Assert the realised mix is near the target (arch §6).

    Tolerance is wider than v1's 0.02 and that is not a weakening: v1 constructed
    the mix by discarding rows, so it could hit any number exactly. A sampled mix
    varies, and at pilot volume — 25-30 documents per type — a single document
    moves a regime's share by three points. A tight bound would fail on ordinary
    sampling noise and teach everyone to ignore it.
    """
    target = mix or MODALITY_MIX
    draws = len(assignment.draws)
    if draws < minimum_draws:
        # Reported, not silently skipped: an unenforced check that looks enforced
        # is worse than one that says it is not running.
        log.warning(
            "%d modality draws is below the %d needed for a %.0f%% tolerance to hold against "
            "ordinary sampling noise; the mix check is not enforced at this size. Realised: %s",
            draws, minimum_draws, tolerance * 100,
            {k: round(v, 3) for k, v in assignment.realised_mix().items()},
        )
        return
    realised = assignment.realised_mix()
    off = {
        mode: (round(realised[mode], 3), target.get(mode, 0.0))
        for mode in MODALITY_MODES
        if abs(realised[mode] - target.get(mode, 0.0)) > tolerance
    }
    if off:
        raise ModeSamplingError(
            f"realised modality mix is more than {tolerance:.0%} from target: {off}. "
            "image_only is the regime with no OCR backstop, so under-sampling it means the "
            "image-only production path is the least-trained one (arch §6)."
        )
