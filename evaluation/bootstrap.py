"""Paired bootstrap over documents (arch v2.1 §15.5).

The v1 gate compared two point estimates against a flat tolerance of 0.001 and
blocked on any movement larger than that. At pilot volume the eval set holds
three or four documents per type, so a metric like classification accuracy can
only take the values 0, 0.25, 0.5, 0.75, 1.0 — the tolerance was **two hundred
and fifty times finer than the measurement could resolve**. And with twelve
metrics each needing to not move down, a genuinely-equal model passed all twelve
with probability near 0.5¹² ≈ 0.02%.

The result was a gate that could not be passed, deliberately built with no
override. That combination does not produce caution; it produces a rule everyone
agrees to ignore.

**What replaces it.** Resample the evaluation DOCUMENTS with replacement, rescore
both models on each resample, and read the distribution of the difference. Paired
because both models saw the same documents — comparing independent samples would
throw away the pairing and widen every interval for nothing.

Two questions, asked separately:

* **Non-inferiority** — is the candidate no worse than the current release by
  more than δ? The lower bound of the 95% CI on (candidate − current) must sit
  above −δ. This is what "did not regress" means when the measurement is noisy.
* **Improvement** — is at least one primary metric actually better? The lower
  bound must sit above zero. Without this a release could pass forever on
  non-inferiority alone, never improving.

Pure standard library. numpy would be faster, but this runs once per gate on a
few hundred documents, and the dependency is not worth the millisecond.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: Resamples per comparison. 10,000 is the arch §15.5 figure: enough that the
#: 2.5th percentile is stable to about a thousandth, which is finer than any δ
#: the gate uses.
RESAMPLES = 10_000

#: Fixed so a gate decision is reproducible. A gate whose verdict changes on
#: re-run is not a gate.
SEED = 42


@dataclass
class ConfidenceInterval:
    """The bootstrap distribution of (candidate − current) for one metric."""

    metric: str
    observed: float
    lower: float
    upper: float
    resamples: int = RESAMPLES
    documents: int = 0

    def non_inferior(self, delta: float) -> bool:
        """Whether the candidate is no worse by more than ``delta``.

        Reads the LOWER bound: the question is how bad this could plausibly be,
        not how good. A point estimate that improved can still have a lower bound
        below −δ when the eval set is small, and that is the honest answer —
        there is not enough evidence to rule out a real regression.
        """
        return self.lower > -abs(delta)

    @property
    def improved(self) -> bool:
        """Whether the improvement is distinguishable from noise."""
        return self.lower > 0.0

    @property
    def width(self) -> float:
        return round(self.upper - self.lower, 6)

    def describe(self) -> str:
        return (
            f"{self.metric}: {self.observed:+.4f} "
            f"[95% CI {self.lower:+.4f}, {self.upper:+.4f}] over {self.documents} document(s)"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "observed_delta": round(self.observed, 6),
            "ci_lower": round(self.lower, 6),
            "ci_upper": round(self.upper, 6),
            "ci_width": self.width,
            "documents": self.documents,
            "resamples": self.resamples,
        }


def paired_bootstrap(
    metric: str,
    current: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    aggregate: Callable[[Sequence[float]], float] | None = None,
    current_weights: Sequence[float] | None = None,
    candidate_weights: Sequence[float] | None = None,
) -> ConfidenceInterval:
    """The 95% CI of (candidate − current), resampling documents with replacement.

    ``current`` and ``candidate`` are **per-document** scores, aligned by index —
    element *i* of each is the same document. That alignment is the pairing, and
    it is what makes the interval tight enough to be useful on a small eval set.

    ``current_weights`` / ``candidate_weights``: how much each document counts in
    its side's metric, aligned the same way. Given, each side of a resample is
    the weighted mean, sum(w·x) / sum(w), over the documents drawn. A metric the
    report averages over ITEMS rather than documents needs it: list_field_recall
    is the mean over every table in the set, so a policy with four tables counts
    four times, and an unweighted mean of per-document scores tested a different
    number from the one the gate reports and the floor reads.
    """
    if len(current) != len(candidate):
        raise ValueError(
            f"{metric}: {len(current)} baseline scores against {len(candidate)} candidate "
            "scores. They must be per-document and aligned by index, or the pairing is lost "
            "and the comparison is between two different document sets."
        )
    n = len(current)
    if n == 0:
        raise ValueError(f"{metric}: no documents to bootstrap over")

    if (current_weights is not None or candidate_weights is not None) and aggregate is not None:
        raise ValueError(f"{metric}: give weights or an aggregate, not both")
    current_mean = _side_mean(metric, "baseline", current, current_weights, aggregate)
    candidate_mean = _side_mean(metric, "candidate", candidate, candidate_weights, aggregate)
    observed = candidate_mean(range(n)) - current_mean(range(n))

    if n == 1:
        # One document resamples to itself every time, so the interval would be
        # a point and the gate would read it as certainty. Report the observed
        # delta with an interval spanning the whole plausible range instead.
        log.warning(
            "%s: bootstrapping over a single document. No interval is meaningful at n=1, so "
            "this is widened to span the whole range — which FAILS non-inferiority. That is "
            "deliberate: absence of evidence is not evidence of absence, the same rule the "
            "gate applies to an unmeasured metric.", metric,
        )
        return ConfidenceInterval(metric, observed, -1.0, 1.0, resamples, n)

    rng = random.Random(f"{seed}:{metric}")
    deltas: list[float] = []
    for _ in range(resamples):
        # ONE index set per resample, applied to both — that is the pairing.
        # Drawing separate indices for each side would compare different
        # document samples and inflate every interval.
        picks = [rng.randrange(n) for _ in range(n)]
        deltas.append(candidate_mean(picks) - current_mean(picks))

    deltas.sort()
    lower = deltas[int(0.025 * resamples)]
    upper = deltas[min(int(0.975 * resamples), resamples - 1)]
    return ConfidenceInterval(metric, observed, lower, upper, resamples, n)


def _side_mean(
    metric: str,
    side: str,
    scores: Sequence[float],
    weights: Sequence[float] | None,
    aggregate: Callable[[Sequence[float]], float] | None,
) -> Callable[[Sequence[int]], float]:
    """One side's metric over the documents at ``picks``: the weighted mean with
    ``weights``, else ``aggregate`` (the plain mean by default)."""
    if weights is None:
        mean = aggregate or (lambda xs: sum(xs) / len(xs))
        return lambda picks: mean([scores[i] for i in picks])
    weights = [float(w) for w in weights]
    if len(weights) != len(scores):
        raise ValueError(
            f"{metric}: {len(weights)} {side} weights for {len(scores)} documents. Weights are "
            "per document and aligned with the scores."
        )
    # Positive, so no resample can draw a total weight of zero.
    if any(not w > 0 for w in weights):
        raise ValueError(f"{metric}: every {side} weight must be above zero")
    products = [w * float(x) for w, x in zip(weights, scores, strict=True)]
    return lambda picks: (
        sum(map(products.__getitem__, picks)) / sum(map(weights.__getitem__, picks))
    )


def binomial_upper_bound(successes: int, trials: int, *, confidence: float = 0.95) -> float:
    """Upper bound of a proportion, by the Wilson score interval.

    Used where the question is "how bad could the error rate really be" on a
    small sample — the §5.4 review thresholds, and the gate's absolute floors on
    error rates. The Wilson interval rather than the normal approximation
    because at n=30 with two errors the normal approximation gives nonsense, and
    n=30 is the size these eval sets actually are.
    """
    if trials <= 0:
        return 1.0
    z = 1.959964 if confidence >= 0.95 else 1.644854
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * ((p * (1 - p) / trials + z * z / (4 * trials * trials)) ** 0.5)
    return min(1.0, (centre + margin) / denominator)
