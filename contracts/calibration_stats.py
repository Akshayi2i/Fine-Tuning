"""How well confidence matches outcome, and how bad a small sample's error rate could be (review A1:
moved from ``evaluation.metrics.coverage_metrics`` and ``evaluation.bootstrap``, which re-export them).
Calibration fits and serves its thresholds with them, and the gate scores with them."""

from __future__ import annotations

import math
from typing import Any

#: How far outside [0, 1] a confidence may sit and still be read as the edge:
#: float rounding of a probability (the exp of a logprob an engine rounds a
#: hair above zero), not a value that is no probability.
CONFIDENCE_TOLERANCE = 1e-6

def probabilities(confidences: list[float]) -> list[float]:
    """``confidences`` as probabilities in [0, 1], refused when one is not.

    A confidence above 1, below 0 or NaN fell in no bin of
    :func:`expected_calibration_error` yet counted in its total, so each one
    pulled ECE toward 0 - the best score - by its share, and a broken
    confidence read as better calibration. Refused with a ``ValueError``
    rather than clipped: such a value is a defect upstream (a NaN feature, a
    calibrator or a logprob conversion gone wrong), and clipped it would be
    scored as a confident, measured probability. Within
    :data:`CONFIDENCE_TOLERANCE` of the range it is float rounding and is read
    as the edge.
    """
    out: list[float] = []
    bad: list[Any] = []
    for confidence in confidences:
        value = float(confidence)
        if (not math.isfinite(value) or value < -CONFIDENCE_TOLERANCE
                or value > 1.0 + CONFIDENCE_TOLERANCE):
            bad.append(confidence)
            continue
        out.append(min(max(value, 0.0), 1.0))
    if bad:
        raise ValueError(
            f"{len(bad)} of {len(confidences)} confidence(s) are not probabilities in [0, 1] "
            f"(e.g. {bad[:5]}). A confidence outside the range or NaN falls in no calibration "
            "bin, and counted in the total it would make the calibration error look smaller "
            "than it is. Find where it was produced (a NaN feature, the calibrator, the logprob "
            "conversion) rather than scoring it."
        )
    return out

def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    *,
    bins: int = 10,
) -> float:
    """ECE — the gap between stated confidence and observed accuracy.

    Answers whether the confidence numbers can be trusted, which is a different
    question from whether the extractions are right. A model can be 95% accurate
    and badly calibrated, and the review routing built on its confidence would
    then send the wrong documents to humans.

    Every confidence must be a probability (:func:`probabilities`): one outside
    [0, 1] or NaN is refused, never left out of the bins while counted in the
    total.
    """
    if len(confidences) != len(correctness):
        raise ValueError(
            f"{len(confidences)} confidences but {len(correctness)} outcomes — they must align"
        )
    if not confidences:
        return 0.0
    confidences = probabilities(confidences)

    total = len(confidences)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (c, ok) for c, ok in zip(confidences, correctness, strict=True)
            if (low < c <= high) or (index == 0 and c == 0.0)
        ]
        if not members:
            continue
        mean_confidence = sum(c for c, _ in members) / len(members)
        accuracy = sum(ok for _, ok in members) / len(members)
        error += (len(members) / total) * abs(accuracy - mean_confidence)
    return round(error, 4)


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
