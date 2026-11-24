"""Post-hoc confidence calibration (arch §5).

Fine-tuned generative models — especially after LoRA fine-tuning on a narrow
task — become **systematically overconfident**. Correct and incorrect outputs
carry similarly high token probabilities, because the model has learned to be
fluent in the target format even when it is wrong about the content. Raw logprobs
are therefore not a usable confidence signal on their own.

The fix is cheap: fit a transform on held-out data comparing raw confidence
against measured correctness, and apply it after the model call. Two methods:

* **temperature scaling** — one parameter, monotonic, the default;
* **isotonic regression** — free-form monotonic, for non-uniform miscalibration.

Fitted **per model version and per document type**, never shared across either:
calibration describes one model's behaviour, and reusing it across versions would
be describing a model that no longer exists.

A separate verifier model would also work and is the documented escalation if
this proves insufficient — but it means another model to train, version, and keep
in sync, plus doubled latency. Start with the cheap approach.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from contracts.calibration_stats import expected_calibration_error

log = logging.getLogger(__name__)

Method = Literal["temperature", "isotonic"]


class CalibrationError(RuntimeError):
    """Raised when a calibration cannot be fitted or is unusable."""


@dataclass
class CalibrationParams:
    """A fitted transform, with the evidence that it helped."""

    method: Method
    doc_type: str
    model_version: str
    temperature: float | None = None
    #: Isotonic breakpoints as (raw, calibrated) pairs, ascending.
    breakpoints: list[tuple[float, float]] = field(default_factory=list)
    ece_before: float = 0.0
    ece_after: float = 0.0
    n_samples: int = 0

    @property
    def improved(self) -> bool:
        return self.ece_after < self.ece_before

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "doc_type": self.doc_type,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "breakpoints": [list(p) for p in self.breakpoints],
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "n_samples": self.n_samples,
            "improved": self.improved,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationParams:
        return cls(
            method=data["method"],
            doc_type=data["doc_type"],
            model_version=data["model_version"],
            temperature=data.get("temperature"),
            breakpoints=[tuple(p) for p in data.get("breakpoints", [])],
            ece_before=data.get("ece_before", 0.0),
            ece_after=data.get("ece_after", 0.0),
            n_samples=data.get("n_samples", 0),
        )


def _apply_temperature(confidence: float, temperature: float) -> float:
    """Temperature-scale a probability through logit space."""
    p = min(max(confidence, 1e-6), 1 - 1e-6)
    logit = math.log(p / (1 - p))
    scaled = logit / max(temperature, 1e-6)
    return 1.0 / (1.0 + math.exp(-scaled))


def fit_temperature(
    confidences: list[float],
    correctness: list[bool],
    *,
    search_range: tuple[float, float] = (0.25, 8.0),
    steps: int = 160,
) -> float:
    """Find the temperature minimising ECE.

    A grid search rather than gradient descent: one bounded parameter, a few
    hundred evaluations, and no optimiser to misconfigure. A temperature above 1
    means the model was overconfident, which is the expected direction here.
    """
    if len(confidences) != len(correctness):
        raise CalibrationError("confidences and correctness must align")
    if not confidences:
        raise CalibrationError("cannot fit calibration on an empty sample")

    low, high = search_range
    best_temperature, best_ece = 1.0, float("inf")
    for index in range(steps + 1):
        temperature = low + (high - low) * index / steps
        scaled = [_apply_temperature(c, temperature) for c in confidences]
        ece = expected_calibration_error(scaled, correctness)
        if ece < best_ece:
            best_temperature, best_ece = temperature, ece
    return round(best_temperature, 4)


def fit_isotonic(
    confidences: list[float],
    correctness: list[bool],
    *,
    bins: int = 10,
) -> list[tuple[float, float]]:
    """Fit a monotonic step function by binning, then enforcing monotonicity.

    Binned rather than a full pool-adjacent-violators fit: at pilot volume there
    are too few samples for a fine-grained isotonic fit to be anything but
    overfitted, and a coarse monotonic curve is the honest resolution.
    """
    if len(confidences) != len(correctness):
        raise CalibrationError("confidences and correctness must align")

    buckets: dict[int, list[bool]] = {}
    for confidence, correct in zip(confidences, correctness, strict=True):
        index = min(int(confidence * bins), bins - 1)
        buckets.setdefault(index, []).append(correct)

    points: list[tuple[float, float]] = []
    for index in sorted(buckets):
        members = buckets[index]
        points.append(((index + 0.5) / bins, sum(members) / len(members)))

    # Enforce monotonicity: confidence that decreases as raw confidence rises
    # would be uninterpretable, whatever the sample says.
    calibrated: list[tuple[float, float]] = []
    running_max = 0.0
    for raw, observed in points:
        running_max = max(running_max, observed)
        calibrated.append((round(raw, 4), round(running_max, 4)))
    return calibrated


def apply_isotonic(confidence: float, breakpoints: list[tuple[float, float]]) -> float:
    """Interpolate a raw confidence through fitted breakpoints."""
    if not breakpoints:
        return confidence
    if confidence <= breakpoints[0][0]:
        return breakpoints[0][1]
    if confidence >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    for (x0, y0), (x1, y1) in zip(breakpoints, breakpoints[1:], strict=False):
        if x0 <= confidence <= x1:
            if x1 == x0:
                return y1
            ratio = (confidence - x0) / (x1 - x0)
            return round(y0 + ratio * (y1 - y0), 6)
    return confidence


def fit_calibration(
    confidences: list[float],
    correctness: list[bool],
    *,
    doc_type: str,
    model_version: str,
    method: Method = "temperature",
    min_samples: int = 30,
) -> CalibrationParams:
    """Fit a calibration transform on held-out data.

    Args:
        min_samples: below this the fit describes noise. It warns rather than
            refuses — an uncalibrated confidence is still better than none, and
            the report says the fit is thin.
    """
    if len(confidences) < min_samples:
        log.warning(
            "fitting calibration on only %d samples (floor %d). The transform will describe "
            "sampling noise as much as miscalibration; treat the result as provisional.",
            len(confidences), min_samples,
        )

    ece_before = expected_calibration_error(confidences, correctness)
    params = CalibrationParams(
        method=method, doc_type=doc_type, model_version=model_version,
        ece_before=ece_before, n_samples=len(confidences),
    )

    if method == "temperature":
        params.temperature = fit_temperature(confidences, correctness)
        calibrated = [_apply_temperature(c, params.temperature) for c in confidences]
    else:
        params.breakpoints = fit_isotonic(confidences, correctness)
        calibrated = [apply_isotonic(c, params.breakpoints) for c in confidences]

    params.ece_after = expected_calibration_error(calibrated, correctness)

    if not params.improved:
        # Surfaced rather than assumed: a fit that does not reduce ECE has not
        # calibrated anything, and shipping it would imply a correction that
        # was never made.
        log.warning(
            "calibration for %s/%s did not improve ECE (%.4f -> %.4f). Do not treat these "
            "confidences as calibrated; try the other method or collect more held-out data.",
            model_version, doc_type, params.ece_before, params.ece_after,
        )
    else:
        log.info(
            "calibration for %s/%s: ECE %.4f -> %.4f on %d samples",
            model_version, doc_type, params.ece_before, params.ece_after, params.n_samples,
        )
    return params


def assert_held_out(train_source_ids: set[str], calibration_source_ids: set[str]) -> None:
    """Calibration must be fitted on data the model did not train on.

    Fitting on training data measures memorisation, produces a transform that
    looks excellent, and silently fails in production on documents the model has
    not seen.
    """
    overlap = train_source_ids & calibration_source_ids
    if overlap:
        raise CalibrationError(
            f"{len(overlap)} document(s) appear in BOTH the training split and the calibration "
            f"set (e.g. {sorted(overlap)[:5]}). Calibration fitted on training data measures "
            "memorisation, not calibration — it would look excellent and fail in production."
        )
