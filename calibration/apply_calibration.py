"""Applying a fitted calibration at inference time (arch §5).

Used by serving (IMPL-11) and the testing harness (IMPL-12). The important
property is what it does when calibration is **missing**: it raises.

Silently returning raw confidence would hand the caller numbers that look
calibrated and are not — and raw confidence from a fine-tuned model is
systematically overconfident, so the review routing built on it would send too
little to humans. A missing calibration is an operational error, not a fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, BlobError
from calibration.fit_calibration import (
    CalibrationError,
    CalibrationParams,
    _apply_temperature,
    apply_isotonic,
)
from calibration.logprob_confidence import FieldConfidence
from common.constants import DEFAULT_REVIEW_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)


@dataclass
class CalibratedField:
    """A field's confidence after calibration, and whether it needs review."""

    field_path: str
    value: Any
    raw_confidence: float
    confidence: float
    needs_review: bool
    reason: str | None = None

    def as_output(self) -> dict[str, Any]:
        """The ``{value, confidence}`` shape the API returns (master §9)."""
        return {"value": self.value, "confidence": round(self.confidence, 4)}


@dataclass
class CalibratedResult:
    """A whole document's calibrated confidence, plus review routing."""

    fields: dict[str, CalibratedField] = field(default_factory=dict)
    overall_confidence: float = 0.0
    review_flags: list[str] = field(default_factory=list)
    #: Each field's feature vector (``calibration.features.FieldFeatures``) by
    #: its values-view path, where a feature calibrator scored it. Kept so the
    #: text-dependent features can be computed again when a page's text arrives
    #: after its fields were read from the image (``calibration.post_text``).
    features: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_flags)


def load_calibration(
    model_version: str, doc_type: str, client: BlobClient
) -> CalibrationParams:
    """Load the fitted transform for a model version and document type.

    Raises when absent. Calibration is fitted per version, so falling back to
    another version's parameters would be describing a different model.
    """
    key = paths.calibration_params(model_version, doc_type)
    try:
        if not client.exists(key):
            raise CalibrationError(
                f"no calibration exists for {model_version}/{doc_type} at {key}. Raw logprobs "
                "from a fine-tuned model are systematically overconfident (arch §5), so returning "
                "them unmarked would produce confidence numbers that cannot be trusted for review "
                "routing. Fit calibration for this version before serving it."
            )
        return CalibrationParams.from_dict(client.read_json(key))
    except BlobError as exc:
        raise CalibrationError(f"could not read calibration for {model_version}/{doc_type}: {exc}") from exc


def calibrate_confidence(confidence: float, params: CalibrationParams) -> float:
    """Map one raw confidence through the fitted transform."""
    if params.method == "temperature":
        if params.temperature is None:
            raise CalibrationError("temperature calibration has no fitted temperature")
        return round(_apply_temperature(confidence, params.temperature), 6)
    return apply_isotonic(confidence, params.breakpoints)


def apply_calibration(
    raw: dict[str, FieldConfidence],
    params: CalibrationParams,
    *,
    review_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
) -> CalibratedResult:
    """Calibrate every field and flag the ones needing review.

    Fields with no usable confidence are flagged for review rather than assigned
    one. An unmappable field has no evidence behind it, and inventing a number
    would be worse than saying so.
    """
    result = CalibratedResult()
    usable: list[float] = []

    for path, confidence in sorted(raw.items()):
        if not confidence.is_usable:
            result.fields[path] = CalibratedField(
                field_path=path, value=confidence.value,
                raw_confidence=0.0, confidence=0.0, needs_review=True,
                reason=f"no confidence available: {confidence.reason}",
            )
            result.review_flags.append(f"{path}:no_confidence")
            continue

        calibrated = calibrate_confidence(confidence.confidence, params)
        needs_review = calibrated < review_threshold
        result.fields[path] = CalibratedField(
            field_path=path, value=confidence.value,
            raw_confidence=confidence.confidence, confidence=calibrated,
            needs_review=needs_review,
            reason="below review threshold" if needs_review else None,
        )
        if needs_review:
            result.review_flags.append(f"{path}:low_confidence")
        usable.append(calibrated)

    result.overall_confidence = round(sum(usable) / len(usable), 4) if usable else 0.0
    return result


def calibrated_output(result: CalibratedResult) -> dict[str, Any]:
    """Render the API output contract (master §9)."""
    return {
        "overall_confidence": result.overall_confidence,
        "fields": {path: f.as_output() for path, f in sorted(result.fields.items())},
        "review_flags": sorted(result.review_flags),
    }
