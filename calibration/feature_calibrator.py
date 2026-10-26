"""Per-field-type feature calibrators (arch v2.1 §5.3).

A logistic regression per field type, over the §5.2 feature vector, fitted on the
**calibration half** of the validation split.

**Why per field type and not per field.** At pilot volume a per-field calibrator
would see a handful of instances each — an `insured_name` calibrator fitted on
twenty-five examples describes those twenty-five examples. Pooling by type gives
each calibrator hundreds of instances while keeping apart the things that
genuinely miscalibrate differently: an identifier is exact-match and unforgiving,
a free-text description is fuzzy and tolerant, and one curve cannot serve both.

**Why logistic regression and not isotonic, yet.** Isotonic is free-form and
needs data to avoid fitting noise; §5.3 puts its threshold at 2,000 labelled
instances per field type. Below that, a two-parameter-per-feature model is the
honest choice. The switch is a config change, not a rewrite.

**The floor that matters.** A field type with fewer than ``MIN_INSTANCES``
labelled instances gets **no calibrator**, and every field of that type routes to
review. Not a default curve, not the pooled one — review. A calibrator fitted on
forty instances produces numbers that look like probabilities and are not, and
the thresholds in §5.4 are defined against a calibrated score.

Gradient descent in pure Python. The design matrix is a few hundred rows by
a dozen columns; scikit-learn would be faster and is already an `eval` extra, but
this keeps calibration runnable on the pod without pulling the eval stack in, and
the fit takes milliseconds.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from calibration.features import FeatureSet, FieldFeatures

log = logging.getLogger(__name__)

#: Below this, a field type has no calibrator and every field of that type is
#: reviewed (arch v2.1 §5.3).
MIN_INSTANCES = 300

#: Above this, isotonic regression replaces logistic (§5.3). Recorded rather
#: than implemented: the switch is a data milestone this corpus has not reached.
ISOTONIC_THRESHOLD = 2_000

LEARNING_RATE = 0.10
ITERATIONS = 400
L2 = 1e-3


class CalibratorError(RuntimeError):
    """Raised when a calibrator cannot be fitted or applied."""


def _sigmoid(z: float) -> float:
    # Split to avoid overflow: exp(710) is inf, and a saturated feature vector
    # reaches that easily once weights grow.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


@dataclass
class FieldTypeCalibrator:
    """One field type's fitted transform, with the evidence it was fitted on."""

    field_type: str
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    instances: int = 0
    positive_rate: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    method: str = "logistic"

    #: Set when the field type had too little data. The calibrator still exists
    #: as a record — "this type was not calibrated, and here is why" — but
    #: refuses to produce a confidence.
    enforced: bool = True
    reason: str | None = None
    #: Features left out of the fit because every instance carried the same
    #: value: they taught nothing, and read at serving - once filled (a span
    #: that reports its runner-up, text that arrives) - their weight would move
    #: every confidence by an amount no data ever priced.
    constant_features: list[str] = field(default_factory=list)

    def predict(self, features: FieldFeatures) -> float:
        """The calibrated probability this field is correct.

        Read by the names the calibrator was fitted on, so one fitted on fewer
        features - or on a retired one (``features.LEGACY_FEATURES``) - reads
        exactly its own vector."""
        if not self.enforced:
            raise CalibratorError(
                f"{self.field_type} has no enforced calibrator ({self.reason}). Every field of "
                "this type routes to review rather than carrying a number nobody measured."
            )
        if self.feature_names:
            values = features.named_values()
            unknown = [name for name in self.feature_names if name not in values]
            if unknown or len(self.feature_names) != len(self.weights):
                raise CalibratorError(
                    f"{self.field_type} calibrator names features {unknown or self.feature_names} it "
                    f"cannot read with {len(self.weights)} weight(s). A calibrator and a feature "
                    "vector that disagree on shape produce a confident number about the wrong thing."
                )
            z = self.bias + sum(w * values[name] for w, name in zip(self.weights, self.feature_names,
                                                                      strict=True))
            return _sigmoid(z)
        vector = features.vector()
        if len(vector) != len(self.weights):
            raise CalibratorError(
                f"{self.field_type} calibrator expects {len(self.weights)} features, got "
                f"{len(vector)}. A calibrator and a feature vector that disagree on shape "
                "produce a confident number about the wrong thing."
            )
        z = self.bias + sum(w * x for w, x in zip(self.weights, vector, strict=True))
        return _sigmoid(z)

    def as_dict(self) -> dict[str, Any]:
        return {
            "field_type": self.field_type,
            "method": self.method,
            "enforced": self.enforced,
            "reason": self.reason,
            "instances": self.instances,
            "positive_rate": round(self.positive_rate, 4),
            "bias": self.bias,
            "weights": dict(zip(self.feature_names, self.weights, strict=True))
            if self.feature_names else {},
            "constant_features": list(self.constant_features),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FieldTypeCalibrator:
        # Ordered by the FEATURE VECTOR, never by the dict's own key order.
        # `to_json` writes with sort_keys=True, so reading insertion order
        # reconstructed the weights alphabetically — a calibrator that survived
        # the round trip with its weights permuted, producing confident numbers
        # about the wrong features. It round-tripped without raising, which is
        # what made it worth a test.
        stored = payload.get("weights", {})
        # Today's features and the retired ones: a calibrator fitted before a
        # feature was retired reads it at the constant it always was.
        names = [n for n in FieldFeatures.known_feature_names() if n in stored]
        unknown = sorted(set(stored) - set(names))
        if unknown:
            raise CalibratorError(
                f"calibrator for {payload.get('field_type')!r} carries unknown features "
                f"{unknown}. It was fitted against a different feature set, so applying it "
                "would score the wrong things."
            )
        return cls(
            field_type=payload["field_type"],
            weights=[stored[n] for n in names],
            bias=float(payload.get("bias", 0.0)),
            instances=int(payload.get("instances", 0)),
            positive_rate=float(payload.get("positive_rate", 0.0)),
            feature_names=names,
            method=payload.get("method", "logistic"),
            enforced=bool(payload.get("enforced", True)),
            reason=payload.get("reason"),
            constant_features=list(payload.get("constant_features") or []),
        )


def fit_field_type(
    features: FeatureSet,
    *,
    min_instances: int = MIN_INSTANCES,
    iterations: int = ITERATIONS,
    learning_rate: float = LEARNING_RATE,
) -> FieldTypeCalibrator:
    """Fit one field type's calibrator, or record why it could not be."""
    names = FieldFeatures.feature_names()
    calibrator = FieldTypeCalibrator(
        field_type=features.field_type,
        instances=features.size,
        feature_names=names,
        weights=[0.0] * len(names),
    )
    if features.size:
        calibrator.positive_rate = sum(features.correct) / features.size

    if features.size < min_instances:
        calibrator.enforced = False
        calibrator.reason = (
            f"{features.size} labelled instance(s), below the {min_instances} floor. A "
            "calibrator fitted on this much data produces numbers that look like "
            "probabilities and are not."
        )
        log.warning("%s: %s Every field of this type routes to review.",
                    features.field_type, calibrator.reason)
        return calibrator

    # A single-class sample cannot be separated, and a curve fitted on one would
    # report the class prior for every field regardless of its features.
    if len(set(features.correct)) < 2:
        calibrator.enforced = False
        calibrator.reason = (
            f"every one of {features.size} instances has the same outcome "
            f"({'all correct' if features.correct[0] else 'all wrong'}), so nothing "
            "distinguishes them and any fit would just report the class prior."
        )
        log.warning("%s: %s", features.field_type, calibrator.reason)
        return calibrator

    # Fitted on the features that vary: a column every instance holds at one
    # value (cross-mode agreement once, a span detail the engine did not report,
    # text agreement on image-only rows) is a second bias at fit time and an
    # unpriced shift at serving the day it is filled.
    keep = [i for i in range(len(names)) if len({vector[i] for vector in features.vectors}) > 1]
    calibrator.constant_features = [names[i] for i in range(len(names)) if i not in keep]
    calibrator.feature_names = [names[i] for i in keep]
    rows = [[vector[i] for i in keep] for vector in features.vectors]
    weights = [0.0] * len(keep)
    bias = 0.0
    n = features.size
    for _ in range(iterations):
        grad_w = [0.0] * len(keep)
        grad_b = 0.0
        for vector, label in zip(rows, features.correct, strict=True):
            z = bias + sum(w * x for w, x in zip(weights, vector, strict=True))
            error = _sigmoid(z) - (1.0 if label else 0.0)
            grad_b += error
            for i, x in enumerate(vector):
                grad_w[i] += error * x
        bias -= learning_rate * grad_b / n
        for i in range(len(weights)):
            # L2 keeps a feature that happens to separate the sample perfectly
            # from acquiring an unbounded weight — which at these volumes is a
            # real risk, not a theoretical one.
            weights[i] -= learning_rate * (grad_w[i] / n + L2 * weights[i])

    calibrator.weights = weights
    calibrator.bias = bias
    if features.size >= ISOTONIC_THRESHOLD:
        log.info(
            "%s has %d instances, past the %d isotonic threshold (§5.3). Still logistic — "
            "switching is a config change, not a rewrite.",
            features.field_type, features.size, ISOTONIC_THRESHOLD,
        )
    return calibrator


@dataclass
class CalibratorSet:
    """Every field type's calibrator for one release and serving format."""

    release_id: str
    serving_format: str
    calibrators: dict[str, FieldTypeCalibrator] = field(default_factory=dict)

    def for_field(self, features: FieldFeatures) -> FieldTypeCalibrator | None:
        return self.calibrators.get(features.field_type)

    def predict(self, features: FieldFeatures) -> float | None:
        """Calibrated confidence, or ``None`` when this field must be reviewed.

        ``None`` is a routing instruction, not a missing value. Callers must send
        it to review rather than substituting a default — a default here is a
        number nobody measured, presented as if it were.
        """
        calibrator = self.for_field(features)
        if calibrator is None or not calibrator.enforced or not features.is_usable:
            return None
        return calibrator.predict(features)

    @property
    def unenforced_types(self) -> list[str]:
        return sorted(t for t, c in self.calibrators.items() if not c.enforced)

    def as_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "serving_format": self.serving_format,
            "feature_names": FieldFeatures.feature_names(),
            "min_instances": MIN_INSTANCES,
            "unenforced_types": self.unenforced_types,
            "calibrators": {t: c.as_dict() for t, c in sorted(self.calibrators.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibratorSet:
        return cls(
            release_id=payload["release_id"],
            serving_format=payload["serving_format"],
            calibrators={
                name: FieldTypeCalibrator.from_dict(body)
                for name, body in payload.get("calibrators", {}).items()
            },
        )

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def fit_calibrators(
    labelled: list[tuple[FieldFeatures, bool]],
    *,
    release_id: str,
    serving_format: str,
    min_instances: int = MIN_INSTANCES,
) -> CalibratorSet:
    """Fit one calibrator per field type, on the calibration half of validation.

    **Per serving format, always.** Quantization moves the logprob distribution,
    so a calibrator fitted on bf16 reports confidence for a distribution FP8 does
    not produce. Sharing one across formats is not an optimisation, it is a
    silently wrong number (§5.3).
    """
    from calibration.features import group_by_field_type

    sets = group_by_field_type(labelled)
    result = CalibratorSet(release_id=release_id, serving_format=serving_format)
    for field_type, feature_set in sorted(sets.items()):
        result.calibrators[field_type] = fit_field_type(
            feature_set, min_instances=min_instances
        )

    if result.unenforced_types:
        log.warning(
            "release %s/%s has no enforced calibrator for %s — every field of those types "
            "routes to review until the corpus supports one.",
            release_id, serving_format, result.unenforced_types,
        )
    return result
