"""LoB accuracy, schema validity, calibration error, and mode accuracy (arch §15).

Four metrics that share a shape — each aggregates per-document outcomes into one
number the promotion gate reads — but each answers a different question:

* **LoB accuracy**, reported **per value**, never folded into overall field
  accuracy: a class that is rare in the corpus must not hide inside a healthy
  aggregate (arch §0b).
* **Schema validity**, which mirrors the Fideon SPEC_07 Stage 3 audit gate.
* **ECE**, which asks whether the confidence numbers are trustworthy — not
  whether the extractions are.
* **Mode accuracy**, split by ``image_only`` and ``scanned``, which feeds the ViT
  escalation gate. It also classifies errors, because that gate keys on the
  perception-vs-reasoning distinction rather than on the accuracy number.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from common.lob import lob_values
from evaluation.metrics.field_accuracy import values_agree

# --------------------------------------------------------------------------
# Line of Business (arch §0b)
# --------------------------------------------------------------------------

@dataclass
class LobReport:
    """LoB detection as a **set** problem, per value (arch v2.1 §15.2).

    Under v1 this was one label per document and accuracy was a fair summary.
    Under a list it is not: a document covering general liability and property,
    predicted as general liability alone, is neither right nor wrong — it is
    complete precision at half recall, and an accuracy number cannot say that.

    So each value gets true positives, false positives and false negatives, and
    the gate reads set-F1. The distinction matters in the direction that costs
    money: a missed line under-states coverage on a certificate.
    """

    #: value -> [true positives, false positives, false negatives]
    per_value: dict[str, list[int]] = field(default_factory=dict)

    #: Documents whose golden LoB list is empty. Predicting a line for one is a
    #: false positive that no per-value recall would otherwise catch, because
    #: there is no truth value to miss.
    undetermined_total: int = 0
    undetermined_correct: int = 0
    #: Documents that carry a line to score. Zero means the metric is not
    #: measured at all — absent, not 0.0.
    scored: int = 0

    def _cell(self, value: str) -> list[int]:
        return self.per_value.setdefault(value, [0, 0, 0])

    @staticmethod
    def _f1(tp: int, fp: int, fn: int) -> float:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    @property
    def overall(self) -> float:
        """Micro set-F1 across every value — the gating number."""
        tp = sum(c[0] for c in self.per_value.values())
        fp = sum(c[1] for c in self.per_value.values())
        fn = sum(c[2] for c in self.per_value.values())
        return self._f1(tp, fp, fn)

    def f1_by_value(self) -> dict[str, float]:
        return {
            value: round(self._f1(*counts), 4)
            for value, counts in sorted(self.per_value.items())
        }

    #: Kept under the v1 name so the manifest field and the gate key do not move
    #: in the same change as the metric's meaning. It reports F1, not accuracy.
    def accuracy_by_value(self) -> dict[str, float]:
        return self.f1_by_value()

    def unmeasured_values(self) -> list[str]:
        """LoB values with no eval documents at all.

        An unmeasured class is not a passing class — it is one whose score nobody
        knows, and the aggregate will not say so. ``tp + fn`` is the support:
        false positives alone mean the value was never in the golden set.
        """
        return sorted(
            v for v in lob_values()
            if (self.per_value.get(v, [0, 0, 0])[0] + self.per_value.get(v, [0, 0, 0])[2]) == 0
        )


def score_lob(documents: list[tuple[dict[str, Any], dict[str, Any]]]) -> LobReport:
    """Score ``line_of_business`` as a set, per value, across the eval set."""
    from common.lob import normalize_lob

    report = LobReport()
    for expected, got in documents:
        if "line_of_business" not in expected:
            # A canonical policy label carries no line_of_business: its line
            # comes from metadata, and the model is never asked for one. Scoring
            # it would read an empty truth and an empty prediction as a 0.0 for
            # every policy, which blocked every policy candidate on a metric the
            # model has no way to move.
            continue
        report.scored += 1
        truth = set(normalize_lob(expected.get("line_of_business")))
        prediction = set(normalize_lob(got.get("line_of_business")))

        if not truth:
            # Nothing to recall. What is measurable is whether the model also
            # declined — inventing a line here is the failure mode, and it shows
            # up as a false positive on whichever value it invented.
            report.undetermined_total += 1
            report.undetermined_correct += int(not prediction)

        for value in truth | prediction:
            cell = report._cell(value)
            if value in truth and value in prediction:
                cell[0] += 1
            elif value in prediction:
                cell[1] += 1
            else:
                cell[2] += 1
    return report


# --------------------------------------------------------------------------
# Schema validity
# --------------------------------------------------------------------------

@dataclass
class SchemaValidityReport:
    valid: int = 0
    total: int = 0
    invalid_source_ids: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.valid / self.total if self.total else 0.0


def score_schema_validity(
    outputs: list[tuple[Any, ...]],
) -> SchemaValidityReport:
    """Validate each output against its document type's schema.

    Each entry is ``(source_id, output, doc_type, acord_form[, lob[, sections]])``;
    the line selects a policy's canonical schema and ``sections`` the slice one
    window was asked for.

    Mirrors the audit gate that runs on every production call, so a validity
    regression is caught before promotion rather than in production. A canonical
    output may arrive in the model's form — no ``confidence``, no ``flagged`` —
    and is enveloped first, exactly as serving does before its own audit: the
    question is whether the model produced the client's tree, not whether it
    wrote fields the pipeline adds.

    One window of a common-model line (an output with ``sections``) is checked
    against that slice's model view (``resolved_schema``), as written: the model
    is never asked for what the pipeline fills in after the merge - a coverage's
    ``coverage_id``, the numbering of every row - and the client's slice
    requires it. A whole common-model document is judged as serving would
    return it (``serving.pipeline.served_schema_verdict``): enveloped, given the
    system fields and every key, then checked against the client's schema. The
    merged answer alone lacks what the pipeline adds and would fail for it.
    """
    from common.canonical import envelope
    from common.schemas import is_canonical, validator_for

    report = SchemaValidityReport()
    for source_id, output, doc_type, acord_form, *rest in outputs:
        lob = rest[0] if rest else None
        sections = rest[1] if len(rest) > 1 else None
        report.total += 1
        if not sections and _is_common_model(doc_type, acord_form, lob):
            from serving.pipeline import served_schema_verdict

            if served_schema_verdict(output, doc_type=doc_type, acord_form=acord_form, lob=lob).valid:
                report.valid += 1
            else:
                report.invalid_source_ids.append(source_id)
            continue
        if sections and _is_common_model(doc_type, acord_form, lob):
            validator = _model_view_validator(doc_type, acord_form, lob, sections)
        else:
            if is_canonical(doc_type, acord_form, lob):
                output = envelope(output, {})
            validator = validator_for(doc_type, acord_form, lob, sections)
        if validator.is_valid(output):
            report.valid += 1
        else:
            report.invalid_source_ids.append(source_id)
    return report


def _is_common_model(doc_type: str, acord_form: str | None, lob: Any) -> bool:
    from common.schemas import SchemaError, is_common_model

    try:
        return doc_type == "policy" and is_common_model(doc_type, acord_form, lob)
    except SchemaError:
        return False


def _model_view_validator(doc_type: str, acord_form: str | None, lob: Any, sections: str) -> Any:
    from jsonschema import Draft202012Validator

    from common.schemas import resolved_schema

    return Draft202012Validator(resolved_schema(doc_type, acord_form, lob, sections))


# --------------------------------------------------------------------------
# Expected Calibration Error
# --------------------------------------------------------------------------

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
    """
    if len(confidences) != len(correctness):
        raise ValueError(
            f"{len(confidences)} confidences but {len(correctness)} outcomes — they must align"
        )
    if not confidences:
        return 0.0

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


def reliability_bins(
    confidences: list[float], correctness: list[bool], *, bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin confidence vs accuracy — the shape behind the ECE number.

    Useful because the direction matters: over-confidence routes too little to
    review, under-confidence routes too much, and one ECE value cannot say which.
    """
    out: list[dict[str, float]] = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            (c, ok) for c, ok in zip(confidences, correctness, strict=True) if low < c <= high
        ]
        if not members:
            continue
        out.append({
            "bin_low": low,
            "bin_high": high,
            "count": len(members),
            "mean_confidence": round(sum(c for c, _ in members) / len(members), 4),
            "accuracy": round(sum(ok for _, ok in members) / len(members), 4),
        })
    return out


# --------------------------------------------------------------------------
# Mode accuracy and error classification (feeds the ViT gate)
# --------------------------------------------------------------------------

@dataclass
class ModeReport:
    """Accuracy per input regime, with document counts.

    Counts travel with the accuracy because the ViT gate needs them: below its
    document floor the gate returns ``insufficient_data`` rather than deciding
    (arch §16c).
    """

    accuracy_by_mode: dict[str, float] = field(default_factory=dict)
    documents_by_mode: dict[str, int] = field(default_factory=dict)
    scanned_accuracy: float = 0.0
    scanned_documents: int = 0
    error_classes: dict[str, int] = field(default_factory=dict)

    def error_mix(self) -> dict[str, float]:
        total = sum(self.error_classes.values())
        return (
            {k: round(v / total, 3) for k, v in sorted(self.error_classes.items())}
            if total else {}
        )


def score_by_mode(
    documents: list[tuple[dict[str, Any], dict[str, Any], str, bool]],
) -> ModeReport:
    """Score accuracy split by modality mode and by scanned-vs-digital.

    Args:
        documents: ``(expected, got, modality_mode, is_scanned)`` tuples.
    """
    from training.vit_gate import classify_error

    by_mode: dict[str, list[float]] = defaultdict(list)
    scanned: list[float] = []
    error_counter: Counter = Counter()

    for expected, got, mode, is_scanned in documents:
        outcomes: list[bool] = []
        for field_path, expected_value in expected.items():
            if isinstance(expected_value, (list, dict)):
                continue
            got_value = got.get(field_path)
            correct = values_agree(expected_value, got_value, field_path)
            outcomes.append(correct)
            if not correct:
                error_counter[classify_error(expected_value, got_value, all_expected=expected)] += 1

        if not outcomes:
            # No scalar fields to compare — a list-only document. Scoring it 0.0
            # counted "nothing to measure" as "everything wrong", halving mode
            # accuracy and pointing the ViT gate at a perception problem that
            # was not there. Skipped, like every other unmeasured thing.
            continue
        accuracy = sum(outcomes) / len(outcomes)
        by_mode[mode].append(accuracy)
        if is_scanned:
            scanned.append(accuracy)

    report = ModeReport(
        accuracy_by_mode={
            mode: round(sum(scores) / len(scores), 4) for mode, scores in sorted(by_mode.items())
        },
        documents_by_mode={mode: len(scores) for mode, scores in sorted(by_mode.items())},
        scanned_accuracy=round(sum(scanned) / len(scanned), 4) if scanned else 0.0,
        scanned_documents=len(scanned),
        error_classes=dict(error_counter),
    )
    return report
