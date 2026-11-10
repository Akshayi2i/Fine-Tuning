"""The ViT escalation gate (arch §3) — decide, don't act.

The Vision Encoder is frozen by default. Unfreezing costs a full Foundation
retrain and risks degrading the pretrained document/OCR capability that the
image-only pathway depends on, so it happens only when the evidence says visual
*reading* is genuinely the bottleneck.

The gate fires when:

1. image-only OR scanned accuracy is below target, **and**
2. the errors are **perception** errors (misread characters, missed checkboxes)
   rather than **schema/reasoning** errors (right value, wrong field).

Condition 2 is the load-bearing one. If the model reads a character correctly but
puts it in the wrong JSON field, that is an LLM or projector problem and touching
the ViT will not help — it will just cost a retrain.

**Three outcomes, not two.** ``insufficient_data`` exists because at pilot volume
image-only carries roughly nine documents of signal per type, and thin data
produces exactly the perception-shaped errors this gate keys on. Forcing a
fire/no-fire answer there would buy an expensive retrain to fix a problem more
documents would have solved. The architecture says the gate must not fire on
pilot data (arch §16c); this is how the code expresses that rather than relying
on someone remembering.
"""

from __future__ import annotations

import functools
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger(__name__)

Decision = Literal["fire", "hold", "insufficient_data"]
ErrorClass = Literal["perception", "schema_reasoning", "omission", "invented", "unknown"]

#: Classes that are not a misreading or a misplacement of a printed value, so
#: they say nothing about the vision tower: a value written where the label has
#: none, and a record that was never classified.
_NOT_READ_ERRORS = frozenset({"invented", "unknown"})

#: Below this many evaluated documents in a subset, the metric is noise and the
#: gate refuses to decide. Nine documents — pilot volume for image-only — is
#: comfortably below it.
MIN_EVAL_DOCUMENTS = 30

#: Share of errors that must be perception-type before the ViT is implicated.
PERCEPTION_ERROR_THRESHOLD = 0.60


@dataclass
class ErrorRecord:
    """One field-level failure, classified by what actually went wrong."""

    source_id: str
    field_path: str
    expected: Any
    got: Any
    error_class: ErrorClass = "unknown"
    modality_mode: str = "ocr_plus_image"
    is_scanned: bool = False


@dataclass
class GateDecision:
    """The gate's answer, and the reasoning behind it."""

    decision: Decision
    rationale: str
    image_only_accuracy: float | None = None
    scanned_accuracy: float | None = None
    image_only_documents: int = 0
    scanned_documents: int = 0
    #: Error mix over the image-only and scanned subsets — the subsets the
    #: accuracy check gates on, and so the only ones whose mix may decide.
    error_mix: dict[str, float] = field(default_factory=dict)
    #: Mix over every error. Reported for context; never gated on.
    overall_error_mix: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""

    @property
    def should_train_vit(self) -> bool:
        return self.decision == "fire"

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "rationale": self.rationale,
            "image_only_accuracy": self.image_only_accuracy,
            "scanned_accuracy": self.scanned_accuracy,
            "image_only_documents": self.image_only_documents,
            "scanned_documents": self.scanned_documents,
            "error_mix": self.error_mix,
            "overall_error_mix": self.overall_error_mix,
            "recommendation": self.recommendation,
        }


def classify_error(expected: Any, got: Any, *, all_expected: dict[str, Any] | None = None) -> ErrorClass:
    """Classify a field error by what actually went wrong.

    * **omission** — nothing was produced. Not a perception problem; the model
      did not attempt the field.
    * **schema_reasoning** — the value produced belongs to a *different* field in
      the same document. The character was read correctly and placed wrongly, so
      the ViT is not implicated.
    * **perception** — the value is a near-miss on the expected string, the
      signature of a misread character.
    * **invented** — a value where the label has none, and that no other field
      of the label holds either. Counted in the report's error totals; left out
      of the ViT error mix, which asks how printed values were misread or
      misplaced.
    """
    if got is None or got == "":
        return "omission"

    got_key, expected_key = _value_key(got), _value_key(expected)
    if all_expected and got_key is not None:
        # The returned value belongs to a DIFFERENT field in the same document:
        # the character was read correctly and placed wrongly, so the ViT is not
        # implicated.
        #
        # Flattened, because `.values()` alone saw only top-level fields. A loss
        # run's confusables live almost entirely inside `claims[]` — an amount
        # lifted from the wrong row is the canonical misplacement error, and it
        # was reaching `_near_miss`, where two similar amounts read as a misread
        # character. That is the one classification this gate must not get
        # wrong: it is what decides whether a Foundation retrain is justified.
        #
        # Compared as the same value written another way - case, spacing, punctuation, "$10,000" and 10000, a date
        # in either layout (2026-11-10 review: 'Off Premises' was called invented beside a gold 'OFF PREMISES', and
        # 10000 beside 10000.0).
        for other_value in _flatten_values(all_expected):
            if _value_key(other_value) == got_key and got_key != expected_key:
                return "schema_reasoning"

    # After the misplacement check: a value the label holds under another field,
    # written where it holds none (a policy number copied into the extra fields),
    # was read right and placed wrong. Only a value the label holds nowhere is
    # invented.
    if expected is None or str(expected).strip() == "":
        return "invented"
    # Casefolded: a value in another case only ('Accounts Receivable.' for 'ACCOUNTS RECEIVABLE') is a near-miss on
    # the expected string, not a value from another field.
    expected_text, got_text = str(expected).strip().casefold(), str(got).strip().casefold()

    if _near_miss(expected_text, got_text):
        return "perception"
    return "schema_reasoning"


#: Keys of a value envelope that hold no value of the label: where it is printed, how sure, whether flagged. A gold
#: citing page 4 does not hold the value 4 (2026-11-10 review: a gold-gap value 7 was classed as placed wrong because
#: the gold cites page 7).
_NOT_VALUES = frozenset({"page_ref", "p", "confidence", "flagged"})

#: Text that is a number as printed: "$10,000", "(1,200.00)", "1,000". Anything else ("CG 00 01") stays text, since
#: ``normalize_currency`` keeps every digit of any text.
_NUMERIC_TEXT = re.compile(r"\(?\s*[$€£]?\s*-?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*\)?")


def _value_key(value: Any) -> tuple[str, Any] | None:
    """What a value is, for asking whether two fields hold the same one: a number as its amount, a date as its day,
    text with case, spacing and punctuation left out (``common.normalize``). None for nothing.

    Remembered per value: a document's every gold value is asked about once per wrong field, and a date is tried
    against a dozen layouts (code review 2026-11-10: about 84 µs a value, some 15 s for a large policy's report)."""
    try:
        return _cached_value_key(value)
    except TypeError:                      # not hashable (a list or object answered for a field): worked out each time
        return _value_key_of(value)


@functools.lru_cache(maxsize=65536, typed=True)
def _cached_value_key(value: Any) -> tuple[str, Any] | None:
    return _value_key_of(value)


def _value_key_of(value: Any) -> tuple[str, Any] | None:
    from common.normalize import normalize_currency, normalize_date, normalize_identifier

    if value is None:
        return None
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", float(value))
    text = str(value).strip()
    if not text:
        return None
    if _NUMERIC_TEXT.fullmatch(text) and (amount := normalize_currency(text)) is not None:
        return ("number", amount)
    if day := normalize_date(text):
        return ("date", day)
    cleaned = normalize_identifier(text)
    return ("text", cleaned) if cleaned else None


def _flatten_values(value: Any) -> list[Any]:
    """Every scalar value in a nested golden record, at any depth - not an envelope's pages or confidence.

    Lists of objects (``claims``, ``coverages``) are where a document's
    type-compatible near-duplicates actually sit, so a misplacement check that
    reads only the top level misses the cases it exists for.
    """
    if isinstance(value, dict):
        return [v for key, item in value.items() if key not in _NOT_VALUES for v in _flatten_values(item)]
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in _flatten_values(item)]
    return [] if value is None else [value]


def _near_miss(expected: str, got: str, *, max_edit_share: float = 0.25) -> bool:
    """Whether two strings differ by a few characters.

    A near-miss is the signature of misreading — ``WC-8842317-01`` returned as
    ``WC-8842317-0l``. A wholly different string is the signature of picking the
    wrong field.
    """
    if abs(len(expected) - len(got)) > max(2, len(expected) * 0.3):
        return False

    # Levenshtein, iterative, no dependency.
    previous = list(range(len(got) + 1))
    for i, a in enumerate(expected, start=1):
        current = [i]
        for j, b in enumerate(got, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    distance = previous[-1]
    return 0 < distance <= max(1, int(len(expected) * max_edit_share))


def in_gated_subsets(errors: list[ErrorRecord]) -> list[ErrorRecord]:
    """The errors from the subsets the accuracy check actually gates on.

    Accuracy is read from image-only and scanned documents, so the error mix has
    to come from the same place. Computed over every error, an ocr_plus_image
    majority — the largest subset, and the one where perception is least
    implicated — drowns the signal: a genuine image-only perception failure lands
    under the 60% threshold and the gate holds while claiming the errors are not
    perception-type. ``modality_mode`` and ``is_scanned`` are recorded on every
    ErrorRecord for exactly this, and were being ignored.
    """
    return [e for e in errors if e.modality_mode == "image_only" or e.is_scanned]


def summarise_error_mix(errors: list[ErrorRecord]) -> dict[str, float]:
    """Share of each error class among classified errors."""
    counts = Counter(e.error_class for e in errors if e.error_class not in _NOT_READ_ERRORS)
    total = sum(counts.values())
    return {cls: round(n / total, 3) for cls, n in counts.items()} if total else {}


def evaluate_gate(
    *,
    image_only_accuracy: float | None,
    scanned_accuracy: float | None,
    image_only_documents: int,
    scanned_documents: int,
    errors: list[ErrorRecord],
    target_accuracy: float = 0.80,
    min_documents: int = MIN_EVAL_DOCUMENTS,
) -> GateDecision:
    """Decide whether to escalate to a ViT LoRA.

    Returns a recommendation only — it never starts training. The decision is
    expensive enough to warrant a human reading the rationale.
    """
    # The mix that decides is scoped to the subsets the accuracy check gates on;
    # the overall mix is reported alongside it so the rationale stays readable.
    gated_errors = in_gated_subsets(errors)
    error_mix = summarise_error_mix(gated_errors)
    overall_mix = summarise_error_mix(errors)

    # 1. Is there enough evidence to decide at all?
    #
    # An accuracy of None means the subset was never scored, and unmeasured is
    # not passing — the same rule the promotion gate applies. Skipping the count
    # check on a None used to fall through to step 2, where `value is not None`
    # skipped it again, so the gate returned `hold` with a rationale asserting
    # both subsets met the target when neither had been measured at all.
    thin = []
    if image_only_accuracy is None:
        thin.append("image-only (never scored)")
    elif image_only_documents < min_documents:
        thin.append(f"image-only ({image_only_documents} documents)")
    if scanned_accuracy is None:
        thin.append("scanned (never scored)")
    elif scanned_documents < min_documents:
        thin.append(f"scanned ({scanned_documents} documents)")

    if thin:
        return GateDecision(
            decision="insufficient_data",
            rationale=(
                f"Evidence too thin to decide: {', '.join(thin)} against a {min_documents}-document "
                "floor. A subset nobody scored has not passed — reporting `hold` there would claim "
                "a target was met that was never measured. And at this volume low accuracy is a "
                "data-volume artifact, because thin data "
                "produces exactly the perception-shaped errors this gate keys on — so firing here "
                "would buy an expensive retrain for a problem more documents would solve "
                "(arch §16c)."
            ),
            image_only_accuracy=image_only_accuracy,
            scanned_accuracy=scanned_accuracy,
            image_only_documents=image_only_documents,
            scanned_documents=scanned_documents,
            error_mix=error_mix,
            overall_error_mix=overall_mix,
            recommendation=(
                "Do not touch the ViT. Collect more image-only and scanned documents, then "
                "re-evaluate. Before considering a ViT LoRA, try the cheaper interventions first: "
                "raise the resolution cap toward 2048px, and rebalance the modality mix to give "
                "image_only a larger share (no new labeling required)."
            ),
        )

    # 2. Is accuracy actually below target?
    below = [
        f"{name} {value:.3f} < {target_accuracy:.2f}"
        for name, value in (("image-only", image_only_accuracy), ("scanned", scanned_accuracy))
        if value is not None and value < target_accuracy
    ]
    if not below:
        return GateDecision(
            decision="hold",
            rationale=(
                f"image-only and scanned accuracy both meet the {target_accuracy:.2f} target. The "
                "frozen ViT is sufficient — and cheaper and faster than the alternative."
            ),
            image_only_accuracy=image_only_accuracy,
            scanned_accuracy=scanned_accuracy,
            image_only_documents=image_only_documents,
            scanned_documents=scanned_documents,
            error_mix=error_mix,
            overall_error_mix=overall_mix,
            recommendation="Keep train_vit: false.",
        )

    # 3. Are the errors actually perception errors?
    perception_share = error_mix.get("perception", 0.0)
    if perception_share < PERCEPTION_ERROR_THRESHOLD:
        dominant = max(error_mix, key=lambda k: error_mix[k]) if error_mix else "unknown"
        return GateDecision(
            decision="hold",
            rationale=(
                f"Accuracy is below target ({'; '.join(below)}) but only {perception_share:.0%} of "
                f"errors in the image-only/scanned subsets are perception-type — the dominant "
                f"class there is {dominant!r}. The model is "
                "reading characters correctly and placing them in the wrong field, which is an "
                "LLM/projector problem. Unfreezing the ViT would not help and would cost a full "
                "Foundation retrain (arch §3)."
            ),
            image_only_accuracy=image_only_accuracy,
            scanned_accuracy=scanned_accuracy,
            image_only_documents=image_only_documents,
            scanned_documents=scanned_documents,
            error_mix=error_mix,
            overall_error_mix=overall_mix,
            recommendation=(
                "Keep train_vit: false. Look at prompt and schema design, corpus coverage for the "
                "failing document type, and confusable discrimination instead."
            ),
        )

    return GateDecision(
        decision="fire",
        rationale=(
            f"Accuracy below target ({'; '.join(below)}) AND {perception_share:.0%} of errors are "
            "perception-type — misread characters and missed checkboxes. The visual reading itself "
            "is the bottleneck, which is the one condition that justifies the cost (arch §3)."
        ),
        image_only_accuracy=image_only_accuracy,
        scanned_accuracy=scanned_accuracy,
        image_only_documents=image_only_documents,
        scanned_documents=scanned_documents,
        error_mix=error_mix,
        overall_error_mix=overall_mix,
        recommendation=(
            "Try the cheap interventions first: raise the resolution cap toward 2048px, and "
            "rebalance the modality mix toward image_only (neither needs new labeling). If both "
            "fail, retrain Foundation with a ViT LoRA — `--train-vit`. Never a full fine-tune: "
            "that risks the pretrained OCR capability the image-only pathway depends on."
        ),
    )


def evaluate_from_report(report: dict[str, Any], **overrides: Any) -> GateDecision:
    """Run the gate against an ``EvalReport.as_dict()`` payload (IMPL-08).

    The keys are read from the report's real shape. Reading invented ones
    (``eval_metrics`` / ``subset_counts`` / ``error_records``) returned None
    accuracies and zero documents for every well-formed report, so the gate
    decided from nothing at all — and, before the fix above, called it `hold`.
    """
    if "gate_metrics" not in report and "by_doc_type" not in report:
        raise ValueError(
            "this is not an EvalReport payload: it has neither `gate_metrics` nor `by_doc_type`. "
            "Guessing at the shape is how the gate came to decide from all-None inputs — pass "
            "EvalReport.as_dict(), or the JSON written from it."
        )

    metrics = report.get("gate_metrics", {})

    # by_doc_type: {doc_type: [subset_dict, ...]}. Documents and error records
    # are per doc_type x subset, so both are summed across doc types.
    subset_documents: dict[str, int] = {}
    errors: list[ErrorRecord] = []
    for subset_reports in report.get("by_doc_type", {}).values():
        for subset_report in subset_reports:
            name = subset_report.get("subset", "")
            subset_documents[name] = subset_documents.get(name, 0) + int(
                subset_report.get("documents", 0)
            )
            for record in subset_report.get("error_records", []):
                errors.append(
                    ErrorRecord(
                        source_id=record.get("source_id", ""),
                        field_path=record.get("field_path", ""),
                        expected=record.get("expected"),
                        got=record.get("got"),
                        error_class=record.get("error_class", "unknown"),
                        # A record inside the image_only subset IS image-only,
                        # whether or not the scorer restated that on the record.
                        modality_mode=record.get(
                            "modality_mode",
                            "image_only" if name == "image_only" else "ocr_plus_image",
                        ),
                        is_scanned=bool(record.get("is_scanned", name == "scanned")),
                    )
                )

    kwargs: dict[str, Any] = {
        "image_only_accuracy": metrics.get("image_only_accuracy"),
        "scanned_accuracy": metrics.get("scanned_accuracy"),
        "image_only_documents": subset_documents.get("image_only", 0),
        "scanned_documents": subset_documents.get("scanned", 0),
        "errors": errors,
    }
    kwargs.update(overrides)
    decision = evaluate_gate(**kwargs)
    log.info("ViT gate: %s — %s", decision.decision, decision.rationale)
    return decision
