"""The promotion gate (arch v2.1 §15.5).

**Why v1's gate was replaced.** It required the candidate to match or beat the
current release on *every* one of twelve metrics, within a flat tolerance of
0.001, with no override — on the reasoning that a gate which can be waived stops
being a guarantee.

That was right about the risk and wrong about the remedy. At pilot volume the
eval set holds three or four documents per type, so classification accuracy can
only take the values 0, 0.25, 0.5, 0.75, 1.0: the tolerance was two hundred and
fifty times finer than the measurement could resolve. With twelve metrics each
needing not to move down, a genuinely-equal model passed all twelve with
probability near 0.5¹² — about 0.02%. And it does not improve at scale: at n=50
the standard error of a proportion is ~0.07, still seventy times the tolerance.

A gate that cannot be passed and cannot be waived is not a strict gate. It is a
rule everyone agrees to ignore, on the afternoon it first blocks something
obviously fine.

**Three conditions replace it**, all of which must hold:

1. **Absolute floors.** Every gating metric meets its §0d floor. This applies to
   the first release too, which has no predecessor — v1's gate passed a first
   version on *zero* measured metrics.
2. **Non-inferiority.** For each metric, the lower bound of the 95% CI on
   (candidate − current), from a paired bootstrap over documents, sits above −δ.
   Per-metric δ, because a 1pp drop in identifier accuracy and a 1pp drop in
   free-text similarity are not the same event.
3. **Improvement.** At least one primary metric improves with its CI lower bound
   above zero — or the release fixes a documented defect. Without this a model
   could pass forever on non-inferiority alone.

**And an override exists**, requiring a named person and a written reason, both
recorded in the gate decision and the release bundle. A waiver nobody can
attribute later is the thing actually worth preventing.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from evaluation.bootstrap import (
    RESAMPLES,
    SEED,
    ConfidenceInterval,
    holm_step_down,
    mean_lower_bound,
    minimum_detectable_drop,
    non_inferiority_test,
    paired_bootstrap,
)

log = logging.getLogger(__name__)

Direction = Literal["higher_is_better", "lower_is_better"]

#: Every gating metric, and which direction counts as improvement.
#: `alias_accuracy` is deliberately absent: it is reported, not gated, because a
#: rare surface variant has too little support for a stable threshold and gating
#: on it would block good models on noise (arch §0c). So is
#: `held_out_carrier_match` — reported, because a one-carrier-per-line slice is
#: too small to gate on and its whole purpose is to be harder than the rest.
GATING_METRICS: dict[str, Direction] = {
    "field_exact_match": "higher_is_better",
    "field_normalized_match": "higher_is_better",
    "field_f1_list_fields": "higher_is_better",
    "list_field_recall": "higher_is_better",
    "schema_validity_rate": "higher_is_better",
    "ocr_arbitration_accuracy": "higher_is_better",
    "image_only_accuracy": "higher_is_better",
    "scanned_accuracy": "higher_is_better",
    "doc_type_classifier_accuracy": "higher_is_better",
    "lob_detection_accuracy": "higher_is_better",
    "ece_confidence": "lower_is_better",
    "confusable_misattribution_rate": "lower_is_better",
    "false_null_rate": "lower_is_better",
    "lossrun_totals_reconciliation_rate": "higher_is_better",
    # Loss Run claims found by claim number with total incurred within a cent
    # (evaluation.metrics.lossrun_table; Fideon SPEC_09 handoff item 8).
    "table_f1": "higher_is_better",
    # Of the values delivered without a review flag, the share that were wrong,
    # on the golden set through serving with the release's thresholds
    # (evaluation.metrics.auto_accept). The number the thresholds promise.
    "auto_accept_error_rate": "lower_is_better",
    # The weakest common-model line's field match (evaluation.run_eval): one
    # line cannot hide behind the others' volume.
    "worst_line_field_match": "higher_is_better",
    # The model must invent nothing (decision 2026-10-24): of the values it
    # wrote that the page text could settle, the share no page of the document
    # prints. On the golden set through serving, read from serving's own
    # printed-values check (ExtractionResult.printed_check), recorded before
    # strict mode removes them, so it measures the model (evaluation.run_eval).
    "hallucination_rate": "lower_is_better",
    # Its output must be consistent (decision 2026-10-24): the share of fields
    # equal between two readings of the same document in the main mode, the
    # second with its windows sent in reverse order (evaluation.golden_eval).
    "run_to_run_agreement": "higher_is_better",
    # What users receive (decision 2026-10-26): field match on the image_only
    # readings of REAL documents, as served - checked on the borrowed MinerU
    # text, as production checks them (evaluation.run_eval). The tier, the
    # improvement rule and non-inferiority read it (SERVED_FIELD_MATCH).
    "served_field_match": "higher_is_better",
    # The mode short PDFs and the fallback read: field match on the
    # ocr_plus_image readings of real documents. Held to its floor alone.
    "real_ocr_plus_image_accuracy": "higher_is_better",
}

#: Gating under arch v2.1 §15.2, but **not enforced yet** — nothing produces
#: them. Each names the phase that turns it on.
#:
#: This distinction is not a softening of the gate; it is the same mistake being
#: avoided twice. `require_all_measured` blocks on any unmeasured metric, and
#: rightly so — absence is not success. But listing a metric no producer emits
#: makes the gate permanently unpassable, which is precisely the defect this
#: rewrite exists to fix. A metric moves up into GATING_METRICS in the same
#: commit as the code that computes it, and `test_module_seams` checks that what
#: the gate demands is what the scorer emits.
PENDING_GATING_METRICS: dict[str, str] = {
    "list_field_precision": "row alignment (§15.1 Hungarian matching) — not written",
}

#: Metrics that only exist when the eval set contains the relevant documents.
#: Their absence means **not applicable**, not "not measured" — an ACORD-only
#: eval set has nothing to say about Loss Run reconciliation, and blocking on it
#: would make the gate a statement about the eval set's composition.
#:
#: Deliberately narrow, because it is otherwise a loophole: a conditional metric
#: is exempt ONLY from the "was it measured" requirement. When it IS present it
#: faces the same floor and the same non-inferiority test as any other. Adding a
#: metric here because it is inconvenient to produce would be the v1 mistake in
#: reverse — a gate that passes by no longer looking.
CONDITIONAL_METRICS: frozenset[str] = frozenset({
    # Only Loss Runs have claim rows and printed totals.
    "lossrun_totals_reconciliation_rate",
    "table_f1",
    # Only canonical outputs carry review flags. When present it faces its
    # floor like any other metric.
    "auto_accept_error_rate",
    # Only common-model lines with enough values scored per line produce it.
    "worst_line_field_match",
    # Only an eval set holding scanned documents measures it. A line whose
    # documents all arrive digital (the CGL originals) has none to hold, and a
    # set that holds scans still holds it to non-inferiority with the baseline.
    "scanned_accuracy",
    # Only page text can settle whether a value is printed: a frozen set whose
    # pages carry no text (page_N.md is optional), or read only from images,
    # has no value to check, and blocking on that would make the gate a
    # statement about how the set was frozen - a frozen set cannot be given
    # text afterwards. Not a way to stop looking: once a promoted release has
    # measured it, the frozen set has the text, and a candidate that does not
    # measure it is blocked as unmeasured (a printed-values check that stopped
    # recording, say).
    "hallucination_rate",
    # Only measured when the golden eval repeats its ``run_to_run_mode`` reading
    # (``run_to_run_check`` in the adapter's training config, on by default). A
    # switch that blocked every release when used would not be a switch; once a
    # promoted release has measured it, every later candidate must too.
    "run_to_run_agreement",
    # Only an eval set holding REAL documents measures them: a set of synthetic
    # twins alone has no real reading. Not a way to stop looking: without
    # served_field_match the release is never production (tier_evidence), and
    # once a promoted release has measured them, every later candidate must too.
    "served_field_match",
    "real_ocr_plus_image_accuracy",
})


def not_applicable_for() -> frozenset[str]:
    """What the adapter structurally cannot produce.

    Derived from what it trains (:func:`common.adapter.not_applicable_metrics`),
    never from a list anyone declared: a metric the adapter CAN produce and did
    not is unmeasured, which blocks.

    ``CONDITIONAL_METRICS`` stays on top of it. That set is about the eval SET
    (no scanned document in it, say), which is a different question from what
    the adapter covers.
    """
    from common.adapter import not_applicable_metrics

    return CONDITIONAL_METRICS | not_applicable_metrics()

#: NOT conditional, deliberately: image_only_accuracy and ocr_arbitration_accuracy.
#: They look like subset metrics, but every document is read in both regimes, so
#: an eval run that produced neither has a defective eval set, and that is worth
#: blocking on. Exempting them would let the eval set quietly stop covering the
#: no-OCR production path while the gate kept passing. scanned_accuracy is
#: conditional (above): scans are not always there to hold.

#: Absolute floors (arch §0d, pilot-exit column). The single source for the
#: thresholds the gate and the pilot share: ``pilot.pilot_run`` derives field
#: match, list recall, schema validity, LoB detection and confusable rate from
#: this table, with the same >= / <= comparison; its relative criteria (image-only
#: gap, alias gap, row completeness) have no floor here. Production values are
#: set by the product owner at pilot exit, which is why these are named pilot
#: floors rather than hardcoded forever.
PILOT_FLOORS: dict[str, float] = {
    "field_exact_match": 0.85,
    "field_normalized_match": 0.85,
    "list_field_recall": 0.75,
    "schema_validity_rate": 0.98,
    "doc_type_classifier_accuracy": 0.95,
    "lob_detection_accuracy": 0.85,
    "confusable_misattribution_rate": 0.05,
    "lossrun_totals_reconciliation_rate": 0.80,
    # Applies to a release that serves Loss Runs; the policy adapter cannot
    # produce it (common.adapter.not_applicable_metrics).
    "table_f1": 0.88,
    # A pilot choice, not a measured one: at most 2 in 100 unreviewed values
    # wrong. The per-type targets are stricter (1% for money, dates, numbers,
    # identifiers; 2% enums; 5% names and addresses), so a release whose
    # thresholds hold on unseen documents passes with room.
    "auto_accept_error_rate": 0.02,
    "worst_line_field_match": 0.85,
    # The field-match floor, on what users receive and on the mode short PDFs
    # and the fallback read (decision 2026-10-26). field_normalized_match keeps
    # its own floor above.
    "served_field_match": 0.85,
    "real_ocr_plus_image_accuracy": 0.85,
    # No ceiling on hallucination_rate and no floor on run_to_run_agreement yet:
    # the user sets them after the first measured run (decision 2026-10-24).
    # Until then each is held to non-inferiority with the promoted release
    # alone, and a first measurement passes on it.
}

#: Non-inferiority margins in absolute units (arch v2.1 §15.5). Per metric,
#: because a 1pp drop on identifiers is a different event from 1.5pp on names:
#: a wrong policy number is a wrong extraction, a slightly-off entity name is
#: usually still matchable.
NON_INFERIORITY_DELTA: dict[str, float] = {
    "field_exact_match": 0.010,
    "field_normalized_match": 0.010,
    "field_f1_list_fields": 0.015,
    "list_field_recall": 0.010,
    "schema_validity_rate": 0.005,
    "ocr_arbitration_accuracy": 0.015,
    "image_only_accuracy": 0.015,
    "scanned_accuracy": 0.015,
    "doc_type_classifier_accuracy": 0.005,
    "lob_detection_accuracy": 0.015,
    "ece_confidence": 0.010,
    "confusable_misattribution_rate": 0.010,
    "false_null_rate": 0.010,
    "lossrun_totals_reconciliation_rate": 0.015,
    "table_f1": 0.015,
    # Half a point: on a 2% ceiling, more would let a release double the wrong
    # values users receive unreviewed and still count as no worse.
    "auto_accept_error_rate": 0.005,
    "worst_line_field_match": 0.015,
    # The same margin as a printed value left empty: a value invented and one
    # left out are the two ways a release stops reading what the page prints.
    "hallucination_rate": 0.010,
    # The margin field_normalized_match was held to, on the number that
    # replaced it in non-inferiority (decision 2026-10-26).
    "served_field_match": 0.010,
    # run_to_run_agreement takes DEFAULT_DELTA.
}
DEFAULT_DELTA = 0.015

#: Gating metrics held to their floor alone (decision 2026-10-26). Their
#: movement against the promoted release is still computed and reported (the
#: verdict's interval), never gated: non-inferiority reads served_field_match,
#: what users receive, instead of the three-mode field_normalized_match; and
#: the ocr_plus_image accuracy on real documents was given a floor of its own,
#: nothing more.
FLOOR_ONLY_METRICS: frozenset[str] = frozenset({
    "field_normalized_match",
    "real_ocr_plus_image_accuracy",
})

#: At least one of these must improve, or the release must fix a documented
#: defect. They are the metrics the business actually experiences: the field
#: match users receive (served_field_match, which replaced the three-mode
#: field_normalized_match on 2026-10-26) and list recall.
PRIMARY_METRICS: tuple[str, ...] = ("served_field_match", "list_field_recall")

#: The gates beside GATING_METRICS that promotion_gate can fail, by the names an
#: override waives them under: the improvement rule, and a continued run's
#: missing per-line comparison with its parent. A continued run's regression on
#: one line is ``line:<line>`` (:func:`is_gate`).
OTHER_GATES: tuple[str, ...] = ("improvement", "line_regression_evidence")

#: The count, in the candidate's metrics, of golden-eval readings that failed to
#: extract twice (``golden_eval.evaluate_version``). Above 0 the gate does not
#: decide: it blocks, and asks for the eval to be run again. Not a gate an
#: override can waive - a decision on part of the set is no decision.
INFRASTRUCTURE_FAILURES = "infrastructure_failures"


def is_gate(name: str) -> bool:
    """Whether ``name`` is a gate :func:`promotion_gate` can fail, so an override
    can waive it: a gating metric, one of :data:`OTHER_GATES`, or ``line:<line>``
    for a line of business with a canonical policy schema."""
    from common.lob import known_lines

    if name in GATING_METRICS or name in OTHER_GATES:
        return True
    kind, sep, line = name.partition(":")
    return bool(sep) and kind == "line" and line in known_lines()


class GateError(RuntimeError):
    """Raised when a gate decision cannot be made."""


def _finite(value: Any) -> bool:
    """A real, finite number (a bool is not one: it is a reading, not a rate)."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


@dataclass
class MetricVerdict:
    """One metric's outcome: its floor, and its movement against the baseline."""

    name: str
    direction: Direction
    candidate: float | None
    current: float | None = None
    floor: float | None = None
    interval: ConfidenceInterval | None = None
    delta: float = DEFAULT_DELTA

    #: Nothing in this run could have produced this metric — no Loss Run in
    #: what the adapter trains, no scan in the eval set. Distinct from `unmeasured`, which means
    #: something could have produced it and did not: that blocks, this does not.
    applicable: bool = True

    #: False for a metric held to its floor alone (:data:`FLOOR_ONLY_METRICS`):
    #: its interval is reported, and a drop does not block.
    comparison_gated: bool = True

    @property
    def unmeasured(self) -> bool:
        return self.candidate is None

    @property
    def not_a_number(self) -> bool:
        """A value is there, but it is no finite number: NaN, an infinity, or
        not a number at all.

        Kept apart from ``unmeasured`` because it fails differently: NaN loses
        every comparison, so ``candidate <= floor`` and ``signed > -delta`` are
        both False - and with no floor and no baseline (a first release's
        hallucination_rate or run_to_run_agreement) nothing compared it, and it
        passed. It blocks instead (fails closed): a rate that came out NaN was
        divided by nothing or broke on the way, and says nothing either way.
        """
        return self.candidate is not None and not _finite(self.candidate)

    @property
    def meets_floor(self) -> bool:
        """Whether the absolute floor is met. Applies with or without a baseline."""
        if self.candidate is None or self.not_a_number:
            return False
        if self.floor is None:
            return True
        return (
            self.candidate >= self.floor if self.direction == "higher_is_better"
            else self.candidate <= self.floor
        )

    @property
    def non_inferior(self) -> bool:
        """Whether the candidate is not meaningfully worse than the baseline.

        With no baseline this is vacuously true — the floor is what gates a first
        release. With a baseline but no per-document scores, it falls back to a
        point comparison against δ, which is weaker than a bootstrap and is
        recorded as such in ``basis``.
        """
        if self.not_a_number:
            return False
        if self.current is None or self.candidate is None:
            return True
        if self.interval is not None:
            return self.interval.non_inferior(self.delta)
        signed = (
            self.candidate - self.current if self.direction == "higher_is_better"
            else self.current - self.candidate
        )
        return signed > -abs(self.delta)

    @property
    def improved(self) -> bool:
        """Improvement distinguishable from noise. Requires an interval."""
        return bool(self.interval and self.interval.improved)

    @property
    def basis(self) -> str:
        if self.current is None:
            return "floor_only"
        return "paired_bootstrap" if self.interval is not None else "point_estimate"

    def describe(self) -> str:
        if self.candidate is None:
            # Two different statements, never collapsed into one: "nothing here
            # could produce it" is a fact about coverage, "it was not measured"
            # is a missing result.
            return f"{self.name}: {'NOT APPLICABLE' if not self.applicable else 'NOT MEASURED'}"
        if self.not_a_number:
            return f"{self.name}: NOT A NUMBER ({self.candidate!r})"
        parts = [f"{self.name}: {self.candidate:.4f}"]
        if self.floor is not None:
            parts.append(f"floor {self.floor:.4f} {'✓' if self.meets_floor else '✗'}")
        if self.interval is not None:
            parts.append(self.interval.describe().split(": ", 1)[1])
        elif self.current is not None:
            parts.append(f"was {self.current:.4f} ({self.candidate - self.current:+.4f})")
        return " | ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "candidate": self.candidate,
            "current": self.current,
            "floor": self.floor,
            "delta": self.delta,
            "meets_floor": self.meets_floor,
            "non_inferior": self.non_inferior,
            "improved": self.improved,
            "basis": self.basis,
            "applicable": self.applicable,
            "comparison_gated": self.comparison_gated,
            "interval": self.interval.as_dict() if self.interval else None,
        }


@dataclass
class GateOverrideRecord:
    """A named, written waiver of specific gates (arch v2.1 §15.5)."""

    approver: str
    reason: str
    waived_gates: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.approver.strip() or self.approver.strip().lower() in {
            "system", "ci", "automated", "n/a", "none", "-",
        }:
            raise GateError(
                "a gate override must name the person accountable for it, not a system "
                "identity. A waiver nobody can attribute later is the thing worth preventing."
            )
        if len(self.reason.strip()) < 20:
            raise GateError(
                "a gate override needs a written reason, not a word. The reason is what a "
                "later reviewer reads when asking why this shipped."
            )
        if not self.waived_gates:
            raise GateError(
                "an override must name the gates it waives. A blanket override is the "
                "unaccountable waiver in a different costume."
            )
        # A misspelt gate waives nothing, and the release it was written for
        # stays blocked by the gate the approver meant.
        unknown = sorted(g for g in self.waived_gates if not is_gate(g))
        if unknown:
            raise GateError(
                f"an override names {unknown}, which is not a gate. Name each gate as the gate "
                f"decision names it: a gating metric ({', '.join(sorted(GATING_METRICS))}), "
                f"{' or '.join(OTHER_GATES)}, or line:<line>."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "approver": self.approver,
            "reason": self.reason,
            "waived_gates": sorted(self.waived_gates),
        }


@dataclass
class GateResult:
    """The gate's decision, and everything behind it."""

    passed: bool
    #: ``production`` or ``interim`` for a release that passed (release_tier);
    #: ``None`` for one that did not.
    tier: str | None = None
    verdicts: list[MetricVerdict] = field(default_factory=list)
    failed_gates: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    is_first_version: bool = False
    override: GateOverrideRecord | None = None


    #: Gates that failed but were waived. Kept separate from `failed_gates` so
    #: "passed with a waiver" is never indistinguishable from "passed".
    waived: list[str] = field(default_factory=list)

    #: ``(gate, reason)`` for every block, so an override removes a reason by the
    #: gate that raised it. Matching on the reason's opening words missed every
    #: reason not phrased as "<gate> ...", so waiving ``improvement`` or a
    #: ``line:`` gate left the release blocked with the waiver recorded.
    blocks: list[tuple[str, str]] = field(default_factory=list)

    #: What the decision rests on that a reader must know, without blocking:
    #: an eval set accepted below its minimum size, for one.
    warnings: list[str] = field(default_factory=list)

    #: Why the promoted release's scores were not compared with the candidate's
    #: (another frozen set, another tenant's set, other scoring code), when they
    #: were not: the candidate was then judged on its floors alone. Not a first
    #: release - there is a predecessor, on another yardstick.
    baseline_not_comparable: str | None = None

    #: A continued run's line-by-line comparison (decisions 2026-10-23 and
    #: 2026-10-26): per line and check - ``parent`` (the run it continued) or
    #: ``drift`` (the last run from the base) - the run held to, the documents,
    #: both field matches, the change, the margin, the one-sided bound at the
    #: Holm-adjusted level, and whether the line was gated or only warned.
    line_verdicts: list[dict[str, Any]] = field(default_factory=list)

    #: What the tier rests on (:func:`tier_evidence`): served_field_match, its
    #: one-sided 95% lower bound over real documents, and why a passed release
    #: is not production. ``None`` when the gate did not decide.
    tier_evidence: TierEvidence | None = None

    @property
    def tier_confidence(self) -> str | None:
        """``confirmed`` or ``unconfirmed`` for a production release; ``None``
        otherwise (:func:`tier_evidence`)."""
        return self.tier_evidence.tier_confidence if self.tier_evidence else None

    @property
    def improved_metrics(self) -> list[str]:
        return [v.name for v in self.verdicts if v.improved]

    def report(self) -> str:
        verdict = "PASS" if self.passed else "BLOCKED"
        if self.passed and self.waived:
            verdict = f"PASS (with {len(self.waived)} waived gate(s))"
        lines = [f"promotion gate: {verdict}"]
        if self.is_first_version:
            lines.append("  first release — gated on absolute floors alone (arch v2.1 §15.5)")
        if self.baseline_not_comparable:
            lines.append(
                "  promoted release on another yardstick — gated on absolute floors alone: "
                f"{self.baseline_not_comparable}"
            )
        lines.extend(f"  {v.describe()}" for v in self.verdicts if not v.unmeasured)
        lines.extend(
            f"  UNMEASURED: {v.name}" for v in self.verdicts if v.unmeasured and v.applicable
        )
        lines.extend(
            f"  not applicable: {v.name}"
            for v in self.verdicts if v.unmeasured and not v.applicable
        )
        lines.extend(f"  {_describe_line_verdict(v)}" for v in self.line_verdicts)
        if self.tier_evidence is not None:
            lines.append(f"  {self.tier_evidence.describe()}")
        lines.extend(f"  BLOCKED: {reason}" for reason in self.blocking_reasons)
        lines.extend(f"  WARNING: {warning}" for warning in self.warnings)
        if self.override:
            lines.append(
                f"  OVERRIDE by {self.override.approver}: {self.override.reason} "
                f"(waived {sorted(self.override.waived_gates)})"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tier": self.tier,
            "tier_confidence": self.tier_confidence,
            "tier_evidence": self.tier_evidence.as_dict() if self.tier_evidence else None,
            "is_first_version": self.is_first_version,
            "baseline_not_comparable": self.baseline_not_comparable,
            "not_applicable": sorted(
                v.name for v in self.verdicts if v.unmeasured and not v.applicable
            ),
            "failed_gates": sorted(self.failed_gates),
            "waived_gates": sorted(self.waived),
            "blocking_reasons": list(self.blocking_reasons),
            "improved_metrics": self.improved_metrics,
            "verdicts": [v.as_dict() for v in self.verdicts],
            "line_verdicts": list(self.line_verdicts),
            "override": self.override.as_dict() if self.override else None,
            "warnings": list(self.warnings),
        }


def promotion_gate(
    candidate_metrics: dict[str, Any],
    current_metrics: dict[str, Any] | None,
    *,
    per_document: dict[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    floors: dict[str, float] | None = None,
    continued_from: str | None = None,
    line_evidence: LineEvidence | None = None,
    line_evidence_missing: str | None = None,
    require_all_measured: bool = True,
    fixes_defect: str | None = None,
    override: GateOverrideRecord | None = None,
    small_eval_set: bool = False,
    warnings: Sequence[str] | None = None,
    baseline_not_comparable: str | None = None,
    served_documents: ServedDocuments | None = None,
) -> GateResult:
    """Decide whether a candidate release may be promoted.

    Args:
        candidate_metrics: aggregate scores over the frozen golden eval set.
        current_metrics: the promoted release's scores, or ``None`` for the first.
        per_document: ``{metric: (current_scores, candidate_scores)}``, aligned by
            index, or ``(current_scores, candidate_scores, current_weights,
            candidate_weights)`` for a metric each document counts in by its
            weight (``paired_bootstrap``). Supplied, each metric is judged by a
            paired bootstrap rather than by comparing two point estimates —
            which is the whole reason this gate can be passed at all. A metric
            with fewer than two paired documents is compared as a point
            estimate instead, with a warning.
        floors: absolute floors, defaulting to the §0d pilot-exit table. Which
            metrics are NOT APPLICABLE comes from what the adapter trains
            (:func:`not_applicable_for`): a policy adapter has no Loss Run to
            reconcile.
        continued_from: the run (or directory) this run's adapter continued, if
            it did: it is then held line by line to the runs before it.
        line_evidence: that line-by-line comparison (:class:`LineEvidence`), or
            ``None`` with ``line_evidence_missing`` saying why none could be made.
        fixes_defect: satisfies the improvement requirement when a release exists
            to fix something rather than to score higher. Checked
            (:func:`documented_defect`): a blank or placeholder value is refused.
        override: a named, written waiver. Validated — an override that cannot be
            attributed is refused.
        small_eval_set: the release was scored on fewer eval documents of a type
            than the frozen set's minimum (a set accepted with ``freeze-eval-set
            --allow-small``). Its scores
            say too little to call a release production, so a pass is capped at
            ``interim`` and warned.
        warnings: what the caller knows about the evidence that a reader of the
            decision must see (per-document scores it could not pair, and why).
            Recorded first in ``GateResult.warnings``.
        baseline_not_comparable: why ``current_metrics`` are not on the
            candidate's yardstick - scored on a set frozen again, on another
            tenant's set, or under other scoring code (decision 2026-10-22).
            Given, the candidate is judged on its floors alone: no
            non-inferiority comparison and no improvement rule, and the reason
            is a warning in the decision. The old release is not scored again.
        served_documents: the candidate's per-document ``served_field_match``
            over its real documents, ``(scores, weights)`` or ``scores``
            alone (the eval report's ``per_document`` and
            ``per_document_weights``). The tier's lower bound is bootstrapped
            from them (:func:`tier_evidence`); without them a passed release is
            interim - production is shown, never assumed.
    """
    result = GateResult(passed=True, is_first_version=current_metrics is None)
    result.warnings.extend(warnings or ())
    if baseline_not_comparable and current_metrics is not None:
        # Numbers from two yardsticks say nothing about each other: a drop may be
        # the new golds, a rise the new label rules.
        result.baseline_not_comparable = baseline_not_comparable
        result.warnings.append(
            f"the promoted release's scores were not compared: {baseline_not_comparable}. The "
            "candidate was judged on its floors alone (decision 2026-10-22): no non-inferiority "
            "comparison and no improvement rule."
        )
        current_metrics, per_document = None, None
    baseline = current_metrics or {}
    floor_table = dict(PILOT_FLOORS) if floors is None else floors
    not_applicable = not_applicable_for()
    samples = per_document or {}

    if fixes_defect is not None:
        fixes_defect = documented_defect(fixes_defect)
    if override is not None:
        override.validate()
        result.override = override

    failures = candidate_metrics.get(INFRASTRUCTURE_FAILURES)
    if failures is not None and (not _finite(failures) or failures > 0):
        # Not a verdict on the model: readings the engine failed twice were not
        # scored, so every metric is over part of the frozen set. Deciding on
        # it would either pass a release on the documents it managed or block
        # it for an outage. No override waives this (it names no gate an
        # override may name, is_gate): the eval is run again.
        _block(result, INFRASTRUCTURE_FAILURES,
               f"{failures} reading(s) of the frozen eval set failed to extract even after a retry, "
               "so they were not scored and the gate does not decide on this measurement: re-run the "
               "gate. The eval report lists them (infrastructure_failures); if the same readings fail "
               "again, the failure is in serving or the frozen document, not the engine.")
        result.passed = False
        result.tier = None
        log.info("%s", result.report())
        return result

    for name, direction in GATING_METRICS.items():
        verdict = MetricVerdict(
            name=name,
            direction=direction,
            candidate=candidate_metrics.get(name),
            current=baseline.get(name),
            floor=floor_table.get(name),
            delta=NON_INFERIORITY_DELTA.get(name, DEFAULT_DELTA),
            # A conditional metric the baseline measured is required: the eval
            # set is frozen, so its documents are the same, and a metric the last
            # release produced and this one did not is something broken (scan
            # detection, a subset's scoring), not a set with nothing to measure.
            applicable=name not in not_applicable
            or (name in CONDITIONAL_METRICS and baseline.get(name) is not None),
            comparison_gated=name not in FLOOR_ONLY_METRICS,
        )
        if current_metrics is not None and verdict.current is None and verdict.candidate is not None:
            # A promoted report scored before the metric was measured: nothing
            # to compare with, so the candidate's first measurement is judged on
            # the floor alone - never blocked for a comparison that cannot be
            # made - and the decision says so rather than reading as compared.
            result.warnings.append(
                f"{name}: the promoted release's eval report does not measure it, so it was judged "
                + (f"on its floor ({verdict.floor:.4f}) alone" if verdict.floor is not None
                   else "on no floor (none is set yet) and no comparison")
                + ": no non-inferiority test. Once a release that measures it is promoted, later "
                "candidates are compared with it."
            )
        if name in samples and verdict.current is not None:
            current_scores, candidate_scores, *weights = samples[name]
            current_weights, candidate_weights = (list(weights) + [None, None])[:2]
            if len(current_scores) == len(candidate_scores) < 2:
                # One document resamples to itself, so the bootstrap widens its
                # interval to the whole range, and that fails non-inferiority for
                # ever: a frozen set holding one scanned document blocked every
                # later candidate on scanned_accuracy, identical scores or not, and
                # a frozen set cannot grow. Too few to bootstrap is judged as a
                # point estimate instead, and says so.
                result.warnings.append(
                    f"{name}: only {len(current_scores)} paired document(s), too few to "
                    "bootstrap, so it is compared as a point estimate against δ and cannot "
                    "count as an improvement."
                )
            else:
                # Sign-normalise for lower-is-better metrics so "improvement" means
                # the same direction for every interval, and the CI can be read the
                # same way regardless of which metric it belongs to.
                if direction == "lower_is_better":
                    current_scores = [-x for x in current_scores]
                    candidate_scores = [-x for x in candidate_scores]
                verdict.interval = paired_bootstrap(
                    name, current_scores, candidate_scores,
                    current_weights=current_weights, candidate_weights=candidate_weights,
                )
        result.verdicts.append(verdict)

        if verdict.unmeasured:
            if require_all_measured and verdict.applicable:
                _block(result, name,
                       f"{name} was not measured. A metric that was not measured has not "
                       "passed — treating its absence as success is how a regression ships.")
            else:
                log.info(
                    "%s is not applicable to this run — nothing it covers produces it. That is "
                    "not a pass, it is a statement about what was scored.", name,
                )
            continue
        if verdict.not_a_number:
            # Before the floor and non-inferiority, which NaN passes by losing
            # every comparison when there is nothing to compare it with.
            _block(result, name,
                   f"{name} is {verdict.candidate!r}, not a finite number. NaN loses every "
                   "comparison, so with no floor and no baseline it was passing unchecked; a "
                   "metric that came out as no number was not measured on anything, and fails "
                   "closed. Find why the scoring produced it (a rate divided by nothing, a "
                   "broken reading) and score again.")
            continue
        if not verdict.meets_floor:
            _block(result, name,
                   f"{name} is {verdict.candidate:.4f} against a floor of {verdict.floor:.4f} "
                   "(arch v2.1 §0d). Floors apply to the first release too.")
        if not verdict.non_inferior and verdict.comparison_gated:
            bound = verdict.interval.lower if verdict.interval else None
            detail = (
                f"95% CI lower bound {bound:+.4f} is below -{verdict.delta:.3f}"
                if bound is not None else
                f"dropped by more than {verdict.delta:.3f} (point estimate; no paired "
                "per-document scores to bootstrap, so this is weaker evidence than one)"
            )
            _block(result, name, f"{name} is not non-inferior: {detail}")

    _require_improvement(result, fixes_defect)
    _require_line_non_regression(result, continued_from, line_evidence, line_evidence_missing)

    # An override lifts only the gates it names, and only ones that actually
    # failed. Waiving a gate that passed would make the record say something
    # untrue about what the approver decided.
    if result.override:
        waived = sorted(set(result.override.waived_gates) & set(result.failed_gates))
        result.waived = waived
        result.failed_gates = [g for g in result.failed_gates if g not in waived]
        result.blocks = [(g, r) for g, r in result.blocks if g not in waived]
        result.blocking_reasons = [r for _, r in result.blocks]
        unused = sorted(set(result.override.waived_gates) - set(waived))
        if unused:
            # In the decision, not only the log: a reader of a passed release
            # must not take a waiver of a gate that never failed for one that
            # was needed.
            warning = (
                f"the override names {unused}, which did not fail here, so it waived nothing "
                "there. Recorded as written."
            )
            log.warning("%s", warning)
            result.warnings.append(warning)

    if small_eval_set:
        result.warnings.append(
            "this release was scored on fewer eval documents of a type than a frozen set's "
            "minimum (a set frozen with freeze-eval-set --allow-small): its intervals are wide, "
            "so a pass is capped at interim until it is scored on a full-size set."

        )
    result.passed = not result.blocking_reasons
    result.tier_evidence = tier_evidence(
        candidate_metrics, served_documents, small_eval_set=small_eval_set, passed=result.passed)
    result.tier = result.tier_evidence.tier
    log.info("%s", result.report())
    return result


#: The metric the tier, the improvement rule and non-inferiority read (decision
#: 2026-10-26): field match on the image_only readings of real documents, as
#: served. What users receive - production reads the page images and holds the
#: values to MinerU's text - where field_normalized_match averages three modes
#: and counts synthetic twins.
SERVED_FIELD_MATCH = "served_field_match"

#: Fideon SPEC_09 §6: a release is PRODUCTION only when its served field match
#: reaches this (and the bound below). One that clears every floor (0.85 and
#: the rest) but not this is promoted as INTERIM - usable, and labelled as not
#: yet at the production bar.
PRODUCTION_FIELD_MATCH = 0.92

#: How sure a production pass must be (decision 2026-10-26): the one-sided
#: lower bound on served_field_match, at :data:`TIER_CONFIDENCE`, bootstrapped
#: over real documents, must reach this.
PRODUCTION_LOWER_BOUND = 0.90

#: A production release whose lower bound reaches this too is ``confirmed``;
#: one between :data:`PRODUCTION_LOWER_BOUND` and this is ``unconfirmed``.
CONFIRMED_LOWER_BOUND = 0.92

#: The bound's confidence: one-sided 95%.
TIER_CONFIDENCE = 0.95

#: Real documents a production release must be scored on: the frozen set's
#: production minimum (``freeze_eval_set.MIN_FROZEN_DOCS_PER_TYPE``, arch v2.1
#: §15.4). The bound's documents are held to it too.
PRODUCTION_MIN_REAL_DOCUMENTS = 150

#: The candidate's per-document served_field_match: ``(scores, weights)``, or
#: the scores alone (each document counting once).
ServedDocuments = tuple[Sequence[float], Sequence[float] | None] | Sequence[float]


@dataclass
class TierEvidence:
    """What a release's tier rests on (:func:`tier_evidence`)."""

    #: ``production`` or ``interim`` for a release that passed; ``None`` for one
    #: that did not.
    tier: str | None
    #: ``confirmed`` (lower bound >= :data:`CONFIRMED_LOWER_BOUND`) or
    #: ``unconfirmed`` for a production release; ``None`` for any other.
    tier_confidence: str | None
    #: served_field_match as the gate metrics carry it.
    point_estimate: float | None
    #: Its one-sided lower bound over real documents; ``None`` with no
    #: per-document scores.
    lower_bound: float | None
    #: The real documents the bound was bootstrapped over.
    documents: int = 0
    #: Why a passed release is not production (empty for a production one).
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "tier_confidence": self.tier_confidence,
            "metric": SERVED_FIELD_MATCH,
            "point_estimate": None if self.point_estimate is None else round(self.point_estimate, 6),
            "lower_bound": None if self.lower_bound is None else round(self.lower_bound, 6),
            "documents": self.documents,
            "confidence": TIER_CONFIDENCE,
            "one_sided": True,
            "resamples": RESAMPLES,
            "seed": SEED,
            "production_point_bar": PRODUCTION_FIELD_MATCH,
            "production_bound_bar": PRODUCTION_LOWER_BOUND,
            "confirmed_bound_bar": CONFIRMED_LOWER_BOUND,
            "minimum_real_documents": PRODUCTION_MIN_REAL_DOCUMENTS,
            "not_production_because": list(self.reasons),
        }

    def describe(self) -> str:
        point = "n/a" if self.point_estimate is None else f"{self.point_estimate:.4f}"
        bound = "n/a" if self.lower_bound is None else f"{self.lower_bound:.4f}"
        text = (f"tier {self.tier or 'none'}"
                + (f" ({self.tier_confidence})" if self.tier_confidence else "")
                + f": {SERVED_FIELD_MATCH} {point}, one-sided {TIER_CONFIDENCE:.0%} lower bound "
                f"{bound} over {self.documents} real document(s)")
        if self.reasons and self.tier is not None:
            text += " - not production: " + "; ".join(self.reasons)
        return text


def _served_scores(served: ServedDocuments | None) -> tuple[list[float], list[float] | None]:
    """``(scores, weights)`` from what :func:`promotion_gate` was given."""
    if served is None:
        return [], None
    if (isinstance(served, tuple) and len(served) == 2
            and all(part is None or not isinstance(part, (int, float)) for part in served)):
        scores, weights = served
        return [float(s) for s in scores or ()], (None if weights is None else [float(w) for w in weights])
    return [float(s) for s in served], None


def tier_evidence(
    metrics: dict[str, Any],
    served_documents: ServedDocuments | None = None,
    *,
    small_eval_set: bool = False,
    passed: bool = True,
) -> TierEvidence:
    """The tier of a release, and what it rests on (decision 2026-10-26).

    PRODUCTION needs all of: the gate passed; at least
    :data:`PRODUCTION_MIN_REAL_DOCUMENTS` real documents (an eval set below the
    frozen minimum, ``small_eval_set``, is not, and neither is a bound over
    fewer); served_field_match >= :data:`PRODUCTION_FIELD_MATCH`; and its
    one-sided :data:`TIER_CONFIDENCE` lower bound, a document-level bootstrap
    over the real documents (``evaluation.bootstrap.mean_lower_bound``,
    10,000 resamples, seeded), >= :data:`PRODUCTION_LOWER_BOUND`. A passed
    release short of any of them is INTERIM, with every reason recorded.

    Missing evidence is never production: no served_field_match (the eval set
    holds no real document read image_only) or no per-document scores to bound
    it is interim, and says which.

    ``tier_confidence`` qualifies a production tier: ``confirmed`` when the
    lower bound itself reaches :data:`CONFIRMED_LOWER_BOUND`, ``unconfirmed``
    when only the point estimate does.
    """
    reasons: list[str] = []
    point = metrics.get(SERVED_FIELD_MATCH)
    point = float(point) if _finite(point) else None
    scores, weights = _served_scores(served_documents)
    bound = None
    if point is None:
        reasons.append(
            f"{SERVED_FIELD_MATCH} was not measured: the eval set holds no real document read "
            "image_only, so what users receive was not scored")
    if scores:
        measured = mean_lower_bound(SERVED_FIELD_MATCH, scores, weights, confidence=TIER_CONFIDENCE)
        bound = measured.lower
    elif point is not None:
        reasons.append(
            f"no per-document {SERVED_FIELD_MATCH} scores to bound it (the candidate's eval report "
            "holds none), so how sure the pass is cannot be shown")
    if small_eval_set:
        reasons.append(
            f"scored on fewer real documents of a type than the frozen set's minimum "
            f"({PRODUCTION_MIN_REAL_DOCUMENTS})")
    if scores and len(scores) < PRODUCTION_MIN_REAL_DOCUMENTS:
        reasons.append(
            f"the bound rests on {len(scores)} real document(s) read image_only, fewer than "
            f"{PRODUCTION_MIN_REAL_DOCUMENTS}")
    if point is not None and round(point, 6) < PRODUCTION_FIELD_MATCH:
        reasons.append(f"{SERVED_FIELD_MATCH} {point:.4f} is below {PRODUCTION_FIELD_MATCH}")
    if bound is not None and round(bound, 6) < PRODUCTION_LOWER_BOUND:
        reasons.append(
            f"its one-sided {TIER_CONFIDENCE:.0%} lower bound {bound:.4f} is below "
            f"{PRODUCTION_LOWER_BOUND}")
    tier = None if not passed else "interim" if reasons else "production"
    confidence = None
    if tier == "production" and bound is not None:
        confidence = "confirmed" if round(bound, 6) >= CONFIRMED_LOWER_BOUND else "unconfirmed"
    return TierEvidence(tier=tier, tier_confidence=confidence, point_estimate=point,
                        lower_bound=bound, documents=len(scores), reasons=reasons)


def release_tier(metrics: dict[str, Any], *, small_eval_set: bool = False,
                 served_documents: ServedDocuments | None = None) -> str:
    """``production`` or ``interim`` for a release that passed the gate
    (:func:`tier_evidence`, which says why)."""
    return str(tier_evidence(metrics, served_documents, small_eval_set=small_eval_set).tier)


def _block(result: GateResult, gate: str, reason: str) -> None:
    if gate not in result.failed_gates:
        result.failed_gates.append(gate)
    result.blocking_reasons.append(reason)
    result.blocks.append((gate, reason))


#: Words that name no defect. ``--fixes-defect`` lifts the improvement rule, so a
#: placeholder would lift it for nothing.
_NOT_A_DEFECT = frozenset({
    "none", "nil", "null", "nothing", "tbd", "todo", "fix", "fixes", "fixed", "bug", "bugs",
    "bugfix", "hotfix", "defect", "issue", "ticket", "yes", "true", "test", "wip", "misc",
    "unknown", "various",
})


def documented_defect(text: str) -> str:
    """``text`` stripped, when it names a defect: a ticket id (``TICKET-12``) or a
    sentence. Refused when blank or a placeholder (``-``, ``n/a``, ``fix``).

    A release that names the defect it fixes skips the improvement rule, so the
    name has to be something a later reader can look up. Any text at all did it
    before: ``--fixes-defect " "`` lifted the rule.
    """
    named = str(text or "").strip()
    letters = "".join(ch for ch in named.lower() if ch.isalnum())
    if len(letters) < 3 or letters in _NOT_A_DEFECT:
        raise GateError(
            f"fixes_defect {text!r} names no defect. It stands in for an improvement, so it must "
            "say which documented defect this release fixes - a ticket id (TICKET-12) or a "
            "sentence a later reviewer can check."
        )
    return named


def _require_improvement(result: GateResult, fixes_defect: str | None) -> None:
    """At least one primary metric must improve, or a defect must be named.

    Without this a model could pass forever on non-inferiority alone — every
    release no worse than the last, none better, and the cycle producing nothing.
    The first release is exempt: there is nothing to improve on. So is a
    release whose predecessor was scored on another yardstick: there is nothing
    to measure an improvement against.
    """
    if result.is_first_version or result.baseline_not_comparable or fixes_defect:
        return
    improved = [v.name for v in result.verdicts if v.name in PRIMARY_METRICS and v.improved]
    if improved:
        return
    measurable = [
        v for v in result.verdicts if v.name in PRIMARY_METRICS and v.interval is not None
    ]
    if not measurable:
        # No per-document scores means no interval, so "improved" is unknowable
        # rather than false. Blocking here would make the improvement rule fire
        # on missing evidence rather than on a missing improvement. Said in the
        # decision, not only the log: a release that passed with the rule
        # unenforced must not read as one that improved.
        warning = (
            f"the improvement rule was not enforced: no primary metric {list(PRIMARY_METRICS)} "
            "had paired per-document scores to bootstrap, so whether one improved is unknowable "
            "and non-inferiority was a point comparison."
        )
        log.warning("%s", warning)
        result.warnings.append(warning)
        return
    _block(
        result, "improvement",
        f"no primary metric improved with its CI lower bound above zero {list(PRIMARY_METRICS)}. "
        "A release must be better at something, or name the documented defect it fixes "
        "(arch v2.1 §15.5).",
    )


#: A continued run's line is GATED only on at least this many real documents
#: scored in both reports (decision 2026-10-26). Below it the line is warned,
#: with the smallest drop its documents could show (``minimum_detectable_drop``):
#: on a handful of documents a one-sided bound is either too wide to pass a
#: line that held or decided by one odd policy. Documents, not values: one
#: policy holds hundreds of values, so a bar on values let a single document
#: decide a line. Its own number, not the split's ``MIN_DOCS_TO_MEASURE_LINE``
#: (whether a line is measured at all).
LINE_MIN_DOCUMENTS = 30

#: The margin each line is held to against the PARENT, the run it continued:
#: one-sided, (candidate − parent) must sit above −this.
LINE_PARENT_MARGIN = 0.015

#: The margin each line is held to against the LAST FRESH run of the chain, the
#: run from the base the chain started at: small losses each under the parent
#: margin add up over cycles, and this catches them once they pass it.
LINE_DRIFT_MARGIN = 0.03

#: Family-wise alpha of each family of line tests (the parent checks, the drift
#: checks), held by Holm's step-down across the lines tested
#: (``evaluation.bootstrap.holm_step_down``).
LINE_FAMILY_ALPHA = 0.05

#: The power the minimum detectable drop of a warned line is stated at.
LINE_DETECTION_POWER = 0.80


@dataclass
class LineComparison:
    """One line's paired per-document field match: the candidate's against one
    run's on the real documents both scored. Aligned by index, as
    :func:`evaluation.bootstrap.paired_bootstrap` pairs."""

    line: str
    #: The run held to: the parent (``LineEvidence.compared``) or the last run
    #: from the base (``LineEvidence.drift``).
    reference: str
    reference_scores: list[float]
    candidate_scores: list[float]
    reference_weights: list[float] | None = None
    candidate_weights: list[float] | None = None
    #: Whether the reference's scores come from its stored answers re-scored by
    #: the scoring code running now (``evaluation.rescore``, decision
    #: 2026-10-26): its own report was on another yardstick only because the
    #: scoring code or its settings moved.
    reference_rescored: bool = False
    #: What the scores are (decision 2026-10-26): ``served_field_match`` per
    #: document - image_only readings of real documents, as served - or
    #: ``field_normalized_match``, the field match over every mode, for a line
    #: the reference's report holds no served scores of (written before they
    #: were kept).
    measure: str = "field_normalized_match"


@dataclass
class LineEvidence:
    """What a continued run's line-by-line gate rests on (decisions 2026-10-23
    and 2026-10-26)."""

    #: Each line against the PARENT, on every real document both scored. The
    #: gate decides which have enough documents to gate (LINE_MIN_DOCUMENTS).
    compared: list[LineComparison] = field(default_factory=list)
    #: Lines the parent scored that share no real document with the candidate,
    #: with that count (0): not compared, and warned.
    thin: dict[str, int] = field(default_factory=dict)
    #: Lines the parent's release serves and the candidate does not.
    dropped: list[str] = field(default_factory=list)
    #: Lines the candidate serves and the parent's release does not.
    added: list[str] = field(default_factory=list)
    #: The runs compared with, parent first, back to the last run from the base.
    chain: list[str] = field(default_factory=list)
    #: Each line against the cumulative-drift reference (``drift_reference``),
    #: on every real document both scored.
    drift: list[LineComparison] = field(default_factory=list)
    #: The run the drift check holds each line to: the last run from the base
    #: when its report is on this yardstick, else the oldest run of the chain
    #: that is (``drift_note`` says so). ``None`` when there is no run older
    #: than the parent to hold to.
    drift_reference: str | None = None
    #: Whether ``drift_reference`` is the run from the base itself.
    drift_reference_is_fresh: bool = False
    #: Why the drift check holds to another run than the last fresh one, or is
    #: not made, when either.
    drift_note: str | None = None
    #: The references re-scored from their stored answers: ``{run: {difference,
    #: rescored_from}}`` - what put their report on another yardstick, and the
    #: yardstick it was scored on before.
    rescored: dict[str, dict[str, Any]] = field(default_factory=dict)


def _weighted_mean(scores: Sequence[float], weights: Sequence[float] | None) -> float:
    weights = list(weights) if weights else [1.0] * len(scores)
    total = sum(weights)
    return sum(s * w for s, w in zip(scores, weights, strict=True)) / total if total else 0.0


def _require_line_non_regression(
    result: GateResult,
    continued_from: str | None,
    evidence: LineEvidence | None,
    missing: str | None,
) -> None:
    """A run that continued an adapter must not have lost ground on any line.

    Continued training compounds drift (decision 2026-10-23): each cycle starts
    from the last one's weights, so a line can slip a little every cycle. Each
    line the candidate serves is checked twice, each time over the line's real
    documents scored in both reports (decision 2026-10-26):

    * **parent** - against the run it continued (``LineEvidence.compared``), at
      margin :data:`LINE_PARENT_MARGIN`;
    * **drift** - against the last run from the base the chain started at
      (``LineEvidence.drift``), at margin :data:`LINE_DRIFT_MARGIN`: losses each
      within the parent margin add up over cycles, and this blocks once they
      pass it.

    Each check is a one-sided paired non-inferiority test, a bootstrap over the
    line's documents (``evaluation.bootstrap.non_inferiority_test``), and each
    family of lines is corrected together by Holm's step-down at family-wise
    :data:`LINE_FAMILY_ALPHA`. A line with fewer than :data:`LINE_MIN_DOCUMENTS`
    shared documents is not gated: it is warned, with the smallest drop its
    documents could show. Every line's evidence - documents, change, bound,
    adjusted alpha, gated or warned - is a line verdict in the decision.

    A line not shown non-inferior blocks as ``line:<line>``, and so does a line
    the parent's release serves and the candidate does not: its policies would
    move to the base model unannounced.

    No comparable parent scores at all (``missing`` says why: the parent's report
    is gone, or on another frozen set or scoring code) blocks
    ``line_regression_evidence``: drift nobody can measure is not drift that did
    not happen. The remedy is the next cycle from the base (``--fresh``).
    """
    if not continued_from:
        return
    if evidence is None:
        _block(
            result, "line_regression_evidence",
            f"this run continued {continued_from}, so each line must be compared with the runs "
            f"before it, and that cannot be done: {missing or 'no per-line scores of the parent'}. "
            "Train the next cycle from the base (finetune or all with --fresh, without --version), "
            "or keep the parent's report on this frozen set.",
        )
        return
    for line in evidence.dropped:
        _block(
            result, f"line:{line}",
            f"{line} is served by {continued_from}'s release and not by this one, so its policies "
            "would go to the base model. Train on that line's documents again, or waive "
            f"line:{line} if dropping it is intended.",
        )
    if evidence.added:
        result.warnings.append(
            f"this run serves {', '.join(evidence.added)}, which {continued_from}'s release does "
            "not: there is no earlier run to compare them with.")
    if evidence.thin:
        result.warnings.append(
            "not compared with the run it continued, no real document of the line scored in both "
            "reports to pair: "
            + ", ".join(f"{line} ({n})" for line, n in sorted(evidence.thin.items())))
    gated = _line_family(result, evidence.compared, check="parent", margin=LINE_PARENT_MARGIN,
                         fresh=False)
    if not gated and not evidence.dropped:
        warning = (f"this run continued {continued_from}, but no line had {LINE_MIN_DOCUMENTS} or "
                   "more real documents scored in both reports, so no line was gated against the "
                   "run it continued.")
        log.warning("%s", warning)
        result.warnings.append(warning)
    if evidence.drift_note:
        result.warnings.append(evidence.drift_note)
    _line_family(result, evidence.drift, check="drift", margin=LINE_DRIFT_MARGIN,
                 fresh=evidence.drift_reference_is_fresh)


def _line_family(result: GateResult, comparisons: Sequence[LineComparison], *, check: str,
                 margin: float, fresh: bool) -> int:
    """One family of line checks (``parent`` or ``drift``): the lines with
    enough shared documents tested together under Holm's step-down, the rest
    warned. Records a verdict per line and blocks each line not shown
    non-inferior. Returns how many lines were gated."""
    tests = {
        c.line: non_inferiority_test(
            f"{check}:{c.line}", c.reference_scores, c.candidate_scores, margin=margin,
            reference_weights=c.reference_weights, candidate_weights=c.candidate_weights)
        for c in comparisons if len(c.candidate_scores) >= LINE_MIN_DOCUMENTS
    }
    holm = holm_step_down(tests, LINE_FAMILY_ALPHA)
    warned: list[str] = []
    for c in sorted(comparisons, key=lambda c: c.line):
        before = _weighted_mean(c.reference_scores, c.reference_weights)
        after = _weighted_mean(c.candidate_scores, c.candidate_weights)
        documents = len(c.candidate_scores)
        detectable = minimum_detectable_drop(
            [a - b for a, b in zip(c.candidate_scores, c.reference_scores, strict=True)],
            alpha=LINE_FAMILY_ALPHA, power=LINE_DETECTION_POWER)
        verdict: dict[str, Any] = {
            "line": c.line,
            "check": check,
            "held_to": c.reference,
            "held_to_rescored": c.reference_rescored,
            "measure": c.measure,
            "documents": documents,
            "reference": round(before, 6),
            "candidate": round(after, 6),
            "delta": round(after - before, 6),
            "margin": margin,
            "minimum_detectable_drop": None if detectable is None else round(detectable, 6),
        }
        outcome = holm.get(c.line)
        if outcome is None:
            verdict.update(status="warned", non_inferior=None, p_value=None, holm_adjusted_p=None,
                           adjusted_alpha=None, bound=None, holm_rank=None, holm_family=len(tests),
                           holm_stopped_by=None)
            warned.append(f"{c.line} ({documents}"
                          + (f", minimum detectable drop {detectable:.3f}" if detectable is not None
                             else ", too few to estimate a detectable drop")
                          + ")")
        else:
            verdict.update(status="gated", non_inferior=outcome.shown,
                           p_value=round(tests[c.line].p_value, 6),
                           holm_adjusted_p=round(outcome.adjusted_p, 6),
                           adjusted_alpha=round(outcome.adjusted_alpha, 6),
                           bound=round(outcome.bound, 6), holm_rank=outcome.rank,
                           holm_family=outcome.family, holm_stopped_by=outcome.stopped_by)
        result.line_verdicts.append(verdict)
        if outcome is not None and not outcome.shown:
            _block(result, f"line:{c.line}",
                   _line_block_reason(verdict, check=check, fresh=fresh, own=tests[c.line].shown(
                       outcome.adjusted_alpha)))
    if warned:
        against = "the run it continued" if check == "parent" else "the last run from the base"
        result.warnings.append(
            f"not gated against {against} ({check}), fewer than {LINE_MIN_DOCUMENTS} real documents "
            "scored in both reports; a drop smaller than the minimum detectable drop (one-sided "
            f"alpha {LINE_FAMILY_ALPHA}, power {LINE_DETECTION_POWER:.0%}) could go unseen at "
            "that n: " + ", ".join(warned))
    return len(tests)


def _line_block_reason(verdict: dict[str, Any], *, check: str, fresh: bool, own: bool) -> str:
    line, held_to = verdict["line"], verdict["held_to"]
    numbers = (
        f"{verdict.get('measure', 'field_normalized_match')} {verdict['reference']:.4f} -> "
        f"{verdict['candidate']:.4f} over "
        f"{verdict['documents']} real document(s); the one-sided lower bound "
        f"{verdict['bound']:+.4f} at Holm-adjusted alpha {verdict['adjusted_alpha']:.4f} "
        f"(line {verdict['holm_rank']} of {verdict['holm_family']}) is "
        + ("above" if own else "not above") + f" -{verdict['margin']:.3f}"
    )
    if own:
        # Its own bound clears the margin: blocked because Holm's step-down
        # stopped at a line before it.
        numbers += (f", but Holm's step-down stopped at {verdict['holm_stopped_by']} before "
                    "reaching it, so it is not shown non-inferior either")
    if check == "parent":
        return (f"{line} is not non-inferior to {held_to}, the run it continued: {numbers}. A "
                "continued run must not lose ground on a line it already read.")
    reference = ("the last run from the base" if fresh
                 else "the oldest run of the chain on this yardstick")
    return (f"{line} has drifted from {held_to}, {reference}: {numbers}. Losses each within the "
            f"parent margin ({LINE_PARENT_MARGIN}) have added up over the cycles: train the next "
            f"cycle from the base (--fresh), or waive line:{line}.")


def _describe_line_verdict(verdict: dict[str, Any]) -> str:
    """One line verdict as :meth:`GateResult.report` prints it."""
    head = (f"line {verdict.get('line')} ({verdict.get('check', 'parent')}) against "
            f"{verdict.get('held_to')}, {verdict.get('measure', 'field_normalized_match')}: "
            f"{verdict.get('reference', 0.0):.4f} -> "
            f"{verdict.get('candidate', 0.0):.4f} over {verdict.get('documents')} document(s)")
    if verdict.get("status") == "warned":
        detectable = verdict.get("minimum_detectable_drop")
        return (head + f", not gated (fewer than {LINE_MIN_DOCUMENTS})"
                + (f", minimum detectable drop {detectable:.3f}" if detectable is not None else ""))
    return (head + f", one-sided bound {verdict.get('bound', 0.0):+.4f} at alpha "
            f"{verdict.get('adjusted_alpha', 0.0):.4f} against -{verdict.get('margin', 0.0):.3f} "
            + ("✓" if verdict.get("non_inferior") else "✗"))


def apply_to_manifest(result: GateResult, manifest: Any, *, gated_against: str | None = None,
                      metrics: dict[str, Any] | None = None) -> Any:
    """Record the gate decision on the candidate's run manifest (IMPL-02).

    The decision travels with the artifact, so "was this promoted, and against
    what" is answerable later without reconstructing it. With ``metrics``, the
    scores gated are recorded too (the registry index and the model card read
    them), with the tier, its confidence and the lower bound it rests on
    (decision 2026-10-26) - from ``result`` unless ``metrics`` names them.
    """
    manifest.promotion.beat_previous_on_all_gates = result.passed and not result.waived
    manifest.promotion.tier = result.tier
    if metrics is not None:
        from registry_utils.models import EvalMetrics

        evidence = result.tier_evidence
        recorded = {
            "tier": result.tier, "tier_confidence": result.tier_confidence,
            "tier_lower_bound": evidence.lower_bound if evidence is not None else None,
            **metrics,
        }
        known = EvalMetrics.model_fields
        manifest.eval_metrics = EvalMetrics(**{
            name: value for name, value in recorded.items()
            if name in known and (value is None or isinstance(value, (int, float, str, dict)))
            and not isinstance(value, bool)
        })
    manifest.promotion.failed_gates = list(result.failed_gates) + [
        f"{g} (WAIVED)" for g in result.waived
    ]
    if gated_against:
        manifest.promotion.gated_against = gated_against

    # A run that never finished is not "evaluated". `launch_and_record` marks a
    # crashed run failed precisely so the registry never claims weights it never
    # wrote — and under arch v2.1 §13 the gate runs AFTER training, so writing
    # "evaluated" unconditionally resurrected a dead run into a publishable one.
    # The metrics being gated in that case came from somewhere else entirely.
    if getattr(manifest, "status", None) == "failed":
        log.warning(
            "not marking %s evaluated: the run is recorded as failed, so its weights were "
            "never written and whatever was scored did not come from them.",
            getattr(manifest, "run_id", "<unknown>"),
        )
        return manifest

    manifest.status = "evaluated"
    return manifest
