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

from data_pipeline.dataset_builder.split_groups import MIN_DOCS_TO_MEASURE_LINE
from evaluation.bootstrap import ConfidenceInterval, paired_bootstrap

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
    # run_to_run_agreement takes DEFAULT_DELTA.
}
DEFAULT_DELTA = 0.015

#: At least one of these must improve, or the release must fix a documented
#: defect. They are the metrics the business actually experiences.
PRIMARY_METRICS: tuple[str, ...] = ("field_normalized_match", "list_field_recall")

#: The gates beside GATING_METRICS that promotion_gate can fail, by the names an
#: override waives them under: the improvement rule, and a continued run's
#: missing per-line comparison with its parent. A continued run's regression on
#: one line is ``line:<line>`` (:func:`is_gate`).
OTHER_GATES: tuple[str, ...] = ("improvement", "line_regression_evidence")


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

    #: A continued run's line-by-line comparison: per line, the run it was held
    #: to, both field matches and the paired interval (decision 2026-10-23).
    line_verdicts: list[dict[str, Any]] = field(default_factory=list)

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
        lines.extend(
            f"  line {v['line']} against {v['held_to']}: {v['reference']:.4f} -> "
            f"{v['candidate']:.4f} (95% CI lower bound {v['interval']['ci_lower']:+.4f})"
            for v in self.line_verdicts
        )
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
        if not verdict.non_inferior:
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
    result.tier = (
        release_tier(candidate_metrics, small_eval_set=small_eval_set) if result.passed else None
    )
    log.info("%s", result.report())
    return result


#: Fideon SPEC_09 §6: a release is PRODUCTION only when its field match reaches this.
#: One that clears every floor (0.85 and the rest) but not this is promoted as
#: INTERIM - usable, and labelled as not yet at the production bar.
PRODUCTION_FIELD_MATCH = 0.92


def release_tier(metrics: dict[str, Any], *, small_eval_set: bool = False) -> str:
    """``production`` or ``interim`` for a release that passed the gate.

    Field match is ``field_normalized_match``, the metric the 0.85 floor is on.
    A release carries one adapter, so its metric is that adapter's. A missing
    value is interim: production has to be shown, not assumed - and so is one
    measured on an eval set accepted below its minimum size.
    """
    if small_eval_set:
        return "interim"
    value = metrics.get("field_normalized_match")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= PRODUCTION_FIELD_MATCH:
        return "production"
    return "interim"


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


#: A continued run's line is compared with the runs before it only on at least
#: this many real documents scored in both reports: the bar the split uses to
#: measure a line at all (``split_groups.MIN_DOCS_TO_MEASURE_LINE``). Documents,
#: not values: one policy read in three modes holds hundreds of values, so a
#: bar on values let a single document decide a line.
LINE_MIN_DOCUMENTS = MIN_DOCS_TO_MEASURE_LINE


@dataclass
class LineComparison:
    """One line's paired per-document field match: the candidate's against the
    run it is held to on that line, the best since the last run from the base.
    Aligned by index, as :func:`evaluation.bootstrap.paired_bootstrap` pairs."""

    line: str
    #: The run held to: the best on this line since the last run from the base.
    reference: str
    reference_scores: list[float]
    candidate_scores: list[float]
    reference_weights: list[float] | None = None
    candidate_weights: list[float] | None = None


@dataclass
class LineEvidence:
    """What a continued run's line-by-line gate rests on (decision 2026-10-23)."""

    compared: list[LineComparison] = field(default_factory=list)
    #: Lines the candidate serves that no earlier run scored on enough shared
    #: real documents, with the most it shared: not compared, and warned.
    thin: dict[str, int] = field(default_factory=dict)
    #: Lines the parent's release serves and the candidate does not.
    dropped: list[str] = field(default_factory=list)
    #: Lines the candidate serves and the parent's release does not.
    added: list[str] = field(default_factory=list)
    #: The runs compared with, parent first, back to the last run from the base.
    chain: list[str] = field(default_factory=list)


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
    line is therefore held to the BEST any run since the last run from the base
    scored on it (``LineComparison.reference``), not only to the parent: a slip
    under δ per cycle still adds up against the best, and blocks once it is
    more than δ.

    The test is the main gate's non-inferiority, per line: a paired bootstrap
    over the line's real documents scored in both reports, whose 95% lower bound
    on (candidate − reference) must sit above −``DEFAULT_DELTA``. A line with
    fewer than :data:`LINE_MIN_DOCUMENTS` such documents is not compared, and is
    named in a warning. A line the parent's release serves and the candidate
    does not blocks as ``line:<line>`` too: its policies would move to the base
    model unannounced.

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
            f"not compared with the runs before, fewer than {LINE_MIN_DOCUMENTS} real documents "
            "scored in both reports: "
            + ", ".join(f"{line} ({n})" for line, n in sorted(evidence.thin.items())))
    if not evidence.compared and not evidence.dropped:
        warning = (f"this run continued {continued_from}, but no line had {LINE_MIN_DOCUMENTS} or "
                   "more real documents scored in both reports, so no line was compared with the "
                   "runs before it.")
        log.warning("%s", warning)
        result.warnings.append(warning)
    for comparison in sorted(evidence.compared, key=lambda c: c.line):
        interval = paired_bootstrap(
            f"line:{comparison.line}", comparison.reference_scores, comparison.candidate_scores,
            current_weights=comparison.reference_weights,
            candidate_weights=comparison.candidate_weights,
        )
        held = interval.non_inferior(DEFAULT_DELTA)
        before = _weighted_mean(comparison.reference_scores, comparison.reference_weights)
        after = _weighted_mean(comparison.candidate_scores, comparison.candidate_weights)
        result.line_verdicts.append({
            "line": comparison.line,
            "held_to": comparison.reference,
            "reference": round(before, 6),
            "candidate": round(after, 6),
            "delta": DEFAULT_DELTA,
            "non_inferior": held,
            "interval": interval.as_dict(),
        })
        if not held:
            _block(
                result, f"line:{comparison.line}",
                f"{comparison.line} is not non-inferior to {comparison.reference}, the best run on it "
                f"since the last run from the base: field match {before:.4f} -> {after:.4f}, 95% CI "
                f"lower bound {interval.lower:+.4f} is not above -{DEFAULT_DELTA:.3f} over "
                f"{interval.documents} document(s). A continued run must not lose ground on a line "
                "it already read.",
            )


def apply_to_manifest(result: GateResult, manifest: Any, *, gated_against: str | None = None,
                      metrics: dict[str, Any] | None = None) -> Any:
    """Record the gate decision on the candidate's run manifest (IMPL-02).

    The decision travels with the artifact, so "was this promoted, and against
    what" is answerable later without reconstructing it. With ``metrics``, the
    scores gated are recorded too (the registry index and the model card read
    them).
    """
    manifest.promotion.beat_previous_on_all_gates = result.passed and not result.waived
    manifest.promotion.tier = result.tier
    if metrics is not None:
        from registry_utils.models import EvalMetrics

        known = EvalMetrics.model_fields
        manifest.eval_metrics = EvalMetrics(**{
            name: value for name, value in metrics.items()
            if name in known and (value is None or isinstance(value, (int, float, dict)))
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
