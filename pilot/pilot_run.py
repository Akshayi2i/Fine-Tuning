"""Experiment C — the pilot training run (IMPL-15 §3, arch §16c).

25–30 annotated documents per type, pilot split ratios (~70/18/12), Foundation
plus per-type adapters, evaluated on the held-out pilot test split.

This is the **minimum experiment that tests the architecture's generalisation
claim**. Experiments A and B cannot: A uses no fine-tuning, and B evaluates on
the training set by design.

Two properties of these numbers matter more than the numbers:

* **They are directional.** At three or four test documents per type, no single
  metric is trustworthy alone. The batch's real job is proving the pipeline works
  end to end before annotation scales to 1000+/type.
* **A failed criterion is a diagnosis, not a verdict.** Each one maps to a
  specific place to look, and every one of those places is a corpus or config
  problem before it is an architecture problem.

**Alias generalization is the one criterion no plumbing test can stand in for.**
It is measured by deliberate hold-out: pick one surface label per
confusable-prone field, keep every document using it out of ``train``, and put
them in the test split. Scoring well on them means the model learned the semantic
mapping rather than memorising label strings — which is the entire claim of the
canonical-field design (master §1.4).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PILOT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PILOT_ROOT / "reports"

PILOT_DOCS_PER_TYPE = (25, 30)

#: How close a held-out variant must stay to its field's dominant label, and how
#: close image-only must stay to OCR-plus-image. Both are *relative* criteria:
#: an absolute floor would fail a hard document type and pass an easy one.
RELATIVE_TOLERANCE = 0.15


class PilotError(RuntimeError):
    """Raised when the pilot cannot be evaluated as specified."""


@dataclass(frozen=True)
class Criterion:
    """One pilot success criterion, with the diagnosis path for its failure."""

    name: str
    threshold: float
    direction: str  # "higher_is_better" | "lower_is_better"
    diagnosis: str
    #: Derived criteria compare two metrics rather than reading one.
    derive: Callable[[dict[str, Any]], float | None] | None = None
    #: The document types the criterion can be measured on; ``None`` for all. A
    #: pilot holding none of them records it as not applicable, never as missed.
    applies_to: frozenset[str] | None = None

    def value(self, metrics: dict[str, Any]) -> float | None:
        if self.derive is not None:
            return self.derive(metrics)
        raw = metrics.get(self.name)
        return float(raw) if isinstance(raw, (int, float)) else None

    def met(self, value: float | None) -> bool:
        if value is None:
            return False
        return value > self.threshold if self.direction == "higher_is_better" else value < self.threshold


def _relative_gap(metrics: dict[str, Any], numerator: str, reference: str) -> float | None:
    """Shortfall of one metric against another, as a share of the reference."""
    a, b = metrics.get(numerator), metrics.get(reference)
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not b:
        return None
    return (b - a) / b


PILOT_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "field_f1", 0.80, "higher_is_better",
        "Prompt and schema design first, then corpus coverage for the weak doc type, then a "
        "targeted per-type retrain on LR/epochs — not the deferred sweep, which measures noise at "
        "this volume.",
    ),
    Criterion(
        "list_field_recall", 0.75, "higher_is_better",
        "Loss Run corpus row-length variety, and whether MinerU's table row count is reaching the "
        "row-completeness signal (IMPL-09). Consider list-specific prompt rules.",
    ),
    Criterion(
        "schema_validity_rate", 0.9999, "higher_is_better",
        "Prompt rules — no fences, explicit null handling — and JSON structural discipline in the "
        "Foundation corpus. This one is 100% or it is a defect; anything else means the model is "
        "improvising structure.",
    ),
    Criterion(
        "image_only_gap", RELATIVE_TOLERANCE, "lower_is_better",
        "The ViT escalation gate (IMPL-06 vit_gate) — but ONLY if the errors are perception-type. "
        "If they are schema or reasoning errors, training the vision encoder fixes nothing and "
        "costs a great deal.",
        derive=lambda m: _relative_gap(m, "image_only_field_f1", "ocr_plus_image_field_f1"),
    ),
    Criterion(
        "row_completeness_detection", 0.80, "higher_is_better",
        "Cross-check the IMPL-09 sources: is MinerU's per-page table row count present in "
        "ocr_meta, and is the document-stated count being parsed?",
    ),
    Criterion(
        "lob_detection_accuracy", 0.85, "higher_is_better",
        "LoB corpus coverage (IMPL-05 corpus_manifest) — check the >=20%-per-value target before "
        "blaming the model.",
        # Measured where the model writes the line. A policy takes its line from
        # the request (detection ships off; scripts/measure_lob_detection.py
        # measures it apart), so a policy-only pilot could never meet this.
        applies_to=frozenset({"acord", "lossrun"}),
    ),
    Criterion(
        "alias_generalization_gap", RELATIVE_TOLERANCE, "lower_is_better",
        "alias_coverage in the corpus manifest: the held-out label is probably under-represented. "
        "Also check the field's semantic gloss — a vague description gives the model nothing to "
        "generalise from.",
    ),
    Criterion(
        "confusable_misattribution_rate", 0.05, "lower_is_better",
        "confusable_example_count: near zero means the corpus taught the mapping but never the "
        "boundary. Also check the gloss carries an explicit exclusion clause (IMPL-01).",
    ),
)

CRITERION_BY_NAME = {c.name: c for c in PILOT_CRITERIA}


@dataclass
class CriterionResult:
    criterion: Criterion
    value: float | None

    @property
    def met(self) -> bool:
        return self.criterion.met(self.value)

    @property
    def unmeasured(self) -> bool:
        return self.value is None

    def describe(self) -> str:
        comparator = ">" if self.criterion.direction == "higher_is_better" else "<"
        if self.value is None:
            return f"{self.criterion.name}: NOT MEASURED (target {comparator} {self.criterion.threshold})"
        return (
            f"{self.criterion.name}: {self.value:.4f} "
            f"(target {comparator} {self.criterion.threshold}) "
            f"{'met' if self.met else 'MISSED'}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion.name,
            "value": None if self.value is None else round(self.value, 4),
            "threshold": self.criterion.threshold,
            "direction": self.criterion.direction,
            "met": self.met,
            "diagnosis": None if self.met else self.criterion.diagnosis,
        }


@dataclass
class PilotReport:
    """Experiment C's output."""

    results: list[CriterionResult] = field(default_factory=list)
    #: Criteria the pilot's document types cannot measure (Criterion.applies_to).
    not_applicable: list[str] = field(default_factory=list)
    documents_by_doc_type: dict[str, int] = field(default_factory=dict)
    held_out_labels: dict[str, str] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return all(r.met for r in self.results) and bool(self.results)

    @property
    def failed(self) -> list[CriterionResult]:
        return [r for r in self.results if not r.met]

    @property
    def under_sized_types(self) -> list[str]:
        low = PILOT_DOCS_PER_TYPE[0]
        return sorted(dt for dt, n in self.documents_by_doc_type.items() if n < low)

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": "C_pilot_run",
            "generated_at": self.generated_at,
            "documents_by_doc_type": dict(sorted(self.documents_by_doc_type.items())),
            "under_sized_doc_types": self.under_sized_types,
            "held_out_surface_labels": dict(sorted(self.held_out_labels.items())),
            "run_ids": sorted(self.run_ids),
            "passed": self.passed,
            "criteria": [r.as_dict() for r in self.results],
            "not_applicable": sorted(self.not_applicable),
            "caveat": (
                "Pilot numbers are directional. At 3-4 test documents per type no single metric is "
                "trustworthy alone; re-baseline properly at real scale and do not carry these "
                "thresholds forward as production gates."
            ),
        }


def evaluate_pilot(
    metrics: dict[str, Any],
    *,
    documents_by_doc_type: dict[str, int] | None = None,
    held_out_labels: dict[str, str] | None = None,
    run_ids: Iterable[str] = (),
) -> PilotReport:
    """Score a pilot run against every criterion and name each failure's diagnosis."""
    report = PilotReport(
        documents_by_doc_type=dict(documents_by_doc_type or {}),
        held_out_labels=dict(held_out_labels or {}),
        run_ids=list(run_ids),
    )
    types = set(report.documents_by_doc_type)

    def applies(criterion: Criterion) -> bool:
        # With no document types given, every criterion counts: a pilot that
        # does not say what it holds cannot excuse itself from one.
        return not types or criterion.applies_to is None or bool(types & criterion.applies_to)

    report.results = [CriterionResult(c, c.value(metrics)) for c in PILOT_CRITERIA if applies(c)]
    report.not_applicable = [c.name for c in PILOT_CRITERIA if not applies(c)]

    for result in report.results:
        if result.unmeasured:
            log.warning("%s was not measured — absent is not passing", result.criterion.name)
        elif not result.met:
            log.warning("%s missed: %s", result.criterion.name, result.criterion.diagnosis)
    return report


# --------------------------------------------------------------------------
# Alias generalization — the deliberate hold-out
# --------------------------------------------------------------------------


def hold_out_surface_label(
    provenance_by_source: dict[str, dict[str, str]],
    *,
    field_path: str,
    surface_label: str,
) -> list[str]:
    """Documents that must be kept out of ``train`` to test alias generalization.

    Every document whose ``field_provenance`` says this field was found under
    this surface label. Put them in the test split: if the model scores well on a
    phrasing it never saw in training, it learned the mapping rather than the
    string.

    Returning the ids rather than mutating a split is deliberate — the caller
    feeds them to ``assign_splits`` alongside its own assignment, so this
    function has no opinion about how splitting works.
    """
    target = surface_label.strip().casefold()
    return sorted(
        source_id
        for source_id, provenance in provenance_by_source.items()
        if (provenance.get(field_path) or "").strip().casefold() == target
    )


def alias_generalization_gap(
    alias_report: Any,
    *,
    field_path: str,
    held_out_label: str,
    min_support: int = 1,
) -> float | None:
    """Shortfall of the held-out label against the field's dominant label.

    Args:
        alias_report: a IMPL-08 ``AliasAccuracyReport``.
        held_out_label: the surface label kept out of ``train``.

    Returns:
        The gap as a share of the dominant label's accuracy, or ``None`` when
        there is not enough support to say anything. ``None`` is the honest
        answer for one document; treating it as zero would report a pass that
        rests on a single example.
    """
    counts = getattr(alias_report, "counts", {}).get(field_path, {})
    if not counts:
        return None

    supported = {
        label: (correct, total)
        for label, (correct, total) in counts.items()
        if total >= min_support
    }
    if held_out_label not in supported or len(supported) < 2:
        return None

    dominant_label = max(
        (lbl for lbl in supported if lbl != held_out_label),
        key=lambda lbl: supported[lbl][1],
    )
    dominant_correct, dominant_total = supported[dominant_label]
    held_correct, held_total = supported[held_out_label]

    dominant_accuracy = dominant_correct / dominant_total
    held_accuracy = held_correct / held_total
    if not dominant_accuracy:
        return None
    return (dominant_accuracy - held_accuracy) / dominant_accuracy


def derive_alias_metric(
    alias_report: Any, held_out_labels: dict[str, str], *, min_support: int = 1
) -> float | None:
    """The worst alias-generalization gap across every held-out label.

    Worst rather than mean: one field that memorised its label strings is the
    finding, and averaging it against three that generalised hides it.
    """
    gaps = [
        gap
        for field_path, label in held_out_labels.items()
        if (gap := alias_generalization_gap(
            alias_report, field_path=field_path, held_out_label=label, min_support=min_support
        )) is not None
    ]
    return max(gaps) if gaps else None


def write_report(report: PilotReport, root: Path = REPORTS_DIR) -> Path:
    path = root / "pilot_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def render(report: PilotReport) -> str:
    lines = ["Experiment C — pilot run (25-30 docs/type, held-out test split)"]
    if report.documents_by_doc_type:
        lines.append("  corpus: " + ", ".join(
            f"{dt}={n}" for dt, n in sorted(report.documents_by_doc_type.items())))
    if report.under_sized_types:
        lines.append(
            f"  NOTE: {', '.join(report.under_sized_types)} below the "
            f"{PILOT_DOCS_PER_TYPE[0]}-document floor — those criteria are indicative only"
        )
    lines += [f"    {r.describe()}" for r in report.results]
    if report.failed:
        lines.append("  diagnosis paths:")
        for result in report.failed:
            lines.append(f"    {result.criterion.name}: {result.criterion.diagnosis}")
    lines.append(f"  -> {'PASS' if report.passed else 'FAIL'}")
    lines.append(
        "     A failed criterion signals a specific issue to diagnose — not a reason to "
        "abandon the architecture."
    )
    return "\n".join(lines)


def assert_manifests_written(run_ids: Sequence[str], client: Any) -> None:
    """Every pilot training run is a registry entry, not an untracked experiment.

    A pilot run whose manifest was skipped is a model nobody can reproduce, and
    "it was only a pilot" is how that becomes normal.
    """
    from registry_utils.query_registry import RegistryQueryError, get

    missing = []
    for run_id in run_ids:
        try:
            get(run_id, client)
        except (RegistryQueryError, KeyError, FileNotFoundError):
            missing.append(run_id)
    if missing:
        raise PilotError(
            f"pilot run(s) {missing} produced no run manifest. Pilot runs are registry entries "
            "like any other (IMPL-02); without one, the corpus version, commit and config that "
            "produced the model are unrecoverable."
        )
