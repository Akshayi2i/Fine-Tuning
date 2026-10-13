"""Experiment A — the zero-shot baseline (IMPL-15 §1, arch §16a).

Run the **untuned** base model over 10 real documents per type and score it with
the same metric code the promotion gate uses. Costs no annotation beyond ground
truth you need anyway, and runs in week one.

What it is really for is not the headline number. It is the **failure-mode
distribution**: whether the base model's problem is schema adherence, list-row
recall, LoB detection, or OCR arbitration. Those four have different remedies,
and one aggregate F1 hides which one you have.

**A thin wrapper, never a parallel path.** It calls the same
``testing.run_extraction.run_document`` that the extraction command calls, which
calls the serving pipeline::

    python -m data_pipeline.ingestion.import_labeled_pdfs --input pilot/baseline_docs \\
           --doc-type policy --tenant pilot        # then OCR them (run_mineru --tenant pilot)
    python -m orchestration.run extract --model base --tenant pilot --source-ids ...

``--model base`` and ``resolve_model_version("base")`` exist precisely so this
experiment, day-zero pre-annotation (IMPL-04) and production all run one
implementation. A separate one here would measure something subtly different from
what ships.

**On "F1".** Each scalar field has exactly one predicted value and one expected
value, so precision, recall and accuracy coincide; the reported field F1 is the
normalized match rate. List fields have a genuine precision/recall split and are
reported separately.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PILOT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PILOT_ROOT / "reports"

#: Enough documents per type to see a failure pattern rather than one bad scan.
BASELINE_DOCS_PER_TYPE = 10

#: Decision bands (IMPL-15 §1). Above the first, the base model already carries a
#: strong prior; below the second, the problem is likely the framing rather than
#: the absence of fine-tuning, and annotation money spent before checking that is
#: money spent on the wrong thing.
STRONG_PRIOR = 0.70
WEAK_PRIOR = 0.40


@dataclass
class BaselineDecision:
    """What the baseline score means for the next step."""

    band: str
    proceed: bool
    recommendation: str
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "band": self.band,
            "proceed": self.proceed,
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


def classify_baseline(field_f1: float) -> BaselineDecision:
    """Map a zero-shot field F1 onto the IMPL-15 decision table."""
    if field_f1 > STRONG_PRIOR:
        return BaselineDecision(
            band="strong_prior",
            proceed=True,
            recommendation="proceed to the pilot corpus",
            rationale=(
                f"zero-shot field F1 {field_f1:.3f} is above {STRONG_PRIOR}: the base model "
                "already carries a strong prior for these documents, so fine-tuning is expected "
                "to reach production quality rather than having to establish the basics."
            ),
        )
    if field_f1 >= WEAK_PRIOR:
        return BaselineDecision(
            band="moderate_prior",
            proceed=True,
            recommendation="proceed, with targeted corpus coverage for the weak areas",
            rationale=(
                f"zero-shot field F1 {field_f1:.3f} sits between {WEAK_PRIOR} and {STRONG_PRIOR}: "
                "fine-tuning will materially improve this, and the failure-mode distribution below "
                "says where to aim the corpus instead of collecting uniformly."
            ),
        )
    return BaselineDecision(
        band="insufficient_prior",
        proceed=False,
        recommendation="review prompt design, schema complexity and document difficulty first",
        rationale=(
            f"zero-shot field F1 {field_f1:.3f} is below {WEAK_PRIOR}. The base model is likely "
            "insufficient as framed, and annotation is the most expensive way to discover that a "
            "prompt or a schema was the problem. Review those before committing the budget."
        ),
    )


@dataclass
class BaselineReport:
    """Experiment A's output."""

    documents: int = 0
    by_doc_type: dict[str, float] = field(default_factory=dict)
    documents_by_doc_type: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    failure_modes: dict[str, float] = field(default_factory=dict)
    weakest_fields: list[tuple[str, float]] = field(default_factory=list)
    failed_documents: list[tuple[str, str]] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def field_f1(self) -> float:
        """Corpus-level field F1 — the number the decision table reads."""
        if not self.by_doc_type:
            return 0.0
        weighted = sum(
            score * self.documents_by_doc_type.get(doc_type, 0)
            for doc_type, score in self.by_doc_type.items()
        )
        # Divided by every document PRESENTED, not just the scored ones. A
        # document the pipeline refused contributes zero — it was measured and
        # it failed. Averaging over successes alone made 8 failures out of 10
        # read as field F1 1.0, band `strong_prior`, proceed=True, on a base
        # model that returned nothing usable for most of the corpus.
        scored_total = sum(self.documents_by_doc_type.values())
        total = max(scored_total, self.documents)
        return weighted / total if total else 0.0

    @property
    def decision(self) -> BaselineDecision:
        return classify_baseline(self.field_f1)

    @property
    def under_sampled_types(self) -> list[str]:
        """Types with fewer than the protocol's 10 documents.

        Reported rather than enforced: a thin type still tells you something, as
        long as nobody reads its score as if it were the full experiment.
        """
        return sorted(
            doc_type for doc_type, count in self.documents_by_doc_type.items()
            if count < BASELINE_DOCS_PER_TYPE
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment": "A_zero_shot_baseline",
            "model": "base",
            "generated_at": self.generated_at,
            "documents": self.documents,
            "field_f1": round(self.field_f1, 4),
            "field_f1_by_doc_type": {k: round(v, 4) for k, v in sorted(self.by_doc_type.items())},
            "documents_by_doc_type": dict(sorted(self.documents_by_doc_type.items())),
            "metrics": {k: round(v, 4) for k, v in sorted(self.metrics.items())},
            "failure_modes": {k: round(v, 4) for k, v in sorted(self.failure_modes.items())},
            "weakest_fields": [(f, round(s, 4)) for f, s in self.weakest_fields],
            "failed_documents": self.failed_documents,
            "under_sampled_doc_types": self.under_sampled_types,
            "decision": self.decision.as_dict(),
        }


def summarise_baseline(
    scored: Sequence[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> BaselineReport:
    """Aggregate per-document results into Experiment A's report.

    Args:
        scored: ``(doc_type, expected, got, per_document_metrics)`` per document.
            ``per_document_metrics`` is what ``run_document`` returns, so this
            reads the same numbers the gate would.
    """
    from evaluation.metrics.field_accuracy import score_all_list_fields, score_fields
    from training.vit_gate import ErrorRecord, classify_error, summarise_error_mix

    report = BaselineReport(documents=len(scored))
    accuracy_by_type: dict[str, list[float]] = {}
    field_scores: dict[str, list[float]] = {}
    recalls: list[float] = []
    schema_valid: list[bool] = []
    lob_hits: list[bool] = []
    errors: list[ErrorRecord] = []

    for doc_type, expected, got, metrics in scored:
        accuracy = score_fields(expected, got)
        accuracy_by_type.setdefault(doc_type, []).append(accuracy.normalized_match)
        for path, score in accuracy.by_field().items():
            field_scores.setdefault(path, []).append(score)

        for result in accuracy.failures():
            errors.append(ErrorRecord(
                source_id=str(metrics.get("document", "")),
                field_path=result.field_path,
                expected=result.expected,
                got=result.got,
                error_class=classify_error(result.expected, result.got, all_expected=expected),
                modality_mode=str(metrics.get("mode", "ocr_plus_image")),
            ))

        for list_report in score_all_list_fields(expected, got).values():
            recalls.append(list_report.recall)

        schema_valid.append(bool(metrics.get("schema_valid", False)))
        if "line_of_business" in expected:
            # Set equality, not ==: LoB is a list under v2.1 and [a, b] must match
            # [b, a]. Comparing lists directly would score correct extractions as
            # misses whenever the model emitted the lines in another order.
            from common.lob import normalize_lob

            lob_hits.append(
                set(normalize_lob(expected["line_of_business"]))
                == set(normalize_lob(got.get("line_of_business")))
            )

    report.by_doc_type = {
        doc_type: sum(scores) / len(scores) for doc_type, scores in accuracy_by_type.items()
    }
    report.documents_by_doc_type = {
        doc_type: len(scores) for doc_type, scores in accuracy_by_type.items()
    }
    report.metrics = {
        "field_normalized_match": report.field_f1,
        "list_field_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "schema_validity_rate": sum(schema_valid) / len(schema_valid) if schema_valid else 0.0,
        "lob_detection_accuracy": sum(lob_hits) / len(lob_hits) if lob_hits else 0.0,
    }
    report.failure_modes = summarise_error_mix(errors)
    report.weakest_fields = sorted(
        ((path, sum(scores) / len(scores)) for path, scores in field_scores.items()),
        key=lambda row: row[1],
    )[:10]
    return report


def run_baseline(
    documents: Sequence[tuple[Any, dict[str, Any]]],
    model: Any,
    classifier: Any,
    calibration: Any,
    **pipeline_kwargs: Any,
) -> BaselineReport:
    """Run the base model over the baseline set and summarise it.

    Args:
        documents: ``(ExtractionRequest, golden_label)`` pairs.
        model: a ``LoadedModel`` resolved from ``base`` — the untuned model with
            no adapter. Passing anything else makes this measure something other
            than a baseline, which is why it is the caller's explicit choice.
    """
    from testing.run_extraction import run_document

    scored: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    failures: list[tuple[str, str]] = []

    for request, golden in documents:
        try:
            result, metrics = run_document(
                request, model, classifier, calibration, golden=golden, **pipeline_kwargs
            )
        except Exception as exc:  # noqa: BLE001 - a document that fails is a finding, not a crash
            # A base model producing unparseable or schema-invalid output IS the
            # measurement here; dropping it would flatter the baseline.
            failures.append((request.source_id, str(exc)))
            log.warning("baseline failed on %s: %s", request.source_id, exc)
            continue
        scored.append((result.doc_type, golden, result.extraction, metrics))

    report = summarise_baseline(scored)
    report.failed_documents = failures
    report.documents = len(documents)
    if failures:
        # A refused document is a schema-invalid document, so it joins the
        # measured invalid ones rather than replacing them. Overwriting the rate
        # with a crash rate discarded every document that returned successfully
        # but failed validation — which, with strict_schema=False, is most of
        # them, and schema validity is one of the four failure modes Experiment A
        # exists to tell apart.
        scored_count = len(scored)
        measured = report.metrics.get("schema_validity_rate", 0.0)
        valid = measured * scored_count
        total = scored_count + len(failures)
        report.metrics["schema_validity_rate"] = valid / total if total else 0.0
    return report


def write_report(report: BaselineReport, root: Path = REPORTS_DIR) -> Path:
    path = root / "zero_shot_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)
    return path


def render(report: BaselineReport) -> str:
    decision = report.decision
    lines = [
        "Experiment A — zero-shot baseline (untuned base model)",
        f"  documents: {report.documents}",
        f"  field F1:  {report.field_f1:.3f}  [{decision.band}]",
    ]
    lines += [f"    {dt}: {score:.3f} ({report.documents_by_doc_type[dt]} docs)"
              for dt, score in sorted(report.by_doc_type.items())]
    lines.append("  where it fails:")
    lines += [f"    {mode}: {share:.0%}" for mode, share in sorted(
        report.failure_modes.items(), key=lambda r: -r[1])]
    if report.under_sampled_types:
        lines.append(
            f"  NOTE: fewer than {BASELINE_DOCS_PER_TYPE} documents for "
            f"{', '.join(report.under_sampled_types)} — those scores are indicative only"
        )
    lines.append(f"  -> {decision.recommendation}")
    lines.append(f"     {decision.rationale}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Experiment A — zero-shot baseline")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(
        "Run this through the extraction command rather than here. Import and OCR the baseline "
        "documents into their own tenant, then extract them with the untuned model:\n"
        "  python -m data_pipeline.ingestion.import_labeled_pdfs --input pilot/baseline_docs "
        "--doc-type policy --tenant pilot\n"
        "  /workspace/venv-ocr/bin/python -m data_pipeline.ocr.run_mineru --doc-type policy "
        "--all-unprocessed --tenant pilot\n"
        "  python -m orchestration.run extract --model base --tenant pilot --source-ids ...\n"
        "then feed its per-document metrics (testing/metrics/base/) to summarise_baseline()."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
