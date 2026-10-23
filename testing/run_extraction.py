"""The extraction/testing harness (IMPL-12, arch §17).

Runs any model version over real PDFs and produces JSON, per-field confidence,
and quality metrics — organised by model version so results are never ambiguous
about their origin.

**It is a thin wrapper over** :func:`serving.pipeline.extract`. Steps 2–7 of the
routine (classify, route, generate, confidence, validate) are the serving
pipeline's, called rather than reproduced. Reimplementing them here would give
two extraction paths and make every number this harness produces a description of
a system that is not the one in production.

``--model`` accepts ``base`` as a first-class value: the untuned model with no
adapter. That is the same path used by the pilot's zero-shot baseline (IMPL-15)
and day-zero pre-annotation (IMPL-04), so all three share one implementation.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from calibration.fit_calibration import CalibrationParams
from serving.pipeline import NO_FULL_TEXT_FLAG, ExtractionRequest, ExtractionResult, extract

log = logging.getLogger(__name__)

TESTING_ROOT = Path(__file__).resolve().parent

#: A document whose served JSON holds less than this share of its page text's
#: words is listed in the summary. The section builder places every line it is
#: given, so a shortfall means text was lost between the page and the JSON.
#: Chosen, not measured: revise on the first real run.
FULL_TEXT_COVERAGE = 0.99


class HarnessError(RuntimeError):
    """Raised when the harness cannot run a document."""


@dataclass
class RunSummary:
    """Batch aggregates — the file compared across model versions.

    This is the testing-time mirror of the IMPL-08 promotion gate: it answers
    "did v3 beat v2?" using the same metrics the gate reads.
    """

    model_version: str
    documents: int = 0
    schema_valid: int = 0
    mean_confidence: float = 0.0
    mean_latency_ms: float = 0.0
    by_doc_type: dict[str, int] = field(default_factory=dict)
    by_mode: dict[str, int] = field(default_factory=dict)
    needing_review: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)
    #: Extracted but not scored, and why: a gold with no review record.
    not_scored: list[tuple[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def schema_validity_rate(self) -> float:
        return self.schema_valid / self.documents if self.documents else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "documents": self.documents,
            "schema_validity_rate": round(self.schema_validity_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "documents_needing_review": self.needing_review,
            "by_doc_type": self.by_doc_type,
            "by_mode": self.by_mode,
            "failed": self.failed,
            "not_scored": self.not_scored,
            **self.metrics,
        }


def results_dir(model_version: str, root: Path = TESTING_ROOT) -> Path:
    """``results/{version}/`` — the version is visible in the path itself."""
    return root / "results" / model_version


def metrics_dir(model_version: str, root: Path = TESTING_ROOT) -> Path:
    return root / "metrics" / model_version


def score_against_ground_truth(
    result: ExtractionResult, golden: dict[str, Any], *, lob: Any = None,
    acord_form: str | None = None,
) -> dict[str, Any]:
    """Per-document metrics, using the IMPL-08 modules.

    The same metric code the promotion gate uses, so "correct" means the same
    thing here as it does there.
    """
    from common.canonical import schema_label, without_system_fields
    from evaluation.metrics.field_accuracy import score_all_list_fields, score_fields

    # The system-supplied fields are filled by serving, not extracted: not scored.
    golden, extraction = without_system_fields(golden), without_system_fields(result.extraction)
    # Scored on what the line's schema can hold, as run_eval scores it.
    golden = schema_label(golden, result.doc_type, acord_form, lob)
    common_model = _is_common_model(result.doc_type, acord_form, lob)
    if common_model:
        # Both sides, as run_eval.build_report narrows them: what serving adds
        # after the merge is not a value the model read.
        extraction = schema_label(extraction, result.doc_type, acord_form, lob)
    # A common-model line's ids are each writer's own numbering: compared by
    # what its links name, never by id (common.structural_ids.comparable_view).
    from common.structural_ids import comparable_for

    golden = comparable_for(golden, result.doc_type, acord_form, lob)
    extraction = comparable_for(extraction, result.doc_type, acord_form, lob)
    model_metrics: dict[str, Any] = {}
    if common_model:
        # Scored as run_eval.score_subset scores the line: nearly every value of
        # it lives in a table (a limit inside its coverage), and skipping table
        # paths scored a policy with every premium and limit wrong as perfect.
        # Links, codes and overflow have their own numbers, as there.
        from common.lob import merge_line
        from evaluation.metrics.common_model import CommonModelTally, core_for_scoring, without_overflow

        line = merge_line(lob)
        accuracy = score_fields(*core_for_scoring(golden, extraction, line), skip_lists=False)
        list_reports = score_all_list_fields(
            without_overflow(golden), without_overflow(extraction), common_model=True)
        tally = CommonModelTally()
        tally.add(golden, extraction, line)
        model_metrics = {k: v for k, v in tally.metrics().items() if v is not None}
    else:
        accuracy = score_fields(golden, extraction)
        list_reports = score_all_list_fields(golden, extraction)

    return {
        **model_metrics,
        "field_exact_match_rate": round(accuracy.exact_match, 4),
        "field_normalized_match_rate": round(accuracy.normalized_match, 4),
        "field_accuracy_by_field": accuracy.by_field(),
        "list_field_f1": {n: round(r.f1, 4) for n, r in list_reports.items()},
        "list_field_recall": {n: round(r.recall, 4) for n, r in list_reports.items()},
        "missed_rows": {n: r.missed_rows for n, r in list_reports.items()},
        "failures": [
            {"field": f.field_path, "expected": f.expected, "got": f.got}
            for f in accuracy.failures()
        ],
    }


def _is_common_model(doc_type: str, acord_form: str | None, lob: Any) -> bool:
    """Whether this is a common-model policy, the test ``comparable_for`` makes."""
    from common.schemas import SchemaError, is_common_model

    try:
        return doc_type == "policy" and is_common_model(doc_type, acord_form, lob)
    except SchemaError:
        return False


def run_document(
    request: ExtractionRequest,
    model: Any,
    classifier: Any,
    calibration: CalibrationParams,
    *,
    golden: dict[str, Any] | None = None,
    **pipeline_kwargs: Any,
) -> tuple[ExtractionResult, dict[str, Any]]:
    """Extract one document through the serving pipeline, then score it.

    ``golden`` is optional. **Confidence is emitted either way** — that is the
    payoff of the calibration work: on a document nobody has labeled, confidence
    is still available to route low-confidence fields to review, which is exactly
    the production case (arch §17).
    """
    result = extract(request, model, classifier, calibration, **pipeline_kwargs)

    metrics: dict[str, Any] = {
        "document": request.source_id,
        "model_version": result.model_version,
        "doc_type": result.doc_type,
        "mode": result.mode,
        "schema_valid": result.schema_valid,
        "overall_confidence": result.overall_confidence,
        "pages_used": result.pages_used,
        "review_flags": result.review_flags,
        # SPEC_21's every-word rule on the served JSON (common.full_text).
        "text_coverage": result.text_coverage,
        "fields": {
            path: {**value, "correct": None}
            for path, value in result.fields.items()
        },
    }

    if golden is not None:
        scored = score_against_ground_truth(
            result, golden, lob=request.known_lob, acord_form=request.known_acord_form)
        metrics.update(scored)
        from common.canonical import schema_label, values_view
        from evaluation.metrics.field_accuracy import flatten_scalars, values_agree

        expected_flat = flatten_scalars(values_view(
            schema_label(golden, result.doc_type, request.known_acord_form, request.known_lob)))
        got_flat = flatten_scalars(values_view(result.extraction))
        for path, value in metrics["fields"].items():
            value["correct"] = values_agree(expected_flat.get(path), got_flat.get(path), path)
    else:
        metrics["ground_truth"] = "not supplied — confidence reported, accuracy not measurable"
    metrics["all_fields"] = all_fields(result, golden, acord_form=request.known_acord_form, lob=request.known_lob)

    return result, metrics


def all_fields(result: ExtractionResult, golden: dict[str, Any] | None, *, acord_form: str | None = None,
               lob: Any = None) -> dict[str, Any] | None:
    """Every field of the document's schema with what the answer holds for it,
    graded against ``golden`` when there is one (testing.comparison.field_report).

    ``None`` when the report cannot be built: it describes the extraction, and
    is never a reason to lose it.
    """
    from testing.comparison import field_report

    try:
        return field_report(result.extraction, golden, doc_type=result.doc_type, acord_form=acord_form,
                            lob=lob, review_flags=result.review_flags)
    except Exception as exc:  # noqa: BLE001 - a report on the answer, never a reason to drop it
        log.warning("%s: the all-fields report could not be built (%s: %s)",
                    result.source_id, type(exc).__name__, exc)
        return None


def write_outputs(
    result: ExtractionResult,
    metrics: dict[str, Any],
    *,
    root: Path = TESTING_ROOT,
) -> tuple[Path, Path]:
    """Write ``results/{version}/{doc}.json`` and its metrics file.

    The results JSON carries ``all_fields`` after ``list_fields`` when the
    metrics hold it (:func:`all_fields`): every field of the schema, each with
    its value and result, in the file a reader opens.
    """
    version = result.model_version
    results_path = results_dir(version, root) / f"{result.source_id}.json"
    metrics_path = metrics_dir(version, root) / f"{result.source_id}.metrics.json"

    report = metrics.get("all_fields")
    answer: dict[str, Any] = {}
    for key, value in result.as_dict().items():
        answer[key] = value
        if key == "list_fields" and report is not None:
            answer["all_fields"] = report
    rest = {key: value for key, value in metrics.items() if key != "all_fields"}
    for path, payload in ((results_path, answer), (metrics_path, rest)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return results_path, metrics_path


def append_registry(
    result: ExtractionResult,
    results_path: Path,
    metrics_path: Path,
    *,
    root: Path = TESTING_ROOT,
    quant_format: str | None = None,
) -> None:
    """Record provenance so an output JSON is never ambiguous about its origin.

    ``model_version`` here is the same tag that resolves to a ``run_manifest``,
    so a result traces back to the corpus version and git commit that produced
    the model that generated it (arch §17).
    """
    registry_path = root / "extraction_registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists() else {"extractions": []}
    )
    registry["extractions"].append({
        "document": result.source_id,
        "doc_type": result.doc_type,
        "model_version": result.model_version,
        "quant_format": quant_format,
        "mode": result.mode,
        "result_path": str(results_path.relative_to(root)),
        "metrics_path": str(metrics_path.relative_to(root)),
        "overall_confidence": result.overall_confidence,
        "schema_valid": result.schema_valid,
        "review_flags": result.review_flags,
        "extracted_at": datetime.now(UTC).isoformat(),
    })
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def summarise(results: list[tuple[ExtractionResult, dict[str, Any]]], model_version: str) -> RunSummary:
    """Aggregate a batch into the cross-version comparison file."""
    summary = RunSummary(model_version=model_version, documents=len(results))
    if not results:
        return summary

    confidences, latencies = [], []
    for result, _metrics in results:
        summary.schema_valid += int(result.schema_valid)
        summary.by_doc_type[result.doc_type] = summary.by_doc_type.get(result.doc_type, 0) + 1
        summary.by_mode[result.mode] = summary.by_mode.get(result.mode, 0) + 1
        summary.needing_review += int(result.needs_review)
        confidences.append(result.overall_confidence)
        if result.latency_ms is not None:
            latencies.append(result.latency_ms)

    summary.mean_confidence = sum(confidences) / len(confidences)
    summary.mean_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    # The every-word rule (common.full_text): how much of the page text reached
    # the JSON, and which documents fall short of it - or had no text to measure.
    coverages = {r.source_id: r.text_coverage for r, _m in results if r.text_coverage is not None}
    if coverages:
        summary.metrics["mean_text_coverage"] = round(sum(coverages.values()) / len(coverages), 4)
        summary.metrics["min_text_coverage"] = min(coverages.values())
        summary.metrics["documents_below_full_text"] = sorted(
            source for source, value in coverages.items() if value < FULL_TEXT_COVERAGE)
    without_text = sorted(r.source_id for r, _m in results if NO_FULL_TEXT_FLAG in r.review_flags)
    if without_text:
        summary.metrics["documents_without_full_text"] = without_text

    scored = [m for _r, m in results if "field_normalized_match_rate" in m]
    if scored:
        summary.metrics["mean_field_normalized_match"] = round(
            sum(m["field_normalized_match_rate"] for m in scored) / len(scored), 4
        )
        recalls = [
            value
            for m in scored
            for value in (m.get("list_field_recall") or {}).values()
        ]
        if recalls:
            summary.metrics["mean_list_field_recall"] = round(sum(recalls) / len(recalls), 4)
    return summary


#: Page images fetched for extraction, kept between runs (git-ignored).
PAGE_CACHE = TESTING_ROOT / "ocr_cache"


def document_request(
    client: Any, source_id: str, *, doc_type: str = "policy", tenant_id: str | None = None,
    mode: str = "ocr_plus_image", images_root: Path = PAGE_CACHE,
    unscored: dict[str, str] | None = None,
) -> tuple[ExtractionRequest, dict[str, Any] | None]:
    """One imported, OCR'd document as an extraction request, and its gold label.

    Read from the store the pipeline itself writes - page images and MinerU text
    under ``processed/``, the label under ``golden-labels/`` - so the document is
    read exactly as serving reads one. The gold label is ``None`` when the
    document has none: the extraction still runs, and is not scored.

    A form's edition left inside its number moves to ``edition_date`` where the
    label rules hold (``label_rules.label_rules_apply``), as the gate's golden
    eval reads it: a gold imported before that rule is not rewritten in Blob,
    and scored as stored it would count the model's ``edition_date`` wrong for
    following the convention.

    A gold with no review record (``export_golden_labels.review_problem``,
    decision 2026-10-22) is not scored: the extraction still runs, and the gold
    comes back ``None`` with a warning saying why - recorded in ``unscored``
    (source id -> why) when the caller passes one.
    """
    from artifact_registry import paths
    from common.lob import merge_line
    from data_pipeline.ingestion.label_rules import under_edition_rule
    from data_pipeline.labeling.export_golden_labels import review_problem
    from training.stage_data import localize_keys

    meta_key = paths.ocr_meta(doc_type, source_id, tenant_id)
    if not client.exists(meta_key):
        raise HarnessError(f"{source_id} has not been OCR'd (no {meta_key}). Import and OCR it first.")
    pages = int(client.read_json(meta_key).get("page_count") or 0)
    if pages < 1:
        raise HarnessError(f"{source_id}: its OCR record lists no pages")
    image_keys = [paths.processed_page(doc_type, source_id, page, "png", tenant_id)
                  for page in range(1, pages + 1)]
    local = localize_keys(client, image_keys, images_root)
    texts: dict[int, str] = {}
    if mode != "image_only":
        for page in range(1, pages + 1):
            key = paths.processed_page(doc_type, source_id, page, "md", tenant_id)
            texts[page] = client.read_text(key) if client.exists(key) else ""

    metadata: dict[str, Any] = {}
    label_meta_key = paths.label_metadata(doc_type, source_id, tenant_id)
    if client.exists(label_meta_key):
        metadata = client.read_json(label_meta_key)
    golden_key = paths.golden_label(doc_type, source_id, tenant_id)
    golden = client.read_json(golden_key) if client.exists(golden_key) else None
    if golden is not None and (problem := review_problem(golden, metadata)):
        log.warning("%s is extracted but NOT scored - its gold has %s", source_id, problem)
        if unscored is not None:
            unscored[source_id] = f"its gold has {problem}"
        golden = None
    if golden is not None:
        golden = under_edition_rule(golden, doc_type, metadata.get("lob"))

    request = ExtractionRequest(
        source_id=source_id,
        image_paths=[local[key] for key in image_keys],
        page_texts=texts,
        modality_mode=mode,
        known_doc_type=doc_type,
        known_acord_form=metadata.get("acord_form"),
        known_lob=merge_line(metadata.get("lob")),
    )
    return request, golden


def run_batch(
    model: Any,
    client: Any,
    source_ids: list[str],
    *,
    doc_type: str = "policy",
    tenant_id: str | None = None,
    mode: str = "ocr_plus_image",
    images_root: Path = PAGE_CACHE,
    root: Path = TESTING_ROOT,
    score: bool = True,
    quant_format: str | None = None,
) -> RunSummary:
    """Extract each document through the serving pipeline; write its JSON, metrics and a summary.

    One document failing is recorded in the summary and the batch carries on.
    No calibrators are loaded here, so every field comes back flagged for review:
    this run measures extraction, not the release's review thresholds.
    """
    from serving.doc_type_classifier import StaticClassifier

    results: list[tuple[ExtractionResult, dict[str, Any]]] = []
    failed: list[tuple[str, str]] = []
    # Documents whose gold has no review record: extracted, never scored.
    unscored: dict[str, str] = {}
    for index, source_id in enumerate(source_ids, start=1):
        try:
            request, golden = document_request(client, source_id, doc_type=doc_type, tenant_id=tenant_id,
                                               mode=mode, images_root=images_root, unscored=unscored)
            calibration = CalibrationParams(method="temperature", doc_type=doc_type,
                                            model_version=model.tag, temperature=1.0)
            result, metrics = run_document(
                request, model, StaticClassifier(doc_type, request.known_acord_form), calibration,
                golden=golden if score else None, strict_schema=False,
            )
        except Exception as exc:  # noqa: BLE001 - one document must not end the batch
            log.error("%s failed: %s: %s", source_id, type(exc).__name__, exc)
            failed.append((source_id, f"{type(exc).__name__}: {exc}"[:300]))
            continue
        if source_id in unscored:
            metrics["ground_truth"] = f"not scored: {unscored[source_id]}"
        results_path, metrics_path = write_outputs(result, metrics, root=root)
        # Each output traced to the model, format and mode that wrote it.
        append_registry(result, results_path, metrics_path, root=root, quant_format=quant_format)
        results.append((result, metrics))
        log.info("[%d/%d] %s -> %s%s", index, len(source_ids), source_id, results_path,
                 f" (field match {metrics['field_normalized_match_rate']:.1%})"
                 if "field_normalized_match_rate" in metrics else "")

    summary = summarise(results, model.tag)
    summary.failed = failed
    summary.not_scored = sorted(unscored.items())
    summary.documents = len(results) + len(failed)
    target = metrics_dir(model.tag, root) / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary.as_dict(), indent=2), encoding="utf-8")
    log.info("summary -> %s", target)
    return summary


def select_source_ids(client: Any, args: argparse.Namespace) -> list[str]:
    """The documents named on the command line: explicit ids, a corpus split, or every label."""
    from artifact_registry import paths
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    if args.source_ids:
        ids = list(args.source_ids)
    elif args.split:
        manifest = client.read_json(paths.corpus_manifest(args.corpus, args.tenant))
        by_split = manifest.get("source_ids_by_split") or {}
        ids = sorted(by_split.get(args.split) or [])
        if not ids:
            raise HarnessError(f"corpus {args.corpus} lists no {args.split} documents")
    else:
        ids = list_labeled_source_ids(client, args.doc_type, args.tenant)
    return ids[: args.limit] if args.limit else ids


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Extract with a chosen model version")
    parser.add_argument("--model", required=True,
                        help="base | v1 | v2 ... - 'base' is the untuned model, no adapter")
    parser.add_argument("--adapter", default=None,
                        help="a LoRA adapter folder (checkpoint-N, or the run's adapter folder; local "
                             "or a Blob prefix) applied to the BASE model on every request, NOT merged. "
                             "Use with --model base")
    parser.add_argument("--label", default=None, help="name to file results under (default: from the adapter)")
    picked = parser.add_mutually_exclusive_group()
    picked.add_argument("--source-ids", nargs="+", default=None, help="imported, OCR'd documents to extract")
    picked.add_argument("--split", choices=["train", "val", "test"], default=None,
                        help="every document of this split of --corpus")
    parser.add_argument("--corpus", default=None, help="the corpus whose split --split reads")
    parser.add_argument("--doc-type", dest="doc_type", default="policy")
    parser.add_argument("--mode", choices=["ocr_plus_image", "image_only"], default="ocr_plus_image")
    parser.add_argument("--no-score", dest="score", action="store_false",
                        help="extract without comparing to the gold labels")
    parser.add_argument("--format", dest="quant_format", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tenant", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.split and not args.corpus:
        parser.error("--split needs --corpus")
    if args.adapter and args.model != "base":
        parser.error("--adapter applies a LoRA to the base model: use --model base with it")

    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("testing.run_extraction", argv, hint="extract"):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from artifact_registry.blob_client import BlobClient
    from inference_core.model_runner import load_model, release_model

    client = BlobClient()
    source_ids = select_source_ids(client, args)
    if not source_ids:
        print("no documents to extract")
        return 1
    model = load_model(args.model, client, quant_format=args.quant_format,
                       adapter=args.adapter, label=args.label)
    try:
        summary = run_batch(model, client, source_ids, doc_type=args.doc_type, tenant_id=args.tenant,
                            mode=args.mode, score=args.score, quant_format=args.quant_format)
    finally:
        release_model(model)
    print(json.dumps(summary.as_dict(), indent=2))
    return 1 if summary.failed and len(summary.failed) == summary.documents else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
