"""Label noise: values a release reads that the page prints, but the gold lacks.

    python -m orchestration.run label-noise --version v4 [--corpus-version v4] [--limit 200] [--tenant T]

A gold that leaves out a value its document prints teaches the model to leave it
out too - and the gate then scores the model wrong for reading it. A person
finds those by reading the document beside the gold; this finds the documents
worth reading first.

The release's adapter (on the base, as served) reads each TRAIN document of the
corpus in ``ocr_plus_image`` - through the golden eval's own generation path
(``evaluation.golden_eval.evaluate``, which calls ``serving.pipeline.extract``).
Each value it reads that the gold has no value for is looked up on the
document's clean page text (MinerU's, before any noise) with serving's
OCR-tolerant search (``common.grounding.ground_tolerant``). Printed there, it is
a candidate: the gold may have missed it. Values the gold has, and values no
page prints, are not this question.

The candidates become a review list through the labeling queue
(``data_pipeline.labeling.active_learning.build_queue`` / ``write_queue``), at
``golden-labels/<tenant>/_review_queue/label-noise-<version>.json``, with each
document's values and the pages that print them beside it
(``...label-noise-<version>.details.json``). **No gold is changed**: a person
reviews each one, and corrects it through the labeling tool and its review
record, as any gold is corrected.

GPU work, on the pod: the command runs detached there (``orchestration.detach``).
Synthetic twins are not read: a twin's gold is its seed's, read once with the seed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: The one reading mode: page text and image, as production reads a document
#: with text. The text is what a printed value is looked up in.
MODE = "ocr_plus_image"

#: Review flag prefix on each queued field.
FLAG = "label_noise:missing_from_gold"


class LabelNoiseError(RuntimeError):
    """The release or corpus cannot be read for label noise."""


@dataclass(frozen=True)
class NoiseFinding:
    """One value the model read, the page prints, and the gold lacks."""

    path: str
    value: Any
    pages: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "value": self.value, "pages": list(self.pages)}


@dataclass
class NoiseResult:
    """A document's findings, in the shape ``active_learning.build_queue_item`` reads.

    ``overall_confidence`` is not a model confidence: it orders the list, the
    document with the most findings first (``1 / (1 + findings)``).
    """

    source_id: str
    doc_type: str
    findings: list[NoiseFinding] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        return 1.0 / (1 + len(self.findings))

    @property
    def fields(self) -> dict[str, Any]:
        # No confidence: each finding is a field to look at, not a score.
        return {finding.path: {"confidence": None} for finding in self.findings}

    @property
    def review_flags(self) -> list[str]:
        return [f"{FLAG}:{finding.path}" for finding in self.findings]


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return not value
    return False


def missing_from_gold(expected: dict[str, Any] | None, got: dict[str, Any] | None,
                      page_texts: dict[int, str]) -> list[NoiseFinding]:
    """The values in ``got`` printed on ``page_texts`` at a path the gold leaves empty.

    Paths as the scorer flattens them (``field_accuracy.flatten_scalars``): a
    table's rows addressable (``vehicles[2].vin``), a list of scalars one field.
    Each value is looked for as the page prints it - an envelope's ``raw``
    (``common.canonical.printed_view``) - with serving's OCR-tolerant search.
    """
    from common.canonical import printed_view
    from common.grounding import ground_tolerant, tolerant_pages
    from evaluation.metrics.field_accuracy import flatten_scalars

    gold = flatten_scalars(expected or {})
    read = flatten_scalars(printed_view(got or {}))
    pages = tolerant_pages(page_texts)
    findings = []
    for path, value in sorted(read.items()):
        if _empty(value) or not _empty(gold.get(path)):
            continue
        values = value if isinstance(value, list) else [value]
        printed, on = [], set()
        for item in values:
            if _empty(item) or isinstance(item, (dict, list)):
                continue
            grounding = ground_tolerant(item, None, pages)
            if grounding.printed:
                printed.append(item)
                on.update(grounding.pages)
        if printed:
            findings.append(NoiseFinding(path, printed if isinstance(value, list) else printed[0],
                                         tuple(sorted(on))))
    return findings


def train_documents(client: BlobClient, corpus_version: str, tenant_id: str | None = None,
                    *, limit: int | None = None) -> list[Any]:
    """The corpus's TRAIN documents as the golden eval reads a document
    (``golden_eval.GoldenDocument``), each gold as training teaches it
    (``golden_eval.verified_label``). Synthetic twins, and golds with no review
    record (they never trained), are left out."""
    from common.adapter import ADAPTER
    from data_pipeline.labeling.export_golden_labels import load_golden_label, review_problem
    from evaluation.freeze_eval_set import is_scanned
    from evaluation.golden_eval import GoldenDocument, merge_line, verified_label
    from orchestration.pipeline_dag import read_page_texts

    key = paths.corpus_manifest(corpus_version, tenant_id)
    if not client.exists(key):
        raise LabelNoiseError(f"corpus {corpus_version} has no manifest at {key}")
    train = sorted((client.read_json(key).get("source_ids_by_split") or {}).get("train") or [])
    if not train:
        raise LabelNoiseError(f"corpus {corpus_version} lists no train documents")

    found: list[tuple[str, str, dict[str, Any], dict[str, Any], int, dict[str, Any]]] = []
    for source_id in train:
        doc_type = next((dt for dt in ADAPTER.doc_types
                         if client.exists(paths.golden_label(dt, source_id, tenant_id))), None)
        meta_key = paths.ocr_meta(doc_type or "policy", source_id, tenant_id)
        if doc_type is None or not client.exists(meta_key):
            log.warning("label noise: %s has no gold or no OCR record any more; skipped", source_id)
            continue
        label, metadata = load_golden_label(source_id, doc_type, client, tenant_id)
        if metadata.get("synthetic") or review_problem(label, metadata):
            continue
        ocr_meta = client.read_json(meta_key)
        pages = int(ocr_meta.get("page_count") or 0)
        if pages < 1:
            continue
        found.append((source_id, doc_type, label, metadata, pages, ocr_meta))
        if limit and len(found) >= limit:
            break
    texts = read_page_texts(client, [[paths.processed_page(dt, sid, page, "md", tenant_id)
                                      for page in range(1, pages + 1)]
                                     for sid, dt, _label, _meta, pages, _ocr in found])
    documents = []
    for (source_id, doc_type, label, metadata, pages, ocr_meta), page_texts in zip(found, texts, strict=True):
        documents.append(GoldenDocument(
            source_id=source_id,
            doc_type=doc_type,
            golden=verified_label(label, page_texts, doc_type=doc_type, lob=metadata.get("lob")),
            image_keys=[paths.processed_page(doc_type, source_id, page, "png", tenant_id)
                        for page in range(1, pages + 1)],
            page_texts=dict(enumerate(page_texts, start=1)),
            acord_form=metadata.get("acord_form"),
            lob=merge_line(metadata.get("lob")),
            is_scanned=is_scanned(ocr_meta),
        ))
    return documents


def corpus_of(client: BlobClient, version: str) -> str:
    """The corpus ``version``'s run trained on (its run manifest)."""
    from common.adapter import ADAPTER
    from registry_utils.query_registry import RegistryQueryError, get

    try:
        return get(ADAPTER.run_id(version), client).dependencies.corpus_version
    except (RegistryQueryError, KeyError, FileNotFoundError, ValueError) as exc:
        raise LabelNoiseError(f"{ADAPTER.run_id(version)}'s run manifest cannot be read ({exc}); "
                              "name the corpus with --corpus-version") from exc


def details_key(queue_key: str) -> str:
    """The findings beside the queue: ``<queue>.details.json``."""
    return queue_key[: -len(".json")] + ".details.json" if queue_key.endswith(".json") else queue_key + ".details"


def run_label_noise(
    client: BlobClient,
    model: Any,
    *,
    version: str,
    corpus_version: str | None = None,
    tenant_id: str | None = None,
    limit: int | None = None,
    evaluate: Callable[..., list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    """Read the corpus's train documents with ``model`` (release ``version``'s
    adapter on the base) and queue the documents whose gold lacks a printed
    value the model read. Returns a summary; writes the queue and its details.

    ``evaluate`` is ``golden_eval.evaluate`` unless a test passes its own.
    """
    from data_pipeline.labeling.active_learning import build_queue, write_queue

    if evaluate is None:
        from evaluation.golden_eval import evaluate

    corpus_version = corpus_version or corpus_of(client, version)
    documents = train_documents(client, corpus_version, tenant_id, limit=limit)
    if not documents:
        raise LabelNoiseError(f"corpus {corpus_version} has no reviewed, OCR'd train document to read")
    log.info("label noise: %s reads %d train document(s) of corpus %s (%s)",
             version, len(documents), corpus_version, MODE)
    triples = evaluate(documents, model, client, paths.staging_train_images_dir(corpus_version, tenant_id),
                       modes=(MODE,), run_to_run=False)
    by_id = {doc.source_id: doc for doc in documents}
    results, failed = [], []
    for expected, got, metadata in triples:
        source_id = metadata.get("source_id")
        if metadata.get("error"):
            failed.append({"source_id": source_id, "error": str(metadata["error"])[:300]})
            continue
        findings = missing_from_gold(expected, got, by_id[source_id].page_texts)
        if findings:
            results.append(NoiseResult(source_id=source_id, doc_type=metadata.get("doc_type") or "policy",
                                       findings=findings))
    queue = build_queue(results, model_version=f"label-noise-{version}")
    key = write_queue(queue, client, tenant_id)
    details = {
        "version": version,
        "corpus_version": corpus_version,
        "mode": MODE,
        "documents_read": len(documents) - len(failed),
        "documents_failed": failed,
        "documents_flagged": len(results),
        "findings": {r.source_id: [f.as_dict() for f in r.findings] for r in results},
        "note": "candidates only: no gold was changed. A person reviews each against its document.",
    }
    client.write_json(details_key(key), details)
    log.info("label noise: %d of %d document(s) flagged, %d finding(s); queue %s",
             len(results), len(documents), sum(len(r.findings) for r in results), key)
    return {"queue": key, "details": details_key(key), "corpus_version": corpus_version,
            "documents": len(documents), "failed": len(failed), "flagged": len(results),
            "findings": sum(len(r.findings) for r in results)}


def main_from_args(args: Any, client: BlobClient | None = None,
                   load: Callable[[str, BlobClient], Any] | None = None) -> dict[str, Any]:
    """``orchestration.run label-noise``: load release ``args.version`` and run it."""
    from inference_core.model_runner import release_model

    client = client or BlobClient()
    if load is None:
        from inference_core.model_runner import load_model as load
    model = load(args.version, client)
    try:
        return run_label_noise(client, model, version=args.version, corpus_version=args.corpus_version or None,
                               tenant_id=args.tenant, limit=args.limit)
    finally:
        release_model(model)


# --------------------------------------------------------------------------
# The confidence-routed queue of unlabelled documents (active_learning's CLI)
# --------------------------------------------------------------------------


def release_calibration(client: BlobClient, version: str, tenant_id: str | None = None) -> tuple[Any, Any]:
    """``(calibrators, thresholds)`` release ``version`` serves with, as serving loads
    them (``vllm_entrypoint.load_release_calibration``); ``(None, None)`` with no
    release bundle or none fitted - every field is then flagged for review."""
    key = paths.release_bundle(version, tenant_id)
    if not client.exists(key):
        log.warning("%s has no release bundle at %s: its drafts are routed without calibrators", version, key)
        return None, None
    from serving.release_router import _as_release
    from serving.vllm_entrypoint import load_release_calibration

    runtime = load_release_calibration(_as_release(client.read_json(key), "bf16"), client)
    return runtime.calibrators, runtime.thresholds


def unlabelled_documents(client: BlobClient, doc_type: str, tenant_id: str | None = None) -> list[str]:
    """Documents OCR'd for ``doc_type`` with no gold yet: the review backlog."""
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    labelled = set(list_labeled_source_ids(client, doc_type, tenant_id))
    processed = {key.split("/")[3] for key in client.list(f"processed/{paths._tenant(tenant_id)}/{doc_type}/")
                 if key.endswith("ocr_meta.json")}
    return sorted(processed - labelled)


def backlog_results(client: BlobClient, model: Any, *, version: str, doc_types: Sequence[str],
                    tenant_id: str | None = None, limit: int | None = None) -> list[Any]:
    """Each unlabelled document extracted by release ``version`` through the
    serving pipeline, with the release's calibrators - the ``ExtractionResult``s
    ``active_learning.build_queue`` routes. A document that fails is logged and
    left out: it goes to a person through the labeling tool as before."""
    from calibration.fit_calibration import CalibrationParams
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import extract
    from testing.run_extraction import document_request

    calibrators, thresholds = release_calibration(client, version, tenant_id)
    images_root = paths.staging_train_images_dir("review-backlog", tenant_id)
    results: list[Any] = []
    for doc_type in doc_types:
        ids = unlabelled_documents(client, doc_type, tenant_id)
        for source_id in ids[:limit] if limit else ids:
            try:
                from pathlib import Path

                request, _gold = document_request(client, source_id, doc_type=doc_type, tenant_id=tenant_id,
                                                  images_root=Path(images_root))
                results.append(extract(
                    request, model, StaticClassifier(doc_type, request.known_acord_form),
                    CalibrationParams(method="temperature", doc_type=doc_type, model_version=version,
                                      temperature=1.0),
                    calibrators=calibrators, thresholds=thresholds, strict_schema=False))
            except Exception as exc:  # noqa: BLE001 - one document must not end the queue
                log.error("review queue: %s not extracted (%s: %s)", source_id, type(exc).__name__,
                          str(exc)[:200])
    return results


__all__: Sequence[str] = ("missing_from_gold", "run_label_noise", "train_documents", "NoiseFinding",
                          "NoiseResult", "LabelNoiseError", "backlog_results", "unlabelled_documents")
