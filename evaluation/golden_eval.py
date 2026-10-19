"""Evaluate a model on the frozen golden eval set, through the serving pipeline.

The promotion gate's numbers come from here. Every document goes through
:func:`serving.pipeline.extract` — never a bespoke inference path — so the gate
measures what production serves: the same routing, the same policy windows and
merge, the same date post-process, the same calibrated confidence. Validation
scoring reads the corpus rows' own prompts, which is right for checkpoint
selection and wrong for a gate: the serving-only steps would be invisible to it.

Layout of the frozen set in Blob (``paths.golden_eval_set_dir(tenant_id)``), one
set per tenant and one directory per document::

    golden-eval-set/{tenant}/manifest.json                  what was frozen, from which corpus
    golden-eval-set/{tenant}/{source_id}/golden.json        the golden label
    golden-eval-set/{tenant}/{source_id}/metadata.json      doc_type, acord_form, lob, is_scanned
    golden-eval-set/{tenant}/{source_id}/page_N.png         page images, 1-based
    golden-eval-set/{tenant}/{source_id}/page_N.md          OCR text per page (optional)

Per tenant because each tenant's releases are gated on its own documents, and
because source ids are numbered per tenant: one shared set would gate every
tenant on whichever froze first, and read another tenant's ``policy_0001`` as
this one's.

Each document is run in all three input modes. ``noisy_ocr_image`` uses the
corpus build's own seeded corruption, and renders the same prompt serving sends
for clean OCR: the noise lives in the data, never in the instruction.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: The modes every golden document is evaluated in.
GOLDEN_MODES: tuple[str, ...] = ("ocr_plus_image", "noisy_ocr_image", "image_only")

_PAGE = re.compile(r"page_(\d+)\.(png|md)$")


class GoldenEvalError(RuntimeError):
    """Raised when the frozen eval set cannot be read or evaluated."""


@dataclass
class GoldenDocument:
    """One frozen eval document: its label, its pages, and what selects its schema."""

    source_id: str
    doc_type: str
    golden: dict[str, Any]
    image_keys: list[str]
    page_texts: dict[int, str] = field(default_factory=dict)
    acord_form: str | None = None
    lob: str | list[str] | None = None
    is_scanned: bool = False
    synthetic: bool = False
    #: The document's carrier was held out of train and val (Fideon SPEC_09
    #: amendment item 4); evaluation reports these documents apart.
    held_out_carrier: bool = False


def load_golden_set(
    client: BlobClient, doc_types: Sequence[str] | None = None, *, tenant_id: str | None = None
) -> list[GoldenDocument]:
    """Every document in this tenant's frozen set, optionally narrowed to ``doc_types``."""
    from evaluation.run_eval import eval_set_keys

    prefix = paths.golden_eval_set_dir(tenant_id)
    by_source: dict[str, list[str]] = {}
    for key in eval_set_keys(client, tenant_id):
        rest = key[len(prefix):].strip("/")
        if "/" in rest:
            by_source.setdefault(rest.split("/", 1)[0], []).append(key)

    documents = []
    for source_id, keys in sorted(by_source.items()):
        base = f"{prefix}/{source_id}"
        if f"{base}/golden.json" not in keys:
            continue
        metadata = (
            client.read_json(f"{base}/metadata.json") if f"{base}/metadata.json" in keys else {}
        )
        doc_type = metadata.get("doc_type")
        if not doc_type:
            raise GoldenEvalError(
                f"{source_id} has no doc_type in metadata.json, so no schema can be selected"
            )
        if doc_types and doc_type not in doc_types:
            continue
        images: dict[int, str] = {}
        texts: dict[int, str] = {}
        for key in keys:
            match = _PAGE.search(key)
            if not match:
                continue
            page = int(match.group(1))
            if match.group(2) == "png":
                images[page] = key
            else:
                texts[page] = client.read_text(key)
        if not images:
            raise GoldenEvalError(f"{source_id} has no page images")
        if sorted(images) != list(range(1, len(images) + 1)):
            raise GoldenEvalError(f"{source_id}: page images are not numbered 1..n")
        documents.append(GoldenDocument(
            source_id=source_id,
            doc_type=doc_type,
            # A rule-added value the page does not print is not expected of
            # the model either (label_verification), as in its training targets.
            golden=verified_label(
                client.read_json(f"{base}/golden.json"), [texts.get(p, "") for p in sorted(images)]
            ),
            image_keys=[images[p] for p in sorted(images)],
            page_texts={p: texts.get(p, "") for p in sorted(images)},
            acord_form=metadata.get("acord_form"),
            lob=merge_line(metadata.get("lob")),
            is_scanned=bool(metadata.get("is_scanned", False)),
            synthetic=bool(metadata.get("synthetic", False)),
            held_out_carrier=bool(metadata.get("held_out_carrier", False)),
        ))
    return documents


def verified_label(label: dict[str, Any], page_texts: list[str]) -> dict[str, Any]:
    """The frozen gold as training would teach it: rule-added values the page
    does not print left out (label_verification), and a form's edition moved out
    of its number, as the label rules now do. A set frozen before that rule is
    not rewritten in Blob; it is read under it."""
    from data_pipeline.dataset_builder.label_verification import verified_label as verify
    from data_pipeline.ingestion.label_rules import split_form_editions

    verified = copy.deepcopy(verify(label, page_texts))
    split_form_editions(verified)
    return verified


def merge_line(lob: Any) -> Any:
    """Classic auto is read as personal auto (common.lob.MERGED_LINES)."""
    from common.lob import merge_line as merged

    return merged(lob)


def evaluate(
    documents: Sequence[GoldenDocument],
    model: Any,
    client: BlobClient,
    images_root: str | Path,
    *,
    calibrators: Any = None,
    thresholds: Any = None,
    modes: Sequence[str] = GOLDEN_MODES,
    seed: int = 42,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """``(expected, got, metadata)`` triples for :func:`run_eval.build_report`.

    A document that fails to extract is scored as an empty answer, never
    dropped: dropping it would score the model on the documents it managed.
    """
    from calibration.fit_calibration import CalibrationParams
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from training.stage_data import localize_keys

    local = localize_keys(
        client, sorted({key for doc in documents for key in doc.image_keys}), images_root
    )
    triples = []
    for doc in documents:
        calibration = CalibrationParams(
            method="temperature", doc_type=doc.doc_type,
            model_version=getattr(model, "tag", "eval"), temperature=1.0,
        )
        for mode in modes:
            texts = dict(doc.page_texts)
            if mode == "noisy_ocr_image":
                corrupted, _details = corrupt_ocr_pages(
                    [texts[p] for p in sorted(texts)], doc.source_id, seed=seed
                )
                texts = dict(zip(sorted(texts), corrupted, strict=True))
            request = ExtractionRequest(
                source_id=doc.source_id,
                image_paths=[local[key] for key in doc.image_keys],
                page_texts={} if mode == "image_only" else texts,
                modality_mode=mode,
                known_doc_type=doc.doc_type,
                known_acord_form=doc.acord_form,
                known_lob=doc.lob,
            )
            metadata: dict[str, Any] = {
                "source_id": doc.source_id,
                "doc_type": doc.doc_type,
                "acord_form": doc.acord_form,
                "lob": doc.lob,
                "modality_mode": mode,
                "is_scanned": doc.is_scanned,
                "synthetic": doc.synthetic,
                "held_out_carrier": doc.held_out_carrier,
                "page_count": len(doc.image_keys),
            }
            try:
                result = extract(
                    request, model, StaticClassifier(doc.doc_type, doc.acord_form), calibration,
                    calibrators=calibrators, thresholds=thresholds, strict_schema=False,
                )
            except Exception as exc:  # noqa: BLE001 - scored as a miss, not dropped
                log.warning("golden eval: %s (%s) failed: %s", doc.source_id, mode, exc)
                metadata["error"] = f"{type(exc).__name__}: {exc}"
                triples.append((doc.golden, {}, metadata))
                continue
            metadata["field_confidence"] = {
                path: body["confidence"] for path, body in result.fields.items()
            }
            if result.reconciliation is not None:
                # The serving path's own report: the gate reads what production did.
                metadata["reconciliation"] = result.reconciliation
            # For the release measurements: which adapter served, and each call's time.
            metadata["adapter"] = result.route_info.get("adapter")
            metadata["window_latencies_ms"] = list(result.window_latencies_ms)
            # Serving's own verdict, the audit gate's: a common-model policy is
            # served with every key filled, so a section no window wrote is there
            # as nulls and only serving knows it was never written.
            # schema_validity_rate reads this verdict (run_eval.score_subset).
            metadata["schema_valid"] = result.schema_valid
            metadata["validation_errors"] = list(result.validation_errors[:4])
            triples.append((doc.golden, result.extraction, metadata))
    return triples


def evaluate_version(
    client: BlobClient,
    model: Any,
    *,
    version: str,
    corpus_version: str,
    scope: Any,
    tenant_id: str | None = None,
    calibrators: Any = None,
    thresholds: Any = None,
) -> dict[str, Any]:
    """Evaluate on this tenant's frozen set, write the eval report where the gate
    reads it, return its dict."""
    from evaluation.freeze_eval_set import is_frozen
    from evaluation.run_eval import assert_eval_set_disjoint, build_report, eval_set_keys

    # is_frozen first, always: it is also what refuses a set still frozen at the
    # root, which this tenant's empty prefix would otherwise let pass unnoticed.
    if not is_frozen(client, tenant_id=tenant_id) and eval_set_keys(client, tenant_id):
        raise GoldenEvalError(
            "the golden eval set has documents but no manifest: a freeze was interrupted. Re-run "
            "freeze-eval-set from the same corpus to finish it before gating on it."
        )
    # Before anything is evaluated: a leaked eval set makes every number meaningless.
    assert_eval_set_disjoint(client, corpus_version, tenant_id)
    documents = [d for d in load_golden_set(client, scope.doc_types, tenant_id=tenant_id)
                 if scope.covers_lob(d.lob)]
    if not documents:
        raise GoldenEvalError(
            f"the frozen eval set at {paths.golden_eval_set_dir(tenant_id)}/ holds no "
            f"{list(scope.doc_types)} documents, so there is nothing to gate on"
        )
    from evaluation.release_measurements import GpuMemorySampler

    with GpuMemorySampler() as memory:
        triples = evaluate(
            documents, model, client,
            paths.staging_train_images_dir(f"golden-{corpus_version}", tenant_id),
            calibrators=calibrators, thresholds=thresholds,
        )
    failed = sum(1 for _e, _g, meta in triples if meta.get("error"))
    report = build_report(version, triples, corpus_version=corpus_version, scope=scope)
    body = report.as_dict()
    body["documents_failed"] = failed
    # Whether the documents were served through fitted calibrators. Without
    # them every field is flagged, and the gate must not reuse this report
    # once the release has calibrators (pipeline_dag.eval_report_metrics).
    body["calibrated"] = calibrators is not None
    body.update(real_only_section(version, triples, corpus_version=corpus_version, scope=scope))
    body["windowing_ceiling"] = windowing_ceiling(documents)
    # P95 latency per adapter and peak GPU memory, for the release bundle
    # (recorded, not gated; evaluation.release_measurements).
    from evaluation.release_measurements import measurements

    # A call with no adapter here ran on the model this eval loaded: the
    # release's merged model, named by its run.
    body["release_measurements"] = measurements(triples, served_as=scope.run_id(version),
                                                peak_mb=memory.peak_mb)
    client.write_json(paths.eval_report(version, scope=scope.name), body)
    log.info("golden eval of %s: %d document-mode run(s), %d failed", version, len(triples), failed)
    return body


def windowing_ceiling(documents: Sequence[GoldenDocument]) -> dict[str, Any] | None:
    """What reading these policies in windows loses before any model runs
    (evaluation.windowing_ceiling): the oracle merge over the same documents
    and pages the gate just scored. Reported beside the gate, never gated -
    it measures the pipeline, not the candidate. Read against
    field_normalized_match it says how much of the gap is the model's.
    """
    from evaluation.windowing_ceiling import ceiling, oracle

    results = []
    for doc in documents:
        if doc.doc_type != "policy":
            continue
        texts = [doc.page_texts.get(p, "") for p in sorted(doc.page_texts)]
        try:
            results.append(oracle(doc.source_id, doc.golden, doc.lob, len(doc.image_keys),
                                  texts if any(texts) else None, doc.synthetic))
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail the gate
            log.warning("windowing ceiling: %s could not be measured: %s", doc.source_id, exc)
    return ceiling(results) if results else None


def real_only_section(
    version: str, triples: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], **report_kwargs: Any
) -> dict[str, Any]:
    """The same report over the real documents alone, when the set mixes in synthetic ones.

    A delivered split can put synthetic twins of held-out sources into the eval
    set. Their layouts are unseen, which makes them a fair test, but their labels
    came from a generator — so the real documents are always scored apart, and
    the gap between the two is how much the generator flatters the model. Reported
    beside the gate, never instead of it.
    """
    from evaluation.run_eval import build_report

    real = [t for t in triples if not t[2].get("synthetic")]
    synthetic = len(triples) - len(real)
    if not synthetic:
        return {}
    section: dict[str, Any] = {
        "composition": {
            "real_documents": len({t[2]["source_id"] for t in real}),
            "synthetic_documents": len({t[2]["source_id"] for t in triples if t[2].get("synthetic")}),
        },
    }
    if real:
        real_report = build_report(version, real, **report_kwargs).as_dict()
        section["real_only"] = {"gate_metrics": real_report.get("gate_metrics")}
    else:
        section["real_only"] = None
        log.warning("the eval set holds no real documents: every score is on generated labels")
    return section


def dumps(body: dict[str, Any]) -> str:  # pragma: no cover - CLI helper
    return json.dumps(body, indent=2, sort_keys=True)
