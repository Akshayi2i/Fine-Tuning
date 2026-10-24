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

Then each document's main mode is read once more, its windows sent in reverse
order, so the engine batches them differently: how often the two answers agree
is ``run_to_run_agreement`` (decision 2026-10-24). And each reading carries
serving's own printed-values record, which ``hallucination_rate`` reads.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: The modes every golden document is evaluated in.
GOLDEN_MODES: tuple[str, ...] = ("ocr_plus_image", "noisy_ocr_image", "image_only")

#: The mode read twice for run_to_run_agreement: the one production reads.
MAIN_MODE = "ocr_plus_image"

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
    """Every document in this tenant's frozen set, optionally narrowed to ``doc_types``.

    Refused, never narrowed, while any of them has no review record in its
    frozen metadata (``export_golden_labels.review_problem``, decision
    2026-10-22): a gold nobody reviewed does not score, and skipping it would
    quietly score a smaller yardstick than every earlier version was scored on.
    """
    from data_pipeline.labeling.export_golden_labels import review_problem
    from evaluation.run_eval import eval_set_keys

    prefix = paths.golden_eval_set_dir(tenant_id)
    by_source: dict[str, list[str]] = {}
    for key in eval_set_keys(client, tenant_id):
        rest = key[len(prefix):].strip("/")
        if "/" in rest:
            by_source.setdefault(rest.split("/", 1)[0], []).append(key)

    documents = []
    unreviewed: list[tuple[str, str]] = []
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
        if problem := review_problem(None, metadata):
            unreviewed.append((source_id, problem))
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
                client.read_json(f"{base}/golden.json"), [texts.get(p, "") for p in sorted(images)],
                doc_type=doc_type, lob=metadata.get("lob"),
            ),
            image_keys=[images[p] for p in sorted(images)],
            page_texts={p: texts.get(p, "") for p in sorted(images)},
            acord_form=metadata.get("acord_form"),
            lob=merge_line(metadata.get("lob")),
            is_scanned=bool(metadata.get("is_scanned", False)),
            synthetic=bool(metadata.get("synthetic", False)),
            held_out_carrier=bool(metadata.get("held_out_carrier", False)),
        ))
    if unreviewed:
        raise GoldenEvalError(
            f"{len(unreviewed)} frozen document(s) have no review record, so they cannot score "
            f"(decision 2026-10-22): {unreviewed[0][0]}: {unreviewed[0][1]}. Others: "
            f"{[s for s, _ in unreviewed[1:6]]}. Record each one's review in its label metadata and "
            "copy it into the frozen set (evaluation.freeze_eval_set.mark_reviewed), or delete the "
            "set and freeze again from a corpus of reviewed golds."
        )
    return documents


def verified_label(
    label: dict[str, Any], page_texts: list[str], *, doc_type: str = "policy", lob: Any = None,
) -> dict[str, Any]:
    """The frozen gold as training would teach it, through the function the
    corpus build reads its golds with (label_verification): a form's edition
    moved out of its number where the label rules hold for ``doc_type`` and
    ``lob`` (label_rules.label_rules_apply), then rule-added values the page does
    not print left out. A set frozen before the edition rule is not rewritten in
    Blob; it is read under it."""
    from data_pipeline.dataset_builder.label_verification import verified_label as verify

    return copy.deepcopy(verify(label, page_texts, doc_type=doc_type, lob=lob))


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
    run_to_run: bool | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """``(expected, got, metadata)`` triples for :func:`run_eval.build_report`.

    A document that fails to extract is scored as an empty answer, never
    dropped: dropping it would score the model on the documents it managed.

    ``run_to_run`` (default: :func:`run_to_run_check`): after the normal
    passes, every document's main mode (:data:`MAIN_MODE`) is extracted again
    with its windows sent in reverse order (``send_order="reversed"``), and the
    second answer rides on the main-mode triple's metadata
    (``run_eval.REPEAT_ANSWER``) for ``run_to_run_agreement``. It costs one more
    main-mode pass over the whole set. A repeat that fails to extract is an
    empty second answer, as a first pass that fails is an empty first one.
    """
    from dataclasses import replace

    from calibration.fit_calibration import CalibrationParams
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
    from evaluation.run_eval import PRINTED_CHECK, REPEAT_ANSWER
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract
    from training.stage_data import localize_keys

    repeat = run_to_run_check() if run_to_run is None else run_to_run
    local = localize_keys(
        client, sorted({key for doc in documents for key in doc.image_keys}), images_root
    )
    #: The main-mode readings to repeat: what was sent, and the metadata the
    #: second answer joins.
    again: list[tuple[GoldenDocument, Any, Any, dict[str, Any]]] = []
    #: Policy readings sent page text that came back with no printed-values record.
    unchecked: list[str] = []
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
            if repeat and mode == MAIN_MODE:
                again.append((doc, request, calibration, metadata))
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
            # Serving's own printed-values check (serving.printed_values): the
            # values it could check and those no page prints, recorded before
            # strict mode removed any. None with no page text: nothing checked.
            metadata[PRINTED_CHECK] = getattr(result, "printed_check", None)
            if (metadata[PRINTED_CHECK] is None and doc.doc_type == "policy" and mode != "image_only"
                    and any(str(text).strip() for text in texts.values())):
                unchecked.append(f"{doc.source_id} ({mode})")
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
    if unchecked:
        log.warning(
            "golden eval: %d policy reading(s) were sent page text and came back with no "
            "printed-values record, so hallucination_rate does not count their values: %s",
            len(unchecked), unchecked[:5],
        )

    # The repeat pass, after every document's own: the same request, its
    # windows sent in reverse order (the merge keeps planned order), so the
    # engine batches them differently.
    for doc, request, calibration, metadata in again:
        try:
            result = extract(
                replace(request, send_order="reversed"), model,
                StaticClassifier(doc.doc_type, doc.acord_form), calibration,
                calibrators=calibrators, thresholds=thresholds, strict_schema=False,
            )
        except Exception as exc:  # noqa: BLE001 - an empty second answer, not dropped
            log.warning("golden eval: %s (%s, repeat) failed: %s", doc.source_id, MAIN_MODE, exc)
            metadata["repeat_error"] = f"{type(exc).__name__}: {exc}"
            metadata[REPEAT_ANSWER] = {}
            continue
        metadata[REPEAT_ANSWER] = result.extraction
    return triples


def run_to_run_check() -> bool:
    """Whether the golden eval reads each document's main mode twice, for
    ``run_to_run_agreement``: ``run_to_run_check`` under ``evaluation`` in the
    adapter's training config (configs/training/unified.yaml), on unless it says
    false. On, it costs one more main-mode pass over the frozen set."""
    from common.adapter import ADAPTER
    from common.config import training_config

    evaluation = training_config(ADAPTER.training_config).get("evaluation") or {}
    return bool(evaluation.get("run_to_run_check", True))


def assert_scoreable(
    client: BlobClient, doc_types: Sequence[str] | None = None, *, tenant_id: str | None = None,
) -> None:
    """Refuse, before any model is loaded, a frozen set the gate cannot score: a
    freeze that was interrupted, or a frozen gold of ``doc_types`` with no review
    record (decision 2026-10-22, checked as :func:`load_golden_set` checks it).

    Reads each frozen document's metadata and nothing else. Without it the gate
    loaded the merged bf16 model, then learned from load_golden_set that a gold
    had no record - and, retried, loaded the model again to learn it twice.
    """
    from data_pipeline.labeling.export_golden_labels import review_problem
    from evaluation.run_eval import eval_set_keys

    _refuse_interrupted_freeze(client, tenant_id)
    prefix = paths.golden_eval_set_dir(tenant_id)
    keys = set(eval_set_keys(client, tenant_id))
    unreviewed: list[tuple[str, str]] = []
    for key in sorted(keys):
        rest = key[len(prefix):].strip("/")
        source_id, _sep, name = rest.partition("/")
        if name != "golden.json":
            continue
        base = f"{prefix}/{source_id}"
        metadata = (
            client.read_json(f"{base}/metadata.json") if f"{base}/metadata.json" in keys else {}
        )
        if doc_types and metadata.get("doc_type") not in doc_types:
            continue
        if problem := review_problem(None, metadata):
            unreviewed.append((source_id, problem))
    if unreviewed:
        raise GoldenEvalError(
            f"{len(unreviewed)} frozen document(s) have no review record, so they cannot score "
            f"(decision 2026-10-22): {unreviewed[0][0]}: {unreviewed[0][1]}. Others: "
            f"{[s for s, _ in unreviewed[1:6]]}. Record each one's review in its label metadata and "
            "copy it into the frozen set (evaluation.freeze_eval_set.mark_reviewed), or delete the "
            "set and freeze again from a corpus of reviewed golds. Refused before any model was loaded."
        )


def _refuse_interrupted_freeze(client: BlobClient, tenant_id: str | None) -> None:
    from evaluation.freeze_eval_set import is_frozen
    from evaluation.run_eval import eval_set_keys

    # is_frozen first, always: it is also what refuses a set still frozen at the
    # root, which this tenant's empty prefix would otherwise let pass unnoticed.
    if not is_frozen(client, tenant_id=tenant_id) and eval_set_keys(client, tenant_id):
        raise GoldenEvalError(
            "the golden eval set has documents but no manifest: a freeze was interrupted. Re-run "
            "freeze-eval-set from the same corpus to finish it before gating on it."
        )


def evaluate_version(
    client: BlobClient,
    model: Any,
    *,
    version: str,
    corpus_version: str,
    tenant_id: str | None = None,
    calibrators: Any = None,
    thresholds: Any = None,
    lines: Collection[str] | None = None,
    adapter_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Evaluate on this tenant's frozen set, write the eval report where the gate
    reads it, return its dict.

    ``adapter_checkpoint`` names the checkpoint ``model`` applies (the pipeline
    passes it), recorded so the report is never reused for another one.

    Each document's main mode is read twice when :func:`run_to_run_check` is
    on (the default): one more main-mode pass, for ``run_to_run_agreement``.

    ``lines`` are the lines of business the release serves: only frozen policies
    of one of them are scored, because serving sends every other policy (another
    line, a package, no line) to the base model, not to this release. ``None``
    scores every frozen document of the adapter's types (``run_eval``'s command
    line). The report records them (``lines_scored``), so the gate reuses it, or
    compares a baseline with it, only for the same lines.
    """
    from common.adapter import ADAPTER
    from common.lob import single_line
    from evaluation.run_eval import assert_eval_set_disjoint, build_report

    _refuse_interrupted_freeze(client, tenant_id)
    # Before anything is evaluated: a leaked eval set makes every number meaningless.
    assert_eval_set_disjoint(client, corpus_version, tenant_id)
    documents = load_golden_set(client, ADAPTER.doc_types, tenant_id=tenant_id)
    if lines is not None:
        served = set(lines)
        documents = [d for d in documents if single_line(d.lob) in served]
    if not documents:
        raise GoldenEvalError(
            f"the frozen eval set at {paths.golden_eval_set_dir(tenant_id)}/ holds no "
            f"{list(ADAPTER.doc_types)} documents"
            + (f" of the lines this release serves ({sorted(lines)})" if lines is not None else "")
            + ", so there is nothing to gate on"
        )
    from evaluation.release_measurements import GpuMemorySampler

    # The repeat pass for run_to_run_agreement: one more main-mode pass.
    repeat = run_to_run_check()
    with GpuMemorySampler() as memory:
        triples = evaluate(
            documents, model, client,
            paths.staging_train_images_dir(f"golden-{corpus_version}", tenant_id),
            calibrators=calibrators, thresholds=thresholds, run_to_run=repeat,
        )
    failed = sum(1 for _e, _g, meta in triples if meta.get("error"))
    report = build_report(version, triples, corpus_version=corpus_version)
    body = report.as_dict()
    body["documents_failed"] = failed
    # Whether each document's main mode was read twice (run_to_run_agreement),
    # and how many second readings failed (scored as empty answers).
    body["run_to_run_check"] = repeat
    body["repeats_failed"] = sum(1 for _e, _g, meta in triples if meta.get("repeat_error"))
    # Which lines were scored (None: every frozen document), so the gate reuses
    # this report, or compares a baseline with it, only for the same lines.
    body["lines_scored"] = sorted(lines) if lines is not None else None
    # Whether the documents were served through fitted calibrators. Without
    # them every field is flagged, and the gate must not reuse this report
    # once the release has calibrators (pipeline_dag.eval_report_metrics).
    body["calibrated"] = calibrators is not None
    # Which frozen set of which tenant this was scored on, under which scoring
    # code, and how many of its documents: the gate compares two reports only
    # on one yardstick, and decides from the real documents' counts whether the
    # release was scored on enough to be production (synthetic twins do not count).
    from evaluation.run_eval import eval_set_record

    body["eval_set"] = eval_set_record(client, tenant_id, documents)
    body.update(real_only_section(version, triples, corpus_version=corpus_version))
    body["windowing_ceiling"] = windowing_ceiling(documents)
    # P95 latency per adapter and peak GPU memory, for the release bundle
    # (recorded, not gated; evaluation.release_measurements).
    from evaluation.release_measurements import measurements

    # A call with no adapter here ran on the model this eval loaded: the base
    # with the release's adapter applied, named by its run.
    body["release_measurements"] = measurements(triples, served_as=ADAPTER.run_id(version),
                                                peak_mb=memory.peak_mb)
    # Which adapter checkpoint was scored: the gate reuses this report only for
    # that checkpoint, never for another one selected since (decision 2026-10-23).
    body["adapter_checkpoint"] = adapter_checkpoint
    client.write_json(paths.eval_report(version, scope=ADAPTER.name), body)

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
