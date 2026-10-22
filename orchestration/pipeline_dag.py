"""The 11 pipeline stages as addressable functions (IMPL-13 §8, arch §13).

``run.py`` composes these; an Airflow DAG or a GitHub Actions workflow could
schedule the same functions unchanged. Keeping them here rather than in the CLI
is what makes ``--from-stage`` a slice of a list instead of a special case in an
argument parser.

Four properties are requirements, not aspirations:

* **Every stage reads and writes Blob or the staging volume, never pod-local disk.**
  A pod is ephemeral; anything written to its own disk is gone at teardown.
* **Every training, evaluation and quantization job writes a run manifest.**
* **Stages are idempotent and resumable.** Re-running a completed stage is a
  no-op, not a duplicate — which is what makes ``--from-stage`` safe to reach for
  after a mid-pipeline failure.
* **Stage 6 is a hard stop.** A candidate that regresses any gating metric does
  not merge, and there is no way round it but a named, written override of the
  gates that failed (arch §15.5), recorded in the decision and the bundle.
  ``all`` therefore never reaches ``package`` on a failed gate.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient
from common.constants import ACTIVE_DOC_TYPES, DAY_ZERO_MIN_LABELS_PER_TYPE
from common.run_ids import is_valid_run_id, version_of
from common.scopes import Scope, default_scope
from orchestration.runpod_controller import RunPodController, StagingVolume
from orchestration.settings import gpu_class_for
from registry_utils.models import RunType

log = logging.getLogger(__name__)

StageStatus = Literal["completed", "skipped", "blocked", "failed"]
Command = Literal["finetune", "package", "manual", "deploy-endpoint", "extract"]


class PipelineError(RuntimeError):
    """Raised when a stage cannot run at all."""


class GateBlocked(RuntimeError):
    """Raised when the promotion gate blocks a candidate.

    A distinct type because it is not a failure — the pipeline worked, and the
    model is the thing that did not qualify. Callers exit non-zero on both, but
    only this one prints per-metric deltas.
    """

    def __init__(self, message: str, gate_result: Any = None) -> None:
        super().__init__(message)
        self.gate_result = gate_result


# --------------------------------------------------------------------------
# Context and results
# --------------------------------------------------------------------------


@dataclass
class StageContext:
    """Everything the stages share. One object, so a stage cannot reach for
    global state and quietly become unrunnable outside the CLI."""

    client: BlobClient
    controller: RunPodController
    out_version: str
    #: Separately permissioned, because ``raw-documents/`` holds unredacted
    #: originals and only ingestion and OCR may reach them (arch §18a). Stages 1
    #: and 2 use this one; everything downstream works from ``processed/``.
    raw_client: BlobClient | None = None
    corpus_version: str = ""
    input_dir: Path | None = None
    #: The types the CORPUS is built over (stages 1-4). What a training run
    #: covers is `scope`, which may be narrower — the corpus is built once and
    #: filtered per scope (`training.corpus_view`).
    doc_types: list[str] = field(default_factory=lambda: list(ACTIVE_DOC_TYPES))

    #: What stages 5-11 train, merge, quantize, gate and publish. Defaults to the
    #: unified scope, whose run ids and artifact paths are unchanged.
    scope: Scope = field(default_factory=default_scope)
    tenant_id: str | None = None
    #: Serving formats, vLLM-native (arch v2.1 §13a). bf16 alone until Phase 0
    #: spike item 9 verifies FP8; the v1 default produced GGUF files the serving
    #: endpoint could not load.
    formats: list[str] = field(default_factory=lambda: ["bf16"])
    dtype: str = "fp16"
    #: Overrides the configured class for the training pod. ``None`` means the
    #: class in ``config/pipeline.yaml``.
    gpu_class: str | None = None

    # -- flags -------------------------------------------------------------
    dry_run: bool = True
    skip_ingest: bool = False
    train_vit: bool = False
    min_labels_per_type: int = DAY_ZERO_MIN_LABELS_PER_TYPE
    push_adapters: bool = False
    skip_quantize: bool = False
    keep_staging: bool = False
    from_blob: bool = False
    git_commit: str | None = None
    seed: int = 42
    #: Retries for a *failed* stage. A blocked gate is never retried — that
    #: would be an override path with extra steps.
    max_attempts: int = 1
    #: Seconds between retries, multiplied by the attempt number. Carried on the
    #: context rather than read from config inside the loop so a caller — a test,
    #: or a run that wants to fail fast — can set it without editing the file.
    retry_backoff_seconds: int = 0

    # -- injected collaborators -------------------------------------------
    #: OCR engine for stage 2. Injected so the DAG is testable without MinerU,
    #: and so swapping the engine (IMPL-03 keeps it swappable) touches one line.
    ocr_engine: Any = None
    #: Produces the candidate's scores against the frozen golden eval set. The
    #: DAG owns the *gate*, not the scoring: scoring needs a model and a GPU,
    #: and pretending otherwise would put a fabricated pass inside the gate.
    metrics_provider: Callable[[StageContext], dict[str, Any]] | None = None
    #: The production version's scores to gate against. ``None`` means this is
    #: the first version and there is nothing to regress against.
    baseline_metrics: dict[str, Any] | None = None
    #: Per-format metrics from the IMPL-12 extraction routine over the frozen
    #: golden eval set, keyed by format. Supplied, the quantize stage applies the
    #: IMPL-10 thresholds and refuses to publish a format that fails.
    quant_metrics: dict[str, dict[str, Any]] | None = None
    #: Per-adapter re-validation results for a Foundation major bump: every
    #: dependent adapter retrained against the new Foundation and gated. Absent
    #: evidence blocks promotion rather than deferring the question (arch §12).
    #: Per-doc-type "has this adapter been revalidated against the new
    #: Foundation" flags, for the arch §12 cascade block.
    revalidation_evidence: dict[str, bool] | None = None
    #: Cross-type regression evidence for a CONTINUED Foundation, as
    #: ``{doc_type: {"current": {...}, "candidate": {...}}}`` — both sides, the
    #: shape `promotion_gate` reads. Separate from `revalidation_evidence`
    #: because that field holds booleans for a different question; reusing it
    #: produced ``{"acord": {"candidate": {...}}}`` with no ``"current"``, which
    #: the gate correctly rejected as empty. The result was that a continued
    #: Foundation could not pass by ANY input: supply metrics and it read as
    #: empty evidence, supply the booleans and it read as no evidence at all.
    cross_type_evidence: dict[str, dict[str, dict[str, float]]] | None = None
    #: A named, written waiver of specific gates (arch v2.1 §15.5), recorded in
    #: the gate decision and the release bundle. ``None`` is the normal case.
    #: From the CLI's ``--override-*`` flags (``run.gate_override_from``).
    gate_override: Any = None
    #: The documented defect this release fixes (a ticket or a sentence). It
    #: satisfies the gate's improvement rule (§15.5) and nothing else: floors
    #: and non-inferiority still apply. ``None`` is the normal case.
    fixes_defect: str | None = None
    #: ``{metric: (current_scores, candidate_scores)}`` aligned by document, for
    #: the gate's paired bootstrap, optionally with ``current_weights,
    #: candidate_weights`` after them. Injected by tests; ``None`` reads both
    #: eval reports' per-document scores (:func:`paired_document_scores`).
    per_document: dict[str, tuple[list[float], ...]] | None = None

    #: Staged checkpoint directories from the training run (arch v2.1 §11.2).
    #: Supplied by the trainer in a real cycle; empty in a dry run, where nothing
    #: was written and there is honestly nothing to choose between.
    checkpoints: list[str] = field(default_factory=list)

    #: The checkpoint early stopping liked. Scored alongside the trailing ones
    #: even when it is not among them — that is exactly the interesting case,
    #: because training continued past what loss preferred.
    best_loss_checkpoint: str | None = None

    #: Injectable so selection is testable without a GPU, and so the scorer is
    #: the SAME one the gate uses: a selector scoring by a different definition
    #: of "correct" picks a checkpoint the gate then rejects.
    checkpoint_scorer: Any = None
    #: ``(ctx, fmt) -> LoadedModel`` for generating with a staged serving format.
    #: ``None`` loads the staged weights into vLLM; tests pass a stub backend.
    serving_model_loader: Any = None

    #: Set once Phase 0 spike item 9 confirms a decoder-only FP8 export loads and
    #: runs in vLLM. Until then FP8 is refused rather than produced untested: an
    #: unverified serving format is a deployment that fails at load, or worse,
    #: one that loads and reads badly (arch v2.1 §13a).
    fp8_verified: bool = False

    #: ``{serving_format: {"calibration": [(features, correct)], "threshold": [...]}}``
    #: from the two halves of the validation split (arch v2.1 §8.2). Keyed by the
    #: exact format: there is no wildcard, because a calibrator fitted on bf16
    #: logprobs reports confidence for a distribution FP8 does not produce.
    calibration_samples: dict[str, Any] = field(default_factory=dict)

    #: The release this cycle produces. Everything gated, promoted and served is
    #: addressed by it (arch v2.1 §12.3).
    release_id: str = ""
    #: Family grouping evidence per doc type, from the dataset build (arch v2.1 §8.2).
    grouping: dict[str, Any] = field(default_factory=dict)

    # -- fitted during the run ---------------------------------------------
    calibrators: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)

    # -- accumulated state -------------------------------------------------
    results: dict[str, StageResult] = field(default_factory=dict)
    manifests: dict[str, Any] = field(default_factory=dict)
    unlabeled_backlog: dict[str, list[str]] = field(default_factory=dict)

    @property
    def raw(self) -> BlobClient:
        """The client permitted to touch ``raw-documents/``."""
        return self.raw_client or self.client

    @property
    def volume(self) -> StagingVolume:
        return self.controller.volume

    @property
    def corpus(self) -> str:
        """The corpus version being trained on; defaults to the output version."""
        return self.corpus_version or self.out_version


@dataclass
class StageResult:
    """What one stage did."""

    name: str
    status: StageStatus
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("completed", "skipped")


@dataclass
class RunReport:
    """The record of one command invocation."""

    command: str
    version: str
    results: list[StageResult] = field(default_factory=list)
    blocked_at: str | None = None
    failed_at: str | None = None
    unlabeled_backlog: dict[str, list[str]] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def ok(self) -> bool:
        return self.blocked_at is None and self.failed_at is None

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "version": self.version,
            "ok": self.ok,
            "started_at": self.started_at,
            "blocked_at": self.blocked_at,
            "failed_at": self.failed_at,
            "unlabeled_backlog": {k: len(v) for k, v in sorted(self.unlabeled_backlog.items())},
            "unlabeled_source_ids": {k: v for k, v in sorted(self.unlabeled_backlog.items()) if v},
            "stages": [
                {"stage": r.name, "status": r.status, "detail": r.detail, **r.data}
                for r in self.results
            ],
        }

    def render(self) -> str:
        lines = [f"{self.command} {self.version}: {'OK' if self.ok else 'FAILED'}"]
        lines += [f"  [{r.status:9}] {r.name}: {r.detail}" for r in self.results]
        for doc_type, backlog in sorted(self.unlabeled_backlog.items()):
            if backlog:
                lines.append(f"  unlabeled backlog ({doc_type}): {len(backlog)} document(s) awaiting review")
        if self.blocked_at:
            lines.append(f"  BLOCKED at {self.blocked_at}")
        if self.failed_at:
            lines.append(f"  FAILED at {self.failed_at}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Stage 1 — ingestion
# --------------------------------------------------------------------------


def _is_ingested(ctx: StageContext) -> bool:
    """Only ``--skip-ingest`` skips ingestion.

    There is no cheap completion check here, and the one that used to be here
    was actively wrong: "any document was ever ingested" is true from cycle two
    onward, so ``finetune --input ./new_batch`` skipped the stage entirely and
    built the corpus from the previous batch.

    Running it is the correct answer rather than a concession. ``ingest_directory``
    is checksum-deduped and idempotent, so a re-run over already-ingested
    documents writes nothing and reports them as duplicates — cheap, and it
    cannot be wrong in the direction that silently loses 500 documents.
    """
    return ctx.skip_ingest


def stage_ingestion(ctx: StageContext) -> StageResult:
    """Checksum-deduped, immutable writes into ``raw-documents/``."""
    from data_pipeline.ingestion.pull_raw_pdfs import ingest_directory

    if ctx.skip_ingest:
        return StageResult("ingestion", "skipped", "--skip-ingest")
    if ctx.input_dir is None:
        raise PipelineError("ingestion needs --input; pass --skip-ingest to reuse what is in Blob")

    # One folder per document type, or one type. A flat folder read once per
    # type stored every PDF as an ACORD, a policy AND a Loss Run: dedup is per
    # type, and the raw layer is write-once.
    typed = [dt for dt in ctx.doc_types if (ctx.input_dir / dt).is_dir()]
    if not typed and len(ctx.doc_types) > 1:
        raise PipelineError(
            f"{ctx.input_dir} has no {'/, '.join(ctx.doc_types)}/ subfolder, so nothing says "
            "which document type each PDF is. Put each type's PDFs in a folder named after it "
            f"({ctx.input_dir.name}/policy/...), or run a scope that covers one type "
            "(--scope policy, --scope personal_lines)."
        )
    skipped = [dt for dt in ctx.doc_types if typed and dt not in typed]
    if skipped:
        log.info("no %s/ folder under %s: nothing to ingest for those types", skipped, ctx.input_dir)

    ingested, duplicates, failed = [], [], []
    for doc_type in typed or ctx.doc_types:
        source = ctx.input_dir / doc_type if typed else ctx.input_dir
        result = ingest_directory(source, doc_type, ctx.raw, tenant_id=ctx.tenant_id)
        ingested += result.ingested
        duplicates += [name for name, _sid in result.skipped_duplicates]
        failed += [f"{name}: {reason}" for name, reason in result.failed]

    return StageResult(
        "ingestion", "completed",
        f"{len(ingested)} ingested, {len(duplicates)} duplicate(s) skipped, {len(failed)} failed",
        {"ingested": len(ingested), "duplicates": len(duplicates), "failed": failed},
    )


# --------------------------------------------------------------------------
# Stage 2 — preprocessing (GPU)
# --------------------------------------------------------------------------


def _is_preprocessed(ctx: StageContext) -> bool:
    from data_pipeline.ocr.run_mineru import find_unprocessed

    return not any(find_unprocessed(ctx.raw, dt, ctx.tenant_id) for dt in ctx.doc_types)


#: The OCR environment on the pod. MinerU 1.x cannot share an environment with
#: vLLM 0.11 / torch 2.8, so OCR has its own (scripts/pod_bootstrap.sh).
OCR_VENV = "/workspace/venv-ocr"


def _assert_ocr_runnable(ctx: StageContext) -> None:
    """Refuse clearly when documents still need OCR and MinerU is not installed here.

    The training environment has no MinerU. OCR runs first, in its own
    environment; once every document has its ocr_meta.json this stage has
    nothing to do and the pipeline carries on in the training environment.
    """
    import importlib.util

    from data_pipeline.ocr.run_mineru import find_unprocessed

    if importlib.util.find_spec("mineru") is not None:
        return
    pending = {dt: len(find_unprocessed(ctx.raw, dt, ctx.tenant_id)) for dt in ctx.doc_types}
    pending = {dt: n for dt, n in pending.items() if n}
    if not pending:
        return
    commands = "\n".join(
        f"  {OCR_VENV}/bin/python -m data_pipeline.ocr.run_mineru --doc-type {dt} --all-unprocessed"
        for dt in pending
    )
    raise PipelineError(
        f"documents still need OCR ({', '.join(f'{dt}: {n}' for dt, n in pending.items())}) and "
        "MinerU is not installed in this environment. It lives in the OCR environment, "
        f"because it cannot share one with vLLM and torch 2.8. Run OCR there first:\n{commands}\n"
        "then re-run this command; preprocessing will find nothing left to do."
    )


def stage_preprocessing(ctx: StageContext) -> StageResult:
    """MinerU OCR plus page rendering at the resolution cap. **Runs on GPU.**"""
    from data_pipeline.ocr.run_mineru import MinerUEngine, find_unprocessed, process_batch

    if ctx.ocr_engine is None:
        _assert_ocr_runnable(ctx)
    engine = ctx.ocr_engine or MinerUEngine()
    processed, skipped, failed = [], [], []
    for doc_type in ctx.doc_types:
        pending = find_unprocessed(ctx.raw, doc_type, ctx.tenant_id)
        if not pending:
            continue
        outcome = process_batch(pending, doc_type, ctx.raw, engine, tenant_id=ctx.tenant_id)
        processed += outcome["processed"]
        skipped += outcome["skipped"]
        failed += [f"{sid}: {reason}" for sid, reason in outcome["failed"]]

    return StageResult(
        "preprocessing", "completed",
        f"{len(processed)} processed, {len(skipped)} unchanged, {len(failed)} failed",
        {"processed": len(processed), "failed": failed},
    )


# --------------------------------------------------------------------------
# Stage 3 — labeling (human, outside the CLI)
# --------------------------------------------------------------------------


def stage_labeling(ctx: StageContext) -> StageResult:
    """Not automated — this stage **reports** on human work and gates on it.

    ``finetune`` ingests and OCRs everything, then builds the corpus from only
    the source_ids that have a validated golden label, reports the rest as a
    backlog, and continues. That is the honest shape: you ingest 500 documents,
    200 are labeled, you train on 200, and the command tells you 300 are waiting
    for a reviewer (IMPL-13 §2).

    It aborts only when the labeled set is empty or below
    ``--min-labels-per-type``, which is the same day-zero floor IMPL-04 uses.
    """
    from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids

    counts, thin = {}, []
    for doc_type in ctx.doc_types:
        labeled = set(list_labeled_source_ids(ctx.client, doc_type, ctx.tenant_id))
        # Counted from processed/ rather than raw-documents/: a document that has
        # not been OCR'd cannot be labeled or trained on either way, and it keeps
        # orchestration out of the container holding unredacted originals.
        processed = {
            key.split("/")[3]
            for key in ctx.client.list(f"processed/{paths._tenant(ctx.tenant_id)}/{doc_type}/")
            if key.endswith("ocr_meta.json")
        }
        ctx.unlabeled_backlog[doc_type] = sorted(processed - labeled)
        counts[doc_type] = len(labeled)
        if len(labeled) < ctx.min_labels_per_type:
            thin.append(f"{doc_type}: {len(labeled)} < {ctx.min_labels_per_type}")

    total = sum(counts.values())
    if total == 0:
        raise PipelineError(
            "no validated golden labels exist, so there is nothing to train on. Labeling is human "
            "work and cannot sit inside an automated command; ingest and OCR have run, and the "
            "documents are waiting in the review tool (IMPL-04)."
        )
    if thin:
        raise PipelineError(
            "the labeled corpus is below the day-zero floor for: " + "; ".join(thin) + ". "
            f"Training on fewer than {ctx.min_labels_per_type} documents per type produces a model "
            "whose eval numbers are noise (IMPL-04). Lower --min-labels-per-type deliberately if "
            "you intend a smoke test rather than a real cycle."
        )

    backlog = sum(len(v) for v in ctx.unlabeled_backlog.values())
    return StageResult(
        "labeling", "completed",
        f"{total} labeled across {len(counts)} type(s); {backlog} document(s) awaiting review",
        {"labeled_by_type": counts, "unlabeled_backlog": backlog},
    )


# --------------------------------------------------------------------------
# Stage 4 — dataset build
# --------------------------------------------------------------------------


def load_labeled_documents(ctx: StageContext) -> list[Any]:
    """Assemble ``SourceDocument``s from Blob for every labeled source_id."""
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument
    from data_pipeline.labeling.export_golden_labels import (
        list_labeled_source_ids,
        load_golden_label,
        review_problem,
    )
    from evaluation.freeze_eval_set import is_scanned

    documents = []
    read_by: dict[str, int] = {}
    unreviewed: list[tuple[str, str]] = []
    for doc_type in ctx.doc_types:
        for source_id in list_labeled_source_ids(ctx.client, doc_type, ctx.tenant_id):
            label, metadata = load_golden_label(source_id, doc_type, ctx.client, ctx.tenant_id)
            # A model-drafted gold nobody reviewed never trains (decision 2026-10-19).
            if problem := review_problem(label, metadata):
                unreviewed.append((source_id, problem))
                continue
            meta_key = paths.ocr_meta(doc_type, source_id, ctx.tenant_id)
            if not ctx.client.exists(meta_key):
                log.warning("%s is labeled but not OCR'd — skipping; re-run stage 2", source_id)
                continue
            ocr_meta = ctx.client.read_json(meta_key)
            if ocr_meta.get("render_only"):
                # Rendered for labeling, never OCR'd: there is no page text. Built
                # into ocr_plus_image rows it would teach the model that every page
                # of a document can be blank while the image is full of values.
                log.warning(
                    "%s was rendered but never OCR'd (render_only) — skipping; run OCR (stage 2) "
                    "to train on it", source_id,
                )
                continue
            # No default. Assuming one page made a document whose meta lost its
            # count train on page 1 alone, against a label for every page.
            page_count = int(ocr_meta.get("page_count") or 0)
            if page_count < 1:
                log.warning(
                    "%s records no pages in its OCR meta — skipping; re-run stage 2", source_id
                )
                continue

            version = str(ocr_meta.get("mineru_version") or "unknown")
            read_by[version] = read_by.get(version, 0) + 1
            documents.append(SourceDocument(
                source_id=source_id,
                doc_type=doc_type,
                golden_label=label,
                # Per page, never joined: the pairing with each page image is
                # what tells the model which text it is looking at, and joining
                # also split every table that crossed a page boundary.
                ocr_pages=[
                    ctx.client.read_text(paths.processed_page(doc_type, source_id, page, "md", ctx.tenant_id))
                    for page in range(1, page_count + 1)
                ],
                image_paths=[
                    paths.processed_page(doc_type, source_id, page, "png", ctx.tenant_id)
                    for page in range(1, page_count + 1)
                ],
                acord_form=metadata.get("acord_form"),
                # Written by both importers. Selects the canonical schema the
                # prompt and the target use; without it a homeowners policy
                # trains against the fallback's field set.
                lob=metadata.get("lob"),
                tenant_id=ctx.tenant_id,
                field_provenance=metadata.get("field_provenance", {}),
                is_scanned=is_scanned(ocr_meta),
                carrier=_document_carrier(metadata, label),
                synthetic=bool(metadata.get("synthetic", False)),
                delivered_split=metadata.get("split"),
                template_id=metadata.get("template_id"),
                delivered_held_out=bool(metadata.get("held_out_carrier", False)),
                twin_index=metadata.get("twin_index") if metadata.get("synthetic") else None,
                render_mode=metadata.get("render_mode"),
                unprinted_values=list(metadata.get("unprinted_values") or []),
            ))
    if unreviewed:
        raise PipelineError(
            f"{len(unreviewed)} labelled document(s) were drafted by a model and nobody reviewed "
            f"them, so they cannot train (decision 2026-10-19): {unreviewed[0][0]}: "
            f"{unreviewed[0][1]}. Others: {[s for s, _ in unreviewed[1:6]]}. Have a person review "
            "them, record reviewer_id and review_date in metadata.json, and re-import."
        )
    assert_one_mineru_version(read_by)
    ctx.grouping = assign_document_groups(ctx, documents)
    return documents


def split_policy(split: dict[str, Any]) -> dict[str, Any]:
    """What the split held out and capped, for the run manifest."""
    return {
        "held_out_carriers_by_line": split.get("held_out_carriers_by_line") or {},
        # A drawn split's hold-out: carriers placed entirely in test, per doc type.
        "held_out_carriers": split.get("held_out_carriers") or {},
        "single_carrier_lines": split.get("single_carrier_lines") or {},
        "lines_not_held_out": split.get("lines_not_held_out") or {},
        "lines_without_carrier": split.get("lines_without_carrier") or {},
        "lines_below_carrier_minimum": split.get("lines_below_carrier_minimum") or {},
        "frozen_eval_set": bool(split.get("frozen_eval_set")),
        "moved_to_test": split.get("moved_to_test") or {},
        "twin_cap": split.get("twin_cap"),
        "twins_dropped": sum((split.get("twins_dropped") or {}).values()),
    }


def assert_one_mineru_version(read_by: dict[str, int]) -> None:
    """Refuse a corpus whose documents were read by more than one MinerU.

    The model learns how MinerU formats its output, so two versions are two input
    distributions - and the manifest records one, taken from a single document,
    so the mix would not even show. ``read_by``: MinerU version -> documents.
    """
    if len(read_by) > 1:
        counts = ", ".join(f"{version} ({n})" for version, n in sorted(read_by.items()))
        raise PipelineError(
            f"the documents were read by more than one MinerU: {counts}. Read them again with one: "
            "run_mineru --all-unprocessed in the OCR environment re-reads those another MinerU read."
        )


def use_delivered_families(documents: list[Any]) -> bool:
    """Switch to the delivery's own split when the documents carry one.

    All or none: a corpus half split upstream and half drawn here has no single
    rule that keeps a family out of two splits. When they do, each document's
    family becomes its source document (``template_id``) — the unit the delivery
    was split by — and families this pipeline detects by text similarity that
    now cross the split are reported rather than merged: several source
    documents of one carrier share a printed form, and a test document on a form
    the model trained on is worth knowing about, not a reason to refuse the
    delivery's split.
    """
    carrying = [d for d in documents if d.delivered_split]
    if not carrying:
        return False
    if len(carrying) != len(documents):
        raise PipelineError(
            f"{len(carrying)} document(s) carry a delivered split and {len(documents) - len(carrying)} "
            "do not. Deliver every document with metadata `split` (train/val/test), or none."
        )
    splits_of: dict[str, set[str]] = {}
    for document in documents:
        splits_of.setdefault(document.family, set()).add(str(document.delivered_split).lower())
    # val/test documents whose text-similar family also has training documents
    on_trained_form = sum(
        1 for d in documents
        if str(d.delivered_split).lower() != "train" and "train" in splits_of[d.family]
    )
    crossing = sum(1 for splits in splits_of.values() if len(splits) > 1)
    for document in documents:
        document.group_id = f"delivered:{document.template_id or document.source_id}"
    if crossing:
        log.warning(
            "delivered split: %d text-similar famil(ies) span splits; %d val/test document(s) share "
            "a printed form with training documents. The split is kept as delivered.",
            crossing, on_trained_form,
        )
    return True


def delivered_split_of(documents: list[Any]) -> dict[str, str]:
    """``family -> split`` for a delivered split, refusing a family in two splits."""
    split_of: dict[str, str] = {}
    for document in documents:
        split = str(document.delivered_split).lower()
        previous = split_of.setdefault(document.family, split)
        if previous != split:
            raise PipelineError(
                f"source {document.template_id or document.source_id} is delivered in both "
                f"{previous} and {split}: one source document and its twins must share a split"
            )
    return split_of


def delivered_held_out_ids(documents: list[Any], assignment: Any) -> list[str]:
    """Test documents the delivery marks as of a carrier no training document has
    (metadata ``held_out_carrier``), not already held out here: recorded with the
    split, so the frozen set marks them and the gate reports them apart."""
    already = set(assignment.held_out_source_ids)
    return sorted(
        d.source_id for d in documents
        if d.delivered_held_out and d.source_id not in already
        and assignment.assignment.get(d.group_id or d.source_id) == "test"
    )


def _document_carrier(metadata: dict[str, Any], label: dict[str, Any]) -> str | None:
    """The carrier a document is held out by: the delivery's own record when it
    has one (the bundle's `carrier`), else the carrier the label names."""
    from common.normalize import normalize_carrier

    recorded = normalize_carrier(metadata.get("carrier")) if metadata.get("carrier") else None
    return recorded or _declared_carrier(label)


def _declared_carrier(label: dict[str, Any]) -> str | None:
    """The carrier the label names. ACORD 25 lists insurers instead; its first
    one is the carrier the certificate is primarily about. A canonical policy
    label holds it as ``carrier.company_name``, inside an envelope.

    Normalised (:func:`common.normalize.normalize_carrier`): the split holds a
    carrier out by this key, and one insurer written three ways was three.
    """
    from common.canonical import values_view
    from common.normalize import normalize_carrier

    label = values_view(label)
    carrier = label.get("carrier")
    if isinstance(carrier, dict):
        # `company_name` on a self-contained schema, `name` on the common model.
        carrier = carrier.get("company_name") or carrier.get("name")
    if isinstance(carrier, str) and carrier.strip():
        return normalize_carrier(carrier)
    insurers = label.get("insurers")
    if isinstance(insurers, list) and insurers and isinstance(insurers[0], dict):
        name = insurers[0].get("name")
        if isinstance(name, str) and name.strip():
            return normalize_carrier(name)
    return None


def _declared_account(label: dict[str, Any]) -> str | None:
    """The insured the label names: ``insured_name`` on a flat label,
    ``named_insured.primary_name`` on a canonical one. Renewals of one account
    group through it, so missing it on canonical labels would let them split."""
    from common.canonical import values_view

    label = values_view(label)
    named = label.get("named_insured")
    account = named.get("primary_name") if isinstance(named, dict) else label.get("insured_name")
    return account if isinstance(account, str) and account.strip() else None


def group_records(documents: list[Any]) -> dict[str, list[Any]]:
    """One :class:`GroupRecord` per family, per doc type, sorted by group id."""
    from data_pipeline.dataset_builder.split_groups import GroupRecord, line_of

    # One GroupRecord per family, per doc type. A document with no detected
    # family is its own group — which reproduces the v1 per-document behaviour
    # for that document rather than leaving it unassigned.
    records: dict[str, dict[str, GroupRecord]] = {}
    for document in documents:
        per_type = records.setdefault(document.doc_type, {})
        record = per_type.get(document.family)
        if record is None:
            per_type[document.family] = GroupRecord(
                group_id=document.family,
                doc_type=document.doc_type,
                source_ids=[document.source_id],
                carrier=document.carrier,
                synthetic=document.synthetic,
                # The line the split balances on. A family is one line in
                # practice (one insured's policy renewed); the first member's
                # stands for it.
                line=line_of(getattr(document, "lob", None)),
            )
        else:
            record.source_ids.append(document.source_id)
            # One synthetic member pins the whole family to train. A family moves
            # as a unit, so a mixed family made splittable — which `and` did,
            # while this comment claimed the opposite — could land in val or
            # test with its generated labels, and eval would score the
            # generator. The real members cost nothing: they still train.
            record.synthetic = record.synthetic or document.synthetic

    return {dt: sorted(v.values(), key=lambda r: r.group_id) for dt, v in records.items()}


def assign_document_groups(ctx: StageContext, documents: list[Any]) -> dict[str, Any]:
    """Stamp every document with its family ``group_id`` (arch v2.1 §8.2).

    Without this every document is its own group, the group-aware split is a
    per-document split under another name, and templates and renewals span train
    and test — the leakage the split exists to prevent, invisible in every metric.

    Grouped within a doc type: ``GroupRecord`` and the split are per type, so a
    group spanning two types would be split independently in each.

    Evidence used: the source file's SHA-256 (as OCR recorded it), a MinHash over the OCR text, and the
    declared carrier + template + insured. The page-1 layout hash is not computed
    — no image-hash dependency is installed — so same-template documents with
    different text group only through a declared template id.
    """
    from data_pipeline.ingestion.dedup_and_group import DocumentFingerprint, assign_groups, minhash

    reports: dict[str, Any] = {}
    by_type: dict[str, list[Any]] = {}
    for document in documents:
        by_type.setdefault(document.doc_type, []).append(document)

    for doc_type, members in sorted(by_type.items()):
        fingerprints = []
        for document in members:
            # The checksum OCR recorded, not the raw metadata: raw-documents/ holds
            # unredacted PII and the dataset build is not allowed to read it (§18a).
            ocr_meta = ctx.client.read_json(
                paths.ocr_meta(doc_type, document.source_id, ctx.tenant_id)
            )
            metadata_key = paths.label_metadata(doc_type, document.source_id, ctx.tenant_id)
            metadata = ctx.client.read_json(metadata_key) if ctx.client.exists(metadata_key) else {}
            text = "\n".join(document.ocr_pages)
            fingerprints.append(DocumentFingerprint(
                source_id=document.source_id,
                doc_type=doc_type,
                # No recorded checksum means no exact-duplicate evidence, not a
                # shared one: the source_id stands in so it matches nothing else.
                content_sha256=(
                    ocr_meta.get("source_checksum") or f"unrecorded:{document.source_id}"
                ),
                layout_phash=None,
                minhash=minhash(text),
                carrier=document.carrier,
                template_id=metadata.get("template_id"),
                account=_declared_account(document.golden_label),
            ))
        report = assign_groups(fingerprints)
        for document in members:
            document.group_id = report.group_of[document.source_id]
        reports[doc_type] = report.as_dict()
    return reports


def _is_corpus_built(ctx: StageContext) -> bool:
    return ctx.client.exists(paths.corpus_manifest(ctx.corpus, ctx.tenant_id))


def exclude_eval_families(
    client: Any, documents: list[Any], *, tenant_id: str | None = None
) -> tuple[list[Any], list[str]]:
    """Drop the frozen eval documents, and every document sharing a family with one.

    Families are assigned over ALL loaded documents first (the frozen ones are
    still labeled and OCR'd, so they load), which is what lets a renewal or a
    same-template sibling of an eval document be recognised here.

    ``documents`` are one tenant's, so they are matched against that tenant's
    frozen set only: source ids are numbered per tenant, and another tenant's
    frozen ``policy_0001`` is a different document.
    """
    from evaluation.run_eval import eval_set_source_ids

    frozen = eval_set_source_ids(client, tenant_id)
    families = {d.family for d in documents if d.source_id in frozen}
    kept = [d for d in documents if d.source_id not in frozen and d.family not in families]
    excluded = sorted(d.source_id for d in documents if d not in kept)
    return kept, excluded


def warn_on_held_out_carriers(manifest: dict[str, Any], documents: list[Any]) -> None:
    """Say when a carrier the eval set holds out is about to be trained on.

    The frozen set's held-out carriers are what makes its "unseen carrier"
    numbers mean anything. Training on a later document of theirs is allowed —
    the data is real — but from then on those numbers measure a seen carrier.
    """
    from data_pipeline.dataset_builder.split_groups import line_of

    held = {c for carriers in (manifest.get("held_out_carriers") or {}).values() for c in carriers}
    by_line = manifest.get("held_out_carriers_by_line") or {}
    seen = sorted({d.carrier for d in documents if d.carrier in held} | {
        f"{d.carrier} ({line_of(getattr(d, 'lob', None))})" for d in documents
        if d.carrier and (by_line.get(d.doc_type) or {}).get(line_of(getattr(d, "lob", None))) == d.carrier
    })
    if seen:
        log.warning(
            "carrier(s) %s are held out in the frozen eval set but now have documents in this "
            "corpus. The eval set's unseen-carrier results no longer measure an unseen carrier.",
            seen,
        )


@dataclass
class CorpusPlan:
    """The corpus as the build would write it, held in memory."""

    documents: list[Any]
    by_type: dict[str, Any]
    assignment: Any
    built: Any
    frozen: bool
    delivered: bool
    modes: Any = None


def plan_corpus(ctx: StageContext) -> CorpusPlan:
    """Load, split and expand the labelled documents - everything the dataset build
    does before it writes. The build and the preflight both call this, so a
    preflight that passes has built exactly the corpus the run will write."""
    from data_pipeline.dataset_builder.build_jsonl import build_corpus, train_source_ids
    from data_pipeline.dataset_builder.sample_modes import assert_mix_is_close, sample_modes
    from data_pipeline.dataset_builder.split_groups import assign_group_splits
    from evaluation.freeze_eval_set import frozen_manifest, is_frozen

    documents = load_labeled_documents(ctx)
    if not documents:
        raise PipelineError("no labeled, OCR'd documents to build a corpus from")

    # Once the tenant's golden eval set is frozen it IS the test set: its
    # documents, and every document in the same family, stay out of the corpus —
    # a renewal of an eval document would otherwise train the model on that
    # document's answers — and the rest split into train and val only. Another
    # tenant's frozen set changes nothing here: this tenant still draws its own
    # test split until it freezes its own.
    frozen = is_frozen(ctx.client, tenant_id=ctx.tenant_id)
    # Before the exclusion below: with a delivered split, "the family of an eval
    # document" is its source document's twins, not every document on its form.
    delivered = use_delivered_families(documents)
    if frozen:
        documents, excluded = exclude_eval_families(ctx.client, documents, tenant_id=ctx.tenant_id)
        if excluded:
            log.info(
                "excluded %d document(s) in the frozen eval set or its families", len(excluded)
            )
        warn_on_held_out_carriers(frozen_manifest(ctx.client, tenant_id=ctx.tenant_id), documents)
        if not documents:
            raise PipelineError("every labeled document is in the frozen eval set or its families")

    by_type = group_records(documents)
    if delivered:
        from data_pipeline.dataset_builder.split_groups import SplitError, assign_delivered_splits

        try:
            assignment = assign_delivered_splits(
                by_type, delivered_split_of(documents), seed=ctx.seed, with_test=not frozen
            )
        except SplitError as exc:
            raise PipelineError(str(exc)) from exc
        assignment.held_out_source_ids.extend(delivered_held_out_ids(documents, assignment))
        log.info("using the delivered split: %s", assignment.counts_by_doc_type)
    else:
        assignment = assign_group_splits(by_type, seed=ctx.seed, with_test=not frozen)
    if frozen:
        # The test set is the frozen one; so are the carriers it holds out. Recorded
        # so the run manifest and model card say what the gate's documents hold out.
        manifest = frozen_manifest(ctx.client, tenant_id=ctx.tenant_id)
        assignment.frozen_eval_set = True
        assignment.held_out_carriers_by_line = dict(manifest.get("held_out_carriers_by_line") or {})
        assignment.held_out_carriers = dict(manifest.get("held_out_carriers") or {})
    # One modality draw per train document per epoch (arch v2.1 §6.1). This is
    # the only sampling step: v1's down-sampler discarded rows to fix a 33/33/33
    # expansion, and running it over epoch rows would drop documents from epochs.
    # At most MAX_TWINS_PER_SEED twins of one seed per render mode train
    # (Fideon SPEC_09 amendment item 5). Before modes are drawn, so a dropped
    # twin draws nothing and the mix check measures what trains.
    from data_pipeline.dataset_builder.split_groups import cap_twins

    documents = cap_twins(documents, assignment, seed=ctx.seed)
    # The global input-mode mix unless the scope's config sets its own
    # (training.data_mix); recorded on the corpus manifest, so a run whose
    # scope wants another mix is refused rather than mislabelled.
    mix = modality_mix_of(ctx)
    modes = sample_modes(train_source_ids(documents, assignment), seed=ctx.seed, mix=mix)
    assert_mix_is_close(modes, mix=mix)
    built = build_corpus(documents, assignment, seed=ctx.seed, mode_assignment=modes)
    return CorpusPlan(documents, by_type, assignment, built, frozen, delivered, modes)


def modality_mix_of(ctx: StageContext) -> dict[str, float]:
    """The input-mode mix this build draws: the scope's, else the global default."""
    from training.data_mix import DataMixError, configured_modality_mix

    try:
        return configured_modality_mix(ctx.scope)
    except DataMixError as exc:
        raise PipelineError(str(exc)) from exc


def corpus_set_aside(documents: Sequence[Any], built: Any) -> tuple[int, float]:
    """``(documents set aside, their share)``: loaded, yet in no row of the built
    corpus. The one figure preflight's corpus check and the build's own guard
    both read, so the two cannot disagree on what >25% means."""
    kept = {row["source_id"] for row in built.all_rows}
    set_aside = len(documents) - len(kept)
    return set_aside, (set_aside / len(documents) if documents else 0.0)


def stage_dataset_build(ctx: StageContext) -> StageResult:
    """Split, expand into modality variants, write JSONL, and pin the corpus.

    The split happens **before** modality expansion, so a document's three
    variants land in one split. Reversing that order leaks a document's own
    content into its evaluation and inflates every number downstream (arch §7).
    """
    from data_pipeline.corpus_manifest import build_manifest
    from data_pipeline.dataset_builder.build_jsonl import train_rows_by_epoch, write_jsonl

    plan = plan_corpus(ctx)
    documents, by_type, assignment, built, frozen, modes = (
        plan.documents, plan.by_type, plan.assignment, plan.built, plan.frozen, plan.modes
    )
    # Refused before anything is written. An epoch file of zero rows trains
    # nothing, and the run would still record a corpus version as built.
    needed = ("train", "val") if frozen else ("train", "val", "test")
    missing = [s for s in needed if not built.rows_by_split.get(s)]
    if missing and ctx.dry_run and "train" not in missing:
        # A fixture-sized dry-run corpus is too small to fill every split; a real
        # build is refused below, and training refuses a run with no val anyway.
        log.warning("corpus %s has no %s rows (dry run, not refused)", ctx.corpus, missing)
    elif missing:
        why = built.cap_report.warning() or "see the build log for rejections"
        raise PipelineError(
            f"corpus {ctx.corpus} has no {'/'.join(missing)} rows from {len(documents)} loaded "
            f"document(s) ({why}). Training needs all three: train to learn from, val to select "
            "a checkpoint and fit calibration, test for the gate. Nothing was written."
        )
    # Preflight's corpus guard, held here too: the one-command chain runs
    # `preflight --skip-corpus` so the corpus is built once, in this stage, and a
    # corpus that set a quarter of its documents aside must still stop before
    # anything trains on it.
    from orchestration.preflight import FAIL_REJECTED_SHARE

    kept = {row["source_id"] for row in built.all_rows}
    set_aside, share = corpus_set_aside(documents, built)
    if not ctx.dry_run and share > FAIL_REJECTED_SHARE:
        why = built.cap_report.warning() or "see the build log for rejections"
        raise PipelineError(
            f"corpus {ctx.corpus} set aside {set_aside} of {len(documents)} document(s) "
            f"({share:.0%}, more than {FAIL_REJECTED_SHARE:.0%}): {why} Nothing was written."
        )

    # Written where training reads them: one file per epoch, one per eval split,
    # all doc types together — one adapter trains on every type (§8.1).
    for epoch, rows in train_rows_by_epoch(built).items():
        ctx.client.write_text(
            paths.corpus_epoch_file(ctx.corpus, epoch, ctx.tenant_id), write_jsonl(rows)
        )
    for split in ("val", "test"):
        ctx.client.write_text(
            paths.corpus_eval_split(ctx.corpus, split, ctx.tenant_id),
            write_jsonl(built.rows_by_split.get(split, [])),
        )

    first = documents[0]
    ocr_environment = ctx.client.read_json(
        paths.ocr_meta(first.doc_type, first.source_id, ctx.tenant_id)
    )
    manifest, coverage = build_manifest(
        corpus_version=ctx.corpus,
        tenant_id=paths._tenant(ctx.tenant_id),
        rows_by_split=built.rows_by_split,
        # Only the documents the corpus kept. One set aside — over budget, or text
        # the trainer would parse as a tag — contributes nothing to training, and
        # counting it would show a line as covered when its documents were dropped.
        golden_labels_by_source={d.source_id: d.golden_label for d in documents if d.source_id in kept},
        provenance_by_source={
            d.source_id: d.field_provenance for d in documents if d.source_id in kept
        },
        split_assignment=assignment.as_dict(),
        # Policies only: a policy's line is metadata (a schema name). ACORD and
        # Loss Run labels carry line_of_business themselves, and the importer copies
        # it into their metadata — keying on "has a metadata lob" dropped them from
        # the enum coverage and counted their values as policy lines.
        lob_by_source={
            d.source_id: d.lob for d in documents
            if d.source_id in kept and d.doc_type == "policy"
        },
        ocr_environment=ocr_environment,
        doc_types=sorted(by_type),
        seed=ctx.seed,
        git_commit=ctx.git_commit or "unknown",
        modality_mix_target=modality_mix_of(ctx),
    )
    ctx.client.write_json(paths.corpus_manifest(ctx.corpus, ctx.tenant_id), manifest)

    for warning in coverage.warnings:
        log.warning("corpus coverage: %s", warning)
    if cap_warning := built.cap_report.warning():
        log.warning("corpus budget: %s", cap_warning)

    return StageResult(
        "dataset_build", "completed",
        f"{len(built.all_rows)} rows from {len(documents)} document(s); {built.summary()}",
        {
            "rows": len(built.all_rows),
            "documents": len(documents),
            "coverage_warnings": list(coverage.warnings),
            "confusable_example_count": coverage.confusable_example_count,
            "grouping": ctx.grouping,
            "modality_draws": modes.as_dict(),
            # Rows over their task budget are rejected, never truncated, and the
            # document is set aside; this is how many, and why.
            "cap_check": built.cap_report.as_dict(),
            # Every document that did not reach the corpus, by reason, and the
            # first few with their reasons — a count in a log line is not a report.
            "documents_kept": len(kept),
            "documents_set_aside": dict(built.set_aside),
            "set_aside_examples": built.skipped[:20],
        },
    )


# --------------------------------------------------------------------------
# Stage 5 — training
# --------------------------------------------------------------------------


def _is_trained(ctx: StageContext) -> bool:
    """One adapter on the volume is a complete training stage (arch v2.1 §4.1).

    v1 also required one staged adapter per document type, because a run was not
    finished until the whole fan-out was. There is no fan-out now.
    """
    # On the pod, the run's own completion marker on the mount: the in-memory
    # volume record dies with the process, so a rerun after a later stage failed
    # retrained from step 0. The directory alone proves nothing - ms-swift
    # creates it when training STARTS - so an interrupted run is not "trained";
    # training runs again and resumes from its last checkpoint (train.resume_point).
    import os

    from training.train import training_completed

    adapter_dir = paths.scoped_staging_adapter_dir(ctx.scope.name, ctx.out_version)
    if os.path.isdir(adapter_dir):
        return training_completed(adapter_dir)
    return ctx.volume.exists(adapter_dir)


def _assert_window_plan(corpus_manifest: dict[str, Any], what: str) -> None:
    """Refuse a corpus holding policies whose windows this code no longer plans.

    The prompt hash cannot catch it: it is computed when the release is
    packaged, from the code doing the packaging, so an adapter trained on
    another window plan carried this code's hash. Serving would then read the
    adapter windows it was never shown (``policy_windows.assert_window_plan``).
    """
    from data_pipeline.dataset_builder.policy_windows import assert_window_plan

    if "policy" not in (corpus_manifest.get("doc_types") or []):
        return
    try:
        assert_window_plan(corpus_manifest, what=what)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc


def stage_training(ctx: StageContext) -> StageResult:
    """ONE unified extractor run (arch v2.1 §4.1).

    v1 trained a Foundation and then fanned out one adapter per document type,
    sequentially, because each sat on top of the Foundation's weights. That
    topology is gone: vLLM applies one LoRA per request, so a Foundation LoRA and
    a per-type LoRA could never both be active on the same call — the stack was
    unservable, not merely awkward. Per-type adapters return only through the
    §4.2 graduation gate, trained on the MERGED foundation and never stacked.
    """
    from common.scopes import ScopeError, assert_one_output_shape
    from registry_utils.models import DataStats
    from training.train import count_examples, train

    corpus_manifest = ctx.client.read_json(paths.corpus_manifest(ctx.corpus, ctx.tenant_id))
    _assert_window_plan(corpus_manifest, f"training run {ctx.out_version}")
    try:
        # A layout family part-way to the common model is two output shapes:
        # no scope trains on documents of both until the family has moved.
        assert_one_output_shape(ctx.scope, [line for line, count in
                                            (corpus_manifest.get("policy_line_counts") or {}).items() if count])
    except ScopeError as exc:
        raise PipelineError(str(exc)) from exc
    counts = corpus_manifest.get("example_counts", {})
    data_stats = DataStats(
        train_examples=count_examples(counts.get("train")),
        val_examples=count_examples(counts.get("val")),
        test_examples=count_examples(counts.get("test")),
        modality_mix=corpus_manifest.get("modality_mix", {}),
        modality_mix_target=corpus_manifest.get("modality_mix_target") or {},
        lob_coverage=corpus_manifest.get("lob_coverage", {}),
        alias_coverage=corpus_manifest.get("alias_coverage", {}),
        confusable_example_count=corpus_manifest.get("confusable_example_count", 0),
        tenant_ids=[paths._tenant(ctx.tenant_id)],
        split_policy=split_policy(corpus_manifest.get("split_assignment") or {}),
    )

    gpu_class = ctx.gpu_class or gpu_class_for("training", scope=ctx.scope.name)
    with ctx.controller.session_pod("training", gpu_class=gpu_class):
        _swift, manifest = train(
            corpus_version=ctx.corpus,
            out_version=ctx.out_version,
            client=ctx.client,
            corpus_manifest=corpus_manifest,
            data_stats=data_stats,
            train_vit=ctx.train_vit,
            dry_run=ctx.dry_run,
            tenant_id=ctx.tenant_id,
            scope=ctx.scope,
        )
        # Keyed by scope, because two scoped runs can be in flight at one
        # version and a single well-known key would hand the gate whichever ran
        # last. `manifest_of` reads it, falling back to the old "foundation" key
        # so a resume against a context built by the previous version still works.
        ctx.manifests[ctx.scope.name] = manifest
        if not ctx.dry_run and not ctx.checkpoints:
            # Nothing else fills these in, so checkpoint selection skipped on
            # every real run and merge took whatever load_best_model_at_end left.
            from evaluation.checkpoint_eval import discover_checkpoints

            staged = paths.scoped_staging_adapter_dir(ctx.scope.name, ctx.out_version)
            ctx.checkpoints, ctx.best_loss_checkpoint = discover_checkpoints(staged)
            if not ctx.checkpoints:
                # A failure, not a warning. ms-swift exiting 0 is not evidence
                # that an adapter exists: zero steps (a resume past max_steps, a
                # tiny corpus) exits cleanly too, and merge would then fall back
                # to the output root, which holds no adapter at all.
                raise PipelineError(
                    f"training for {manifest.run_id} exited cleanly but wrote no checkpoint "
                    f"under {staged}, so there is no adapter to select or merge. Check the "
                    "ms-swift log for the step count."
                )
        # The staging volume is where merge, quantize and push look for the
        # weights. Without this mark the artifacts exist and the pipeline cannot
        # find them.
        ctx.volume.mark(paths.scoped_staging_adapter_dir(ctx.scope.name, ctx.out_version))

    if ctx.push_adapters and not ctx.dry_run:
        # Belt and braces: the adapter is tens of MB, so pushing it now costs
        # little and means a reclaimed volume loses only the merged model. The
        # adapter itself — the checkpoint that loss preferred, or the last one —
        # not the placeholder note this used to write in its place.
        # The adapter files only, as package publishes them - never the
        # checkpoint's resume state, which would stay in the published prefix.
        from artifact_registry.transfer import push_scoped_adapter

        adapter = ctx.best_loss_checkpoint or ctx.checkpoints[-1]
        blob_dir = push_scoped_adapter(adapter, ctx.scope.name, ctx.out_version, client=ctx.client)
        log.info("pushed the adapter from %s -> %s", adapter, blob_dir)

    return StageResult(
        "training", "completed",
        f"trained {manifest.run_id} on {manifest.data_stats.train_examples} examples",
        {
            "runs": [manifest.run_id],
            "run_type": manifest.run_type,
            "push_adapters": ctx.push_adapters,
        },
    )


# --------------------------------------------------------------------------
# Stage 6 — checkpoint selection (arch v2.1 §11.2)
# --------------------------------------------------------------------------


def _rediscover_checkpoints(ctx: StageContext) -> None:
    """Fill ``ctx.checkpoints`` from the staging volume when this process did
    not train. ``--from-stage checkpoint_eval``, ``--from-stage merge`` and a
    training stage skipped as already complete all left it empty, so checkpoint
    selection "skipped" and merge fell back to the ms-swift output root — which
    holds a ``v0-<timestamp>/`` directory, not a loadable adapter."""
    if ctx.dry_run or ctx.checkpoints:
        return
    from evaluation.checkpoint_eval import discover_checkpoints

    staged = paths.scoped_staging_adapter_dir(ctx.scope.name, ctx.out_version)
    if Path(staged).is_dir():
        ctx.checkpoints, ctx.best_loss_checkpoint = discover_checkpoints(staged)


def _selection_settings(ctx: StageContext) -> dict[str, Any]:
    """The validation cut's settings for checkpoint selection, from the scope's config."""
    from common.config import training_config
    from common.constants import MODALITY_MODES

    evaluation = training_config(ctx.scope.training_config)["evaluation"]
    # The reading modes the finalists' whole-document pass reads (none = all).
    modes = [str(m) for m in (evaluation.get("selection_document_modes") or [])]
    unknown = sorted(set(modes) - set(MODALITY_MODES))
    if unknown:
        raise PipelineError(
            f"selection_document_modes names {unknown}, which no validation row is read in; "
            f"the reading modes are {list(MODALITY_MODES)}"
        )
    return {
        "validation_sample_rows": int(evaluation.get("validation_sample_rows") or 0),
        "selection_tie_break_margin": float(evaluation.get("selection_tie_break_margin") or 0.0),
        "selection_document_finalists": int(evaluation.get("selection_document_finalists") or 0),
        "selection_guard_margin": float(evaluation.get("selection_guard_margin") or 0.0),
        "selection_document_modes": modes,
    }


def _checkpoint_scorer(ctx: StageContext) -> Any:
    """The scorer checkpoint selection uses: one vLLM engine per GPU, the
    candidates scored at once, when training had several GPUs
    (evaluation.checkpoint_eval.ParallelScorer); else one engine in this process."""
    from common.config import training_config
    from evaluation.checkpoint_eval import parallel_vllm_scorer, scoring_gpus, vllm_scorer
    from training.train import training_gpus

    selection = _selection_settings(ctx)
    settings = {
        "client": ctx.client,
        # The scope's own validation view: scoring a policy checkpoint on
        # Loss Runs it never trained on measures the base model.
        "val_path": paths.corpus_scope_eval_split(ctx.corpus, "val", ctx.scope.name, ctx.tenant_id),
        "images_root": paths.staging_train_images_dir(ctx.corpus, ctx.tenant_id),
        "sample_rows": int(selection["validation_sample_rows"]),
        # The finalists' whole-document pass (and a near-tie) reads these modes only.
        "full_modes": selection["selection_document_modes"],
    }
    distributed = training_config(ctx.scope.training_config).get("distributed") or {}
    gpus = scoring_gpus()[: training_gpus(distributed)]
    if len(gpus) > 1:
        staged = paths.scoped_staging_adapter_dir(ctx.scope.name, ctx.out_version)
        return parallel_vllm_scorer(**settings, gpus=gpus, work_dir=Path(staged) / "checkpoint_scores")
    return vllm_scorer(**settings)


def stage_checkpoint_eval(ctx: StageContext) -> StageResult:
    """Pick the checkpoint that ships, by GENERATED field F1.

    Validation loss drove selection under v1, and it is a poor proxy: averaged
    over every token, it is dominated by the easy copy tokens — schema keys, JSON
    punctuation, boilerplate — the model gets right within a hundred steps. A
    checkpoint can improve on loss while getting worse at the values, which is
    the only thing the gate reads.

    Skipped, explicitly, when nothing was staged: a dry run never launched
    ms-swift, so there is nothing to choose between, and merging the staged
    adapter directory is the honest fallback rather than a silent one.
    """
    from evaluation.checkpoint_eval import CheckpointEvalError, select_best

    _rediscover_checkpoints(ctx)
    checkpoints = sorted(ctx.checkpoints or [])
    if not checkpoints:
        return StageResult(
            "checkpoint_eval", "skipped",
            "no staged checkpoints to choose between; merge takes the staged adapter directory",
            {"selected": None},
        )
    # No separate dry-run branch: a dry run never launched ms-swift, so it has no
    # checkpoints, and the check above already covers it. A second condition
    # meaning the same thing is one that can disagree with the first.
    scorer = ctx.checkpoint_scorer
    if scorer is None:  # pragma: no cover - needs a GPU
        try:
            scorer = _checkpoint_scorer(ctx)
        except CheckpointEvalError as exc:
            # A split or setting the scorer cannot use fails identically on a
            # retry - after loading another engine - so it is not retried.
            raise PipelineError(f"checkpoint selection cannot start for {ctx.out_version}: {exc}") from exc

    settings = _selection_settings(ctx)
    try:
        report = select_best(checkpoints, scorer, best_loss=ctx.best_loss_checkpoint)
        full = getattr(scorer, "full", None)
        if settings["selection_document_finalists"] >= 2:
            # The finalists on every validation row, judged on whole documents
            # (which also settles any near-tie on the sample).
            from evaluation.checkpoint_eval import choose_on_documents

            report = choose_on_documents(report, full, finalists=settings["selection_document_finalists"],
                                         guard_margin=settings["selection_guard_margin"],
                                         modes=settings["selection_document_modes"])
        elif full is not None:
            from evaluation.checkpoint_eval import break_tie

            report = break_tie(report, full, settings["selection_tie_break_margin"])
    except CheckpointEvalError as exc:
        raise PipelineError(
            f"no checkpoint could be selected for {ctx.out_version}: {exc}. Merging an "
            "arbitrary one would ship a model nobody measured."
        ) from exc
    finally:
        # The base engine is done once selection is; merge and calibration need
        # the card for the next model.
        if callable(getattr(scorer, "close", None)):
            scorer.close()

    ctx.client.write_json(
        paths.checkpoint_selection(ctx.out_version, scope=ctx.scope.name), report.as_dict()
    )
    detail = f"selected {report.selected} (margin {report.margin:+.4f} field F1)"
    if report.loss_and_f1_disagreed:
        detail += f"; validation loss would have shipped {report.best_loss_checkpoint}"
    from evaluation.checkpoint_eval import generation_cost

    cost = generation_cost(scorer)
    if cost:
        detail += f"; {cost['rows']} row(s) generated in {cost['seconds']:.0f} s"
    return StageResult("checkpoint_eval", "completed", detail,
                       {**report.as_dict(), **({"generation": cost} if cost else {})})


# --------------------------------------------------------------------------
# Stage 6 — evaluation and the promotion gate (HARD STOP)
# --------------------------------------------------------------------------


def _major(version: str) -> str:
    """The major component of ``v2``, ``v2.1``, ``foundation-v3`` — all ``2``/``3``."""
    tag = (version_of(version) if is_valid_run_id(version) else version).lstrip("vV")
    return tag.split(".")[0] or tag


def is_major_bump(previous: str | None, candidate: str) -> bool:
    """Whether the candidate is a new Foundation **major** version.

    A patch that continues from the previous Foundation is a different case,
    handled by the gate's own cross-type evidence rule.
    """
    if not previous:
        return False
    return _major(previous) != _major(candidate)


@dataclass
class CascadeWorkList:
    """Which adapters a Foundation bump invalidates, and whether they have passed.

    A Foundation move changes the weights every per-type adapter was trained on
    top of, so promoting it without re-validating them ships three models that
    were never evaluated against the base they now sit on (arch §12). The
    dependency is recorded on every adapter manifest, so this is a query rather
    than an audit.
    """

    previous_foundation: str | None
    dependents: list[str] = field(default_factory=list)
    revalidated: list[str] = field(default_factory=list)
    outstanding: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return bool(self.outstanding)

    def describe(self) -> str:
        return (
            f"Foundation major bump from {self.previous_foundation}: "
            f"{len(self.dependents)} dependent adapter(s), "
            f"{len(self.outstanding)} still to re-validate ({', '.join(self.outstanding)})"
        )


def foundation_upgrade_work_list(ctx: StageContext) -> CascadeWorkList:
    """The dependent-adapter re-validation list for a Foundation major bump."""
    from registry_utils.query_registry import adapters_depending_on, latest_promoted

    # "unified" under arch v2.1 §4.1. A graduated per-type adapter (§4.2) still
    # records the unified run it was trained on as its foundation_version, so the
    # dependency-upgrade rule is unchanged — only the run_type it looks for moved.
    previous = latest_promoted(
        ctx.client, _promoted_lineage(ctx.scope), scope=ctx.scope.name
    )
    work = CascadeWorkList(previous_foundation=previous)
    if not is_major_bump(previous, ctx.out_version):
        return work

    work.dependents = adapters_depending_on(previous or "", ctx.client)
    evidence = ctx.revalidation_evidence or {}
    work.revalidated = sorted(a for a in work.dependents if evidence.get(a))
    work.outstanding = sorted(a for a in work.dependents if not evidence.get(a))
    return work


def eval_report_metrics(ctx: StageContext) -> dict[str, Any]:
    """The default metrics provider: score the frozen golden eval set.

    Runs IMPL-08's ``run_eval`` — the same code the promotion gate's numbers are
    supposed to come from — after asserting the eval set does not overlap the
    corpus. If an eval report for this version already exists (a resumed run, or
    a scoring pass done separately), it is read rather than recomputed.

    Wiring this in is what makes ``finetune`` completable: the CLI supplied no
    provider at all, so every real run trained a Foundation and three adapters on
    an A100 and then aborted at the gate.
    """
    from evaluation.run_eval import assert_eval_set_disjoint

    assert_eval_set_disjoint(ctx.client, ctx.corpus, ctx.tenant_id)

    # This scope's report, not the unified one. Reading the unscoped key gated a
    # policy candidate on the UNIFIED model's numbers — or found no report and
    # failed — while stage_push recorded the scoped key on the manifest.
    summary_key = paths.eval_report(ctx.out_version, scope=ctx.scope.name)
    calibrators, thresholds = release_calibration(ctx, "bf16")
    if ctx.client.exists(summary_key):
        report = ctx.client.read_json(summary_key)
        metrics = report.get("gate_metrics") or report.get("candidate_metrics") or {}
        if metrics and calibrators is not None and report.get("calibrated") is not True:
            # Scored before this release had calibrators: every field was
            # flagged, so auto_accept_error_rate read a trivial 0.0. Not reused.
            log.info("the eval report at %s was made without calibrators; scoring again "
                     "with the release's", summary_key)
        elif metrics:
            log.info("gate metrics read from %s", summary_key)
            return dict(metrics)

    if ctx.dry_run:
        raise PipelineError(
            f"no eval report at {summary_key}, and a dry run loads no model to produce one. "
            "The gate will not judge a candidate on metrics nobody measured."
        )

    # No report yet: produce it. The frozen golden set goes through the serving
    # pipeline with the merged bf16 model and the release's own bf16 calibrators,
    # so the gate measures what production would serve — windows, merge, date
    # post-process and calibrated confidence included.
    from evaluation.golden_eval import evaluate_version
    from inference_core.model_runner import release_model

    loader = ctx.serving_model_loader or _staged_serving_model
    model = loader(ctx, "bf16")
    try:
        report = evaluate_version(
            ctx.client, model,
            version=ctx.out_version, corpus_version=ctx.corpus, scope=ctx.scope,
            tenant_id=ctx.tenant_id,
            calibrators=calibrators, thresholds=thresholds,
        )
    finally:
        release_model(model)
    return dict(report.get("gate_metrics") or {})


def release_calibration(ctx: StageContext, fmt: str) -> tuple[Any, Any]:
    """This release's calibrators and thresholds for one serving format.

    From this process when calibrate ran in it, otherwise from where calibrate
    saved them. A gate resumed in a new process (``--from-stage
    evaluation_gate``) had neither, so the golden set was served uncalibrated:
    everything flagged, nothing accepted, and auto_accept_error_rate a trivial
    0.0 that measured nothing.
    """
    calibrators, thresholds = ctx.calibrators.get(fmt), ctx.thresholds.get(fmt)
    if calibrators is not None or not ctx.release_id:
        return calibrators, thresholds
    key = paths.release_calibrators(ctx.release_id, fmt, ctx.tenant_id)
    if not ctx.client.exists(key):
        return None, None
    from calibration.feature_calibrator import CalibratorSet
    from calibration.thresholds import ThresholdSet

    body = ctx.client.read_json(key)
    calibrators = CalibratorSet.from_dict(body["calibrators"])
    thresholds = ThresholdSet.from_dict(body["thresholds"])
    ctx.calibrators[fmt], ctx.thresholds[fmt] = calibrators, thresholds
    log.info("loaded the %s calibrators of %s from %s", fmt, ctx.release_id, key)
    return calibrators, thresholds


def default_baseline_metrics(ctx: StageContext) -> dict[str, Any] | None:
    """The production version's scores, read from its own eval report.

    ``None`` means there is no promoted version yet — a genuine first version,
    which the gate handles. It never means "could not find one".
    """
    key = _promoted_eval_report_key(ctx)
    if key is None:
        return None
    report = ctx.client.read_json(key)
    return report.get("gate_metrics") or report.get("candidate_metrics") or None


def _promoted_eval_report_key(ctx: StageContext) -> str | None:
    """Where the promoted version's eval report lives; ``None`` with no report."""
    from registry_utils.query_registry import latest_promoted

    promoted = latest_promoted(
        ctx.client, _promoted_lineage(ctx.scope), scope=ctx.scope.name
    )
    if not promoted:
        return None
    # The promoted run's OWN scope decides where its report lives, and it is this
    # scope's previous release by construction — `latest_promoted` was already
    # filtered by scope. Reading the unscoped key compared a policy candidate
    # against unified numbers, or silently degraded to "first version" and
    # dropped the regression check altogether.
    key = paths.eval_report(version_of(promoted), scope=ctx.scope.name)
    if not ctx.client.exists(key):
        log.warning(
            "promoted version %s has no eval report at %s, so this candidate is gated as a first "
            "version — nothing to regress against. That is a gap in the previous cycle's records, "
            "not a pass.", promoted, key,
        )
        return None
    return key


def paired_document_scores(
    ctx: StageContext, warnings: list[str] | None = None,
) -> dict[str, tuple[list[float], list[float], list[float], list[float]]] | None:
    """Each gate metric's per-document scores, promoted vs candidate, paired.

    Read from the two eval reports (``EvalReport.per_document_scores``) and
    aligned on the documents both scored, in sorted order: that alignment is
    what the paired bootstrap resamples. Each comes as ``(current, candidate,
    current_weights, candidate_weights)``: a document counts in the bootstrap as
    it counts in the reported metric (``EvalReport.per_document_weights``; one
    each in a report written before they were kept).

    Paired only when both reports were scored on the same frozen set of this
    tenant (``eval_set``, :func:`evaluation.run_eval.eval_set_record`). Source ids
    are numbered per tenant and reused when a set is frozen again, and the
    report path names no tenant, so matching on ids alone paired documents
    across two yardsticks. ``None`` whenever pairing is not possible - another
    set or tenant, no shared documents, a report without per-document scores or
    without the set it was scored on - with the reason added to ``warnings``:
    the gate then compares point estimates and cannot enforce the improvement
    rule, and the decision has to say so.
    """
    notes = warnings if warnings is not None else []
    promoted_key = _promoted_eval_report_key(ctx)
    if promoted_key is None:
        return None
    candidate_key = paths.eval_report(ctx.out_version, scope=ctx.scope.name)
    if not ctx.client.exists(candidate_key):
        _not_paired(notes, f"the candidate has no eval report at {candidate_key}")
        return None
    promoted_report = ctx.client.read_json(promoted_key)
    candidate_report = ctx.client.read_json(candidate_key)
    reason = _unpairable(ctx, promoted_key, promoted_report, candidate_key, candidate_report)
    if reason:
        _not_paired(notes, reason)
        return None
    current, candidate = promoted_report["per_document"], candidate_report["per_document"]
    current_w = promoted_report.get("per_document_weights") or {}
    candidate_w = candidate_report.get("per_document_weights") or {}
    paired: dict[str, tuple[list[float], list[float], list[float], list[float]]] = {}
    for name in sorted(set(current) & set(candidate)):
        shared = sorted(set(current[name]) & set(candidate[name]))
        if set(current[name]) != set(candidate[name]):
            note = (
                f"{name}: the two eval reports scored different documents of one frozen set "
                f"({len(current[name])} promoted, {len(candidate[name])} candidate, "
                f"{len(shared)} shared); paired on the shared ones."
            )
            log.warning("%s", note)
            notes.append(note)
        if shared:
            paired[name] = (
                [float(current[name][d]) for d in shared],
                [float(candidate[name][d]) for d in shared],
                [float((current_w.get(name) or {}).get(d, 1.0)) for d in shared],
                [float((candidate_w.get(name) or {}).get(d, 1.0)) for d in shared],
            )
    if not paired:
        _not_paired(notes, f"{promoted_key} and {candidate_key} share no scored document")
    return paired or None


def _unpairable(ctx: StageContext, promoted_key: str, promoted: dict[str, Any],
                candidate_key: str, candidate: dict[str, Any]) -> str | None:
    """Why two eval reports' per-document scores cannot be paired; ``None`` when they can."""
    from evaluation.run_eval import EVAL_SET_IDENTITY

    for key, report in ((promoted_key, promoted), (candidate_key, candidate)):
        if not report.get("per_document"):
            return f"{key} has no per-document scores (written before they were kept)"
        if not all((report.get("eval_set") or {}).get(k) for k in EVAL_SET_IDENTITY):
            return (f"{key} does not record which frozen eval set it was scored on (written "
                    "before reports recorded it), so its documents cannot be shown to be these")
    promoted_set, candidate_set = promoted["eval_set"], candidate["eval_set"]
    tenant = paths._tenant(ctx.tenant_id)
    if {promoted_set["tenant_id"], candidate_set["tenant_id"]} != {tenant}:
        return (f"the promoted report was scored on tenant {promoted_set['tenant_id']!r}'s set "
                f"and the candidate on tenant {candidate_set['tenant_id']!r}'s, gating tenant "
                f"{tenant!r}; source ids are numbered per tenant")
    if any(promoted_set[k] != candidate_set[k] for k in EVAL_SET_IDENTITY):
        return (f"the two reports were scored on different frozen sets (promoted: frozen "
                f"{promoted_set['frozen_at']} from {promoted_set['frozen_from_corpus']}; "
                f"candidate: frozen {candidate_set['frozen_at']} from "
                f"{candidate_set['frozen_from_corpus']}), so one source id may be another "
                "document or another gold")
    return None


def _not_paired(notes: list[str], reason: str) -> None:
    note = (f"per-document scores were not paired: {reason}. Non-inferiority falls back to a "
            "point comparison.")
    log.warning("%s", note)
    notes.append(note)


def scope_eval_set_size(ctx: StageContext, manifest: dict[str, Any]) -> tuple[dict[str, int], str]:
    """How many frozen documents of each type this scope was scored on, and
    where the count was read: ``(counts, source)``.

    From the candidate's own eval report when it records them (``eval_set``):
    exactly the documents it was scored on, after the scope's doc types and
    lines. Otherwise from the frozen manifest: a line-scoped policy scope counts
    the documents of its lines (``documents_by_line``, which counts every
    document recording a line, so it can only overstate), any other scope its
    types (``documents_by_doc_type``). Empty when neither says.

    Not the manifest's ``small_set_accepted``: that describes the whole set. A
    GL-only scope scored on 9 GL policies of a set frozen with 150 policies was
    let through to production, and a policy scope scored on 200 policies was
    capped because the set held 9 Loss Runs.
    """
    key = paths.eval_report(ctx.out_version, scope=ctx.scope.name)
    report = ctx.client.read_json(key) if ctx.client.exists(key) else {}
    recorded_set = report.get("eval_set") or {}
    recorded = recorded_set.get("documents_by_doc_type")
    # The report path names no tenant: another tenant's report says nothing here.
    if (isinstance(recorded, dict) and recorded
            and recorded_set.get("tenant_id") == paths._tenant(ctx.tenant_id)):
        return {t: int(recorded.get(t) or 0) for t in ctx.scope.doc_types}, "eval_report"
    if not manifest:
        return {}, "unknown"
    if ctx.scope.lines:
        by_line = manifest.get("documents_by_line") or {}
        # A package's lines are joined with "+" (split_groups.line_of).
        count = sum(int(n) for line, n in by_line.items() if ctx.scope.covers_lob(line.split("+")))
        return {t: count for t in ctx.scope.doc_types}, "frozen_manifest.documents_by_line"
    by_type = manifest.get("documents_by_doc_type") or {}
    return {t: int(by_type.get(t) or 0) for t in ctx.scope.doc_types}, "frozen_manifest.documents_by_doc_type"


def stage_evaluation_gate(ctx: StageContext) -> StageResult:
    """Score the candidate against the frozen golden eval set, then gate.

    **This stage is a hard stop inside ``package``** (arch v2.1 §13). It runs
    after merge, quantize and calibrate, so a regression stops the RELEASE, not
    the build: the merged model exists on the staging volume, but nothing is
    published, the release bundle is never written, and the command exits
    non-zero with per-metric verdicts. There is no ``--force``. There is a named,
    written override (``ctx.gate_override``, §15.5; ``--override-approver``,
    ``--override-reason`` and ``--override-gate`` on ``package`` and ``all``),
    recorded in the gate decision and the bundle.
    """
    from evaluation.freeze_eval_set import MIN_FROZEN_DOCS_PER_TYPE, frozen_manifest
    from evaluation.gating import apply_to_manifest, promotion_gate

    provider = ctx.metrics_provider or eval_report_metrics
    candidate = provider(ctx)
    baseline = ctx.baseline_metrics
    per_document = ctx.per_document
    # Why per-document scores could not be paired, for the decision's warnings.
    pairing: list[str] = []
    if baseline is None:
        baseline = default_baseline_metrics(ctx)
        # Paired only against the promoted report's own documents: an injected
        # baseline has no per-document scores to pair with.
        if per_document is None and baseline is not None:
            per_document = paired_document_scores(ctx, warnings=pairing)

    # Read the manifest back from the registry when this run did not train it —
    # a resume, or --from-stage evaluation_gate, leaves ctx.manifests empty, and
    # the gate then saw continued_from=None and silently dropped the cross-type
    # requirement a continued Foundation is supposed to face. It also skipped
    # recording the decision, while stage_push published the run regardless.
    foundation = manifest_of(ctx)
    if foundation is None:
        from registry_utils.query_registry import RegistryQueryError
        from registry_utils.query_registry import get as get_manifest

        try:
            run_id = ctx.scope.run_id(ctx.out_version)
            foundation = get_manifest(run_id, ctx.client)
            ctx.manifests[ctx.scope.name] = foundation
        except (RegistryQueryError, KeyError, FileNotFoundError) as exc:
            raise PipelineError(
                f"no run manifest for {run_id}, so the gate cannot tell "
                "whether this Foundation continued from a previous checkpoint — and a continued "
                "one must show cross-type regression evidence before promotion (arch §12). "
                f"Re-run training for this version rather than gating blind ({exc})."
            ) from exc
    # The set the candidate was scored on. A scope scored on fewer than the
    # minimum documents of a type it covers caps the release at interim, with a
    # warning - whatever the whole set's own flag says (scope_eval_set_size).
    # A count nothing records is not shown to be large enough.
    eval_set = frozen_manifest(ctx.client, tenant_id=ctx.tenant_id)
    scored_on, counted_from = scope_eval_set_size(ctx, eval_set)
    small = not scored_on or any(n < MIN_FROZEN_DOCS_PER_TYPE for n in scored_on.values())
    result = promotion_gate(
        candidate,
        baseline,
        small_eval_set=small,
        warnings=pairing,
        # Per-document scores turn non-inferiority into a paired bootstrap and
        # let the improvement rule be enforced (arch §15.5).
        per_document=per_document,
        fixes_defect=ctx.fixes_defect,
        continued_from=getattr(foundation, "continued_from", None),
        # Passed through verbatim: the operator supplies both sides, because
        # only they know which promoted per-type report is the baseline.
        cross_type_evidence=ctx.cross_type_evidence or None,
        override=ctx.gate_override,
        # Decides which metrics are NOT APPLICABLE and supplies this scope's
        # floors. Without it a policy-only run blocks for ever on Loss Run
        # reconciliation, which nothing it covers could have produced.
        scope=ctx.scope,
    )

    # `gate_decision`, not `eval_report`. Writing here used to clobber the
    # scored EvalReport at the same key, taking `by_doc_type` and every error
    # record with it — which is what `vit_gate` reads to decide whether the
    # vision encoder is the bottleneck.
    report_key = paths.gate_decision(ctx.out_version, scope=ctx.scope.name)
    # Before the decision is written, not after: `package --from-stage package`
    # reads the recorded decision, and one saying "passed" for a version the
    # cascade then blocked let it be published.
    cascade = foundation_upgrade_work_list(ctx) if result.passed else None
    cascade_blocked = bool(cascade and cascade.blocking)
    decision = {
        "version": ctx.out_version,
        "candidate_metrics": candidate,
        "gate_metrics": candidate,
        "baseline_metrics": baseline,
        "passed": result.passed and not cascade_blocked,
        "cascade_blocked": cascade.describe() if cascade_blocked else None,
        "failed_gates": result.failed_gates,
        # The full verdict per metric — floor, interval and basis — not just a
        # delta. "Why was this blocked" needs the evidence, not the difference.
        "verdicts": [v.as_dict() for v in result.verdicts],
        "improved_metrics": result.improved_metrics,
        "waived_gates": sorted(result.waived),
        "override": result.override.as_dict() if result.override else None,
        # How many documents each metric's bootstrap paired; empty when the gate
        # compared point estimates (a first release, or reports without them).
        "paired_documents": {name: len(pair[0]) for name, pair in sorted((per_document or {}).items())},
        "fixes_defect": ctx.fixes_defect,
        "eval_set": {
            "small_set_accepted": bool(eval_set.get("small_set_accepted")),
            "documents_by_doc_type": eval_set.get("documents_by_doc_type", {}),
            "frozen_from_corpus": eval_set.get("frozen_from_corpus"),
            "frozen_at": eval_set.get("frozen_at"),
            # What decided the interim cap: this scope's documents per type,
            # where that count was read, and the minimum it was held to.
            "scope_documents_by_doc_type": scored_on,
            "scope_documents_counted_from": counted_from,
            "minimum_per_doc_type": MIN_FROZEN_DOCS_PER_TYPE,
            "small_for_scope": small,
        },
        "warnings": list(result.warnings),
        # Fideon SPEC_09 §6: production (field match >= 0.92) or interim; None if
        # blocked, by the gate or by the cascade.
        "tier": result.tier if not cascade_blocked else None,
    }
    decision["scope"] = ctx.scope.name
    decision["doc_types"] = list(ctx.scope.doc_types)
    ctx.client.write_json(report_key, decision)
    # Also under the release, where the bundle points. The candidate metrics are
    # the merged bf16 model's, so this is bf16's gate run and no other format's:
    # a quantized format inherits nothing from it (arch v2.1 §13a).
    ctx.client.write_json(
        paths.release_gate_decision(ctx.release_id, "bf16", ctx.tenant_id), decision
    )
    ctx.volume.write(
        paths.staging_eval_report(ctx.out_version, scope=ctx.scope.name), json.dumps(candidate)
    )

    apply_to_manifest(result, foundation, metrics=candidate)
    if cascade_blocked:
        foundation.promotion.tier = None
    from registry_utils.write_run_manifest import write_manifest

    write_manifest(foundation, ctx.client)

    if not result.passed:
        raise GateBlocked(
            f"promotion gate blocked {ctx.out_version} — not packaging.\n{result.report()}",
            gate_result=result,
        )

    if cascade_blocked:
        raise GateBlocked(
            f"{ctx.out_version} passed its own gate but cannot be promoted yet. "
            f"{cascade.describe()}. Every per-type adapter was trained on top of the previous "
            "Foundation's weights, so promoting this one without re-validating them ships models "
            "that were never evaluated against the base they now sit on (arch §12). Retrain and "
            "gate each dependent, then supply the results as revalidation_evidence.",
            gate_result=result,
        )

    return StageResult(
        "evaluation_gate", "completed",
        "gate passed" + (" (first version, no baseline)" if result.is_first_version else ""),
        {"metrics": candidate, "failed_gates": [], "cascade_dependents": cascade.dependents},
    )


# --------------------------------------------------------------------------
# Stage 7 — merge
# --------------------------------------------------------------------------


def _is_merged(ctx: StageContext) -> bool:
    """One merged model is a complete merge stage (arch v2.1 §4.1)."""
    return ctx.volume.exists(paths.staging_merged_model_dir(ctx.out_version, scope=ctx.scope.name))


def selected_checkpoint(ctx: StageContext) -> str | None:
    """The checkpoint the §11.2 selector picked, from this process or from Blob.

    ONE reader for merge and package: the adapter published must be the one
    merged. A resumed process (``--from-stage merge`` or ``package``) did not run
    checkpoint_eval, but an earlier one did and wrote its choice down.
    """
    selection = ctx.results.get("checkpoint_eval")
    selected = (selection.data or {}).get("selected") if selection else None
    if selected is None:
        key = paths.checkpoint_selection(ctx.out_version, scope=ctx.scope.name)
        if ctx.client.exists(key):
            selected = ctx.client.read_json(key).get("selected")
    return selected


def stage_merge(ctx: StageContext) -> StageResult:
    """PEFT ``merge_and_unload()`` — ONE adapter into the bf16 base.

    v1 merged Foundation first and then the per-type LoRA on top, producing one
    model per document type. There is one adapter now (arch v2.1 §4.1), so there
    is one merge and one model. A graduated per-type adapter (§4.2) is never
    merged: it is applied at serving time on top of these merged weights, one
    LoRA per request.
    """
    from common.config import base_model_config
    from training.merge import merge, plan_merge

    base = base_model_config()["model"]
    selected = selected_checkpoint(ctx)
    if selected is None and not ctx.dry_run:
        raise PipelineError(
            f"no checkpoint was selected for {ctx.out_version}: run checkpoint_eval first. "
            "Merging the ms-swift output root would merge a directory with no adapter in it."
        )

    plan = plan_merge(
        base_model=f"{base['model_id']}@{base['revision']}",
        version=ctx.out_version,
        dtype=ctx.dtype,  # type: ignore[arg-type]
        selected_checkpoint=selected,
        scope=ctx.scope.name,
    )
    output = merge(plan, dry_run=ctx.dry_run)
    ctx.volume.mark(output)

    return StageResult(
        "merge", "completed", plan.describe(),
        {"merged": [ctx.scope.name], "selected_checkpoint": selected},
    )


# --------------------------------------------------------------------------
# Stage 8 — quantize (package)
# --------------------------------------------------------------------------


def _is_quantized(ctx: StageContext) -> bool:
    if ctx.skip_quantize:
        return True
    # ONE model under arch v2.1 §4.1, so one set of formats — not one per
    # document type. bf16 is the merged model itself and is never exported, so a
    # bf16-only cycle has nothing to check and is complete once merge is.
    quantized = [f for f in ctx.formats if f != "bf16"]
    if not quantized:
        return True
    return all(
        ctx.volume.exists(
            paths.staging_quantized_model_dir(ctx.out_version, fmt, scope=ctx.scope.name)
        )
        for fmt in quantized
    )


def stage_quantize(ctx: StageContext) -> StageResult:
    """Produce the vLLM-native serving formats (arch v2.1 §13a).

    bf16 is the merged model itself and is never re-exported. FP8 is produced
    only once Phase 0 spike item 9 has verified that a decoder-only export — with
    the vision tower, the mergers and lm_head excluded — loads and runs in vLLM.

    GGUF is not produced here at all. It is llama.cpp's format, the endpoint runs
    vLLM, and v1 spent conversion and eval compute every cycle on a file nothing
    could deploy. An edge build goes through ``postprocessing.quantize.export_gguf``
    on request, and is validated in llama.cpp rather than by this gate.
    """
    from postprocessing.quantize import plan_quantization, quantize

    if ctx.skip_quantize:
        return StageResult("quantize", "skipped", "--skip-quantize")

    # ONE model under arch v2.1 §4.1, so one plan — not one per document type.
    plan = plan_quantization(
        version=ctx.out_version,
        formats=ctx.formats,
        fp8_verified=ctx.fp8_verified,
        scope=ctx.scope.name,
    )
    outputs = quantize(plan, dry_run=ctx.dry_run)
    for fmt, directory in outputs.items():
        if fmt != "bf16":   # bf16 IS the merged model; already marked by merge
            ctx.volume.mark(directory)
    produced: dict[str, list[str]] = {ctx.scope.name: sorted(outputs)}

    # The threshold gate, between quantize and push (IMPL-13 §4). It runs only
    # when the caller supplied per-format metrics: scoring each GGUF needs the
    # IMPL-12 extraction routine on a GPU, and a gate that invented numbers to
    # have something to judge would be worse than one that says it has none.
    validation: dict[str, Any] = {}
    if ctx.quant_metrics:
        from postprocessing.validate_quant import assert_servable, validate_quant

        report = validate_quant(ctx.quant_metrics, serving_formats=ctx.formats)
        assert_servable(report, ctx.formats)          # raises; no override path
        validation = report.manifest_entry()
    elif not ctx.skip_quantize:
        log.warning(
            "no per-format metrics supplied, so the quantization thresholds were not applied. "
            "Cycle 1 serves bf16, which IS the reference, so there is nothing to compare — but a "
            "QUANTIZED format that reaches serving unvalidated has not passed (IMPL-10 §4)."
        )

    return StageResult(
        "quantize", "completed", f"formats {ctx.formats}",
        {"produced": produced, "quant_threshold_results": validation},
    )


# --------------------------------------------------------------------------
# Stage 9 — calibrate (package)
# --------------------------------------------------------------------------


def _staged_serving_model(ctx: StageContext, fmt: str) -> Any:  # pragma: no cover - needs a GPU
    """The staged model for one serving format, loaded for vLLM generation."""
    from inference_core.model_runner import LoadedModel, VLLMBackend
    from inference_core.runner_config import load_runner_config
    from registry_utils.query_registry import ResolvedModel

    weights = (
        paths.staging_merged_model_dir(ctx.out_version, scope=ctx.scope.name) if fmt == "bf16"
        else paths.staging_quantized_model_dir(ctx.out_version, fmt, scope=ctx.scope.name)
    )
    config = load_runner_config("vllm")
    resolved = ResolvedModel(
        tag=f"{ctx.out_version}:{fmt}", kind="merged", merged_model=weights, from_staging=True,
    )
    return LoadedModel(tag=resolved["tag"], resolved=resolved,
                       backend=VLLMBackend(resolved, config), config=config)


def collect_calibration_samples(ctx: StageContext) -> dict[str, Any]:
    """Generate the validation split with each staged serving format.

    Per format, never shared: quantization moves the logprob distribution, so
    each format's features come from its own generations (arch v2.1 §5.3).
    Before this, nothing produced calibration samples, and calibrate skipped on
    every real run — every field of every release routed to review.
    """
    from evaluation.validation_generation import (
        assert_generations_usable,
        calibration_samples,
        generate_validation,
        read_rows,
    )
    from inference_core.model_runner import release_model

    val_key = paths.corpus_scope_eval_split(ctx.corpus, "val", ctx.scope.name, ctx.tenant_id)
    if not ctx.client.exists(val_key):
        log.warning("no validation split at %s; nothing to calibrate on", val_key)
        return {}
    rows = read_rows(ctx.client.read_text(val_key))
    if ctx.serving_model_loader is None:
        # The real vLLM path opens images by local path; the rows hold Blob keys.
        from training.stage_data import localize_rows

        rows = localize_rows(
            rows, ctx.client, paths.staging_train_images_dir(ctx.corpus, ctx.tenant_id)
        )
    loader = ctx.serving_model_loader or _staged_serving_model

    samples: dict[str, Any] = {}
    formats = ["bf16"] if ctx.skip_quantize else list(ctx.formats)
    for fmt in formats:
        # One engine at a time: each format's model is freed before the next
        # loads, or the second format finds the card still full of the first.
        model = loader(ctx, fmt)
        stats: dict[str, Any] = {}
        try:
            generations = generate_validation(rows, model, stats=stats)
        finally:
            release_model(model)
        log.info("calibration (%s): %s row(s) in %s s, %s call(s) of up to %s, slowest %s s", fmt,
                 stats.get("generated"), stats.get("seconds"), stats.get("calls"), stats.get("batch_rows"),
                 stats.get("slowest_call_seconds"))
        assert_generations_usable(generations, what=f"calibration ({fmt})")
        halves = calibration_samples(generations)
        # Only a format with evidence gets an entry. An empty entry reads as
        # "samples supplied" and would fit a calibrator on nothing.
        if any(halves.values()):
            samples[fmt] = halves
    return samples


def stage_calibrate(ctx: StageContext) -> StageResult:
    """Fit confidence calibrators and review thresholds, per serving format.

    The stage v1 did not have. Without it the serving path has no calibrated
    confidence at all — it reads a raw aggregate and compares it to a hardcoded
    0.70, which is a guess wearing a decimal point (arch v2.1 §5.4).

    **Per serving format, always.** Quantization moves the logprob distribution,
    so a calibrator fitted on bf16 reports confidence for a distribution FP8 does
    not produce. Sharing one across formats is not an optimisation, it is a
    silently wrong number.

    **Two halves of validation, never one.** Calibrators are fitted on the
    calibration half and thresholds chosen on the threshold half (§8.2). The
    calibrator has already been pulled toward the errors in its own half, so a
    threshold chosen there prices risk the model has already been shown — the
    difference between a guarantee and a hope.
    """
    from calibration.feature_calibrator import fit_calibrators
    from calibration.thresholds import auto_accept_error_rate, fit_thresholds

    if not ctx.calibration_samples and not ctx.dry_run:
        ctx.calibration_samples = collect_calibration_samples(ctx)

    if not ctx.calibration_samples:
        # Honest rather than silent: without labelled validation features there
        # is nothing to fit, and shipping an uncalibrated release means every
        # field routes to review. That is a correct outcome, and an operator
        # should know it happened.
        return StageResult(
            "calibrate", "skipped",
            "no labelled validation features supplied, so no calibrator or threshold could be "
            "fitted. Every field will route to review until one is.",
            {"calibrated_formats": [], "unenforced_types": []},
        )

    fitted: dict[str, Any] = {}
    for fmt in ctx.formats:
        samples: dict[str, Any] = (
            ctx.calibration_samples.get(fmt) or {}
        )
        if not samples:
            log.warning(
                "no calibration samples for %s, so it ships uncalibrated and every field of "
                "every type routes to review. A format served without its own calibrator "
                "reports confidence for a distribution it does not produce (arch v2.1 §5.3).",
                fmt,
            )
            continue

        calibrators = fit_calibrators(
            samples.get("calibration", []), release_id=ctx.release_id, serving_format=fmt
        )
        scored_by_type: dict[str, list[tuple[float, bool]]] = {}
        for features, was_correct in samples.get("threshold", []):
            confidence = calibrators.predict(features)
            if confidence is not None:
                scored_by_type.setdefault(features.field_type, []).append(
                    (confidence, was_correct)
                )

        thresholds = fit_thresholds(
            scored_by_type, release_id=ctx.release_id, serving_format=fmt
        )
        ctx.client.write_json(
            paths.release_calibrators(ctx.release_id, fmt, ctx.tenant_id),
            {"calibrators": calibrators.as_dict(), "thresholds": thresholds.as_dict()},
        )
        ctx.calibrators[fmt] = calibrators
        ctx.thresholds[fmt] = thresholds
        fitted[fmt] = {
            "unenforced_types": thresholds.unenforced_types,
            "guarantees": thresholds.guarantees(),
            # The gating metric: the rate of wrong values that reached a user
            # without a human looking (arch v2.1 §15.2).
            "auto_accept_error_rate": round(
                auto_accept_error_rate(scored_by_type, thresholds), 4
            ),
            # Whether the confidence numbers mean what they say, on the half the
            # calibrators were not fitted on: the gap between stated confidence
            # and observed accuracy, per field type and over all fields.
            "expected_calibration_error": _calibration_errors(scored_by_type),
        }

    detail = "; ".join(
        f"{fmt}: {len(body['guarantees'])} field type(s), "
        f"auto-accept error {body['auto_accept_error_rate']:.2%}, "
        f"calibration error {body['expected_calibration_error']['all']:.3f}"
        for fmt, body in sorted(fitted.items())
    ) or "nothing fitted"
    return StageResult("calibrate", "completed", detail, {"by_format": fitted})


def _calibration_errors(scored_by_type: dict[str, list[tuple[float, bool]]]) -> dict[str, float]:
    """Expected calibration error per field type and over every field ("all")."""
    from evaluation.metrics.coverage_metrics import expected_calibration_error

    def ece(pairs: list[tuple[float, bool]]) -> float:
        return round(expected_calibration_error([c for c, _ in pairs], [ok for _, ok in pairs]), 4)

    out = {field_type: ece(pairs) for field_type, pairs in sorted(scored_by_type.items())}
    out["all"] = ece([pair for pairs in scored_by_type.values() for pair in pairs])
    return out


# --------------------------------------------------------------------------
# Stage 11 — package: push artifacts and write the release bundle
# --------------------------------------------------------------------------


def manifest_of(ctx: StageContext) -> Any:
    """This scope's run manifest, or ``None``.

    Falls back to the v2.1 ``"foundation"`` key so a `--from-stage` resume
    against a context built before scopes existed still finds it.
    """
    return ctx.manifests.get(ctx.scope.name) or (
        ctx.manifests.get("foundation") if ctx.scope.is_unified else None
    )


def _promoted_lineage(scope: Scope) -> RunType:
    """Which run type counts as "the previous release" for this scope.

    The baseline a policy run is gated against is the previous POLICY run, not
    whichever run happens to be newest.
    """
    return "unified" if scope.is_unified else "scoped"


def assert_release_id(ctx: StageContext) -> None:
    """Refuse to package without a valid, operator-named release id.

    Everything calibrate, the gate and the bundle write is addressed by it
    (arch v2.1 §12.3). It is named explicitly rather than derived: a derived id
    moves between a failed run and its ``--from-stage`` resume, splitting one
    release's calibrators and gate decisions across two ids.
    """
    if paths.is_valid_release_id(ctx.release_id):
        return
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    existing = {
        key[len(paths.releases_root(ctx.tenant_id)):].strip("/").split("/", 1)[0]
        for key in ctx.client.list(paths.releases_root(ctx.tenant_id) + "/")
    }
    suggestion = paths.next_release_id(existing, now.year, now.month)
    given = f"{ctx.release_id!r} is not a valid release id" if ctx.release_id else "no --release-id"
    raise PipelineError(
        f"{given}. package addresses every calibrator, gate decision and bundle by release "
        f"id (release-YYYY.M.N). The next free id this month is {suggestion}: re-run with "
        f"--release-id {suggestion}, and pass the same id when resuming."
    )


def assert_gate_passed(ctx: StageContext) -> None:
    """Refuse to package a version whose promotion gate did not pass.

    ``finetune`` stops at a blocked gate, but ``package`` is its own command and
    read nothing the gate wrote — so a blocked version could be published by
    running package next. The recorded decision is the authority; an override is
    already folded into its ``passed``.
    """
    key = paths.gate_decision(ctx.out_version, scope=ctx.scope.name)
    if not ctx.client.exists(key):
        if ctx.dry_run:
            log.warning("no gate decision at %s; a dry run packages without one", key)
            return
        raise PipelineError(
            f"no gate decision at {key}, so {ctx.out_version} was never judged. Run the "
            "evaluation gate (`finetune --from-stage evaluation_gate`) before packaging."
        )
    decision = ctx.client.read_json(key)
    if not decision.get("passed"):
        raise PipelineError(
            f"{ctx.out_version} did not pass its promotion gate (failed: "
            f"{decision.get('failed_gates')}), so it is not packaged. Fix the model, or record a "
            "written override through the gate (arch v2.1 §15.5) — not by running package."
        )


def resolve_corpus_version(ctx: StageContext) -> None:
    """The corpus a standalone ``package`` works against: the one its run trained on.

    ``package`` had no way to name it, so ``ctx.corpus`` fell back to the MODEL
    version. Calibration then found no validation rows and skipped, the eval-set
    leak check compared against a corpus that did not exist and passed, and the
    bundle build failed on the missing corpus manifest - after the weights were
    already published. Read from the run manifest; refused when neither that nor
    ``--corpus-version`` names a corpus that exists.
    """
    if ctx.corpus_version or ctx.dry_run:
        return
    from registry_utils.query_registry import RegistryQueryError
    from registry_utils.query_registry import get as get_manifest

    run_id = ctx.scope.run_id(ctx.out_version)
    try:
        recorded = get_manifest(run_id, ctx.client).dependencies.corpus_version
    except (RegistryQueryError, KeyError, FileNotFoundError, AttributeError):
        recorded = ""
    recorded = str(recorded or "").removeprefix("corpus/")
    if recorded:
        ctx.corpus_version = recorded
        log.info("packaging %s against corpus %s, from its run manifest", run_id, recorded)
        return
    if ctx.client.exists(paths.corpus_manifest(ctx.out_version, ctx.tenant_id)):
        return  # the corpus shares the model version's name
    raise PipelineError(
        f"no corpus is known for {ctx.out_version}: its run manifest ({run_id}) records none and "
        f"there is no corpus named {ctx.out_version}. Pass --corpus-version with the corpus the "
        "version trained on; calibration and the eval-set leak check both read it."
    )


def assert_staged(ctx: StageContext) -> None:
    """Fail loudly, with remediation, when the version is not on the volume."""
    import os

    if ctx.from_blob:
        return
    expected = paths.scoped_staging_adapter_dir(ctx.scope.name, ctx.out_version)
    # The mount itself, as _is_trained reads it: the volume record lives in the
    # process that trained, so a `package` run later - a new process - found
    # nothing recorded and refused a version that was sitting on the disk.
    if ctx.volume.exists(expected) or os.path.isdir(expected):
        return
    raise PipelineError(
        f"version {ctx.out_version} is not on the staging volume — expected {expected}. "
        "The volume is working storage and may have been reclaimed since finetune ran. "
        "Re-run `finetune --from-stage merge` to rebuild it, or pass --from-blob if the adapters "
        "were pushed with `finetune --push-adapters` (IMPL-13 §4)."
    )


def _file_hash(*files: Path) -> str:
    """SHA-256 over files, in the order given, each prefixed by its name."""
    import hashlib

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_release_bundle(ctx: StageContext) -> tuple[Any, list[str]]:
    """Assemble the release bundle for this cycle (arch v2.1 §12.3).

    Returns ``(bundle, reasons_not_promoted)``. The bundle is ``promoted`` only
    when every serving format has its own calibrator and its own gate run;
    otherwise it is ``gated`` (it passed the gate that did run) and the reasons
    say what is missing. Promoting by default and listing the gaps would make the
    record claim a guarantee nobody measured.
    """
    from common.config import base_model_config
    from common.prompts import prompt_hash
    from registry_utils.models import GateOverride, ReleaseBundle

    root = Path(__file__).resolve().parent.parent
    base = base_model_config()["model"]
    corpus_manifest = ctx.client.read_json(paths.corpus_manifest(ctx.corpus, ctx.tenant_id))
    _assert_window_plan(corpus_manifest, f"release {ctx.release_id}")

    formats = ["bf16"] if ctx.skip_quantize else list(ctx.formats)
    serving_formats = {
        # bf16 is the merged model itself; it is never re-exported.
        fmt: paths.merged_model_dir(ctx.out_version, scope=ctx.scope.name) if fmt == "bf16"
        else paths.quantized_model_dir(ctx.out_version, fmt, scope=ctx.scope.name)
        for fmt in formats
    }
    # In memory when calibrate ran in this process; in Blob when package runs on
    # its own (`package`, or a resumed `--from-stage package`). Reading memory
    # alone marked every resumed release "gated: no calibrator" although the
    # calibrators it points at were written by the earlier run.
    calibrators = {
        fmt: paths.release_calibrators(ctx.release_id, fmt, ctx.tenant_id)
        for fmt in formats
        if fmt in ctx.calibrators
        or ctx.client.exists(paths.release_calibrators(ctx.release_id, fmt, ctx.tenant_id))
    }
    gate_reports = {
        fmt: paths.release_gate_decision(ctx.release_id, fmt, ctx.tenant_id)
        for fmt in formats
        if ctx.client.exists(paths.release_gate_decision(ctx.release_id, fmt, ctx.tenant_id))
    }

    # Production or interim, as the bf16 gate decided it (gating.release_tier).
    bf16_decision = gate_reports.get("bf16")
    tier = ctx.client.read_json(bf16_decision).get("tier") if bf16_decision else None

    reasons = [
        f"{fmt} has no calibrator, so its confidence is not calibrated (arch v2.1 §5.3)"
        for fmt in formats if fmt not in calibrators
    ] + [
        f"{fmt} has no gate run of its own, and a quantized format inherits nothing from "
        "the bf16 run (arch v2.1 §13a)"
        for fmt in formats if fmt not in gate_reports
    ]

    override = None
    if ctx.gate_override is not None:
        override = GateOverride(
            approver=ctx.gate_override.approver,
            reason=ctx.gate_override.reason,
            waived_gates=list(ctx.gate_override.waived_gates),
        )

    # The training environment's lock (scripts/lock_requirements.sh): every
    # resolved version the model was trained, calibrated and gated with (§10.2).
    # pyproject.toml only if the lock is missing, which records ranges, not versions.
    lock = root / "requirements-train.lock"
    lockfile = lock if lock.exists() else root / "pyproject.toml"

    try:
        bundle = ReleaseBundle(
            release_id=ctx.release_id,
            status="gated" if reasons else "promoted",
            tier=tier,
            tenant_scope=paths._tenant(ctx.tenant_id),
            scope=ctx.scope.name,
            # What this release may SERVE. Serving routes each document type to
            # the narrowest promoted release covering it, so an empty list —
            # which is what every release written before scopes existed carries —
            # reads as "every active type".
            doc_types=[] if ctx.scope.is_unified else list(ctx.scope.serves),
            lines=sorted(ctx.scope.lines),
            base_model=f"{base['model_id']}@{base['revision']}",
            adapter=ctx.scope.run_id(ctx.out_version),
            merged_model=paths.merged_model_dir(ctx.out_version, scope=ctx.scope.name),
            serving_formats=serving_formats,
            calibrators=calibrators,
            gate_reports=gate_reports,
            prompt_hash=prompt_hash(),
            schema_versions={
                str(k): str(v) for k, v in (corpus_manifest.get("schema_versions") or {}).items()
            },
            ocr_pin={
                "mineru_version": corpus_manifest.get("mineru_version"),
                "ocr_device": corpus_manifest.get("ocr_device"),
            },
            vision_config_hash=_file_hash(root / "configs" / "shared" / "vision.yaml"),
            vllm_config_hash=_file_hash(root / "configs" / "inference" / "vllm_serving.yaml"),
            lockfile_hash=_file_hash(lockfile),
            override=override,
        )
    except ValueError as exc:
        raise PipelineError(f"the release bundle for {ctx.release_id} is invalid: {exc}") from exc
    record_release_measurements(ctx, bundle)
    return bundle, reasons


def record_release_measurements(ctx: StageContext, bundle: Any) -> None:
    """Latency, GPU memory and the routing table, onto the bundle (Fideon SPEC_09 handoff item 6).

    Latency and memory are read from this release's eval report and recorded,
    not gated. The routing check fails the release step: every line is routed
    through the serving plan as it stands with this release promoted, and a
    line that cannot be routed, or one this release was trained for that would
    not reach it, stops the release here.
    """
    from serving.release_router import build_serving_plan
    from serving.routing_check import RoutingCheckError, plan_with_candidate, routing_table

    report_key = paths.eval_report(ctx.out_version, scope=ctx.scope.name)
    report = ctx.client.read_json(report_key) if ctx.client.exists(report_key) else {}
    measured = report.get("release_measurements") or {}
    bundle.latency_p95_ms_by_adapter = {
        adapter: entry.get("p95_ms") for adapter, entry in
        (measured.get("latency_p95_ms_by_adapter") or {}).items()
    }
    bundle.peak_gpu_memory_mb = measured.get("peak_gpu_memory_mb")
    bundle.measured_with = measured.get("measured_with")
    if not measured:
        log.warning("%s: no release measurements in %s; latency and GPU memory are not recorded",
                    bundle.release_id, report_key)

    from serving.release_router import ServingPlanError
    from serving.vllm_entrypoint import release_pins

    try:
        # With the serving config's rollback pins, as the endpoint will route.
        plan = plan_with_candidate(
            build_serving_plan(ctx.client, tenant_id=ctx.tenant_id, pins=release_pins()),
            json.loads(bundle.model_dump_json()))
        bundle.routing = routing_table(plan, bundle.release_id, bundle.lines)
    except (RoutingCheckError, ServingPlanError) as exc:
        raise PipelineError(f"{bundle.release_id} is not released: {exc}") from exc


def write_release_bundle(ctx: StageContext, bundle: Any) -> None:
    """Write the bundle, and replace its row in the tenant's release index."""
    ctx.client.write_json(
        paths.release_bundle(bundle.release_id, ctx.tenant_id),
        json.loads(bundle.model_dump_json()),
    )
    index_key = paths.release_index(ctx.tenant_id)
    rows = ctx.client.read_json(index_key) if ctx.client.exists(index_key) else []
    rows = [r for r in rows if r.get("release_id") != bundle.release_id] + [bundle.index_row()]
    ctx.client.write_json(index_key, sorted(rows, key=lambda r: r["release_id"]))


def _clear_staged_scope(ctx: StageContext) -> int:
    """Drop the staged artifacts this scope owns, and nothing else."""
    owned = [
        paths.scoped_staging_adapter_dir(ctx.scope.name, ctx.out_version),
        paths.staging_merged_model_dir(ctx.out_version, scope=ctx.scope.name),
        paths.staging_eval_report(ctx.out_version, scope=ctx.scope.name),
    ]
    if not ctx.skip_quantize:
        owned += [
            paths.staging_quantized_model_dir(ctx.out_version, fmt, scope=ctx.scope.name)
            for fmt in ctx.formats
        ]
    return sum(ctx.volume.clear(path) + _remove_from_mount(path) for path in owned)


def _remove_from_mount(path: str) -> int:
    """Delete a staged artifact from the real volume; 1 when something went.

    The volume record is in memory, so clearing it freed nothing: every
    packaged version left its adapter and merged model (tens of GB) on the
    mount. Only ever a path inside the staging root, never the root itself.
    """
    import os
    import shutil

    root = os.path.realpath(paths.staging_root())
    target = os.path.realpath(path)
    if not os.path.exists(target) or target == root or not target.startswith(root + os.sep):
        return 0
    if os.path.isdir(target):
        shutil.rmtree(target)
    else:
        os.remove(target)
    log.info("removed %s from the staging volume", path)
    return 1


def _record_dry_run_push(ctx: StageContext) -> dict[str, str]:
    """A dry run trained nothing, so it records where each artifact WOULD go."""
    scope = ctx.scope
    pushed: dict[str, str] = {}
    blob_dir = paths.scoped_adapter_dir(scope.name, ctx.out_version)
    ctx.client.write_json(f"{blob_dir}/adapter_config.json", {
        "version": ctx.out_version,
        "kind": "foundation" if scope.is_unified else "scoped",
        "scope": scope.name,
        "doc_types": list(scope.doc_types),
        "doc_type": None,
    })
    pushed[f"adapter:{scope.name}"] = blob_dir

    merged_dir = paths.merged_model_dir(ctx.out_version, scope=scope.name)
    ctx.client.write_json(f"{merged_dir}/config.json", {"dtype": ctx.dtype})
    pushed[f"merged:{scope.name}"] = merged_dir

    if not ctx.skip_quantize:
        for fmt in ctx.formats:
            quant_dir = paths.quantized_model_dir(ctx.out_version, fmt, scope=scope.name)
            ctx.client.write_json(f"{quant_dir}/config.json", {"format": fmt})
            pushed[f"quantized:{scope.name}:{fmt}"] = quant_dir
    return pushed


def _push_weights(ctx: StageContext) -> dict[str, str]:
    """Upload the adapter, the merged model and each quantized format to Blob.

    Package used to write a one-line JSON at each destination — a placeholder
    ``adapter_config.json`` and ``config.json`` — and publish the run against
    them, so a "published" release pointed serving at prefixes holding no
    weights. Every upload goes through :mod:`artifact_registry.transfer`, which
    refuses a missing source directory rather than creating an empty prefix.
    """
    from artifact_registry.transfer import push_merged_model, push_quantized, push_scoped_adapter

    scope = ctx.scope
    adapter = selected_checkpoint(ctx) or paths.scoped_staging_adapter_dir(
        scope.name, ctx.out_version
    )
    pushed = {
        f"adapter:{scope.name}": push_scoped_adapter(
            adapter, scope.name, ctx.out_version, client=ctx.client
        ),
        f"merged:{scope.name}": push_merged_model(
            paths.staging_merged_model_dir(ctx.out_version, scope=scope.name), ctx.out_version,
            client=ctx.client, scope=scope.name,
        ),
    }
    if not ctx.skip_quantize:
        for fmt in ctx.formats:
            if fmt == "bf16":
                continue   # bf16 IS the merged model, pushed above
            pushed[f"quantized:{scope.name}:{fmt}"] = push_quantized(
                paths.staging_quantized_model_dir(ctx.out_version, fmt, scope=scope.name),
                ctx.out_version, fmt, client=ctx.client, scope=scope.name,
            )
    return pushed


def stage_push(ctx: StageContext) -> StageResult:
    """Copy adapters, merged model and quantized models into Blob, then flip the
    manifest from ``staged`` to ``published``.

    The layouts mirror each other deliberately (IMPL-13 §3), so this copies
    rather than translates — a translation step is where a path convention drifts
    between the two stores and an artifact becomes unfindable.
    """
    from registry_utils.query_registry import get as get_manifest
    from registry_utils.query_registry import list_runs
    from registry_utils.write_run_manifest import mark_published

    assert_gate_passed(ctx)
    assert_staged(ctx)
    # Built - and its routing check run - before anything is pushed: a release the
    # check stops must not leave weights in Blob and manifests marked published
    # with no bundle. It records paths only, so it needs nothing pushed yet.
    bundle, not_promoted = build_release_bundle(ctx)
    pushed: dict[str, str] = {}

    # ONE adapter and ONE merged model (arch v2.1 §4.1). v1 fanned this out per
    # document type; that topology is gone, because vLLM applies one LoRA per
    # request and a Foundation plus a per-type adapter could never both be
    # active. A graduated per-type adapter (§4.2) is published by its own run,
    # not by this one.
    #
    # Through the IMPL-02 §3 helper's path, never assembled here. The only place
    # that built these inline is the place that published a Foundation against
    # paths that were never produced.
    scope = ctx.scope
    if ctx.dry_run:
        pushed.update(_record_dry_run_push(ctx))
    else:
        pushed.update(_push_weights(ctx))

    quantized = [] if ctx.skip_quantize else [f for f in ctx.formats if f != "bf16"]
    published: list[str] = []
    for row in list_runs(ctx.client, scope=scope.name):
        run_id = str(row.get("run_id", ""))
        # Parsed, not suffix-matched. `endswith(version)` matched every scope's
        # run at that tag, so packaging one scope published all of them — each
        # against THIS scope's paths, advertising Blob prefixes that hold another
        # scope's weights or nothing at all.
        if not is_valid_run_id(run_id) or version_of(run_id) != ctx.out_version:
            continue
        # A run that crashed or never finished has no weights to publish.
        # `launch_and_record` sets "failed" precisely so the registry never
        # claims weights a dead run never wrote; publishing on run_id alone
        # undid that, advertising Blob paths for an adapter that OOM'd at step
        # 40 and sending serving to fetch an empty prefix.
        status = row.get("status")
        if status not in ("trained", "evaluated", "promoted"):
            log.warning(
                "not publishing %s: its status is %r, so its weights were never written. "
                "Re-run training for this version rather than publishing a path to nothing.",
                run_id, status,
            )
            continue
        doc_type = row.get("doc_type")
        is_graduated = row.get("run_type") == "per_type_adapter"
        run_kind: paths.AdapterKind = "doc_type" if is_graduated else "foundation"
        manifest = get_manifest(run_id, ctx.client)
        manifest.artifacts.eval_report = paths.eval_report(ctx.out_version, scope=scope.name)
        # The unified run owns the adapter AND the merged and quantized models —
        # there is one of each under arch v2.1 §4.1. A graduated per-type adapter
        # (§4.2) owns only its own weights; it is merged into nothing, because it
        # is applied at serving time on top of the merged foundation.
        # A scoped run owns its OWN merged and quantized models, under its own
        # paths. Only a §4.2 graduated adapter owns none: it is applied on top of
        # a merged model rather than being one.
        owns_a_model = not is_graduated
        mark_published(
            manifest,
            ctx.client,
            adapter_weights=(
                paths.adapter_dir(run_kind, ctx.out_version, doc_type) if is_graduated
                else paths.scoped_adapter_dir(scope.name, ctx.out_version)
            ),
            merged_model=(
                paths.merged_model_dir(ctx.out_version, scope=scope.name) if owns_a_model else None
            ),
            # The first QUANTIZED format. formats[0] is bf16, which is the merged
            # model and is never exported, so this pointed at an empty prefix.
            quantized_model=(
                paths.quantized_model_dir(ctx.out_version, quantized[0], scope=scope.name)
                if owns_a_model and quantized else None
            ),
            quantized_formats=list(quantized) if owns_a_model else [],
        )
        published.append(run_id)

    # The unit of promotion (arch v2.1 §12.3). Built after the artifacts are
    # pushed, so every path it pins exists in Blob.
    write_release_bundle(ctx, bundle)
    for reason in not_promoted:
        log.warning("%s is %s, not promoted: %s", bundle.release_id, bundle.status, reason)

    cleared = 0
    if not ctx.keep_staging and published:
        # Only after something was actually published. The comment used to claim
        # the manifest flip verified the push, but nothing checked that any flip
        # happened — so a run where no manifest matched the version deleted every
        # staged adapter, merged model and GGUF and reported success.
        #
        # And only THIS scope's paths: clearing the whole staging root while
        # packaging `policy` would delete `unified`'s staged merged model, which
        # is the only copy until its own package run publishes it.
        cleared = _clear_staged_scope(ctx)
    elif not ctx.keep_staging:
        log.warning(
            "no run manifest matched version %s, so nothing was published and the staging volume "
            "is left intact. Clearing it here would delete the only copy of the weights. Check "
            "that training wrote its manifests before re-running package.", ctx.out_version,
        )

    return StageResult(
        "package", "completed",
        f"pushed {len(pushed)} artifact location(s), published {len(published)} manifest(s), "
        f"release {bundle.release_id} {bundle.status}"
        + (f", cleared {cleared} staged path(s)" if cleared else ""),
        {
            "pushed": pushed, "published": published, "staging_cleared": cleared,
            "release_id": bundle.release_id, "release_status": bundle.status,
            "not_promoted_because": not_promoted,
        },
    )


# --------------------------------------------------------------------------
# Stages 10 and 11 — outside finetune/package
# --------------------------------------------------------------------------


def stage_deploy(ctx: StageContext) -> StageResult:
    """The persistent serving endpoint pulls the promoted artifact."""
    version = ctx.controller.deploy_endpoint(ctx.out_version, dry_run=ctx.dry_run)
    return StageResult("serving", "completed", f"endpoint -> {version}", {"version": version})


def stage_feedback(ctx: StageContext) -> StageResult:
    """Low-confidence extraction output feeds the next corpus version (IMPL-04).

    Driven by ``extract``, not by a build: the loop closes when documents are
    actually processed, and there is nothing to feed back at build time.
    """
    return StageResult(
        "feedback_loop", "skipped",
        "driven by `extract` plus the IMPL-04 active-learning queue, not by a build command",
    )


# --------------------------------------------------------------------------
# The DAG
# --------------------------------------------------------------------------


def _deterministic_errors() -> tuple[type[BaseException], ...]:
    """Failures a retry would only repeat, after re-doing every step before them.

    A training stage that fails on a missing page image, an over-length row, a
    refused configuration or a crashed ``swift sft`` fails identically the second
    time — after re-materializing, re-staging and re-measuring the corpus, and,
    for a crash at hour five, after a second full training run. Retries are for
    transient faults (a throttled Blob read), not for these. Imported lazily so
    the DAG stays importable without the training stack.
    """
    import subprocess

    from training.corpus_view import CorpusViewError
    from training.length_check import LengthCheckError
    from training.stage_data import StagingError
    from training.train import TrainingError

    return (TrainingError, StagingError, LengthCheckError, CorpusViewError,
            subprocess.CalledProcessError)


@dataclass(frozen=True)
class Stage:
    """One addressable stage."""

    number: int
    name: str
    command: Command
    gpu: bool
    run: Callable[[StageContext], StageResult]
    is_complete: Callable[[StageContext], bool] | None = None


STAGES: tuple[Stage, ...] = (
    Stage(1, "ingestion", "finetune", False, stage_ingestion, _is_ingested),
    Stage(2, "preprocessing", "finetune", True, stage_preprocessing, _is_preprocessed),
    Stage(3, "labeling", "finetune", False, stage_labeling, None),
    Stage(4, "dataset_build", "finetune", True, stage_dataset_build, _is_corpus_built),
    Stage(5, "training", "finetune", True, stage_training, _is_trained),
    # Between training and the gate: the gate scores what ships, so what ships
    # has to be chosen first (arch v2.1 §11.2).
    Stage(6, "checkpoint_eval", "finetune", True, stage_checkpoint_eval, None),
    Stage(7, "merge", "finetune", True, stage_merge, _is_merged),
    Stage(8, "quantize", "package", True, stage_quantize, _is_quantized),
    # Calibrate BEFORE the gate: the gate reads auto_accept_error_rate, which is
    # the rate of wrong values that reached a user without a human looking — and
    # that number does not exist until thresholds are chosen (arch v2.1 §5.4).
    Stage(9, "calibrate", "package", True, stage_calibrate, None),
    # The gate moved AFTER merge, quantize and calibrate (arch v2.1 §13). Under
    # v1 it ran before merge, so it scored the bare adapter — not the merged
    # model, and certainly not the merged model in each serving format.
    # Quantization degrades exactly what was fine-tuned in, so an FP8 release
    # inherits nothing from bf16's result.
    Stage(10, "evaluation_gate", "package", True, stage_evaluation_gate, None),
    Stage(11, "package", "package", False, stage_push, None),
    Stage(12, "serving", "deploy-endpoint", False, stage_deploy, None),
    Stage(13, "feedback_loop", "extract", False, stage_feedback, None),
)

STAGE_BY_NAME: dict[str, Stage] = {s.name: s for s in STAGES}

#: Stage 3 is human work and stage 6 must re-judge every run, so neither is
#: skippable by an idempotency check. That is deliberate: an "already gated"
#: shortcut would let a re-run inherit a pass it did not earn.
FINETUNE_STAGES = tuple(s for s in STAGES if s.command == "finetune")
PACKAGE_STAGES = tuple(s for s in STAGES if s.command == "package")

#: Stages 1-4 build the corpus, which every scope shares. Running them per scope
#: would re-ingest, re-OCR and re-split the same documents — and a second split
#: draw is the leakage `corpus_view` exists to avoid.
SHARED_STAGES = tuple(s for s in STAGES if s.name in
                      ("ingestion", "preprocessing", "labeling", "dataset_build"))

#: Stages 5-11 produce one scope's adapter, model, calibrators and release. They
#: run once per scope, each against its own StageContext.
PER_SCOPE_STAGES = tuple(
    s for s in STAGES
    if s.command in ("finetune", "package") and s not in SHARED_STAGES
)


def stages_for(command: str) -> tuple[Stage, ...]:
    """The stages a command runs. ``all`` is ``finetune`` + ``package``, and
    never extraction — that is the rule the command surface exists to express."""
    if command == "finetune":
        return FINETUNE_STAGES
    if command == "package":
        return PACKAGE_STAGES
    if command == "all":
        return FINETUNE_STAGES + PACKAGE_STAGES
    raise PipelineError(
        f"{command!r} does not map to pipeline stages. `extract` runs the §17 extraction routine "
        "through serving/pipeline.py, not the build DAG."
    )


@dataclass
class MultiScopeReport:
    """One shared corpus build, then one report per scope.

    Kept as its own type rather than a merged RunReport: each scope produces an
    independent adapter, gate verdict and release, and flattening them would make
    "did it pass" unanswerable for any one of them.
    """

    command: str
    version: str
    shared: RunReport | None = None
    by_scope: dict[str, RunReport] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        reports = ([self.shared] if self.shared else []) + list(self.by_scope.values())
        return all(r.ok for r in reports)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def render(self) -> str:
        lines = [f"{self.command} {self.version}: {'OK' if self.ok else 'FAILED'}"]
        if self.shared:
            lines.append(self.shared.render())
        for name, report in self.by_scope.items():
            lines.append(f"-- scope {name} --")
            lines.append(report.render())
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "version": self.version,
            "ok": self.ok,
            "shared": self.shared.as_dict() if self.shared else None,
            "by_scope": {name: r.as_dict() for name, r in self.by_scope.items()},
        }


def run_multi_scope(
    build_context: Callable[[Scope], StageContext],
    scopes: Sequence[Scope],
    *,
    command: str = "all",
    from_stage: str | None = None,
) -> MultiScopeReport:
    """Build the corpus once, then run stages 5-11 once per scope.

    Each scope gets its own ``StageContext`` — its own release id, checkpoints,
    calibrators and manifests — because they are independent runs that happen to
    share a corpus.

    **A block in one scope does not stop the others.** They are separate
    adapters: a policy regression says nothing about the lossrun model, and
    stopping the second run would only mean re-running the first. The aggregate
    exit code is still non-zero, so a pipeline never mistakes "one passed" for
    "all passed".
    """
    if not scopes:
        raise PipelineError("no scope to run; name at least one with --scope")

    first = build_context(scopes[0])
    report = MultiScopeReport(command=command, version=first.out_version)

    shared = [s for s in _selected(command, from_stage) if s in SHARED_STAGES]
    if shared:
        # Over the union of every scope's types, so one build serves them all.
        first.doc_types = sorted({dt for scope in scopes for dt in scope.doc_types})
        report.shared = run_stages(first, tuple(shared), command=command)
        if not report.shared.ok:
            return report

    per_scope = [s for s in _selected(command, from_stage) if s in PER_SCOPE_STAGES]
    for scope in scopes:
        ctx = first if scope is scopes[0] else build_context(scope)
        ctx.scope = scope
        report.by_scope[scope.name] = run_stages(ctx, tuple(per_scope), command=command)
    return report


def _selected(command: str, from_stage: str | None) -> tuple[Stage, ...]:
    return stages_from(from_stage, command) if from_stage else stages_for(command)


def stages_from(stage_name: str, command: str = "finetune") -> tuple[Stage, ...]:
    """Resume mid-pipeline: the named stage and everything after it."""
    selected = stages_for(command)
    names = [s.name for s in selected]
    if stage_name not in names:
        raise PipelineError(
            f"unknown stage {stage_name!r} for {command!r}; expected one of {names}"
        )
    return selected[names.index(stage_name):]


def _already_complete(stage: Stage, ctx: StageContext) -> bool:
    if stage.is_complete is None:
        return False
    try:
        return stage.is_complete(ctx)
    except Exception as exc:  # noqa: BLE001 - an unreadable check means "not complete"
        log.debug("completion check for %s failed (%s); running the stage", stage.name, exc)
        return False


def run_stages(ctx: StageContext, stages: Sequence[Stage], *, command: str = "finetune") -> RunReport:
    """Execute stages in order, stopping at the first block or failure.

    A completed stage is **skipped**, not re-run. That is what makes
    ``--from-stage`` usable after a mid-pipeline failure without wondering
    whether it will duplicate the work that already succeeded.

    A *failed* stage is retried up to ``ctx.max_attempts`` — pods and object
    stores fail transiently. A **blocked** gate is never retried: a retry loop
    around the gate would be an override path with extra steps.
    """
    # `package` operates on what `finetune` staged, so a reclaimed volume is a
    # precondition failure for the whole command. Checked here rather than inside
    # a stage: --skip-quantize turned the first stage into a no-op, and the check
    # disappeared with it.
    report = RunReport(command=command, version=ctx.out_version)
    preconditions = []
    if command == "package" and stages:
        preconditions += [resolve_corpus_version, assert_staged]
    if any(stage.command == "package" for stage in stages):
        # Before any work, including a whole `all` training run: an invalid id
        # used to surface only when calibrate tried to SAVE, after fitting.
        preconditions.append(assert_release_id)
    for precondition in preconditions:
        try:
            precondition(ctx)
        except PipelineError as exc:
            # Reported, not raised: the operator should get the same rendered
            # remediation as any other failure rather than a traceback. Attached
            # to the first stage, because that is where the work would have
            # started and where `--from-stage` will resume.
            result = StageResult(stages[0].name, "failed", str(exc))
            ctx.results[stages[0].name] = result
            report.results.append(result)
            report.failed_at = stages[0].name
            log.error("%s", exc)
            return report

    from common.logging_setup import reassert

    for stage in stages:
        # A library imported by the stage before (ms-swift, vLLM) may have
        # silenced our log lines; every stage starts with them back on.
        reassert()
        if _already_complete(stage, ctx):
            result = StageResult(stage.name, "skipped", "already complete — no-op")
            ctx.results[stage.name] = result
            report.results.append(result)
            log.info("stage %d %s: already complete, skipping", stage.number, stage.name)
            continue

        attempts = 0
        began = time.monotonic()
        while True:
            attempts += 1
            log.info("stage %d %s%s%s", stage.number, stage.name,
                     " [GPU]" if stage.gpu else "",
                     f" (attempt {attempts})" if attempts > 1 else "")
            try:
                result = stage.run(ctx)
                break
            except GateBlocked as blocked:
                result = _timed(StageResult(
                    stage.name, "blocked", str(blocked),
                    {"failed_gates": list(getattr(blocked.gate_result, "failed_gates", []))},
                ), began)
                ctx.results[stage.name] = result
                report.results.append(result)
                report.blocked_at = stage.name
                report.unlabeled_backlog = dict(ctx.unlabeled_backlog)
                log.error("%s", blocked)
                _write_progress(report, ctx)
                return report
            except (PipelineError, paths.PathError) as exc:
                # Deterministic: the stage cannot run at all with these inputs,
                # so a second attempt re-does every listing and fails with the
                # identical message after an entirely pointless backoff. Retries
                # are for transient faults — a throttled Blob read, a pod that
                # dropped — not for "below the day-zero floor" or a malformed
                # --out-version.
                result = _timed(StageResult(stage.name, "failed", str(exc)), began)
                ctx.results[stage.name] = result
                report.results.append(result)
                report.failed_at = stage.name
                log.error("stage %s cannot run: %s (not retried)", stage.name, exc)
                _write_progress(report, ctx)
                return report
            except Exception as exc:  # noqa: BLE001 - retried, then recorded and stops the run
                if attempts < max(1, ctx.max_attempts) and not isinstance(
                    exc, _deterministic_errors()
                ):
                    # Configured in pipeline.yaml and previously read by nothing,
                    # so both attempts fired within milliseconds — inside the same
                    # throttle window that caused the first failure.
                    delay = ctx.retry_backoff_seconds * attempts
                    log.warning("stage %s failed (%s); retrying in %ss", stage.name, exc, delay)
                    if delay:
                        time.sleep(delay)
                    continue
                result = _timed(StageResult(stage.name, "failed", str(exc)), began)
                ctx.results[stage.name] = result
                report.results.append(result)
                report.failed_at = stage.name
                report.unlabeled_backlog = dict(ctx.unlabeled_backlog)
                log.error("stage %s failed after %d attempt(s): %s", stage.name, attempts, exc)
                _write_progress(report, ctx)
                return report

        result = _timed(result, began)
        ctx.results[stage.name] = result
        report.results.append(result)
        log.info("stage %d %s: %s in %s", stage.number, stage.name, result.status,
                 _duration(result.data["wall_seconds"]))
        _write_progress(report, ctx)

    report.unlabeled_backlog = dict(ctx.unlabeled_backlog)
    return report


def _timed(result: StageResult, began: float) -> StageResult:
    """``result`` with the stage's wall time since ``began`` - every attempt
    included - in its data: what a run's time went on, measured, not guessed."""
    result.data["wall_seconds"] = round(time.monotonic() - began, 1)
    return result


def _duration(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {rest:02d}s" if hours else f"{minutes}m {rest:02d}s"


#: Where the run report is written after every stage, on the pod's volume.
PROGRESS_DIR_ENV = "FIDEON_LOG_DIR"


def progress_report_name(report: RunReport, ctx: StageContext) -> str:
    """``run-<command>-<scope>-<tenant>-<version>.json``: two scopes or tenants
    trained under one version tag (CGL v1, personal-lines v1) keep their own."""
    scope = ctx.scope.name if getattr(ctx, "scope", None) is not None else "unified"
    return f"run-{report.command}-{scope}-{paths._tenant(ctx.tenant_id)}-{report.version}.json"


def _write_progress(report: RunReport, ctx: StageContext) -> None:
    """The run report so far, written after every stage to
    ``$FIDEON_LOG_DIR`` (default /workspace/logs) when that folder exists: a
    run that dies part-way still says what each finished stage took."""
    folder = Path(os.environ.get(PROGRESS_DIR_ENV) or "/workspace/logs")
    if not folder.is_dir():
        return
    target = folder / progress_report_name(report, ctx)
    try:
        target.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")
    except OSError as exc:  # a report that cannot be written never stops the run
        log.warning("run report not written to %s: %s", target, exc)
