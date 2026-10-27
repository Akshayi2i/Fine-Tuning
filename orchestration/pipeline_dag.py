"""The 11 pipeline stages as addressable functions (IMPL-13 §8, arch §13).

``run.py`` composes these; an Airflow DAG or a GitHub Actions workflow could
schedule the same functions unchanged. Keeping them here rather than in the CLI
is what makes ``--from-stage`` a slice of a list instead of a special case in an
argument parser.

Four properties are requirements, not aspirations:

* **Every stage reads and writes Blob or the staging volume, never pod-local disk.**
  A pod is ephemeral; anything written to its own disk is gone at teardown.
* **Every training and evaluation job writes a run manifest.**
* **Stages are idempotent and resumable.** Re-running a completed stage is a
  no-op, not a duplicate — which is what makes ``--from-stage`` safe to reach for
  after a mid-pipeline failure.
* **The gate (stage 8) is a hard stop.** A candidate that regresses any gating
  metric is not packaged, and there is no way round it but a named, written override of the
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
from common.adapter import ADAPTER
from common.constants import ACTIVE_DOC_TYPES, DAY_ZERO_MIN_LABELS_PER_TYPE
from common.run_ids import is_valid_run_id, version_of
from orchestration.runpod_controller import RunPodController, StagingVolume
from orchestration.settings import gpu_class_for

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
    #: The types the corpus is built over (stages 1-4): the adapter's
    #: (``common.adapter``), policies.
    doc_types: list[str] = field(default_factory=lambda: list(ADAPTER.doc_types))
    tenant_id: str | None = None
    #: Serving formats. bf16 only: the release is its adapter, served as a LoRA
    #: on the bf16 base, and nothing is merged or quantized (decision 2026-10-23).
    formats: list[str] = field(default_factory=lambda: ["bf16"])
    #: Start from the base model, not the promoted adapter (``--fresh``): after
    #: a data deletion, or to reset drift (decision 2026-10-23).
    fresh: bool = False
    #: Overrides the configured class for the training pod. ``None`` means the
    #: class in ``config/pipeline.yaml``.
    gpu_class: str | None = None

    # -- flags -------------------------------------------------------------
    dry_run: bool = True
    skip_ingest: bool = False
    #: Documents the OCR stage reads at once against the served MinerU
    #: (``run_mineru.process_batch``). 1 reads one at a time, as before.
    ocr_workers: int = 1
    #: Train on a working tree with uncommitted changes (``--allow-dirty``): a
    #: real training run refuses one otherwise (``training.train.assert_clean_tree``).
    allow_dirty: bool = False
    train_vit: bool = False
    min_labels_per_type: int = DAY_ZERO_MIN_LABELS_PER_TYPE
    push_adapters: bool = False
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
    #: Per-adapter re-validation results for a Foundation major bump: every
    #: dependent adapter retrained against the new Foundation and gated. Absent
    #: evidence blocks promotion rather than deferring the question (arch §12).
    #: Per-doc-type "has this adapter been revalidated against the new
    #: Foundation" flags, for the arch §12 cascade block.
    revalidation_evidence: dict[str, bool] | None = None

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

    #: ``{serving_format: {"calibration": [(features, correct)], "threshold": [...]}}``
    #: from the two halves of the validation split (arch v2.1 §8.2). Keyed by the
    #: exact format: there is no wildcard, because a calibrator fitted on bf16
    #: logprobs reports confidence for a distribution FP8 does not produce.
    calibration_samples: dict[str, Any] = field(default_factory=dict)

    #: Family grouping evidence per doc type, from the dataset build (arch v2.1 §8.2).
    grouping: dict[str, Any] = field(default_factory=dict)
    #: ``(MinerU version, tier) -> documents`` the dataset build loaded, held to
    #: the served pin before a corpus is planned (:func:`assert_corpus_ocr_pin`).
    ocr_readings: dict[tuple[str, str | None], int] = field(default_factory=dict)
    #: ``ocr_meta.json`` bodies by key, read once per run by the OCR stage's
    #: completion check and the stage (``run_mineru.find_unprocessed``); cleared
    #: once the stage has written new ones.
    ocr_meta_cache: dict[str, Any] = field(default_factory=dict)

    # -- fitted during the run ---------------------------------------------
    calibrators: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)

    # -- accumulated state -------------------------------------------------
    results: dict[str, StageResult] = field(default_factory=dict)
    manifests: dict[str, Any] = field(default_factory=dict)
    unlabeled_backlog: dict[str, list[str]] = field(default_factory=dict)
    #: Set by :func:`continuation` when the cycle continues a parent whose eval
    #: report was on another yardstick only because the scoring code or
    #: settings moved, and was re-scored from its stored answers (decision
    #: 2026-10-26): ``{"run_id", "difference", "reason", "rescored_from"}``.
    parent_report_rescored: dict[str, Any] | None = None

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

    @property
    def release_id(self) -> str:
        """The release this cycle produces: its version (decision 2026-10-22).
        Everything gated, promoted and served is addressed by it (arch v2.1 §12.3)."""
        return self.out_version


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
        # What to run next, as a stage names it (package: shadow, then promote).
        steps = [step for r in self.results for step in (r.data or {}).get("next_steps") or []]
        if steps and self.ok:
            lines.append("  next steps:")
            lines += [f"    {step}" for step in steps]
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
    others = sorted(dt for dt in ACTIVE_DOC_TYPES
                    if dt not in ctx.doc_types and (ctx.input_dir / dt).is_dir())
    if others and not typed:
        # Read flat, the other types' folders would be stored as policies, for
        # good: the raw layer is write-once.
        raise PipelineError(
            f"{ctx.input_dir} holds {', '.join(o + '/' for o in others)} but no "
            f"{'/, '.join(ctx.doc_types)}/ folder: read as one folder, every PDF under it, "
            "theirs too, would be stored as a policy. Put the policies in "
            f"{ctx.input_dir.name}/policy/, or ingest a folder of policies only."
        )
    if others:
        log.info("%s under %s: not ingested, the adapter trains %s only",
                 ", ".join(o + "/" for o in others), ctx.input_dir, list(ctx.doc_types))
    if not typed and len(ctx.doc_types) > 1:
        raise PipelineError(
            f"{ctx.input_dir} has no {'/, '.join(ctx.doc_types)}/ subfolder, so nothing says "
            "which document type each PDF is. Put each type's PDFs in a folder named after it "
            f"({ctx.input_dir.name}/policy/...)."
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


def _ocr_pin_for(engine: Any) -> tuple[str | None, str | None]:
    """``(MinerU version, tier)`` a stored reading must match to count as read.

    A served engine's own. With none (``MINERU_URL`` unset), the served pin in
    ``configs/shared/ocr.yaml`` all the same: a document another MinerU read -
    the in-process 3.4.5 - is still unread, and is never built into a corpus a
    release would be refused for (:func:`assert_corpus_ocr_pin`). An injected
    engine that says nothing of itself (a test's) holds a reading to its
    existence alone.
    """
    if engine is None:
        from common.config import served_mineru

        pin = served_mineru()
        return pin["version"], pin["tier"]
    if hasattr(engine, "environment"):
        return engine.environment().mineru_version, getattr(engine, "tier", None)
    return None, None


def _pending_ocr(ctx: StageContext, engine: Any) -> dict[str, list[str]]:
    """``{doc type: source ids}`` not yet read as :func:`_ocr_pin_for` asks.

    Every ``ocr_meta.json`` is read once per run, not once per question
    (``ctx.ocr_meta_cache``): the completion check and the stage itself ask in turn."""
    from data_pipeline.ocr.run_mineru import find_unprocessed

    version, tier = _ocr_pin_for(engine)
    pending = {}
    for doc_type in ctx.doc_types:
        ids = find_unprocessed(ctx.raw, doc_type, ctx.tenant_id, mineru_version=version,
                               mineru_tier=tier, meta_cache=getattr(ctx, "ocr_meta_cache", None))
        if ids:
            pending[doc_type] = ids
    return pending


def _is_preprocessed(ctx: StageContext) -> bool:
    from data_pipeline.ocr.served_ocr import ServedMinerUEngine

    # A document another MinerU read still needs reading, by the served one.
    return not _pending_ocr(ctx, ctx.ocr_engine or ServedMinerUEngine.from_environment())


#: The OCR environment on the pod. MinerU 1.x cannot share an environment with
#: vLLM 0.11 / torch 2.8, so OCR has its own (scripts/pod_bootstrap.sh).
OCR_VENV = "/workspace/venv-ocr"


def _assert_ocr_runnable(ctx: StageContext, pending: dict[str, list[str]]) -> None:
    """Refuse clearly when documents still need OCR and no served MinerU is named.

    The corpus is read by the served MinerU (``MINERU_URL``, the pin in
    ``configs/shared/ocr.yaml``), from anywhere that reaches it. Without one,
    only a MinerU installed here at the pinned version may read: another one -
    the OCR environment's 3.4.5 - would write a corpus serving refuses.
    """
    import importlib.util

    from common.config import served_mineru

    if not pending:
        return
    pin = served_mineru()
    counts = ", ".join(f"{dt}: {len(ids)}" for dt, ids in pending.items())
    if importlib.util.find_spec("mineru") is not None:
        from data_pipeline.ocr.mineru_version import get_mineru_version

        local = get_mineru_version()
        if local == pin["version"]:
            return
        why = (f"the MinerU installed here is {local}, not the served pin {pin['version']} "
               f"(tier {pin['tier']}): a corpus it read would train a release serving refuses")
    else:
        why = "no served MinerU is named (MINERU_URL) and none is installed here"
    raise PipelineError(
        f"documents still need reading by the served MinerU ({counts}), and {why}. Set MINERU_URL to "
        f"the served MinerU {pin['version']}/{pin['tier']} (MINERU_TOKEN its token) and re-run, or read "
        f"them first: {ocr_remediation(list(pending), ctx.tenant_id)}. Preprocessing then finds nothing "
        "left to do."
    )


def stage_preprocessing(ctx: StageContext) -> StageResult:
    """MinerU OCR plus page rendering at the resolution cap.

    Through the served MinerU when ``$MINERU_URL`` names one (``configs/shared/ocr.yaml``),
    ``ctx.ocr_workers`` documents at once: it runs anywhere that reaches it.
    Otherwise a MinerU installed here at the pinned version, one document at a time.
    """
    from data_pipeline.ocr.run_mineru import MinerUEngine, process_batch
    from data_pipeline.ocr.served_ocr import ServedMinerUEngine

    engine = ctx.ocr_engine or ServedMinerUEngine.from_environment()
    processed, skipped, failed = [], [], []
    try:
        # A served engine also re-reads what another MinerU read: one corpus, one MinerU.
        pending = _pending_ocr(ctx, engine)
        if engine is None and pending:
            _assert_ocr_runnable(ctx, pending)
            engine = MinerUEngine()
        # Concurrently against a served MinerU only: the in-process one holds the GPU.
        workers = max(1, int(getattr(ctx, "ocr_workers", 1) or 1)) if hasattr(engine, "environment") else 1
        for doc_type, ids in pending.items():
            outcome = process_batch(ids, doc_type, ctx.raw, engine, tenant_id=ctx.tenant_id, workers=workers)
            processed += outcome["processed"]
            skipped += outcome["skipped"]
            failed += [f"{sid}: {reason}" for sid, reason in outcome["failed"]]
    finally:
        # The readings just written are not the ones cached.
        if getattr(ctx, "ocr_meta_cache", None) is not None:
            ctx.ocr_meta_cache.clear()

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
    #: (MinerU version, tier) -> documents: held to the served pin (plan_corpus).
    readings: dict[tuple[str, str | None], int] = {}
    #: Each document's page-text keys, read together below.
    page_keys: list[list[str]] = []
    unreviewed: list[tuple[str, str]] = []
    #: Documents read with the other read_every_page_by_ocr than configs/shared/ocr.yaml's.
    from common.config import ocr_every_page
    from data_pipeline.ocr.run_mineru import read_as_configured

    every_page = ocr_every_page()
    other_setting: list[str] = []
    for doc_type in ctx.doc_types:
        for source_id in list_labeled_source_ids(ctx.client, doc_type, ctx.tenant_id):
            label, metadata = load_golden_label(source_id, doc_type, ctx.client, ctx.tenant_id)
            # A gold with no review record never trains (decision 2026-10-22).
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
            reading = (version, ocr_meta.get("mineru_tier"))
            readings[reading] = readings.get(reading, 0) + 1
            if not read_as_configured(ocr_meta, every_page):
                other_setting.append(source_id)
            page_keys.append([paths.processed_page(doc_type, source_id, page, "md", ctx.tenant_id)
                              for page in range(1, page_count + 1)])
            documents.append(SourceDocument(
                source_id=source_id,
                doc_type=doc_type,
                golden_label=label,
                # Per page, never joined: the pairing with each page image is
                # what tells the model which text it is looking at, and joining
                # also split every table that crossed a page boundary. Read
                # below, every document's pages at once (read_page_texts).
                ocr_pages=[],
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
    if other_setting:
        raise PipelineError(
            f"{len(other_setting)} labelled document(s) were read with read_every_page_by_ocr "
            f"{not every_page}, and configs/shared/ocr.yaml reads with {every_page}: a release trained "
            f"on them would be refused by serving at start-up ({other_setting[:5]}). Read them again: "
            f"{ocr_remediation()} --force"
        )
    if unreviewed:
        raise PipelineError(
            f"{len(unreviewed)} labelled document(s) have no review record, so they cannot train "
            f"(decision 2026-10-22): {unreviewed[0][0]}: {unreviewed[0][1]}. Others: "
            f"{[s for s, _ in unreviewed[1:6]]}. Have a person review them, record reviewer_id and "
            "review_date in metadata.json (a twin inherits its seed's through prepare_bundles), "
            "and re-import."
        )
    assert_one_mineru_version(read_by)
    for document, texts in zip(documents, read_page_texts(ctx.client, page_keys), strict=True):
        document.ocr_pages = texts
    ctx.ocr_readings = readings
    ctx.grouping = assign_document_groups(ctx, documents)
    return documents


#: Page texts the corpus build reads at once. One Blob GET per page: a corpus of
#: 20,000 pages read one after another was minutes of a paid pod doing nothing.
PAGE_READ_WORKERS = 16


def read_page_texts(client: BlobClient, keys_per_document: Sequence[Sequence[str]],
                    workers: int = PAGE_READ_WORKERS) -> list[list[str]]:
    """Every document's page texts, in its page order: ``keys_per_document``
    read through a pool of ``workers`` threads. The same texts in the same
    order as reading them one by one; the first read that fails raises."""
    from concurrent.futures import ThreadPoolExecutor

    flat = [key for keys in keys_per_document for key in keys]
    if workers <= 1 or len(flat) <= 1:
        texts = [client.read_text(key) for key in flat]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="page-text") as pool:
            texts = list(pool.map(client.read_text, flat))
    out, at = [], 0
    for keys in keys_per_document:
        out.append(texts[at:at + len(keys)])
        at += len(keys)
    return out


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
            f"the documents were read by more than one MinerU: {counts}. Read them again with one, "
            f"the served pin: {ocr_remediation()}"
        )


def ocr_remediation(doc_types: Sequence[str] = ("policy",), tenant_id: str | None = None) -> str:
    """The command that reads every document not yet read by the served MinerU
    (``configs/shared/ocr.yaml``) - those never OCR'd and those another MinerU,
    or another tier, read."""
    from common.config import served_mineru

    pin = served_mineru()
    tenant = f" --tenant {tenant_id}" if tenant_id else ""
    commands = "; ".join(
        f"python -m data_pipeline.ocr.run_mineru --doc-type {doc_type} --all-unprocessed "
        f"--mineru-url \"$MINERU_URL\" --workers 4{tenant}" for doc_type in doc_types)
    return (f"{commands} - with MINERU_URL the served MinerU {pin['version']} (tier {pin['tier']}, "
            "configs/shared/ocr.yaml); it needs no GPU or MinerU where it runs")


def assert_corpus_ocr_pin(readings: dict[tuple[str, str | None], int]) -> None:
    """Refuse a corpus any document of which another MinerU read than the served one.

    Serving reads every page with the pinned MinerU (``common.config.served_mineru``)
    and the Azure worker refuses, at start-up, a release whose corpus another
    version read (``scripts.served_adapter.assert_ocr_pin``): trained here, gated
    and promoted, such a corpus is an outage. Held as
    :func:`registry_utils.releases.ocr_pin_refusal` holds a release - the version,
    and the tier where the reading records one. ``readings``: ``(version, tier)
    -> documents`` (:func:`load_labeled_documents`).
    """
    from registry_utils.releases import ocr_pin_refusal

    off = {reading: n for reading, n in readings.items()
           if ocr_pin_refusal({"mineru_version": None if reading[0] == "unknown" else reading[0],
                               "mineru_tier": reading[1]})}
    if not off:
        return
    from common.config import served_mineru

    pin = served_mineru()
    counts = ", ".join(f"{version}{f'/{tier}' if tier else ''} ({n})"
                       for (version, tier), n in sorted(off.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")))
    raise PipelineError(
        f"{sum(off.values())} labelled document(s) were read by another MinerU than the served one, "
        f"{pin['version']} (tier {pin['tier']}): {counts}. A release trained on them would be refused "
        f"by serving at start-up. Read them with the served MinerU first: {ocr_remediation()}"
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
                holds_real=not document.synthetic,
                # The line the split balances on. A family is one line in
                # practice (one insured's policy renewed); the first member's
                # stands for it.
                line=line_of(getattr(document, "lob", None)),
            )
        else:
            record.source_ids.append(document.source_id)
            if not document.synthetic and not record.holds_real:
                # The real member's carrier names the family: a twin's may be
                # generated, and the carrier hold-out counts real carriers.
                record.carrier = document.carrier
                record.holds_real = True
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

    Held out by the split, or by the delivery (the freeze records a carrier its
    documents are marked with, ``held_out_carrier``). A line's entry is one
    carrier, or a list where more than one is held out of it. Both sides are
    compared as :func:`common.normalize.normalize_carrier` keys, so a carrier
    recorded under an older key ("dryden_mutual") still matches its documents.
    """
    from common.normalize import normalize_carrier
    from data_pipeline.dataset_builder.split_groups import line_of

    def keys(carriers: Any) -> set[str]:
        listed = [carriers] if isinstance(carriers, str) else list(carriers or ())
        return {key for c in listed if isinstance(c, str) and (key := normalize_carrier(c))}

    held = keys([c for carriers in (manifest.get("held_out_carriers") or {}).values() for c in carriers])
    by_line = manifest.get("held_out_carriers_by_line") or {}
    seen = sorted({d.carrier for d in documents if keys(d.carrier) & held} | {
        f"{d.carrier} ({line_of(getattr(d, 'lob', None))})" for d in documents
        if keys(d.carrier) & keys((by_line.get(d.doc_type) or {}).get(line_of(getattr(d, "lob", None))))
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
    # Before anything is split or trained: a corpus another MinerU read trains a
    # release the Azure worker refuses at start-up.
    assert_corpus_ocr_pin(ctx.ocr_readings)

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
    # A cycle that will continue the promoted adapter (stage_training asks the
    # same question): every document its chain trained on stays in train, or
    # the continued adapter is validated on what it memorised. continuation()
    # refuses to continue a chain whose documents cannot be read.
    parent, _fresh = continuation(ctx)
    continues = ADAPTER.run_id(parent) if parent is not None else None
    trained = lineage_training_documents(ctx, continues)[0] if continues else set()
    if delivered:
        from data_pipeline.dataset_builder.split_groups import SplitError, assign_delivered_splits

        try:
            assignment = assign_delivered_splits(
                by_type, delivered_split_of(documents), seed=ctx.seed, with_test=not frozen,
                trained=trained,
            )
        except SplitError as exc:
            raise PipelineError(str(exc)) from exc
        assignment.held_out_source_ids.extend(delivered_held_out_ids(documents, assignment))
        log.info("using the delivered split: %s", assignment.counts_by_doc_type)
    else:
        assignment = assign_group_splits(by_type, seed=ctx.seed, with_test=not frozen,
                                         trained=trained)
    assignment.continues = continues
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
    # The global input-mode mix unless the training config sets its own
    # (training.data_mix); recorded on the corpus manifest, so a run that
    # wants another mix is refused rather than mislabelled.
    mix = modality_mix_of(ctx)
    modes = sample_modes(train_source_ids(documents, assignment), seed=ctx.seed, mix=mix)
    assert_mix_is_close(modes, mix=mix)
    built = build_corpus(documents, assignment, seed=ctx.seed, mode_assignment=modes)
    return CorpusPlan(documents, by_type, assignment, built, frozen, delivered, modes)


def modality_mix_of(ctx: StageContext) -> dict[str, float]:
    """The input-mode mix this build draws: the training config's, else the global default."""
    from training.data_mix import DataMixError, configured_modality_mix

    try:
        return configured_modality_mix()
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
        # What building the targets did (verification, placed pages, halves,
        # windows dropped): all zeros in the manifest until passed.
        target_counts=dict(built.target_counts),
    )
    # The served MinerU's tier beside its version: both are the OCR pin a release
    # carries (registry_utils.releases.ocr_pin_refusal).
    if ocr_environment.get("mineru_tier"):
        manifest["mineru_tier"] = ocr_environment["mineru_tier"]
    # And whether every page was read by OCR (configs/shared/ocr.yaml): every
    # document's reading was held to the setting (load_labeled_documents refuses
    # one read otherwise), so it is the corpus's - not the first document's
    # record alone, which a reading from before the setting does not carry.
    from common.config import ocr_every_page

    manifest["read_every_page_by_ocr"] = ocr_every_page()
    # Each kept gold's content as stored, so a later cycle can tell whether a gold
    # this corpus trained on has changed since (continued_training.fresh_when_golds_changed).
    manifest["gold_sha256"] = stored_gold_hashes(
        ctx.client, {d.source_id: d.doc_type for d in documents if d.source_id in kept}, ctx.tenant_id)
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

    adapter_dir = paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version)
    if os.path.isdir(adapter_dir):
        return training_completed(adapter_dir)
    return ctx.volume.exists(adapter_dir)


def _assert_corpus_pins(corpus_manifest: dict[str, Any], what: str) -> None:
    """Refuse a corpus this code would not build today.

    - **Its golds' review rule.** A corpus built before every gold needed a
      review record skips the dataset build's check when reused
      (``--corpus-version``), so its unreviewed golds would train.
    - **A policy corpus's window plan.** The prompt hash cannot catch it: it is
      computed when the release is packaged, from the code doing the
      packaging, so an adapter trained on another window plan carried this
      code's hash. Serving would then read the adapter windows it was never
      shown (``policy_windows.assert_window_plan``).
    """
    from data_pipeline.corpus_manifest import REVIEW_RULE_VERSION
    from data_pipeline.dataset_builder.policy_windows import assert_window_plan

    rule = corpus_manifest.get("review_rule_version")
    if rule != REVIEW_RULE_VERSION:
        raise PipelineError(
            f"{what} rests on a corpus built under review rule {rule or 'unrecorded'}, and every "
            f"gold must now pass rule {REVIEW_RULE_VERSION}: a person's review record on every "
            "gold, a twin its seed's (decision 2026-10-22). Rebuild the corpus, so the dataset "
            "build checks each gold, and retrain."
        )
    if "policy" not in (corpus_manifest.get("doc_types") or []):
        return
    try:
        assert_window_plan(corpus_manifest, what=what)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc


def staging_parent_adapter_dir(version: str) -> str:
    """Where cycle ``version`` keeps the adapter it continues, on the staging volume."""
    return paths.staging_parent_adapter_dir(ADAPTER.name, version)


def continuation(ctx: StageContext) -> tuple[str | None, str | None]:
    """``(parent version, None)`` when this cycle continues the promoted adapter;
    ``(None, why)`` when it starts from the base model (decision 2026-10-23).

    A cycle continues the tenant's promoted release - its adapter's weights,
    trained further on new and existing data, on the same base model. It starts
    from the base instead when asked (``--fresh``: after a data deletion, or to
    reset drift), when nothing is promoted, and whenever continuing would teach
    the adapter on top of what it can no longer use:

    - its run manifest or its adapter in Blob cannot be read;
    - it was trained on another base model;
    - it has another LoRA rank, alpha or set of target modules (the weights
      would not fit), or was saved with another dropout, rsLoRA or bias
      setting (ms-swift would keep its own) - a dropout only when the
      config gives a number: ``dropout: auto`` keeps the parent's;
    - it was packaged with another prompt hash or window plan: a template,
      schema or section-map change, so its outputs have another shape;
    - it has no eval report, or one on another yardstick (the set frozen
      again, or another tenant's) that its stored answers cannot be re-scored
      onto: the gate holds a continued run to its parent line by line, and
      would block this cycle only after training it.

    A parent's report on another yardstick only because the scoring code or
    settings moved is re-scored from its stored answers by the code running
    now (``evaluation.rescore``, decision 2026-10-26; the old model is never
    run again) and the cycle continues, recording it on the context
    (``parent_report_rescored``). A parent scored before answers were stored
    has none to re-score, and the cycle starts from the base as before.
    """
    from artifact_registry.transfer import ADAPTER_FILES
    from common.config import base_model_config, training_config
    from common.prompts import prompt_hash
    from data_pipeline.dataset_builder.policy_windows import window_plan_id
    from evaluation.rescore import rescore_report
    from evaluation.run_eval import eval_set_record, yardstick_difference
    from registry_utils.query_registry import RegistryQueryError
    from registry_utils.query_registry import get as get_manifest
    from training.train import adapter_config_mismatch

    ctx.parent_report_rescored = None
    if ctx.fresh:
        return None, "asked for (--fresh)"
    promoted = promoted_release(ctx)
    if promoted is None:
        return None, "no release of the policy adapter is promoted for this tenant"
    release_id, parent = promoted
    run_id = ADAPTER.run_id(parent)
    try:
        manifest = get_manifest(run_id, ctx.client)
    except (RegistryQueryError, KeyError, FileNotFoundError, ValueError):
        # ValueError: not JSON, or a manifest this build cannot validate.
        return None, f"the promoted release's run manifest ({run_id}) cannot be read"
    base = base_model_config()["model"]
    wanted = f"{base['model_id']}@{base['revision']}"
    if manifest.dependencies.base_model != wanted:
        return None, f"{run_id} trained on base {manifest.dependencies.base_model}, this cycle on {wanted}"
    cfg = training_config(ADAPTER.training_config)
    lora = cfg["lora"]
    modules = list(lora["target_modules"]) + (["merger"] if lora.get("include_vision_projector") else [])
    recorded = manifest.training_config
    for name, now, then in (("LoRA rank", lora["rank"], recorded.lora_rank),
                            ("LoRA alpha", lora["alpha"], recorded.lora_alpha),
                            ("target modules", sorted(modules), sorted(recorded.target_modules))):
        if now != then:
            return None, f"{run_id} has {name} {then}, this cycle's config {now}"
    adapter_dir = paths.scoped_adapter_dir(ADAPTER.name, parent)
    missing = [name for name in ADAPTER_FILES if not ctx.client.exists(f"{adapter_dir}/{name}")]
    if missing:
        return None, f"{run_id}'s adapter is not in Blob at {adapter_dir} (no {', '.join(missing)})"
    try:
        saved = ctx.client.read_json(f"{adapter_dir}/adapter_config.json")
    except ValueError as exc:  # not JSON
        return None, f"{run_id}'s adapter_config.json cannot be read ({type(exc).__name__})"
    # ms-swift continues an adapter with the settings it was SAVED with, not the
    # config's (training.train.adapter_config_mismatch): a changed dropout or
    # rsLoRA would never apply, while the manifest recorded it. Under
    # `dropout: auto` the parent's dropout is the one wanted (resolve_lora_dropout).
    differs = adapter_config_mismatch(saved, lora)
    if differs:
        return None, (f"{run_id}'s adapter was trained with other LoRA settings ({differs}); "
                      "continued, it would train with its own")
    bundle_key = paths.release_bundle(release_id, ctx.tenant_id)
    if not ctx.client.exists(bundle_key):
        return None, f"the promoted release {release_id} has no bundle at {bundle_key}"
    bundle = ctx.client.read_json(bundle_key)
    if bundle.get("prompt_hash") != prompt_hash():
        return None, (f"{parent} was packaged with another prompt hash: a template, schema or "
                      "section-map change since, so its outputs have another shape")
    if bundle.get("window_plan_version") != window_plan_id():
        return None, f"{parent} was trained on policies cut with another window plan"
    report_key = paths.eval_report(parent, scope=ADAPTER.name)
    if not ctx.client.exists(report_key):
        return None, (f"{run_id} has no eval report at {report_key}, so the gate could not "
                      "compare this cycle with it line by line")
    current = eval_set_record(ctx.client, ctx.tenant_id, [])
    different = yardstick_difference(ctx.client.read_json(report_key).get("eval_set"), current,
                                     ctx.tenant_id)
    rescored_from: dict[str, Any] | None = None
    if different:
        # Other scoring code or settings alone: its stored answers re-scored
        # by the code running now. Another frozen set or tenant, or answers
        # missing or incomplete, are refused, and the cycle starts fresh.
        rescored, why = rescore_report(ctx.client, parent, ctx.tenant_id, eval_set=current)
        if rescored is None:
            return None, (f"{run_id}'s eval report is on another yardstick, so the gate could not "
                          f"compare this cycle with it line by line: {different}. {why}")
        rescored_from = {"run_id": run_id, "difference": different, "reason": why,
                         "rescored_from": rescored.get("rescored_from")}
    trained, unread = lineage_training_documents(ctx, run_id)
    if unread:
        return None, (f"the documents {run_id}'s chain of runs trained on cannot be read ({unread}), "
                      "so the split could not keep them out of validation, where the continued "
                      "adapter would be scored on documents it memorised")
    # The operator's own rules (pipeline.yaml continued_training, all off by default).
    if why := fresh_by_rule(ctx, run_id, trained):
        return None, why
    if rescored_from is not None:
        log.info("%s continues %s, whose eval report was re-scored: %s", ctx.out_version, run_id,
                 rescored_from["reason"])
    ctx.parent_report_rescored = rescored_from
    return parent, None


def lineage_training_documents(ctx: StageContext, run_id: str) -> tuple[set[str], str | None]:
    """The source ids ``run_id`` and every run it continued trained on, down
    ``parent_run_id`` to the run from the base: ``(ids, None)``, or ``(the ids
    read so far, why the rest cannot be read)``.

    Read from each run's corpus (``source_ids_by_split["train"]``): a document
    in the train split of any corpus on the chain is one the continued adapter
    has seen, so a cycle continuing it must not validate on it: checkpoint
    selection and the calibration fits would read a memorised document as an
    unseen one (the split pins it to train,
    ``split_groups.assign_group_splits(trained=...)``). The
    corpus's train split, not the narrower train view: a document the view
    left out is pinned too, which costs one validation document at most and
    never lets a seen one through.

    A run whose manifest or corpus manifest cannot be read, or one continued
    by hand from a directory (``continued_from`` with no parent run), leaves the
    chain's documents unknown: absent evidence is not a clean split.
    """
    from registry_utils.query_registry import RegistryQueryError
    from registry_utils.query_registry import get as get_manifest

    trained: set[str] = set()
    seen: set[str] = set()
    current: str | None = run_id
    while current:
        if current in seen:
            return trained, f"the chain of runs loops at {current}"
        seen.add(current)
        try:
            manifest = get_manifest(current, ctx.client)
        except (RegistryQueryError, KeyError, FileNotFoundError, ValueError) as exc:
            return trained, f"the run manifest of {current} cannot be read ({type(exc).__name__})"
        key = paths.corpus_manifest(manifest.dependencies.corpus_version, ctx.tenant_id)
        try:
            corpus = ctx.client.read_json(key) if ctx.client.exists(key) else None
        except ValueError as exc:  # not JSON
            return trained, f"{current}'s corpus manifest at {key} cannot be read ({type(exc).__name__})"
        by_split = (corpus or {}).get("source_ids_by_split")
        if not isinstance(by_split, dict):
            return trained, f"{current}'s corpus manifest at {key} does not list its train documents"
        trained.update(str(s) for s in by_split.get("train") or [])
        if manifest.continued_from and not manifest.parent_run_id:
            return trained, (f"{current} continued {manifest.continued_from}, a directory, so what "
                             "the weights it started from trained on is not recorded")
        current = manifest.parent_run_id
    return trained, None


def corpus_validates_on_trained(ctx: StageContext, corpus_manifest: dict[str, Any],
                                parent: str) -> str | None:
    """Why this corpus cannot be trained by a cycle continuing ``parent``: its
    val or test split holds documents ``parent``'s chain trained on, or that
    chain's documents cannot be read. ``None`` when it can.

    A corpus is built for one continuation (``plan_corpus`` pins what that
    chain trained on), but one reused with ``--corpus-version``, or built while
    another release was promoted, was not; continued on it, the adapter would
    pick its checkpoint and fit its calibration on documents it memorised.
    """
    trained, unread = lineage_training_documents(ctx, ADAPTER.run_id(parent))
    if unread:
        return f"the documents {ADAPTER.run_id(parent)}'s chain trained on cannot be read ({unread})"
    by_split = corpus_manifest.get("source_ids_by_split") or {}
    seen = {split: sorted(trained & set(by_split.get(split) or [])) for split in ("val", "test")}
    seen = {split: ids for split, ids in seen.items() if ids}
    if not seen:
        return None
    shown = "; ".join(f"{split}: {len(ids)} ({', '.join(ids[:3])})" for split, ids in seen.items())
    return (f"corpus {ctx.corpus} holds documents {ADAPTER.run_id(parent)}'s chain trained on outside "
            f"train ({shown}): the continued adapter would select its checkpoint and fit "
            "calibration on documents it memorised. Rebuild the corpus in this cycle to continue")


# --------------------------------------------------------------------------
# Fresh or continue: the operator's rules (pipeline.yaml continued_training)
# --------------------------------------------------------------------------


def gold_hash(label: Any) -> str:
    """A gold's content: the sha256 of its canonical JSON (keys sorted)."""
    import hashlib

    text = json.dumps(label, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def stored_gold_hashes(client: BlobClient, doc_type_by_source: dict[str, str],
                       tenant_id: str | None, workers: int = PAGE_READ_WORKERS) -> dict[str, str | None]:
    """``{source id: gold_hash}`` of each gold as stored in Blob (before any rule
    reads it), None where there is none any more. Read through a pool."""
    from concurrent.futures import ThreadPoolExecutor

    def one(item: tuple[str, str]) -> str | None:
        source_id, doc_type = item
        key = paths.golden_label(doc_type, source_id, tenant_id)
        return gold_hash(client.read_json(key)) if client.exists(key) else None

    items = sorted(doc_type_by_source.items())
    if not items:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items))), thread_name_prefix="gold") as pool:
        return dict(zip((s for s, _ in items), pool.map(one, items), strict=True))


def _chain_corpora(ctx: StageContext, run_id: str) -> list[tuple[str, dict[str, Any]]]:
    """``(run id, its corpus manifest)`` down ``run_id``'s chain of continued runs,
    newest first. Only called once :func:`lineage_training_documents` read the
    whole chain, so every manifest here is readable."""
    from registry_utils.query_registry import get as get_manifest

    chain: list[tuple[str, dict[str, Any]]] = []
    current: str | None = run_id
    while current and current not in {r for r, _ in chain}:
        manifest = get_manifest(current, ctx.client)
        key = paths.corpus_manifest(manifest.dependencies.corpus_version, ctx.tenant_id)
        chain.append((current, ctx.client.read_json(key) if ctx.client.exists(key) else {}))
        current = manifest.parent_run_id
    return chain


def fresh_by_rule(ctx: StageContext, run_id: str, trained: set[str]) -> str | None:
    """Why this cycle starts from the base by the operator's rules, or None.

    ``continued_training`` in ``orchestration/config/pipeline.yaml``, every rule
    off by default (:func:`orchestration.settings.continued_training_rules`):

    - ``fresh_every_n_cycles: N`` - fresh once ``run_id``'s chain (the run from
      the base and every run continuing it) is N cycles long: every N-th cycle
      starts again from the base;
    - ``fresh_when_new_share_above: S`` - fresh when more than S of the
      documents this cycle can train on (labelled, not in the frozen eval set)
      are new to the chain. Measured before the split, so the dataset build and
      training reach one answer; an upper bound of the new share of train,
      since every document the chain trained on stays in train;
    - ``fresh_when_golds_changed: true`` - fresh when a gold the chain trained
      on changed content since (or was deleted): compared by the hash each
      corpus records (``gold_sha256``). A chain corpus that records none (built
      before 2026-10-26) cannot show it, and goes fresh too, saying so.

    ``trained``: the documents the chain trained on (:func:`lineage_training_documents`).
    """
    from orchestration.settings import continued_training_rules

    rules = continued_training_rules()
    every = rules["fresh_every_n_cycles"]
    share_above = rules["fresh_when_new_share_above"]
    golds = rules["fresh_when_golds_changed"]
    if not (every or share_above is not None or golds):
        return None
    chain = _chain_corpora(ctx, run_id)
    if every and len(chain) >= every:
        return (f"{run_id}'s chain is {len(chain)} cycle(s) long ({chain[-1][0]} from the base), and "
                f"continued_training.fresh_every_n_cycles is {every}: every {every}th cycle starts from the base")
    if share_above is not None:
        from data_pipeline.labeling.export_golden_labels import list_labeled_source_ids
        from evaluation.freeze_eval_set import frozen_manifest, is_frozen

        labelled = {s for dt in ctx.doc_types for s in list_labeled_source_ids(ctx.client, dt, ctx.tenant_id)}
        if is_frozen(ctx.client, tenant_id=ctx.tenant_id):
            labelled -= set(frozen_manifest(ctx.client, tenant_id=ctx.tenant_id).get("source_ids") or [])
        new = labelled - trained
        share = len(new) / len(labelled) if labelled else 0.0
        if share > share_above:
            return (f"{len(new)} of the {len(labelled)} documents this cycle can train on ({share:.0%}) are new "
                    f"to {run_id}'s chain, above continued_training.fresh_when_new_share_above ({share_above:.0%})")
    if golds:
        return golds_changed(ctx, run_id, chain)
    return None


def golds_changed(ctx: StageContext, run_id: str, chain: list[tuple[str, dict[str, Any]]]) -> str | None:
    """Why the golds ``run_id``'s chain trained on are not all as they trained, or None."""
    recorded: dict[str, str] = {}
    for current, corpus in reversed(chain):                 # oldest first: the newest hash wins
        hashes = corpus.get("gold_sha256")
        if not isinstance(hashes, dict):
            return (f"continued_training.fresh_when_golds_changed is on, and {current}'s corpus "
                    f"{corpus.get('corpus_version') or '(unreadable)'} records no gold hashes (built before "
                    "2026-10-26), so whether a gold it trained on changed cannot be told: from the base")
        train = set((corpus.get("source_ids_by_split") or {}).get("train") or [])
        recorded.update({s: h for s, h in hashes.items() if s in train})
    doc_type_by_source = {}
    for source_id in recorded:
        for doc_type in ctx.doc_types:
            if ctx.client.exists(paths.golden_label(doc_type, source_id, ctx.tenant_id)):
                doc_type_by_source[source_id] = doc_type
                break
    now = stored_gold_hashes(ctx.client, doc_type_by_source, ctx.tenant_id)
    changed = sorted(s for s, h in recorded.items() if now.get(s) != h)
    if not changed:
        return None
    return (f"{len(changed)} gold(s) {run_id}'s chain trained on changed or were deleted since "
            f"({', '.join(changed[:5])}), and continued_training.fresh_when_golds_changed is on")


def stage_parent_adapter(ctx: StageContext, parent: str) -> str:
    """Copy the parent's adapter to the staging volume; return the local directory.
    A dry run copies nothing and returns where it would go."""
    local = staging_parent_adapter_dir(ctx.out_version)
    if not ctx.dry_run:
        copied = ctx.client.download_dir(paths.scoped_adapter_dir(ADAPTER.name, parent), local)
        log.info("copied %s's adapter (%d file(s)) to %s", ADAPTER.run_id(parent), copied, local)
        ctx.volume.mark(local)
    return local


def stage_training(ctx: StageContext) -> StageResult:

    """ONE run of the policy adapter (decision 2026-10-22), continuing the
    promoted adapter or starting from the base (``continuation``, decision
    2026-10-23).

    v1 trained a Foundation and then fanned out one adapter per document type,
    sequentially, because each sat on top of the Foundation's weights. That
    topology is gone: vLLM applies one LoRA per request, so a Foundation LoRA and
    a per-type LoRA could never both be active on the same call. Each cycle now
    trains the one adapter (``common.adapter``), always on the original base
    model, never on a merged one.
    """
    from registry_utils.models import DataStats
    from training.train import count_examples, train

    corpus_manifest = ctx.client.read_json(paths.corpus_manifest(ctx.corpus, ctx.tenant_id))
    _assert_corpus_pins(corpus_manifest, f"training run {ctx.out_version}")
    # Continue the promoted adapter, or start from the base, and why.
    parent, fresh_reason = continuation(ctx)
    if parent is not None and (held := corpus_validates_on_trained(ctx, corpus_manifest, parent)):
        # Built for another chain (or reused): from the base, the documents in
        # val are unseen again, so the corpus is sound for a fresh run.
        parent, fresh_reason = None, held
    continue_from = parent_run_id = None
    if parent is None:
        log.info("%s starts from the base model: %s", ctx.out_version, fresh_reason)
    else:
        parent_run_id = ADAPTER.run_id(parent)
        continue_from = stage_parent_adapter(ctx, parent)
        log.info("%s continues %s (its adapter, at %s)", ctx.out_version, parent_run_id, continue_from)
    # One output shape is checked by training.train, on the documents the
    # adapter's view really trains (common.adapter.assert_one_output_shape).
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

    gpu_class = ctx.gpu_class or gpu_class_for("training")
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
            continue_from=continue_from,
            parent_run_id=parent_run_id,
            fresh_reason=fresh_reason,
            # A real run refuses a dirty tree unless asked (--allow-dirty).
            allow_dirty=getattr(ctx, "allow_dirty", False),
        )
        ctx.manifests[ADAPTER.name] = manifest

        if not ctx.dry_run and not ctx.checkpoints:
            # Nothing else fills these in, so checkpoint selection skipped on
            # every real run and merge took whatever load_best_model_at_end left.
            from evaluation.checkpoint_eval import discover_checkpoints

            staged = paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version)
            ctx.checkpoints, ctx.best_loss_checkpoint = discover_checkpoints(staged)
            if not ctx.checkpoints:
                # A failure, not a warning. ms-swift exiting 0 is not evidence
                # that an adapter exists: zero steps (a resume past max_steps, a
                # tiny corpus) exits cleanly too, and package would then fall
                # back to the output root, which holds no adapter at all.
                raise PipelineError(
                    f"training for {manifest.run_id} exited cleanly but wrote no checkpoint "
                    f"under {staged}, so there is no adapter to select or publish. Check the "
                    "ms-swift log for the step count."
                )
        # The staging volume is where calibration, the gate and push look for
        # the adapter. Without this mark the artifacts exist and the pipeline cannot
        # find them.
        ctx.volume.mark(paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version))

    # --push-adapters pushes the SELECTED checkpoint, once checkpoint selection
    # has chosen it (stage_checkpoint_eval): pushed here, it was the one
    # validation loss preferred, which package --from-blob could not use.

    return StageResult(
        "training", "completed",
        f"trained {manifest.run_id} on {manifest.data_stats.train_examples} examples, "
        + (f"continuing {parent_run_id}" if parent_run_id else f"from the base ({fresh_reason})"),
        {
            "runs": [manifest.run_id],
            "run_type": manifest.run_type,
            "push_adapters": ctx.push_adapters,
            "continued": parent_run_id,
            "fresh_reason": fresh_reason,
            # The parent's eval report re-scored from its stored answers so the
            # gate can compare this run with it (decision 2026-10-26).
            "parent_report_rescored": ctx.parent_report_rescored if parent_run_id else None,
        },
    )


# --------------------------------------------------------------------------
# Stage 6 — checkpoint selection (arch v2.1 §11.2)
# --------------------------------------------------------------------------


def _rediscover_checkpoints(ctx: StageContext) -> None:
    """Fill ``ctx.checkpoints`` from the staging volume when this process did
    not train. ``--from-stage checkpoint_eval``, ``--from-stage calibrate`` and a
    training stage skipped as already complete all left it empty, so checkpoint
    selection "skipped" and the next stage fell back to the ms-swift output root — which
    holds a ``v0-<timestamp>/`` directory, not a loadable adapter."""
    if ctx.dry_run or ctx.checkpoints:
        return
    from evaluation.checkpoint_eval import discover_checkpoints

    staged = paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version)
    if Path(staged).is_dir():
        ctx.checkpoints, ctx.best_loss_checkpoint = discover_checkpoints(staged)


def _selection_settings(ctx: StageContext) -> dict[str, Any]:
    """The validation cut's settings for checkpoint selection, from the training config."""
    from common.config import training_config
    from common.constants import MODALITY_MODES

    evaluation = training_config(ADAPTER.training_config)["evaluation"]
    # The reading modes the finalists' whole-document pass reads (none = all).
    modes = [str(m) for m in (evaluation.get("selection_document_modes") or [])]
    unknown = sorted(set(modes) - set(MODALITY_MODES))
    if unknown:
        raise PipelineError(
            f"selection_document_modes names {unknown}, which no validation row is read in; "
            f"the reading modes are {list(MODALITY_MODES)}"
        )
    # The second reading covers the gate's own repeat mode only (image_only):
    # what the gate holds a release to, at the cost of that mode's rows. A pass
    # that does not read the mode (the standard profile reads ocr_plus_image)
    # reads all its rows twice, as before, rather than measure nothing.
    from evaluation.golden_eval import run_to_run_mode

    repeat_mode = run_to_run_mode()
    repeat_modes = [repeat_mode] if not modes or repeat_mode in modes else []
    if not repeat_modes:
        log.warning("selection: the finalists' pass reads %s, not the gate's repeat mode %s; its rows are "
                    "all read twice for the run-to-run guard", modes, repeat_mode)
    return {
        "validation_sample_rows": int(evaluation.get("validation_sample_rows") or 0),
        "selection_tie_break_margin": float(evaluation.get("selection_tie_break_margin") or 0.0),
        "selection_document_finalists": int(evaluation.get("selection_document_finalists") or 0),
        "selection_guard_margin": float(evaluation.get("selection_guard_margin") or 0.0),
        "selection_document_modes": modes,
        # The finalists' pass read twice, for the run-to-run guard; on unless
        # the config says false (decision 2026-10-24).
        "selection_run_to_run_check": bool(evaluation.get("selection_run_to_run_check", True)),
        # ...the second time in these modes only (none = every row of the pass).
        "selection_repeat_modes": repeat_modes,
        # The checkpoint nearest two-thirds of the final step is a candidate too
        # (checkpoint_eval.two_thirds_checkpoint): the evidence for two epochs.
        "selection_two_thirds_candidate": bool(evaluation.get("selection_two_thirds_candidate", True)),
        # Calibration fits from the selected finalist's saved answers rather
        # than generating the validation split again (collect_calibration_samples).
        "calibration_reuses_selection": bool(evaluation.get("calibration_reuses_selection", True)),
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
        # The adapter's own validation view: the policy rows it trains on.
        "val_path": paths.corpus_scope_eval_split(ctx.corpus, "val", ADAPTER.name, ctx.tenant_id),
        "images_root": paths.staging_train_images_dir(ctx.corpus, ctx.tenant_id),
        "sample_rows": int(selection["validation_sample_rows"]),
        # The finalists' whole-document pass (and a near-tie) reads these modes only.
        "full_modes": selection["selection_document_modes"],
        # ...and is read twice, for the run-to-run guard, when there are
        # finalists to judge: a near-tie alone reads nothing twice.
        "repeat": bool(selection["selection_run_to_run_check"]
                       and selection["selection_document_finalists"] >= 2),
        # ...the gate's repeat mode only (image_only), where the pass reads it.
        "repeat_modes": selection["selection_repeat_modes"],
    }
    distributed = training_config(ADAPTER.training_config).get("distributed") or {}
    gpus = scoring_gpus()[: training_gpus(distributed)]
    # The run's working dir: each finalist's pass is saved here, and the
    # selected one's is what calibration fits from (_finalist_answers).
    work_dir = Path(paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version)) / "checkpoint_scores"
    if len(gpus) > 1:
        return parallel_vllm_scorer(**settings, gpus=gpus, work_dir=work_dir, save_dir=work_dir / "finalists")
    return vllm_scorer(**settings, save_dir=work_dir / "finalists")


def stage_checkpoint_eval(ctx: StageContext) -> StageResult:
    """Pick the checkpoint that ships, by GENERATED field F1.

    Validation loss drove selection under v1, and it is a poor proxy: averaged
    over every token, it is dominated by the easy copy tokens — schema keys, JSON
    punctuation, boilerplate — the model gets right within a hundred steps. A
    checkpoint can improve on loss while getting worse at the values, which is
    the only thing the gate reads.

    Skipped, explicitly, when nothing was staged: a dry run never launched
    ms-swift, so there is nothing to choose between, and packaging the staged
    adapter directory is the honest fallback rather than a silent one.

    With ``--push-adapters`` the selected checkpoint is pushed to Blob here, so
    a reclaimed volume does not lose it (``package --from-blob``).
    """
    from evaluation.checkpoint_eval import CheckpointEvalError, select_best

    _rediscover_checkpoints(ctx)
    checkpoints = sorted(ctx.checkpoints or [])
    if not checkpoints:
        return StageResult(
            "checkpoint_eval", "skipped",
            "no staged checkpoints to choose between; package takes the staged adapter directory",
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
    two_thirds = None
    if settings["selection_two_thirds_candidate"]:
        from evaluation.checkpoint_eval import two_thirds_checkpoint

        two_thirds = two_thirds_checkpoint(checkpoints)
    try:
        report = select_best(checkpoints, scorer, best_loss=ctx.best_loss_checkpoint, two_thirds=two_thirds)
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
        # The base engine is done once selection is; calibration needs the card
        # for the next model.
        if callable(getattr(scorer, "close", None)):
            scorer.close()

    from evaluation.checkpoint_eval import predicted_gate_field_match, two_thirds_outcome

    report.two_thirds = two_thirds_outcome(report, sample_margin=settings["selection_tie_break_margin"])
    report.finalist_pass = _persist_finalist_pass(ctx, scorer, report)
    report.early_warning = predicted_gate_field_match(report.gate_mirror_accuracy, _gate_mirror_history(ctx))
    ctx.client.write_json(
        paths.checkpoint_selection(ctx.out_version, scope=ADAPTER.name), report.as_dict()
    )
    if ctx.push_adapters and not ctx.dry_run:
        # Belt and braces: the adapter is small, so pushing it now costs little
        # and means a reclaimed volume does not lose it. The checkpoint
        # selected - the one calibration, the gate and package use - and the
        # adapter files only, never the checkpoint's resume state.
        from artifact_registry.transfer import push_scoped_adapter

        blob_dir = push_scoped_adapter(str(report.selected), ADAPTER.name, ctx.out_version,
                                       client=ctx.client)
        _record_pushed(ctx, str(report.selected))
        log.info("pushed the selected adapter %s -> %s", report.selected, blob_dir)
    # The margin is the SELECTED checkpoint's on the sample (negative when whole
    # documents or the full split chose one the sample ranks behind), so the
    # detail says which decision overrode the sample rather than reading as a lead.
    overrode = ("whole validation documents" if (report.document_choice or {}).get("changed_choice")
                else "the full validation split" if (report.tie_break or {}).get("changed_choice")
                else None)
    detail = (f"selected {report.selected} (margin {report.margin:+.4f} field F1 on the validation "
              f"sample{f'; chosen on {overrode} over the sample leader' if overrode else ''})")
    if report.loss_and_f1_disagreed:
        detail += f"; validation loss would have shipped {report.best_loss_checkpoint}"
    detail += _selection_summary(report)
    from evaluation.checkpoint_eval import generation_cost

    cost = generation_cost(scorer)
    if cost:
        detail += f"; {cost['rows']} row(s) generated in {cost['seconds']:.0f} s"
    return StageResult("checkpoint_eval", "completed", detail,
                       {**report.as_dict(), **({"generation": cost} if cost else {})})


def _selection_summary(report: Any) -> str:
    """What the selection summary adds to the winner: the gate mirror (or the
    fallback the finalists were judged on), the early warning, and whether the
    two-thirds checkpoint won or tied - the evidence for moving to two epochs."""
    parts: list[str] = []
    choice = report.document_choice or {}
    if report.gate_mirror_accuracy is not None:
        parts.append(f"gate mirror {report.gate_mirror_accuracy:.4f}")
    elif choice.get("field_match_fallback"):
        parts.append(f"no gate mirror ({choice['field_match_fallback']})")
    warning = report.early_warning or {}
    if warning.get("summary"):
        parts.append(str(warning["summary"]))
        log.info("selection: %s", warning["summary"])
    third = report.two_thirds or {}
    if third.get("checkpoint"):
        name = Path(str(third["checkpoint"])).name
        if "won" not in third:
            verdict = "not scored"
        elif third["won"]:
            verdict = "WON"
        elif third.get("tied"):
            verdict = f"tied (behind by {third.get('behind_by')} on {third.get('measured_on')}, within {third.get('margin')})"
        else:
            verdict = f"lost (behind by {third.get('behind_by')} on {third.get('measured_on')})"
        parts.append(f"two-thirds candidate {name} (step {third.get('step')} of {third.get('final_step')}): {verdict}")
    return "".join(f"; {part}" for part in parts)


def _persist_finalist_pass(ctx: StageContext, scorer: Any, report: Any) -> dict[str, Any]:
    """The selected checkpoint's finalist pass, kept: its answers, logprobs
    included, to Blob beside the run's artifacts (``validation_generations_key``;
    the run's working dir has them already), which calibration fits from; and
    its validation error tallies (``validation_errors_key``), for the tallies'
    reader. Neither is fatal: without them calibration generates again."""
    from evaluation.checkpoint_eval import saved_finalist_pass

    found = saved_finalist_pass(scorer, report.selected)
    if not found:
        return {"saved": False, "reason": "the selected checkpoint's finalist pass was not saved (no "
                "whole-document pass, or it failed); calibration generates the validation split itself"}
    run_id = ADAPTER.run_id(ctx.out_version)
    generations_key = paths.validation_generations_key(run_id, ctx.tenant_id)
    errors_key = paths.validation_errors_key(run_id, ctx.tenant_id)
    record: dict[str, Any] = {"saved": True, "checkpoint": report.selected, "local": found["generations"]}
    try:
        from evaluation.validation_generation import read_generations

        with open(found["generations"], encoding="utf-8") as handle:
            header, _ = read_generations(handle.readline())
        record.update(settings=header.get("settings"), modes=header.get("modes"), rows=header.get("rows"))
        ctx.client.upload_file(found["generations"], generations_key)
        record["blob"] = generations_key
        tallies = json.loads(Path(found["tallies"]).read_text(encoding="utf-8"))
        ctx.client.write_json(errors_key, {
            "version": 1, "run_id": run_id, "checkpoint": str(report.selected),
            **{name: tallies.get(name) or {} for name in ("by_field", "by_line", "by_group", "by_mode")},
        })
        record["validation_errors"] = errors_key
    except Exception as exc:  # noqa: BLE001 - kept where it could be; calibration falls back
        log.warning("selection: could not keep the selected finalist's pass in Blob: %s", exc)
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _validation_gate_mirror(ctx: StageContext) -> float | None:
    """The selected checkpoint's validation gate mirror, from this process's
    checkpoint selection or its record in Blob; None when selection measured
    none (no real image_only document, or a version selected before it was)."""
    selection = ctx.results.get("checkpoint_eval")
    value = (selection.data or {}).get("validation_gate_mirror_accuracy") if selection else None
    if value is None:
        key = paths.checkpoint_selection(ctx.out_version, scope=ADAPTER.name)
        try:
            value = ctx.client.read_json(key).get("validation_gate_mirror_accuracy") if ctx.client.exists(key) else None
        except Exception as exc:  # noqa: BLE001 - reported, never gated: an unreadable record is no number
            log.warning("gate: could not read the validation gate mirror from %s: %s", key, exc)
            value = None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _gate_mirror_history(ctx: StageContext) -> list[dict[str, Any]]:
    """Each earlier cycle of the adapter that recorded both numbers: its
    selected checkpoint's validation gate mirror (its checkpoint selection) and
    the gate's field match - the tier's number - on the golden set (its gate
    decision). What the early warning's offset is the mean of."""
    current = paths.version_number(ctx.out_version)
    history: list[dict[str, Any]] = []
    for number in range(1, current or 0):
        version = f"v{number}"
        selection_key = paths.checkpoint_selection(version, scope=ADAPTER.name)
        decision_key = paths.gate_decision(version, scope=ADAPTER.name)
        try:
            if not (ctx.client.exists(selection_key) and ctx.client.exists(decision_key)):
                continue
            mirror = ctx.client.read_json(selection_key).get("validation_gate_mirror_accuracy")
            gate = (ctx.client.read_json(decision_key).get("candidate_metrics") or {}).get("field_normalized_match")
        except Exception as exc:  # noqa: BLE001 - an unreadable cycle is left out, not fatal
            log.warning("selection: could not read %s's gate mirror or gate decision: %s", version, exc)
            continue
        history.append({"version": version, "gate_mirror_accuracy": mirror, "gate_field_match": gate})
    return history


# --------------------------------------------------------------------------
# Stage 8 — evaluation and the promotion gate (HARD STOP)
# --------------------------------------------------------------------------


def _major(version: str) -> str:
    """The major component of ``v2``, ``v2.1``, ``foundation-v3`` — all ``2``/``3``."""
    tag = (version_of(version) if is_valid_run_id(version) else version).lstrip("vV")
    return tag.split(".")[0] or tag


def is_major_bump(previous: str | None, candidate: str) -> bool:
    """Whether the candidate is a new Foundation **major** version.

    A run that continues the promoted adapter is a different case, held line by
    line to the runs before it (``gating._require_line_non_regression``).
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
    from registry_utils.query_registry import adapters_depending_on

    # "unified" under arch v2.1 §4.1. A graduated per-type adapter (§4.2) still
    # records the unified run it was trained on as its foundation_version, so the
    # dependency-upgrade rule is unchanged — only the run_type it looks for moved.
    promoted = promoted_version(ctx)
    previous = ADAPTER.run_id(promoted) if promoted else None
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
    a scoring pass done separately), it is read rather than recomputed - when it
    records that it was scored on this tenant's frozen set as it stands now, by
    the scoring code running now (``run_eval.EVAL_SET_IDENTITY``). The report
    path names no tenant, so another tenant's report there, or one that does not
    say whose set it was scored on, is scored again as if it were absent. So is
    one scored before the set was frozen again (corrected golds: one source id
    may now be another document or another gold) or under other scoring code
    (other label rules or scorers): its numbers are another yardstick's, and
    the gate would pass on them while recording the current set beside them.

    Wiring this in is what makes ``finetune`` completable: the CLI supplied no
    provider at all, so every real run trained a Foundation and three adapters on
    an A100 and then aborted at the gate.
    """
    from evaluation.run_eval import assert_eval_set_disjoint, eval_set_record

    assert_eval_set_disjoint(ctx.client, ctx.corpus, ctx.tenant_id)

    # The adapter's report, where stage_push records it on the manifest.
    summary_key = paths.eval_report(ctx.out_version, scope=ADAPTER.name)
    calibrators, thresholds = release_calibration(ctx, "bf16")
    # The lines the release will serve, and so the documents it is scored on:
    # serving sends every other policy to the base model.
    lines = served_lines(ctx)
    # The checkpoint scored: a report of another one (selected again since) is
    # never this release's.
    scored = adapter_checkpoint(ctx)
    not_reused: str | None = None
    if ctx.client.exists(summary_key):
        report = ctx.client.read_json(summary_key)
        metrics = report.get("gate_metrics") or report.get("candidate_metrics") or {}
        recorded = report.get("eval_set") or {}
        tenant = paths._tenant(ctx.tenant_id)
        if not metrics:
            not_reused = "it holds no gate metrics"
        elif "lines_scored" in report and report["lines_scored"] != lines:
            not_reused = (f"it was scored on lines {report['lines_scored'] or 'all'}, and this release "
                          f"serves {lines}")
        elif "adapter_checkpoint" in report and report["adapter_checkpoint"] != scored:
            not_reused = (f"it scored checkpoint {report['adapter_checkpoint']}, and the checkpoint "
                          f"selected now is {scored}")
        elif not recorded.get("tenant_id"):
            not_reused = ("it does not record which tenant's frozen set it was scored on (written "
                          "before reports recorded it), so it cannot be shown to be this tenant's")
        elif recorded["tenant_id"] != tenant:
            not_reused = (f"it was scored on tenant {recorded['tenant_id']!r}'s frozen set, and this "
                          f"candidate gates tenant {tenant!r}")
        elif stale := _other_yardstick(recorded, eval_set_record(ctx.client, ctx.tenant_id, [])):
            not_reused = stale
        elif calibrators is not None and report.get("calibrated") is not True:
            # Scored before this release had calibrators: every field was
            # flagged, so auto_accept_error_rate read a trivial 0.0.
            not_reused = "it was made without calibrators, and this release has them"
        else:
            log.info("gate metrics read from %s", summary_key)
            return dict(metrics)
        log.info("the eval report at %s is not reused: %s. Scoring again.", summary_key, not_reused)

    if ctx.dry_run:
        found = (f"the eval report at {summary_key} is not reused ({not_reused})" if not_reused
                 else f"no eval report at {summary_key}")
        raise PipelineError(
            f"{found}, and a dry run loads no model to produce one. "
            "The gate will not judge a candidate on metrics nobody measured."
        )

    # No report yet: produce it. The frozen golden set goes through the serving
    # pipeline with the bf16 base, this cycle's adapter applied, and the release's
    # own bf16 calibrators, so the gate measures what production would serve —
    # windows, the merge of their outputs, date post-process and calibrated
    # confidence included.
    from evaluation.golden_eval import assert_scoreable, evaluate_version
    from inference_core.model_runner import release_model

    # Before the model is loaded: a frozen gold with no review record is refused
    # by the scoring, and learning that after loading the model bought
    # nothing.
    assert_scoreable(ctx.client, ADAPTER.doc_types, tenant_id=ctx.tenant_id)
    if not lines:
        raise PipelineError(
            f"{ctx.out_version} trained on no single-line policy of a known line (its run's train "
            "view and its corpus record none), so it has no line to serve or to be scored on."
        )
    loader = ctx.serving_model_loader or _staged_serving_model
    model = loader(ctx, "bf16")
    try:
        report = evaluate_version(
            ctx.client, model,
            version=ctx.out_version, corpus_version=ctx.corpus, tenant_id=ctx.tenant_id,
            calibrators=calibrators, thresholds=thresholds, lines=lines,
            adapter_checkpoint=scored,
        )
    finally:
        release_model(model)
    return dict(report.get("gate_metrics") or {})


def _other_yardstick(recorded: dict[str, Any], current: dict[str, Any]) -> str | None:
    """Why an eval report's ``eval_set`` (``recorded``) is not the yardstick the
    gate holds now (``current``, ``run_eval.eval_set_record`` of the set and the
    code as they stand); ``None`` when it is.

    Compared field by field rather than through ``yardstick_difference``, which
    compares two releases' reports and would call this one "the promoted
    release". The tenant is checked by the caller. A report that records no
    freeze or no fingerprint (written before reports recorded them) cannot be
    shown to be on today's yardstick, so it is not.
    """
    then = (recorded.get("frozen_at"), recorded.get("frozen_from_corpus"))
    now = (current.get("frozen_at"), current.get("frozen_from_corpus"))
    if not all(then):
        return ("it does not record which freeze of the eval set it was scored on (written before "
                "reports recorded it), so it cannot be shown to be scored on the set as it stands")
    if then != now and not all(now):
        return (f"it was scored on the set frozen {then[0]} from {then[1]}, and no set is frozen "
                "now: what it was scored on is not the gate's yardstick")
    if then != now:
        return (f"it was scored on the set frozen {then[0]} from {then[1]}, and the gate holds the "
                f"set frozen {now[0]} from {now[1]}: the set was frozen again (corrected golds), so "
                "one source id may be another document or another gold")
    if recorded.get("scoring_fingerprint") != current.get("scoring_fingerprint"):
        return (f"it was scored by other scoring code (fingerprint "
                f"{recorded.get('scoring_fingerprint') or 'unrecorded'}, the code now "
                f"{current.get('scoring_fingerprint')}): other label rules, label verification or "
                "scorers read and score the same gold differently")
    return None


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
    fitted_on, selected = body.get("adapter_checkpoint"), adapter_checkpoint(ctx)
    if fitted_on is not None and fitted_on != selected:
        raise PipelineError(
            f"the {fmt} calibrators at {key} were fitted on checkpoint {fitted_on}, and the "
            f"checkpoint selected now is {selected}: confidence from one checkpoint's logprobs "
            "says nothing about another's. Run package again from calibrate."
        )
    calibrators = CalibratorSet.from_dict(body["calibrators"])
    thresholds = ThresholdSet.from_dict(body["thresholds"])
    ctx.calibrators[fmt], ctx.thresholds[fmt] = calibrators, thresholds
    log.info("loaded the %s calibrators of %s from %s", fmt, ctx.release_id, key)
    return calibrators, thresholds


def default_baseline_metrics(ctx: StageContext) -> dict[str, Any] | None:
    """The production version's scores, read from its own eval report.

    ``None`` means there is no promoted version yet — a genuine first version,
    which the gate handles. It never means "could not find one".

    This tenant's promoted release (:func:`promoted_version`), never another
    tenant's newer one: its report as the gate compares the candidate with it
    (:func:`promoted_baseline_report`, re-scored when only the scoring code or
    settings moved). Whether its scores can be compared with the candidate's at
    all is :func:`baseline_not_comparable`.
    """
    found = promoted_baseline_report(ctx)
    if found is None:
        return None
    report = found[1]
    return report.get("gate_metrics") or report.get("candidate_metrics") or None


def baseline_not_comparable(ctx: StageContext) -> str | None:
    """Why the promoted release's eval report is not on the candidate's
    yardstick; ``None`` when it is, or when there is no promoted report.

    Another yardstick is another frozen set (the set was frozen again), another
    tenant's set, or other scoring code or settings
    (``run_eval.yardstick_difference``), or a report that does not record which.
    Other scoring code or settings alone: the promoted release's stored answers
    are re-scored by the code running now (:func:`promoted_baseline_report`,
    decision 2026-10-26) and compared - the old model is never run again. When
    they cannot be (another set or tenant, answers missing or incomplete, a
    release scored before answers were stored), decision 2026-10-22: the gate
    judges the candidate on its floors alone and says why in the decision.

    The candidate's yardstick is its own eval report's record. A candidate with
    no record (scores a metrics provider supplied) is taken as scored on the set
    and the code as they stand now.
    """
    from evaluation.run_eval import yardstick_difference

    found = promoted_baseline_report(ctx)
    if found is None:
        return None
    promoted_key, promoted_report, rescoring = found
    if unusable := _unusable_baseline(ctx, promoted_key, promoted_report):
        return unusable
    reason = yardstick_difference(promoted_report.get("eval_set"), _candidate_eval_set(ctx), ctx.tenant_id)
    if not reason:
        return None
    refused = f". {rescoring['reason']}" if rescoring and not rescoring.get("rescored") else ""
    return f"{reason} (promoted report {promoted_key}){refused}"


def promoted_baseline_report(ctx: StageContext) -> tuple[str, dict[str, Any], dict[str, Any] | None] | None:
    """``(where, report, rescoring)``: the promoted release's eval report as the
    gate compares the candidate with it; ``None`` with no promoted report.

    Its own report when it is on the candidate's yardstick. When it is on
    another only because the scoring code or settings moved, its stored answers
    re-scored by the code running now (``evaluation.rescore.rescore_report``,
    decision 2026-10-26; the old model is never run again), so non-inferiority
    and the improvement rule apply again: ``where`` then ends "(re-scored)" and
    ``rescoring`` is ``{"rescored": True, "difference", "reason",
    "rescored_from"}``. When they cannot be re-scored - another frozen set or
    tenant, answers missing or incomplete, a release scored before answers were
    stored - its own report, and ``{"rescored": False, "difference", "reason"}``:
    the candidate is judged on its floors alone, as before. ``rescoring`` is
    ``None`` when nothing had to be re-scored, or the report cannot be compared
    whatever its yardstick (:func:`_unusable_baseline`).

    Re-scored once per gate however often it is asked: the re-scoring is
    remembered per Blob client (``evaluation.rescore``).
    """
    from evaluation.rescore import rescore_report
    from evaluation.run_eval import yardstick_difference

    key = _promoted_eval_report_key(ctx)
    if key is None:
        return None
    report = ctx.client.read_json(key)
    candidate = _candidate_eval_set(ctx)
    different = yardstick_difference(report.get("eval_set"), candidate, ctx.tenant_id)
    if different is None or _unusable_baseline(ctx, key, report):
        return key, report, None
    version = promoted_version(ctx)
    rescored, why = (rescore_report(ctx.client, version, ctx.tenant_id, eval_set=candidate) if version
                     else (None, "no promoted version"))
    if rescored is None:
        return key, report, {"rescored": False, "difference": different, "reason": why}
    log.info("the promoted release's report is compared re-scored: %s", why)
    return f"{key} (re-scored)", rescored, {"rescored": True, "difference": different, "reason": why,
                                            "rescored_from": rescored.get("rescored_from")}


def baseline_rescoring(ctx: StageContext) -> dict[str, Any] | None:
    """What :func:`promoted_baseline_report` re-scored of the promoted release's
    report, or why it could not, for the gate's decision; ``None`` when
    nothing had to be."""
    found = promoted_baseline_report(ctx)
    return found[2] if found else None


def _unusable_baseline(ctx: StageContext, key: str, report: dict[str, Any]) -> str | None:
    """Why the promoted release's report cannot be compared whatever its
    yardstick: scored without its calibrators, or on other lines."""
    if report.get("calibrated") is False:
        # Scored without its calibrators (an ad-hoc `run_eval`, which writes the
        # same key): every field flagged, so auto_accept_error_rate reads 0.0.
        return (f"the promoted release's report at {key} was scored without its "
                "calibrators")
    candidate_lines = served_lines(ctx)
    if "lines_scored" in report and report["lines_scored"] != candidate_lines:
        return (f"the promoted release was scored on lines {report['lines_scored'] or 'all'} "
                f"and the candidate serves {candidate_lines} (promoted report {key})")
    return None


def _candidate_eval_set(ctx: StageContext) -> dict[str, Any]:
    """The candidate's yardstick: its own eval report's record, or - with none
    (scores a metrics provider supplied) - the set and the code as they stand now."""
    from evaluation.run_eval import eval_set_record

    candidate_key = paths.eval_report(ctx.out_version, scope=ADAPTER.name)
    candidate = (ctx.client.read_json(candidate_key).get("eval_set")
                 if ctx.client.exists(candidate_key) else None)
    return candidate or eval_set_record(ctx.client, ctx.tenant_id, [])


def promoted_version(ctx: StageContext) -> str | None:
    """The adapter version of this tenant's promoted release, other than this
    cycle's own; ``None`` when none is promoted (:func:`promoted_release`)."""
    promoted = promoted_release(ctx)
    return promoted[1] if promoted else None


def promoted_release(ctx: StageContext) -> tuple[str, str] | None:
    """``(release id, adapter version)`` of this tenant's promoted release,
    other than this cycle's own; ``None`` when none is promoted, or when it is
    a release of another model.

    Read from the tenant's release index - what serving serves - not from run
    manifests: no real run ever set a manifest's status to promoted, so every
    candidate was gated as a first version, with no regression check at all.
    A release named by version is its adapter's version; one named before
    releases were (``release-YYYY.M.N``) gives its adapter's, and keeps its own
    name for its bundle.
    """
    from common.run_ids import RunIdError, parse_run_id
    from serving.release_router import promoted_rows

    for row in promoted_rows(ctx.client, ctx.tenant_id):
        release_id = str(row.get("release_id") or "")
        if release_id == ctx.release_id:
            continue
        if paths.version_number(release_id) is not None:
            return release_id, release_id
        try:
            parsed = parse_run_id(str(row.get("adapter") or ""))
        except (RunIdError, ValueError):
            parsed = None
        if parsed is None or parsed.lineage != ADAPTER.name:
            # Another model (an older unified extractor, say): the policy
            # adapter's report at that number would be someone else's.
            log.warning("promoted release %s is not a release of the policy adapter (adapter %r); "
                        "nothing to regress against", release_id, row.get("adapter"))
            return None
        return release_id, parsed.version
    return None


def _promoted_eval_report_key(ctx: StageContext) -> str | None:
    """Where this tenant's promoted release's eval report lives; ``None`` with no report."""
    promoted = promoted_version(ctx)
    if not promoted:
        return None
    key = paths.eval_report(promoted, scope=ADAPTER.name)
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
) -> dict[str, tuple[list[float], ...]] | None:
    """Each gate metric's per-document scores, promoted vs candidate, paired.

    Read from the two eval reports (``EvalReport.per_document_scores``) and
    aligned on the documents both scored, in sorted order: that alignment is
    what the paired bootstrap resamples. Each comes as ``(current, candidate,
    current_weights, candidate_weights)``: a document counts in the bootstrap as
    it counts in the reported metric (``EvalReport.per_document_weights``). A
    metric either report holds no weights for (written before they were kept)
    comes as ``(current, candidate)`` and is bootstrapped unweighted on both
    sides, with a warning: weighting one side only compared a weighted mean with
    an unweighted one, and identical scores bootstrapped as an improvement.

    Paired only when both reports were scored on one yardstick
    (:func:`evaluation.run_eval.yardstick_difference`): source ids are numbered
    per tenant and reused when a set is frozen again, the report path names no
    tenant, and a gold is read under the running code's label rules. The gate
    does not ask for pairs then (:func:`baseline_not_comparable`: floors only).
    ``None`` whenever pairing is not possible - another yardstick, no shared
    documents, a report without per-document scores - with the reason added to
    ``warnings``; on one yardstick the gate then compares point estimates and
    cannot enforce the improvement rule, and the decision has to say so.
    """
    from evaluation.run_eval import yardstick_difference

    notes = warnings if warnings is not None else []
    # The promoted report as the gate compares it: re-scored when only the
    # scoring code or settings moved (promoted_baseline_report).
    found = promoted_baseline_report(ctx)
    if found is None:
        return None
    promoted_key, promoted_report, _rescoring = found
    candidate_key = paths.eval_report(ctx.out_version, scope=ADAPTER.name)
    if not ctx.client.exists(candidate_key):
        _not_paired(notes, f"the candidate has no eval report at {candidate_key}")
        return None
    candidate_report = ctx.client.read_json(candidate_key)
    # The yardstick first: two yardsticks' scores are not paired even where both
    # reports hold per-document scores, and are not compared at all.
    different = yardstick_difference(
        promoted_report.get("eval_set"), candidate_report.get("eval_set"), ctx.tenant_id)
    if different:
        _not_paired(notes, different, "The candidate is judged on its floors alone (decision "
                    "2026-10-22, baseline_not_comparable).")
        return None
    reason = _unpairable(ctx, promoted_key, promoted_report, candidate_key, candidate_report)
    if reason:
        _not_paired(notes, reason)
        return None
    current, candidate = promoted_report["per_document"], candidate_report["per_document"]
    current_w = promoted_report.get("per_document_weights") or {}
    candidate_w = candidate_report.get("per_document_weights") or {}
    paired: dict[str, tuple[list[float], ...]] = {}
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
        if not shared:
            continue
        scores = ([float(current[name][d]) for d in shared],
                  [float(candidate[name][d]) for d in shared])
        weights = [w.get(name) or {} for w in (current_w, candidate_w)]
        if all(d in side for side in weights for d in shared):
            current_weights, candidate_weights = ([float(side[d]) for d in shared] for side in weights)
            paired[name] = (*scores, current_weights, candidate_weights)
            continue
        unweighted = [side for side, w in zip(("promoted", "candidate"), weights, strict=True)
                      if not all(d in w for d in shared)]
        note = (
            f"{name}: the {' and '.join(unweighted)} report(s) hold no per-document weights "
            "for it (written before they were kept), so both sides are bootstrapped "
            "unweighted, each document once - not the item-weighted mean the report shows."
        )
        log.warning("%s", note)
        notes.append(note)
        paired[name] = scores
    if not paired:
        _not_paired(notes, f"{promoted_key} and {candidate_key} share no scored document")
    return paired or None


def _unpairable(ctx: StageContext, promoted_key: str, promoted: dict[str, Any],
                candidate_key: str, candidate: dict[str, Any]) -> str | None:
    """Why two eval reports of one yardstick cannot have their per-document
    scores paired; ``None`` when they can. Two yardsticks are refused before
    this (:func:`paired_document_scores`)."""
    for key, report in ((promoted_key, promoted), (candidate_key, candidate)):
        if not report.get("per_document"):
            return f"{key} has no per-document scores (written before they were kept)"
    return None


def _not_paired(notes: list[str], reason: str,
                consequence: str = "Non-inferiority falls back to a point comparison.") -> None:
    note = f"per-document scores were not paired: {reason}. {consequence}"
    log.warning("%s", note)
    notes.append(note)


def eval_set_size(
    ctx: StageContext, manifest: dict[str, Any], warnings: list[str] | None = None,
) -> tuple[dict[str, int], str]:
    """How many frozen REAL documents of each type the candidate was scored on,
    and where the count was read: ``(counts, source)``. Synthetic twins do not
    count toward the production minimum (decision 2026-10-22).

    From the candidate's own eval report when it records them
    (``eval_set.real_documents_by_doc_type``): exactly the documents it was
    scored on. Otherwise from the frozen manifest's
    ``real_documents_by_doc_type``, of the types the adapter trains. Empty when
    neither says - a manifest frozen before it counted real documents, with a
    report that does not either - and the reason is added to ``warnings``: its
    counts include twins, so they cannot show the minimum was met, and the
    release is capped at interim.

    Not the manifest's ``small_set_accepted``: that describes the whole set,
    other types included.
    """
    notes = warnings if warnings is not None else []
    key = paths.eval_report(ctx.out_version, scope=ADAPTER.name)
    report = ctx.client.read_json(key) if ctx.client.exists(key) else {}
    recorded_set = report.get("eval_set") or {}
    recorded = recorded_set.get("real_documents_by_doc_type")
    # The report path names no tenant: another tenant's report says nothing here.
    if (isinstance(recorded, dict)
            and recorded_set.get("tenant_id") == paths._tenant(ctx.tenant_id)):
        return {t: int(recorded.get(t) or 0) for t in ADAPTER.doc_types}, "eval_report"
    if not manifest:
        return {}, "unknown"
    counted = "real_documents_by_doc_type"
    counts = manifest.get(counted)
    if not isinstance(counts, dict):
        note = (
            f"how many real documents the candidate was scored on is unknown: the frozen manifest has "
            f"no {counted} (frozen before the manifest counted real documents apart from synthetic "
            "twins, which do not count toward the minimum, decision 2026-10-22) and the "
            "candidate's eval report does not record them, so the release is capped at interim. "
            "Scoring the candidate again records them in its report."
        )
        log.warning("%s", note)
        notes.append(note)
        return {}, "unknown"
    return {t: int(counts.get(t) or 0) for t in ADAPTER.doc_types}, f"frozen_manifest.{counted}"


def served_document_scores(
    ctx: StageContext, warnings: list[str] | None = None,
) -> tuple[list[float], list[float]] | None:
    """The candidate's per-document ``served_field_match`` over its real
    documents, ``(scores, weights)`` in source-id order, read from its eval
    report (``per_document`` and ``per_document_weights``): what the gate
    bootstraps the tier's lower bound from (``gating.tier_evidence``, decision
    2026-10-26).

    ``None`` when the report is not there, is another tenant's (the report path
    names no tenant), or holds no such scores (written before they were kept,
    or a set with no real document read image_only): the release is then
    interim, never production on missing data, and the reason is added to
    ``warnings`` when the report is another tenant's.
    """
    from evaluation.gating import SERVED_FIELD_MATCH

    notes = warnings if warnings is not None else []
    key = paths.eval_report(ctx.out_version, scope=ADAPTER.name)
    report = ctx.client.read_json(key) if ctx.client.exists(key) else {}
    if not report:
        return None
    tenant = (report.get("eval_set") or {}).get("tenant_id")
    if tenant is not None and tenant != paths._tenant(ctx.tenant_id):
        notes.append(f"the eval report at {key} was scored on tenant {tenant!r}'s set, so its "
                     f"per-document {SERVED_FIELD_MATCH} does not bound this candidate's tier")
        return None
    scores = (report.get("per_document") or {}).get(SERVED_FIELD_MATCH) or {}
    if not scores:
        return None
    weights = (report.get("per_document_weights") or {}).get(SERVED_FIELD_MATCH) or {}
    ids = sorted(scores)
    return ([float(scores[d]) for d in ids], [float(weights.get(d) or 1.0) for d in ids])


def line_evidence(
    ctx: StageContext, parent_run_id: str, warnings: list[str] | None = None,
) -> tuple[Any, str | None]:
    """What a continued run is held to, line by line (decisions 2026-10-23 and
    2026-10-26): ``(gating.LineEvidence, None)``, or ``(None, why)`` when no
    comparison can be made - the candidate's or the parent's eval report
    missing, the parent's on another yardstick (``run_eval.yardstick_difference``),
    or either written before reports recorded each document's line.

    Each line the candidate serves is paired, on the line's real documents both
    reports scored, on the served field match (``served_field_match`` per
    document: image_only readings as served, decision 2026-10-26) - or, for a
    line a reference's report holds none of (written before it was kept), on
    ``field_normalized_match``, with a warning; ``LineComparison.measure``
    records which - with two runs:

    * the PARENT (``compared``), the run it continued;
    * the cumulative-drift reference (``drift``): the last run from the base,
      found by walking ``parent_run_id`` down the chain to the run whose
      manifest names no parent. When that run's report is missing or on another
      yardstick, or the chain cannot be walked back to it, the oldest run of
      the chain on this yardstick is held to instead, and ``drift_note`` says
      so. When that is the parent itself there is no drift check: the parent
      check, at the smaller margin, already covers it.

    Every line with at least one shared document is passed on; whether it has
    enough to be gated is the gate's call (``gating.LINE_MIN_DOCUMENTS``). The
    parent and the last run from the base, when their report is on another
    yardstick only because the scoring code or its settings moved, are held to
    their stored answers re-scored by the code running now
    (``evaluation.rescore.rescored_or_none``, decision 2026-10-26), recorded in
    ``rescored``. A run on the chain whose report is missing or on another
    yardstick is left out with a warning - except the parent, which is required. Lines the parent's
    release serves (its report's ``lines_scored``) and the candidate does not
    are ``dropped``; lines the parent scored that share no document with the
    candidate are ``thin``.
    """
    from evaluation.gating import SERVED_FIELD_MATCH, LineComparison, LineEvidence
    from evaluation.rescore import rescored_or_none
    from evaluation.run_eval import line_document_scores, yardstick_difference
    from registry_utils.query_registry import RegistryQueryError
    from registry_utils.query_registry import get as get_manifest

    notes = warnings if warnings is not None else []
    candidate_key = paths.eval_report(ctx.out_version, scope=ADAPTER.name)
    if not ctx.client.exists(candidate_key):
        return None, f"the candidate has no eval report at {candidate_key}"
    candidate_report = ctx.client.read_json(candidate_key)
    mine_by_line = line_document_scores(candidate_report)
    if mine_by_line is None:
        return None, (f"the candidate's report at {candidate_key} records no document's line "
                      "(written before reports recorded them)")

    # The runs since the last run from the base, parent first, with their
    # per-line document scores; and that run from the base, when the walk
    # reaches it.
    chain: list[tuple[str, dict[str, dict[str, tuple[float, float]]], dict[str, Any]]] = []
    fresh_run: str | None = None
    run_id: str | None = parent_run_id
    seen: set[str] = set()
    rescored: dict[str, dict[str, Any]] = {}
    while run_id and run_id not in seen:
        seen.add(run_id)
        stopped: str | None = None
        try:
            next_run = get_manifest(run_id, ctx.client).parent_run_id
        except (RegistryQueryError, KeyError, FileNotFoundError, ValueError) as exc:
            next_run = None
            stopped = (f"the chain of runs this one continued stops at {run_id}: its run manifest "
                       f"cannot be read ({type(exc).__name__})")
        is_fresh = stopped is None and not next_run
        key = paths.eval_report(version_of(run_id), scope=ADAPTER.name)
        report = ctx.client.read_json(key) if ctx.client.exists(key) else None
        if report is None:
            why = f"it has no eval report at {key}"
        else:
            why = yardstick_difference(report.get("eval_set"), candidate_report.get("eval_set"),
                                       ctx.tenant_id)
            if why and (run_id == parent_run_id or is_fresh):
                # A reference on another yardstick only because the scoring
                # code or its settings moved is re-scored from its stored
                # answers, as the gate's baseline and the continuation are
                # (evaluation.rescore, decision 2026-10-26): a continued cycle
                # whose parent was re-scored to continue it is not then blocked
                # at the gate for the same report.
                said: list[str] = []
                again = rescored_or_none(ctx.client, version_of(run_id), ctx.tenant_id,
                                         candidate_report.get("eval_set"), notes=said)
                if again is not None:
                    rescored[run_id] = {"difference": why, "rescored_from": again.get("rescored_from")}
                    notes.extend(said)
                    report, why = again, None
                elif said:
                    why = f"{why}; {said[-1]}"
        scores = line_document_scores(report) if report is not None and not why else None
        if scores is None and not why:
            why = f"its report at {key} records no document's line (written before reports recorded them)"
        if why:
            if run_id == parent_run_id:
                return None, f"{run_id}: {why}"
            notes.append(f"{run_id}, on the chain of runs this one continued, is not compared line by "
                         f"line: {why}")
        else:
            chain.append((run_id, scores, report))
        if stopped:
            notes.append(stopped)
            break
        if is_fresh:
            fresh_run = run_id
        run_id = next_run

    # What each line is compared on (decision 2026-10-26): the served field
    # match - image_only readings of real documents, as served - and, for a
    # line a reference holds none of, the field match over every mode.
    mine_served = line_document_scores(candidate_report, SERVED_FIELD_MATCH) or {}

    def paired(reference: str, report: dict[str, Any],
               by_line: dict[str, dict[str, tuple[float, float]]], line: str) -> Any:
        theirs_served = line_document_scores(report, SERVED_FIELD_MATCH) or {}
        for measure, mine, theirs in (
                (SERVED_FIELD_MATCH, mine_served.get(line) or {}, theirs_served.get(line) or {}),
                ("field_normalized_match", mine_by_line.get(line) or {}, by_line.get(line) or {})):
            shared = sorted(set(theirs) & set(mine))
            if not shared:
                continue
            if measure != SERVED_FIELD_MATCH:
                notes.append(
                    f"{line} against {reference}: compared on field_normalized_match, the field match "
                    f"over every reading mode - {reference}'s report or the candidate's holds no "
                    f"{SERVED_FIELD_MATCH} for this line's documents (written before it was kept, or "
                    "no image_only reading)")
            return LineComparison(
                line=line, reference=reference,
                reference_scores=[theirs[d][0] for d in shared],
                candidate_scores=[mine[d][0] for d in shared],
                reference_weights=[theirs[d][1] for d in shared],
                candidate_weights=[mine[d][1] for d in shared],
                reference_rescored=reference in rescored,
                measure=measure,
            )
        return None

    parent_scores, parent_report = chain[0][1], chain[0][2]
    parent_lines = set(parent_report.get("lines_scored") or parent_scores)
    served = set(served_lines(ctx))
    evidence = LineEvidence(dropped=sorted(parent_lines - served), added=sorted(served - parent_lines),
                            chain=[run for run, _scores, _report in chain], rescored=rescored)
    for line in sorted(served):
        comparison = paired(parent_run_id, parent_report, parent_scores, line)
        if comparison is not None:
            evidence.compared.append(comparison)
        elif line in parent_lines:
            evidence.thin[line] = 0

    # The cumulative-drift reference: the last run from the base, else the
    # oldest run of the chain on this yardstick.
    oldest, oldest_scores, oldest_report = chain[-1]
    if fresh_run is not None and fresh_run == oldest:
        evidence.drift_reference_is_fresh = True
    else:
        why = (f"{fresh_run}, the last run from the base, has no report on this yardstick"
               if fresh_run is not None else
               "the chain of runs could not be walked back to the run from the base")
        evidence.drift_note = (
            f"the cumulative drift check holds each line to {oldest}, the oldest run of the chain "
            f"on this yardstick: {why}" if oldest != parent_run_id else
            f"no cumulative drift check: {why}, and no run older than the parent is on this yardstick")
    if oldest != parent_run_id:
        evidence.drift_reference = oldest
        evidence.drift = [c for c in (paired(oldest, oldest_report, oldest_scores, line)
                                      for line in sorted(served)) if c is not None]
    return evidence, None


def stage_evaluation_gate(ctx: StageContext) -> StageResult:

    """Score the candidate against the frozen golden eval set, then gate.

    **This stage is a hard stop inside ``package``** (arch v2.1 §13). It runs
    after calibrate, so a regression stops the RELEASE, not the build: the
    adapter exists on the staging volume, but nothing is
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
    # Why per-document scores could not be paired, or the set's size could
    # not be counted, for the decision's warnings.
    pairing: list[str] = []
    # Why the promoted release is on another yardstick, when it is: the
    # candidate is then judged on its floors alone (decision 2026-10-22).
    not_comparable: str | None = None
    # Whether the promoted release's report was re-scored to be compared, or
    # why it could not be (promoted_baseline_report, decision 2026-10-26).
    rescoring: dict[str, Any] | None = None
    if baseline is None:
        baseline = default_baseline_metrics(ctx)
        if baseline is not None:
            not_comparable = baseline_not_comparable(ctx)
            rescoring = baseline_rescoring(ctx)
        # Paired only against the promoted report's own documents: an injected
        # baseline has no per-document scores to pair with.
        if per_document is None and baseline is not None and not not_comparable:
            per_document = paired_document_scores(ctx, warnings=pairing)

    # Read the manifest back from the registry when this run did not train it —
    # a resume, or --from-stage evaluation_gate, leaves ctx.manifests empty, and
    # the gate then saw no parent and silently dropped the line-by-line
    # comparison a continued run is held to. It also skipped recording the
    # decision, while stage_push published the run regardless.
    foundation = manifest_of(ctx)
    if foundation is None:
        from registry_utils.query_registry import RegistryQueryError
        from registry_utils.query_registry import get as get_manifest

        try:
            run_id = ADAPTER.run_id(ctx.out_version)
            foundation = get_manifest(run_id, ctx.client)
            ctx.manifests[ADAPTER.name] = foundation
        except (RegistryQueryError, KeyError, FileNotFoundError) as exc:
            raise PipelineError(
                f"no run manifest for {run_id}, so the gate cannot tell "
                "whether this run continued an adapter — and a continued one is held line by "
                "line to the runs before it (decision 2026-10-23). "
                f"Re-run training for this version rather than gating blind ({exc})."
            ) from exc
    # The set the candidate was scored on. Fewer real documents than the
    # minimum caps the release at interim, with a warning (eval_set_size). A
    # count nothing records is not shown to be large enough.
    eval_set = frozen_manifest(ctx.client, tenant_id=ctx.tenant_id)
    scored_on, counted_from = eval_set_size(ctx, eval_set, warnings=pairing)
    small = not scored_on or any(n < MIN_FROZEN_DOCS_PER_TYPE for n in scored_on.values())
    # A run that continued an adapter: held, line by line, to the runs before it
    # since the last run from the base (line_evidence). One continued by hand
    # from a directory (training.train --continue-from) names no parent run, so
    # there is nothing to compare it with.
    parent_run = getattr(foundation, "parent_run_id", None)
    continued = parent_run or getattr(foundation, "continued_from", None)
    evidence, line_missing = (line_evidence(ctx, parent_run, warnings=pairing)
                              if parent_run else (None, None))
    if continued and not parent_run:
        line_missing = f"it continued {continued}, a directory, not a run of the adapter"
    # The candidate's per-document served field match: the tier's lower bound
    # is bootstrapped from it (decision 2026-10-26).
    served = served_document_scores(ctx, warnings=pairing)
    result = promotion_gate(
        candidate,
        baseline,
        baseline_not_comparable=not_comparable,
        small_eval_set=small,
        warnings=pairing,
        # Per-document scores turn non-inferiority into a paired bootstrap and
        # let the improvement rule be enforced (arch §15.5).
        per_document=per_document,
        fixes_defect=ctx.fixes_defect,
        # A continued run is held, line by line, to the run it continued
        # (decision 2026-10-23): continued training compounds drift.
        continued_from=continued,
        line_evidence=evidence,
        line_evidence_missing=line_missing,
        override=ctx.gate_override,
        served_documents=served,
    )

    # `gate_decision`, not `eval_report`. Writing here used to clobber the
    # scored EvalReport at the same key, taking `by_doc_type` and every error
    # record with it — which is what `vit_gate` reads to decide whether the
    # vision encoder is the bottleneck.
    report_key = paths.gate_decision(ctx.out_version, scope=ADAPTER.name)
    # Before the decision is written, not after: `package --from-stage package`
    # reads the recorded decision, and one saying "passed" for a version the
    # cascade then blocked let it be published.
    cascade = foundation_upgrade_work_list(ctx) if result.passed else None
    cascade_blocked = bool(cascade and cascade.blocking)
    decision = {
        "version": ctx.out_version,
        "candidate_metrics": candidate,
        "gate_metrics": candidate,
        # A promoted release on another yardstick is recorded, never compared:
        # its scores sit apart, with why (decision 2026-10-22).
        "baseline_metrics": None if not_comparable else baseline,
        "baseline_not_comparable": not_comparable,
        "baseline_metrics_not_compared": baseline if not_comparable else None,
        # Its stored answers re-scored by the scoring code running now, so it
        # was compared (decision 2026-10-26); or why they could not be.
        "baseline_rescored": bool(rescoring and rescoring.get("rescored") and not not_comparable),
        "baseline_rescoring": rescoring,
        "passed": result.passed and not cascade_blocked,
        "cascade_blocked": cascade.describe() if cascade_blocked else None,
        "failed_gates": result.failed_gates,
        # The full verdict per metric — floor, interval and basis — not just a
        # delta. "Why was this blocked" needs the evidence, not the difference.
        "verdicts": [v.as_dict() for v in result.verdicts],
        "improved_metrics": result.improved_metrics,
        "waived_gates": sorted(result.waived),
        "override": result.override.as_dict() if result.override else None,
        # How many documents each metric's bootstrap paired: the metrics the
        # gate judged by a paired bootstrap, and no other. A metric paired on
        # one document is compared as a point estimate, so it is not here.
        "paired_documents": {
            v.name: v.interval.documents for v in result.verdicts if v.interval is not None
        },
        # A continued run's line-by-line comparison (decisions 2026-10-23 and
        # 2026-10-26): the runs on the chain, each line's evidence against the
        # parent and against the cumulative-drift reference (documents, change,
        # bound, Holm-adjusted alpha, gated or warned), and the lines not
        # compared - or why no comparison could be made.
        "line_comparison": (
            {"continued": continued, "runs_compared": list(evidence.chain),
             "parent": parent_run, "drift_reference": evidence.drift_reference,
             "drift_reference_is_fresh": evidence.drift_reference_is_fresh,
             "drift_note": evidence.drift_note,
             # The references re-scored from their stored answers (#15), and why.
             "rescored_references": dict(evidence.rescored),
             "lines": list(result.line_verdicts), "not_compared": dict(evidence.thin),
             "dropped": list(evidence.dropped), "added": list(evidence.added)}
            if evidence is not None
            else {"continued": continued, "missing": line_missing} if continued else None
        ),
        "blocking_reasons": list(result.blocking_reasons),
        "fixes_defect": ctx.fixes_defect,
        "eval_set": {
            "small_set_accepted": bool(eval_set.get("small_set_accepted")),
            "documents_by_doc_type": eval_set.get("documents_by_doc_type", {}),
            "real_documents_by_doc_type": eval_set.get("real_documents_by_doc_type"),
            "frozen_from_corpus": eval_set.get("frozen_from_corpus"),
            "frozen_at": eval_set.get("frozen_at"),
            # What decided the interim cap: the real documents per type the
            # candidate was scored on, where that count was read, and the
            # minimum it was held to.
            "scored_documents_by_doc_type": scored_on,
            "scored_documents_counted_from": counted_from,
            "minimum_per_doc_type": MIN_FROZEN_DOCS_PER_TYPE,
            "small": small,
        },
        "warnings": list(result.warnings),
        # Fideon SPEC_09 §6 and decision 2026-10-26: production (served field
        # match >= 0.92, its one-sided 95% lower bound >= 0.90, 150 real
        # documents) or interim; None if blocked, by the gate or by the cascade.
        "tier": result.tier if not cascade_blocked else None,
        # confirmed (the bound >= 0.92 too) or unconfirmed, for production.
        "tier_confidence": result.tier_confidence if not cascade_blocked else None,
        # What the tier rests on: the point estimate, the bound, the real
        # documents it was bootstrapped over, the bars, and why not production.
        "tier_evidence": result.tier_evidence.as_dict() if result.tier_evidence else None,
        # Beside the tier and the golden field match it reads: the selected
        # checkpoint's validation gate mirror (checkpoint selection) - the pair
        # the next cycles' early warning predicts the gate from. Not gated.
        "validation_gate_mirror_accuracy": _validation_gate_mirror(ctx),
    }
    decision["adapter"] = ADAPTER.run_id(ctx.out_version)
    decision["doc_types"] = list(ADAPTER.doc_types)

    ctx.client.write_json(report_key, decision)
    # Also under the release, where the bundle points. The candidate metrics are
    # the bf16 base's with the adapter applied: bf16 is the only format served
    # (decision 2026-10-23).
    ctx.client.write_json(
        paths.release_gate_decision(ctx.release_id, "bf16", ctx.tenant_id), decision
    )
    ctx.volume.write(
        paths.staging_eval_report(ctx.out_version, scope=ADAPTER.name), json.dumps(candidate)
    )

    # The manifest's metrics carry the gate mirror beside the gate's own, and
    # the tier as the decision records it - none when the cascade blocked it -
    # with its confidence and lower bound (EvalMetrics, decision 2026-10-26).
    apply_to_manifest(result, foundation, metrics={
        **candidate, "validation_gate_mirror_accuracy": decision["validation_gate_mirror_accuracy"],
        "tier": decision["tier"], "tier_confidence": decision["tier_confidence"],
        "tier_lower_bound": (decision["tier_evidence"] or {}).get("lower_bound")})
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
        "gate passed" + (" (first version, no baseline)" if result.is_first_version else
                         " (floors only: the promoted release is on another yardstick)"
                         if result.baseline_not_comparable else "")
        # The run summary shows the tier, how sure it is, and the bound.
        + (f"; {result.tier_evidence.describe()}" if result.tier_evidence else ""),
        {"metrics": candidate, "failed_gates": [], "cascade_dependents": cascade.dependents,
         "validation_gate_mirror_accuracy": decision["validation_gate_mirror_accuracy"],
         "tier": decision["tier"], "tier_confidence": decision["tier_confidence"],
         "tier_evidence": decision["tier_evidence"]},
    )


# --------------------------------------------------------------------------
# The selected adapter, as calibration, the gate and package load it
# --------------------------------------------------------------------------


def selected_checkpoint(ctx: StageContext) -> str | None:
    """The checkpoint the §11.2 selector picked, from this process or from Blob.

    ONE reader for calibration, the gate and package: the adapter published must
    be the one scored. A resumed process (``--from-stage calibrate`` or
    ``package``) did not run checkpoint_eval, but an earlier one did and wrote
    its choice down.
    """
    selection = ctx.results.get("checkpoint_eval")
    selected = (selection.data or {}).get("selected") if selection else None
    if selected is None:
        key = paths.checkpoint_selection(ctx.out_version, scope=ADAPTER.name)
        if ctx.client.exists(key):
            selected = ctx.client.read_json(key).get("selected")
    return selected


def adapter_checkpoint(ctx: StageContext) -> str:
    """The checkpoint this version ships, as calibration, the gate and package
    name it: the selected one, else the staged adapter directory. Recorded with
    the eval report and the calibrators, so neither is reused for another."""
    return selected_checkpoint(ctx) or paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version)


def serving_adapter(ctx: StageContext) -> str:
    """Where calibration and the gate load that checkpoint from: the staging
    volume, or with ``--from-blob`` the version's adapter in Blob, which
    :func:`assert_staged` has shown to be that checkpoint, pushed."""
    if ctx.from_blob:
        return paths.scoped_adapter_dir(ADAPTER.name, ctx.out_version)
    return adapter_checkpoint(ctx)


def _record_pushed(ctx: StageContext, source: str) -> None:
    """Note in the checkpoint selection which checkpoint was pushed to Blob, so
    ``package --from-blob`` can tell the adapter there is the one selected."""
    key = paths.checkpoint_selection(ctx.out_version, scope=ADAPTER.name)
    selection = ctx.client.read_json(key) if ctx.client.exists(key) else {}
    ctx.client.write_json(key, {**selection, "pushed": source})


# --------------------------------------------------------------------------
# Stage 7 — calibrate (package)
# --------------------------------------------------------------------------


def _staged_serving_model(ctx: StageContext, fmt: str) -> Any:  # pragma: no cover - needs a GPU
    """The model as it will be served: the bf16 base with this cycle's selected
    adapter applied to every request, never merged (decision 2026-10-23).
    Calibration and the gate score what is served."""
    from inference_core.model_runner import load_model

    if fmt != "bf16":
        raise PipelineError(f"{fmt} is not served: the release is a LoRA on the bf16 base")
    adapter = serving_adapter(ctx)
    return load_model("base", ctx.client, backend="vllm", adapter=adapter,
                      label=f"{ADAPTER.run_id(ctx.out_version)}:bf16")


def collect_calibration_samples(ctx: StageContext) -> dict[str, Any]:
    """The validation split's calibration samples, per staged serving format.

    Per format, never shared: quantization moves the logprob distribution, so
    each format's features come from its own generations (arch v2.1 §5.3).
    Before this, nothing produced calibration samples, and calibrate skipped on
    every real run — every field of every release routed to review.

    The generations are the selected checkpoint's own answers from its
    finalists' pass where they can be (:func:`_finalist_answers`), and only
    the rows they do not cover are generated (:func:`_calibration_generate`).

    Per format, the samples are by reading mode
    (``calibration.population.samples_by_mode``: ``{mode: {half: [(features,
    correct)]}}``), so the stage fits on the modes the profile fits on -
    image_only under fast, every mode under standard.
    """
    samples, _how = _collect_calibration_samples(ctx)
    return samples


def _collect_calibration_samples(ctx: StageContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """:func:`collect_calibration_samples`, and per format how its generations
    were come by: reused from the finalists' pass, generated, or both, and why."""
    from calibration.population import samples_by_mode
    from evaluation.validation_generation import (
        assert_generations_usable,
        read_rows,
        reading_key,
    )

    val_key = paths.corpus_scope_eval_split(ctx.corpus, "val", ADAPTER.name, ctx.tenant_id)
    if not ctx.client.exists(val_key):
        log.warning("no validation split at %s; nothing to calibrate on", val_key)
        return {}, {}
    rows = read_rows(ctx.client.read_text(val_key))

    samples: dict[str, Any] = {}
    how: dict[str, Any] = {}
    for fmt in ctx.formats:
        reused, why = _finalist_answers(ctx, rows, fmt)
        remaining = [row for row in rows if reading_key(row) not in reused] if reused else rows
        generated, made = _calibration_generate(ctx, remaining, fmt, val_key) if remaining else ([], {})
        if reused:
            # Keys are unique here (_finalist_answers refuses a split where they are not).
            answers = {**reused, **{reading_key(g.row): g for g in generated}}
            generations = [answers[reading_key(row)] for row in rows]
        else:
            generations = generated
        how[fmt] = {
            # Which path was taken (review #28): the finalist's answers, fresh
            # generations, or the finalist's plus the rows they do not cover.
            "source": ("finalist_answers" if reused and not remaining
                       else "finalist_answers+generated" if reused else "generated"),
            "why": why, "reused_rows": len(reused), "generated_rows": len(remaining), **made,
        }
        log.info("calibration (%s): %d row(s) reused from the finalists' pass, %d generated (%s)",
                 fmt, len(reused), len(remaining), why)
        assert_generations_usable(generations, what=f"calibration ({fmt})")
        by_mode = samples_by_mode(generations)
        # Only a format with evidence gets an entry. An empty entry reads as
        # "samples supplied" and would fit a calibrator on nothing.
        if any(rows for halves in by_mode.values() for rows in halves.values()):
            samples[fmt] = by_mode
    return samples, how


def _finalist_pass_record(ctx: StageContext) -> dict[str, Any]:
    """Where checkpoint selection kept the selected finalist's answers
    (``finalist_pass`` in its record), from this process or from Blob."""
    selection = ctx.results.get("checkpoint_eval")
    record = (selection.data or {}).get("finalist_pass") if selection else None
    if record is None:
        key = paths.checkpoint_selection(ctx.out_version, scope=ADAPTER.name)
        if ctx.client.exists(key):
            record = ctx.client.read_json(key).get("finalist_pass")
    return dict(record or {})


def _calibration_generation_settings() -> dict[str, Any]:
    """The generation settings calibration generates with: the served engine's
    (``_staged_serving_model`` loads the vLLM backend)."""
    from evaluation.validation_generation import generation_settings
    from inference_core.runner_config import load_runner_config

    return generation_settings(load_runner_config("vllm"))


def _finalist_answers(ctx: StageContext, rows: list[dict[str, Any]], fmt: str) -> tuple[dict[str, Any], str]:
    """The selected checkpoint's answers from its finalists' pass, by reading
    (``reading_key``), restored onto this split's rows - and why, or why none.

    Guarded: reused only when they were made by the checkpoint calibrated
    (``adapter_checkpoint``), in the format served (bf16), with the generation
    settings calibration would generate with, on these rows (each answer's row
    digest); anything else and calibration generates them itself. Nothing is
    reused partly: a single answer to another split refuses them all.
    """
    from evaluation.validation_generation import (
        read_generations,
        reading_key,
        restored_generation,
        row_digest,
    )

    if fmt != "bf16":
        return {}, f"selection generated in bf16, not {fmt}"
    if not _selection_settings(ctx)["calibration_reuses_selection"]:
        return {}, "calibration_reuses_selection is off"
    record = _finalist_pass_record(ctx)
    if not record.get("saved"):
        return {}, record.get("reason") or "checkpoint selection kept no finalist answers"
    text = None
    try:
        if record.get("local") and Path(record["local"]).is_file():
            text = Path(record["local"]).read_text(encoding="utf-8")
        elif record.get("blob") and ctx.client.exists(record["blob"]):
            text = ctx.client.read_text(record["blob"])
    except OSError as exc:
        return {}, f"the finalist answers could not be read: {exc}"
    if text is None:
        return {}, "the finalist answers are neither in the run's working dir nor in Blob"
    header, records = read_generations(text)
    checkpoint = adapter_checkpoint(ctx)
    if header.get("checkpoint") != checkpoint:
        return {}, f"they were made by {header.get('checkpoint')}, and {checkpoint} is calibrated"
    expected = _calibration_generation_settings()
    if header.get("settings") != expected:
        return {}, f"they were generated with other settings ({header.get('settings')}, calibration: {expected})"
    keys = [reading_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        return {}, "two validation rows are the same reading, so an answer cannot be put back on its row"
    by_key = dict(zip(keys, rows, strict=True))
    reused: dict[str, Any] = {}
    for answer in records:
        row = by_key.get(answer.get("key"))
        if row is None or answer.get("digest") != row_digest(row):
            return {}, f"they answer another validation split (first mismatch: {answer.get('key')})"
        reused[answer["key"]] = restored_generation(row, answer)
    return reused, (f"{len(reused)} answer(s) of {checkpoint}'s finalists' pass "
                    f"({', '.join(header.get('modes') or [])}), same checkpoint and generation settings")


def _calibration_gpus() -> list[str]:
    """The GPUs calibration may generate on: those selection scored on."""
    from common.config import training_config
    from evaluation.checkpoint_eval import scoring_gpus
    from training.train import training_gpus

    distributed = training_config(ADAPTER.training_config).get("distributed") or {}
    return scoring_gpus()[: training_gpus(distributed)]


def _calibration_generate(ctx: StageContext, rows: list[dict[str, Any]], fmt: str,
                          val_key: str) -> tuple[list[Any], dict[str, Any]]:
    """``rows`` generated with the served model, in their order, and what it
    cost: sharded across every GPU (``checkpoint_eval.generate_sharded``, the
    selected adapter on the base, as selection generates) when the pod has
    several; else - or when that fails - with one engine here."""
    from evaluation.validation_generation import generate_validation, reading_key
    from inference_core.model_runner import release_model
    from training.stage_data import localize_rows

    images_root = paths.staging_train_images_dir(ctx.corpus, ctx.tenant_id)
    gpus = _calibration_gpus() if ctx.serving_model_loader is None and fmt == "bf16" else []
    if len(gpus) > 1:
        from evaluation.checkpoint_eval import CheckpointEvalError, generate_sharded

        adapter = serving_adapter(ctx)
        localize_rows(rows, ctx.client, images_root)   # fetched once, here, for the workers to read
        try:
            done = generate_sharded(
                rows=rows, val_path=val_key, images_root=images_root, sample_rows=0, full=False,
                full_modes=None, checkpoints=[adapter], gpus=gpus, kind="calibration",
                work_dir=Path(paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version)) / "calibration",
                only_keys=[reading_key(row) for row in rows],
                # --from-blob: the adapter is the version's Blob copy, fetched by each worker.
                prepare_adapters=not Path(adapter).is_dir())[adapter]
        except CheckpointEvalError as exc:
            done = None
            log.warning("calibration (%s): cannot shard the generation (%s); one engine generates it", fmt, exc)
        if done is not None and done.error is None:
            return done.generations, {"gpus": len(gpus), "shards": done.stats}
        if done is not None:
            log.warning("calibration (%s): the sharded generation failed (%s); one engine generates it again",
                        fmt, done.error)
    if ctx.serving_model_loader is None:
        # The real vLLM path opens images by local path; the rows hold Blob keys.
        rows = localize_rows(rows, ctx.client, images_root)
    loader = ctx.serving_model_loader or _staged_serving_model
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
    return generations, {"gpus": 1, **stats}


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
    from calibration.population import (
        HALVES,
        fit_release_calibration,
        production_samples,
        scored_by_field_type,
    )
    from calibration.thresholds import auto_accept_error_rate
    from common.config import STANDARD_PROFILE

    # How each format's generations were come by (review #28): the selected
    # finalist's saved answers, fresh generation, or both - and why.
    generation: dict[str, Any] = {}
    if not ctx.calibration_samples and not ctx.dry_run:
        ctx.calibration_samples, generation = _collect_calibration_samples(ctx)

    if not ctx.calibration_samples:
        # Honest rather than silent: without labelled validation features there
        # is nothing to fit, and shipping an uncalibrated release means every
        # field routes to review. That is a correct outcome, and an operator
        # should know it happened.
        return StageResult(
            "calibrate", "skipped",
            "no labelled validation features supplied, so no calibrator or threshold could be "
            "fitted. Every field will route to review until one is.",
            {"calibrated_formats": [], "unenforced_types": [], "generation": generation},
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

        # By reading mode (collect_calibration_samples), fitted on the modes the
        # profile fits on: image_only under fast - the mode production reads -
        # every mode pooled under standard (calibration.population). Samples
        # supplied as two halves carry no mode: pooled, as before.
        supplied = set(samples) <= set(HALVES)
        by_mode = {"supplied": samples} if supplied else samples
        release = fit_release_calibration(by_mode, release_id=ctx.release_id, serving_format=fmt,
                                          profile=STANDARD_PROFILE if supplied else None)
        calibrators, thresholds = release.calibrators, release.thresholds
        scored_by_type = scored_by_field_type(
            calibrators, production_samples(by_mode, STANDARD_PROFILE if supplied else None)["threshold"])
        ctx.client.write_json(
            paths.release_calibrators(ctx.release_id, fmt, ctx.tenant_id),
            {"calibrators": calibrators.as_dict(), "thresholds": thresholds.as_dict(),
             # Fitted on this checkpoint's generations, and never reused for another.
             "adapter_checkpoint": adapter_checkpoint(ctx)},
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
            # What it was fitted on: the profile, the modes, the rows of each
            # half and mode, and the calibration error in each mode.
            "report": release.report,
        }

    detail = "; ".join(
        f"{fmt}: {len(body['guarantees'])} field type(s), "
        f"auto-accept error {body['auto_accept_error_rate']:.2%}, "
        f"calibration error {body['expected_calibration_error']['all']:.3f}"
        for fmt, body in sorted(fitted.items())
    ) or "nothing fitted"
    for fmt, body in sorted(fitted.items()):
        detail += f"; {fmt} fitted on {', '.join(body['report'].get('fitted_on_modes') or [])}"
    for fmt, body in sorted(generation.items()):
        detail += (f"; {fmt} fitted from {body['source']} ({body['reused_rows']} row(s) reused, "
                   f"{body['generated_rows']} generated)")
    return StageResult("calibrate", "completed", detail, {"by_format": fitted, "generation": generation})


def _calibration_errors(scored_by_type: dict[str, list[tuple[float, bool]]]) -> dict[str, float]:
    """Expected calibration error per field type and over every field ("all")."""
    from evaluation.metrics.coverage_metrics import expected_calibration_error

    def ece(pairs: list[tuple[float, bool]]) -> float:
        return round(expected_calibration_error([c for c, _ in pairs], [ok for _, ok in pairs]), 4)

    out = {field_type: ece(pairs) for field_type, pairs in sorted(scored_by_type.items())}
    out["all"] = ece([pair for pairs in scored_by_type.values() for pair in pairs])
    return out


# --------------------------------------------------------------------------
# Stage 9 — package: push the adapter and write the release bundle
# --------------------------------------------------------------------------


def manifest_of(ctx: StageContext) -> Any:
    """This cycle's run manifest, or ``None`` when this process did not train it."""
    return ctx.manifests.get(ADAPTER.name)



def assert_not_released(ctx: StageContext) -> None:
    """Refuse to package a version that already has a promoted or archived release.

    Packaging it again re-runs its gate against the release live now: a pass
    promoted it with no named person (a rollback by another route), a failure
    overwrote the decision that let it be promoted, so ``promote`` could no
    longer put it back. A rollback is ``promote --version vN``; a new model is a
    new cycle. A ``gated`` release (every release package writes, never promoted
    yet) may be packaged again; its shadow check is then run again, since the
    report names the bundle it read (``registry_utils.releases.shadow_problem``).
    """
    index_key = paths.release_index(ctx.tenant_id)
    rows = ctx.client.read_json(index_key) if ctx.client.exists(index_key) else []
    status = next((r.get("status") for r in rows or [] if r.get("release_id") == ctx.release_id), None)
    if status in ("promoted", "archived"):
        raise PipelineError(
            f"{ctx.release_id} is already released ({status}) for tenant "
            f"{paths._tenant(ctx.tenant_id)}. To serve it again: python -m orchestration.run "
            f"promote --version {ctx.release_id} --by \"<person>\". A new model is a new cycle."
        )


def assert_gate_passed(ctx: StageContext) -> None:

    """Refuse to package a version whose promotion gate did not pass.

    ``finetune`` stops at a blocked gate, but ``package`` is its own command and
    read nothing the gate wrote — so a blocked version could be published by
    running package next. The recorded decision is the authority; an override is
    already folded into its ``passed``.
    """
    key = paths.gate_decision(ctx.out_version, scope=ADAPTER.name)
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

    run_id = ADAPTER.run_id(ctx.out_version)
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
        _assert_in_blob(ctx)
        return
    expected = paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version)
    # The mount itself, as _is_trained reads it: the volume record lives in the
    # process that trained, so a `package` run later - a new process - found
    # nothing recorded and refused a version that was sitting on the disk.
    if ctx.volume.exists(expected) or os.path.isdir(expected):
        return
    raise PipelineError(
        f"version {ctx.out_version} is not on the staging volume — expected {expected}. "
        "The volume is working storage and may have been reclaimed since finetune ran. "
        "Re-run `finetune --version <it> --from-stage training` to rebuild it, or pass --from-blob "
        "if the selected checkpoint was pushed (`finetune --push-adapters`, or an earlier "
        "package run) (IMPL-13 §4)."
    )


def _assert_in_blob(ctx: StageContext) -> None:
    """``--from-blob``: the version's adapter is in Blob, and it is the
    checkpoint selected (the push recorded it in the checkpoint selection), so
    calibration, the gate and the release are all that one checkpoint. A dry
    run loads nothing, so it checks nothing."""
    from artifact_registry.transfer import ADAPTER_FILES

    if ctx.dry_run:
        return
    blob_dir = paths.scoped_adapter_dir(ADAPTER.name, ctx.out_version)
    missing = [name for name in ADAPTER_FILES if not ctx.client.exists(f"{blob_dir}/{name}")]
    if missing:
        raise PipelineError(
            f"--from-blob: {blob_dir} holds no {', '.join(missing)}, so version {ctx.out_version}'s "
            "adapter is not in Blob. Re-run finetune for it, with --push-adapters to keep a copy "
            "in Blob."
        )
    key = paths.checkpoint_selection(ctx.out_version, scope=ADAPTER.name)
    pushed = (ctx.client.read_json(key) if ctx.client.exists(key) else {}).get("pushed")
    wanted = adapter_checkpoint(ctx)
    if pushed != wanted:
        raise PipelineError(
            f"--from-blob: the adapter at {blob_dir} is not shown to be the checkpoint selected for "
            f"{ctx.out_version} ({wanted}): {key} records {pushed or 'no push'}. Calibration, the "
            "gate and the release must all be that checkpoint."
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

    Returns ``(bundle, reasons_not_promotable)``. The bundle is always
    ``gated``: package never promotes (decision 2026-10-26, #44); a person
    does, after a shadow check on the real server path (``orchestration.run
    shadow``, then ``promote``). The reasons say what ``promote`` would refuse
    it for: a serving format with no calibrator or no gate run of its own.
    Listing the gaps on a promotable record would make it claim a guarantee
    nobody measured.
    """
    from common.config import base_model_config
    from common.prompts import prompt_hash
    from registry_utils.models import GateOverride, ReleaseBundle

    root = Path(__file__).resolve().parent.parent
    base = base_model_config()["model"]
    corpus_manifest = ctx.client.read_json(paths.corpus_manifest(ctx.corpus, ctx.tenant_id))
    _assert_corpus_pins(corpus_manifest, f"release {ctx.release_id}")

    formats = list(ctx.formats)
    # The release IS its adapter, served as a LoRA on the bf16 base: nothing is
    # merged, so there is no other format (decision 2026-10-23).
    serving_formats = {fmt: paths.scoped_adapter_dir(ADAPTER.name, ctx.out_version) for fmt in formats}
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

    lines = trained_lines(ctx, corpus_manifest)
    if not lines:
        # An empty list reads as "every line" at serving (a bundle written before
        # lines were recorded), so a release with none would serve every line.
        message = (f"release {ctx.release_id} has no line of business it trained on (no "
                   "single-line policy of a known line in its train view or corpus), so it "
                   "would serve every line.")
        if not ctx.dry_run:
            raise PipelineError(message + " Train it on policies of known lines.")
        log.warning("dry run: %s", message)

    # The training environment's lock (scripts/lock_requirements.sh): every
    # resolved version the model was trained, calibrated and gated with (§10.2).
    # pyproject.toml only if the lock is missing, which records ranges, not versions.
    lock = root / "requirements-train.lock"
    lockfile = lock if lock.exists() else root / "pyproject.toml"

    try:
        bundle = ReleaseBundle(
            release_id=ctx.release_id,
            # Never promoted here (decision 2026-10-26, #44): `shadow`, then `promote`.
            status="gated",
            tier=tier,
            tenant_scope=paths._tenant(ctx.tenant_id),
            scope=ADAPTER.name,
            # What this release may SERVE: the types and lines it trained on.
            # Serving sends a policy of any other line to the base model.
            doc_types=list(ADAPTER.doc_types),
            lines=lines,
            base_model=f"{base['model_id']}@{base['revision']}",
            adapter=ADAPTER.run_id(ctx.out_version),
            served_as="lora_on_base",
            serving_formats=serving_formats,
            calibrators=calibrators,
            gate_reports=gate_reports,
            prompt_hash=prompt_hash(),
            window_plan_version=corpus_manifest.get("window_plan_version"),
            schema_versions={
                str(k): str(v) for k, v in (corpus_manifest.get("schema_versions") or {}).items()
            },
            ocr_pin={
                "mineru_version": corpus_manifest.get("mineru_version"),
                "ocr_device": corpus_manifest.get("ocr_device"),
                # The served MinerU's tier, when the corpus records one
                # (registry_utils.releases.ocr_pin_refusal holds it to the pin).
                **({"mineru_tier": corpus_manifest["mineru_tier"]}
                   if corpus_manifest.get("mineru_tier") else {}),
                # Whether every page was read by OCR, where the corpus recorded it.
                **({"read_every_page_by_ocr": corpus_manifest["read_every_page_by_ocr"]}
                   if "read_every_page_by_ocr" in corpus_manifest else {}),
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


def trained_lines(ctx: StageContext, corpus_manifest: dict[str, Any]) -> list[str]:
    """The lines of business the adapter trained on, which its release serves.

    From the run's own train view (``data_stats.examples_by_line``). Single
    lines only: the view trains no package and no policy without a line
    (``training.corpus_view._filter``), so a line seen only inside a package was
    never trained. Each key is read as its schema's name (``workers_comp`` is
    ``wc``, ``common.lob.lob_lines``).

    A run that recorded no view falls back to the corpus's
    ``policy_line_counts``, which count a package once per line, so a line seen
    only in a package would count there. No run this code trains reaches it:
    ``training.train`` always records the view, and refuses a view with no
    train rows.
    """
    from common.lob import known_lines, lob_lines

    manifest = manifest_of(ctx)
    if manifest is None:
        from registry_utils.query_registry import RegistryQueryError
        from registry_utils.query_registry import get as get_manifest

        try:
            manifest = get_manifest(ADAPTER.run_id(ctx.out_version), ctx.client)
        except (RegistryQueryError, KeyError, FileNotFoundError):
            manifest = None
    by_line = (getattr(getattr(manifest, "data_stats", None), "examples_by_line", None) or {})
    train = by_line.get("train") if isinstance(by_line, dict) else None
    if train:
        found: set[str] = set()
        for key in train:
            if "," not in str(key) and str(key) != "unknown":
                found |= lob_lines(str(key))
    else:
        found = {line for line, count in (corpus_manifest.get("policy_line_counts") or {}).items()
                 if count}
    return sorted(found & known_lines())


def served_lines(ctx: StageContext) -> list[str]:
    """The lines this cycle's release serves (:func:`trained_lines`), from its
    corpus's manifest when the run manifest records none."""
    key = paths.corpus_manifest(ctx.corpus, ctx.tenant_id)
    corpus = ctx.client.read_json(key) if ctx.client.exists(key) else {}
    return trained_lines(ctx, corpus)


def record_release_measurements(ctx: StageContext, bundle: Any) -> None:
    """Latency, GPU memory and the routing table, onto the bundle (Fideon SPEC_09 handoff item 6).

    Latency and memory are read from this release's eval report and recorded,
    not gated. The routing check fails the release step: every line is routed
    through the serving plan as it stands with this release promoted, and a
    line that cannot be routed, or one this release was trained for that would
    not reach it, stops the release here.
    """
    from serving.routing_check import RoutingCheckError, plan_with_candidate, routing_table

    report_key = paths.eval_report(ctx.out_version, scope=ADAPTER.name)
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

    try:
        plan = plan_with_candidate(json.loads(bundle.model_dump_json()))
        bundle.routing = routing_table(plan, bundle.release_id, bundle.lines)
    except RoutingCheckError as exc:
        raise PipelineError(f"{bundle.release_id} is not released: {exc}") from exc


def write_release_bundle(ctx: StageContext, bundle: Any) -> None:
    """Write the bundle and its row in the tenant's release index, ``gated``.

    Package never promotes and archives nothing (decision 2026-10-26, #44):
    it auto-promoted every release that passed its gate, interim tier
    included, and archived the one serving. A person promotes now, after the
    release is read on the real server path beside the promoted one
    (:func:`release_next_steps`: ``orchestration.run shadow``, then
    ``promote``). A dry run never promoted, and still does not.
    """
    from registry_utils.releases import write_index

    written = bundle.model_copy(update={"status": "gated"}) if bundle.status != "gated" else bundle
    ctx.client.write_json(
        paths.release_bundle(bundle.release_id, ctx.tenant_id),
        json.loads(written.model_dump_json()),
    )
    index_key = paths.release_index(ctx.tenant_id)
    rows = ctx.client.read_json(index_key) if ctx.client.exists(index_key) else []
    rows = [r for r in rows if r.get("release_id") != bundle.release_id] + [written.index_row()]
    write_index(ctx.client, rows, ctx.tenant_id)


def release_next_steps(ctx: StageContext, bundle: Any, not_promotable: list[str]) -> list[str]:
    """What the operator runs after ``package`` (decision 2026-10-26, #44): the
    shadow check of the gated release on the real server path, then its
    promotion by a named person. A release ``promote`` would refuse says what
    to fix first; a dry run pushed placeholders, so it has nothing to read."""
    from registry_utils.releases import promoted_release_id
    from scripts.served_adapter import lora_name

    release = bundle.release_id
    if ctx.dry_run:
        return [f"dry run: {release} pushed placeholders, so it is neither shadowed nor promoted"]
    if not_promotable:
        return [f"{release} cannot be promoted yet ({'; '.join(not_promotable)}): package it again once fixed"]
    tenant = f" --tenant {ctx.tenant_id}" if ctx.tenant_id else ""
    current = promoted_release_id(ctx.client, ctx.tenant_id, excluding=release)
    served = (f"{lora_name(release)} and {lora_name(current)} (vllm serve ... --max-loras 2)" if current
              else f"{lora_name(release)}")
    interim = " --accept-interim" if bundle.tier == "interim" else ""
    return [
        f"1. shadow:  python -m orchestration.run shadow --version {release} --model-url <vLLM URL> "
        f"--mineru-url <MinerU URL>{tenant}   (the pod's vLLM serving {served})",
        f"2. promote: python -m orchestration.run promote --version {release} --by \"<person>\"{tenant}{interim}",
    ]


def _clear_staged_scope(ctx: StageContext) -> int:
    """Drop the staged artifacts this version owns, and nothing else."""
    owned = [
        paths.scoped_staging_adapter_dir(ADAPTER.name, ctx.out_version),
        paths.staging_eval_report(ctx.out_version, scope=ADAPTER.name),
        staging_parent_adapter_dir(ctx.out_version),
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
    name = ADAPTER.name
    pushed: dict[str, str] = {}
    blob_dir = paths.scoped_adapter_dir(name, ctx.out_version)
    ctx.client.write_json(f"{blob_dir}/adapter_config.json", {
        "version": ctx.out_version,
        "kind": "scoped",
        "scope": name,
        "doc_types": list(ADAPTER.doc_types),
        "doc_type": None,
    })
    pushed[f"adapter:{name}"] = blob_dir
    return pushed


def _push_weights(ctx: StageContext) -> dict[str, str]:
    """Upload the selected adapter to Blob: it is the release (decision
    2026-10-23) - nothing is merged or quantized.

    Package used to write a one-line JSON at each destination — a placeholder
    ``adapter_config.json`` — and publish the run against it, so a "published"
    release pointed serving at a prefix holding no weights. The upload goes
    through :mod:`artifact_registry.transfer`, which refuses a missing source
    directory rather than creating an empty prefix.
    """
    from artifact_registry.transfer import push_scoped_adapter

    name = ADAPTER.name
    adapter = adapter_checkpoint(ctx)
    if ctx.from_blob:
        # _assert_in_blob showed the adapter there is this checkpoint, pushed.
        blob_dir = paths.scoped_adapter_dir(name, ctx.out_version)
        log.info("%s is in Blob already at %s (pushed from %s)",
                 ADAPTER.run_id(ctx.out_version), blob_dir, adapter)
        return {f"adapter:{name}": blob_dir}
    pushed = {f"adapter:{name}": push_scoped_adapter(adapter, name, ctx.out_version, client=ctx.client)}
    # So a release left gated, its staging cleared, can be packaged again with --from-blob.
    _record_pushed(ctx, adapter)
    return pushed


def stage_push(ctx: StageContext) -> StageResult:
    """Copy the selected adapter into Blob, then flip the manifest from
    ``staged`` to ``published``.

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

    # ONE adapter (arch v2.1 §4.1), and nothing merged: the release is served as
    # a LoRA on the base (decision 2026-10-23).
    #
    # Through the IMPL-02 §3 helper's path, never assembled here. The only place
    # that built these inline is the place that published a Foundation against
    # paths that were never produced.
    if ctx.dry_run:
        pushed.update(_record_dry_run_push(ctx))
    else:
        pushed.update(_push_weights(ctx))

    published: list[str] = []
    for row in list_runs(ctx.client, scope=ADAPTER.name):
        run_id = str(row.get("run_id", ""))
        # Parsed, not suffix-matched: `endswith(version)` once matched every
        # run at that tag.
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
        manifest = get_manifest(run_id, ctx.client)
        manifest.artifacts.eval_report = paths.eval_report(ctx.out_version, scope=ADAPTER.name)
        # The run's adapter is the whole release: nothing is merged.
        mark_published(
            manifest,
            ctx.client,
            adapter_weights=paths.scoped_adapter_dir(ADAPTER.name, ctx.out_version),
        )
        published.append(run_id)

    # The unit of promotion (arch v2.1 §12.3). Built after the artifacts are
    # pushed, so every path it pins exists in Blob.
    write_release_bundle(ctx, bundle)
    for reason in not_promoted:
        log.warning("%s is %s, and promote would refuse it: %s", bundle.release_id, bundle.status, reason)
    next_steps = release_next_steps(ctx, bundle, not_promoted)

    cleared = 0
    if not ctx.keep_staging and published:
        # Only after something was actually published. The comment used to claim
        # the manifest flip verified the push, but nothing checked that any flip
        # happened — so a run where no manifest matched the version deleted every
        # staged adapter, merged model and GGUF and reported success.
        #
        # And only THIS version's paths: clearing the whole staging root would
        # delete another version's staged adapter, the only copy until its
        # own package run publishes it.
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
            # What promote would refuse the release for (package itself never promotes).
            "not_promoted_because": not_promoted,
            # Printed by the run summary (RunReport.render): shadow, then promote.
            "next_steps": next_steps,
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

    The gate's own refusals too: an unreviewed or interrupted frozen set, a
    leaked eval set, a refused override. A retry loaded the model again to
    fail on the same frozen gold.
    """
    import subprocess

    from evaluation.freeze_eval_set import FreezeError
    from evaluation.gating import GateError
    from evaluation.golden_eval import GoldenEvalError
    from evaluation.run_eval import EvalError
    from training.corpus_view import CorpusViewError
    from training.length_check import LengthCheckError
    from training.stage_data import StagingError
    from training.train import TrainingError

    return (TrainingError, StagingError, LengthCheckError, CorpusViewError,
            subprocess.CalledProcessError, GoldenEvalError, EvalError, FreezeError, GateError)


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
    # No merge and no quantize (decision 2026-10-23): the release IS the
    # adapter, served as a LoRA on the bf16 base, so calibration and the gate
    # score the base with the adapter applied - what is served.
    #
    # Calibrate BEFORE the gate: the gate reads auto_accept_error_rate, which is
    # the rate of wrong values that reached a user without a human looking — and
    # that number does not exist until thresholds are chosen (arch v2.1 §5.4).
    Stage(7, "calibrate", "package", True, stage_calibrate, None),
    Stage(8, "evaluation_gate", "package", True, stage_evaluation_gate, None),
    Stage(9, "package", "package", False, stage_push, None),
    Stage(10, "serving", "deploy-endpoint", False, stage_deploy, None),
    Stage(11, "feedback_loop", "extract", False, stage_feedback, None),
)

STAGE_BY_NAME: dict[str, Stage] = {s.name: s for s in STAGES}

#: Stage 3 is human work and stage 6 must re-judge every run, so neither is
#: skippable by an idempotency check. That is deliberate: an "already gated"
#: shortcut would let a re-run inherit a pass it did not earn.
FINETUNE_STAGES = tuple(s for s in STAGES if s.command == "finetune")
PACKAGE_STAGES = tuple(s for s in STAGES if s.command == "package")

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
    # a stage: a stage skipped as a no-op once took the check with it.
    report = RunReport(command=command, version=ctx.out_version)
    preconditions = []
    if command == "package" and stages:
        preconditions += [resolve_corpus_version, assert_staged]
    if any(stage.command == "package" for stage in stages):
        # Before any work: packaging a released version again re-runs its gate
        # against the release live now and overwrites its own decision.
        preconditions.append(assert_not_released)
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
            _write_progress(report, ctx)            # and told, as any stage that failed
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
                # --version.
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
    """``run-<command>-<tenant>-<version>.json``: two tenants' runs keep their own."""
    return f"run-{report.command}-{paths._tenant(ctx.tenant_id)}-{report.version}.json"


def _write_progress(report: RunReport, ctx: StageContext) -> None:
    """The run report so far, written after every stage to
    ``$FIDEON_LOG_DIR`` (default /workspace/logs) when that folder exists: a
    run that dies part-way still says what each finished stage took.

    Called after every stage that ran, so also where each one is told
    (``orchestration.notify.after_stage``): its wall time to the tracker, and a
    webhook post for a stage that failed or a gate that blocked."""
    from orchestration.notify import after_stage

    after_stage(report, ctx)
    folder = Path(os.environ.get(PROGRESS_DIR_ENV) or "/workspace/logs")
    if not folder.is_dir():
        return
    target = folder / progress_report_name(report, ctx)
    try:
        target.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")
    except OSError as exc:  # a report that cannot be written never stops the run
        log.warning("run report not written to %s: %s", target, exc)
