"""``RunManifest`` — the lineage record for every training run (arch §12).

One manifest per run, Foundation and per-type alike, no exceptions. It is the
single record linking *code version + data version + config + resulting metrics
+ artifact location*, and without it these questions have no answer:

* which corpus version produced ``lossrun-adapter-v2.1``?
* why did ACORD accuracy drop between v2 and v3?
* which adapters depend on ``foundation-v2.0`` and must be re-validated now that
  Foundation has moved?

The manifest is written **even when the weights are only staged** on the RunPod
volume, because a volume is working storage with no durability guarantee. Without
that rule, a reclaimed volume means a training run that happened and left no
trace (master §12a).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: ``scoped`` is what every run of the policy adapter writes (``common.adapter``):
#: trained on the base, covering a named set of document types, and filed by
#: its name (``registry/scope/policy/``) rather than by a doc_type.
#:
#: ``unified``, ``foundation`` and ``per_type_adapter`` are the kinds older runs
#: wrote — one adapter over every type, v1's Foundation and its per-type
#: adapters. They stay so those manifests still load, which is the whole point
#: of a lineage record.
RunType = Literal["unified", "foundation", "per_type_adapter", "scoped"]
#: ``training`` exists because the manifest is written BEFORE ms-swift is
#: launched — the run_id has to be reserved and the config recorded even if
#: the run dies. Without it the pre-launch manifest defaulted to "trained",
#: so a pod that OOM'd at step 40 left a registry entry asserting a trained
#: adapter and a staging path holding nothing.
RunStatus = Literal["training", "trained", "evaluated", "promoted", "archived", "failed"]
ArtifactStatus = Literal["staged", "published"]
SweepPhase = Literal["lr", "epochs", "rank"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Dependencies(_Base):
    """Everything needed to reproduce the run.

    Reproducibility is the point: the first time your best model turns out to be
    one you cannot re-create, this is what makes it recoverable.
    """

    base_model: str = Field(..., description="qwen3-vl-8b-instruct@<hf_revision_pin>")
    corpus_version: str = Field(..., description="e.g. corpus/v3")
    code_git_commit: str = Field(..., description="exact repo commit that ran this")

    #: Which Foundation this adapter sits on. None for a Foundation run. This is
    #: what makes the arch §12 dependency-upgrade rule a query rather than an audit.
    foundation_version: str | None = None

    #: The model learns HOW MinerU formats its output, so a version bump is
    #: distribution shift, not a dependency bump (arch §8a).
    mineru_version: str | None = None
    ocr_device: str | None = Field(None, description="cuda | cpu — part of the pin if output differs")

    #: A change to either forces a corpus rebuild and a new training cycle (arch §7).
    schema_version: str | None = None
    prompt_template_version: str | None = None

    @field_validator("base_model")
    @classmethod
    def _revision_must_be_pinned(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError(
                "base_model must pin a revision as 'model_id@revision'. An unpinned base means "
                "two runs can silently use different weights, which makes a regression unattributable."
            )
        return v


class TrainingConfig(_Base):
    """The configuration that actually ran — not what a YAML file says today."""

    #: Required, not defaulted. These three described the *technique* while
    #: silently defaulting to QLoRA / NF4 / paged-8-bit, so a run that held the
    #: base in bf16 and never passed them recorded itself as a 4-bit QLoRA run.
    #: A manifest that misreports the technique is worse than no manifest: the
    #: whole point is that "why did this regress" is answerable from the record,
    #: and base precision is exactly the kind of change that causes a regression.
    technique: Literal["LoRA", "QLoRA"]

    #: ``bf16_frozen_base``, or ``<quant_type>_<double|single>_quant_<dtype>_compute``
    #: under 4-bit. Built by ``training.base_precision.manifest_descriptor`` from
    #: the config that ran, never hand-written at the call site.
    base_quantization: str

    lora_rank: int
    lora_alpha: int
    lora_dropout: float = 0.05
    bias: str = "none"
    learning_rate: float
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: int
    optimizer: str
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int
    effective_batch_size: int
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    target_modules: list[str]

    #: When true it is LoRA-on-ViT, never a full fine-tune: full FT risks
    #: degrading the pretrained OCR ability the image-only pathway depends on.
    vit_trainable: bool = False
    vit_method: Literal["frozen", "lora"] = "frozen"

    resolution_cap_px: int
    #: The image resize budget the trainer was actually given, as passed in its
    #: environment (``MAX_PIXELS`` / ``IMAGE_MAX_TOKEN_NUM`` ...). Recorded because
    #: ``resolution_cap_px`` is the render cap, not what the processor resized to.
    pixel_budget: dict[str, int] | None = None
    max_seq_len: int
    seed: int

    @model_validator(mode="after")
    def _vit_never_fully_fine_tuned(self) -> TrainingConfig:
        if self.vit_trainable and self.vit_method != "lora":
            raise ValueError(
                "vit_trainable is set but vit_method is not 'lora'. The §3 escalation is "
                "'add a ViT LoRA', never 'unfreeze and train the encoder' — full fine-tuning "
                "risks the pretrained document/OCR capability the image-only path relies on."
            )
        return self


class DataStats(_Base):
    """What the run actually trained on."""

    train_examples: int
    val_examples: int
    test_examples: int
    modality_mix: dict[str, float] = Field(default_factory=dict)

    #: Per-LoB-value share (arch §0b). Recorded so a later regression can be
    #: checked against coverage before it is blamed on the model.
    lob_coverage: dict[str, float] = Field(default_factory=dict)

    #: Per canonical field -> observed surface label -> count (arch §0c).
    alias_coverage: dict[str, dict[str, int]] = Field(default_factory=dict)

    #: Documents where a field and one of its confusables co-occur. Zero is a
    #: corpus defect: the model learns the mapping but never the boundary.
    confusable_example_count: int = 0

    tenant_ids: list[str] = Field(default_factory=list)

    #: Carriers held out per line, lines with a single carrier, families moved
    #: into test for the hold-out, and twins dropped by the per-seed cap
    #: (Fideon SPEC_09 amendment items 4 and 5).
    split_policy: dict[str, Any] = Field(default_factory=dict)

    #: The input-mode mix the corpus was drawn against (``modality_mix`` is
    #: what it came to).
    modality_mix_target: dict[str, float] = Field(default_factory=dict)

    #: The configured synthetic_fraction and scanned_share, and per line
    #: what its train documents came to (Fideon SPEC_09 amendment items 6, 7).
    data_mix: dict[str, Any] = Field(default_factory=dict)

    #: split -> line -> documents, real, synthetic and rows, of the run's view.
    examples_by_line: dict[str, dict[str, dict[str, int]]] = Field(default_factory=dict)

    #: Recorded, not enforced, while de-identification is blocked (IMPL-05 §1).
    deidentified: bool = False
    image_redaction: str = "unresolved"


class EvalMetrics(_Base):
    """Scores against the frozen golden eval set (arch §15). All optional —
    a manifest exists from the moment training completes, before evaluation."""

    field_exact_match: float | None = None
    field_normalized_match: float | None = None
    field_f1_list_fields: float | None = None
    list_field_recall: float | None = None
    schema_validity_rate: float | None = None
    ece_confidence: float | None = None
    ocr_arbitration_accuracy: float | None = None
    image_only_accuracy: float | None = None
    scanned_accuracy: float | None = None
    doc_type_classifier_accuracy: float | None = None

    #: Own metric, per value, never folded into the aggregate (arch §0b).
    lob_detection_accuracy: float | None = None
    lob_accuracy_by_value: dict[str, float] = Field(default_factory=dict)

    #: Reported, not gating — rare aliases have too little support to gate on.
    alias_accuracy: dict[str, dict[str, float]] = Field(default_factory=dict)

    #: GATING. Systematic rather than random: it means the model has collapsed
    #: two distinct entities, and the output is fluent enough to pass every
    #: structural check (arch §0c).
    confusable_misattribution_rate: float | None = None

    #: Loss Runs only: claims table F1 and the share of verifiable Loss Runs
    #: whose claims reconciled with their printed totals.
    table_f1: float | None = None
    lossrun_totals_reconciliation_rate: float | None = None

    #: Field match on the documents of carriers held out of train and val in
    #: their line (Fideon SPEC_09 amendment item 4). Reported, not gated.
    held_out_carrier_match: float | None = None

    latency_ms_per_doc: float | None = None


class Artifacts(_Base):
    """Where the outputs live, and whether they are durable yet."""

    #: ``staged`` = on the RunPod volume, no durability guarantee.
    #: ``published`` = in Blob. ``package`` flips this.
    status: ArtifactStatus = "staged"
    staging_path: str | None = None

    adapter_weights: str | None = None
    merged_model: str | None = None
    quantized_model: str | None = None
    quantized_formats: list[str] = Field(default_factory=list)
    eval_report: str | None = None
    calibration_params: str | None = None

    @model_validator(mode="after")
    def _published_artifacts_have_a_path(self) -> Artifacts:
        if self.status == "published" and not (self.adapter_weights or self.merged_model):
            raise ValueError(
                "artifacts.status is 'published' but no Blob path is recorded. A published "
                "manifest that points nowhere is worse than a staged one — it claims durability "
                "the artifact does not have."
            )
        return self


class Promotion(_Base):
    """The gate decision. A candidate is promoted only if it matches or beats the
    current production version on **every** gating metric — no override path."""

    gated_against: str | None = None
    beat_previous_on_all_gates: bool | None = None
    failed_gates: list[str] = Field(default_factory=list)
    promoted_by: str | None = None
    promoted_at: datetime | None = None
    #: Fideon SPEC_09 §6: production (field match >= 0.92) or interim; None if
    #: not gated or blocked.
    tier: Literal["production", "interim"] | None = None


class RunManifest(_Base):
    """One training run, completely described."""

    run_id: str
    run_type: RunType

    #: The §4.2 graduated per-type adapter this run IS — never the set it covers.
    #: Kept exactly as it was so `index_row`, `list_runs(doc_type=)` and
    #: `adapters_depending_on` keep meaning what they meant.
    doc_type: str | None = None

    #: The adapter's name for a run of it (``common.adapter``: ``policy``), or
    #: the training scope of an older run. ``None`` reads as ``unified``, which
    #: is correct for every manifest written before scopes existed.
    scope: str | None = None

    #: The document types this run actually trained on. Empty reads as "the
    #: active set at the time", again correct for existing manifests. This is
    #: what a scoped run records instead of ``doc_type``.
    doc_types: list[str] = Field(default_factory=list)

    tenant_id: str | None = None
    status: RunStatus = "trained"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    #: Sweep runs are first-class registry entries, not untracked side
    #: experiments (arch §11a). Sweeps are deferred until after the pilot; the
    #: fields exist now so nothing changes when they are turned on.
    is_sweep_run: bool = False
    sweep_id: str | None = None
    sweep_phase: SweepPhase | None = None

    #: The adapter directory this run's weights started from, when it continued
    #: training rather than starting from the base (``training.train
    #: --continue-from``, or the parent's adapter a cycle continues).
    continued_from: str | None = None

    #: The run whose adapter this one continued (``policy-v3``), when a cycle
    #: trained incrementally (decision 2026-10-23). Its documents are this run's
    #: too: the purge plan follows the chain, and the gate holds every line to
    #: its parent's score.
    parent_run_id: str | None = None

    #: Why this cycle started from the base rather than continuing the promoted
    #: adapter (``--fresh``, nothing promoted, another window plan ...). None
    #: when it continued, or for a run written before cycles continued.
    fresh_reason: str | None = None

    dependencies: Dependencies
    training_config: TrainingConfig
    data_stats: DataStats
    eval_metrics: EvalMetrics = Field(default_factory=EvalMetrics)
    artifacts: Artifacts = Field(default_factory=Artifacts)
    promotion: Promotion = Field(default_factory=Promotion)

    @model_validator(mode="after")
    def _consistency(self) -> RunManifest:
        if self.run_type == "per_type_adapter":
            if not self.doc_type:
                raise ValueError("a per_type_adapter run must name its doc_type")
            if not self.dependencies.foundation_version:
                raise ValueError(
                    "a per_type_adapter must record the foundation_version it was trained on. "
                    "Without it, 'which adapters does this Foundation bump invalidate?' becomes "
                    "a manual audit instead of a query (arch §12)."
                )
        if self.run_type in ("unified", "foundation") and self.doc_type:
            # Kept exactly as it was. The guarantee — a run claiming to span
            # everything must not secretly be one type — matters MORE once scopes
            # exist, not less. A narrower run is `scoped`, which says so.
            raise ValueError(
                f"a {self.run_type} run spans all doc types and must not name one"
            )
        if self.run_type in ("unified", "foundation") and self.scope not in (None, "unified"):
            raise ValueError(
                f"a {self.run_type} run is the unified scope; it cannot name scope "
                f"{self.scope!r}. A narrower run is run_type='scoped'."
            )
        if self.run_type == "scoped":
            if not self.scope:
                raise ValueError(
                    "a scoped run must name its scope — it is the run-id lineage and the "
                    "artifact path segment, so without it the artifacts are unaddressable"
                )
            if not self.doc_types:
                raise ValueError(
                    f"scoped run {self.run_id!r} records no doc_types. What it covers is the "
                    "one thing a scoped run exists to state, and the gate reads it to decide "
                    "which metrics are not applicable."
                )
            if self.doc_type:
                raise ValueError(
                    "a scoped run records doc_types (what it covers), not doc_type (the §4.2 "
                    "graduated adapter it would be). Naming both makes the lineage ambiguous."
                )
        if self.is_sweep_run and not self.sweep_id:
            raise ValueError("a sweep run must carry its sweep_id to be groupable")
        if self.status == "promoted" and not self.promotion.promoted_at:
            raise ValueError("a promoted run must record when, and by whom, it was promoted")
        return self

    def index_row(self) -> dict[str, Any]:
        """The flat row for ``registry_index.json`` — the whole history at a glance."""
        return {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "doc_type": self.doc_type,
            # Absent on every row written before scopes existed; readers take
            # `row.get("scope") or "unified"`, which is what those rows mean.
            "scope": self.scope or "unified",
            "doc_types": list(self.doc_types),
            "status": self.status,
            "artifact_status": self.artifacts.status,
            "created_at": self.created_at.isoformat(),
            # Whose corpus trained it, and the gate's tier: the version table's columns.
            "tenant_id": self.tenant_id,
            "tier": self.promotion.tier,
            "corpus_version": self.dependencies.corpus_version,
            "foundation_version": self.dependencies.foundation_version,
            "code_git_commit": self.dependencies.code_git_commit,
            "is_sweep_run": self.is_sweep_run,
            # Which version this one continued; None for a run from the base.
            "parent_run_id": self.parent_run_id,
            "field_exact_match": self.eval_metrics.field_exact_match,
            "list_field_recall": self.eval_metrics.list_field_recall,
            "lob_detection_accuracy": self.eval_metrics.lob_detection_accuracy,
            "confusable_misattribution_rate": self.eval_metrics.confusable_misattribution_rate,
        }


# --------------------------------------------------------------------------
# Release bundles (arch v2.1 §12.3)
# --------------------------------------------------------------------------

ReleaseStatus = Literal["candidate", "gated", "promoted", "archived", "rejected"]


class GateOverride(_Base):
    """A recorded decision to promote past a failed gate (arch v2.1 §15.5).

    v1 had no override at all, on the reasoning that a gate which can be waived
    stops being a guarantee. That was right about the risk and wrong about the
    remedy: the v1 gate demanded improvement on twelve metrics within 0.001,
    which at pilot volume is finer than the eval set can resolve, so nothing
    could ever be promoted and the rule would have been broken in practice
    rather than in the open.

    An override is therefore allowed and **recorded** — named approver, written
    reason, and the specific gates waived. A waiver nobody can attribute later is
    the thing actually worth preventing.
    """

    approver: str = Field(..., description="a person, not a service account")
    reason: str = Field(..., min_length=20, description="why this ships despite the gate")
    waived_gates: list[str] = Field(..., min_length=1)
    approved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("approver")
    @classmethod
    def _named_person(cls, v: str) -> str:
        if not v.strip() or v.strip().lower() in {"system", "ci", "automated", "n/a", "none"}:
            raise ValueError(
                "a gate override must name the person accountable for it, not a system identity"
            )
        return v.strip()


class ReleaseBundle(_Base):
    """What is gated, promoted, served, and selected by ``--model``.

    ``--model v2`` was ambiguous: it named weights but not the prompt that
    rendered their input, the schema they were trained against, the calibrators
    that turn their logprobs into confidence, or the OCR version whose formatting
    the model learned. Changing any one of those changes behaviour, so the unit
    of promotion has to pin all of them together (arch v2.1 §12.3).

    **Calibrators and gate reports are per serving format.** Quantization moves
    the logprob distribution, so a calibrator fitted on bf16 is wrong for FP8 —
    which is why every format gets its own fit and its own gate run, and why
    these are maps rather than single values.
    """

    release_id: str = Field(..., description="the cycle's version, e.g. v3")

    status: ReleaseStatus = "candidate"
    #: Fideon SPEC_09 §6: ``production`` when field match >= 0.92, ``interim`` when the
    #: release passed every floor below that (evaluation.gating.release_tier).
    #: ``None`` on bundles written before tiers existed.
    tier: Literal["production", "interim"] | None = None
    #: Recorded, not gated (Fideon SPEC_09 handoff item 6): P95 milliseconds of
    #: one generation call per adapter, and the peak GPU memory, from the
    #: golden eval; and where every line of business routes once this release
    #: is promoted.
    latency_p95_ms_by_adapter: dict[str, Any] = Field(default_factory=dict)
    peak_gpu_memory_mb: float | None = None
    measured_with: str | None = None
    routing: list[dict[str, Any]] = Field(default_factory=list)
    tenant_scope: str

    #: The adapter behind this release (``common.adapter``: ``policy``), or the
    #: training scope of an older one. Defaulted so every bundle already in Blob
    #: loads unchanged; serving reads it to find the adapter's weights.
    scope: str = "unified"

    #: The document types this release may SERVE. **Empty means "every active
    #: type"**, which is what a bundle written before types were recorded means.
    doc_types: list[str] = Field(default_factory=list)
    #: The lines of business this release serves: the lines it trained on.
    #: Serving sends a policy of any other line to the base model. Empty means
    #: every line — every bundle written before lines were recorded.
    lines: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    base_model: str = Field(..., description="Qwen/Qwen3-VL-8B-Instruct@<hf_revision>")
    adapter: str = Field(..., description="the run whose adapter this is, e.g. policy-v3")
    #: The merged model's path, for a release served as one. Empty for a
    #: ``lora_on_base`` release: nothing is merged (decision 2026-10-23).
    merged_model: str = ""

    #: How the release is served. ``lora_on_base``: the base model with this
    #: release's adapter applied per request, never merged (every release since
    #: 2026-10-23; ``serving_formats.bf16`` is then the adapter's directory).
    #: ``merged``: one merged model (every bundle written before).
    served_as: Literal["merged", "lora_on_base"] = "merged"

    #: format -> artifact path. bf16 is the reference and must always be present:
    #: every quantization threshold in §13b is an absolute margin against it, so
    #: a bundle without it has nothing to measure its own FP8 drop from.
    serving_formats: dict[str, str] = Field(..., min_length=1)
    calibrators: dict[str, str] = Field(default_factory=dict)
    gate_reports: dict[str, str] = Field(default_factory=dict)

    prompt_hash: str
    #: The window plan the corpus behind a policy release was cut with
    #: (``policy_windows.window_plan_id``). Serving refuses a policy release
    #: whose plan is not the one its code reads with: the prompt hash cannot
    #: tell, since budgets and the window rule change windows, not prompts.
    window_plan_version: str | None = None
    schema_versions: dict[str, str] = Field(default_factory=dict)
    ocr_pin: dict[str, Any] = Field(default_factory=dict)
    vision_config_hash: str
    vllm_config_hash: str

    #: ms-swift, transformers, peft, torch, flash-attn and vllm pinned together.
    #: Recorded because the trainer stack's behaviour is version-dependent and a
    #: run reproduced against different pins is not the same run (§10.2).
    lockfile_hash: str

    override: GateOverride | None = None

    @model_validator(mode="after")
    def _consistency(self) -> ReleaseBundle:
        if "bf16" not in self.serving_formats:
            raise ValueError(
                "every release must carry the bf16 reference: the §13b quantization thresholds "
                "are absolute margins against it, so without it a quantized format has no "
                "baseline to be measured against."
            )
        if self.served_as == "lora_on_base" and set(self.serving_formats) != {"bf16"}:
            raise ValueError(
                f"a release served as a LoRA on the bf16 base serves bf16 only, not "
                f"{sorted(self.serving_formats)}: nothing is merged, so there is no model to quantize"
            )
        if self.served_as == "merged" and not self.merged_model:
            raise ValueError("a release served as a merged model must name it (merged_model)")
        unknown = set(self.calibrators) - set(self.serving_formats)
        if unknown:
            raise ValueError(
                f"calibrators {sorted(unknown)} name formats this release does not serve"
            )
        if self.status == "promoted":
            missing = sorted(set(self.serving_formats) - set(self.calibrators))
            if missing:
                raise ValueError(
                    f"formats {missing} are served with no calibrator. Confidence would be raw "
                    "logprobs, which are systematically overconfident, and every review threshold "
                    "downstream is defined against a calibrated score (arch v2.1 §5.3)."
                )
            ungated = sorted(set(self.serving_formats) - set(self.gate_reports))
            if ungated and not self.override:
                raise ValueError(
                    f"formats {ungated} were promoted without a gate run. Quantization degrades "
                    "exactly what was fine-tuned in, so a format inherits nothing from bf16's "
                    "result (arch v2.1 §13a)."
                )
        return self

    def index_row(self) -> dict[str, Any]:
        """The flat row for the release index."""
        return {
            "release_id": self.release_id,
            "status": self.status,
            "tenant_scope": self.tenant_scope,
            "scope": self.scope,
            "doc_types": list(self.doc_types),
            "created_at": self.created_at.isoformat(),
            "adapter": self.adapter,
            "base_model": self.base_model,
            "serving_formats": sorted(self.serving_formats),
            "overridden": self.override is not None,
            "override_approver": self.override.approver if self.override else None,
        }
