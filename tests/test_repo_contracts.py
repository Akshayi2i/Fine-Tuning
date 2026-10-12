"""Repo-level contracts (IMPL-01 §§11-14, IMPL-03, IMPL-12).

The checks nothing else covers because they are about the repository rather than
a module: the config files carrying every parameter the architecture specifies,
``.env.example`` naming every variable the code reads, and the provenance record
that ties an extraction result back to the run that produced it.

The env-var check is here because its absence already cost something: the
``ALLOW_EXTERNAL_PREANNOTATION`` guard was documented in ``.env.example`` and
read by no module, so the compliance refusal it describes did not exist. A
documented variable nobody reads and a read variable nobody documents are the
same class of bug in opposite directions, and both are caught below.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from common.config import load_yaml

ROOT = Path(__file__).resolve().parent.parent

CODE_PACKAGES = (
    "common", "artifact_registry", "registry_utils", "data_pipeline", "inference_core",
    "training", "evaluation", "calibration", "serving", "postprocessing", "testing",
    "orchestration", "pilot",
)


# --------------------------------------------------------------------------
# The repo tree (IMPL-01 §1, master §5)
# --------------------------------------------------------------------------


REQUIRED_DIRS = (
    "common", "configs", "configs/training", "configs/sweeps", "configs/inference",
    "configs/deepspeed", "schemas", "schemas/examples", "schemas/aliases", "prompts",
    "artifact_registry", "registry_utils", "data_pipeline", "data_pipeline/ingestion",
    "data_pipeline/ocr", "data_pipeline/labeling", "data_pipeline/dataset_builder",
    "training", "training/callbacks", "inference_core", "evaluation", "evaluation/metrics",
    "calibration", "postprocessing", "serving", "testing", "testing/prompts",
    "orchestration", "orchestration/config", "pilot", "tests", "tests/fixtures",
)


@pytest.mark.parametrize("relative", REQUIRED_DIRS)
def test_the_repo_tree_matches_the_master_layout(relative: str):
    assert (ROOT / relative).is_dir(), f"{relative}/ is missing from the tree (master §5)"


def test_every_code_package_is_importable_as_a_package():
    """A directory without ``__init__.py`` imports by accident under one runner
    and fails under another."""
    for package in CODE_PACKAGES:
        assert (ROOT / package / "__init__.py").exists(), f"{package} has no __init__.py"


def test_every_package_is_declared_for_packaging():
    """A package missing from pyproject installs as nothing on a pod, and the
    failure surfaces as an ImportError halfway through a training run."""
    import tomllib

    declared = set(tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
                   ["tool"]["setuptools"]["packages"])
    missing = [p for p in CODE_PACKAGES if p not in declared]
    assert not missing, f"packages not declared in pyproject: {missing}"


# --------------------------------------------------------------------------
# .env.example completeness (IMPL-01 §14)
# --------------------------------------------------------------------------


#: Variables consumed by third-party libraries rather than by this code. They
#: belong in the template even though no module names them.
LIBRARY_OWNED_VARS = frozenset({
    "HF_TOKEN", "HF_MODEL_REVISION", "WANDB_API_KEY", "MLFLOW_TRACKING_URI",
    "RUNPOD_ENDPOINT_ID",
    # MinerU's own: the name of its config file under the home directory.
    "MINERU_TOOLS_CONFIG_JSON",
    # Hugging Face's cache location, kept on the volume on the pod.
    "HF_HOME",
    # Hugging Face: RunPod images enable hf_transfer, which our environments lack.
    "HF_HUB_ENABLE_HF_TRANSFER",
})


#: Set by the runtime around the code, never by an operator: tmux sets TMUX in
#: every session it runs. Documenting them in .env.example would invite setting
#: them by hand, which would make a foreground job believe it was detached.
RUNTIME_SET_VARS = frozenset({
    "TMUX",
    # The OS's own: read so ms-swift is found next to the running Python.
    "PATH",
    # CUDA's own, set by the job launcher (scripts/pod_run.sh) or a scheduler:
    # which GPUs checkpoint selection may run its engines on. Never in .env -
    # the pod sources it, and an empty value there would hide every GPU.
    "CUDA_VISIBLE_DEVICES",
    # vLLM's own, set by the code for the engine it builds when generation is
    # asked to be repeatable (inference_core.model_runner) - never by a person.
    "VLLM_BATCH_INVARIANT",
})


def env_vars_in_code() -> set[str]:
    """Every environment-variable name the code names.

    Matches the quoted name anywhere in a module rather than only inside an
    ``env(...)`` call, because a name is just as much "read" when it is bound to
    a module constant first — which is exactly how the external-pre-annotation
    guard is written, and a call-site-only scan would report it as unread.
    """
    pattern = re.compile(r'["\']([A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,})["\']')
    found: set[str] = set()
    for package in CODE_PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            found.update(pattern.findall(path.read_text(encoding="utf-8")))
    return found


def env_vars_in_template() -> set[str]:
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z][A-Z0-9_]{2,})=", text, re.M))


def test_every_env_var_the_code_reads_is_documented():
    """An undocumented variable is one nobody sets, and the default it falls
    back to is rarely the one anybody intended."""
    read_at_a_call_site = re.compile(
        r'(?:env|os\.environ\.get|os\.getenv)\(\s*["\']([A-Z][A-Z0-9_]{2,})["\']'
        r'|os\.environ\[\s*["\']([A-Z][A-Z0-9_]{2,})["\']'
    )
    called: set[str] = set()
    for package in CODE_PACKAGES:
        for path in (ROOT / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for a, b in read_at_a_call_site.findall(path.read_text(encoding="utf-8")):
                called.add(a or b)

    undocumented = called - env_vars_in_template() - RUNTIME_SET_VARS
    assert not undocumented, (
        f"read by the code but absent from .env.example: {sorted(undocumented)}"
    )


def test_every_documented_env_var_is_read_by_something():
    """The mirror-image bug, and the one that actually bit: a documented guard
    that no module reads is a guard that does not exist."""
    orphaned = env_vars_in_template() - env_vars_in_code() - LIBRARY_OWNED_VARS
    assert not orphaned, (
        f"documented in .env.example but read by no module: {sorted(orphaned)}. Either wire it or "
        "remove it — a variable that only exists in the template describes behaviour the code "
        "does not have."
    )


def test_no_secret_value_is_committed_in_the_template():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "AccountKey=..." in text or "AccountKey=" in text
    for line in text.splitlines():
        if line.startswith(("RUNPOD_API_KEY", "HF_TOKEN", "WANDB_API_KEY")):
            assert line.split("=", 1)[1].strip() == "", f"{line!r} carries a value"


# --------------------------------------------------------------------------
# Training config completeness (IMPL-01 §11, arch §11 full tables)
# --------------------------------------------------------------------------


#: Every parameter arch §11's full specification tables name, by config section.
#: The summary table gives the shape; these are what the entrypoints assemble,
#: and a config missing one silently takes a library default instead.
REQUIRED_TRAINING_PARAMS: dict[str, tuple[str, ...]] = {
    "optimization": (
        "learning_rate", "lr_scheduler_type", "num_train_epochs",
        "optim", "adam_beta1", "adam_beta2", "adam_epsilon", "weight_decay", "max_grad_norm",
    ),
    "batch": (
        "per_device_train_batch_size", "gradient_accumulation_steps", "effective_batch_size",
    ),
    "lora": ("rank", "alpha", "dropout", "target_modules", "bias"),
}

#: ONE config under arch v2.1 §4.1. No per-type training config exists until a
#: type passes the §4.2 graduation gate.
TRAINING_CONFIGS = ("unified",)


@pytest.mark.parametrize("name", TRAINING_CONFIGS)
def test_training_config_carries_every_specified_parameter(name: str):
    config = load_yaml(ROOT / "configs" / "training" / f"{name}.yaml")
    for section, params in REQUIRED_TRAINING_PARAMS.items():
        assert section in config, f"{name}.yaml has no {section} section"
        missing = [p for p in params if p not in config[section]]
        assert not missing, (
            f"{name}.yaml {section} is missing {missing}. An absent parameter takes a library "
            "default rather than the specified value, and the manifest then records a config that "
            "is not the one that ran (arch §11)."
        )


def test_there_is_exactly_one_launch_path_training_config():
    """v1 shipped four — a Foundation plus one per document type — and they
    drifted. The topology that needed them is gone: vLLM applies one LoRA per
    request, so a Foundation and a per-type adapter could never both be active
    (arch v2.1 §4.1)."""
    configs = {p.stem for p in (ROOT / "configs" / "training").glob("*.yaml")}
    # An adapter's config (Fideon SPEC_09 amendment item 7: per-adapter data
    # mix) is no second launch path: it extends unified and may name only its
    # data, so it cannot drift from it on anything that trains.
    adapters = {name for name in configs
                if load_yaml(ROOT / "configs" / "training" / f"{name}.yaml").get("extends") == "unified"}
    for name in adapters:
        own = set(load_yaml(ROOT / "configs" / "training" / f"{name}.yaml")) - {"extends"}
        assert own <= {"data_mix", "modality_mix"}, f"{name}.yaml overrides {sorted(own)} of unified"
    assert configs - adapters <= {"unified", "per_type_adapter", "sweep"}, (
        f"unexpected training configs: {sorted(configs - adapters - {'unified', 'per_type_adapter', 'sweep'})}"
    )
    assert "unified" in configs


def test_the_learning_rate_is_the_v2_1_value():
    """1e-4. v1 ran the Foundation at 2e-4, which is aggressive for ~25 documents
    per type and a rank-64 adapter (arch v2.1 §11.1)."""
    config = load_yaml(ROOT / "configs" / "training" / "unified.yaml")
    assert float(config["optimization"]["learning_rate"]) == 1.0e-4


def test_warmup_is_counted_in_steps_not_a_ratio():
    """At pilot volume a 0.03 ratio over a handful of steps rounds to zero
    warmup, and the first optimizer step lands at full learning rate on a
    freshly-initialised adapter (arch v2.1 §11.1)."""
    optimization = load_yaml(ROOT / "configs" / "training" / "unified.yaml")["optimization"]
    assert "warmup_steps" in optimization
    assert "warmup_ratio" not in optimization


def test_weight_decay_is_zero_on_a_low_rank_adapter():
    """Decay pulls B toward zero, which is where it starts — a shrinkage prior on
    the update itself, not the regularisation it is on a full fine-tune. Dropout
    does that job here (arch v2.1 §11.1)."""
    optimization = load_yaml(ROOT / "configs" / "training" / "unified.yaml")["optimization"]
    assert float(optimization["weight_decay"]) == 0.0
    assert float(load_yaml(ROOT / "configs" / "training" / "unified.yaml")["lora"]["dropout"]) >= 0.05


def test_the_effective_batch_is_small_enough_to_produce_updates():
    """At ~25 documents per type an effective batch of 32 is most of an epoch in
    one step, so a run sees a handful of updates and the cosine schedule never
    gets anywhere (arch v2.1 §11.1)."""
    batch = load_yaml(ROOT / "configs" / "training" / "unified.yaml")["batch"]
    assert int(batch["effective_batch_size"]) <= 8


def test_the_mergers_are_not_a_lora_target():
    """A reversal of v1, which appended "merger" to the target list. Tower and
    connector LoRA in vLLM is experimental with known mixed-adapter batching
    risks, and arbitration is learned in the DECODER, where image tokens and OCR
    text tokens attend to each other — the mergers never see OCR text at all
    (arch v2.1 §9a)."""
    lora = load_yaml(ROOT / "configs" / "training" / "unified.yaml")["lora"]
    assert lora["include_vision_projector"] is False
    assert "merger" not in lora["target_modules"]
    assert set(lora["target_modules"]) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }


def test_the_memory_settings_that_make_the_largest_cap_affordable_are_on():
    """Without use_logits_to_keep the LM head produces a 151k-vocabulary
    distribution at every position of a 32k sequence, which dominates activation
    memory on its own (arch v2.1 §9.3)."""
    config = load_yaml(ROOT / "configs" / "training" / "unified.yaml")
    assert config["batch"]["gradient_checkpointing"] is True
    assert config["batch"]["bf16"] is True
    memory = config["memory"]
    assert memory["use_logits_to_keep"] is True
    assert memory["padding_free"] is True
    # Off: one row per device with padding_free leaves no padding to save, and it
    # made ms-swift encode every row before step 1 (1.5 h on the CGL full run).
    assert memory["length_grouped_sampling"] is False


def test_deepspeed_is_not_the_first_reach_for_vram():
    """Under LoRA, ZeRO-2 shards ~1.5GB of adapter optimizer state and is close
    to a no-op. Sequence parallelism comes first (arch v2.1 §9.3)."""
    distributed = load_yaml(ROOT / "configs" / "training" / "unified.yaml")["distributed"]
    assert distributed["deepspeed_config"] is None
    assert "sequence_parallel_size" in distributed


def test_checkpoint_retention_covers_what_the_selector_scores():
    """checkpoint_eval generates over the last 3 checkpoints plus the best-loss
    one, so fewer than 4 would discard a candidate before it is scored
    (arch v2.1 §11.2)."""
    evaluation = load_yaml(ROOT / "configs" / "training" / "unified.yaml")["evaluation"]
    assert int(evaluation["save_total_limit"]) >= 4


def test_early_stopping_selects_on_loss_and_says_so():
    """Field F1 needs generation, which the training loop's eval does not do
    efficiently for a VLM. Loss here is for early stopping only — a separate vLLM
    job selects what ships (arch v2.1 §11.2)."""
    evaluation = load_yaml(ROOT / "configs" / "training" / "unified.yaml")["evaluation"]
    assert evaluation["metric_for_best_model"] == "eval_loss"
    assert evaluation["greater_is_better"] is False


def test_the_base_is_held_in_bf16_by_default():
    """arch §9: LoRA on a bf16 base, not QLoRA. Both serving paths — the merged
    model (IMPL-10) and vLLM's LoRA hot-swap (IMPL-11) — hold the base in
    bf16/fp16, so a 4-bit training base means the adapter compensates for
    quantization error in weights it is never served against.

    This asserts the DEFAULT, not the only permitted value: `load_in_4bit` is a
    live flag for VRAM-constrained pods. What it prevents is the flag flipping
    without the arch §9 rationale and the pod-class guidance moving with it."""
    quant = load_yaml(ROOT / "configs" / "base_model.yaml")["quantization"]
    assert quant["load_in_4bit"] is False
    # The bnb_* keys stay populated so flipping the flag needs no other edit.
    assert quant["bnb_4bit_quant_type"] == "nf4"
    assert quant["bnb_4bit_use_double_quant"] is True


@pytest.mark.parametrize("name", TRAINING_CONFIGS)
def test_the_optimizer_is_not_a_bitsandbytes_paged_one(name: str):
    """`paged_adamw_8bit` was carried over from the QLoRA default. Under bf16
    LoRA only ~150-200M params train, so fp32 AdamW state is affordable and
    removes a second quantization approximation from the loop. Paging also never
    addressed the long-sequence spikes it was commented as addressing — those are
    activation and logit spikes, not optimizer state."""
    config = load_yaml(ROOT / "configs" / "training" / f"{name}.yaml")
    assert config["optimization"]["optim"] == "adamw_torch"


# --------------------------------------------------------------------------
# Sweep configs (IMPL-01 §12) — authored this cycle, executed later
# --------------------------------------------------------------------------


SWEEP_PHASES = (
    ("phase1_lr", "lr", "learning_rate"),
    ("phase2_epochs", "epochs", "num_train_epochs"),
    ("phase3_rank", "rank", "lora_rank"),
)


@pytest.mark.parametrize(("filename", "phase", "swept"), SWEEP_PHASES)
def test_each_sweep_phase_defines_its_grid_metric_and_budget(filename, phase, swept):
    config = load_yaml(ROOT / "configs" / "sweeps" / f"{filename}.yaml")

    assert config["phase"] == phase
    assert config["metric"]["name"] and config["metric"]["goal"] in ("minimize", "maximize")
    assert any(k.startswith("budget") for k in config), f"{filename} declares no budget"

    grid = config["grid"]
    assert "foundation" in grid, f"{filename} sweeps nothing for the Foundation"
    assert swept in grid["foundation"], f"{filename} does not vary {swept}"
    assert len(grid["foundation"][swept]) >= 2, "a one-candidate grid is not a sweep"


def test_the_phases_are_ordered_by_dependency():
    """Learning rate first — highest impact, and every later phase is
    conditioned on its winner."""
    phase1 = load_yaml(ROOT / "configs" / "sweeps" / "phase1_lr.yaml")
    phase2 = load_yaml(ROOT / "configs" / "sweeps" / "phase2_epochs.yaml")
    phase3 = load_yaml(ROOT / "configs" / "sweeps" / "phase3_rank.yaml")

    assert "depends_on" not in phase1
    assert phase2["depends_on"] == "phase1_lr"
    assert phase3["depends_on"] == "phase2_epochs"


def test_the_sweep_ranks_on_the_metric_checkpoint_selection_reads():
    """Validation loss can fall while field extraction gets worse, and F1 is
    what the promotion gate reads."""
    from evaluation.checkpoint_eval import CheckpointScore

    # The name checkpoint selection reads, so the sweep and the selector cannot
    # rank on differently named metrics. "field_f1" matched nothing the scorer
    # emits, which would have left every candidate unmeasured.
    selection_metric = "field_normalized_match"
    assert CheckpointScore("x/checkpoint-1", 1, {selection_metric: 0.5}).field_f1 == 0.5, (
        "checkpoint selection no longer reads field_normalized_match; update the sweep with it"
    )
    for phase in ("phase2_epochs.yaml", "phase3_rank.yaml"):
        config = load_yaml(ROOT / "configs" / "sweeps" / phase)
        assert config["metric"]["name"] == selection_metric, phase
        assert config["metric"]["goal"] == "maximize"


def test_the_rank_phase_is_not_run_by_default():
    """Rank is the least likely of the three to be the bottleneck; it runs only
    if F1 plateaus after phases 1 and 2."""
    config = load_yaml(ROOT / "configs" / "sweeps" / "phase3_rank.yaml")
    assert config["run_by_default"] is False


def test_the_sweep_exists_but_is_not_wired_into_any_command():
    """Built, and deliberately not part of `finetune` or `all`.

    Running it is an operator decision taken after the pilot, not a stage that
    fires because a pipeline reached it — 9-12 training runs is not something a
    build should start on its own.
    """
    from orchestration import pipeline_dag, settings

    assert (ROOT / "training" / "sweep.py").exists()
    assert settings.pipeline_config()["deferred"]["sweep"] is False
    assert not any("sweep" in stage.name for stage in pipeline_dag.STAGES)


def test_the_sweep_refuses_to_run_at_pilot_volume():
    """The deferral reason, now enforced in code rather than in a doc: a sweep at
    25-30 documents per type measures which documents landed in a 3-4 document
    test split, and promotes that as a hyperparameter finding."""
    import pytest

    from training.sweep import SweepError, assert_enough_data

    with pytest.raises(SweepError, match="below the"):
        assert_enough_data({"policy": 28})


def test_quantization_thresholds_gate_package_when_metrics_exist():
    """IMPL-13 §4 — the gate sits between quantize and push. It applies only when
    per-format metrics were measured; scoring each GGUF needs a GPU, and a gate
    that invented numbers to have something to judge would be worse than one
    that says it has none."""
    import inspect

    from orchestration import pipeline_dag

    source = inspect.getsource(pipeline_dag.stage_quantize)
    assert "assert_servable" in source
    assert "quant_threshold_results" in source


# --------------------------------------------------------------------------
# GPU-vs-CPU OCR benchmark (IMPL-03 §7)
# --------------------------------------------------------------------------


def test_the_benchmark_times_the_gpu_and_makes_no_cpu_comparison(monkeypatch):
    """OCR is GPU-only, so there is no second device to compare against. What
    used to be a speed/cost measurement is now a constraint: MinerU's CPU path
    produces different markdown, so a corpus spanning both devices is built from
    two distributions (arch §8a)."""
    from data_pipeline.ocr import mineru_version, run_mineru

    monkeypatch.setattr(mineru_version, "cuda_available", lambda: (True, "L40S"))

    class _Engine:
        def __init__(self, device):
            assert device == "cuda", "the benchmark asked for a device other than cuda"

        def process(self, pdf_bytes, *, device, max_long_side_px):
            assert device == "cuda"
            return [run_mineru.PageOutput(page_number=n, markdown="x", image_bytes=b"i")
                    for n in (1, 2)]

    result = run_mineru.benchmark_gpu(b"%PDF", _Engine, max_long_side_px=1792)

    assert result["device"] == "cuda"
    assert result["pages"] == 2
    assert "seconds_per_page" in result
    assert "cpu" not in result and "speedup" not in result


def test_the_ocr_path_refuses_to_run_without_a_gpu(monkeypatch):
    """A silent CPU fallback would finish the job and write markdown from a
    different distribution than the corpus was built on, with no error anywhere.
    Failing is the only way that becomes visible."""
    import pytest as _pytest

    from data_pipeline.ocr import mineru_version

    monkeypatch.setattr(mineru_version, "cuda_available", lambda: (False, None))
    with _pytest.raises(mineru_version.MinerUVersionError, match="GPU-only"):
        mineru_version.resolve_device(None)


# --------------------------------------------------------------------------
# Extraction provenance (IMPL-12 §9)
# --------------------------------------------------------------------------


def test_a_result_traces_back_to_the_model_that_produced_it(tmp_path):
    """An output JSON with no recorded model version is a number nobody can
    attribute to a training run."""
    import json

    from serving.pipeline import ExtractionResult
    from testing.run_extraction import append_registry, write_outputs

    result = ExtractionResult(
        source_id="policy_0001", doc_type="policy", model_version="v2",
        mode="ocr_plus_image", schema_valid=True, overall_confidence=0.91,
        extraction={"insured_name": "Rivera Fabrication LLC"},
    )
    results_path, metrics_path = write_outputs(result, {"document": "policy_0001"}, root=tmp_path)
    append_registry(result, results_path, metrics_path, root=tmp_path, quant_format="q5_k_m")

    registry = json.loads((tmp_path / "extraction_registry.json").read_text(encoding="utf-8"))
    row = registry["extractions"][0]
    assert row["model_version"] == "v2"       # -> resolves to a run_manifest
    assert row["quant_format"] == "q5_k_m"
    assert row["document"] == "policy_0001"
    assert row["extracted_at"]
    assert (tmp_path / row["result_path"]).exists()
    assert (tmp_path / row["metrics_path"]).exists()


def test_results_are_organised_by_model_version_in_the_path(tmp_path):
    """`results/{version}/` — the version is visible in the path itself, so a
    file cannot be mistaken for another model's output."""
    from serving.pipeline import ExtractionResult
    from testing.run_extraction import write_outputs

    for version in ("base", "v2"):
        result = ExtractionResult(
            source_id="policy_0001", doc_type="policy", model_version=version,
            mode="ocr_plus_image", schema_valid=True, overall_confidence=0.5,
        )
        results_path, _metrics = write_outputs(result, {}, root=tmp_path)
        assert results_path.parent.name == version

    assert (tmp_path / "results" / "base" / "policy_0001.json").exists()
    assert (tmp_path / "results" / "v2" / "policy_0001.json").exists()


def test_appending_twice_accumulates_rather_than_overwrites(tmp_path):
    """A registry that keeps only the last extraction is not a registry."""
    import json

    from serving.pipeline import ExtractionResult
    from testing.run_extraction import append_registry, write_outputs

    for source_id in ("policy_0001", "policy_0002"):
        result = ExtractionResult(
            source_id=source_id, doc_type="policy", model_version="v2",
            mode="ocr_plus_image", schema_valid=True, overall_confidence=0.5,
        )
        paths_ = write_outputs(result, {}, root=tmp_path)
        append_registry(result, *paths_, root=tmp_path)

    registry = json.loads((tmp_path / "extraction_registry.json").read_text(encoding="utf-8"))
    assert [r["document"] for r in registry["extractions"]] == ["policy_0001", "policy_0002"]


# --------------------------------------------------------------------------
# Reference prompt files (IMPL-12 §2)
# --------------------------------------------------------------------------


def test_prompt_files_match_the_renderer():
    """The requirement the spec states as a test rather than a discipline.

    Two hand-kept copies of a prompt diverge by one character eventually, and
    prompt drift between corpus build and inference degrades a fine-tuned model
    while showing up in no training metric.
    """
    from testing.render_prompts import drifted

    problems = drifted()
    assert not problems, (
        "reference prompts no longer match common.prompts: "
        + "; ".join(problems)
        + ". Regenerate with `python -m testing.render_prompts --write` — never hand-edit them."
    )


def test_a_hand_edited_prompt_file_is_caught(tmp_path):
    """Without this the drift check could be comparing nothing."""
    from testing.render_prompts import drifted, prompt_path, write_all

    write_all(tmp_path)
    assert not drifted(tmp_path)

    victim = prompt_path("policy", "ocr_plus_image", None, tmp_path)
    victim.write_text(victim.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert any("policy.prompt.txt" in p for p in drifted(tmp_path))


def test_a_missing_prompt_file_is_drift_not_a_pass(tmp_path):
    from testing.render_prompts import drifted, prompt_path, write_all

    write_all(tmp_path)
    prompt_path("lossrun", "image_only", None, tmp_path).unlink()
    assert any("missing" in p for p in drifted(tmp_path))


def test_a_stale_prompt_file_is_reported_and_removed(tmp_path):
    """A leftover from a renamed target is a reference nothing regenerates, and
    the drift check would otherwise never look at it."""
    from testing.render_prompts import drifted, write_all

    write_all(tmp_path)
    orphan = tmp_path / "acord.prompt.txt"       # the old per-type name
    orphan.write_text("stale", encoding="utf-8")
    assert any("stale" in p for p in drifted(tmp_path))

    write_all(tmp_path)
    assert not orphan.exists()
    assert not drifted(tmp_path)


def test_each_acord_form_gets_its_own_reference():
    """25, 125 and 140 have different schemas, so one acord file pinned to a
    single form would misrepresent the other two."""
    from common.constants import ACORD_FORMS
    from testing.render_prompts import PROMPTS_DIR

    for form in ACORD_FORMS:
        assert (PROMPTS_DIR / f"acord_{form}.prompt.txt").exists()
    assert not (PROMPTS_DIR / "acord.prompt.txt").exists()


def test_the_banner_is_not_part_of_the_compared_prompt():
    """The provenance header must not leak into what the model would see."""
    from testing.render_prompts import PROMPTS_DIR, body_of, render

    text = (PROMPTS_DIR / "policy.prompt.txt").read_text(encoding="utf-8")
    assert text.startswith("# GENERATED")
    body = body_of(text)
    assert not body.startswith("#")
    assert body == render("policy", "ocr_plus_image")


def test_the_prompt_files_carry_no_alias_strings():
    """The same negative requirement as the rendered prompt: mapping surface
    labels to canonical keys is the model's job, and a prompt that lists them
    caps the system at a hand-written list."""
    from common import aliases, schemas
    from testing.render_prompts import PROMPTS_DIR

    # Whatever the schemas themselves say reaches the prompts by construction;
    # same rule as tests/test_schema_contract.py.
    #
    # Built from `schema_selectors`, not `all_schema_keys`: the latter returns
    # two-tuples and so cannot express a line of business, and it puts the LOB in
    # the acord_form slot where `schema_key` ignores it. Every per-LOB schema
    # would resolve back to the generic policy one, and every canonical schema's
    # own `title` — "Classic Auto policy" — would be reported as a leaked alias.
    embedded = "".join(
        schemas.schema_text(dt, form, lob) for dt, form, lob in schemas.schema_selectors()
    )

    for path in sorted(PROMPTS_DIR.glob("*.prompt.txt")):
        text = path.read_text(encoding="utf-8")
        for doc_type in ("policy", "lossrun", "acord"):
            for field_path, entry in aliases.load_registry(doc_type).items():
                for alias in entry.aliases:
                    # A one-word alias collides with ordinary prose ("Date",
                    # "Insured"); multi-word aliases are unambiguous. Same rule
                    # as test_prompt_carries_descriptions_but_no_alias_strings.
                    if " " not in alias or alias in embedded:
                        continue
                    assert alias not in text, (
                        f"{path.name} contains the alias {alias!r} for {field_path}"
                    )


def test_the_combined_spec_is_not_hand_edited():
    """It says "Do not edit this file directly" and, until this test, nothing
    enforced that. It drifted: IMPL-10 was rewritten for the v2.1 serving formats
    while the combined copy still described the GGUF matrix, so the repo held two
    contradictory answers to "what does quantize produce" and both looked
    official."""
    from scripts.combine_specs import drifted

    assert not drifted(), (
        "IMPL_ALL_COMBINED.md no longer matches the individual specs. Regenerate with "
        "`python -m scripts.combine_specs --write` — never hand-edit it."
    )
