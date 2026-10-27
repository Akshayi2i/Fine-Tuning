"""Incremental adapter training: base + LoRA, continued, never merged (decision 2026-10-23).

A cycle continues the tenant's promoted release: its adapter's weights are
trained further, on the same base model. It starts from the base instead when
asked (``--fresh``: after a data deletion, or to reset drift), when nothing is
promoted, and whenever continuing would teach the adapter on top of what it can
no longer use - another base, another LoRA shape, an adapter that is not in
Blob, another prompt hash or window plan, a parent that cannot be read - or
could not be judged afterwards: a parent with no eval report on the current
yardstick, which the gate's line rule would need.

Three things follow from it, and are guarded here:

* **A continued run is held, line by line, to the best since the base.**
  Continued training compounds drift, so each line is held to the best any run
  since the last run from the base scored on it: a line that slips a little
  each cycle blocks once the slips add up to more than the margin.
* **A purge follows the chain.** A run that continued one trained on a deleted
  document holds what that run learned, whatever its own corpus lists.
* **Nothing is merged or quantized.** The release is its adapter, served as a
  LoRA on the bf16 base: the bundle, the serving plan, ``extract --model vN``
  and the push all say so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import run as cli
from orchestration.pipeline_dag import (
    PipelineError,
    continuation,
    line_evidence,
    run_stages,
    stage_parent_adapter,
    stages_for,
    staging_parent_adapter_dir,
)
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import (
    PASSING_METRICS,
    _cli_context,
    _scored_on,
    make_context,
    promote_in_index,
    seed_corpus,
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


#: When the tenant's eval set was frozen, as ``_scored_on`` records it.
FROZEN_AT = "2026-10-01T00:00:00+00:00"


@pytest.fixture
def released(client, controller) -> BlobClient:
    """Release v1, packaged and promoted: its run manifest, its adapter in Blob
    (saved on the config's LoRA settings), its bundle on this code's prompt hash
    and window plan, and its eval report on the frozen set as it stands -
    everything a cycle needs to continue it."""
    from evaluation.freeze_eval_set import manifest_key

    seed_corpus(client)
    report = run_stages(make_context(client, controller, out_version="v1"), stages_for("all"),
                        command="all")
    assert report.ok, report.render()
    promote_in_index(client, "v1")
    client.write_json(manifest_key(), {"frozen_at": FROZEN_AT, "frozen_from_corpus": "v1"})
    client.write_json(paths.eval_report("v1", scope="policy"),
                      _report({"auto": [0.95] * 6}, policy=200))
    write_adapter(client, "v1")
    return client


def write_adapter(client: BlobClient, version: str, **config) -> None:
    """A trained adapter in Blob, as package pushes it: its saved config, on the
    LoRA settings of configs/training/unified.yaml unless ``config`` says
    otherwise, and its weights."""
    from common.config import training_config

    lora = training_config("unified")["lora"]
    # PEFT saves a number; under `auto`, the rate a small corpus is given.
    dropout = 0.1 if lora["dropout"] == "auto" else lora["dropout"]
    blob_dir = paths.scoped_adapter_dir("policy", version)
    client.write_json(f"{blob_dir}/adapter_config.json", {
        "r": lora["rank"], "lora_alpha": lora["alpha"], "lora_dropout": dropout,
        "bias": lora["bias"], "use_rslora": bool(lora.get("use_rslora")),
        "target_modules": list(lora["target_modules"]), **config})
    client.write_bytes(f"{blob_dir}/adapter_model.safetensors", b"weights")


def _edit_manifest(client: BlobClient, run_id: str, edit) -> None:
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import write_manifest

    manifest = get(run_id, client)
    edit(manifest)
    write_manifest(manifest, client)


def _edit_bundle(client: BlobClient, release_id: str, **fields) -> None:
    key = paths.release_bundle(release_id)
    client.write_json(key, {**client.read_json(key), **fields})


# --------------------------------------------------------------------------
# continuation: the promoted release, or the base and why
# --------------------------------------------------------------------------


def test_a_cycle_continues_the_promoted_release_when_everything_matches(released, controller):
    from common.prompts import prompt_hash
    from data_pipeline.dataset_builder.policy_windows import window_plan_id

    bundle = released.read_json(paths.release_bundle("v1"))
    assert bundle["prompt_hash"] == prompt_hash() and bundle["window_plan_version"] == window_plan_id()
    assert continuation(make_context(released, controller, out_version="v2")) == ("v1", None)


def test_fresh_is_asked_for_and_wins_over_everything(released, controller):
    assert continuation(make_context(released, controller, out_version="v2", fresh=True)) == (
        None, "asked for (--fresh)")


def test_with_nothing_promoted_the_cycle_starts_from_the_base(client, controller):
    """The first cycle, and every tenant whose index promotes nothing: another
    tenant's promoted release is not this one's to continue."""
    assert continuation(make_context(client, controller, out_version="v1")) == (
        None, "no release of the policy adapter is promoted for this tenant")
    promote_in_index(client, "v1", "acme")
    assert continuation(make_context(client, controller, out_version="v2")) == (
        None, "no release of the policy adapter is promoted for this tenant")


def test_a_parent_whose_run_manifest_cannot_be_read_is_not_continued(released, controller):
    released.delete(paths.run_manifest("policy-v1", "scoped", scope="policy"))
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None
    assert why == "the promoted release's run manifest (policy-v1) cannot be read"


@pytest.mark.parametrize("corrupt", ["newer_field", "not_json"])
def test_a_parent_manifest_that_does_not_load_is_not_continued(released, controller, corrupt):
    key = paths.run_manifest("policy-v1", "scoped", scope="policy")
    if corrupt == "newer_field":
        released.write_json(key, {**released.read_json(key), "field_from_a_newer_build": 1})
    else:
        released.write_text(key, "{not json")
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None and "cannot be read" in why


def test_a_parent_trained_on_another_base_is_not_continued(released, controller):
    from common.config import base_model_config

    base = base_model_config()["model"]
    other = f"{base['model_id']}@0000000000000000000000000000000000000000"

    def edit(manifest):
        manifest.dependencies.base_model = other

    _edit_manifest(released, "policy-v1", edit)
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None
    assert why == (f"policy-v1 trained on base {other}, this cycle on "
                   f"{base['model_id']}@{base['revision']}")


@pytest.mark.parametrize("field, value, said", [
    ("lora_rank", 32, "LoRA rank 32, this cycle's config 64"),
    ("lora_alpha", 64, "LoRA alpha 64, this cycle's config 128"),
    ("target_modules", ["q_proj"], "target modules ['q_proj'], this cycle's config"),
])
def test_a_parent_of_another_lora_shape_is_not_continued(released, controller, field, value, said):
    """Its weights would not fit the adapter this cycle's config builds."""
    _edit_manifest(released, "policy-v1", lambda m: setattr(m.training_config, field, value))
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None
    assert why.startswith(f"policy-v1 has {said}"), why


@pytest.mark.parametrize("gone", ["adapter_config.json", "adapter_model.safetensors"])
def test_a_parent_whose_adapter_is_not_in_blob_is_not_continued(released, controller, gone):
    """Both files: a dry run's placeholder holds a config and no weights."""
    adapter = paths.scoped_adapter_dir("policy", "v1")
    released.delete(f"{adapter}/{gone}")
    assert continuation(make_context(released, controller, out_version="v2")) == (
        None, f"policy-v1's adapter is not in Blob at {adapter} (no {gone})")


@pytest.mark.parametrize("saved", [{"use_rslora": True}, {"bias": "all"}])
def test_a_parent_saved_with_other_lora_settings_is_not_continued(released, controller, saved):
    """ms-swift continues an adapter with the settings it was saved with: an
    rsLoRA or bias change in the config would never apply to it. (Nor would a
    numeric dropout change; the shipped `dropout: auto` keeps the parent's,
    tests/test_review_training_engine.py.)"""
    write_adapter(released, "v1", **saved)
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None
    assert why.startswith("policy-v1's adapter was trained with other LoRA settings ("), why


def test_a_promoted_release_named_before_versions_is_continued_by_its_own_bundle(released, controller):
    """``release-YYYY.M.N`` serves adapter policy-v1: its bundle is under its own
    name, not under v1's."""
    legacy = "release-2026.10.1"
    released.write_json(paths.release_bundle(legacy), released.read_json(paths.release_bundle("v1")))
    released.delete(paths.release_bundle("v1"))
    index = paths.release_index()
    released.write_json(index, [{**row, "release_id": legacy} for row in released.read_json(index)])
    assert continuation(make_context(released, controller, out_version="v2")) == ("v1", None)


def test_a_parent_packaged_with_another_prompt_hash_is_not_continued(released, controller):
    """A template, schema or section-map change: the parent's outputs have
    another shape than the one this cycle trains."""
    _edit_bundle(released, "v1", prompt_hash="0" * 64)
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None and why.startswith("v1 was packaged with another prompt hash")

    # No bundle at all records no hash, so none that matches.
    released.delete(paths.release_bundle("v1"))
    assert continuation(make_context(released, controller, out_version="v2")) == (
        None, f"the promoted release v1 has no bundle at {paths.release_bundle('v1')}")


def test_a_parent_cut_with_another_window_plan_is_not_continued(released, controller):
    _edit_bundle(released, "v1", window_plan_version="1")
    assert continuation(make_context(released, controller, out_version="v2")) == (
        None, "v1 was trained on policies cut with another window plan")


def test_a_parent_with_no_eval_report_is_not_continued(released, controller):
    """The gate holds a continued run to its parent line by line; with no parent
    report it would block the cycle - after training it. Refused up front."""
    report_key = paths.eval_report("v1", scope="policy")
    released.delete(report_key)
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None
    assert why == (f"policy-v1 has no eval report at {report_key}, so the gate could not compare "
                   "this cycle with it line by line")


@pytest.mark.parametrize("case, said", [
    ("other_scoring_code", "different scoring code"),
    ("frozen_again", "the set was frozen again"),
])
def test_a_parent_scored_on_another_yardstick_is_not_continued(released, controller, case, said):
    """Other scoring code, or the set frozen again since the parent was scored:
    its per-line scores say nothing on this cycle's yardstick."""
    from evaluation.freeze_eval_set import manifest_key

    report_key = paths.eval_report("v1", scope="policy")
    if case == "other_scoring_code":
        released.write_json(report_key, _report({"homeowners": (0.95, 400)}, policy=200,
                                                fingerprint="0123456789abcdef"))
    else:
        released.write_json(manifest_key(), {"frozen_at": "2026-10-20T00:00:00+00:00",
                                             "frozen_from_corpus": "v1"})
    parent, why = continuation(make_context(released, controller, out_version="v2"))
    assert parent is None
    assert why.startswith("policy-v1's eval report is on another yardstick"), why
    assert said in why


# --------------------------------------------------------------------------
# stage_training: the parent staged and recorded
# --------------------------------------------------------------------------


def test_the_first_cycle_records_why_it_started_from_the_base(released):
    from registry_utils.query_registry import get

    first = get("policy-v1", released)
    assert first.parent_run_id is None and first.continued_from is None
    assert first.fresh_reason == "no release of the policy adapter is promoted for this tenant"


def test_a_continued_cycle_records_its_parent_on_the_manifest_and_in_the_stage(released, controller):
    from registry_utils.query_registry import get

    ctx = make_context(released, controller, out_version="v2")
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.ok, report.render()

    training = ctx.results["training"]
    assert training.data["continued"] == "policy-v1" and training.data["fresh_reason"] is None
    assert training.detail.endswith("continuing policy-v1")
    manifest = get("policy-v2", released)
    assert manifest.parent_run_id == "policy-v1" and manifest.fresh_reason is None
    # The weights it starts from: the parent's adapter, staged on the volume.
    assert manifest.continued_from == staging_parent_adapter_dir("v2")
    assert staging_parent_adapter_dir("v2") == paths.staging_parent_adapter_dir("policy", "v2")
    assert manifest.index_row()["parent_run_id"] == "policy-v1"
    # A dry run copies nothing.
    assert not ctx.volume.exists(staging_parent_adapter_dir("v2"))


def test_a_fresh_cycle_records_why_and_names_no_parent(released, controller):
    from registry_utils.query_registry import get

    ctx = make_context(released, controller, out_version="v2", fresh=True)
    assert run_stages(ctx, stages_for("finetune"), command="finetune").ok

    training = ctx.results["training"]
    assert training.data["continued"] is None
    assert training.data["fresh_reason"] == "asked for (--fresh)"
    assert training.detail.endswith("from the base (asked for (--fresh))")
    manifest = get("policy-v2", released)
    assert manifest.parent_run_id is None and manifest.continued_from is None
    assert manifest.fresh_reason == "asked for (--fresh)"
    assert manifest.index_row()["parent_run_id"] is None


def test_a_real_cycle_copies_the_parent_s_adapter_to_the_volume(client, controller, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", (tmp_path / "volume").as_posix())
    blob = paths.scoped_adapter_dir("policy", "v1")
    client.write_json(f"{blob}/adapter_config.json", {"r": 64, "lora_alpha": 128})
    client.write_bytes(f"{blob}/adapter_model.safetensors", b"weights")

    dry = make_context(client, controller, out_version="v2")
    assert stage_parent_adapter(dry, "v1") == staging_parent_adapter_dir("v2")
    assert not Path(staging_parent_adapter_dir("v2")).exists(), "a dry run copied the adapter"

    ctx = make_context(client, controller, out_version="v2", dry_run=False)
    local = stage_parent_adapter(ctx, "v1")
    assert local == staging_parent_adapter_dir("v2")
    assert json.loads(Path(local, "adapter_config.json").read_text(encoding="utf-8"))["r"] == 64
    assert Path(local, "adapter_model.safetensors").read_bytes() == b"weights"
    assert ctx.volume.exists(local)


def test_packaging_clears_the_staged_parent_with_the_version_s_own_paths(client, controller):
    from orchestration.pipeline_dag import _clear_staged_scope

    ctx = make_context(client, controller, out_version="v2")
    ctx.volume.mark(staging_parent_adapter_dir("v2"))
    ctx.volume.mark(staging_parent_adapter_dir("v3"))
    _clear_staged_scope(ctx)
    assert not ctx.volume.exists(staging_parent_adapter_dir("v2"))
    assert ctx.volume.exists(staging_parent_adapter_dir("v3")), "another cycle's parent was cleared"


# --------------------------------------------------------------------------
# training.train: warm start, fresh optimizer; resumes its own checkpoints
# --------------------------------------------------------------------------


def test_a_continued_run_takes_the_continued_training_values_where_set():
    from training.train import optimization_for

    cfg = {"optimization": {"num_train_epochs": 3, "learning_rate": 1e-4, "warmup_steps": 10},
           "continued_training": {"num_train_epochs": 1, "learning_rate": None}}
    assert optimization_for(cfg, continuing=False) == cfg["optimization"]
    assert optimization_for(cfg, continuing=True) == {
        "num_train_epochs": 1, "learning_rate": 1e-4, "warmup_steps": 10}
    assert cfg["optimization"]["num_train_epochs"] == 3, "the config itself was changed"
    assert optimization_for({"optimization": {"learning_rate": 1e-4}}, continuing=True) == {
        "learning_rate": 1e-4}


def test_the_shipped_config_trains_a_continued_run_as_a_fresh_one_until_measured():
    """Null keeps the fresh run's value: the continued values are to be set
    from a measured continued run, not by guess."""
    from common.config import training_config
    from training.train import optimization_for

    cfg = training_config("unified")
    assert cfg["continued_training"] == {"num_train_epochs": None, "learning_rate": None,
                                         # Rehearsal off: every document trains every epoch.
                                         "rehearsal_old_share": None,
                                         "rehearsal_weight_by_line_errors": False}
    assert optimization_for(cfg, continuing=True) == cfg["optimization"]


def test_the_continued_values_reach_ms_swift_and_the_manifest(monkeypatch):
    from common.config import training_config
    from training import train as T

    cfg = training_config("unified")
    shorter = {**cfg, "continued_training": {"num_train_epochs": 1, "learning_rate": 5e-5}}
    monkeypatch.setattr(T, "training_config", lambda _name=None: shorter)
    epochs = [f"corpus/default/v1/train/epoch_{i}.jsonl" for i in (1, 2, 3, 4)]

    fresh, fresh_recorded = T.build_training_config(corpus_paths=epochs, output_dir="/o")
    assert fresh.args["learning_rate"] == cfg["optimization"]["learning_rate"]
    assert len(fresh.args["dataset"]) == cfg["optimization"]["num_train_epochs"]
    assert "adapters" not in fresh.args

    swift, recorded = T.build_training_config(corpus_paths=epochs, output_dir="/o",
                                              resume_from="/staging/parent-adapters/policy/v2")
    assert swift.args["adapters"] == ["/staging/parent-adapters/policy/v2"]
    assert swift.args["learning_rate"] == 5e-5 and swift.args["dataset"] == epochs[:1]
    assert (recorded.learning_rate, recorded.epochs) == (5e-5, 1)
    assert (fresh_recorded.learning_rate, fresh_recorded.epochs) != (5e-5, 1)


def test_the_run_fingerprint_names_the_parent():
    """A run continued from v3 and one from the base are not one run, so neither
    resumes the other's checkpoints."""
    from types import SimpleNamespace

    from training.train import run_fingerprint

    recorded = SimpleNamespace(lora_rank=64, lora_alpha=128, learning_rate=1e-4, epochs=3)
    fresh = run_fingerprint("v1", {"rows": 10}, recorded)
    continued = run_fingerprint("v1", {"rows": 10}, recorded, parent_run_id="policy-v3")
    assert fresh["parent_run_id"] is None and continued["parent_run_id"] == "policy-v3"
    assert fresh != continued
    assert run_fingerprint("v1", {"rows": 10}, recorded, parent_run_id="policy-v3") == continued
    assert run_fingerprint("v1", {"rows": 10}, recorded, parent_run_id="policy-v2") != continued


def _as_if_on_the_pod(monkeypatch, tmp_path):
    """A real (non-dry) train() here, as tests/test_stage_data runs one: no CUDA
    and no ms-swift template, the volume a temporary folder, launch recorded."""
    from tests.test_stage_data import _stub_masking_encoder, _stub_tokens
    from training import train as T

    monkeypatch.setenv(T.OFF_POD_ENV, "1")
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path))
    # A pod's clean checkout, whatever state this one is in (a real run refuses a dirty tree).
    monkeypatch.setattr(T, "worktree_state", lambda: (False, None))
    monkeypatch.setattr(T, "_masking_encoder", lambda swift: _stub_masking_encoder())
    monkeypatch.setattr(T, "_token_counter", lambda: _stub_tokens)
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    launched: list = []
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))
    return launched


def _train(client, **kwargs):
    from registry_utils.models import DataStats
    from training import train as T

    return T.train(corpus_version="v1", out_version="v9", client=client, corpus_manifest={},
                   data_stats=DataStats(train_examples=0, val_examples=0, test_examples=0), **kwargs)


def test_a_dry_run_continuing_an_adapter_records_where_it_starts(client, monkeypatch, tmp_path):
    from tests.test_stage_data import _seed

    _seed(client)
    _as_if_on_the_pod(monkeypatch, tmp_path)
    parent = "/runpod-volume/staging/parent-adapters/policy/v9"
    swift, manifest = _train(client, dry_run=True, continue_from=parent, parent_run_id="policy-v8")
    assert swift.args["adapters"] == [parent] and "resume_from_checkpoint" not in swift.args
    assert manifest.continued_from == parent and manifest.parent_run_id == "policy-v8"
    assert manifest.fresh_reason is None

    _swift, fresh = _train(client, dry_run=True, fresh_reason="asked for (--fresh)")
    assert "adapters" not in _swift.args
    assert fresh.parent_run_id is None and fresh.fresh_reason == "asked for (--fresh)"


def test_a_continued_run_starts_from_the_parent_and_resumes_its_own_checkpoint(client, monkeypatch,
                                                                                tmp_path):
    """A continued run is a NEW run from the parent's weights (ms-swift
    ``adapters``, fresh optimizer). Interrupted, it resumes its own checkpoint -
    which already started from the parent - and does not load the parent again."""
    from tests.test_stage_data import _seed
    from tests.test_training_continuation import matching_adapter_config
    from tests.test_training_resume import _checkpoint
    from training import train as T

    _seed(client)
    launched = _as_if_on_the_pod(monkeypatch, tmp_path / "volume")
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_config.json").write_text(json.dumps(matching_adapter_config()), encoding="utf-8")

    def interrupted(config):
        launched.append(config)
        raise RuntimeError("GPU lost at step 80")

    monkeypatch.setattr(T, "launch", interrupted)
    with pytest.raises(RuntimeError, match="GPU lost"):
        _train(client, dry_run=False, continue_from=str(parent), parent_run_id="policy-v8")
    first = launched[-1]
    assert first.args["adapters"] == [str(parent)]
    assert "resume_from_checkpoint" not in first.args

    staging = Path(paths.scoped_staging_adapter_dir("policy", "v9"))
    checkpoint = _checkpoint(staging / "v0-20261023-100000", 70)
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))
    _swift, manifest = _train(client, dry_run=False, continue_from=str(parent), parent_run_id="policy-v8")
    resumed = launched[-1]
    assert resumed.args["resume_from_checkpoint"] == str(checkpoint)
    assert "adapters" not in resumed.args, "the resumed run loaded the parent's adapter as well"
    assert manifest.parent_run_id == "policy-v8" and manifest.status == "trained"


def test_a_checkpoint_of_a_run_from_another_parent_is_not_resumed(client, monkeypatch, tmp_path):
    from tests.test_stage_data import _seed
    from tests.test_training_continuation import matching_adapter_config
    from tests.test_training_resume import _checkpoint
    from training import train as T

    _seed(client)
    _as_if_on_the_pod(monkeypatch, tmp_path / "volume")
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "adapter_config.json").write_text(json.dumps(matching_adapter_config()), encoding="utf-8")
    monkeypatch.setattr(T, "launch", lambda config: (_ for _ in ()).throw(RuntimeError("GPU lost")))
    with pytest.raises(RuntimeError):
        _train(client, dry_run=False, continue_from=str(parent), parent_run_id="policy-v8")
    _checkpoint(Path(paths.scoped_staging_adapter_dir("policy", "v9")) / "v0-x", 70)

    with pytest.raises(T.TrainingError, match="parent_run_id"):
        _train(client, dry_run=False)                       # from the base this time


# --------------------------------------------------------------------------
# The gate: a continued run is held, line by line, to the runs before it
# --------------------------------------------------------------------------


def _metrics(**over) -> dict[str, float]:
    return {**PASSING_METRICS, **over}


def _compared(line: str, before: list[float], after: list[float], reference: str = "policy-v1"):
    from evaluation.gating import LineComparison

    return LineComparison(line=line, reference=reference, reference_scores=list(before),
                          candidate_scores=list(after))


def _gate(evidence, **kwargs):
    from evaluation.gating import promotion_gate

    return promotion_gate(_metrics(), None, continued_from="policy-v2", line_evidence=evidence,
                          **kwargs)


def test_a_continued_run_that_lost_no_ground_passes():
    from evaluation.gating import LineEvidence

    # 30 documents a line: the fewest a line is gated on (decision 2026-10-26).
    result = _gate(LineEvidence(compared=[
        _compared("auto", [0.90] * 30, [0.90] * 30),
        _compared("wc", [0.90] * 30, [0.95] * 30),
        # A drop inside the margin (LINE_PARENT_MARGIN), on every document alike.
        _compared("homeowners", [0.90] * 30, [0.889] * 30),
    ], chain=["policy-v2", "policy-v1"]))
    assert result.passed, result.report()
    assert [v["line"] for v in result.line_verdicts] == ["auto", "homeowners", "wc"]
    assert all(v["non_inferior"] for v in result.line_verdicts)


def test_a_line_that_fell_beyond_the_margin_blocks_a_continued_run():
    from evaluation.gating import LINE_PARENT_MARGIN, LineEvidence

    assert LINE_PARENT_MARGIN == 0.015
    result = _gate(LineEvidence(compared=[_compared("auto", [0.95] * 30, [0.90] * 30),
                                          _compared("wc", [0.90] * 30, [0.90] * 30)]))
    assert result.failed_gates == ["line:auto"]
    [reason] = result.blocking_reasons
    assert "auto is not non-inferior to policy-v1" in reason and "0.9500 -> 0.9000" in reason
    assert result.as_dict()["line_verdicts"][0]["held_to"] == "policy-v1"


def test_a_drop_on_one_document_of_a_line_still_counts():
    """Paired over the line's documents: one document losing a third of its
    values is not averaged away by the others that held."""
    from evaluation.gating import LineEvidence

    before = [0.9, 0.8, 0.95, 0.7, 0.9, 0.85] * 5
    after = [0.6, *before[1:]]
    result = _gate(LineEvidence(compared=[_compared("auto", before, after)]))
    assert result.failed_gates == ["line:auto"]


def test_a_drop_of_exactly_the_margin_is_judged_the_same_whatever_the_digits():
    """Float subtraction put one drop of exactly δ just above -δ and another
    just below it: one passed and one blocked."""
    from evaluation.bootstrap import ConfidenceInterval

    for parent, candidate in ((0.0505, 0.0355), (0.9, 0.885)):
        drop = candidate - parent
        assert ConfidenceInterval("m", drop, drop, drop).non_inferior(0.015) is False, (parent, candidate)
    assert ConfidenceInterval("m", -0.0149, -0.0149, -0.0149).non_inferior(0.015)


def test_a_line_the_parent_served_and_the_candidate_does_not_blocks():
    from evaluation.gating import LineEvidence

    result = _gate(LineEvidence(dropped=["homeowners"], added=["wc"]))
    assert result.failed_gates == ["line:homeowners"]
    assert ("homeowners is served by policy-v2's release and not by this one"
            in result.blocking_reasons[0])
    assert any("this run serves wc" in w for w in result.warnings), result.warnings


def test_a_continued_run_with_nothing_to_compare_against_is_blocked():
    """Drift nobody can measure is not drift that did not happen: the remedy is
    the next cycle, from the base."""
    missing = "policy-v1: it has no eval report at eval-reports/scope/policy/v1/summary.json"
    result = _gate(None, line_evidence_missing=missing)
    assert result.failed_gates == ["line_regression_evidence"]
    [reason] = result.blocking_reasons
    assert missing in reason and "next cycle" in reason and "--fresh, without --version" in reason


def test_lines_with_too_few_documents_are_named_and_not_compared():
    """Under 30 shared documents a line is warned, not gated (decision
    2026-10-26); a line sharing none is named apart."""
    from evaluation.gating import LINE_MIN_DOCUMENTS, LineEvidence

    assert LINE_MIN_DOCUMENTS == 30
    result = _gate(LineEvidence(compared=[_compared("auto", [0.95] * 4, [0.60] * 4)],
                                thin={"wc": 0}))
    assert result.passed, result.report()
    assert any("fewer than 30 real documents" in w and "auto (4, minimum detectable drop" in w
               for w in result.warnings), result.warnings
    assert any("no real document of the line scored in both reports" in w and "wc (0)" in w
               for w in result.warnings), result.warnings
    assert any("no line had 30 or more real documents" in w for w in result.warnings), result.warnings
    [verdict] = result.line_verdicts
    assert verdict["status"] == "warned" and verdict["non_inferior"] is None


def test_a_run_from_the_base_is_not_held_to_a_parent():
    from evaluation.gating import promotion_gate

    assert promotion_gate(_metrics(), None, line_evidence=None).passed


def test_the_line_gates_are_named_gates_and_the_cross_type_ones_are_gone():
    import inspect

    from evaluation import gating

    assert gating.OTHER_GATES == ("improvement", "line_regression_evidence")
    assert gating.is_gate("line:homeowners") and gating.is_gate("line_regression_evidence")
    assert not gating.is_gate("line:not_a_line") and not gating.is_gate("cross_type:acord")
    assert not gating.is_gate("cross_type_regression_evidence")
    params = inspect.signature(gating.promotion_gate).parameters
    assert {"continued_from", "line_evidence", "line_evidence_missing"} <= set(params)
    assert "cross_type_evidence" not in params
    assert not hasattr(gating, "_require_cross_type_evidence")
    assert not hasattr(gating, "LINE_MIN_VALUES")


# --------------------------------------------------------------------------
# The per-line evidence, from the eval reports down the chain
# --------------------------------------------------------------------------


def _report(docs: dict[str, list[float]], **eval_set) -> dict:
    """An eval report as ``EvalReport.as_dict`` writes it, as far as the gate
    reads it: the gate metrics, the yardstick, and each real document's field
    match and line - ``{line: [score of document 1, 2, ...]}``, the documents
    named ``<line>_0001`` and on, so two reports share them."""
    scores = {f"{line}_{i:04d}": score for line, values in docs.items()
              for i, score in enumerate(values, 1)}
    return {
        "gate_metrics": dict(PASSING_METRICS),
        "eval_set": _scored_on(**eval_set),
        "per_document": {"field_normalized_match": scores},
        "per_document_weights": {"field_normalized_match": dict.fromkeys(scores, 1.0)},
        "per_document_lines": {source: source.rsplit("_", 1)[0] for source in scores},
    }


def test_the_report_records_each_real_single_line_document_s_line():
    from evaluation.run_eval import build_report, line_document_scores

    gold = {"policy": {"policy_number": "HO-1", "insured_name": "A B"}}

    def reading(source: str, lob: list[str], **meta) -> tuple:
        return (gold, gold, {"source_id": source, "doc_type": "policy", "lob": lob,
                             "modality_mode": "ocr_plus_image", **meta})

    body = build_report("v", [
        reading("p1", ["homeowners"]),
        reading("p2", ["homeowners"], synthetic=True),          # a twin counts toward nothing
        reading("p3", ["homeowners", "personal_auto"]),         # a package goes to the base model
        reading("p4", ["classic_auto"]),                         # classic auto is personal auto
    ]).as_dict()
    assert body["per_document_lines"] == {"p1": "homeowners", "p4": "personal_auto"}
    assert line_document_scores(body) == {"homeowners": {"p1": (1.0, 1.0)},
                                          "personal_auto": {"p4": (1.0, 1.0)}}
    # Written before reports recorded documents' lines.
    assert line_document_scores({"per_document": body["per_document"]}) is None


def _chain_context(client, controller, monkeypatch, parents: dict[str, str | None],
                   served=("auto", "wc")):
    """Candidate v3, continuing policy-v2, with these run manifests' parents
    (a run missing from ``parents`` has no readable manifest) and these lines
    served."""
    from types import SimpleNamespace

    from orchestration import pipeline_dag
    from registry_utils import query_registry

    def get(run_id, _client):
        if run_id not in parents:
            raise query_registry.RegistryQueryError(f"no manifest for {run_id}")
        return SimpleNamespace(parent_run_id=parents[run_id])

    monkeypatch.setattr(query_registry, "get", get)
    monkeypatch.setattr(pipeline_dag, "served_lines", lambda ctx: list(served))
    return make_context(client, controller, out_version="v3")


def _reports(client, **by_version: dict) -> None:
    for version, report in by_version.items():
        client.write_json(paths.eval_report(version, scope="policy"), report)


ROOTED = {"policy-v2": "policy-v1", "policy-v1": None}


def test_each_line_is_held_to_its_parent_and_to_the_run_from_the_base(client, controller, monkeypatch):
    """Decision 2026-10-26: the reference is the PARENT, and the run from the
    base is the cumulative-drift reference. auto slipped 0.01 a cycle (0.95,
    0.94, 0.93): within the parent margin, and 0.02 in all, within the drift
    margin, so it passes - the drift check blocks once the losses pass 0.03."""
    from evaluation.gating import promotion_gate

    ctx = _chain_context(client, controller, monkeypatch, ROOTED)
    _reports(client, v1=_report({"auto": [0.95] * 30, "wc": [0.80] * 30}),
             v2=_report({"auto": [0.94] * 30, "wc": [0.85] * 30}),
             v3=_report({"auto": [0.93] * 30, "wc": [0.85] * 30}))
    evidence, why = line_evidence(ctx, "policy-v2")
    assert why is None and evidence.chain == ["policy-v2", "policy-v1"]
    assert {c.line: c.reference for c in evidence.compared} == {"auto": "policy-v2", "wc": "policy-v2"}
    assert evidence.drift_reference == "policy-v1" and evidence.drift_reference_is_fresh
    assert {c.line: c.reference for c in evidence.drift} == {"auto": "policy-v1", "wc": "policy-v1"}
    result = promotion_gate(_metrics(), None, continued_from="policy-v2", line_evidence=evidence)
    assert result.passed, result.report()
    assert {(v["line"], v["check"], v["held_to"]) for v in result.line_verdicts} == {
        ("auto", "parent", "policy-v2"), ("wc", "parent", "policy-v2"),
        ("auto", "drift", "policy-v1"), ("wc", "drift", "policy-v1")}


def test_on_a_tie_the_nearer_run_is_held_to(client, controller, monkeypatch):
    ctx = _chain_context(client, controller, monkeypatch, ROOTED, served=("auto",))
    _reports(client, v1=_report({"auto": [0.9] * 6}), v2=_report({"auto": [0.9] * 6}),
             v3=_report({"auto": [0.9] * 6}))
    evidence, _why = line_evidence(ctx, "policy-v2")
    assert [c.reference for c in evidence.compared] == ["policy-v2"]


def test_the_parent_s_report_is_required(client, controller, monkeypatch):
    ctx = _chain_context(client, controller, monkeypatch, ROOTED)
    assert line_evidence(ctx, "policy-v2") == (
        None, f"the candidate has no eval report at {paths.eval_report('v3', scope='policy')}")
    _reports(client, v3=_report({"auto": [0.9] * 6}))
    assert line_evidence(ctx, "policy-v2") == (
        None, f"policy-v2: it has no eval report at {paths.eval_report('v2', scope='policy')}")
    _reports(client, v2=_report({"auto": [0.9] * 6}, frozen_at="2026-09-01T00:00:00+00:00"))
    evidence, why = line_evidence(ctx, "policy-v2")
    assert evidence is None and why.startswith("policy-v2: the set was frozen again")
    old = _report({"auto": [0.9] * 6})
    del old["per_document_lines"]
    _reports(client, v2=old)
    evidence, why = line_evidence(ctx, "policy-v2")
    assert evidence is None and "records no document's line" in why


def test_a_run_down_the_chain_on_another_yardstick_is_left_out_and_named(client, controller,
                                                                         monkeypatch):
    ctx = _chain_context(client, controller, monkeypatch, ROOTED)
    _reports(client, v1=_report({"auto": [0.99] * 6}, frozen_at="2026-09-01T00:00:00+00:00"),
             v2=_report({"auto": [0.9] * 6}), v3=_report({"auto": [0.9] * 6}))
    warnings: list[str] = []
    evidence, why = line_evidence(ctx, "policy-v2", warnings)
    assert why is None and evidence.chain == ["policy-v2"]
    assert [c.reference for c in evidence.compared] == ["policy-v2"]
    assert any(w.startswith("policy-v1, on the chain of runs this one continued, is not compared "
                            "line by line: the set was frozen again") for w in warnings), warnings


def test_the_chain_stops_where_a_manifest_cannot_be_read_and_never_loops(client, controller,
                                                                        monkeypatch):
    _reports(client, v1=_report({"auto": [0.9] * 6}), v2=_report({"auto": [0.9] * 6}),
             v3=_report({"auto": [0.9] * 6}))
    ctx = _chain_context(client, controller, monkeypatch, {"policy-v2": "policy-v1"})
    warnings: list[str] = []
    evidence, _why = line_evidence(ctx, "policy-v2", warnings)
    assert evidence.chain == ["policy-v2", "policy-v1"]
    assert any("the chain of runs this one continued stops at policy-v1" in w for w in warnings)

    looped = _chain_context(client, controller, monkeypatch,
                            {"policy-v2": "policy-v1", "policy-v1": "policy-v2"})
    evidence, _why = line_evidence(looped, "policy-v2")
    assert evidence.chain == ["policy-v2", "policy-v1"]


def test_a_line_with_too_few_shared_documents_is_named_not_compared(client, controller, monkeypatch):
    """Every line sharing a document is passed on, with its documents; the gate
    decides it has too few to gate (decision 2026-10-26) and warns."""
    from evaluation.gating import promotion_gate

    ctx = _chain_context(client, controller, monkeypatch, ROOTED)
    _reports(client, v1=_report({"auto": [0.9] * 4}), v2=_report({"auto": [0.9] * 4}),
             v3=_report({"auto": [0.9] * 6, "wc": [0.9] * 6}))
    evidence, _why = line_evidence(ctx, "policy-v2")
    assert [(c.line, len(c.candidate_scores)) for c in evidence.compared] == [("auto", 4)]
    assert evidence.thin == {}
    # wc is new: nothing before it scored it, so it is added, not thin.
    assert evidence.added == ["wc"] and evidence.dropped == []
    result = promotion_gate(_metrics(), None, continued_from="policy-v2", line_evidence=evidence)
    assert result.passed and {v["status"] for v in result.line_verdicts} == {"warned"}


def test_lines_the_parent_s_release_served_and_the_candidate_does_not_are_dropped(
        client, controller, monkeypatch):
    ctx = _chain_context(client, controller, monkeypatch, ROOTED)
    parent = _report({"auto": [0.9] * 6})
    parent["lines_scored"] = ["auto", "homeowners"]
    _reports(client, v1=_report({"auto": [0.9] * 6}), v2=parent, v3=_report({"auto": [0.9] * 6}))
    evidence, _why = line_evidence(ctx, "policy-v2")
    assert evidence.dropped == ["homeowners"] and evidence.added == ["wc"]


def _continued_v2_gate(client, controller, *, parent: dict, candidate: dict):
    """v2, continuing the promoted v1, through the gate with these eval reports."""
    client.write_json(paths.eval_report("v1", scope="policy"), parent)
    client.write_json(paths.eval_report("v2", scope="policy"), candidate)
    ctx = make_context(client, controller, out_version="v2")
    report = run_stages(ctx, stages_for("all"), command="all")
    assert ctx.results["training"].data["continued"] == "policy-v1"
    return report, client.read_json(paths.gate_decision("v2", scope="policy"))


def test_a_continued_cycle_that_held_every_line_is_released(released, controller):
    report, decision = _continued_v2_gate(
        released, controller,
        parent=_report({"equipment_breakdown": [0.95] * 6}, policy=200),
        candidate=_report({"equipment_breakdown": [0.96] * 6}, policy=200))
    assert report.ok, report.render()
    assert decision["passed"] and decision["failed_gates"] == []
    comparison = decision["line_comparison"]
    assert comparison["continued"] == "policy-v1" and comparison["runs_compared"] == ["policy-v1"]
    assert [(v["line"], v["held_to"]) for v in comparison["lines"]] == [("equipment_breakdown", "policy-v1")]


def test_a_continued_cycle_that_lost_a_line_is_blocked_at_the_gate(released, controller):
    report, decision = _continued_v2_gate(
        released, controller,
        parent=_report({"equipment_breakdown": [0.95] * 30}, policy=200),
        candidate=_report({"equipment_breakdown": [0.90] * 30}, policy=200))
    assert report.blocked_at == "evaluation_gate"
    assert "line:equipment_breakdown" in decision["failed_gates"]
    assert any("equipment_breakdown is not non-inferior to policy-v1" in r for r in decision["blocking_reasons"])
    assert "package" not in {r.name for r in report.results}


def test_a_continued_cycle_whose_parent_s_report_went_missing_is_blocked(released, controller):
    """continuation() refuses a parent with no report before training; one that
    goes after (a package run later) leaves the gate nothing to hold the cycle
    to, and it is blocked rather than passed on floors."""
    trained = make_context(released, controller, out_version="v2")
    assert run_stages(trained, stages_for("finetune"), command="finetune").ok
    assert trained.results["training"].data["continued"] == "policy-v1"
    released.delete(paths.eval_report("v1", scope="policy"))
    released.write_json(paths.eval_report("v2", scope="policy"),
                        _report({"auto": [0.9] * 6}, policy=200))

    report = run_stages(make_context(released, controller, out_version="v2"), stages_for("package"),
                        command="package")
    assert report.blocked_at == "evaluation_gate"
    decision = released.read_json(paths.gate_decision("v2", scope="policy"))
    assert decision["failed_gates"] == ["line_regression_evidence"]
    assert "policy-v1: it has no eval report" in report.results[-1].detail
    assert decision["line_comparison"]["missing"].startswith("policy-v1: it has no eval report")


def test_a_run_continued_from_a_directory_has_no_parent_to_compare_with(released, controller):
    """training.train --continue-from names a directory, not a run of the
    adapter: there is no parent report to hold it to, so it is blocked."""
    ctx = make_context(released, controller, out_version="v2", fresh=True)
    assert run_stages(ctx, stages_for("finetune"), command="finetune").ok
    _edit_manifest(released, "policy-v2",
                   lambda m: setattr(m, "continued_from", "/runpod-volume/staging/some-adapter"))

    package = make_context(released, controller, out_version="v2")
    report = run_stages(package, stages_for("package"), command="package")
    assert report.blocked_at == "evaluation_gate"
    decision = released.read_json(paths.gate_decision("v2", scope="policy"))
    assert decision["failed_gates"] == ["line_regression_evidence"]
    assert ("it continued /runpod-volume/staging/some-adapter, a directory, not a run of the adapter"
            in report.results[-1].detail)


# --------------------------------------------------------------------------
# No merge: the release is its adapter, served as a LoRA on the bf16 base
# --------------------------------------------------------------------------


def test_the_release_bundle_is_the_adapter_on_the_bf16_base(released, controller):
    from orchestration.pipeline_dag import build_release_bundle

    bundle, _reasons = build_release_bundle(make_context(released, controller, out_version="v1"))
    assert bundle.served_as == "lora_on_base" and bundle.merged_model == ""
    assert bundle.serving_formats == {"bf16": paths.scoped_adapter_dir("policy", "v1")}


def test_a_release_cannot_be_packaged_in_a_format_nothing_produces(released, controller):
    from orchestration.pipeline_dag import build_release_bundle

    ctx = make_context(released, controller, out_version="v1", formats=["bf16", "fp8"])
    with pytest.raises(PipelineError, match="serves bf16 only"):
        build_release_bundle(ctx)


def _bundle(**over):
    from registry_utils.models import ReleaseBundle

    body = dict(release_id="v3", tenant_scope="default", base_model="Qwen/Qwen3-VL-8B-Instruct@abc",
                adapter="policy-v3", serving_formats={"bf16": paths.scoped_adapter_dir("policy", "v3")},
                prompt_hash="h", vision_config_hash="v", vllm_config_hash="l", lockfile_hash="k")
    body.update(over)
    return ReleaseBundle(**body)


def test_the_release_bundle_refuses_what_a_lora_on_the_base_cannot_serve():
    lora = _bundle(served_as="lora_on_base")
    assert lora.merged_model == ""
    with pytest.raises(ValueError, match="serves bf16 only"):
        _bundle(served_as="lora_on_base", serving_formats={
            "bf16": paths.scoped_adapter_dir("policy", "v3"), "fp8": "quantized-models/x"})
    # Every bundle written before is a merged one, and must still name its model.
    with pytest.raises(ValueError, match="must name it"):
        _bundle()
    merged = _bundle(merged_model=paths.merged_model_dir("v3", scope="policy"),
                     serving_formats={"bf16": paths.merged_model_dir("v3", scope="policy")})
    assert merged.served_as == "merged"


def test_the_push_and_the_serving_model_are_the_selected_adapter(client, controller, monkeypatch):
    """Calibration and the gate load the base with the selected checkpoint on
    every request - what is served - and nothing else is servable."""
    from inference_core import model_runner
    from orchestration.pipeline_dag import _staged_serving_model

    loaded: list[tuple] = []
    monkeypatch.setattr(model_runner, "load_model",
                        lambda tag, client, **kw: loaded.append((tag, kw)) or "model")
    ctx = make_context(client, controller, out_version="v1")
    assert _staged_serving_model(ctx, "bf16") == "model"
    tag, kwargs = loaded[-1]
    assert tag == "base" and kwargs["backend"] == "vllm" and kwargs["label"] == "policy-v1:bf16"
    assert kwargs["adapter"] == paths.scoped_staging_adapter_dir("policy", "v1")

    chosen = f"{paths.scoped_staging_adapter_dir('policy', 'v1')}/v0-1/checkpoint-150"
    client.write_json(paths.checkpoint_selection("v1", scope="policy"), {"selected": chosen})
    _staged_serving_model(ctx, "bf16")
    assert loaded[-1][1]["adapter"] == chosen
    with pytest.raises(PipelineError, match="fp8 is not served"):
        _staged_serving_model(ctx, "fp8")


# --------------------------------------------------------------------------
# Serving a release that is a LoRA on the base
# --------------------------------------------------------------------------


def _lora_release(client, release_id="v3", *, serving_formats=None, lines=("homeowners",),
                  **over) -> dict:
    """A promoted release as package now writes it: served as a LoRA on the
    base, its bf16 artifact the adapter's directory."""
    from tests.test_release_router import promote

    bundle = promote(client, release_id, lines=lines, **over)
    bundle.update(served_as="lora_on_base", merged_model="", serving_formats=(
        {"bf16": paths.scoped_adapter_dir("policy", release_id)} if serving_formats is None
        else serving_formats))
    client.write_json(paths.release_bundle(release_id), bundle)
    return bundle


def test_a_lora_release_is_served_from_its_adapter_directory(client):
    from serving.release_router import build_serving_plan

    _lora_release(client, serving_formats={"bf16": "adapters/scope/policy/v7"})
    release = build_serving_plan(client).release
    assert release is not None and release.served_as == "lora_on_base"
    # The bundle's bf16 artifact, not one derived from the adapter's name.
    assert release.adapter_dir == "adapters/scope/policy/v7"
    assert release.merged_model == "" and release.adapter == "policy-v3"


def test_a_lora_release_with_no_adapter_to_serve_is_skipped(client):
    from serving.release_router import build_serving_plan

    _lora_release(client, "v3", created_at="2026-10-01T00:00:00+00:00")
    nothing = _lora_release(client, "v4", serving_formats={}, created_at="2026-11-01T00:00:00+00:00")
    nothing["adapter"] = None
    client.write_json(paths.release_bundle("v4"), nothing)

    assert build_serving_plan(client).release.release_id == "v3"


def test_a_lora_release_is_not_served_in_a_format_it_was_not_packaged_in(client):
    """Not the bf16 adapter under another format's name, with no calibrator."""
    from serving.release_router import build_serving_plan

    _lora_release(client)
    assert build_serving_plan(client, fmt="bf16").release.release_id == "v3"
    assert build_serving_plan(client, fmt="fp8").release is None


def test_an_engine_serving_a_lora_release_loads_the_base(client):
    """Even for a release trained on every line: the engine holds the base and
    applies the adapter per request, as there is no merged model to hold."""
    from serving.release_router import build_serving_plan
    from serving.vllm_entrypoint import serves_base_with_loras

    bundle = _lora_release(client, lines=())
    assert serves_base_with_loras(build_serving_plan(client))
    # The same release as an older merged bundle: its merged model is the engine.
    merged = paths.merged_model_dir("v3", scope="policy")
    client.write_json(paths.release_bundle("v3"), {
        **bundle, "served_as": "merged", "merged_model": merged, "serving_formats": {"bf16": merged}})
    assert not serves_base_with_loras(build_serving_plan(client))


# --------------------------------------------------------------------------
# extract --model vN: the base with that version's adapter
# --------------------------------------------------------------------------


def _write_adapter_run(client, status: str) -> None:
    from registry_utils.models import Artifacts
    from registry_utils.write_run_manifest import write_manifest
    from tests.test_registry import _scoped

    write_manifest(_scoped("policy-v3", "policy", ["policy"], artifacts=Artifacts(
        status=status, adapter_weights=paths.scoped_adapter_dir("policy", "v3"),
        staging_path=paths.scoped_staging_adapter_dir("policy", "v3") if status == "staged" else None,
    )), client)


def test_a_published_version_loads_as_the_base_with_its_adapter(client, tmp_path, monkeypatch):
    from inference_core import model_runner
    from inference_core.model_runner import EchoBackend, generate, load_model
    from registry_utils.query_registry import resolve_model_version

    monkeypatch.setattr(model_runner, "ADAPTER_CACHE", tmp_path / "cache")
    _write_adapter_run(client, "published")
    blob = paths.scoped_adapter_dir("policy", "v3")
    client.write_json(f"{blob}/adapter_config.json", {"r": 64, "lora_alpha": 128})
    client.write_bytes(f"{blob}/adapter_model.safetensors", b"weights")

    resolved = resolve_model_version("v3", client)
    assert resolved["kind"] == "base_plus_adapter" and resolved["merged_model"] is None
    assert resolved["foundation_adapter"] == blob

    echo = EchoBackend('{"a": 1}')
    model = load_model("v3", client, backend_impl=echo)
    assert model.tag == "v3" and not model.is_base
    assert Path(model.default_adapter, "adapter_config.json").is_file()
    generate(model, [{"role": "user", "content": "x"}])
    assert echo.calls[-1]["adapter"] == model.default_adapter


def test_a_staged_version_loads_its_adapter_from_the_volume(client, tmp_path, monkeypatch):
    from inference_core import model_runner
    from inference_core.model_runner import EchoBackend, load_model

    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", (tmp_path / "volume").as_posix())
    _write_adapter_run(client, "staged")
    prepared: list[str] = []
    real = model_runner.prepare_adapter
    monkeypatch.setattr(model_runner, "prepare_adapter",
                        lambda adapter, client, config: prepared.append(adapter) or real(adapter, client, config))
    # As ms-swift leaves it: checkpoints under a versioned run directory, no
    # adapter at the output root. With no selection recorded, the latest one.
    run = Path(paths.scoped_staging_adapter_dir("policy", "v3")) / "v0-20261023-120000"
    for step in (100, 200):
        (run / f"checkpoint-{step}").mkdir(parents=True)
        (run / f"checkpoint-{step}" / "adapter_config.json").write_text('{"r": 64}', encoding="utf-8")

    model = load_model("v3", client, backend_impl=EchoBackend())
    assert [Path(p) for p in prepared] == [run / "checkpoint-200"]
    assert Path(model.default_adapter) == (run / "checkpoint-200").resolve()
    assert model.resolved["from_staging"] is True


def test_a_version_has_no_quantized_model_to_load(client):
    from inference_core.model_runner import EchoBackend, load_model
    from registry_utils.query_registry import RegistryQueryError

    _write_adapter_run(client, "published")
    with pytest.raises(RegistryQueryError, match="nothing is merged, so there is no fp8 model"):
        load_model("v3", client, backend_impl=EchoBackend(), quant_format="fp8")


# --------------------------------------------------------------------------
# A purge follows the chain of continued runs
# --------------------------------------------------------------------------


def _purge_runs(client, *runs: tuple[str, str, str | None], corpora: dict[str, list[str]]) -> dict:
    """``(run_id, corpus, parent_run_id)`` runs of tenant broker_a, written in the
    order given, and each corpus's train split; returns the purge plan's actions
    for policy_0001 by run."""
    from common.run_ids import version_of
    from data_pipeline.ingestion import pull_raw_pdfs as ingest
    from registry_utils.write_run_manifest import write_manifest
    from tests.test_registry import _deps, _scoped

    for corpus, ids in corpora.items():
        client.write_json(paths.corpus_manifest(corpus, "broker_a"),
                          {"source_ids_by_split": {"train": ids}})
    for run_id, corpus, parent in runs:
        version = version_of(run_id)
        manifest = _scoped(
            run_id, "policy", ["policy"], tenant_id="broker_a", parent_run_id=parent,
            continued_from=paths.staging_parent_adapter_dir("policy", version) if parent else None,
            dependencies=_deps(corpus_version=f"corpus/{corpus}"))
        manifest.artifacts.adapter_weights = paths.scoped_adapter_dir("policy", version)
        write_manifest(manifest, client)
    plan = ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)
    return {entry["run_id"]: entry for entry in plan["trained_weights"]}


def test_a_run_continued_from_one_trained_on_the_document_needs_a_retrain(client):
    """policy-v2's own corpus never listed the document, but it started from
    policy-v1's weights, which learned it."""
    runs = _purge_runs(client, ("policy-v1", "v1", None), ("policy-v2", "v2", "policy-v1"),
                       corpora={"v1": ["policy_0001"], "v2": ["policy_0002"]})
    assert runs["policy-v1"]["action"] == "needs retrain or delete"
    assert runs["policy-v2"]["action"] == "continued policy-v1, listed here: needs retrain or delete"
    assert runs["policy-v2"]["adapter_weights"] == paths.scoped_adapter_dir("policy", "v2")


def test_the_purge_follows_the_chain_down_every_level_in_any_order(client):
    """Three levels, written newest first: v3 continued v2, which continued v1.
    A fresh v4 and the v5 continued from it never saw the document."""
    client.write_json(paths.release_index("broker_a"), [{"release_id": "v3", "adapter": "policy-v3"}])
    runs = _purge_runs(
        client,
        ("policy-v3", "v3", "policy-v2"), ("policy-v2", "v2", "policy-v1"), ("policy-v1", "v1", None),
        ("policy-v4", "v4", None), ("policy-v5", "v5", "policy-v4"),
        corpora={"v1": ["policy_0001"], "v2": ["policy_0002"], "v3": ["policy_0003"],
                 "v4": ["policy_0004"], "v5": ["policy_0005"]})
    assert runs["policy-v2"]["action"] == "continued policy-v1, listed here: needs retrain or delete"
    assert runs["policy-v3"]["action"] == "continued policy-v2, listed here: needs retrain or delete"
    assert runs["policy-v3"]["releases"] == ["v3"]
    assert "policy-v4" not in runs and "policy-v5" not in runs


def test_a_parent_to_check_by_hand_is_checked_with_its_children(client):
    """policy-v1's corpus manifest is gone, so whether it trained on the document
    cannot be read: every run down its chain is checked by hand with it."""
    runs = _purge_runs(client, ("policy-v1", "v1", None), ("policy-v2", "v2", "policy-v1"),
                       ("policy-v3", "v3", "policy-v2"),
                       corpora={"v2": ["policy_0002"], "v3": ["policy_0003"]})
    assert runs["policy-v1"]["action"] == "check by hand: the run's corpus manifest is missing"
    assert runs["policy-v2"]["action"] == "continued policy-v1, listed here: check by hand with it"
    assert runs["policy-v3"]["action"] == "continued policy-v2, listed here: check by hand with it"


def test_a_continued_run_whose_own_corpus_lists_the_document_is_listed_for_itself(client):
    runs = _purge_runs(client, ("policy-v1", "v1", None), ("policy-v2", "v2", "policy-v1"),
                       corpora={"v1": ["policy_0009"], "v2": ["policy_0001"]})
    assert runs["policy-v2"]["action"] == "needs retrain or delete"
    assert "policy-v1" not in runs


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["finetune", "all"])
def test_fresh_reaches_the_context(client, controller, command):
    argv = [command, "--input", "./intake", "--version", "v2"]
    assert _cli_context(client, controller, argv + ["--fresh"]).fresh is True
    assert _cli_context(client, controller, argv).fresh is False


def test_package_continues_or_not_as_its_cycle_trained(client, controller):
    """``--fresh`` decides where training starts; package trains nothing."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["package", "--version", "v2", "--fresh"])
    assert _cli_context(client, controller, ["package", "--version", "v2"]).fresh is False


@pytest.mark.parametrize("flags", [["--formats", "bf16", "fp8"], ["--skip-quantize"], ["--dtype", "bf16"]])
@pytest.mark.parametrize("command", [["package", "--version", "v2"], ["all", "--input", "./intake"],
                                     ["finetune", "--input", "./intake"]])
def test_the_merge_and_quantize_flags_are_gone(command, flags):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(command + flags)
