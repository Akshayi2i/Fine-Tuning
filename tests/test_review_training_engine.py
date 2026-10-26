"""The training engine's review fixes (W2): dropout auto, throughput levers and
the benchmark mode, supervised tokens per group, the two-thirds checkpoint,
reproducibility (settings hash, dirty tree) and the dead early-stopping config.

Synthetic fixtures only; nothing here needs a GPU or ms-swift.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.config import training_config as real_training_config
from training import train as T

EPOCH_PATHS = [f"corpus/default/v1/train/epoch_{i}.jsonl" for i in (1, 2, 3, 4)]


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _with_config(monkeypatch, **blocks) -> dict:
    """The shipped training config with ``blocks`` laid over it, as training.train reads it."""
    cfg = copy.deepcopy(real_training_config("unified"))
    for name, values in blocks.items():
        cfg[name] = {**cfg.get(name, {}), **values}
    monkeypatch.setattr(T, "training_config", lambda name=None: cfg)
    return cfg


def _as_if_on_the_pod(monkeypatch, tmp_path, *, dirty: bool = False) -> list:
    """A real (non-dry) run here: no CUDA, no ms-swift, launch recorded."""
    from tests.test_stage_data import _stub_masking_encoder, _stub_tokens

    monkeypatch.setenv(T.OFF_POD_ENV, "1")
    monkeypatch.delenv(T.ALLOW_DIRTY_ENV, raising=False)
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", str(tmp_path / "volume"))
    monkeypatch.setattr(T, "worktree_state", lambda: (True, "d1f7") if dirty else (False, None))
    monkeypatch.setattr(T, "_masking_encoder", lambda swift: _stub_masking_encoder())
    monkeypatch.setattr(T, "_token_counter", lambda: _stub_tokens)
    monkeypatch.setattr(T, "validate_all", lambda **_kw: None)
    launched: list = []
    monkeypatch.setattr(T, "launch", lambda config: launched.append(config))
    return launched


def _train(client, **kwargs):
    from registry_utils.models import DataStats

    return T.train(corpus_version="v1", out_version="v9", client=client, corpus_manifest={},
                   data_stats=DataStats(train_examples=0, val_examples=0, test_examples=0), **kwargs)


# --------------------------------------------------------------------------
# #5 lora.dropout: auto
# --------------------------------------------------------------------------


def test_auto_takes_the_lower_rate_only_when_every_line_has_200_real_documents():
    big = {"homeowners": {"real": 250, "synthetic": 0}, "wc": {"real": 200, "synthetic": 40}}
    assert T.auto_dropout(big)[0] == 0.05
    # One line short - synthetic documents do not count towards the 200.
    short = {**big, "umbrella": {"real": 150, "synthetic": 300, "documents": 450}}
    value, why = T.auto_dropout(short)
    assert value == 0.10 and "umbrella 150" in why
    assert T.auto_dropout({})[0] == 0.10 and T.auto_dropout(None)[0] == 0.10


def test_a_numeric_dropout_works_exactly_as_before(monkeypatch):
    big = {"homeowners": {"real": 900}}
    assert T.resolve_lora_dropout({"dropout": 0.1}, train_lines=big) == (
        0.1, {"configured": 0.1, "chosen": 0.1, "source": "config"})
    _with_config(monkeypatch, lora={"dropout": 0.07})
    swift, recorded = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o")
    assert swift.args["lora_dropout"] == 0.07 and recorded.lora_dropout == 0.07
    assert recorded.lora_dropout_choice["source"] == "config"
    with pytest.raises(T.TrainingError, match="lora.dropout"):
        T.configured_dropout({"dropout": "high"})


def test_the_shipped_config_is_auto():
    """Shipped as auto: today every line has under 200 real documents, so a run
    from the base gets 0.10 - what the number was - and a continued one keeps
    its parent's; the rule then applies itself when the data arrives."""
    assert T.is_auto_dropout(real_training_config("unified")["lora"]["dropout"])


def test_auto_never_trips_the_mismatch_refusal_a_number_still_does():
    from tests.test_training_continuation import LORA, SAVED

    auto = {**LORA, "dropout": "auto"}
    for saved in (0.05, 0.1, 0.0, 1e-9):
        assert T.adapter_config_mismatch({**SAVED, "lora_dropout": saved}, auto) is None
    # An adapter saved with no dropout at all cannot say what it trains with.
    assert T.adapter_config_mismatch({**SAVED, "lora_dropout": None}, auto) == (
        "lora_dropout is missing from the adapter, auto in the config")
    assert "not a number" in T.adapter_config_mismatch({**SAVED, "lora_dropout": "x"}, auto)
    # A number in the config: the refusal as it was.
    assert T.adapter_config_mismatch({**SAVED, "lora_dropout": 0.1}, LORA) == (
        "lora_dropout is 0.1 in the adapter, 0.05 in the config")


def test_a_fresh_auto_run_chooses_from_its_corpus_and_records_why(client, monkeypatch, tmp_path):
    from tests.test_stage_data import _seed

    _seed(client)
    _as_if_on_the_pod(monkeypatch, tmp_path)
    swift, manifest = _train(client, dry_run=True)
    assert swift.args["lora_dropout"] == 0.10 == manifest.training_config.lora_dropout
    choice = manifest.training_config.lora_dropout_choice
    assert choice["configured"] == "auto" and choice["source"] == "corpus"
    assert "fewer than 200 real training documents" in choice["reason"]

    # Every line over the floor: the lower rate.
    monkeypatch.setattr(T, "AUTO_DROPOUT_MIN_REAL_DOCUMENTS", 1)
    swift, manifest = T.train(corpus_version="v1", out_version="v10", client=client, corpus_manifest={},
                              data_stats=manifest.data_stats, dry_run=True)
    assert swift.args["lora_dropout"] == 0.05 == manifest.training_config.lora_dropout


def test_a_continued_auto_run_keeps_its_parents_dropout(client, monkeypatch, tmp_path):
    """The parent was saved at 0.05 while a fresh run would get 0.10: not
    refused, and trained, recorded and passed at the parent's value - the one
    ms-swift keeps for a continued adapter, and the one its adapter_config.json
    will carry again."""
    from tests.test_stage_data import _seed
    from tests.test_training_continuation import _adapter

    _seed(client)
    launched = _as_if_on_the_pod(monkeypatch, tmp_path)
    parent = _adapter(tmp_path / "parent", lora_dropout=0.05)
    _swift, manifest = _train(client, dry_run=False, continue_from=parent, parent_run_id="policy-v8")
    assert launched[-1].args["lora_dropout"] == 0.05 == manifest.training_config.lora_dropout
    choice = manifest.training_config.lora_dropout_choice
    assert choice["source"] == "parent" and "adapter_config.json" in choice["reason"]


def test_a_continued_run_whose_dropout_cannot_be_read(tmp_path):
    lora = {"dropout": "auto"}
    missing = str(tmp_path / "nowhere")
    with pytest.raises(T.TrainingError, match="cannot be read"):
        T.resolve_lora_dropout(lora, train_lines={}, continue_from=missing)
    # A dry run cannot see the pod's volume: the corpus decides, and says so.
    value, choice = T.resolve_lora_dropout(lora, train_lines={}, continue_from=missing, dry_run=True)
    assert value == 0.10 and "dry run" in choice["reason"]


def test_a_dry_run_reads_the_parents_dropout_from_its_manifest(client, monkeypatch):
    """The pod's volume is not visible from a dry run: the parent run's manifest is."""
    manifest = SimpleNamespace(training_config=SimpleNamespace(lora_dropout=0.05))
    monkeypatch.setattr("registry_utils.query_registry.get",
                        lambda run_id, c: manifest if run_id == "policy-v3" else None)
    value, where = T.parent_dropout("/runpod-volume/staging/parent-adapters/policy/v4", "policy-v3", client)
    assert value == 0.05 and "policy-v3" in where


def test_the_last_checkpoints_saved_dropout_is_checked_after_training(tmp_path, caplog):
    run = tmp_path / "v0-x"
    for step in (100, 200, 300):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    (run / "checkpoint-300" / "adapter_config.json").write_text(json.dumps({"lora_dropout": 0.1}))
    assert T.training_summary(tmp_path, lora_dropout=0.1)["adapter_lora_dropout"] == 0.1
    with caplog.at_level("WARNING"):
        T.training_summary(tmp_path, lora_dropout=0.05)
    assert "saved with lora_dropout 0.1, the manifest records 0.05" in caplog.text


# --------------------------------------------------------------------------
# #29 Throughput levers and the benchmark mode
# --------------------------------------------------------------------------


def test_the_throughput_levers_are_not_passed_until_set(monkeypatch):
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o")
    assert not set(T.THROUGHPUT_KEYS) & set(swift.args), "null is ms-swift's own default"

    _with_config(monkeypatch, batch={"vit_gradient_checkpointing": False, "dataloader_num_workers": 4,
                                     "per_device_eval_batch_size": 4})
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o")
    assert (swift.args["vit_gradient_checkpointing"], swift.args["dataloader_num_workers"],
            swift.args["per_device_eval_batch_size"]) == (False, 4, 4)
    argv = swift.to_cli()
    assert argv[argv.index("--vit_gradient_checkpointing") + 1] == "false"


@pytest.mark.parametrize("bad", [{"vit_gradient_checkpointing": "no"}, {"dataloader_num_workers": -1},
                                 {"per_device_eval_batch_size": 0}, {"dataloader_num_workers": True}])
def test_a_lever_of_the_wrong_kind_is_refused(bad):
    with pytest.raises(T.TrainingError, match="batch."):
        T.throughput_args(bad)


def test_the_levers_are_not_part_of_the_run_fingerprint(monkeypatch):
    from common import config

    cfg = copy.deepcopy(real_training_config("unified"))
    current = {"cfg": cfg}
    monkeypatch.setattr(config, "training_config", lambda name: current["cfg"])
    first = T._settings_hash()
    current["cfg"] = {**cfg, "batch": {**cfg["batch"], "vit_gradient_checkpointing": False,
                                       "dataloader_num_workers": 8, "per_device_eval_batch_size": 4}}
    assert T._settings_hash() == first


def _log_entries(times: list[float], memory: float = 40.5) -> list[dict]:
    """ms-swift's logging.jsonl at logging_steps 1: elapsed seconds per step."""
    n = len(times)
    out = [{"loss": 1.0, "memory(GiB)": memory - (n - k), "train_speed(iter/s)": round(k / t, 6),
            "global_step/max_steps": f"{k}/{n}"} for k, t in enumerate(times, start=1)]
    return out + [{"train_runtime": times[-1], "train_steps_per_second": n / times[-1]}]


def test_step_times_come_from_ms_swifts_per_step_log():
    timings = T.step_timings(_log_entries([30.0, 32.0, 34.0, 36.5, 38.0]))
    assert timings["first_step"] == pytest.approx(30.0, rel=1e-4)
    assert timings["per_step"] == pytest.approx([2.0, 2.0, 2.5, 1.5], rel=1e-3)
    assert timings["peak_memory_gib"] == 40.5 and timings["train_runtime"] == 38.0
    assert T.step_timings([{"garbage": 1}])["per_step"] == []


def test_the_suggestions_follow_the_measurement_plan_one_lever_at_a_time():
    batch = {"per_device_train_batch_size": 1, "effective_batch_size": 8}
    suggest = T.suggest_next_experiment
    assert "vit_gradient_checkpointing: false" in suggest(batch, peak_memory_gib=60, gpu_memory_gib=140)
    batch["vit_gradient_checkpointing"] = False
    assert "per_device_train_batch_size: 2" in suggest(batch, peak_memory_gib=60, gpu_memory_gib=140)
    # No room for a second row: straight to the workers.
    assert "dataloader_num_workers: 4" in suggest(batch, peak_memory_gib=100, gpu_memory_gib=140)
    batch["per_device_train_batch_size"] = 2
    batch["dataloader_num_workers"] = 4
    assert "dataloader_num_workers: 8" in suggest(batch, peak_memory_gib=100, gpu_memory_gib=140)
    batch["dataloader_num_workers"] = 8
    assert "every lever" in suggest(batch, peak_memory_gib=100, gpu_memory_gib=140)


def test_the_benchmark_runs_n_steps_saves_nothing_and_writes_benchmark_json(client, monkeypatch, tmp_path):
    from registry_utils.query_registry import RegistryQueryError, get
    from tests.test_stage_data import _seed

    _seed(client)
    launched = _as_if_on_the_pod(monkeypatch, tmp_path, dirty=True)   # editing a lever dirties the tree

    def fake_swift(config):
        launched.append(config)
        log = Path(config.args["output_dir"]) / "v0-20261026" / "logging.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text("".join(json.dumps(e) + "\n" for e in _log_entries([20.0, 22.0, 24.0, 26.0])))

    monkeypatch.setattr(T, "launch", fake_swift)
    result = T.benchmark(corpus_version="v1", out_version="v9", client=client, steps=4)

    (config,) = launched
    assert config.args["max_steps"] == 4 and config.args["logging_steps"] == 1
    assert config.args["save_strategy"] == "no" and config.args["eval_strategy"] == "no"
    assert not {"val_dataset", "save_steps", "eval_steps", "save_total_limit"} & set(config.args)
    assert all(Path(p).is_file() for p in config.args["dataset"])
    written = json.loads((T.benchmark_dir("v9") / T.BENCHMARK_FILE).read_text(encoding="utf-8"))
    assert written["step_seconds"]["median"] == pytest.approx(2.0, rel=1e-3) == result["step_seconds"]["p90"]
    assert written["peak_memory_gib"] == 40.5
    assert written["tokens_per_second_estimated"] > 0 and written["length_histogram"]
    assert written["supervised_by_group"] and written["code"]["dirty"] is True
    assert "vit_gradient_checkpointing" in written["next_experiment"]
    # Not a run: no manifest, no version reserved, nothing in the run's staging.
    with pytest.raises((RegistryQueryError, KeyError, FileNotFoundError)):
        get("policy-v9", client)
    assert not Path(paths.scoped_staging_adapter_dir("policy", "v9")).exists()


def test_a_dry_run_benchmark_only_shows_the_command(client, monkeypatch, tmp_path):
    from tests.test_stage_data import _seed

    _seed(client)
    launched = _as_if_on_the_pod(monkeypatch, tmp_path)
    result = T.benchmark(corpus_version="v1", out_version="v9", client=client, steps=3, dry_run=True)
    assert result["dry_run"] and "--max_steps" in result["argv"] and not launched
    with pytest.raises(T.TrainingError, match="at least 1"):
        T.benchmark(corpus_version="v1", out_version="v9", client=client, steps=0, dry_run=True)


def test_the_benchmark_cli_detaches_on_the_pod_before_anything_runs(monkeypatch):
    import orchestration.detach as detach

    calls = []
    monkeypatch.setattr(detach, "detach_module_if_needed",
                        lambda module, argv, hint=None: calls.append((module, list(argv), hint)) or True)
    monkeypatch.setattr(T, "BlobClient", lambda: pytest.fail("ran attached"))
    argv = ["--corpus", "v1", "--out-version", "v9", "--benchmark-steps", "30"]
    assert T.main(argv) == 0
    assert calls == [("training.train", argv, "train-benchmark")]


# --------------------------------------------------------------------------
# #36 Supervised tokens per group
# --------------------------------------------------------------------------


def _staged(path: Path, answers: list[str]) -> str:
    rows = [{"messages": [{"role": "system", "content": "s" * 40}, {"role": "user", "content": "u" * 400},
                          {"role": "assistant", "content": answer}], "images": []} for answer in answers]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path.resolve())


def test_the_length_report_totals_supervised_tokens_per_group(tmp_path, caplog):
    from tests.test_stage_data import _stub_tokens
    from training.length_check import HISTOGRAM_BUCKET, measure

    epoch = _staged(tmp_path / "epoch_1.jsonl", ["d" * 400, "a" * 2000, "l" * 1600, "a" * 400])
    val = _staged(tmp_path / "val.jsonl", ["x" * 40_000])
    groups = {epoch: ["decl", "arrays", "lineblk", "arrays"], val: ["decl"]}
    report = measure([epoch, val], max_length=10**6, count_tokens=_stub_tokens, min_pixels=1,
                     max_pixels=10**7, groups=groups, stats_files=[epoch])
    summary = report.summary()
    by_group = summary["supervised_by_group"]
    assert {g: c["tokens"] for g, c in by_group.items()} == {"decl": 101, "arrays": 501 + 101, "lineblk": 401}
    assert by_group["arrays"]["rows"] == 2 and summary["rows"] == 4      # the validation sample is not counted
    assert summary["supervised_tokens"] == 101 + 602 + 401
    assert sum(summary["length_histogram"].values()) == 4
    assert all(int(edge) % HISTOGRAM_BUCKET == 0 for edge in summary["length_histogram"])
    # arrays + lineblk = 1003 of 1104 supervised tokens: the declarations are under-weighted.
    assert summary["heavy_group_share"] > 0.6 and summary["warnings"]

    balanced = measure([epoch], max_length=10**6, count_tokens=_stub_tokens, min_pixels=1, max_pixels=10**7,
                       groups={epoch: ["decl", "decl", "lineblk", "arrays"]})
    assert balanced.heavy_share() < 0.6 and not balanced.warnings()
    # Without groups every row is "unknown", and nothing is said.
    plain = measure([epoch], max_length=10**6, count_tokens=_stub_tokens, min_pixels=1, max_pixels=10**7)
    assert set(plain.summary()["supervised_by_group"]) == {"unknown"} and not plain.warnings()


def test_a_real_run_records_the_length_report_on_its_manifest(client, monkeypatch, tmp_path):
    from tests.test_stage_data import _seed

    _seed(client)
    _as_if_on_the_pod(monkeypatch, tmp_path)
    _swift, manifest = _train(client, dry_run=False)
    stats = manifest.data_stats.token_stats
    assert stats["rows"] > 0 and stats["supervised_tokens"] > 0 and stats["length_histogram"]
    # Each staged row's section group, as staging records it.
    assert "unknown" not in stats["supervised_by_group"], stats["supervised_by_group"]


# --------------------------------------------------------------------------
# #30 The two-thirds checkpoint
# --------------------------------------------------------------------------


def _rotate(limit: int, total: int, every: int, best: int | None) -> list[int]:
    """The HF Trainer's checkpoint rotation: after each save, the oldest go
    beyond ``limit``, the best-loss checkpoint kept (moved to the end)."""
    kept: list[int] = []
    for step in [*range(every, total + 1, every), *([total] if total % every else [])]:
        kept.append(step)
        order = sorted(kept)
        if best in order:
            order.remove(best)
            order.append(best)
        kept = order[max(0, len(order) - limit):]
    return sorted(kept)


@pytest.mark.parametrize("total, every, best", [(300, 10, 50), (301, 10, None), (90, 7, 7), (10, 3, 3)])
def test_the_limit_keeps_the_checkpoint_nearest_two_thirds(total, every, best):
    limit = T.keep_two_thirds_limit(total, every)
    saves = [*range(every, total + 1, every), *([total] if total % every else [])]
    nearest = min(saves, key=lambda s: (abs(s - 2 * total / 3), s))
    assert nearest in _rotate(limit, total, every, best)
    # ...and a step count one save longer than planned still keeps it.
    assert nearest in _rotate(limit, total + every, every, best)
    # The point of it: a small configured limit alone would have lost it.
    assert nearest not in _rotate(2, total, every, best) or total <= 3 * every


def test_the_configured_limit_is_raised_never_lowered(monkeypatch):
    _with_config(monkeypatch, evaluation={"evaluations_per_run": 30, "save_total_limit": 1,
                                          "checkpoint_every_steps": 1000},
                 distributed={"gpus": 1})
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o", train_rows=2400)
    assert swift.args["save_steps"] == 10
    assert swift.args["save_total_limit"] == T.keep_two_thirds_limit(300, 10) == 13
    # The shipped config: the larger of its own cadence's limit and the floor.
    monkeypatch.setattr(T, "training_config", real_training_config)
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o", train_rows=2400)
    cadence = T._checkpoint_cadence(T._eval_interval(2400, configured=50, evaluations=10,
                                                     effective_batch=8),
                                    real_training_config("unified")["evaluation"])
    assert swift.args["save_total_limit"] == max(cadence["save_total_limit"],
                                                 T.keep_two_thirds_limit(300, cadence["save_steps"]))


def test_the_two_thirds_step_is_recorded_beside_the_completion_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "write_manifest", lambda manifest, client: None)
    monkeypatch.setattr(T, "launch", lambda config: None)
    old, run = tmp_path / "v0-old", tmp_path / "v1-new"
    for step in (500, 900):
        (old / f"checkpoint-{step}").mkdir(parents=True)
    for step in (70, 140, 210, 280, 300):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    (run / "checkpoint-300" / "trainer_state.json").write_text(json.dumps({"save_steps": 70}))
    manifest = SimpleNamespace(status="training", run_id="policy-v9",
                               artifacts=SimpleNamespace(staging_path=str(tmp_path)),
                               training_config=SimpleNamespace(lora_dropout=0.1))
    import os

    os.utime(old, (1, 1))                       # the most recent run is the one recorded
    T.launch_and_record(object(), manifest, None)
    summary = T.read_training_summary(tmp_path)
    assert summary["run_id"] == "policy-v9" and summary["final_step"] == 300
    assert summary["two_thirds_step"] == 210 and summary["two_thirds_checkpoint"].endswith("checkpoint-210")
    assert T.training_completed(tmp_path)


# --------------------------------------------------------------------------
# #46 Reproducibility
# --------------------------------------------------------------------------


def test_the_settings_hash_covers_the_format_profile_and_the_base_model(monkeypatch, tmp_path):
    from common import config

    fmt, base = tmp_path / "format.yaml", tmp_path / "base_model.yaml"
    fmt.write_text("profile: fast\n")
    base.write_text("model: {model_id: qwen, revision: abc, local_dir: /workspace/models}\n")
    monkeypatch.setattr(config, "SHARED_FORMAT_CONFIG", fmt)
    monkeypatch.setattr(config, "BASE_MODEL_CONFIG", base)
    first = T._settings_hash()
    fmt.write_text("profile: standard\n")
    assert T._settings_hash() != first
    fmt.write_text("profile: fast\n")
    base.write_text("model: {model_id: qwen, revision: def, local_dir: /workspace/models}\n")
    assert T._settings_hash() != first
    # Where a pod keeps the weights is not which weights: a resume elsewhere matches.
    base.write_text("model: {model_id: qwen, revision: abc, local_dir: /mnt/other}\n")
    assert T._settings_hash() == first


def test_a_real_run_on_a_dirty_tree_is_refused_unless_allowed(client, monkeypatch, tmp_path):
    from tests.test_stage_data import _seed

    _seed(client)
    launched = _as_if_on_the_pod(monkeypatch, tmp_path, dirty=True)
    with pytest.raises(T.TrainingError, match="--allow-dirty"):
        _train(client, dry_run=False)
    assert not launched
    assert not Path(paths.staging_train_data_dir("policy", "v9")).exists()     # refused before staging

    _swift, manifest = _train(client, dry_run=False, allow_dirty=True)
    assert launched and manifest.dependencies.code_dirty is True
    assert manifest.dependencies.code_diff_sha256 == "d1f7"


def test_the_pipeline_can_allow_a_dirty_tree_by_environment(monkeypatch):
    monkeypatch.setattr(T, "worktree_state", lambda: (True, "d1f7"))
    monkeypatch.delenv(T.ALLOW_DIRTY_ENV, raising=False)
    with pytest.raises(T.TrainingError):
        T.assert_clean_tree()
    monkeypatch.setenv(T.ALLOW_DIRTY_ENV, "1")
    T.assert_clean_tree()
    monkeypatch.setattr(T, "worktree_state", lambda: (None, None))     # no git: not refused, recorded unknown
    monkeypatch.delenv(T.ALLOW_DIRTY_ENV)
    T.assert_clean_tree()


def test_a_dry_run_on_a_dirty_tree_is_recorded_not_refused(client, monkeypatch, tmp_path):
    from tests.test_stage_data import _seed

    _seed(client)
    _as_if_on_the_pod(monkeypatch, tmp_path, dirty=True)
    _swift, manifest = _train(client, dry_run=True)
    assert manifest.dependencies.code_dirty is True and manifest.dependencies.code_diff_sha256 == "d1f7"


def test_the_worktree_state_hashes_the_diff_and_the_untracked_names(monkeypatch):
    import subprocess

    state = {"status": b"", "diff": b"", "untracked": b""}

    def fake_git(argv, **_kw):
        if argv[1] == "status":
            out = state["status"]
        elif argv[1] == "diff":
            out = state["diff"]
        else:
            out = state["untracked"]
        return SimpleNamespace(stdout=out)

    monkeypatch.setattr(subprocess, "run", fake_git)
    assert T.worktree_state() == (False, None)
    state.update(status=b" M training/train.py\n", diff=b"-a\n+b\n")
    dirty, first = T.worktree_state()
    assert dirty is True and len(first) == 64
    state["untracked"] = b"training/new.py\n"
    assert T.worktree_state()[1] != first
    state.update(untracked=b"", diff=b"-a\n+c\n")
    assert T.worktree_state()[1] != first

    def no_git(*_a, **_kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert T.worktree_state() == (None, None)


# --------------------------------------------------------------------------
# #A8 No dead early-stopping config
# --------------------------------------------------------------------------


def test_no_early_stopping_is_configured_or_kept_around():
    import importlib.util

    assert "early_stopping_patience" not in real_training_config("unified")["evaluation"]
    assert importlib.util.find_spec("training.callbacks.early_stopping") is None
    swift, _ = T.build_training_config(corpus_paths=EPOCH_PATHS, output_dir="/o")
    assert not any("early_stopping" in key for key in swift.args)
