"""Resuming an interrupted training run from its last checkpoint (training.train).

Before: a crash meant training from step 0 again, because a checkpoint was saved
only at an evaluation and nothing resumed from one; and "trained" meant "the
output directory exists", which ms-swift creates when training starts.
"""

from __future__ import annotations

import json

from training import train as T


def _checkpoint(run, step, *, optimizer=True, state=True, adapter=True):
    ckpt = run / f"checkpoint-{step}"
    ckpt.mkdir(parents=True)
    if state:
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    if adapter:
        (ckpt / "adapter_model.safetensors").write_bytes(b"w")
    if optimizer:
        (ckpt / "optimizer.pt").write_bytes(b"o")
    return ckpt


def test_an_interrupted_run_resumes_from_its_latest_complete_checkpoint(tmp_path):
    run = tmp_path / "v0-20261001-100000"
    _checkpoint(run, 70)
    latest = _checkpoint(run, 140)
    _checkpoint(run, 210, optimizer=False)          # killed while saving
    assert T.resume_point(tmp_path) == latest


def test_a_finished_run_is_not_resumed(tmp_path):
    _checkpoint(tmp_path / "v0-x", 70)
    T.mark_training_complete(tmp_path, "policy-v1")
    assert T.training_completed(tmp_path)
    assert T.resume_point(tmp_path) is None


def test_nothing_to_resume_from_an_empty_or_missing_directory(tmp_path):
    assert T.resume_point(tmp_path / "absent") is None
    assert T.resume_point(tmp_path) is None


def test_resume_points_fall_between_evaluations_and_candidates_survive():
    cadence = T._checkpoint_cadence(250, {"checkpoint_every_steps": 70, "save_total_limit": 4})
    assert cadence["save_steps"] == 63                             # at least every 70 steps
    assert cadence["eval_steps"] % cadence["save_steps"] == 0      # an evaluation lands on a save
    assert cadence["eval_steps"] == 252                            # ~where the cut put it (250)
    assert cadence["save_total_limit"] == 4 * 4 + 1                # the 4 evaluation checkpoints kept


def test_a_short_run_saves_at_its_evaluations():
    cadence = T._checkpoint_cadence(29, {"checkpoint_every_steps": 70, "save_total_limit": 4})
    assert cadence == {"save_steps": 29, "eval_steps": 29, "save_total_limit": 5}


def test_completion_is_recorded_only_when_ms_swift_returns(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    written = []
    monkeypatch.setattr(T, "write_manifest", lambda manifest, client: written.append(manifest.status))
    manifest = SimpleNamespace(status="training", run_id="policy-v1",
                               artifacts=SimpleNamespace(staging_path=str(tmp_path)))

    monkeypatch.setattr(T, "launch", lambda config: (_ for _ in ()).throw(RuntimeError("GPU lost")))
    with pytest.raises(RuntimeError):
        T.launch_and_record(object(), manifest, None)
    assert not T.training_completed(tmp_path) and written[-1] == "failed"

    monkeypatch.setattr(T, "launch", lambda config: None)
    T.launch_and_record(object(), manifest, None)
    assert T.training_completed(tmp_path) and written[-1] == "trained"


def test_checkpoint_selection_compares_evaluation_checkpoints_only(tmp_path):
    from evaluation.checkpoint_eval import discover_checkpoints

    run = tmp_path / "v0-x"
    for step in (70, 140, 210, 280, 350, 420, 490, 560, 600):
        _checkpoint(run, step)
    (run / "checkpoint-600" / "trainer_state.json").write_text(
        json.dumps({"global_step": 600, "eval_steps": 280, "save_steps": 70}), encoding="utf-8")
    checkpoints, _best = discover_checkpoints(str(tmp_path))
    assert [c.rsplit("-", 1)[1] for c in checkpoints] == ["280", "560", "600"]   # evals + last


def test_the_pipeline_treats_an_unfinished_run_on_the_mount_as_not_trained(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from artifact_registry import paths
    from orchestration import pipeline_dag as D

    asked = []
    monkeypatch.setattr(paths, "scoped_staging_adapter_dir",
                        lambda scope, version: asked.append((scope, version)) or str(tmp_path))
    ctx = SimpleNamespace(out_version="v1", volume=SimpleNamespace(exists=lambda path: True))
    _checkpoint(tmp_path / "v0-x", 70)
    assert D._is_trained(ctx) is False          # directory and checkpoints, but no marker
    T.mark_training_complete(tmp_path, "policy-v1")
    assert D._is_trained(ctx) is True
    assert set(asked) == {("policy", "v1")}     # the policy adapter's staged run

