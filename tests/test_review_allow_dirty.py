"""Review 2026-10-26: ``--allow-dirty`` reaches training from the pipeline.

``training.train`` refuses a real training run on a working tree with
uncommitted changes unless ``allow_dirty`` (``--allow-dirty``, or
``FIDEON_ALLOW_DIRTY_TREE=1``). ``orchestration.run finetune`` / ``all`` take
the flag - parsed before the command detaches on the pod, and carried in the
argv the detached run is started with - and the training stage passes it on.
"""

from __future__ import annotations

import pytest

from orchestration import run as cli


@pytest.mark.parametrize("command", ["finetune", "all"])
def test_the_flag_is_taken_and_reaches_the_context(command):
    args = cli.build_parser().parse_args([command, "--skip-ingest", "--allow-dirty"])
    assert cli.flag_problem(args) is None
    assert cli.build_context(args, client=object(), controller=object()).allow_dirty is True
    args = cli.build_parser().parse_args([command, "--skip-ingest"])
    assert cli.build_context(args, client=object(), controller=object()).allow_dirty is False


def test_package_does_not_take_it():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["package", "--version", "v1", "--allow-dirty"])


def test_the_flag_is_in_the_argv_the_detached_run_starts_with(monkeypatch):
    from orchestration import detach

    started = {}
    monkeypatch.setattr(detach, "should_detach", lambda: True)
    monkeypatch.setattr(detach, "detach", lambda command, hint: started.setdefault("command", command) and "run")
    monkeypatch.setattr(cli, "fresh_problem", lambda args, client=None: None)
    assert cli.main(["finetune", "--skip-ingest", "--allow-dirty"]) == 0
    assert "--allow-dirty" in started["command"]


@pytest.mark.parametrize("allowed", [True, False])
def test_the_training_stage_passes_it_to_training(allowed, monkeypatch):
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from orchestration.pipeline_dag import run_stages, stages_for
    from orchestration.runpod_controller import LocalBackend, RunPodController
    from tests.test_orchestration import make_context, seed_corpus
    from training import train as T

    seen = []
    real = T.train

    def spy(**kwargs):
        seen.append(kwargs.get("allow_dirty"))
        return real(**kwargs)

    monkeypatch.setattr(T, "train", spy)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    controller = RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")
    seed_corpus(client)
    run_stages(make_context(client, controller, allow_dirty=allowed), stages_for("finetune"), command="finetune")
    assert seen == [allowed]
