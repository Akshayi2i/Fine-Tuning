"""The frozen eval set is checked before the work that needs it, not at the gate.

The gate scores ``golden-eval-set/{tenant}/``, a one-time copy of a corpus's test split that nothing in the
build makes. Found missing at stage 8, it was learned of after the whole training run. A gated run now stops
in its first seconds, and training says so while there is time.
"""

from __future__ import annotations

import logging

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation.freeze_eval_set import manifest_key
from evaluation.golden_eval import _freeze_hint
from orchestration.pipeline_dag import (
    PipelineError,
    Stage,
    StageResult,
    assert_eval_set_frozen,
    run_stages,
    warn_eval_set_not_frozen,
)
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import make_context


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def _freeze(client, tenant_id=None) -> None:
    client.write_json(manifest_key(tenant_id=tenant_id), {"frozen_from_corpus": "v1", "source_ids": ["policy_0001"]})


def _gate_stage(ran: list[str]) -> Stage:
    def run(_ctx):
        ran.append("gate")
        return StageResult("evaluation_gate", "completed")

    return Stage(8, "evaluation_gate", "package", False, run)


def test_a_gated_run_stops_when_the_eval_set_is_not_frozen(client, controller):
    ctx = make_context(client, controller, dry_run=False, corpus_version="v1")
    with pytest.raises(PipelineError) as stop:
        assert_eval_set_frozen(ctx)
    message = str(stop.value)
    assert "not frozen" in message and "Nothing was run" in message
    assert "freeze-eval-set --corpus v1" in message and "--mark-reviewed" in message


def test_a_frozen_set_lets_the_run_go_on(client, controller):
    _freeze(client)
    assert_eval_set_frozen(make_context(client, controller, dry_run=False, corpus_version="v1"))


def test_a_dry_run_checks_nothing(client, controller):
    assert_eval_set_frozen(make_context(client, controller, dry_run=True))


def test_one_tenants_frozen_set_does_not_stand_for_anothers(client, controller):
    _freeze(client, tenant_id="acme")
    with pytest.raises(PipelineError):
        assert_eval_set_frozen(make_context(client, controller, dry_run=False, tenant_id="globex",
                                            corpus_version="v1"))


def test_run_stages_stops_at_its_start_and_never_scores(client, controller):
    ran: list[str] = []
    ctx = make_context(client, controller, dry_run=False, corpus_version="v1")
    report = run_stages(ctx, (_gate_stage(ran),), command="all")
    assert ran == []
    assert report.failed_at == "evaluation_gate"
    assert "freeze-eval-set" in report.results[-1].detail


def test_a_frozen_set_lets_run_stages_reach_the_gate(client, controller):
    _freeze(client)
    ran: list[str] = []
    ctx = make_context(client, controller, dry_run=False, corpus_version="v1")
    run_stages(ctx, (_gate_stage(ran),), command="all")
    assert ran == ["gate"]


def test_a_run_that_builds_the_corpus_is_not_stopped_for_a_set_it_cannot_have_yet(client, controller):
    ran: list[str] = []
    ctx = make_context(client, controller, dry_run=False, corpus_version="v1")
    build = Stage(4, "dataset_build", "finetune", False,
                  lambda _c: ran.append("build") or StageResult("dataset_build", "completed"))
    run_stages(ctx, (build, _gate_stage(ran)), command="all")
    assert ran == ["build", "gate"]


def test_training_warns_when_the_set_is_not_frozen(client, controller, caplog):
    ctx = make_context(client, controller, dry_run=False, corpus_version="v1")
    with caplog.at_level(logging.WARNING):
        warn_eval_set_not_frozen(ctx)
    assert any("NOT frozen" in r.getMessage() and "freeze-eval-set" in r.getMessage() for r in caplog.records)


def test_training_is_quiet_when_the_set_is_frozen(client, controller, caplog):
    _freeze(client)
    ctx = make_context(client, controller, dry_run=False, corpus_version="v1")
    with caplog.at_level(logging.WARNING):
        warn_eval_set_not_frozen(ctx)
    assert not caplog.records


def test_the_gates_error_names_the_step_that_makes_the_set(client):
    hint = _freeze_hint(client, "v3", None)
    assert "not frozen yet" in hint and "freeze-eval-set --corpus v3" in hint and "--mark-reviewed" in hint


def test_the_gates_error_adds_no_hint_when_the_set_is_frozen(client):
    _freeze(client)
    assert _freeze_hint(client, "v3", None) == ""
