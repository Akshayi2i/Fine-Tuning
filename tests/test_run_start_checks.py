"""Checks at the start of a real run, for what used to be found hours in.

A SAS token that runs out mid-cycle, a registry that will not take a write, and a corpus already built whose
test split was never frozen each failed at the first use, after the training run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, BlobError, InMemoryBackend
from artifact_registry.sas import sas_expiry, sas_problems
from evaluation.freeze_eval_set import manifest_key
from orchestration.pipeline_dag import (
    PipelineError,
    Stage,
    StageResult,
    assert_storage_ready,
    run_stages,
)
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import make_context

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


@pytest.fixture(autouse=True)
def no_sas_in_the_environment(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_SAS_TOKEN", raising=False)
    monkeypatch.delenv("AZURE_RAW_SAS_TOKEN", raising=False)


def _token(expiry: datetime) -> str:
    return f"sv=2022-11-02&ss=b&srt=sco&sp=rwdlac&se={expiry.strftime('%Y-%m-%dT%H:%M:%SZ')}&sig=SECRETSIG"


# --------------------------------------------------------------------------
# The SAS expiry
# --------------------------------------------------------------------------


def test_the_expiry_is_read_from_the_token():
    assert sas_expiry(_token(NOW)) == NOW
    assert sas_expiry("?" + _token(NOW)) == NOW
    assert sas_expiry("se=2026-12-01&sig=x") == datetime(2026, 12, 1, tzinfo=UTC)


def test_a_token_naming_no_expiry_is_not_judged():
    assert sas_expiry("") is None and sas_expiry(None) is None
    assert sas_expiry("sv=2022&sig=x") is None and sas_expiry("se=soon") is None


@pytest.mark.parametrize("left,verdict", [
    (timedelta(hours=-3), "refused"), (timedelta(minutes=30), "refused"),
    (timedelta(hours=5), "warned"), (timedelta(days=3), "fine"),
])
def test_what_each_remaining_time_means(left, verdict):
    refusals, warnings = sas_problems({"AZURE_STORAGE_SAS_TOKEN": _token(NOW + left)}, NOW)
    assert {"refused": bool(refusals), "warned": bool(warnings), "fine": not refusals and not warnings}[verdict]


def test_the_message_names_the_setting_and_never_the_token():
    refusals, _ = sas_problems({"AZURE_RAW_SAS_TOKEN": _token(NOW - timedelta(hours=1))}, NOW)
    assert refusals and "AZURE_RAW_SAS_TOKEN" in refusals[0] and "SECRETSIG" not in refusals[0]


# --------------------------------------------------------------------------
# The start of a run
# --------------------------------------------------------------------------


def test_an_expired_token_stops_a_real_run_before_any_work(client, controller, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", _token(datetime.now(UTC) - timedelta(hours=2)))
    with pytest.raises(PipelineError, match="expired.*Nothing was run") as stop:
        assert_storage_ready(make_context(client, controller, dry_run=False))
    assert "SECRETSIG" not in str(stop.value)


def test_a_token_with_a_day_left_only_warns(client, controller, monkeypatch, caplog):
    monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", _token(datetime.now(UTC) + timedelta(hours=5)))
    with caplog.at_level("WARNING"):
        assert_storage_ready(make_context(client, controller, dry_run=False))
    assert any("expires at" in r.getMessage() for r in caplog.records)


def test_a_registry_that_takes_no_write_stops_the_run_at_its_start(client, controller, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise BlobError("403 AuthorizationFailure")

    monkeypatch.setattr(client, "write_json", refuse)
    with pytest.raises(PipelineError, match="this run could not save its result.*Nothing has been run"):
        assert_storage_ready(make_context(client, controller, dry_run=False))


def test_the_probe_leaves_nothing_behind(client, controller):
    assert_storage_ready(make_context(client, controller, dry_run=False))
    assert not [k for k in client.list("") if k.endswith("_preflight.json")]


def test_a_dry_run_touches_no_storage(client, controller, monkeypatch):
    monkeypatch.setattr(client, "write_json", lambda *_a, **_k: pytest.fail("a dry run wrote"))
    assert_storage_ready(make_context(client, controller, dry_run=True))


def test_run_stages_reports_a_bad_login_as_the_first_stage_failing(client, controller, monkeypatch):
    monkeypatch.setenv("AZURE_STORAGE_SAS_TOKEN", _token(datetime.now(UTC) - timedelta(hours=2)))
    ran: list[str] = []

    def run(_ctx):
        ran.append("ran")
        return StageResult("calibrate", "completed")

    report = run_stages(make_context(client, controller, dry_run=False), (Stage(7, "calibrate", "package", False, run),),
                        command="all")
    assert ran == [] and report.failed_at == "calibrate"


# --------------------------------------------------------------------------
# `all` on a corpus that is already built
# --------------------------------------------------------------------------


def _stages(ran: list[str]) -> tuple[Stage, ...]:
    def build(_ctx):
        ran.append("build")
        return StageResult("dataset_build", "completed")

    def gate(_ctx):
        ran.append("gate")
        return StageResult("evaluation_gate", "completed")

    return Stage(4, "dataset_build", "finetune", False, build), Stage(8, "evaluation_gate", "package", False, gate)


def test_all_on_a_corpus_already_built_needs_the_frozen_set_before_it_starts(client, controller):
    client.write_json(paths.corpus_manifest("v1"), {"corpus_version": "v1"})
    ran: list[str] = []
    ctx = make_context(client, controller, dry_run=False, corpus_version="v1")
    report = run_stages(ctx, _stages(ran), command="all")
    assert ran == [] and report.failed_at == "dataset_build"
    assert "not frozen" in report.results[-1].detail


def test_all_on_a_corpus_already_built_goes_on_when_the_set_is_frozen(client, controller):
    client.write_json(paths.corpus_manifest("v1"), {"corpus_version": "v1"})
    client.write_json(manifest_key(), {"frozen_from_corpus": "v1"})
    ran: list[str] = []
    run_stages(make_context(client, controller, dry_run=False, corpus_version="v1"), _stages(ran), command="all")
    assert ran == ["build", "gate"]


def test_all_on_a_new_corpus_is_not_stopped(client, controller):
    ran: list[str] = []
    run_stages(make_context(client, controller, dry_run=False, corpus_version="v9"), _stages(ran), command="all")
    assert ran == ["build", "gate"]
