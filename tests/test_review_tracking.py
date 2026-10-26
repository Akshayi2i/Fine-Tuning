"""Review 2026-10-26 #45: tracking and notifications.

- Every manifest write mirrors the run - and each stage's wall time - into
  MLflow when ``MLFLOW_TRACKING_URI`` is set (``log_to_tracker``, which had no
  caller); a no-op otherwise.
- The ``notifications`` block of pipeline.yaml is read: a stage that failed, a
  gate that blocked and a command that finished post to a webhook (HTTPS only,
  ``FIDEON_NOTIFY_WEBHOOK_URL``) naming the run, version, stage, status, tier
  and a short reason - never a document's content.
"""

from __future__ import annotations

import sys
import types

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import notify
from orchestration.runpod_controller import LocalBackend, RunPodController
from registry_utils import write_run_manifest as W

HOOK = "https://hooks.example.test/fideon"


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


@pytest.fixture
def posted(monkeypatch) -> list[dict]:
    """Every webhook post, as its decoded JSON; the URL set to an HTTPS hook."""
    import json

    sent: list[dict] = []
    monkeypatch.setenv("FIDEON_NOTIFY_WEBHOOK_URL", HOOK)
    monkeypatch.setattr(notify, "_post", lambda url, data, timeout: sent.append(
        {"url": url, **json.loads(data.decode("utf-8"))}))
    return sent


class _FakeMlflow(types.ModuleType):
    """The calls log_to_tracker makes, recorded; runs found again by their tag."""

    def __init__(self):
        super().__init__("mlflow")
        self.runs: dict[str, dict] = {}
        self.current: str | None = None

    def search_runs(self, filter_string, max_results, output_format):
        run_id = filter_string.split("'")[1]
        return [types.SimpleNamespace(info=types.SimpleNamespace(run_id=run_id))] if run_id in self.runs else []

    def start_run(self, run_id=None, run_name=None, tags=None):
        fake = self

        class _Run:
            def __enter__(self_inner):
                key = run_id or (tags or {})["fideon_run_id"]
                fake.runs.setdefault(key, {"params": {}, "metrics": {}, "tags": dict(tags or {}), "starts": 0})
                fake.runs[key]["starts"] += 1
                fake.current = key

            def __exit__(self_inner, *exc):
                fake.current = None
                return False

        return _Run()

    def log_params(self, params):
        self.runs[self.current]["params"].update(params)

    def log_metrics(self, metrics):
        self.runs[self.current]["metrics"].update(metrics)

    def set_tag(self, key, value):
        self.runs[self.current]["tags"][key] = value


# --------------------------------------------------------------------------
# The tracker
# --------------------------------------------------------------------------


def test_the_tracker_is_a_no_op_without_mlflow_configured(client, monkeypatch):
    from tests.test_registry import _scoped

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)            # importing it would fail
    W.write_manifest(_scoped("policy-v7", "policy", ["policy"]), client)
    W.record_stage_seconds("policy-v7", "training", 12.5)


def test_each_manifest_write_and_each_stage_reach_one_tracker_run(client, monkeypatch):
    from tests.test_registry import _scoped

    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setattr(W, "_STAGE_SECONDS", {})

    W.record_stage_seconds("policy-v7", "dataset_build", 30.0)
    W.write_manifest(_scoped("policy-v7", "policy", ["policy"]), client)
    W.record_stage_seconds("policy-v7", "training", 3600.0)

    assert list(fake.runs) == ["policy-v7"], "one MLflow run per run id, found again by its tag"
    run = fake.runs["policy-v7"]
    assert run["starts"] == 3
    assert run["metrics"]["stage_seconds.dataset_build"] == 30.0
    assert run["metrics"]["stage_seconds.training"] == 3600.0
    assert run["params"]["lora_rank"] == 16 and run["tags"]["status"]


def test_a_tracker_that_fails_never_fails_the_write(client, monkeypatch):
    from tests.test_registry import _scoped

    broken = _FakeMlflow()
    broken.search_runs = lambda **kw: (_ for _ in ()).throw(ConnectionError("tracker down"))
    monkeypatch.setitem(sys.modules, "mlflow", broken)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    assert W.write_manifest(_scoped("policy-v8", "policy", ["policy"]), client)


def test_the_pipeline_records_each_stages_wall_time(client, controller, monkeypatch):
    from orchestration.pipeline_dag import run_stages, stages_for
    from tests.test_orchestration import make_context, seed_corpus

    monkeypatch.setattr(W, "_STAGE_SECONDS", {})
    seed_corpus(client)
    run_stages(make_context(client, controller), stages_for("finetune"), command="finetune")
    times = W._STAGE_SECONDS["policy-v1"]
    assert {"labeling", "dataset_build", "training"} <= set(times)
    assert all(isinstance(s, float) for s in times.values())


# --------------------------------------------------------------------------
# The webhook
# --------------------------------------------------------------------------


def test_nothing_is_posted_without_a_webhook(monkeypatch):
    monkeypatch.delenv("FIDEON_NOTIFY_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(notify, "_post", lambda *a: pytest.fail("posted without a URL"))
    assert notify.notify("stage_failure", {"event": "stage_failure"}) is False


def test_an_unencrypted_webhook_is_refused_and_the_run_goes_on(monkeypatch):
    monkeypatch.setenv("FIDEON_NOTIFY_WEBHOOK_URL", "http://hooks.example.test/x")
    monkeypatch.setattr(notify, "_post", lambda *a: pytest.fail("posted over http"))
    assert notify.notify("stage_failure", {"event": "stage_failure"}) is False


def test_an_event_switched_off_is_not_posted(posted):
    settings = {"webhook_url_env": "FIDEON_NOTIFY_WEBHOOK_URL", "on_stage_failure": False,
                "on_gate_block": True, "on_run_complete": True, "timeout_seconds": 5}
    assert notify.notify("stage_failure", {"event": "stage_failure"}, settings=settings) is False
    assert posted == []


def test_the_body_names_the_run_and_carries_a_short_reason_only():
    long = "the corpus has no val rows " + "x" * 500 + "\nsecond line with page text"
    body = notify.event_body("stage_failure", run_id="policy-v3", version="v3", stage="dataset_build",
                             status="failed", tier=None, reason=long, command="finetune", tenant="t")
    assert set(body) == {"source", "event", "run_id", "version", "command", "tenant", "stage", "status",
                         "tier", "reason"}
    assert len(body["reason"]) <= notify.MAX_REASON_CHARS and "second line" not in body["reason"]


def test_a_failed_stage_posts_once_and_the_run_completion_says_how_it_ended(client, controller, posted):
    from orchestration.pipeline_dag import PipelineError, Stage, run_stages
    from tests.test_orchestration import make_context

    def broken(_ctx):
        raise PipelineError("no labeled, OCR'd documents to build a corpus from")

    ctx = make_context(client, controller, out_version="v5")
    report = run_stages(ctx, (Stage(4, "dataset_build", "finetune", False, broken),), command="finetune")
    notify.run_complete(report, ctx)

    assert [p["event"] for p in posted] == ["stage_failure", "run_complete"]
    failure, done = posted
    assert failure["url"] == HOOK
    assert (failure["run_id"], failure["version"], failure["stage"], failure["status"]) == (
        "policy-v5", "v5", "dataset_build", "failed")
    assert "no labeled" in failure["reason"]
    assert (done["status"], done["stage"], done["command"]) == ("failed", "dataset_build", "finetune")


def test_a_gate_block_posts_its_event(client, controller, posted):
    from orchestration.pipeline_dag import GateBlocked, Stage, run_stages
    from tests.test_orchestration import make_context

    def blocked(_ctx):
        raise GateBlocked("field_normalized_match below its floor")

    ctx = make_context(client, controller, out_version="v6")
    run_stages(ctx, (Stage(8, "evaluation_gate", "package", False, blocked),), command="all")
    [event] = posted
    assert (event["event"], event["stage"], event["status"]) == ("gate_block", "evaluation_gate", "blocked")


def test_a_command_refused_before_its_first_stage_posts_its_failure(client, controller, posted):
    """``package`` of a version nothing staged fails its precondition: told too."""
    from orchestration.pipeline_dag import Stage, StageResult, run_stages
    from tests.test_orchestration import make_context

    ctx = make_context(client, controller, out_version="v6")
    run_stages(ctx, (Stage(7, "calibrate", "package", False, lambda _c: StageResult("calibrate", "completed")),),
               command="package")
    [event] = posted
    assert (event["event"], event["stage"], event["status"]) == ("stage_failure", "calibrate", "failed")


def test_a_completed_run_posts_its_completion(client, controller, posted):
    from orchestration.pipeline_dag import Stage, StageResult, run_stages
    from tests.test_orchestration import make_context

    ctx = make_context(client, controller, out_version="v9")
    report = run_stages(ctx, (Stage(3, "labeling", "finetune", False,
                                    lambda _c: StageResult("labeling", "completed", "ok")),), command="finetune")
    notify.run_complete(report, ctx)
    [event] = posted
    assert (event["event"], event["status"], event["reason"]) == ("run_complete", "completed", None)


def test_the_shipped_config_posts_every_event():
    from orchestration.settings import notification_settings

    settings = notification_settings()
    assert settings["webhook_url_env"] == "FIDEON_NOTIFY_WEBHOOK_URL"
    assert settings["on_stage_failure"] and settings["on_gate_block"] and settings["on_run_complete"]
