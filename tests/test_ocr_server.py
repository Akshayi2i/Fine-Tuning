"""The pipeline runs the OCR server for the OCR stage and stops it after (orchestration.ocr_server)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from orchestration.ocr_server import OcrServerError, managed_ocr_server


class Runner:
    """A stand-in for subprocess.run: records what the script was asked to do."""

    def __init__(self, fail: str | None = None):
        self.calls: list[str] = []
        self.fail = fail

    def __call__(self, command, **_kwargs):
        action = command[-1]
        self.calls.append(action)
        return SimpleNamespace(returncode=1 if action == self.fail else 0, stdout="", stderr="boom")


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.delenv("MINERU_URL", raising=False)
    monkeypatch.delenv("MINERU_TOKEN", raising=False)
    monkeypatch.delenv("FIDEON_REMOTE_TOKEN", raising=False)
    file = tmp_path / ".env"
    file.write_text('MINERU_URL="http://localhost:8001"\nMINERU_TOKEN="made-by-start"\n', encoding="utf-8")
    return file


def _clock():
    ticks = iter(range(0, 100_000, 10))
    return lambda: float(next(ticks))


def test_nothing_to_read_starts_nothing(env):
    runner = Runner()
    with managed_ocr_server(False, runner=runner, can_start=lambda: True, env_file=env) as engine:
        assert engine is None
    assert runner.calls == []


def test_a_server_somebody_else_runs_is_used_as_it_is_and_never_stopped(env, monkeypatch):
    monkeypatch.setenv("MINERU_URL", "https://pod-8001.proxy.runpod.net")
    runner = Runner()
    with managed_ocr_server(True, runner=runner, probe=lambda *a, **k: False, can_start=lambda: True,
                            env_file=env) as engine:
        assert engine is None
    assert runner.calls == []


def test_a_local_server_already_up_is_used_and_left_running(env, monkeypatch):
    monkeypatch.setenv("MINERU_URL", "http://localhost:8001")
    runner = Runner()
    with managed_ocr_server(True, runner=runner, probe=lambda *a, **k: True, can_start=lambda: True,
                            env_file=env) as engine:
        assert engine is None
    assert runner.calls == []


def test_with_no_server_installed_nothing_is_started_and_the_stage_refuses_as_before(env):
    runner = Runner()
    with managed_ocr_server(True, runner=runner, can_start=lambda: False, env_file=env) as engine:
        assert engine is None
    assert runner.calls == []


def test_a_needed_server_is_started_waited_for_used_and_stopped(env):
    runner = Runner()
    answers = iter([False, False, True])           # loading, loading, then it answers
    seen = {}
    with managed_ocr_server(True, runner=runner, probe=lambda *a, **k: next(answers), sleep=lambda s: None,
                            clock=_clock(), can_start=lambda: True, env_file=env) as engine:
        assert engine is not None and engine.reader.mineru_url.startswith("http://localhost:8001")
        seen["url"], seen["token"] = os.environ["MINERU_URL"], os.environ["MINERU_TOKEN"]
        assert runner.calls == ["start"]           # still up inside the block
    assert runner.calls == ["start", "stop"]       # the GPU is free after it
    assert seen == {"url": "http://localhost:8001", "token": "made-by-start"}
    assert "MINERU_URL" not in os.environ and "MINERU_TOKEN" not in os.environ    # put back


def test_it_is_stopped_even_when_the_reading_fails(env):
    runner = Runner()
    with pytest.raises(RuntimeError, match="reading failed"), managed_ocr_server(
            True, runner=runner, probe=lambda *a, **k: True, sleep=lambda s: None,
            clock=_clock(), can_start=lambda: True, env_file=env):
        raise RuntimeError("reading failed")
    assert runner.calls == ["start", "stop"]


def test_a_server_that_never_answers_is_an_error_and_is_stopped(env):
    runner = Runner()
    with pytest.raises(OcrServerError, match="did not answer"), managed_ocr_server(
            True, runner=runner, probe=lambda *a, **k: False, sleep=lambda s: None,
            clock=_clock(), can_start=lambda: True, env_file=env, timeout_s=60):
        pass
    assert runner.calls == ["start", "stop"]


def test_a_failed_start_is_an_error_and_the_server_is_still_stopped(env):
    runner = Runner(fail="start")
    with pytest.raises(OcrServerError, match="start failed"), managed_ocr_server(
            True, runner=runner, can_start=lambda: True, env_file=env):
        pass
    assert runner.calls == ["start", "stop"]


def test_a_stop_that_fails_is_reported_not_raised(env, caplog):
    runner = Runner(fail="stop")
    with managed_ocr_server(True, runner=runner, probe=lambda *a, **k: True, sleep=lambda s: None,
                            clock=_clock(), can_start=lambda: True, env_file=env):
        pass
    assert runner.calls == ["start", "stop"]
    assert "stop failed" in caplog.text


def test_the_ocr_stage_reads_through_the_server_it_started(env, monkeypatch):
    """stage_preprocessing: a pending document, no engine, a server the stage starts."""
    from orchestration import pipeline_dag as D

    runner = Runner()
    engine_used = []
    monkeypatch.setattr("orchestration.ocr_server.subprocess.run", runner)
    monkeypatch.setattr("orchestration.ocr_server.healthy", lambda *a, **k: True)
    monkeypatch.setattr("orchestration.ocr_server.installed", lambda: True)
    monkeypatch.setattr("orchestration.ocr_server.REPO", env.parent)
    monkeypatch.setattr(D, "_pending_ocr", lambda ctx, engine: {"policy": ["policy_0001"]})
    monkeypatch.setattr("data_pipeline.ocr.run_mineru.process_batch",
                        lambda ids, doc_type, raw, engine, **kw: (engine_used.append((engine, kw["workers"])),
                                                                  {"processed": ids, "skipped": [], "failed": []})[1])
    ctx = SimpleNamespace(ocr_engine=None, raw=None, tenant_id="default", ocr_workers=1, ocr_meta_cache=None)
    result = D.stage_preprocessing(ctx)
    assert result.status == "completed" and result.detail.startswith("1 processed")
    [(engine, workers)] = engine_used
    assert engine.reader.mineru_url.startswith("http://localhost:8001") and workers == 4
    assert runner.calls == ["start", "stop"]
