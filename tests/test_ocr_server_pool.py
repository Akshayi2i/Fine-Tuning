"""Page reading on every GPU: one MinerU server per GPU, the documents shared between them.

One server reads about 1.7 pages a second whether it is sent one document or four at a time
(measured 2026-11-14 on the 4x H200 pod), so `scripts/mineru_server.sh start` runs one per GPU
and the OCR stage reads through all of them (data_pipeline.ocr.served_ocr.PooledServedEngine).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_pipeline.ocr.served_ocr import (
    PooledServedEngine,
    ServedMinerUEngine,
    served_engine,
    served_urls,
)
from orchestration.ocr_server import OcrServerError, managed_ocr_server

FOUR = ",".join(f"http://localhost:{18001 + i}" for i in range(4))
SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" / "mineru_server.sh").read_text(encoding="utf-8")


class Runner:
    def __init__(self):
        self.calls: list[str] = []
        self.stop_ports: list[str | None] = []

    def __call__(self, command, **kwargs):
        self.calls.append(command[-1])
        if command[-1] == "stop":
            self.stop_ports.append((kwargs.get("env") or {}).get("MINERU_STOP_PORTS"))
        return SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture
def env(monkeypatch, tmp_path):
    for key in ("MINERU_URL", "MINERU_URLS", "MINERU_TOKEN", "FIDEON_REMOTE_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    file = tmp_path / ".env"
    file.write_text(f'MINERU_URL="http://localhost:18001"\nMINERU_URLS="{FOUR}"\nMINERU_TOKEN="made-by-start"\n',
                    encoding="utf-8")
    return file


def _clock():
    ticks = iter(range(0, 100_000, 10))
    return lambda: float(next(ticks))


# --- the addresses -------------------------------------------------------------------------------------------------

def test_a_server_list_is_split_on_commas_and_blanks_are_dropped():
    assert served_urls(" http://a:1 , ,http://b:2,") == ["http://a:1", "http://b:2"]
    assert served_urls(None) == [] and served_urls("") == []


def test_one_address_is_one_engine_and_several_are_a_pool():
    assert type(served_engine(["http://localhost:18001"])) is ServedMinerUEngine
    pool = served_engine(served_urls(FOUR), "t")
    assert isinstance(pool, PooledServedEngine) and pool.urls == served_urls(FOUR) and len(pool.engines) == 4


def test_from_the_environment_every_server_is_read_through_before_the_one(monkeypatch):
    monkeypatch.setenv("MINERU_URL", "http://localhost:18001")
    monkeypatch.setenv("MINERU_URLS", FOUR)
    engine = ServedMinerUEngine.from_environment()
    assert isinstance(engine, PooledServedEngine) and len(engine.engines) == 4
    monkeypatch.delenv("MINERU_URLS")
    assert type(ServedMinerUEngine.from_environment()) is ServedMinerUEngine


# --- the pool ------------------------------------------------------------------------------------------------------

def test_the_pool_records_what_one_server_records():
    pool = served_engine(served_urls(FOUR))
    single = ServedMinerUEngine("http://localhost:18001")
    assert pool.environment() == single.environment() and pool.meta() == single.meta()
    assert (pool.version, pool.tier) == (single.version, single.tier)


def test_servers_held_to_different_pins_are_refused(monkeypatch):
    versions = iter(["4.0.12", "4.0.12", "4.0.11", "4.0.12"])

    class Engine:
        def __init__(self, url, token=None, **_kwargs):
            self.version, self.tier = next(versions), "basic"

    monkeypatch.setattr("data_pipeline.ocr.served_ocr.ServedMinerUEngine", Engine)
    with pytest.raises(ValueError, match="different pins"):
        PooledServedEngine(served_urls(FOUR))
    with pytest.raises(ValueError, match="at least one URL"):
        PooledServedEngine([])


def test_documents_read_at_once_are_spread_evenly_over_the_servers():
    pool = served_engine(served_urls(FOUR))
    release = threading.Event()
    started = threading.Barrier(9)
    taken: list[int] = []

    class Fake:
        def __init__(self, index):
            self.index = index

        def process(self, pdf_bytes, **_kwargs):
            taken.append(self.index)
            started.wait(timeout=10)
            release.wait(timeout=10)
            return [self.index]

    pool.engines = [Fake(i) for i in range(4)]
    results = []
    threads = [threading.Thread(target=lambda: results.append(pool.process(b"%PDF", device="cuda",
                                                                           max_long_side_px=1)))
               for _ in range(8)]
    for thread in threads:
        thread.start()
    started.wait(timeout=10)                       # all 8 in flight at once
    assert sorted(taken) == [0, 0, 1, 1, 2, 2, 3, 3]
    release.set()
    for thread in threads:
        thread.join(timeout=10)
    assert len(results) == 8 and pool._busy == [0, 0, 0, 0]


class Server:
    """A served MinerU: answers, or - dead - fails at once."""

    def __init__(self, index, calls, dead=False):
        self.index, self.calls, self.dead = index, calls, dead

    def process(self, pdf_bytes, **_kwargs):
        self.calls.append(self.index)
        if self.dead or pdf_bytes == b"broken":
            raise RuntimeError(f"server {self.index} failed")
        return [self.index]


def _read(pool, data=b"%PDF"):
    return pool.process(data, device="cuda", max_long_side_px=1)


def test_a_document_a_dead_server_fails_is_read_on_another():
    calls: list[int] = []
    pool = served_engine(served_urls(FOUR)[:2])
    pool.engines = [Server(0, calls, dead=True), Server(1, calls)]
    assert _read(pool) == [1] and calls == [0, 1]
    assert pool._busy == [0, 0] and pool._failed_in_a_row == [1, 0]


def test_a_server_that_keeps_failing_is_taken_out_and_gets_nothing_more(caplog):
    calls: list[int] = []
    pool = served_engine(served_urls(FOUR))
    pool.engines = [Server(0, calls, dead=True)] + [Server(i, calls) for i in (1, 2, 3)]
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert _read(pool) != [0]
    assert pool.out == {0}
    assert any("taken out of the pool" in r.getMessage() for r in caplog.records)
    calls.clear()
    for _ in range(12):
        _read(pool)
    assert 0 not in calls and len(calls) == 12                       # the dead one never again, nothing failed


def test_a_document_that_fails_on_two_servers_is_the_document_s_fault():
    calls: list[int] = []
    pool = served_engine(served_urls(FOUR))
    pool.engines = [Server(i, calls) for i in range(4)]
    with pytest.raises(RuntimeError, match="failed"):
        _read(pool, b"broken")
    assert len(calls) == 2 and pool.out == set()                     # tried twice, then its error stands
    assert _read(pool) is not None


def test_the_last_server_is_never_taken_out():
    calls: list[int] = []
    pool = served_engine(served_urls(FOUR)[:2])
    pool.engines = [Server(0, calls, dead=True), Server(1, calls, dead=True)]
    for _ in range(5):
        with pytest.raises(RuntimeError):
            _read(pool)
    assert len(pool.out) == 1 and pool._busy == [0, 0]


def test_a_pool_shares_the_cpus_between_its_documents_in_flight():
    from data_pipeline.ocr.served_ocr import render_share

    pool = served_engine(served_urls(FOUR))
    assert all(engine.reader.render_workers == render_share(16) for engine in pool.engines)
    assert render_share(16) == max(1, (os.cpu_count() or 1) // 16)
    one = served_engine(served_urls(FOUR)[:1])
    assert one.reader.render_workers == min(8, os.cpu_count() or 1)  # a single server as before


def test_the_list_else_the_one_rule_is_the_same_everywhere():
    from data_pipeline.ocr.served_ocr import configured_urls

    assert configured_urls({"MINERU_URLS": FOUR, "MINERU_URL": "http://localhost:18001"}) == served_urls(FOUR)
    assert configured_urls({"MINERU_URLS": "  ", "MINERU_URL": "http://localhost:18001"}) == ["http://localhost:18001"]
    assert configured_urls({}) == []


# --- the server manager --------------------------------------------------------------------------------------------

def test_every_server_started_is_waited_for_read_through_and_stopped(env):
    runner = Runner()
    answers = {url: iter([False, True]) for url in served_urls(FOUR)}
    answers["http://localhost:18004"] = iter([False, False, False, True])     # the last one loads slowest
    seen = {}

    def probe(url, token, **_kwargs):
        seen.setdefault("probed", set()).add(url)
        return next(answers[url])

    with managed_ocr_server(True, runner=runner, probe=probe, sleep=lambda s: None, clock=_clock(),
                            can_start=lambda: True, env_file=env) as engine:
        assert isinstance(engine, PooledServedEngine) and engine.urls == served_urls(FOUR)
        seen["env"] = (os.environ["MINERU_URL"], os.environ["MINERU_URLS"], os.environ["MINERU_TOKEN"])
    assert runner.calls == ["start", "stop"]
    assert seen["probed"] == set(served_urls(FOUR))
    assert seen["env"] == ("http://localhost:18001", FOUR, "made-by-start")
    assert not {"MINERU_URL", "MINERU_URLS", "MINERU_TOKEN"} & set(os.environ)          # put back


def test_a_server_that_never_answers_is_left_out_and_the_others_read(env, caplog):
    runner = Runner()
    with caplog.at_level("WARNING"), managed_ocr_server(
            True, runner=runner, probe=lambda url, *a, **k: not url.endswith("18003"), sleep=lambda s: None,
            clock=_clock(), can_start=lambda: True, env_file=env, timeout_s=60) as engine:
        assert engine.urls == ["http://localhost:18001", "http://localhost:18002", "http://localhost:18004"]
        assert os.environ["MINERU_URLS"] == ",".join(engine.urls)        # what the smoke command reads
    assert any("18003" in r.getMessage() and "reading with the 3 that do" in r.getMessage() for r in caplog.records)
    assert runner.calls == ["start", "stop"] and runner.stop_ports == [None]          # all of ours stopped


def test_no_server_answering_is_an_error_and_all_are_stopped(env):
    runner = Runner()
    with pytest.raises(OcrServerError, match="18001.*18004"), managed_ocr_server(
            True, runner=runner, probe=lambda *a, **k: False, sleep=lambda s: None,
            clock=_clock(), can_start=lambda: True, env_file=env, timeout_s=60):
        pass
    assert runner.calls == ["start", "stop"]


def test_servers_already_up_are_used_as_they_are_and_never_stopped(env, monkeypatch):
    monkeypatch.setenv("MINERU_URL", "http://localhost:18001")
    monkeypatch.setenv("MINERU_URLS", FOUR)
    runner = Runner()
    with managed_ocr_server(True, runner=runner, probe=lambda *a, **k: True, can_start=lambda: True,
                            env_file=env) as engine:
        assert engine is None
    assert runner.calls == []
    # one of the four is down: start brings it up (it skips those already running), and the block stops that one
    # only - the three that were up before it are somebody's
    with managed_ocr_server(True, runner=runner, probe=lambda url, *a, **k: url != "http://localhost:18002"
                            or runner.calls == ["start"], sleep=lambda s: None, clock=_clock(),
                            can_start=lambda: True, env_file=env) as engine:
        assert isinstance(engine, PooledServedEngine) and len(engine.urls) == 4
    assert runner.calls == ["start", "stop"] and runner.stop_ports == ["18002"]


def test_a_list_with_a_server_somebody_else_runs_is_never_started_or_stopped(env, monkeypatch):
    monkeypatch.setenv("MINERU_URLS", "http://localhost:18001,https://pod-18002.proxy.runpod.net")
    runner = Runner()
    with managed_ocr_server(True, runner=runner, probe=lambda *a, **k: False, can_start=lambda: True,
                            env_file=env) as engine:
        assert engine is None
    assert runner.calls == []


def test_the_ocr_stage_keeps_every_server_full(env, monkeypatch):
    """stage_preprocessing with four servers started: 4 jobs each, 16 documents at once."""
    from orchestration import pipeline_dag as D

    runner = Runner()
    used = []
    monkeypatch.setattr("orchestration.ocr_server.subprocess.run", runner)
    monkeypatch.setattr("orchestration.ocr_server.healthy", lambda *a, **k: True)
    monkeypatch.setattr("orchestration.ocr_server.installed", lambda: True)
    monkeypatch.setattr("orchestration.ocr_server.REPO", env.parent)
    monkeypatch.setattr(D, "_pending_ocr", lambda ctx, engine: {"policy": ["policy_0001"]})
    monkeypatch.setattr("data_pipeline.ocr.run_mineru.process_batch",
                        lambda ids, doc_type, raw, engine, **kw: (used.append((engine, kw["workers"])),
                                                                  {"processed": ids, "skipped": [], "failed": []})[1])
    ctx = SimpleNamespace(ocr_engine=None, raw=None, tenant_id="default", ocr_workers=1, ocr_meta_cache=None)
    assert D.stage_preprocessing(ctx).status == "completed"
    [(engine, workers)] = used
    assert isinstance(engine, PooledServedEngine) and len(engine.engines) == 4 and workers == 16
    assert runner.calls == ["start", "stop"]


def test_servers_somebody_else_runs_get_the_workers_asked_for(monkeypatch):
    """A pool of remote servers (not ours): --ocr-workers stands, it is not raised to 4 per server."""
    from orchestration import pipeline_dag as D

    used = []
    monkeypatch.setattr(D, "_pending_ocr", lambda ctx, engine: {"policy": ["policy_0001"]})
    monkeypatch.setattr("data_pipeline.ocr.run_mineru.process_batch",
                        lambda ids, doc_type, raw, engine, **kw: (used.append(kw["workers"]),
                                                                  {"processed": ids, "skipped": [], "failed": []})[1])
    remote = SimpleNamespace(urls=["https://a.example.test", "https://b.example.test"], environment=lambda: None)
    ctx = SimpleNamespace(ocr_engine=remote, raw=None, tenant_id="default", ocr_workers=2, ocr_meta_cache=None)
    assert D.stage_preprocessing(ctx).status == "completed" and used == [2]


# --- the smoke run and the script ----------------------------------------------------------------------------------

def test_the_smoke_run_accepts_a_server_list_alone():
    from orchestration.smoke_run import assert_served_ocr

    monkey_installed = "orchestration.ocr_server.installed"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(monkey_installed, lambda: False)
        assert_served_ocr(["ocr"], {"MINERU_URLS": FOUR})                # no MINERU_URL: still named

def test_the_smoke_ocr_step_reads_through_every_server(tmp_path, monkeypatch):
    from orchestration.smoke_run import commands

    monkeypatch.setenv("MINERU_URL", "http://localhost:18001")
    monkeypatch.setenv("MINERU_URLS", FOUR)
    ocr = commands(batch_dir=tmp_path, subset_dir=tmp_path / "s", tenant="smoke", version="v0",
                   check_out=tmp_path / "c")["ocr"]
    assert ocr[ocr.index("--mineru-url") + 1] == FOUR and ocr[ocr.index("--workers") + 1] == "16"
    monkeypatch.delenv("MINERU_URLS")
    ocr = commands(batch_dir=tmp_path, subset_dir=tmp_path / "s", tenant="smoke", version="v0",
                   check_out=tmp_path / "c")["ocr"]
    assert ocr[ocr.index("--mineru-url") + 1] == "http://localhost:18001" and ocr[ocr.index("--workers") + 1] == "4"


def test_the_script_runs_one_server_per_gpu_and_stops_them_all():
    assert 'MINERU_SERVERS:-auto' in SCRIPT and "nvidia-smi -L" in SCRIPT
    assert "export CUDA_VISIBLE_DEVICES=$i" in SCRIPT and "port=$((PORT + i))" in SCRIPT
    assert '("MINERU_URLS", ",".join(urls))' in SCRIPT
    # every server gets a folder of its own (each writes a lock file where it runs)
    assert 'echo "$STATE/gpu$1"' in SCRIPT and 'cd "$dir"' in SCRIPT
    stop = SCRIPT[SCRIPT.index("  stop)"):]
    assert "running_ports" in stop and "fideon-mineru-server(-[0-9]+)?" in stop
    # the pipeline names the ports it started, so a server that was up before it stays up
    assert 'ports="$MINERU_STOP_PORTS"' in stop
    check = SCRIPT[SCRIPT.index("  check)"):SCRIPT.index("  stop)")]
    assert 'urls="${MINERU_URLS:-${MINERU_URL:-}}"' in check
