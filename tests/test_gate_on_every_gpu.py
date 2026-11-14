"""The gate's readings made on every GPU at once (evaluation.golden_eval.read_on_every_gpu).

The gate's in-process engine reads one document at a time, so on the 4x H200 pod one GPU read the whole frozen
set. Each GPU's worker (evaluation.gate_shard_worker) now reads a share into the gate's saved readings, and the
gate takes them and scores everything together, as before.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import gate_shard_worker, golden_eval
from evaluation.golden_eval import GoldenDocument, split_by_pages
from evaluation.reading_cache import ReadingCache
from tests.test_invention_and_agreement import _fake_extract, _golden_docs


def _docs(pages: dict[str, int]) -> list[GoldenDocument]:
    return [GoldenDocument(source_id=sid, doc_type="policy", golden={}, lob="gl",
                           image_keys=[f"k/{sid}/{n}.png" for n in range(count)])
            for sid, count in pages.items()]


def _evaluate(documents, cache):
    return golden_eval.evaluate(documents, object(), object(), "/tmp", run_to_run=True, repeat_mode="image_only",
                                cache=cache)


# --- the split -----------------------------------------------------------------------------------------------------

def test_the_split_holds_every_document_once_with_near_equal_pages():
    pages = {"a": 120, "b": 90, "c": 60, "d": 40, "e": 30, "f": 20, "g": 10, "h": 5}
    shares = split_by_pages(_docs(pages), 4)
    assert sorted(sid for share in shares for sid in share) == sorted(pages)
    totals = [sum(pages[s] for s in share) for share in shares]
    assert totals[0] == 120                                                # the longest document, alone
    assert max(totals[1:]) - min(totals[1:]) <= 10                         # the rest shared evenly (90/85/80)
    assert split_by_pages(list(reversed(_docs(pages))), 4) == shares        # the same split however listed


def test_fewer_documents_than_gpus_leaves_shares_empty():
    assert split_by_pages(_docs({"a": 3}), 4) == [["a"], [], [], []]


# --- what the gate scores does not change --------------------------------------------------------------------------

def test_readings_made_in_shares_are_the_gate_s_own_and_it_reads_none_again(monkeypatch, tmp_path):
    calls: list = []
    _fake_extract(monkeypatch, calls)
    single = _evaluate(_golden_docs(), ReadingCache(tmp_path / "one_gpu", "k"))      # the gate as it was

    calls.clear()
    shared = ReadingCache(tmp_path / "every_gpu", "k")
    first, second = _golden_docs()
    _evaluate([first], ReadingCache(shared.dir.parent, "k"))                           # GPU 0's worker
    _evaluate([second], ReadingCache(shared.dir.parent, "k"))                          # GPU 1's worker
    assert len(calls) == 8                                                            # 2 x 3 modes, 2 repeats
    calls.clear()
    gate = _evaluate(_golden_docs(), ReadingCache(shared.dir.parent, "k"))            # then the gate
    assert calls == []                                                                # it read nothing
    assert [(e, g, m) for e, g, m in gate] == [(e, g, m) for e, g, m in single]       # and scores the same


def test_the_gate_reads_itself_what_a_worker_left(monkeypatch, tmp_path):
    calls: list = []
    _fake_extract(monkeypatch, calls)
    first, _second = _golden_docs()
    _evaluate([first], ReadingCache(tmp_path, "k"))                                   # GPU 1's worker failed
    calls.clear()
    triples = _evaluate(_golden_docs(), ReadingCache(tmp_path, "k"))
    assert {sid for sid, _mode, _order in calls} == {"p2"} and len(triples) == 6


def test_the_workers_and_the_gate_name_the_saved_readings_alike(monkeypatch):
    monkeypatch.setenv("FIDEON_GATE_CACHE", "1")
    monkeypatch.setattr("registry_utils.write_run_manifest.capture_git_commit", lambda: "abc123")
    docs = _golden_docs()
    common = dict(version="v1", corpus_version="v1", tenant_id="default", documents=docs,
                  adapter_checkpoint="checkpoint-857", calibrators={"x": 1}, thresholds=None,
                  settings=(True, "image_only", 1.0, 1.0))
    by_label = golden_eval.gate_reading_key(model_tag="policy-v1:bf16", **common)
    cache = golden_eval.reading_cache_for(None, "default", "v1", "v1", docs, SimpleNamespace(tag="policy-v1:bf16"),
                                          "checkpoint-857", {"x": 1}, None, settings=(True, "image_only", 1.0, 1.0))
    assert cache.key == by_label
    # every document of the set is in the name, whichever share a worker reads
    assert golden_eval.gate_reading_key(model_tag="policy-v1:bf16", **{**common, "documents": docs[:1]}) != by_label


# --- the worker ----------------------------------------------------------------------------------------------------

def _spec(docs, key, **extra):
    return {"version": "v1", "corpus_version": "v1", "tenant_id": "default", "calibrators": None,
            "thresholds": None, "adapter_checkpoint": "checkpoint-857", "adapter": "/ckpt",
            "label": "policy-v1:bf16", "key": key, "settings": golden_eval.gate_settings(),
            "shares": split_by_pages(docs, 2), "documents": docs, **extra}


@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    docs = _golden_docs()
    monkeypatch.setattr(golden_eval, "gate_documents", lambda *a: pytest.fail("the worker loaded the set again"))
    monkeypatch.setattr(golden_eval, "gate_reading_key", lambda **kw: f"key-{kw['model_tag']}")
    monkeypatch.setattr("evaluation.reading_cache.default_root", lambda: tmp_path)
    monkeypatch.setattr("inference_core.model_runner.release_model", lambda model: released.append(model))
    read: list = []
    monkeypatch.setattr(golden_eval, "evaluate",
                        lambda documents, model, client, root, **kw: (read.append([d.source_id for d in documents]),
                                                                      ["t"] * (3 * len(documents)))[1])
    released: list = []
    return SimpleNamespace(docs=docs, read=read, released=released)


def test_a_worker_reads_its_share_only_and_releases_the_gpu(worker_env):
    spec = _spec(worker_env.docs, "key-policy-v1:bf16")
    loaded = []
    result = gate_shard_worker.read_share(spec, 1, client=object(),
                                          loader=lambda adapter, label: loaded.append((adapter, label))
                                          or SimpleNamespace(tag=label))
    assert result["ok"] and result["documents"] == 1 and result["readings"] == 3
    assert worker_env.read == [spec["shares"][1]] and loaded == [("/ckpt", "policy-v1:bf16")]
    assert len(worker_env.released) == 1


def test_a_worker_whose_saved_readings_would_have_another_name_refuses_before_loading_the_model(worker_env):
    spec = _spec(worker_env.docs, "key-something-else")
    result = gate_shard_worker.read_share(spec, 0, client=object(), loader=lambda *a: pytest.fail("loaded"))
    assert not result["ok"] and "would not find" in result["error"] and worker_env.read == []


def test_a_model_loaded_under_another_tag_reads_nothing_and_is_let_go(worker_env):
    spec = _spec(worker_env.docs, "key-policy-v1:bf16")
    result = gate_shard_worker.read_share(spec, 0, client=object(),
                                          loader=lambda adapter, label: SimpleNamespace(tag="something-else"))
    assert not result["ok"] and "would not find" in result["error"] and worker_env.read == []
    assert len(worker_env.released) == 1


def test_a_worker_that_cannot_find_its_documents_reads_nothing(worker_env):
    spec = _spec(worker_env.docs, "key-policy-v1:bf16", shares=[["p1", "gone"], ["p2"]])
    result = gate_shard_worker.read_share(spec, 0, client=object(), loader=lambda *a: pytest.fail("loaded"))
    assert not result["ok"] and "not among the gate's documents" in result["error"]


# --- the gate starting them ----------------------------------------------------------------------------------------

class FakeProc:
    """A worker process: done at once with ``code``, or - ``code`` None - reading until stopped."""

    def __init__(self, out: Path, result: dict | None, code: int | None):
        if result is not None:
            out.write_text(json.dumps(result), encoding="utf-8")
        self.returncode = code
        self.stopped = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.stopped, self.returncode = True, -15

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


def _start(monkeypatch, tmp_path, docs, outcome, **kwargs):
    monkeypatch.setenv("FIDEON_GATE_CACHE", "1")
    monkeypatch.setattr(golden_eval, "_refuse_interrupted_freeze", lambda client, tenant: None)
    monkeypatch.setattr(golden_eval, "gate_documents", lambda client, tenant, lines: docs)
    monkeypatch.setattr(golden_eval, "gate_reading_key", lambda **kw: "k")
    monkeypatch.setattr("evaluation.reading_cache.default_root", lambda: tmp_path / "cache")
    started, procs = [], []

    def spawn(command, env, log_path):
        out = Path(command[command.index("--out") + 1])
        started.append((env["CUDA_VISIBLE_DEVICES"], int(command[command.index("--share") + 1]), env))
        result, code = outcome(env["CUDA_VISIBLE_DEVICES"])
        procs.append(FakeProc(out, result, code))
        return procs[-1]

    monkeypatch.setattr("evaluation.checkpoint_eval._spawn", spawn)
    done = golden_eval.read_on_every_gpu(
        object(), version="v1", corpus_version="v1", tenant_id="default", calibrators=None, thresholds=None,
        lines=None, adapter_checkpoint="checkpoint-857", adapter="/ckpt", label="policy-v1:bf16",
        gpus=["0", "1", "2", "3"], work_dir=tmp_path / "work", poll_seconds=0,
        **{"sleep": lambda s: None, **kwargs})
    return done, started, procs


def test_one_worker_per_gpu_each_on_its_own_card(monkeypatch, tmp_path):
    docs = _docs({"a": 9, "b": 7, "c": 5, "d": 3, "e": 1})
    done, started, _procs = _start(monkeypatch, tmp_path, docs, lambda gpu: ({"ok": True, "summary": "fine"}, 0))
    assert [(gpu, share) for gpu, share, _env in started] == [("0", 0), ("1", 1), ("2", 2), ("3", 3)]
    assert all(env["FIDEON_NO_DETACH"] == "1" for *_, env in started)
    spec = pickle.loads((tmp_path / "work" / "gate-spec.pkl").read_bytes())
    assert spec["key"] == "k" and spec["label"] == "policy-v1:bf16" and spec["shares"] == split_by_pages(docs, 4)
    assert [d.source_id for d in spec["documents"]] == [d.source_id for d in docs]      # no worker reloads them
    assert done["gpus"] == 4 and done["documents"] == 5


def test_a_failed_worker_is_reported_and_the_gate_goes_on(monkeypatch, tmp_path, caplog):
    docs = _docs({"a": 9, "b": 7, "c": 5, "d": 3})
    with caplog.at_level("WARNING"):
        done, _started, _procs = _start(monkeypatch, tmp_path, docs,
                                        lambda gpu: (None, 1) if gpu == "2" else ({"ok": True}, 0))
    assert done["workers"]["2"]["exit"] == 1
    assert any("GPU 2 did not finish its share" in r.getMessage() for r in caplog.records)


def test_workers_that_keep_nothing_for_too_long_are_stopped(monkeypatch, tmp_path, caplog):
    docs = _docs({"a": 9, "b": 7, "c": 5, "d": 3})
    ticks = iter(range(0, 100_000, 600))                  # ten minutes a poll
    with caplog.at_level("WARNING"):
        done, _started, procs = _start(monkeypatch, tmp_path, docs,
                                       lambda gpu: ({"ok": True}, 0) if gpu == "0" else (None, None),
                                       stall_s=45 * 60, clock=lambda: float(next(ticks)))
    assert [p.stopped for p in procs] == [False, True, True, True]
    assert all(done["workers"][gpu].get("stopped") for gpu in ("1", "2", "3"))
    assert any("no reading kept for 45 min" in r.getMessage() for r in caplog.records)


def test_an_error_while_waiting_leaves_no_worker_on_a_gpu(monkeypatch, tmp_path):
    docs = _docs({"a": 9, "b": 7})
    seen: list = []
    real = FakeProc.__init__

    def track(self, *args):
        real(self, *args)
        seen.append(self)

    monkeypatch.setattr(FakeProc, "__init__", track)

    def sleep(_seconds):
        raise KeyboardInterrupt                           # Ctrl-C, or anything else going wrong mid-wait

    with pytest.raises(KeyboardInterrupt):
        _start(monkeypatch, tmp_path, docs, lambda gpu: (None, None), sleep=sleep)
    assert len(seen) == 2 and all(p.stopped for p in seen)


def test_a_share_kept_whole_starts_no_worker_and_progress_is_the_status_line(monkeypatch, tmp_path, caplog):
    import re

    from evaluation.reading_cache import ReadingCache

    docs = _docs({"a": 9, "b": 7})
    monkeypatch.setattr("evaluation.reading_cache.default_root", lambda: tmp_path / "cache")
    settings = golden_eval.gate_settings()
    kept = ReadingCache(tmp_path / "cache", "k")
    for sid, mode, kind in golden_eval.planned_readings(docs[:1], settings):     # share 0 ("a") done before
        kept.put(sid, mode, {"done": True}, kind)
    with caplog.at_level("INFO"):
        done, started, _procs = _start(monkeypatch, tmp_path, docs, lambda gpu: ({"ok": True}, 0))
    assert [share for _gpu, share, _env in started] == [1]
    status = re.compile(r"golden eval: (\d+)/(\d+) reading")                  # orchestration/status.py's pattern
    lines = [m.groups() for r in caplog.records if (m := status.search(r.getMessage()))]
    planned = len(golden_eval.planned_readings(docs, settings))
    assert lines and lines[0] == (str(planned // 2), str(planned))


def test_the_gate_s_own_refusal_comes_before_any_gpu_reads(monkeypatch, tmp_path):
    monkeypatch.setenv("FIDEON_GATE_CACHE", "1")

    def refuse(client, tenant):
        raise golden_eval.GoldenEvalError("the last freeze of the eval set was interrupted")

    monkeypatch.setattr(golden_eval, "_refuse_interrupted_freeze", refuse)
    monkeypatch.setattr("evaluation.checkpoint_eval._spawn", lambda *a: pytest.fail("a worker was started"))
    with pytest.raises(golden_eval.GoldenEvalError, match="interrupted"):
        golden_eval.read_on_every_gpu(
            object(), version="v1", corpus_version="v1", tenant_id="default", calibrators=None, thresholds=None,
            lines=None, adapter_checkpoint="c", adapter="/ckpt", label="l", gpus=["0", "1"], work_dir=tmp_path)


def test_one_gpu_or_no_saved_readings_starts_no_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(golden_eval, "gate_documents", lambda client, tenant, lines: _golden_docs())
    monkeypatch.setattr("evaluation.checkpoint_eval._spawn", lambda *a: pytest.fail("a worker was started"))
    kwargs = dict(version="v1", corpus_version="v1", tenant_id="default", calibrators=None, thresholds=None,
                  lines=None, adapter_checkpoint="c", adapter="/ckpt", label="l", work_dir=tmp_path)
    monkeypatch.setenv("FIDEON_GATE_CACHE", "1")
    assert golden_eval.read_on_every_gpu(object(), gpus=["0"], **kwargs)["skipped"]
    monkeypatch.setenv("FIDEON_GATE_CACHE", "0")                     # nowhere to keep them: the gate reads all
    assert golden_eval.read_on_every_gpu(object(), gpus=["0", "1"], **kwargs)["skipped"]


def test_the_gate_stage_shares_its_reading_only_with_several_gpus_and_never_fails_on_it(monkeypatch, caplog):
    from orchestration import pipeline_dag as D

    calls, copied = [], []
    monkeypatch.setattr("evaluation.golden_eval.read_on_every_gpu", lambda client, **kw: calls.append(kw) or {})
    monkeypatch.setattr(D, "serving_adapter", lambda ctx: "adapters/policy/v1/")              # a Blob prefix
    monkeypatch.setattr("inference_core.model_runner.load_runner_config", lambda backend: backend)
    monkeypatch.setattr("inference_core.model_runner.prepare_adapter",
                        lambda adapter, client, config: copied.append((adapter, config)) or "/cache/adapter")
    ctx = SimpleNamespace(client=object(), out_version="v1", corpus="v1", tenant_id="default")
    monkeypatch.setattr(D, "_calibration_gpus", lambda: ["0"])
    D._read_gate_on_every_gpu(ctx, calibrators=None, thresholds=None, lines=["gl"], scored="c")
    assert calls == []
    monkeypatch.setattr(D, "_calibration_gpus", lambda: ["0", "1", "2", "3"])
    D._read_gate_on_every_gpu(ctx, calibrators=None, thresholds=None, lines=["gl"], scored="c")
    assert calls[0]["gpus"] == ["0", "1", "2", "3"] and calls[0]["label"] == D._serving_label(ctx)
    # copied down once, here, and the workers handed the local copy
    assert copied == [("adapters/policy/v1/", "vllm")] and calls[0]["adapter"] == "/cache/adapter"

    def broken(client, **kw):
        raise OSError("disk full")

    monkeypatch.setattr("evaluation.golden_eval.read_on_every_gpu", broken)
    with caplog.at_level("WARNING"):
        D._read_gate_on_every_gpu(ctx, calibrators=None, thresholds=None, lines=["gl"], scored="c")
    assert any("reads them all on one GPU" in r.getMessage() for r in caplog.records)
