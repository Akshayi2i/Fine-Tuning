"""Several documents' requests in shared calls of the in-process engine (inference_core.batching_backend)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from inference_core.batching_backend import BatchingBackend
from inference_core.model_runner import Generation, ModelBackend


class _BatchOnly(ModelBackend):
    """An engine that runs batches only, records each call, and refuses a prompt holding "BAD" by failing the whole
    call - as vLLM does."""

    def __init__(self, delay: float = 0.0) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.delay = delay
        self.closed = False
        self.busy = threading.Lock()

    def generate(self, messages, config, adapter=None):
        return self.generate_batch([messages], [config], adapter)[0]

    def generate_batch(self, messages_list, configs, adapter=None):
        assert self.busy.acquire(blocking=False), "two engine calls at once"
        try:
            time.sleep(self.delay)
            texts = [m[0]["content"] for m in messages_list]
            self.calls.append((texts, adapter))
            if any("BAD" in t for t in texts):
                return [RuntimeError("prompt refused") for _ in texts]
            return [Generation(text=f"answer to {t}", tokens=[]) for t in texts]
        finally:
            self.busy.release()

    def supports_logprobs(self):
        return False

    def close(self):
        self.closed = True


def _msg(text):
    return [{"role": "user", "content": text}]


def test_requests_sent_one_at_a_time_go_to_the_engine_in_shared_calls():
    engine = _BatchOnly(delay=0.05)
    backend = BatchingBackend(engine, max_per_call=8, gather_seconds=0.2)
    assert backend.supports_submit()
    futures = [backend.submit(_msg(f"doc{d}-window{w}"), SimpleNamespace()) for d in range(3) for w in range(4)]
    answers = [f.result(timeout=10) for f in futures]
    backend.stop()
    assert [a.text for a in answers] == [f"answer to doc{d}-window{w}" for d in range(3) for w in range(4)]
    assert len(engine.calls) <= 3 and max(len(texts) for texts, _ in engine.calls) == 8
    assert backend.requests == 12 and not engine.closed          # stop() leaves the engine loaded


def test_a_call_failed_whole_is_sent_again_in_halves_so_only_the_bad_request_fails():
    engine = _BatchOnly()
    backend = BatchingBackend(engine, max_per_call=8, gather_seconds=0.2)
    futures = [backend.submit(_msg(text), SimpleNamespace()) for text in ("a", "b", "BAD", "d", "e")]
    results = [f.result(timeout=10) for f in futures]
    backend.stop()
    assert [type(r).__name__ for r in results] == ["Generation", "Generation", "RuntimeError", "Generation",
                                                   "Generation"]


def test_each_call_is_one_adapters():
    engine = _BatchOnly()
    backend = BatchingBackend(engine, gather_seconds=0.2)
    futures = [backend.submit(_msg(f"r{i}"), SimpleNamespace(), adapter="a" if i % 2 else "b") for i in range(6)]
    [f.result(timeout=10) for f in futures]
    backend.stop()
    assert all(len({adapter}) == 1 for _texts, adapter in engine.calls)
    assert {adapter for _texts, adapter in engine.calls} == {"a", "b"}


def test_every_future_settles_when_the_engine_raises():
    class Broken(_BatchOnly):
        def generate_batch(self, messages_list, configs, adapter=None):
            raise RuntimeError("engine fell over")

    backend = BatchingBackend(Broken(), gather_seconds=0.05)
    futures = [backend.submit(_msg(f"r{i}"), SimpleNamespace()) for i in range(3)]
    assert all(isinstance(f.result(timeout=10), RuntimeError) for f in futures)
    backend.stop()


def test_the_gate_reads_documents_side_by_side_and_gives_the_engine_back(monkeypatch):
    from evaluation import golden_eval

    engine = _BatchOnly()
    model = SimpleNamespace(backend=engine)
    seen = []

    def evaluate_read(documents, model, *args):
        seen.append((type(model.backend).__name__, golden_eval.documents_in_flight(model)))
        return golden_eval.GoldenTriples()

    monkeypatch.setattr(golden_eval, "_evaluate_read", evaluate_read)
    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {})
    golden_eval.evaluate([], model, object(), "/tmp", run_to_run=False)
    assert seen == [("BatchingBackend", golden_eval.DOCUMENTS_IN_FLIGHT)]
    assert model.backend is engine and not engine.closed
