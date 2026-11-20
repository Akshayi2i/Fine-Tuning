"""Several documents' requests in one engine call: the in-process engine made to take requests one at a time.

The in-process vLLM engine (``model_runner.VLLMBackend``) runs a call to its end and cannot take a request while
another runs, so the golden eval read one document at a time on it (``golden_eval.documents_in_flight``), and a
document's page groups seldom filled a call: cycle 2's gate read ~17 page groups a minute per GPU against ~30 rows a
minute in checkpoint selection, ~10.5 hours for the frozen set on three GPUs.

:class:`BatchingBackend` wraps such an engine. :meth:`~BatchingBackend.submit` queues a request and returns its
future at once, so several documents are read side by side (``supports_submit``) and serving sends each page group
as it is ready (``serving.pipeline._read_as_answered``). One thread takes what is queued - waiting
:data:`GATHER_SECONDS` for more when the queue runs short, never more than :data:`MAX_REQUESTS_PER_CALL` - and sends
it to the engine in ONE call, a call per adapter.

What each request is answered with does not change: the engine still answers each request on its own sampling
parameters, and serving merges a document's answers in planned order whichever finished first. A call vLLM fails
whole (one prompt it refuses fails every prompt of the call) is sent again in halves, down to the request that
failed it, so one document's bad page never fails another document's page group - as serving's own
``_batch_in_halves`` does within a document.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from inference_core.model_runner import Generation, ModelBackend, RunnerConfig

log = logging.getLogger(__name__)

#: Requests in one engine call at most: serving's own cap per call (``serving.pipeline.MAX_WINDOWS_PER_BATCH``),
#: the size its call timeout is set for.
MAX_REQUESTS_PER_CALL = 96

#: How long the sending thread waits for more requests once the queue holds fewer than a full call.
GATHER_SECONDS = 0.5


@dataclass
class _Queued:
    messages: list[dict[str, Any]]
    config: RunnerConfig
    adapter: str | None
    future: Future


class BatchingBackend(ModelBackend):
    """``inner`` (a backend that runs batches only) answering requests sent one at a time, in shared calls."""

    def __init__(self, inner: ModelBackend, *, max_per_call: int = MAX_REQUESTS_PER_CALL,
                 gather_seconds: float = GATHER_SECONDS) -> None:
        self.inner = inner
        self.max_per_call = max(1, int(max_per_call))
        self.gather_seconds = max(0.0, float(gather_seconds))
        self._queue: list[_Queued] = []
        self._ready = threading.Condition()
        self._engine = threading.Lock()          # the engine runs one call at a time, whoever sends it
        self._closed = False
        self._thread: threading.Thread | None = None
        #: Engine calls made and requests sent, for the log and the tests.
        self.calls = 0
        self.requests = 0

    def __getattr__(self, name: str) -> Any:
        # What the wrapped backend offers besides the backend interface (its engine, its LoRA settings).
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    # -- the backend interface ------------------------------------------------------------------------------------

    def supports_logprobs(self) -> bool:
        return self.inner.supports_logprobs()

    def supports_submit(self) -> bool:
        return True

    def generate(self, messages: list[dict[str, Any]], config: RunnerConfig,
                 adapter: str | None = None) -> Generation:
        with self._engine:
            return self.inner.generate(messages, config, adapter=adapter)

    def generate_batch(self, messages_list: list[list[dict[str, Any]]], configs: list[RunnerConfig],
                       adapter: str | None = None) -> list[Generation | Exception]:
        with self._engine:
            return self.inner.generate_batch(messages_list, configs, adapter=adapter)

    def submit(self, messages: list[dict[str, Any]], config: RunnerConfig,
               adapter: str | None = None) -> Future:
        future: Future = Future()
        with self._ready:
            if self._closed:
                raise RuntimeError("the batching backend is closed")
            self._queue.append(_Queued(messages, config, adapter, future))
            if self._thread is None:
                self._thread = threading.Thread(target=self._send_loop, name="batching-backend", daemon=True)
                self._thread.start()
            self._ready.notify()
        return future

    def stop(self) -> None:
        """Send what is queued, then end the sending thread; the wrapped engine stays loaded."""
        with self._ready:
            self._closed = True
            self._ready.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=600)

    def close(self) -> None:
        self.stop()
        self.inner.close()

    # -- the sending thread ---------------------------------------------------------------------------------------

    def _take(self) -> list[_Queued]:
        """The next call's requests: one adapter's, up to :attr:`max_per_call`, after a short wait for more."""
        with self._ready:
            while not self._queue and not self._closed:
                self._ready.wait()
            if not self._queue:
                return []
            deadline = time.monotonic() + self.gather_seconds
            while len(self._queue) < self.max_per_call and not self._closed:
                left = deadline - time.monotonic()
                if left <= 0 or not self._ready.wait(timeout=left):
                    break
            adapter = self._queue[0].adapter
            taken = [item for item in self._queue if item.adapter == adapter][:self.max_per_call]
            chosen = {id(item) for item in taken}
            self._queue = [item for item in self._queue if id(item) not in chosen]
            return taken

    def _send_loop(self) -> None:
        while True:
            taken = self._take()
            if not taken:
                return
            try:
                results = self._answered(taken)
            except Exception as exc:  # noqa: BLE001 - every future must settle, or its reader waits forever
                results = [exc] * len(taken)
            for item, result in zip(taken, results, strict=True):
                if not item.future.done():
                    item.future.set_result(result)

    def _answered(self, taken: list[_Queued]) -> list[Generation | Exception]:
        """One engine call for ``taken``; a call failed whole is sent again in halves, down to its bad request."""
        with self._engine:
            results = self.inner.generate_batch([item.messages for item in taken], [item.config for item in taken],
                                                adapter=taken[0].adapter)
        self.calls += 1
        self.requests += len(taken)
        failed_whole = (len(taken) > 1 and all(isinstance(r, Exception) for r in results)
                        and len({str(r) for r in results}) == 1)
        if not failed_whole:
            return list(results)
        log.warning("an engine call of %d request(s) failed whole (%s): sent again in halves",
                    len(taken), results[0])
        middle = len(taken) // 2
        return self._answered(taken[:middle]) + self._answered(taken[middle:])
