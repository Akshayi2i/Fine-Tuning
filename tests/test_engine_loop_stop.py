"""The in-process vLLM engine stops an answer that repeats itself, as the served model's stream does (2026-11-11).

``LLM.chat`` returns only finished answers, so a loop in selection ran to the 12,288-token limit while production
stops the same answer early. ``model_runner.watched_chat`` runs chat's steps itself with each answer watched.
These tests run it on a stand-in engine that writes one piece of text per step, as vLLM's DELTA outputs do.
"""

from __future__ import annotations

import itertools
import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from inference_core.model_runner import STOPPED_REPEATING, VLLMBackend, watched_chat
from inference_core.runner_config import RunnerConfig


@pytest.fixture(autouse=True)
def _vllm_kinds(monkeypatch):
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = SimpleNamespace(DELTA="delta", FINAL_ONLY="final_only")
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)


def _pieces(text: str, size: int = 40) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


class _Engine:
    """``LLM`` as watched_chat uses it: each request writes its answer one piece (one token) per step."""

    def __init__(self, answers: list[str], *, empty_step_at: int | None = None) -> None:
        self.answers = answers
        self.empty_step_at = empty_step_at
        self.request_counter = itertools.count()
        self.llm_engine = self
        self.running: dict[str, list[str]] = {}
        self.written: dict[str, int] = {}
        self.added: list[tuple[str, Any, Any]] = []
        self.aborted: list[str] = []

    def preprocess_chat(self, messages):
        return [{"prompt": i} for i, _ in enumerate(messages)]

    def add_request(self, request_id, prompt, params, *, lora_request=None, tokenization_kwargs=None):
        self.added.append((request_id, params.output_kind, lora_request))
        self.running[request_id] = _pieces(self.answers[prompt["prompt"]])
        self.written[request_id] = 0

    def has_unfinished_requests(self):
        return bool(self.running)

    def abort_request(self, request_ids):
        for request_id in request_ids:
            self.aborted.append(request_id)
            self.running.pop(request_id, None)

    def step(self):
        outputs = []
        for request_id, pieces in list(self.running.items()):
            index = self.written[request_id]
            if self.empty_step_at is not None and index == self.empty_step_at and not getattr(self, "_emptied", False):
                # A step that wrote no token: vLLM 0.11.0 hands back the whole logprobs list with it.
                self._emptied = True
                every = [_logprob(p, n) for n, p in enumerate(pieces[:index])]
                outputs.append(_output(request_id, "", [], every, None, False))
                continue
            piece = pieces[index]
            self.written[request_id] = index + 1
            done = index + 1 == len(pieces)
            outputs.append(_output(request_id, piece, [index], [_logprob(piece, index)], "stop" if done else None, done))
            if done:
                del self.running[request_id]
        return outputs


def _logprob(piece: str, token_id: int) -> dict:
    return {token_id: SimpleNamespace(logprob=-0.01, decoded_token=piece, rank=1)}


def _output(request_id, text, token_ids, logprobs, finish_reason, finished):
    completion = SimpleNamespace(text=text, token_ids=token_ids, logprobs=logprobs, finish_reason=finish_reason)
    return SimpleNamespace(request_id=request_id, outputs=[completion], prompt_token_ids=[1, 2, 3], finished=finished)


def _params():
    return SimpleNamespace(output_kind=None)


def _answer(rows: int) -> str:
    return json.dumps({"forms_and_endorsements": [{"form_number": f"BP {n:02d} 18", "title": f"Form title {n}"}
                                                  for n in range(rows)]})


def _loop(rows: int) -> str:
    return json.dumps({"coverages": [{"coverage_code": "BOPGL_BI_WEBSITES", "coverage_name": {"v": f"Form {n}", "p": [17]},
                                      "limits": [{"amount": {"r": "$25,000", "v": 25000.0, "p": [16]}}]}
                                     for n in range(rows)]})


def _generation(output):
    return VLLMBackend._to_generation(output, RunnerConfig(), 1.0)


def test_a_finished_answer_comes_back_whole_with_every_token_and_logprob():
    answer = _answer(30)
    engine = _Engine([answer])
    [output] = watched_chat(engine, [["messages"]], [_params()], "adapter")
    generation = _generation(output)
    assert generation.text == answer and "".join(generation.tokens) == answer
    assert len(generation.token_logprobs) == len(_pieces(answer))
    assert generation.finish_reason == "stop" and not generation.truncated()
    assert generation.prompt_tokens == 3
    assert engine.added == [("0", "delta", "adapter")] and engine.aborted == []


def test_a_loop_is_stopped_early_and_counted_as_cut_off_while_the_others_finish():
    loop, real = _loop(600), _answer(60)
    engine = _Engine([real, loop, real])
    outputs = watched_chat(engine, [["a"], ["b"], ["c"]], [_params(), _params(), _params()], None)
    first, stopped, third = (_generation(o) for o in outputs)
    assert engine.aborted == ["1"]
    assert stopped.finish_reason == STOPPED_REPEATING and stopped.truncated()
    assert len(stopped.text) < len(loop) / 2 and loop.startswith(stopped.text)
    assert first.text == real and third.text == real               # in the order asked, the others untouched


def test_a_step_that_writes_nothing_adds_no_logprobs():
    answer = _answer(10)
    engine = _Engine([answer], empty_step_at=3)
    [output] = watched_chat(engine, [["m"]], [_params()], None)
    generation = _generation(output)
    assert generation.text == answer
    assert len(generation.token_logprobs) == len(_pieces(answer))


def test_both_engine_calls_go_through_the_watched_path():
    """generate and generate_batch: neither may call LLM.chat, which would run a loop to the limit."""
    import inspect

    source = inspect.getsource(VLLMBackend.generate) + inspect.getsource(VLLMBackend.generate_batch)
    assert "watched_chat(" in source and ".chat(" not in source.replace("watched_chat(", "")
