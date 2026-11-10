"""Production's model server is started as the release was trained and calibrated for, or it is not served.

The gate reads with its own engine, which forbids whitespace in structured answers and returns raw logprobs itself
(inference_core.model_runner.STRUCTURED_OUTPUTS_ENGINE). Production reads through the pod's stock vLLM, where only the
server's start flags decide both (vLLM 0.11: xgrammar takes disable_any_whitespace from the server's config alone),
and nothing checked them (2026-11-09).
"""

from __future__ import annotations

import json
import logging

import pytest

from inference_core.model_runner import ModelRunnerError
from inference_core.openai_backend import OpenAIChatBackend

ONE_TOKEN = {" ": 220, "\n": 198, "  ": 256, "\t": 197, " banana": 43096}


class FakeServer(OpenAIChatBackend):
    """The two settings as a vLLM 0.11 server applies them, without a network."""

    def __init__(self, *, whitespace_rule: bool = True, logprobs_mode: str = "raw_logprobs",
                 tokenize: bool = True, max_model_len: int | None = 32768, reads_images: bool = True,
                 version: str | None = "0.11.0", spacing: str = '{"a": "x", "b": 1}') -> None:
        self.base_url, self.model, self.token, self.timeout = "https://pod-8000.proxy.runpod.net", "policy-v3", None, 30
        self.whitespace_rule, self.logprobs_mode, self.tokenize = whitespace_rule, logprobs_mode, tokenize
        self.max_model_len, self.reads_images = max_model_len, reads_images
        self.version, self.spacing = version, spacing
        self.asked: list[dict] = []

    def _request(self, path, body=None, *, stream=False, timeout=None):
        if path == "/version":
            if self.version is None:
                raise ModelRunnerError("404 from /version: Not Found")
            if self.version == "html":
                raise json.JSONDecodeError("Expecting value", "<html>", 0)    # a proxy's page, as _request reads it
            if self.version == "list":
                return ["0.11.0"]
            return {"version": self.version}
        if path == "/v1/models":
            # As vLLM 0.11 lists them: the length on the base model's entry, the adapter naming it as parent.
            return {"data": [{"id": "/workspace/models/Qwen3-VL-8B-Instruct", "max_model_len": self.max_model_len},
                             {"id": "policy-v3", "parent": "/workspace/models/Qwen3-VL-8B-Instruct"}]}
        request = json.loads(body.decode("utf-8"))
        self.asked.append({"path": path, **request})
        if path == "/tokenize":
            if not self.tokenize:
                raise ModelRunnerError("404 from /tokenize: Not Found")
            token = ONE_TOKEN.get(request["prompt"])
            return {"tokens": [token] if token else [1, 2]}
        content = request.get("messages", [{}])[0].get("content")
        if isinstance(content, list) and any(part.get("type") == "image_url" for part in content):
            if not self.reads_images:
                # vLLM 0.11.0 serving Qwen3-VL with a LoRA, without inference_core.vllm_patches' fix.
                raise ModelRunnerError("500 from /v1/chat/completions: AssertionError in _lora_shrink: "
                                       "token_lora_mapping.size(0) == M")
            assert request["max_tokens"] == 1 and set(request["mm_processor_kwargs"]) == {"min_pixels", "max_pixels"}
            return {"choices": [{"message": {"content": "OK"}}]}
        if "response_format" in request and "logit_bias" not in request:
            assert request["response_format"]["json_schema"]["name"] == "spacing"
            return {"choices": [{"message": {"content": self.spacing}}]}
        if "response_format" in request:
            pushed = request["logit_bias"]
            assert set(pushed) == {str(t) for t in (220, 198, 256, 197)} and set(pushed.values()) == {100}
            # With the rule the grammar allows only "{}"; without it the pushed whitespace wins where it may.
            return {"choices": [{"message": {"content": "{}" if self.whitespace_rule else "{        "}}]}
        assert request["logit_bias"] == {"43096": 100} and request["max_tokens"] == 1
        written, alternatives = {
            "raw_logprobs": (-14.2, [-0.003, -6.1]),        # taken before the bias: the word is unlikely
            "processed_logprobs": (-0.0, [-0.0, -95.4]),    # taken after it: the pushed word is near certain
            "raw_logits": (17.5, [31.0, 17.5]),            # logits, not log-probabilities
        }[self.logprobs_mode]
        return {"choices": [{"logprobs": {"content": [
            {"token": " banana", "logprob": written,
             "top_logprobs": [{"token": "OK", "logprob": alternatives[0]}, {"token": "Ok", "logprob": alternatives[1]}]},
        ]}}]}


def test_a_server_started_as_configured_has_no_problems():
    server = FakeServer()
    assert server.server_setting_problems() == []
    # spacing, whitespace, logprobs, a page image
    assert [a["path"] for a in server.asked].count("/v1/chat/completions") == 4
    assert all(a.get("temperature") == 0.0 for a in server.asked if a["path"] != "/tokenize")


def test_whitespace_allowed_in_structured_answers_is_found():
    [problem] = FakeServer(whitespace_rule=False).server_setting_problems()
    assert "whitespace" in problem and "disable_any_whitespace" in problem


@pytest.mark.parametrize("mode,said", [("processed_logprobs", "processed log-probabilities"),
                                       ("raw_logits", "logits, not log-probabilities")])
def test_logprobs_other_than_raw_are_found(mode, said):
    [problem] = FakeServer(logprobs_mode=mode).server_setting_problems()
    assert said in problem and "--logprobs-mode raw_logprobs" in problem


def test_both_wrong_are_both_said():
    assert len(FakeServer(whitespace_rule=False, logprobs_mode="raw_logits").server_setting_problems()) == 2


def test_production_refuses_a_server_set_up_otherwise():
    from azure_service.runtime import check_server_settings
    from serving.vllm_entrypoint import ColdStartError

    check_server_settings(FakeServer())
    with pytest.raises(ColdStartError, match="not started as the release was trained and calibrated for"):
        check_server_settings(FakeServer(logprobs_mode="processed_logprobs"))


def test_a_server_that_cannot_be_probed_is_warned_about_not_refused(caplog):
    from azure_service.runtime import check_server_settings

    with caplog.at_level(logging.WARNING):
        check_server_settings(FakeServer(tokenize=False))
    assert "could not be checked" in caplog.text


def test_a_server_with_a_shorter_context_than_training_is_found():
    """Review 2026-11-10, finding 3: windows are sized, and the model trained, for 32,768 tokens."""
    [problem] = FakeServer(max_model_len=16384).server_setting_problems()
    assert "16384" in problem and "--max-model-len 32768" in problem
    assert FakeServer(max_model_len=None).server_setting_problems() == []      # a server that does not say


def test_a_server_running_another_vllm_is_found():
    """2026-11-10 review: the serving pod runs no code of ours, so nothing pinned its vLLM."""
    [problem] = FakeServer(version="0.11.1").server_setting_problems()
    assert "vLLM 0.11.1" in problem and "vllm==0.11.0" in problem
    [problem] = FakeServer(version=None).server_setting_problems()
    assert "could not be read" in problem


def test_the_version_check_names_an_unreadable_answer_and_takes_a_local_build():
    """Code review 2026-11-10: a page that is not JSON, or JSON of another shape, crashed the check; a local build
    tag of the release was refused."""
    assert FakeServer(version="0.11.0+cu128").server_setting_problems() == []
    [problem] = FakeServer(version="html").server_setting_problems()
    assert "could not be read" in problem
    [problem] = FakeServer(version="list").server_setting_problems()
    assert "of no stated version" in problem


def test_the_pinned_vllm_is_the_one_the_serving_lock_installs():
    from pathlib import Path

    from inference_core.openai_backend import PINNED_VLLM

    lock = (Path(__file__).resolve().parents[1] / "requirements-serve.lock").read_text(encoding="utf-8")
    assert f"vllm=={PINNED_VLLM}" in lock


def test_structured_answers_spaced_otherwise_than_training_are_found():
    """Training answers are json.dumps' ", " and ": "; that the server writes the same was only a comment."""
    [problem] = FakeServer(spacing='{"a":"x","b":1}').server_setting_problems()
    assert "spaces structured answers otherwise" in problem and '{"a":"x","b":1}' in problem


def test_a_server_that_cannot_read_a_page_image_through_the_adapter_is_found():
    """Review 2026-11-10, finding 1: a stock vLLM 0.11.0 fails Qwen3-VL's first image request through a LoRA; the
    text-only probes passed it, and production reads every page image through the adapter."""
    [problem] = FakeServer(reads_images=False).server_setting_problems()
    assert "page image" in problem and "vllm_patches" in problem
