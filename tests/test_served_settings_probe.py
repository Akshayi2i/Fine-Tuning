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
                 tokenize: bool = True) -> None:
        self.base_url, self.model, self.token, self.timeout = "https://pod-8000.proxy.runpod.net", "policy-v3", None, 30
        self.whitespace_rule, self.logprobs_mode, self.tokenize = whitespace_rule, logprobs_mode, tokenize
        self.asked: list[dict] = []

    def _request(self, path, body=None, *, stream=False, timeout=None):
        request = json.loads(body.decode("utf-8"))
        self.asked.append({"path": path, **request})
        if path == "/tokenize":
            if not self.tokenize:
                raise ModelRunnerError("404 from /tokenize: Not Found")
            token = ONE_TOKEN.get(request["prompt"])
            return {"tokens": [token] if token else [1, 2]}
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
    assert [a["path"] for a in server.asked].count("/v1/chat/completions") == 2
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
