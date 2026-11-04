"""A served document's content never reaches the logs: not its values, not an answer the model got wrong."""

from __future__ import annotations

import json
import logging

import pytest

from inference_core.model_runner import Generation, ModelRunnerError, ModelTransportError
from serving.pipeline import PipelineError
from tests.test_policy_windows import _ScriptedBackend, _serve_with, client  # noqa: F401 - the fixture

SECRET = "SECRET-INSURED-NAME"


class _Answers(_ScriptedBackend):
    """Every answer holds the secret: written whole, cut off, or as text that is not JSON."""

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def generate(self, messages, config, adapter=None) -> Generation:
        group, pages = self._window(messages, config)
        self.windows.append((group, pages))
        whole = json.dumps({"carrier": {"company_name": {"raw": SECRET, "parsed": SECRET, "page_ref": pages[:1]}}})
        text = {"whole": whole, "cut": whole[:len(whole) // 2 + 20], "not_json": f"the insured is {SECRET}"}[self.shape]
        tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
        return Generation(text=text, tokens=tokens, token_logprobs=[-0.01] * len(tokens),
                          finish_reason="length" if self.shape == "cut" else "stop")


@pytest.mark.parametrize("shape", ["whole", "cut", "not_json"])
def test_an_answer_never_reaches_the_logs_however_it_ended(client, caplog, shape):  # noqa: F811
    assert SECRET in _Answers(shape).generate(
        [{"role": "system", "content": ""}, {"role": "user", "content": "<page 1 of 2>"}],
        type("C", (), {"json_schema": {}})()).text         # the backend does write it
    with caplog.at_level(logging.DEBUG):
        try:
            _serve_with(client, _Answers(shape))
        except PipelineError as exc:
            assert SECRET not in str(exc), "an error a job records holds no document content"
    assert caplog.records, "the run logged something, so the check below looked at it"
    assert not [r.getMessage() for r in caplog.records if SECRET in r.getMessage()]


def test_a_page_text_never_reaches_the_logs(client, caplog):  # noqa: F811
    from tests import test_policy_windows as windows

    marked = [f"{text} {SECRET}" for text in windows.TEXTS]
    original, windows.TEXTS = windows.TEXTS, marked
    try:
        with caplog.at_level(logging.DEBUG):
            _serve_with(client, _ScriptedBackend())
    finally:
        windows.TEXTS = original
    assert not [r.getMessage() for r in caplog.records if SECRET in r.getMessage()]


@pytest.mark.parametrize("kind", [ModelTransportError, ModelRunnerError])
def test_a_server_error_is_the_servers_words_cut_short(kind):
    """What reaches the logs from a failed call: the server's own reply, at most 300 characters
    (``openai_backend._error_body``). The request and its pages are never in it."""
    import io
    import urllib.error
    from unittest import mock

    from inference_core.openai_backend import OpenAIChatBackend

    backend = OpenAIChatBackend("https://pod.example.test", "tok", model="m")
    status = 502 if kind is ModelTransportError else 400
    error = urllib.error.HTTPError("https://pod.example.test/v1/chat/completions", status, "x", {},
                                   io.BytesIO(b"maximum context length exceeded " + b"x" * 5000))
    with mock.patch("common.transport.open_url", side_effect=error), pytest.raises(kind) as raised:
        backend._request("/v1/chat/completions", json.dumps({"messages": SECRET}).encode())
    assert SECRET not in str(raised.value) and len(str(raised.value)) < 400
