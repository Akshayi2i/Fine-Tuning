"""Decoding as configured, and as consistent as it can be (review 2026-10-26):
near ties and run-to-run agreement (#23), structured outputs that cannot be
silently off (#24), the server's sampling pinned and its prompt length held to
the pages (#8)."""

from __future__ import annotations

import io
import json
import sys
import types

import pytest

from inference_core.span_map import map_compact_field_spans, map_field_spans

# --------------------------------------------------------------------------
# #23 the top-2 margin, from every backend to the served flag
# --------------------------------------------------------------------------


def _tokens(text: str, size: int = 3) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def test_a_span_s_margin_is_its_weakest_token_s_lead():
    text = '{"a": "12345", "b": "x"}'
    tokens = _tokens(text)
    logprobs = [-0.01] * len(tokens)
    second = [-5.0] * len(tokens)
    a = map_field_spans(text, tokens, logprobs)["a"]
    second[a.token_start + 1] = -0.05                       # one token of "a" nearly another
    spans = map_field_spans(text, tokens, logprobs, token_top2_logprobs=second)

    assert spans["a"].top2_margin == pytest.approx(0.04)
    assert spans["b"].top2_margin == pytest.approx(4.99)
    # No second-best given, or one that does not align: no margin.
    assert map_field_spans(text, tokens, logprobs)["a"].top2_margin is None
    assert map_field_spans(text, tokens, logprobs, token_top2_logprobs=second[:-1])["a"].top2_margin is None
    # A token with none listed is left out of the minimum.
    partial = [None] * len(tokens)
    assert map_field_spans(text, tokens, logprobs, token_top2_logprobs=partial)["a"].top2_margin is None


def test_a_compact_answer_keeps_its_margins_under_the_full_paths():
    text = '{"premium": {"total": {"r": "$1,665", "v": 1665, "p": [1]}}}'
    tokens = _tokens(text)
    spans = map_compact_field_spans(text, tokens, [-0.01] * len(tokens),
                                    token_top2_logprobs=[-0.02] * len(tokens))
    assert spans["premium.total.raw"].top2_margin == pytest.approx(0.01)


def test_a_near_tie_is_flagged_on_a_served_value_only():
    from inference_core.span_map import FieldSpan
    from serving.value_flags import near_tie_flags

    spans = {
        "policy.policy_number": FieldSpan("policy.policy_number", "PA-1", 0, 4, mapped=True, top2_margin=0.02),
        "policy.effective_date": FieldSpan("policy.effective_date", "04/01/2026", 0, 4, mapped=True,
                                           top2_margin=0.5),
        "policy.rating_state": FieldSpan("policy.rating_state", None, 0, 4, mapped=True, top2_margin=0.0),
        "premium.total": FieldSpan("premium.total", 1665.0, 0, 4, mapped=True, top2_margin=None),
    }
    assert near_tie_flags(spans, 0.1) == ["policy.policy_number:near_tie"]
    assert near_tie_flags(spans, 0.6) == ["policy.effective_date:near_tie", "policy.policy_number:near_tie"]


def test_the_margin_is_configured_and_the_engine_asks_for_the_second_best():
    from inference_core.runner_config import load_runner_config
    from serving.value_flags import near_tie_margin

    assert near_tie_margin() == pytest.approx(0.1)
    assert load_runner_config().top_logprobs == 2


def _sse(*chunks: dict) -> list[bytes]:
    return [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks] + [b"data: [DONE]\n"]


def test_the_served_model_s_alternatives_are_read_from_its_stream():
    import time

    from inference_core.openai_backend import _read_stream

    def scored(token, logprob, *alternatives):
        return {"token": token, "logprob": logprob,
                "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alternatives]}

    stream = _sse(
        {"choices": [{"delta": {"content": '{"a"'}, "logprobs": {"content": [
            scored('{"', -0.01, ('{"', -0.01), ("{", -4.0)), scored('a"', -0.2, ("b", -0.1), ('a"', -0.2))]}}]},
        {"choices": [{"delta": {"content": ": 1}"}, "logprobs": {"content": [scored(": 1}", -0.01)]},
                      "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 1234}},
    )
    answer = _read_stream(stream, started=time.perf_counter())

    assert answer["tokens"] == ['{"', 'a"', ": 1}"]
    # The best OTHER token: the second when the token written led, the first when
    # the schema masked the model's own first choice; none when none is listed.
    assert answer["top2"] == [-4.0, -0.1, None]
    assert answer["prompt_tokens"] == 1234


class _Answer:
    def __init__(self, body: dict) -> None:
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return io.BytesIO(self.body)

    def __exit__(self, *exc):
        return False


def _backend(monkeypatch, stream=False):
    from inference_core import openai_backend

    monkeypatch.setattr(openai_backend.OpenAIChatBackend, "_check_served", lambda self, name: None)
    return openai_backend.OpenAIChatBackend("http://localhost:8000", model="m", stream=stream)


def test_the_payload_pins_the_server_s_sampling_and_asks_for_two_logprobs(monkeypatch):
    from inference_core.runner_config import load_runner_config

    backend = _backend(monkeypatch)
    body = backend.payload([{"role": "user", "content": "hi"}], load_runner_config())

    assert body["repetition_penalty"] == 1.0 and body["top_k"] == -1
    assert body["presence_penalty"] == 0.0 and body["frequency_penalty"] == 0.0
    assert body["temperature"] == 0.0 and body["top_p"] == 1.0
    assert body["logprobs"] is True and body["top_logprobs"] == 2


def test_an_unstreamed_answer_carries_its_second_best_and_drops_the_end_token_s(monkeypatch):
    from inference_core.runner_config import load_runner_config

    backend = _backend(monkeypatch)
    reply = {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop", "logprobs": {"content": [
        {"token": '{"a": ', "logprob": -0.01, "top_logprobs": [{"token": '{"a": ', "logprob": -0.01},
                                                                {"token": '{"b": ', "logprob": -0.03}]},
        {"token": "1}", "logprob": -0.5, "top_logprobs": [{"token": "2}", "logprob": -0.4}]},
        {"token": "<|im_end|>", "logprob": 0.0, "top_logprobs": [{"token": "x", "logprob": -9.0}]}]}}],
        "usage": {"prompt_tokens": 99}}
    monkeypatch.setattr(backend, "_request", lambda path, body=None, **kw: reply)
    generation = backend.generate([{"role": "user", "content": "hi"}], load_runner_config())

    assert generation.tokens == ['{"a": ', "1}"]
    assert generation.token_top2_logprobs == [-0.03, -0.4]
    spans = map_field_spans(generation.text, generation.tokens, generation.token_logprobs,
                            token_top2_logprobs=generation.token_top2_logprobs)
    assert spans["a"].top2_margin == pytest.approx(-0.1)      # the model preferred another: a tie and worse


class _Logprob:
    def __init__(self, logprob, rank, token):
        self.logprob, self.rank, self.decoded_token = logprob, rank, token


def test_the_in_process_engine_carries_its_second_best_too():
    from types import SimpleNamespace

    from inference_core.model_runner import VLLMBackend
    from inference_core.runner_config import RunnerConfig

    completion = SimpleNamespace(
        text='{"a": 1}', token_ids=[7, 8, 9], finish_reason="stop",
        logprobs=[{7: _Logprob(-0.01, 1, '{"a": '), 3: _Logprob(-3.0, 2, "x")},
                  {8: _Logprob(-0.3, 2, "1}"), 4: _Logprob(-0.2, 1, "2}")},
                  {9: _Logprob(0.0, 1, "<|im_end|>")}])
    generation = VLLMBackend._to_generation(SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2]),
                                            RunnerConfig(), 10.0)

    assert generation.tokens == ['{"a": ', "1}"]
    assert generation.token_top2_logprobs == [-3.0, -0.2]
    assert generation.prompt_tokens == 2


# --------------------------------------------------------------------------
# #24 structured outputs: the current class, the old one, or a refusal
# --------------------------------------------------------------------------


def _fake_vllm(monkeypatch, **classes):
    sampling = types.ModuleType("vllm.sampling_params")
    for name, value in classes.items():
        setattr(sampling, name, value)
    vllm = types.ModuleType("vllm")
    vllm.sampling_params = sampling

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm.SamplingParams = SamplingParams
    lora = types.ModuleType("vllm.lora.request")
    lora.LoRARequest = object
    for name, module in (("vllm", vllm), ("vllm.sampling_params", sampling), ("vllm.lora", types.ModuleType("x")),
                         ("vllm.lora.request", lora)):
        monkeypatch.setitem(sys.modules, name, module)


class _Params:
    def __init__(self, json=None):
        self.json = json


def test_structured_outputs_prefer_the_current_class(monkeypatch):
    from inference_core.model_runner import VLLMBackend
    from inference_core.runner_config import RunnerConfig

    _fake_vllm(monkeypatch, StructuredOutputsParams=_Params, GuidedDecodingParams=object)
    params = VLLMBackend._sampling_params(RunnerConfig(json_schema={"type": "object"}, top_logprobs=2))

    assert params.kwargs["structured_outputs"].json == {"type": "object"}
    assert "guided_decoding" not in params.kwargs
    assert params.kwargs["logprobs"] == 2
    assert params.kwargs["repetition_penalty"] == 1.0 and params.kwargs["top_k"] == -1
    assert params.kwargs["presence_penalty"] == 0.0 and params.kwargs["frequency_penalty"] == 0.0


def test_structured_outputs_fall_back_to_the_deprecated_class(monkeypatch):
    from inference_core.model_runner import VLLMBackend
    from inference_core.runner_config import RunnerConfig

    _fake_vllm(monkeypatch, GuidedDecodingParams=_Params)
    params = VLLMBackend._sampling_params(RunnerConfig(json_schema={"type": "object"}))
    assert params.kwargs["guided_decoding"].json == {"type": "object"}


def test_no_structured_outputs_class_is_fatal_when_a_schema_is_asked_for(monkeypatch):
    from inference_core.model_runner import ModelRunnerError, VLLMBackend
    from inference_core.runner_config import RunnerConfig

    _fake_vllm(monkeypatch)
    with pytest.raises(ModelRunnerError, match="neither StructuredOutputsParams nor GuidedDecodingParams"):
        VLLMBackend._sampling_params(RunnerConfig(json_schema={"type": "object"}))
    # Unconstrained (the evaluation's format-health run) needs neither.
    assert "structured_outputs" not in VLLMBackend._sampling_params(RunnerConfig()).kwargs


# --------------------------------------------------------------------------
# #8 the server's prompt length against the pages
# --------------------------------------------------------------------------


def _png(path, width, height):
    from PIL import Image

    Image.new("RGB", (width, height), "white").save(path)
    return str(path)


class _Counter:
    """A tokenizer stand-in: a token every four characters."""

    def encode(self, text, add_special_tokens=False):
        return list(range((len(text) + 3) // 4))


def test_a_prompt_is_expected_from_its_pages_and_its_text(tmp_path):
    from inference_core.prompt_tokens import check_prompt, expected_prompt
    from training.length_check import image_tokens

    budget = (200704, 1254400)
    pages = [_png(tmp_path / "p1.png", 960, 1248), _png(tmp_path / "p2.png", 1792, 1385)]
    messages = [{"role": "system", "content": "x" * 400},
                {"role": "user", "content": [{"type": "image", "image": pages[0]}, {"type": "text", "text": "y" * 80},
                                             {"type": "image", "image": pages[1]}]}]
    visual = sum(image_tokens(w, h, min_pixels=budget[0], max_pixels=budget[1])
                 for w, h in ((960, 1248), (1792, 1385)))

    counted = expected_prompt(messages, budget=budget, tokenizer=_Counter())
    assert counted["image_tokens"] == visual and counted["text_tokens"] == 120
    assert counted["expected"] == visual + 120
    assert check_prompt(visual + 130, counted)["plausible"]
    # Pages resized to the standard budget instead: far more tokens than these pages hold.
    assert not check_prompt(visual * 2, counted)["plausible"]

    bounded = expected_prompt(messages, budget=budget, tokenizer=None)
    assert bounded["expected"] is None and bounded["low"] == visual
    checked = check_prompt(visual + 300, bounded)
    assert checked["plausible"] and checked["ratio"] == round((visual + 300) / visual, 3)
    assert not check_prompt(visual - 1, bounded)["plausible"]           # fewer than its images alone
    # A page whose size cannot be read: nothing to hold the server to.
    gone = [{"role": "user", "content": [{"type": "image", "image": str(tmp_path / "missing.png")}]}]
    assert expected_prompt(gone, budget=budget, tokenizer=None) is None


def test_a_window_s_prompt_check_is_recorded_and_warned_once(tmp_path, monkeypatch, caplog):
    import inference_core.prompt_tokens as prompt_tokens
    from inference_core.model_runner import Generation
    from serving.pipeline import _check_prompt_tokens

    monkeypatch.setattr(prompt_tokens, "_tokenizer", lambda: None)
    page = _png(tmp_path / "p1.png", 960, 1248)
    messages = [{"role": "user", "content": [{"type": "image", "image": page}, {"type": "text", "text": "hello"}]}]
    warned: list[str] = []
    stat: dict = {"pages": [1]}
    _check_prompt_tokens("doc", stat, Generation(text="{}", prompt_tokens=1180), messages, warned)
    assert stat["prompt_check"]["plausible"] and not warned

    with caplog.at_level("WARNING"):
        for _ in range(2):
            stat = {"pages": [1]}
            _check_prompt_tokens("doc", stat, Generation(text="{}", prompt_tokens=2424), messages, warned)
            assert stat["prompt_check"]["plausible"] is False
    assert warned == ["doc"] and sum("may resize pages" in r.message for r in caplog.records) == 1
    # No prompt length reported, or a failed window: nothing recorded.
    stat = {}
    _check_prompt_tokens("doc", stat, Generation(text="{}"), messages, warned)
    _check_prompt_tokens("doc", stat, RuntimeError("x"), messages, warned)
    assert stat == {}


# --------------------------------------------------------------------------
# #23 run-to-run agreement on the server path (latency_check --repeat)
# --------------------------------------------------------------------------


def _row(document, attempt, number, lob="personal_auto", run="standard", error=None):
    if error:
        return {"document": document, "pdf": document, "lob": lob, "run": run, "pass": attempt, "error": error}
    return {"document": document, "pdf": document, "lob": lob, "run": run, "pass": attempt,
            "extraction": {"policy": {"policy_number": {"raw": number, "parsed": number, "page_ref": [1]},
                                      "effective_date": {"raw": "04/01/2026", "parsed": "04/01/2026",
                                                         "page_ref": [1]}}}}


def test_each_document_read_twice_is_compared_with_itself():
    from testing.latency_check import agreement_table, run_to_run

    report = run_to_run([
        _row("a.pdf", 1, "PA-1"), _row("b.pdf", 1, "PA-2"), _row("c.pdf", 1, "PA-3"),
        _row("a.pdf", 2, "PA-1"), _row("b.pdf", 2, "PA-9"), _row("c.pdf", 2, None, error="boom"),
    ])
    by_document = {entry["document"]: entry for entry in report["documents"]}
    assert by_document["a.pdf"]["agreement"] == 1.0
    assert by_document["b.pdf"]["agreement"] == 0.5
    assert report["unpaired"] == 1 and "c.pdf" not in by_document
    assert report["overall"] == pytest.approx(0.75)
    assert "| **overall** | | 0.75 |" in agreement_table(report)


def test_repeat_reads_every_document_once_then_again(monkeypatch, tmp_path):
    """All documents, then all again: a document's two readings are apart."""
    import testing.latency_check as latency_check

    read: list[str] = []

    def timed(pdf, lob, **kwargs):
        read.append(f"{pdf.name}:{kwargs['key']}")
        return {**_row(pdf.name, 0, "PA-1"), "timings_s": {"fields": 1.0, "model": 1.0, "full_text": 1.0,
                                                         "total": 1.0},
                "windows": {"split": 0, "failed": 0, "stopped_repeating": 0, "rounds": 1, "answer_tokens": 1,
                            "slowest_window": None},
                "pages": 1, "windows_planned": 1, "accuracy": None, "sent_mb": 0.0, "schema_valid": True,
                "text_coverage": None}

    pdfs = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(b"%PDF")
        pdfs.append(path)

    class _Stub:
        model = "m"
        images = "inline"
        sent_bytes = 0

        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(latency_check, "time_document", timed)
    monkeypatch.setattr(latency_check, "setup_problem", lambda args, env: None)
    monkeypatch.setattr("orchestration.detach.detach_module_if_needed", lambda *a, **k: False)
    monkeypatch.setattr("inference_core.openai_backend.OpenAIChatBackend", _Stub)
    monkeypatch.setattr("data_pipeline.ocr.served_ocr.ServedOcr", _Stub)
    monkeypatch.setattr("inference_core.model_runner.load_model", lambda *a, **k: object())
    out = tmp_path / "out"
    code = latency_check.main(["--model-url", "http://localhost:8000", "--mineru-url", "http://localhost:8001",
                               "--doc", f"{pdfs[0]}=personal_auto", "--doc", f"{pdfs[1]}=personal_auto",
                               "--repeat", "--out", str(out)])

    assert code == 0
    assert read == ["a.pdf:a", "b.pdf:b", "a.pdf:a.repeat", "b.pdf:b.repeat"]
    report = json.loads((out / "run_to_run.json").read_text(encoding="utf-8"))
    assert report["overall"] == 1.0 and len(report["documents"]) == 2
    assert "Run-to-run agreement" in (out / "summary.md").read_text(encoding="utf-8")


def test_extract_sends_a_near_tie_to_review(monkeypatch):
    """A window whose policy-number tokens all but chose others: flagged
    near_tie and sent to review; the rest, decided, are not."""
    from inference_core.model_runner import Generation
    from tests.test_review_printed_controls import _serve
    from tests.test_schema2_serving import _Replay

    original = _Replay.generate

    def generate(self, messages, config, adapter=None):
        plain = original(self, messages, config, adapter)
        start = plain.text.find('"PA-2026-0042"')
        offsets, second = 0, []
        for token in plain.tokens:
            near = start >= 0 and offsets < start + 14 and offsets + len(token) > start
            second.append(-0.03 if near else -9.0)
            offsets += len(token)
        return Generation(text=plain.text, tokens=plain.tokens, token_logprobs=plain.token_logprobs,
                          token_top2_logprobs=second)

    monkeypatch.setattr(_Replay, "generate", generate)
    result, _texts = _serve(monkeypatch=monkeypatch)

    assert "policy.policy_number:near_tie" in result.review_flags
    assert result.extraction["policy"]["policy_number"]["flagged"] is True
    assert result.extraction["policy"]["effective_date"]["flagged"] is False
    assert not any(flag.endswith(":near_tie") and not flag.startswith("policy.policy_number")
                   for flag in result.review_flags)
