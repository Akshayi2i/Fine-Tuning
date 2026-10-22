"""IMPL-07 — the shared inference primitive.

This is the file that guards **test == prod at the primitive level**. If
``build_messages`` produces different output in evaluation, serving, and testing,
then their numbers describe three different systems, and no downstream test would
notice.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from inference_core import model_runner, span_map
from inference_core.input_builder import (
    InputBuilderError,
    build_messages,
    build_training_row,
    page_images_for,
)
from inference_core.model_runner import EchoBackend, Generation, ModelRunnerError, load_model
from inference_core.runner_config import RunnerConfig, load_runner_config

PAGES = ["processed/default/policy/policy_0001/page_1.png"]
#: One markdown string PER PAGE, aligned with PAGES — never one joined blob.
OCR = ["# POLICY\n\n**Applicant** Rivera Fabrication LLC\n"]


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


# --------------------------------------------------------------------------
# The parity guarantee
# --------------------------------------------------------------------------

def test_build_messages_is_byte_identical_across_contexts():
    """Evaluation, serving, and testing must assemble inputs the same way.

    They call this function; the test asserts repeated calls agree, which is what
    makes the shared primitive meaningful rather than merely conventional.
    """
    a = build_messages("policy", PAGES, OCR, "ocr_plus_image")
    b = build_messages("policy", PAGES, OCR, "ocr_plus_image")
    assert json.dumps(a.messages) == json.dumps(b.messages)
    assert a.prompt_fingerprint == b.prompt_fingerprint


def test_training_row_and_inference_share_one_prompt():
    """The corpus row and the request must carry the same system prompt.

    Divergence here degrades a fine-tuned model and is invisible in training
    metrics, because training never sees the serving prompt (arch §7).
    """
    row = build_training_row(
        "policy", "policy_0001", PAGES, OCR, "ocr_plus_image", '{"insured_name":"X"}'
    )
    served = build_messages("policy", PAGES, OCR, "ocr_plus_image")
    assert row["messages"][0] == served.messages[0]
    assert row["messages"][1] == served.messages[1]
    assert row["messages"][2]["role"] == "assistant"      # only the corpus row has one


def test_prompt_drift_against_the_corpus_is_detected():
    built = build_messages("policy", PAGES, OCR, "ocr_plus_image")
    built.assert_matches_corpus(
        {"prompt_template_version": built.prompt_template_version,
         "schema_version": built.schema_version}
    )
    with pytest.raises(InputBuilderError, match="drift"):
        built.assert_matches_corpus({"prompt_template_version": "0.9.0"})


# --------------------------------------------------------------------------
# Modality handling
# --------------------------------------------------------------------------

def test_image_only_omits_the_ocr_block_and_declares_the_absence():
    built = build_messages("policy", PAGES, None, "image_only")
    content = built.messages[1]["content"]

    # Images plus marker-only text blocks — no OCR body anywhere.
    assert [b["type"] for b in content] == ["image", "text"]
    assert content[1]["text"] == "<page 1 of 1>"
    assert "no OCR text" in built.system_prompt()


def test_ocr_supplied_to_image_only_is_refused():
    """Silently accepting it would serve the wrong regime — and the prompt has
    already told the model no OCR exists."""
    with pytest.raises(InputBuilderError, match="image_only was requested but OCR text"):
        build_messages("policy", PAGES, OCR, "image_only")


def test_missing_ocr_for_an_ocr_mode_is_refused():
    with pytest.raises(InputBuilderError, match="requires OCR text"):
        build_messages("policy", PAGES, None, "ocr_plus_image")


def test_noisy_ocr_renders_the_same_prompt_as_ocr_plus_image():
    """The noise is in the data, not the instruction (arch §6)."""
    a = build_messages("policy", PAGES, OCR, "ocr_plus_image")
    b = build_messages("policy", PAGES, ["corrupted OCR"], "noisy_ocr_image")
    assert a.system_prompt() == b.system_prompt()


def test_at_least_one_image_is_always_required():
    """Every mode is image-bearing; image_only removes OCR, not the pages."""
    with pytest.raises(InputBuilderError, match="at least one page image"):
        build_messages("policy", [], OCR, "ocr_plus_image")


# --------------------------------------------------------------------------
# Multi-page ordering
# --------------------------------------------------------------------------

def test_page_order_is_preserved_exactly():
    """Ordering is positionally meaningful under Interleaved-MRoPE (arch §3)."""
    pages = [f"page_{i}.png" for i in range(1, 5)]
    texts = [f"text for page {i}" for i in range(1, 5)]
    built = build_messages("policy", pages, texts, "ocr_plus_image")
    images = [b["image"] for b in built.messages[1]["content"] if b["type"] == "image"]
    assert images == pages

    reversed_built = build_messages("policy", list(reversed(pages)), texts, "ocr_plus_image")
    reversed_images = [b["image"] for b in reversed_built.messages[1]["content"] if b["type"] == "image"]
    assert reversed_images == list(reversed(pages))
    assert images != reversed_images


def test_a_windows_pages_come_back_in_page_order():
    """A window reads a subset of the pages; out-of-order pages change what the model sees."""
    from artifact_registry.paths import processed_page

    paths = page_images_for(
        {"page_count": 10}, processed_page, "policy", "policy_0001", None, pages=[7, 2, 4]
    )
    assert [p.rsplit("page_", 1)[1] for p in paths] == ["2.png", "4.png", "7.png"]


def test_out_of_range_page_is_refused():
    from artifact_registry.paths import processed_page

    with pytest.raises(InputBuilderError, match="out of range"):
        page_images_for({"page_count": 3}, processed_page, "policy", "policy_0001", None, pages=[9])


# --------------------------------------------------------------------------
# Span mapping
# --------------------------------------------------------------------------

def _tokenize(text: str, size: int = 4) -> tuple[list[str], list[float]]:
    tokens = [text[i:i + size] for i in range(0, len(text), size)]
    return tokens, [-0.01 * (i % 5) for i in range(len(tokens))]


def test_scalar_null_and_list_row_fields_all_map():
    text = ('{"carrier":"Sentinel","valuation_date":null,'
            '"claims":[{"claim_number":"WC24-00817","paid":12400.0}]}')
    tokens, logprobs = _tokenize(text)
    spans = span_map.map_field_spans(text, tokens, logprobs)

    assert spans["carrier"].value == "Sentinel"
    assert spans["valuation_date"].value is None          # null is a real value
    assert spans["claims[0].paid"].value == 12400.0       # rows addressed individually
    assert all(s.mapped for s in spans.values())
    assert all(s.token_logprobs for s in spans.values())


def test_a_string_is_scored_on_its_own_tokens_not_its_quotes():
    """The tokens around a value's quotes are the JSON's (``": "``, ``",``):
    near-certain under any reading, they say nothing about the value."""
    text = '{"policy_number": "HO-778812", "year": "2015"}'
    tokens = ['{"', 'policy', '_number', '": "', 'HO', '-77', '8812', '", "', 'year', '": "', '2015', '"}']
    logprobs = [0.0, 0.0, 0.0, -0.001, -0.5, -0.4, -0.3, -0.001, 0.0, -0.001, -0.2, -0.001]
    spans = span_map.map_field_spans(text, tokens, logprobs)
    assert spans["policy_number"].token_logprobs == [-0.5, -0.4, -0.3]
    assert spans["year"].token_logprobs == [-0.2]
    empty = span_map.map_field_spans('{"a": ""}', ['{"', 'a', '": ', '""', '}'], [0.0, 0.0, 0.0, -0.7, 0.0])
    assert empty["a"].token_logprobs == [-0.7]                 # an empty string keeps its quotes


def test_row_count_supports_the_completeness_signal():
    """A missing row emits no tokens, so counting generated rows is the only way
    to compare against the document's own count (IMPL-09)."""
    text = '{"claims":[{"n":"a"},{"n":"b"},{"n":"c"}]}'
    tokens, logprobs = _tokenize(text)
    spans = span_map.map_field_spans(text, tokens, logprobs)
    assert span_map.row_count(spans, "claims") == 3


def test_misaligned_logprobs_are_refused():
    """A misaligned pair silently attributes one field's probability to another."""
    text = '{"a":"b"}'
    tokens, _ = _tokenize(text)
    with pytest.raises(span_map.SpanMapError, match="must align"):
        span_map.map_field_spans(text, tokens, [-0.1])


def test_unmappable_fields_are_reported_not_dropped():
    """An unmapped field has no confidence. Dropping it would make it
    indistinguishable from a clean, confident extraction."""
    text = '{"a":"value"}'
    tokens = ["{", '"a"', ":", '"different"', "}"]   # do not reconstruct the text
    logprobs = [-0.1] * len(tokens)
    spans = span_map.map_field_spans(text, tokens, logprobs)

    assert not spans["a"].mapped
    unmapped = span_map.unmapped_fields(spans)
    assert unmapped and unmapped[0][0] == "a"
    assert "reconstruct" in unmapped[0][1]


def test_strict_mode_raises_on_token_misalignment():
    text = '{"a":"value"}'
    with pytest.raises(span_map.SpanMapError, match="cannot be trusted"):
        span_map.map_field_spans(text, ["nope"], [-0.1], strict=True)


def test_escaped_strings_and_unicode_are_handled():
    text = '{"name":"Rivera \\"Fab\\" LLC","note":"caf\\u00e9"}'
    tokens, logprobs = _tokenize(text)
    spans = span_map.map_field_spans(text, tokens, logprobs)
    assert spans["name"].value == 'Rivera "Fab" LLC'
    assert spans["note"].value == "café"


def test_trailing_content_after_json_is_rejected():
    """The model is prompted for JSON only, no fences. Trailing text means the
    output is not what the schema gate will accept either."""
    with pytest.raises(span_map.SpanMapError, match="trailing content"):
        span_map.scan_json_spans('{"a":1} some explanation')


# --------------------------------------------------------------------------
# Runner config and generation
# --------------------------------------------------------------------------

def test_runner_config_is_greedy_and_logprob_enabled():
    config = load_runner_config()
    assert config.is_greedy, "sampling breaks reproducibility and the calibration fitted on it"
    assert config.logprobs


def test_generation_fingerprint_changes_with_settings():
    a = RunnerConfig(temperature=0.0)
    b = RunnerConfig(temperature=0.7)
    assert a.fingerprint() != b.fingerprint()


def test_missing_logprobs_fail_loudly():
    """Returning an extraction with no confidence would look like a clean result
    and could not be risk-routed."""
    with pytest.raises(ModelRunnerError, match="no usable logprobs"):
        Generation(text="{}", tokens=["{", "}"], token_logprobs=[]).assert_logprobs()


def test_truncation_is_detectable():
    """On a long Loss Run, truncation drops claim rows — a recall failure
    per-field confidence cannot see."""
    assert Generation(text="{", finish_reason="length").truncated()
    assert not Generation(text="{}", finish_reason="stop").truncated()


def test_load_base_needs_no_registry_entry(client):
    """`base` is the shared path for the pilot baseline, day-zero pre-annotation,
    and `extract --model base`."""
    model = load_model("base", client, backend_impl=EchoBackend())
    assert model.is_base
    assert model.resolved["foundation_adapter"] is None


def test_generate_returns_text_with_aligned_logprobs(client):
    response = '{"insured_name":"Rivera Fabrication LLC","line_of_business":["workers_comp"]}'
    model = load_model("base", client, backend_impl=EchoBackend(response))
    built = build_messages("policy", PAGES, OCR, "ocr_plus_image")

    result = model_runner.generate(model, built.messages)
    assert result.text == response
    assert result.has_logprobs
    assert result.latency_ms is not None

    spans = span_map.map_field_spans(result.text, result.tokens, result.token_logprobs)
    assert spans["insured_name"].value == "Rivera Fabrication LLC"
    assert spans["insured_name"].mapped


def test_foundation_only_generation_is_a_valid_path(client):
    """`adapter=None` is the classifier's low-confidence fallback (arch §4a),
    not an error."""
    backend = EchoBackend()
    model = load_model("base", client, backend_impl=backend)
    built = build_messages("policy", PAGES, OCR, "ocr_plus_image")
    model_runner.generate(model, built.messages, adapter=None)
    assert backend.calls[0]["adapter"] is None


def test_backend_without_logprobs_is_refused(client):
    class NoLogprobs(EchoBackend):
        def supports_logprobs(self) -> bool:
            return False

    with pytest.raises(ModelRunnerError, match="cannot return logprobs"):
        load_model("base", client, backend_impl=NoLogprobs())


# --------------------------------------------------------------------------
# Layering
# --------------------------------------------------------------------------

def test_inference_core_does_not_import_higher_layers():
    """IMPL-07 sits *below* calibration, serving, evaluation and testing.

    They import it; it must never import them, or the cycle it exists to break
    reappears.
    """
    import inspect

    forbidden = ("calibration", "serving", "evaluation", "testing")
    for module in (model_runner, span_map):
        source = inspect.getsource(module)
        for name in forbidden:
            assert f"import {name}" not in source, f"{module.__name__} imports {name}"
            assert f"from {name}" not in source, f"{module.__name__} imports from {name}"


# --------------------------------------------------------------------------
# Page pairing — the structure problems 1-3 were about
# --------------------------------------------------------------------------

def test_each_image_is_followed_by_its_own_page_text():
    """Positional pairing. Concatenating the pages into one block left the model
    to work out which text belonged to which image by content matching — feasible
    on a 3-page ACORD, not on a 40-page policy."""
    pages = [f"page_{i}.png" for i in range(1, 4)]
    texts = [f"body of page {i}" for i in range(1, 4)]
    content = build_messages("policy", pages, texts, "ocr_plus_image").messages[1]["content"]

    assert [b["type"] for b in content] == ["image", "text", "image", "text", "image", "text"]
    for i in range(3):
        assert content[2 * i]["image"] == pages[i]
        assert texts[i] in content[2 * i + 1]["text"]


def test_every_page_text_carries_its_page_marker():
    texts = ["a", "b", "c"]
    content = build_messages("policy", ["p1.png", "p2.png", "p3.png"], texts,
                             "ocr_plus_image").messages[1]["content"]
    markers = [b["text"].splitlines()[0] for b in content if b["type"] == "text"]
    assert markers == ["<page 1 of 3>", "<page 2 of 3>", "<page 3 of 3>"]


def test_a_routed_subset_keeps_its_true_page_numbers():
    """`<page 9 of 20>` tells the model it is holding a fragment, so the fields
    that are missing are missing because they live on pages it was not shown."""
    content = build_messages("policy", ["page_9.png", "page_14.png"], ["nine", "fourteen"],
                             "ocr_plus_image", page_numbers=[9, 14],
                             total_pages=20).messages[1]["content"]
    markers = [b["text"].splitlines()[0] for b in content if b["type"] == "text"]
    assert markers == ["<page 9 of 20>", "<page 14 of 20>"]


def test_a_routed_request_is_a_subsequence_of_the_full_document():
    """The shape long policies are served in must be the shape they trained on —
    fewer pairs, not a different structure. Nothing in the suite compared user
    turns before, which is why the old one-call-per-page reading went
    unnoticed. A window is that shape with fewer pairs."""
    pages = [f"page_{i}.png" for i in range(1, 5)]
    texts = [f"t{i}" for i in range(1, 5)]

    full = build_messages("policy", pages, texts, "ocr_plus_image").messages[1]["content"]
    routed = build_messages("policy", [pages[0], pages[2]], [texts[0], texts[2]],
                            "ocr_plus_image", page_numbers=[1, 3],
                            total_pages=4).messages[1]["content"]

    assert [b["type"] for b in routed] == ["image", "text", "image", "text"]
    assert routed[0] == full[0] and routed[1] == full[1]      # page 1's pair, unchanged
    assert routed[2] == full[4] and routed[3] == full[5]      # page 3's pair, unchanged


def test_a_mismatched_page_count_is_refused():
    """A short or long text list would shift every page's text onto the wrong
    image, and every value after the shift would be attributed to the wrong page."""
    with pytest.raises(InputBuilderError, match="one text per image"):
        build_messages("policy", ["p1.png", "p2.png"], ["only one"], "ocr_plus_image")


def test_a_table_crossing_a_page_boundary_is_not_joined():
    """A blank line ends a Markdown table, so joining page 9's text to page 10's
    split every continued table in two and left the second half headerless."""
    page_9 = "| CLM-1044 | closed | 4200 |\n| CLM-1045 | open | 0 |"
    page_10 = "| CLM-1046 | open | 1750 |"
    content = build_messages("lossrun", ["page_9.png", "page_10.png"], [page_9, page_10],
                             "ocr_plus_image").messages[1]["content"]

    texts = [b["text"] for b in content if b["type"] == "text"]
    assert len(texts) == 2, "the two pages must stay in separate blocks"
    assert page_9 in texts[0] and page_10 in texts[1]
    assert page_9 + "\n\n" + page_10 not in "".join(texts)


def test_image_only_sends_markers_but_never_ocr():
    """image_only removes the OCR body, not page identity. Without the marker a
    an image_only window could not tell the model it is holding pages 9 and
    14 of 20 rather than a two-page document."""
    content = build_messages("policy", ["p9.png", "p14.png"], None, "image_only",
                             page_numbers=[9, 14], total_pages=20).messages[1]["content"]

    assert [b["type"] for b in content] == ["image", "text", "image", "text"]
    assert [b["text"] for b in content if b["type"] == "text"] == [
        "<page 9 of 20>", "<page 14 of 20>"
    ]


# --------------------------------------------------------------------------
# The alignment invariant the whole confidence signal rests on
# --------------------------------------------------------------------------

def test_tokens_that_reconstruct_the_text_are_accepted():
    from inference_core.model_runner import assert_tokens_reconstruct

    assert_tokens_reconstruct('{"a":1}', ['{"a', '":1', "}"], "test")   # does not raise


def test_tokens_that_do_not_reconstruct_the_text_are_refused():
    """The failure this guards is invisible everywhere else: offsets after the
    divergence shift, so each field reports its neighbour's confidence while the
    numbers stay in range and the schema still validates."""
    from inference_core.model_runner import ModelRunnerError, assert_tokens_reconstruct

    with pytest.raises(ModelRunnerError, match="do not reconstruct"):
        assert_tokens_reconstruct('{"insured_name":"Rivera"}', ['{"insured_name"', ':"Rivera"'], "test")


def test_a_dropped_special_token_is_caught():
    """`skip_special_tokens=True` on the token list but not the text is the
    likeliest way to break this, and it shortens the rebuild by exactly the
    characters the special token occupied."""
    from inference_core.model_runner import ModelRunnerError, assert_tokens_reconstruct

    with pytest.raises(ModelRunnerError, match="first difference at"):
        assert_tokens_reconstruct("hello world", ["hello", "world"], "test")


def test_the_error_names_where_the_divergence_starts():
    """"They differ" is not actionable on a 4,000-character generation."""
    from inference_core.model_runner import ModelRunnerError, assert_tokens_reconstruct

    text = "x" * 200 + "DIVERGES HERE"
    with pytest.raises(ModelRunnerError) as excinfo:
        assert_tokens_reconstruct(text, ["x" * 200 + "DIFFERENT"], "test")

    message = str(excinfo.value)
    assert "first difference at 202" in message      # where V and F diverge
    assert "DIVERGES" in message and "DIFFERENT" in message


def test_the_echo_backend_satisfies_the_same_invariant():
    """The stub must obey the contract the real backends are held to, or tests
    pass against a shape production never produces."""
    from inference_core.model_runner import assert_tokens_reconstruct

    generation = EchoBackend('{"insured_name":"Rivera Fabrication LLC"}').generate(
        [], load_runner_config("vllm")
    )
    assert_tokens_reconstruct(generation.text, generation.tokens, "EchoBackend")
    assert generation.has_logprobs


def test_the_hf_backend_refuses_a_per_request_adapter_swap():
    """Transformers binds adapters at load time. Silently ignoring a mismatch
    would serve one document type's weights under another's routing decision."""
    from inference_core.model_runner import HFBackend, ModelRunnerError
    from registry_utils.query_registry import ResolvedModel

    backend = HFBackend(
        ResolvedModel(tag="v2", type_adapter="adapters/policy/v2", base_model="base"),
        load_runner_config("hf"),
    )
    with pytest.raises(ModelRunnerError, match="binds adapters at load time"):
        backend.generate([], load_runner_config("hf"), adapter="adapters/lossrun/v2")


def test_a_version_with_no_servable_weights_is_refused():
    """vLLM serves a merged model or the base; a bare adapter stack has to be
    merged first."""
    from inference_core.model_runner import ModelRunnerError, VLLMBackend
    from registry_utils.query_registry import ResolvedModel

    backend = VLLMBackend(ResolvedModel(tag="v2"), load_runner_config("vllm"))
    with pytest.raises((ModelRunnerError, ImportError)):
        backend.generate([], load_runner_config("vllm"))


def _vllm_output(text, pieces, token_ids=None):
    from types import SimpleNamespace as NS

    token_ids = token_ids or list(range(len(pieces)))
    steps = [{tid: NS(decoded_token=piece, logprob=-0.1 * i, rank=1)}
             for i, (tid, piece) in enumerate(zip(token_ids, pieces, strict=True))]
    completion = NS(text=text, token_ids=token_ids, logprobs=steps, finish_reason="stop")
    return NS(outputs=[completion])


def test_the_end_of_turn_token_vllm_returns_outside_the_text_is_dropped():
    """vLLM lists <|im_end|> among the tokens but not in the text; every
    completed validation answer failed reconstruction and scored as empty."""
    from inference_core.model_runner import VLLMBackend

    gen = VLLMBackend._to_generation(
        _vllm_output('{"a":1}', ['{"a', '":1', "}", "<|im_end|>"]), _runner_config(), 1.0)
    assert gen.tokens == ['{"a', '":1', "}"] and len(gen.token_logprobs) == 3


def test_anything_else_that_breaks_reconstruction_is_still_refused():
    from inference_core.model_runner import ModelRunnerError, VLLMBackend

    with pytest.raises(ModelRunnerError, match="do not reconstruct"):
        VLLMBackend._to_generation(
            _vllm_output('{"a":1}', ['{"a', '":2', "}", "<|im_end|>"]), _runner_config(), 1.0)
    with pytest.raises(ModelRunnerError, match="do not reconstruct"):
        VLLMBackend._to_generation(
            _vllm_output('{"a":1}', ['{"a', "}", "<|im_end|>"]), _runner_config(), 1.0)


def _runner_config():
    from inference_core.runner_config import RunnerConfig

    return RunnerConfig()


def test_a_fields_confidence_reads_the_printed_value_not_its_reformatting():
    """``parsed`` is written after ``raw`` and largely decided by it: its tokens
    are near-certain whether the reading was right or not."""
    from common.canonical import collapse_spans

    text = '{"policy": {"effective_date": {"raw": "6/4/25", "parsed": "06/04/2025", "page_ref": [1]}}}'
    spans = span_map.map_field_spans(text, list(text), [-0.3 if text[i] in "6/425" and i < 60 else -0.001
                                                         for i in range(len(text))])
    collapsed = collapse_spans(spans)
    assert collapsed["policy.effective_date"] is spans["policy.effective_date.raw"]
    no_raw = collapse_spans({"a.parsed": spans["policy.effective_date.parsed"],
                             "a.raw": span_map.FieldSpan("a.raw", None, 0, 0)})
    assert no_raw["a"] is spans["policy.effective_date.parsed"]      # nothing read: parsed stands in
