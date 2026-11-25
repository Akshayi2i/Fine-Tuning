"""Review Phase 5 (2026-11-25): P6 and P4 in the standard template, the classifier (P8), and one
setting for a schema-invalid answer (R7)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.prompts import render_classifier_prompt, render_system_prompt

# --------------------------------------------------------------------------
# P6: "null" in the standard template, said once both ways on a canonical document
# --------------------------------------------------------------------------


@pytest.mark.format_profile("standard")
def test_a_canonical_documents_printed_but_unparsed_value_keeps_raw() -> None:
    text = render_system_prompt("policy", "ocr_plus_image")
    assert "A value the document prints that cannot be put in its format keeps `raw`, with `parsed` null." in text
    assert "a component is missing, `parsed` is null" in text
    assert "If the document supports none of them, `parsed` is null" in text
    assert "a component is missing, output null" not in text


@pytest.mark.format_profile("standard")
@pytest.mark.parametrize("doc_type,form", [("lossrun", None), ("acord", "25")])
def test_a_flat_document_still_outputs_null(doc_type: str, form: str | None) -> None:
    """ACORD forms and Loss Runs keep every key, null for absence: "output null" is right there."""
    text = render_system_prompt(doc_type, "ocr_plus_image", form)
    assert "a component is missing, output null" in text and "`parsed` is null" not in text


# --------------------------------------------------------------------------
# P4 in the standard template: the image-only page sentence
# --------------------------------------------------------------------------


@pytest.mark.format_profile("standard")
@pytest.mark.parametrize("doc_type,form", [("policy", None), ("lossrun", None), ("acord", "125")])
def test_the_standard_image_only_prompt_never_promises_page_text(doc_type: str, form: str | None) -> None:
    image_only = render_system_prompt(doc_type, "image_only", form)
    assert "no OCR text" in image_only and "own text" not in image_only
    assert "followed by its page marker, `<page N of M>`" in image_only
    assert "own text" in render_system_prompt(doc_type, "ocr_plus_image", form)


# --------------------------------------------------------------------------
# P8: the classifier
# --------------------------------------------------------------------------


def test_the_classifier_prompt_says_it_reads_the_opening_pages() -> None:
    for kwargs in ({}, {"line_only": True}):
        text = render_classifier_prompt(**kwargs)
        assert "whole document" not in text
        assert "every page you are given - the document's opening pages" in text


def _generation(text: str, chunk: int = 3, logprob: float = -0.05) -> SimpleNamespace:
    tokens = [text[i:i + chunk] for i in range(0, len(text), chunk)]
    return SimpleNamespace(text=text, tokens=tokens, token_logprobs=[logprob] * len(tokens))


def test_the_document_types_confidence_is_measured_not_stated() -> None:
    """The model wrote 0.99; its tokens say otherwise. Routing acts on the tokens."""
    from serving.doc_type_classifier import ZeroShotClassifier

    answer = '{"doc_type": "policy", "acord_form": null, "lob": null, "confidence": 0.99}'
    generation = _generation(answer, logprob=-0.5)
    classifier = ZeroShotClassifier(SimpleNamespace(is_base=True), lambda *a, **k: generation)
    result = classifier.classify(["p1.png"], None)
    assert result.stated_confidence == 0.99
    assert 0.0 < result.confidence < 0.5                     # exp of the "policy" tokens' summed logprobs
    no_logprobs = SimpleNamespace(text=answer, tokens=[], token_logprobs=[])
    unmeasured = ZeroShotClassifier(SimpleNamespace(is_base=True), lambda *a, **k: no_logprobs).classify(["p.png"], None)
    assert unmeasured.confidence == 0.0 and unmeasured.stated_confidence == 0.99


def test_an_acord_form_with_no_number_stays_unusable() -> None:
    from serving.doc_type_classifier import ZeroShotClassifier

    generation = _generation('{"doc_type": "acord", "acord_form": null, "confidence": 0.9}', logprob=-0.01)
    result = ZeroShotClassifier(SimpleNamespace(is_base=True), lambda *a, **k: generation).classify(["p.png"], None)
    assert result.confidence == 0.0


def test_the_largest_classifier_request_fits_its_cap() -> None:
    """Measured 2026-11-25: with 4,000 characters of text, ~4,236 estimated tokens against
    classify's 4,096 (configs/shared/sequence.yaml). The text is cut to what fits."""
    from common.config import sequence_for_task
    from common.lob import known_lines
    from contracts.token_estimates import (
        TEMPLATE_OVERHEAD_TOKENS,
        estimate_text_tokens,
        estimate_visual_tokens,
    )
    from serving.doc_type_classifier import CLASSIFIER_PAGES, classifier_messages

    cap = sequence_for_task("classify")
    page_text = ("Policy No. 12-3456789 Effective 04/01/2026 Limit $1,000,000 " * 200)    # digits cost most
    for lines, line_only in ((None, False), (sorted(known_lines()), True)):
        system, user = classifier_messages(["a.png", "b.png"], page_text, lines=lines, line_only=line_only)
        text = user["content"][-1]["text"]
        total = (estimate_text_tokens(system["content"]) + estimate_visual_tokens(CLASSIFIER_PAGES, "classify")
                 + estimate_text_tokens(text) + TEMPLATE_OVERHEAD_TOKENS + int(cap["max_output_tokens"]))
        assert total <= int(cap["max_seq_len"]), (lines is not None, total)


# --------------------------------------------------------------------------
# R7: one setting for a schema-invalid answer (decision 2026-11-25: served, flagged)
# --------------------------------------------------------------------------


def test_the_worker_and_the_endpoint_read_one_setting() -> None:
    from serving.read_settings import ReadSettings
    from serving.vllm_entrypoint import serving_thresholds

    tuning = serving_thresholds()
    assert tuning["strict_schema"] is False                 # serve_flagged, the shipped value
    worker = ReadSettings.served(tuning)
    endpoint = ReadSettings.endpoint(adapter_map=None, plan=None, release_runtimes=None, base_model_loaded=True,
                                     tuning=tuning)
    assert worker.strict_schema is endpoint.strict_schema is False


def test_refuse_refuses_in_both_and_an_unknown_value_is_refused() -> None:
    from common.config import ConfigError
    from serving.read_settings import ReadSettings
    from serving.vllm_entrypoint import invalid_answers_refused

    assert invalid_answers_refused({"schema": {"invalid_answers": "refuse"}}) is True
    assert ReadSettings.served({"strict_schema": True}).strict_schema is True
    with pytest.raises(ConfigError, match="invalid_answers"):
        invalid_answers_refused({"schema": {"invalid_answers": "maybe"}})
