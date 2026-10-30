"""Extraction never skips MinerU (decision 2026-10-30): an endpoint request carries
MinerU's text of every page - in the prompt (page_texts, ocr_plus_image) or held back
to check an image-only answer (printed_texts) - and a job whose MinerU text fails for
good is not a finished one (tests/test_azure_service.py)."""

from __future__ import annotations

import pytest

from serving.vllm_entrypoint import ServingError, build_request

PAGES = ["p1.png", "p2.png"]


def payload(**more):
    return {"source_id": "d1", "image_paths": PAGES, "lob": "gl", **more}


def test_a_request_with_no_mineru_text_is_refused():
    for mode in ("image_only", "ocr_plus_image"):
        with pytest.raises(ServingError, match="never skips MinerU"):
            build_request(payload(modality_mode=mode))
    with pytest.raises(ServingError, match="never skips MinerU"):
        build_request(payload(modality_mode="image_only", printed_texts={}))        # empty text is no text


def test_a_request_may_carry_the_text_in_the_prompt():
    request = build_request(payload(modality_mode="ocr_plus_image", page_texts={"1": "Declarations", "2": "Schedule"}))
    assert request.page_texts == {1: "Declarations", 2: "Schedule"} and request.printed_texts is None
    assert build_request(payload(modality_mode="ocr_plus_image", ocr_text="all pages joined")).ocr_text


def test_an_image_only_request_carries_its_text_to_check_the_answer_against():
    request = build_request(payload(modality_mode="image_only", printed_texts={"1": "Declarations", "2": "Schedule"}))
    assert request.modality_mode == "image_only" and request.page_texts == {}
    assert request.printed_texts == {1: "Declarations", 2: "Schedule"}


@pytest.mark.parametrize(("body", "message"), [
    ({"modality_mode": "image_only", "printed_texts": {"1": "only page one"}}, "printed_texts covers pages"),
    ({"modality_mode": "image_only", "printed_texts": {"one": "x", "2": "y"}}, "printed_texts keys"),
    ({"modality_mode": "ocr_plus_image", "page_texts": {"1": "a", "2": "b"}, "printed_texts": {"1": "a", "2": "b"}},
     "carries its text as page_texts"),
    ({"modality_mode": "image_only", "ocr_text": "text shown to an image-only prompt"}, "image_only was requested"),
])
def test_text_that_does_not_match_the_request_is_refused(body, message):
    with pytest.raises(ServingError, match=message):
        build_request(payload(**body))
