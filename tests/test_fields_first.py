"""serving/fields_first.py: the fields held to the pages they cite before they are served."""

from __future__ import annotations

import pytest

from serving.fields_first import PENDING_FLAG, _pending, cited_pages


def test_the_pages_the_fields_cite_are_every_page_ref_within_the_document():
    extraction = {
        "policy": {"policy_number": {"raw": "PA-1", "page_ref": [1]}, "insured": None},
        "vehicles": [{"vin": {"raw": "1HGCM", "page_ref": [3, 4]}}, {"year": {"raw": "2019", "page_ref": [3]}}],
        "additional_fields": [{"label": "Agent", "raw": "J. Smith", "page_ref": [9, 120]}],
        "forms_and_endorsements": [{"form_number": {"raw": "PP 00 01", "page_ref": "7"}}],   # not a list: ignored
    }
    assert cited_pages(extraction, page_count=10) == [1, 3, 4, 9]
    assert cited_pages({}, page_count=10) == []


@pytest.mark.parametrize(("flag", "becomes"), [
    ("policy.policy_number:not_printed_removed", f"policy.policy_number:{PENDING_FLAG}"),
    ("vehicles[0].vin:not_printed", f"vehicles[0].vin:{PENDING_FLAG}"),
    ("policy.policy_number:low_confidence", "policy.policy_number:low_confidence"),
    ("text_sections:no_page_text", "text_sections:no_page_text"),
])
def test_not_found_on_the_cited_pages_is_pending_not_printed_nowhere(flag, becomes):
    assert _pending(flag) == becomes


def test_an_image_only_read_is_looked_over_for_missed_fields_when_its_text_comes(monkeypatch):
    import serving.pipeline as pipeline
    from serving.pipeline import attach_full_text
    from tests.test_full_text import _compact_auto, _serve_replay

    result, _b, _a = _serve_replay(_compact_auto(), 3, strict=False)
    result.mode, result.printed_check = "image_only", {}               # read from images; check aside
    seen = []
    monkeypatch.setattr(pipeline, "_likely_missed",
                        lambda extraction, texts, lob: seen.append(sorted(texts)) or ["policy.insured:likely_missed"])
    attach_full_text(result, {1: "Declarations", 2: "Schedule page 2", 3: "Schedule page 3"}, lob="personal_auto")
    assert seen == [[1, 2, 3]] and "policy.insured:likely_missed" in result.review_flags
    # A read that had its text all along was looked over by extract already.
    result.mode, seen[:] = "ocr_plus_image", []
    attach_full_text(result, {1: "Declarations"}, lob="personal_auto")
    assert seen == []
