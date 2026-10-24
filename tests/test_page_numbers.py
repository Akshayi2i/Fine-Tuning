"""A ``page_ref`` that names no page is read as no page, never a crash (QA 2026-10-24)."""

from __future__ import annotations

import pytest

from common.canonical import page_numbers


@pytest.mark.parametrize(("refs", "pages"), [
    ([1, 3], [1, 3]),
    (["12", " 4 "], [12, 4]),          # digits: the page they name, never [1, 2]
    (["p1", "1-3", 2.5, None], []),    # names no page
    ([True, 2], [2]),                  # a bool is not a page
    ("7", []),                         # not a list
    (None, []),
])
def test_page_numbers_are_whole_numbers_only(refs, pages):
    assert page_numbers(refs) == pages


def test_a_gold_value_citing_no_readable_page_does_not_stop_the_build():
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document

    value = {"raw": "Granite Mutual", "parsed": "Granite Mutual", "page_ref": ["p1"],
             "confidence": {"score": 1.0, "source": "audit"}, "flagged": False}
    number = {**value, "raw": "WC-1", "parsed": "WC-1", "page_ref": [1]}
    doc = SourceDocument(source_id="policy_0001", doc_type="policy",
                         golden_label={"carrier": {"company_name": value}, "policy": {"policy_number": number}},
                         ocr_pages=["Declarations Granite Mutual WC-1"],
                         image_paths=["processed/default/policy/policy_0001/page_1.png"],
                         lob="workers_comp", tenant_id="default")
    rows, _report = expand_document(doc, "train", modes=("ocr_plus_image",))
    assert rows                                                   # built, not refused with a ValueError
