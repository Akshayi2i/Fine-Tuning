"""scripts/check_answers_fit_decoder.py: a training answer production's decoder could never write is found (D4)."""

from __future__ import annotations

import json

from data_pipeline.dataset_builder.policy_windows import plan_windows, window_target
from inference_core.input_builder import build_training_row
from scripts.check_answers_fit_decoder import check_rows

LABEL = {"policy": {"policy_number": {"raw": "GL-1001", "parsed": "GL-1001", "page_ref": [3]}}}


def _decl() -> tuple[dict, list[int]]:
    """The first declarations window of a 10-page gl policy and its target, as the corpus build cuts it."""
    plan = next(p for p in plan_windows("gl", list(range(1, 11)), None, None) if p.group == "decl")
    return window_target(LABEL, "gl", plan), list(plan.pages)


def _row(answer: dict, pages: list[int]) -> dict:
    return build_training_row("policy", "policy_0001", [f"p{n}.png" for n in pages], None, "image_only",
                              json.dumps(answer), lob="gl", sections="decl", split="train",
                              page_numbers=pages, total_pages=10)


def test_an_answer_the_decoder_can_write_fits():
    target, pages = _decl()
    row = _row(target, pages)
    assert row.get("answer_format")                          # compact, as production decodes it
    assert check_rows([row]) == {"checked": 1, "misfit": 0, "reasons": {}, "rows": []}


def test_an_answer_citing_a_page_the_window_does_not_show_is_found():
    """Production holds page references to the window's own pages (with_page_bounds): page 9 cannot be written."""
    target, pages = _decl()
    target["policy"]["policy_number"]["page_ref"] = [9]
    report = check_rows([_row(target, pages)])
    assert report["misfit"] == 1 and "9" in report["rows"][0]["first"]


def test_an_answer_with_a_key_outside_the_slice_is_found():
    target, pages = _decl()
    report = check_rows([_row({**target, "made_up_section": {"x": 1}}, pages)])
    assert report["misfit"] == 1 and "made_up_section" in report["rows"][0]["first"]
