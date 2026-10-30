"""Serving must send the model what training showed it.

Each test pins one place the two sides diverged: the schema a policy was read
against, which image sat beside which text, what a blank page said, how pages
were resized, which fields were dates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.canonical import with_output_dates
from common.normalize import infer_field_kind
from inference_core.input_builder import EMPTY_PAGE_TEXT, build_messages
from serving.pipeline import ExtractionRequest, PipelineError, _image_for
from serving.vllm_entrypoint import ServingError, build_request, request_lob

LOB_SCHEMAS = Path(__file__).resolve().parent.parent / "configs" / "canonical schema" / "LOB Schema"


def _payload(**over):
    # Every request carries MinerU's text of every page (extraction never skips MinerU).
    return {"source_id": "s1", "image_paths": ["a/page_1.png", "a/page_2.png"],
            "page_texts": {"1": "Declarations", "2": "Schedule"}, **over}


# --------------------------------------------------------------------------
# Line of business
# --------------------------------------------------------------------------


def test_the_line_of_business_reaches_the_request():
    assert build_request(_payload(lob="homeowners")).known_lob == "homeowners"
    assert build_request(_payload(line_of_business="workers_comp")).known_lob == "workers_comp"


def test_a_package_policy_may_name_several_lines():
    assert request_lob({"lob": ["general_liability", "commercial_property"]}) == [
        "general_liability", "commercial_property",
    ]


def test_a_line_with_no_schema_is_refused_not_quietly_defaulted():
    with pytest.raises(ServingError, match="no canonical policy schema"):
        build_request(_payload(lob="underwater_basket_weaving"))


def test_a_malformed_line_is_refused():
    with pytest.raises(ServingError):
        request_lob({"lob": []})
    with pytest.raises(ServingError):
        request_lob({"lob": 7})


# --------------------------------------------------------------------------
# Pages and images
# --------------------------------------------------------------------------


def test_page_texts_must_number_one_per_image():
    with pytest.raises(ServingError, match="exactly pages 1..2"):
        build_request(_payload(page_texts={"1": "a", "3": "b"}))
    request = build_request(_payload(page_texts={"2": "b", "1": "a"}))
    assert sorted(request.page_texts) == [1, 2]


def test_an_image_only_page_finds_its_own_image_by_position():
    request = ExtractionRequest(
        source_id="s", image_paths=["x/scan-a.png", "x/scan-b.png", "x/scan-c.png"],
        modality_mode="image_only",
    )
    assert _image_for(request, 3) == "x/scan-c.png"


def test_the_page_convention_matches_the_file_name_not_a_substring():
    request = ExtractionRequest(source_id="s", image_paths=[
        f"x/page_{n}.png" for n in (10, 1, 11, 2)
    ])
    assert _image_for(request, 1) == "x/page_1.png"
    assert _image_for(request, 2) == "x/page_2.png"


def test_a_page_beyond_the_images_is_an_error():
    request = ExtractionRequest(source_id="s", image_paths=["x/a.png"])
    with pytest.raises(PipelineError, match="page 2"):
        _image_for(request, 2)


# --------------------------------------------------------------------------
# One prompt for a blank page
# --------------------------------------------------------------------------


def test_a_blank_page_gets_the_placeholder_in_training_and_serving():
    built = build_messages(
        "lossrun", ["p1.png", "p2.png"], ["text", "   "], "ocr_plus_image",
        check_resolution_parity=False,
    )
    texts = [b["text"] for b in built.messages[1]["content"] if b["type"] == "text"]
    assert texts[1].endswith(EMPTY_PAGE_TEXT)
    assert not texts[0].endswith(EMPTY_PAGE_TEXT)


# --------------------------------------------------------------------------
# One pixel budget
# --------------------------------------------------------------------------


def test_training_and_serving_resize_to_the_same_budget():
    from common.adapter import ADAPTER
    from common.config import pixel_budget
    from training.train import _pixel_budget

    env = _pixel_budget()
    assert (int(env["MIN_PIXELS"]), int(env["MAX_PIXELS"])) == pixel_budget(ADAPTER.tasks)
    assert pixel_budget(ADAPTER.tasks) == pixel_budget()


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------


def _leaf_paths(node, path=""):
    if isinstance(node, dict):
        props = node.get("properties")
        if props:
            for key, sub in props.items():
                yield from _leaf_paths(sub, f"{path}.{key}" if path else key)
        if "items" in node:
            yield from _leaf_paths(node["items"], f"{path}[0]")
        for key in ("anyOf", "oneOf", "allOf"):
            for sub in node.get(key, []):
                yield from _leaf_paths(sub, path)
        if "$ref" in node and not props:
            yield path


def _schema_files():
    return sorted(LOB_SCHEMAS.glob("*.json"))


def test_the_lob_schemas_are_present():
    assert len(_schema_files()) >= 30


@pytest.mark.parametrize("schema_file", _schema_files(), ids=lambda p: p.stem)
def test_every_date_field_in_every_schema_is_formatted(schema_file):
    """MM/DD/YYYY is a guarantee only if every date field is recognised as one."""
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    for path in _leaf_paths(schema):
        words = path.rsplit(".", 1)[-1].replace("[0]", "").casefold().split("_")
        if {"date", "dates", "dated"} & set(words):
            assert infer_field_kind(path) == "date", path


def test_not_every_word_containing_date_is_a_date():
    assert infer_field_kind("policy.update_reason") != "date"
    assert infer_field_kind("candidate") != "date"


def test_an_element_of_a_list_of_dates_is_formatted():
    doc = {"report_due_dates": [{"raw": "Jan 5, 2026", "parsed": "2026-01-05", "page_ref": [1]}]}
    assert with_output_dates(doc)["report_due_dates"][0]["parsed"] == "01/05/2026"


# --------------------------------------------------------------------------
# One page threshold, and a prompt hash that sees every prompt input
# --------------------------------------------------------------------------


def test_serving_takes_no_page_threshold_of_its_own():
    """The window plan is the corpus build's; a serving-only knob moved it."""
    from serving.vllm_entrypoint import serving_thresholds

    assert "page_threshold" not in serving_thresholds()


def test_the_prompt_hash_covers_doc_type_templates_schemas_and_sections():
    from common.prompts import prompt_input_files

    names = {p.as_posix() for p in prompt_input_files()}
    assert any(n.endswith("prompts/doc_types/policy.jinja") for n in names)
    assert any("LOB Schema" in n for n in names)
    assert any(n.endswith("schema_sections.yaml") for n in names)


def test_a_release_packaged_on_other_prompts_is_refused_at_cold_start():
    from common.prompts import prompt_hash
    from serving.release_router import ServedRelease, ServingPlan
    from serving.vllm_entrypoint import ColdStartError, assert_prompt_hash

    def plan(h):
        return ServingPlan(ServedRelease("v3", "base@abc1234", "merged", doc_types=("policy",),
                                         prompt_hash=h))

    assert_prompt_hash(plan(prompt_hash()))
    assert_prompt_hash(plan(""))
    with pytest.raises(ColdStartError, match="prompt hash"):
        assert_prompt_hash(plan("0" * 64))
