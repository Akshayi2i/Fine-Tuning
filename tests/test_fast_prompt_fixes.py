"""The fast-layout prompt says one thing at a time (review P1, P3, P4; prompt version of 2026-11-24).

Each test reads every window request a line sends (every group, both modes), as
``common.prompts`` renders it for the model.
"""

from __future__ import annotations

import pytest

from common.model_view import PRINTED_STRING_FIELDS
from common.schema_sections import groups_for
from testing.render_prompts import REFERENCE_MODES, REFERENCE_WINDOW_LINES, render

WINDOWS = [(lob, group, mode) for lob in REFERENCE_WINDOW_LINES for group in groups_for(lob) for mode in REFERENCE_MODES]

READ_ELSEWHERE = "are read by other requests"
EXCEPTION = "is still written in that entry's own field"


def _request(lob: str, group: str, mode: str) -> str:
    return render("policy", mode, None, lob, group)


@pytest.mark.parametrize("lob,group,mode", WINDOWS)
def test_a_request_that_sends_values_elsewhere_keeps_its_printed_fields(lob, group, mode) -> None:
    """P1: "a value that belongs under another part is read there: leave it out
    here" contradicted a coverage's form_refs, which asks for the form numbers
    printed with it. Where both are in one request, the exception is said, naming the fields."""
    text = _request(lob, group, mode)
    printed = [field for fields in PRINTED_STRING_FIELDS.values() for field in fields if f"  {field}: " in text]
    if READ_ELSEWHERE in text and printed:
        assert EXCEPTION in text
        for field in printed:
            assert f"own field, though it is also read elsewhere: `{field}`" in text or f"`{field}`" in text


def test_the_exception_is_in_the_schedules_request_and_nowhere_else() -> None:
    with_it = {(lob, group) for lob, group, mode in WINDOWS if EXCEPTION in _request(lob, group, mode)}
    assert with_it == {(lob, "arrays") for lob in REFERENCE_WINDOW_LINES}


@pytest.mark.parametrize("lob,group,mode", WINDOWS)
def test_the_field_list_names_the_answer_keys_the_answer_uses(lob, group, mode) -> None:
    """P3: the answer writes `r`, `v`, `p`; the field list said `raw` and `parsed`.
    `page_ref` stays where the compact answer keeps it: an additional field's entry."""
    text = _request(lob, group, mode)
    assert "`raw`" not in text and "`parsed`" not in text and " in parsed" not in text
    for line in text.splitlines():
        if "page_ref" in line:
            assert line.strip().startswith("page_ref:"), line


def test_an_additional_field_entry_keeps_page_ref_in_the_compact_answer() -> None:
    from common.canonical import to_compact_answer

    entry = {"label": {"raw": "Agent", "parsed": "Agent", "page_ref": [1]},
             "value": {"raw": "A-1", "parsed": "A-1", "page_ref": [1]}, "page_ref": [1]}
    compact = to_compact_answer({"additional_fields": [entry]})["additional_fields"][0]
    assert compact["page_ref"] == [1] and compact["value"] == {"v": "A-1", "p": [1]}


@pytest.mark.parametrize("lob,group", [(lob, group) for lob in REFERENCE_WINDOW_LINES for group in groups_for(lob)])
def test_the_image_only_request_never_says_a_page_comes_with_its_text(lob, group) -> None:
    """P4: it said "no OCR text", then "each page ... followed by that page's own text"."""
    image_only = _request(lob, group, "image_only")
    with_text = _request(lob, group, "ocr_plus_image")
    assert "no OCR text" in image_only
    assert "own text" not in image_only and "followed by its page marker, `<page N of M>`" in image_only
    assert "own text" in with_text
    for text in (image_only, with_text):
        assert "Where the numbers skip, you are seeing selected pages" in text
