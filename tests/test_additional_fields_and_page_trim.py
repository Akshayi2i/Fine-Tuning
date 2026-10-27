"""Window targets: page references limited to the window, additional_fields placed
by their own pages, text_sections never taught (Fideon SPEC_09 amendment item 1,
SPEC_21; handoff item 1)."""

from __future__ import annotations

import copy

import pytest

from data_pipeline.dataset_builder import policy_windows
from data_pipeline.dataset_builder.policy_windows import (
    PolicyWindowPlan,
    TargetReport,
    _additional_within,
    _within,
    window_target,
)


def _fv(value, pages):
    return {"raw": value, "parsed": value, "page_ref": pages}


def _plan(pages, *, single=False, group="decl"):
    return PolicyWindowPlan(group=group, window_index=0, pages=tuple(pages), single=single)


#: A self-contained line: LABEL's shape, and the path that keeps additional_fields
#: entries as written (a common-model line, such as general liability or, since
#: property.json 2.0.0, property, envelopes their values; tests/test_schema2_*.py
#: pin that).
LINE = "equipment_breakdown"

LABEL = {
    "carrier": {"company_name": _fv("Granite Mutual", [1, 2, 9, 30])},
    "named_insured": {"primary_name": _fv("Rivera", [12])},
}


def test_a_value_cites_only_the_windows_pages_and_the_stored_label_is_untouched():
    label = copy.deepcopy(LABEL)
    kept = _within(label["carrier"], "carrier", {1, 2, 3}, _plan([1, 2, 3]), TargetReport())
    assert kept["company_name"]["page_ref"] == [1, 2]
    assert label["carrier"]["company_name"]["page_ref"] == [1, 2, 9, 30]


def test_a_field_whose_pages_are_outside_the_window_is_absent():
    assert _within(LABEL["named_insured"], "named_insured", {1, 2, 3}, _plan([1, 2, 3]), None) == {}


def test_every_policy_row_cites_only_its_own_windows_pages():
    import json

    from data_pipeline.dataset_builder.build_jsonl import expand_document
    from tests.test_policy_windows import LOB, _document

    rows, _ = expand_document(_document(lob=LOB), "train", modes=("ocr_plus_image",))
    assert rows
    for row in rows:
        target = json.loads(row["messages"][-1]["content"][0]["text"]
                            if isinstance(row["messages"][-1]["content"], list)
                            else row["messages"][-1]["content"])
        cited = _page_refs(target)
        assert cited <= set(row["window_pages"]), (row["window_pages"], cited)


def _page_refs(node):
    if isinstance(node, dict):
        found = {int(p) for p in node.get("page_ref") or []} if "page_ref" in node else set()
        for key, value in node.items():
            if key != "page_ref":
                found |= _page_refs(value)
        return found
    if isinstance(node, list):
        return set().union(*(_page_refs(item) for item in node)) if node else set()
    return set()


# --------------------------------------------------------------------------
# additional_fields
# --------------------------------------------------------------------------

ADDITIONAL = [
    {"label": "Roof age", "value": "12 years", "section_hint": "dwelling", "page_ref": [2, 14]},
    {"label": "Pool on premises", "value": "Yes", "section_hint": "dwelling", "page_ref": [9]},
    {"label": "Blank", "value": None, "page_ref": [1]},
    {"label": "No page", "value": "x"},
]


def test_additional_fields_are_placed_by_their_own_pages_and_trimmed():
    kept = _additional_within(ADDITIONAL, "additional_fields", {1, 2, 3}, _plan([1, 2, 3]), TargetReport())
    assert kept == [{"label": "Roof age", "value": "12 years", "section_hint": "dwelling", "page_ref": [2]}]
    assert ADDITIONAL[0]["page_ref"] == [2, 14]                       # the label itself is untouched
    single = _additional_within(ADDITIONAL, "additional_fields", {1, 2, 3}, _plan([1, 2, 3], single=True), None)
    assert [e["label"] for e in single] == ["Roof age", "No page"]   # no page: only where one window reads all


@pytest.fixture
def schema_with_both(monkeypatch):
    """A line schema that defines additional_fields and text_sections, as SPEC_21's do."""
    monkeypatch.setattr("common.canonical.schema_label", lambda label, *a, **k: label)
    monkeypatch.setattr("common.schema_sections.sections_for",
                        lambda group, lob=None: ["carrier", "additional_fields", "text_sections"])
    monkeypatch.setattr("common.schemas.required_fields", lambda *a, **k: [])
    monkeypatch.setattr("common.schemas.resolved_schema", lambda *a, **k: {"properties": {}})


def test_a_gold_with_additional_fields_keeps_them_in_the_target(schema_with_both):
    label = {**LABEL, "additional_fields": ADDITIONAL}
    target = window_target(label, LINE, _plan([1, 2, 3]))
    assert [e["label"] for e in target["additional_fields"]] == ["Roof age"]
    assert target["additional_fields"][0]["page_ref"] == [2]


def test_no_target_contains_text_sections(schema_with_both):
    label = {**LABEL, "text_sections": [{"heading": "Conditions", "text": "...", "page_ref": [1]}]}
    assert "text_sections" not in window_target(label, LINE, _plan([1, 2, 3]))
    assert "text_sections" in policy_windows.EXCLUDED_FROM_TARGETS


def test_an_additional_field_whose_envelope_states_nothing_is_not_taught():
    empty = [{"label": "Roof age", "value": {"raw": None, "parsed": None, "page_ref": [2]}, "page_ref": [2]}]
    assert _additional_within(empty, "additional_fields", {1, 2, 3}, _plan([1, 2, 3]), None,
                              common_model=True) == []
