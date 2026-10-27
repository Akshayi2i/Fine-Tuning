"""Training targets in the key order the decoding grammar writes (common.canonical.in_schema_order).

xgrammar builds each object's grammar from its schema properties in order, so a
target that put form_title before form_number taught an order the grammar
refuses: every form came back without its number on the smoke run, and 58% of
the delivered labels' windows carried at least one such object.
"""

from __future__ import annotations

from common.canonical import in_schema_order, training_target
from common.schemas import resolved_schema


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


# A self-contained line, whose forms carry form_title.
LINE = "equipment_breakdown"

SCHEMA = {
    "type": "object",
    "$defs": {"Row": {"type": "object", "properties": {"form_number": {}, "edition_date": {},
                                                       "form_title": {}}}},
    "properties": {
        "carrier": {"type": "object", "properties": {"company_name": {}, "naic": {}}},
        "forms": {"type": "array", "items": {"$ref": "#/$defs/Row"}},
    },
}


def test_keys_follow_the_schema_at_every_depth():
    label = {"forms": [{"form_title": "Homeowners 3", "form_number": "HO 00 03"}],
             "carrier": {"naic": "123", "company_name": "Granite"}}
    ordered = in_schema_order(label, SCHEMA)
    assert list(ordered) == ["carrier", "forms"]
    assert list(ordered["carrier"]) == ["company_name", "naic"]
    assert list(ordered["forms"][0]) == ["form_number", "form_title"]


def test_values_are_untouched_and_unknown_keys_keep_their_place_last():
    label = {"extra": 1, "carrier": {"naic": "123"}}
    ordered = in_schema_order(label, SCHEMA)
    assert ordered == label and list(ordered) == ["carrier", "extra"]


def test_a_canonical_training_target_is_in_schema_order():
    label = {
        "forms_and_endorsements": [{"form_title": _env("Homeowners 3"), "form_number": _env("HO 00 03")}],
        "policy": {"policy_number": _env("HO-1")},
    }
    target = training_target(label, "policy", None, LINE)
    schema_order = list(resolved_schema("policy", None, LINE)["properties"])
    assert list(target) == sorted(target, key=schema_order.index)
    assert list(target["forms_and_endorsements"][0]) == ["form_number", "form_title"]
    assert list(target["forms_and_endorsements"][0]["form_number"]) == ["raw", "parsed", "page_ref"]


def test_a_window_target_is_in_its_slices_order():
    from data_pipeline.dataset_builder.policy_windows import plan_windows, window_target

    label = {"forms_and_endorsements": [
        {"form_title": _env("Homeowners 3", 2), "form_number": _env("HO 00 03", 2)}]}
    for plan in plan_windows(LINE, [1, 2, 3]):
        target = window_target(label, LINE, plan)
        for row in target.get("forms_and_endorsements", []):
            assert list(row) == ["form_number", "form_title"]
