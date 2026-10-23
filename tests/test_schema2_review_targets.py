"""Review findings on the common-model training targets and the model view.

A window target is what the model is taught to write, and the model view is
the grammar it is held to when it writes. The two must agree on every gold the
client's schema accepts: a reference the slice leaves out is left out of the
target too; the view encodes no client rule a window cannot satisfy; a code
field takes the codes its value can really hold; a label is one part of its own
line; and a fragment of a row is taught only in a window that shows what
identifies it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from common import schemas as S
from common.canonical import CanonicalLabelError
from common.schema_sections import groups_for
from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
HOME = json.loads((S.CANONICAL_ROOT / "common schema" / "examples" / "homeowners_minimal.json")
                  .read_text(encoding="utf-8"))
AUTO = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))
AUTO_PAGES = [1, 2, 3, 4, 5, 6]
LINES = (
    "homeowners", "personal_auto", "dwelling_fire", "ocean_marine",
    "motorcycle", "recreational_vehicle", "personal_umbrella",
)


def _fv(raw, parsed, pages):
    return {"raw": raw, "parsed": parsed, "confidence": {"score": 1.0, "source": "deterministic"},
            "page_ref": pages, "flagged": False}


def _null():
    return {"raw": None, "parsed": None, "confidence": {"score": 1.0, "source": "deterministic"},
            "page_ref": [], "flagged": False}


def _client_valid(gold, lob="personal_auto"):
    assert list(S.iter_validation_errors(gold, "policy", None, lob)) == []


def _targets(gold, lob="personal_auto", pages=AUTO_PAGES):
    report = TargetReport()
    out = [(plan, window_target(gold, lob, plan, report)) for plan in plan_windows(lob, pages, 1)]
    return out, report


def _assert_writable(targets, lob="personal_auto"):
    for plan, target in targets:
        view = S.resolved_schema("policy", None, lob, plan.group)
        errors = [e.message for e in Draft202012Validator(view).iter_errors(target)]
        assert not errors, (plan.group, plan.pages, errors[:3])


def _coverage(gold, coverage_id):
    (row,) = [c for c in gold["coverages"] if c["coverage_id"] == coverage_id]
    return row


# T1 - a reference the slice leaves out is left out of the target -----------


@pytest.mark.parametrize("lob", LINES)
def test_the_slice_and_the_target_leave_out_the_same_references(lob):
    """One set for both sides: every reference field the target builder drops
    for a group is missing from that group's slice, and every one it keeps is
    still there - a field dropped on one side only is a target the slice cannot
    hold, or a slot the model is never taught to fill."""
    from common.schema_sections import references

    full = S.load_schema("policy", None, lob)["$defs"]
    for group in groups_for(lob):
        dropped = S.cross_group_references(lob, group)
        sliced = S.load_schema("policy", None, lob, group)["$defs"]
        for name, definition in sliced.items():
            shown = set(definition.get("properties") or {})
            assert not dropped & shown, (group, name, sorted(dropped & shown))
            whole = set(full[name].get("properties") or {})
            assert (whole & set(references(lob))) - dropped <= shown, (group, name)


def test_a_premium_item_for_one_vehicle_is_taught_without_its_vehicle():
    """The declarations window cannot see the vehicles, so its slice has no
    applies_to on a premium item. The value was carried into the target anyway,
    and writing the target raised on a gold the client's schema accepts."""
    gold = copy.deepcopy(AUTO)
    gold["premium"]["items"] = [{"description": _fv("Collision", "Collision", [1]),
                                 "amount": _fv("$380.00", 380.0, [1]),
                                 "coverage_code": "X_COLLISION", "applies_to": ["veh_1"]}]
    _client_valid(gold)
    targets, report = _targets(gold)
    _assert_writable(targets)
    # The declarations' first window (pages 1-5) holds the premium item.
    decl = next(t for p, t in targets if p.group == "decl")
    (item,) = decl["premium"]["items"]
    assert "applies_to" not in item and item["coverage_code"] == "X_COLLISION"
    assert "decl:premium.items[0].applies_to=veh_1" in report.dangling


def test_an_endorsement_on_one_vehicle_is_taught_without_its_vehicle():
    gold = copy.deepcopy(AUTO)
    gold["forms_and_endorsements"][0]["applies_to"] = ["veh_2"]
    _client_valid(gold)
    targets, report = _targets(gold)
    _assert_writable(targets)
    forms = [f for p, t in targets if p.group == "forms" for f in t.get("forms_and_endorsements") or []]
    assert forms and all("applies_to" not in f for f in forms)
    assert "forms:forms_and_endorsements[0].applies_to=veh_2" in report.dangling


# T2 - the view encodes no rule a window cannot satisfy ---------------------


def _view_def(lob, name):
    return S.resolved_schema("policy", None, lob)["$defs"][name]


@pytest.mark.parametrize("lob", LINES)
def test_a_deductible_is_a_plain_closed_object_and_a_limit_splits_only_on_its_percentage(lob):
    defs = S.resolved_schema("policy", None, lob)["$defs"]
    if "Deductible" in defs:
        deductible = defs["Deductible"]
        assert deductible["additionalProperties"] is False and "properties" in deductible
        assert not {"anyOf", "allOf", "if", "then", "else"} & set(deductible)
        assert "enum" in S.resolved_schema("policy", None, lob)["$defs"]["DeductibleType"]
    if "Limit" in defs:
        variants = defs["Limit"]["anyOf"]
        assert [sorted(v["required"]) for v in variants] == [
            ["basis_coverage_code", "limit_type", "percentage"], ["limit_type"]]
        assert "percentage" not in variants[1]["properties"]
        assert all("const" not in v["properties"]["limit_type"] for v in variants)


def test_a_flat_deductible_whose_amount_is_on_another_page_is_still_taught(no_window_overlap):
    """The amount on page 3, the peril on page 4: the window over pages 4-6
    holds a flat deductible with no amount. The view required one, so the target
    was a row the decoder could never write."""
    gold = copy.deepcopy(AUTO)
    deductible = _coverage(gold, "cov_2")["deductibles"][0]
    deductible["amount"]["page_ref"] = [3]
    deductible["peril"] = _fv("All perils", "All perils", [4])
    _client_valid(gold)
    targets, _ = _targets(gold)
    _assert_writable(targets)
    rows = [d for p, t in targets if p.pages == (4, 5, 6) and p.group == "arrays"
            for c in t.get("coverages") or [] for d in c.get("deductibles") or []]
    assert {"deductible_type": "flat"}.items() <= rows[0].items() and "amount" not in rows[0]


def test_a_flat_deductible_with_no_amount_stated_is_still_taught():
    gold = copy.deepcopy(AUTO)
    deductible = _coverage(gold, "cov_2")["deductibles"][0]
    deductible["amount"] = _null()
    deductible["peril"] = _fv("All perils", "All perils", [4])
    _client_valid(gold)
    targets, _ = _targets(gold)
    _assert_writable(targets)


def test_a_sublimit_with_no_description_stated_is_still_taught():
    gold = copy.deepcopy(AUTO)
    _coverage(gold, "cov_1")["limits"].append(
        {"limit_type": "sublimit", "amount": _fv("$1,000", 1000, [4]), "description": _null()})
    _client_valid(gold)
    targets, _ = _targets(gold)
    _assert_writable(targets)
    limits = [lim for p, t in targets for c in t.get("coverages") or [] for lim in c.get("limits") or []]
    assert any(lim["limit_type"] == "sublimit" and "description" not in lim for lim in limits)


@pytest.mark.parametrize("gold,lob,pages", [(HOME, "homeowners", [1, 2, 3]),
                                            (AUTO, "personal_auto", AUTO_PAGES)])
def test_every_window_target_of_the_fixtures_is_one_its_decoder_can_write(gold, lob, pages):
    targets, _ = _targets(gold, lob, pages)
    assert targets
    _assert_writable(targets, lob)


def test_a_target_its_decoder_cannot_write_is_refused():
    """A code the client's pattern accepts but the line does not list: the
    grammar would force another code in its place, so the target is never
    taught - the build names the path instead."""
    gold = copy.deepcopy(AUTO)
    _coverage(gold, "cov_2")["coverage_code"] = "PA_NOT_A_CODE"
    _client_valid(gold)
    with pytest.raises(CanonicalLabelError, match=r"coverages\[1\]\.coverage_code"):
        _targets(gold)


# T3 - an underlying policy's code is another line's -----------------------


def test_an_umbrellas_underlying_policy_takes_another_lines_coverage_code():
    """An umbrella sits over an auto or home policy: its underlying coverage is
    coded X_VEHICLE_LIABILITY, which the umbrella's own list cannot name."""
    view = S.resolved_schema("policy", None, "personal_umbrella")
    schema = {"$defs": view["$defs"], "$ref": "#/$defs/UnderlyingPolicy"}
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"coverage_code": "X_VEHICLE_LIABILITY"})
    assert view["$defs"]["UnderlyingPolicy"]["properties"]["coverage_code"]["type"] == "string"
    # A limit's basis still names one of the line's own coverages.
    limit = view["$defs"]["Limit"]["anyOf"][0]["properties"]["basis_coverage_code"]
    assert limit["$ref"] == "#/$defs/CoverageCode"


# T4 - a label is one part, of its own line ---------------------------------


def test_a_two_part_label_under_one_common_model_line_is_refused():
    gold = copy.deepcopy(AUTO)
    gold["lob_parts"].append({"part_id": "part_2", "lob": "personal_umbrella",
                              "title": _fv("Umbrella", "Umbrella", [1])})
    with pytest.raises(CanonicalLabelError, match="two-part or foreign-line label"):
        _targets(gold)


def test_a_part_of_another_line_is_refused():
    gold = copy.deepcopy(AUTO)
    gold["lob_parts"][0]["lob"] = "homeowners"
    with pytest.raises(CanonicalLabelError, match="holds one part, of personal_auto itself"):
        _targets(gold)


def test_a_part_of_a_line_read_as_this_one_is_written_as_this_line():
    """classic_auto is read as personal_auto (common.lob.merge_line): the part
    is taught as the line the schema holds it to."""
    gold = copy.deepcopy(AUTO)
    gold["lob_parts"][0]["lob"] = "classic_auto"
    targets, _ = _targets(gold)
    _assert_writable(targets)
    decl = next(t for p, t in targets if p.group == "decl")
    assert [part["lob"] for part in decl["lob_parts"]] == ["personal_auto"]
    assert AUTO["lob_parts"][0]["lob"] == "personal_auto"


# T5 - only a printed value identifies a row -------------------------------


def test_a_coverage_fragment_without_its_name_is_an_orphan(no_window_overlap):
    """Collision's premium also printed on page 2, and the liability's first
    limit on page 3: the window over pages 1-3 shows neither coverage's name.
    The coverage code rides with every fragment and the limits have no printed
    identifier, so the fragments were taught as nameless coverages."""
    gold = copy.deepcopy(AUTO)
    _coverage(gold, "cov_2")["premium"]["page_ref"] = [4, 2]
    _coverage(gold, "cov_1")["limits"][0]["amount"]["page_ref"] = [4, 3]
    targets, report = _targets(gold)
    _assert_writable(targets)
    (first,) = [t for p, t in targets if p.group == "arrays" and p.pages == (1, 2, 3)]
    assert "coverages" not in first
    assert {"arrays:coverages[0]", "arrays:coverages[1]"} <= set(report.orphaned)
    (second,) = [t for p, t in targets if p.group == "arrays" and p.pages == (4, 5, 6)]
    assert [c["coverage_code"] for c in second["coverages"]][:2] == ["X_VEHICLE_LIABILITY", "X_COLLISION"]


def test_a_coverage_is_identified_by_its_printed_name():
    from data_pipeline.dataset_builder.policy_windows import _row_identifiers

    assert _row_identifiers(AUTO["coverages"], "coverages", lob="personal_auto",
                            common_model=True) == ["coverage_name"]
    assert _row_identifiers(AUTO["vehicles"], "vehicles", lob="personal_auto",
                            common_model=True)[0] == "vin"
