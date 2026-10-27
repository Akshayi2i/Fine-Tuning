"""A common-model line's ids, merge and every-key fill: the cases review found wrong.

Each test is a document the branch served or scored wrongly - a link taught to
the wrong vehicle, a building named by its writer's location numbering, a
served JSON the client's schema rejects, a unit left as a phantom second row -
and what it should have been. The last tests pin the self-contained lines,
whose merge and fill none of this may change.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from common.canonical import values_view, with_all_keys, with_system_fields
from common.schemas import _sources, _strip_prefixed, is_common_model, iter_validation_errors, load_schema
from common.structural_ids import comparable_view, reference_pairs, renumber_structural_ids
from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target
from evaluation.metrics.common_model import CommonModelTally
from serving.policy_merge import PolicyWindow, merge_policy_windows

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
AUTO = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))
COMMON_MODEL_LINES = ("homeowners", "personal_auto", "dwelling_fire", "ocean_marine", "motorcycle",
                      "recreational_vehicle", "personal_umbrella")


def _v(raw, page=1, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": [page]}


def _merge(*answers, lob="personal_auto", group="arrays"):
    """One window per answer, on pages 1, 2, 3... in the order given."""
    windows = [PolicyWindow(group, [i + 1], copy.deepcopy(a)) for i, a in enumerate(answers)]
    return merge_policy_windows(windows, lob=lob)


def _envelopes(node):
    if isinstance(node, dict):
        if "raw" in node and "page_ref" in node:
            yield node
        for value in node.values():
            yield from _envelopes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _envelopes(value)


# --------------------------------------------------------------------------
# Renumbering a window's ids
# --------------------------------------------------------------------------


def test_a_reference_that_reads_like_a_new_id_is_still_dangling():
    """The only vehicle is the label's veh_2, so it becomes veh_1. A coverage on
    the label's veh_1 - a vehicle this document does not hold - must not be
    relinked to it because the old id now spells a new one."""
    doc = {"vehicles": [{"unit_id": "veh_2", "vin": _v("B")}],
           "coverages": [{"coverage_code": "X_COLLISION", "applies_to": ["veh_1"]},
                         {"coverage_code": "X_UM", "applies_to": ["veh_2"]}]}
    out, report = renumber_structural_ids(doc, "personal_auto")
    assert "applies_to" not in out["coverages"][0]
    assert out["coverages"][1]["applies_to"] == ["veh_1"]
    assert report.dangling == ["coverages[0].applies_to=veh_1"]


def test_an_alias_to_an_id_already_renumbered_still_resolves():
    """What the fallback is for: extra_index naming a row by its new id."""
    doc = {"vehicles": [{"unit_id": "veh_2", "vin": _v("B")}],
           "coverages": [{"coverage_code": "X_UM", "applies_to": ["w2/veh_9"]}]}
    out, report = renumber_structural_ids(doc, "personal_auto", extra_index={"w2/veh_9": "veh_1"})
    assert out["coverages"][0]["applies_to"] == ["veh_1"] and not report.dangling


@pytest.mark.format_profile("standard")
def test_a_window_showing_only_the_second_vehicle_teaches_no_link_to_the_first(no_window_overlap):
    """Vehicle 1 on page 2, vehicle 2 on page 5, the coverages of both on pages
    4-5. The window over pages 4-6 shows only vehicle 2: the first vehicle's
    coverages lose their link (counted), not gain vehicle 2's."""
    gold = copy.deepcopy(AUTO)
    for envelope in _envelopes(gold["vehicles"][1]):
        envelope["page_ref"] = [5]
    report = TargetReport()
    (target,) = [window_target(gold, "personal_auto", plan, report)
                 for plan in plan_windows("personal_auto", [1, 2, 3, 4, 5, 6], 1)
                 if plan.group == "arrays" and tuple(plan.pages) == (4, 5, 6)]
    assert [values_view(v["vin"]) for v in target["vehicles"]] == [values_view(AUTO["vehicles"][1]["vin"])]
    links = [(values_view(c["premium"]), c.get("applies_to")) for c in target["coverages"]]
    assert links == [(410.0, None), (380.0, None), (455.0, ["veh_1"]), (420.0, ["veh_1"])]
    assert {"arrays:coverages[0].applies_to=veh_1", "arrays:coverages[1].applies_to=veh_1"} <= set(report.dangling)


# --------------------------------------------------------------------------
# Scoring links by the units' own keys
# --------------------------------------------------------------------------


def _homeowners(location_ids):
    """Two locations, building 1 at each, a Coverage A on each building; the
    locations numbered with ``location_ids`` (location 1's id, location 2's)."""
    first, second = location_ids
    return {
        "locations": [{"unit_id": first, "location_number": _v("1", parsed=1)},
                      {"unit_id": second, "location_number": _v("2", parsed=2)}],
        "buildings": [{"unit_id": "bldg_1", "location_ref": first, "building_number": _v("1", parsed=1)},
                      {"unit_id": "bldg_2", "location_ref": second, "building_number": _v("1", parsed=1)}],
        "coverages": [{"coverage_code": "HO_COV_A", "coverage_name": _v("Dwelling"), "applies_to": ["bldg_1"],
                       "premium": _v("$100", parsed=100)},
                      {"coverage_code": "HO_COV_A", "coverage_name": _v("Dwelling"), "applies_to": ["bldg_2"],
                       "premium": _v("$200", parsed=200)}],
    }


@pytest.mark.parametrize("lob", ["homeowners", "dwelling_fire"])
def test_a_building_is_named_by_its_locations_key_not_its_writers_id(lob):
    """The answer numbers the two locations the other way round. Its buildings,
    their locations and the coverages on them are all right, so nothing about
    the comparison may change."""
    gold, answer = _homeowners(("loc_1", "loc_2")), _homeowners(("loc_2", "loc_1"))
    assert comparable_view(answer, lob) == comparable_view(gold, lob)
    assert Counter(reference_pairs(answer, lob)) == Counter(reference_pairs(gold, lob))
    tally = CommonModelTally()
    tally.add(comparable_view(gold, lob), comparable_view(answer, lob), lob)
    assert tally.metrics()["reference_accuracy"] == 1.0


def test_one_missed_location_does_not_cost_the_links_to_the_next():
    """The answer missed location 1, so its location 2 is loc_1: the link from
    the coverage to the building at location 2 is still the same link."""
    gold = {"locations": [{"unit_id": "loc_1", "location_number": _v("1", parsed=1)},
                          {"unit_id": "loc_2", "location_number": _v("2", parsed=2)}],
            "buildings": [{"unit_id": "bldg_1", "location_ref": "loc_2", "building_number": _v("1", parsed=1)}],
            "coverages": [{"coverage_code": "HO_COV_A", "coverage_name": _v("Dwelling"), "applies_to": ["bldg_1"]}]}
    answer = {"locations": [{"unit_id": "loc_1", "location_number": _v("2", parsed=2)}],
              "buildings": [{"unit_id": "bldg_1", "location_ref": "loc_1", "building_number": _v("1", parsed=1)}],
              "coverages": [{"coverage_code": "HO_COV_A", "coverage_name": _v("Dwelling"),
                             "applies_to": ["bldg_1"]}]}
    tally = CommonModelTally()
    tally.add(comparable_view(gold, "homeowners"), comparable_view(answer, "homeowners"), "homeowners")
    assert tally.metrics()["reference_accuracy"] == 1.0
    (applies_to,) = {tuple(c["applies_to"]) for c in comparable_view(answer, "homeowners")["coverages"]}
    assert applies_to == ("buildings:location_ref=locations:location_number=2,building_number=1",)


def test_a_wrong_code_costs_the_code_not_the_link():
    """A coverage with its printed name, its premium and its vehicle right, and
    the wrong code: the code score drops, the link score does not."""
    answer = copy.deepcopy(AUTO)
    answer["coverages"][1]["coverage_code"] = "X_COMPREHENSIVE"
    tally = CommonModelTally()
    tally.add(comparable_view(AUTO, "personal_auto"), comparable_view(answer, "personal_auto"), "personal_auto")
    metrics = tally.metrics()
    assert metrics["reference_accuracy"] == 1.0 and metrics["coverage_code_accuracy"] < 1.0


def test_a_coverage_with_no_printed_name_is_still_known_by_its_code():
    def doc(code):
        return {"vehicles": [{"unit_id": "veh_1", "vin": _v("A")}],
                "coverages": [{"coverage_code": code, "applies_to": ["veh_1"]}]}

    assert reference_pairs(doc("X_UM"), "personal_auto") != reference_pairs(doc("X_COLLISION"), "personal_auto")


# --------------------------------------------------------------------------
# The every-key fill
# --------------------------------------------------------------------------


def _envelope(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page],
            "confidence": {"score": 0.9, "source": "vlm"}, "flagged": False}


def _served(lob):
    """A served answer as the pipeline holds it before the fill: system fields
    supplied, a coverage with a limit row, a form row."""
    answer = {"policy": {"policy_number": _envelope("P-1")},
              "coverages": [{"coverage_id": "cov_1", "coverage_code": "X_OTHER", "coverage_name": _envelope("Other"),
                             "limits": [{"limit_type": "per_occurrence", "amount": _envelope(100000)}]}],
              "forms_and_endorsements": [{"form_number": _envelope("PP 00 01")}]}
    return with_system_fields(answer, page_count=2, source_file_name="x.pdf", lob=lob, modality="native_pdf")


@pytest.mark.parametrize("lob", COMMON_MODEL_LINES)
def test_the_filled_answer_is_valid_against_the_lines_full_schema(lob):
    """The fill used to add an empty `percentage` to every limit (which then
    requires a `basis_coverage_code` it cannot write) and `page_range: []` to
    every form (which must hold two pages): every served answer was invalid."""
    filled = with_all_keys(_served(lob), _strip_prefixed(load_schema("policy", None, lob), ("fideon:",)))
    assert list(iter_validation_errors(filled, "policy", None, lob)) == []
    (limit,) = filled["coverages"][0]["limits"]
    assert "percentage" not in limit and limit["amount"]["parsed"] == 100000
    assert "page_range" not in filled["forms_and_endorsements"][0]
    assert filled["forms_and_endorsements"][0]["title"]["raw"] is None      # the rest is still filled


def test_a_value_the_model_wrote_is_never_taken_out_by_the_fill():
    """A percentage the model read stays, even without its basis: the fill
    judges only what it adds."""
    served = _served("personal_auto")
    served["coverages"][0]["limits"][0]["percentage"] = _envelope(10)
    schema = _strip_prefixed(load_schema("policy", None, "personal_auto"), ("fideon:",))
    assert with_all_keys(served, schema)["coverages"][0]["limits"][0]["percentage"]["parsed"] == 10


def test_a_dependency_in_the_older_keyword_is_honoured_too():
    schema = {"type": "object", "properties": {
        "rate": {"type": ["number", "null"]}, "basis": {"type": "string"}},
        "dependencies": {"rate": ["basis"]}}
    assert with_all_keys({}, schema) == {}
    assert with_all_keys({"basis": "x"}, schema) == {"rate": None, "basis": "x"}


def test_no_other_schema_has_what_the_fill_now_leaves_out():
    """Why the self-contained lines, ACORD and loss runs fill exactly as before:
    none of their schemas has a table that must hold rows (a page reference
    inside an envelope aside - the fill never reaches it) or a key that requires
    another. The fallback composes the common model since 1.1.0, and is filled
    as the common-model lines are."""
    def found(node, path=""):
        if isinstance(node, dict):
            if (node.get("minItems") or 0) > 0 and not path.endswith("/page_ref"):
                yield f"{path}: minItems"
            for keyword in ("dependentRequired", "dependencies"):
                if keyword in node:
                    yield f"{path}: {keyword}"
            for key, value in node.items():
                yield from found(value, f"{path}/{key}")
        elif isinstance(node, list):
            for value in node:
                yield from found(value, path)

    checked = 0
    for key in _sources():
        if key.startswith("acord:"):
            schema = load_schema("acord", key.split(":", 1)[1], None)
        elif key.startswith("policy"):
            lob = key.split(":", 1)[1] if ":" in key else None
            if is_common_model("policy", None, lob):
                continue
            schema = load_schema("policy", None, lob)
        else:
            schema = load_schema(key, None, None)
        assert list(found(schema)) == [], key
        checked += 1
    # 9 self-contained lines (statutory_disability reads with health since the owner's
    # map routed it there), ACORD 25/125/140, loss runs
    assert checked >= 13


# --------------------------------------------------------------------------
# Merging units
# --------------------------------------------------------------------------


_VIN_ONLY = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-A")}]}
_NUMBER_ONLY = {"vehicles": [{"unit_id": "veh_1", "vehicle_number": _v("1", parsed=1)}],
                "coverages": [{"coverage_code": "X_COLLISION", "coverage_name": _v("Collision"),
                               "applies_to": ["veh_1"]}]}
_BOTH = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-A"), "vehicle_number": _v("1", parsed=1)}]}


@pytest.mark.parametrize("order", [(_VIN_ONLY, _NUMBER_ONLY, _BOTH), (_NUMBER_ONLY, _VIN_ONLY, _BOTH)],
                         ids=["vin-number-both", "number-vin-both"])
def test_a_row_stating_both_keys_joins_the_two_halves_read_before_it(order):
    """A VIN-only row and a number-only row cannot be compared; the window
    that prints both shows they are one vehicle, whichever came first."""
    merged = _merge(*order)
    vehicles = merged.extraction["vehicles"]
    assert [(v["unit_id"], v["vin"]["raw"], v["vehicle_number"]["parsed"]) for v in vehicles] == [
        ("veh_1", "VIN-A", 1)]
    assert merged.extraction["coverages"][0]["applies_to"] == ["veh_1"]
    assert merged.review_flags == []


def _dwelling(page, **values):
    return {"locations": [{"unit_id": "loc_1", "location_number": _v("1", page, 1)}],
            "buildings": [{"unit_id": "bldg_1", "location_ref": "loc_1", **values}],
            "coverages": [{"coverage_code": "HO_COV_A", "coverage_name": _v("Dwelling", page),
                           "applies_to": ["bldg_1"],
                           "limits": [{"limit_type": "per_occurrence", "amount": _v("$300,000", page, 300000)}]}]}


@pytest.mark.parametrize("second", [{"year_built": _v("1990", 2, 1990)}, {"roof_type": _v("Asphalt", 2)}],
                         ids=["same-values", "other-values"])
def test_a_dwelling_with_no_building_number_read_twice_is_one_building(second):
    merged = _merge(_dwelling(1, year_built=_v("1990", 1, 1990)), _dwelling(2, **second), lob="homeowners")
    (building,) = merged.extraction["buildings"]
    (coverage,) = merged.extraction["coverages"]
    assert building["unit_id"] == "bldg_1" and building["location_ref"] == "loc_1"
    assert coverage["applies_to"] == ["bldg_1"]
    assert "buildings:joined_without_identifier" in merged.review_flags


def test_two_unnumbered_buildings_one_window_wrote_stay_two():
    answer = {"locations": [{"unit_id": "loc_1", "location_number": _v("1", parsed=1)}],
              "buildings": [{"unit_id": "bldg_1", "location_ref": "loc_1", "year_built": _v("1990", parsed=1990)},
                            {"unit_id": "bldg_2", "location_ref": "loc_1", "roof_type": _v("Metal")}]}
    assert len(_merge(answer, lob="homeowners").extraction["buildings"]) == 2


def test_unnumbered_buildings_that_disagree_stay_apart():
    merged = _merge(_dwelling(1, year_built=_v("1990", 1, 1990)), _dwelling(2, year_built=_v("2004", 2, 2004)),
                    lob="homeowners")
    assert len(merged.extraction["buildings"]) == 2


def test_a_building_naming_no_location_joins_the_only_building():
    first = _dwelling(1, year_built=_v("1990", 1, 1990))
    second = {"buildings": [{"unit_id": "bldg_1", "roof_type": _v("Asphalt", 2)}]}
    merged = _merge(first, second, lob="homeowners")
    (building,) = merged.extraction["buildings"]
    assert building["roof_type"]["raw"] == "Asphalt" and building["location_ref"] == "loc_1"
    assert "buildings:joined_without_identifier" in merged.review_flags


def test_a_lienholder_and_a_discount_on_two_vehicles_stay_two_rows_each():
    answer = {
        "vehicles": [{"unit_id": "veh_1", "vin": _v("A")}, {"unit_id": "veh_2", "vin": _v("B")}],
        "interested_parties": [
            {"role": "lienholder", "name": _v("First Bank"), "loan_number": _v("L-1"), "applies_to": ["veh_1"]},
            {"role": "lienholder", "name": _v("First Bank"), "loan_number": _v("L-2"), "applies_to": ["veh_2"]}],
        "rating_modifiers": [
            {"modifier_type": "discount", "description": _v("Anti-theft"), "percent": _v("5%", parsed=5),
             "applies_to": ["veh_1"]},
            {"modifier_type": "discount", "description": _v("Anti-theft"), "percent": _v("10%", parsed=10),
             "applies_to": ["veh_2"]}],
    }
    merged = _merge(answer)
    parties, modifiers = merged.extraction["interested_parties"], merged.extraction["rating_modifiers"]
    assert [(p["loan_number"]["raw"], p["applies_to"]) for p in parties] == [("L-1", ["veh_1"]), ("L-2", ["veh_2"])]
    assert [(m["percent"]["parsed"], m["applies_to"]) for m in modifiers] == [(5, ["veh_1"]), (10, ["veh_2"])]
    assert not merged.conflicts


@pytest.mark.parametrize("title", ["Personal Auto Policy", "Auto Policy"])
def test_a_declarations_window_read_in_two_halves_is_one_part(title):
    first = {"lob_parts": [{"part_id": "part_1", "lob": "personal_auto", "title": _v("Personal Auto Policy")}],
             "coverages": [{"coverage_code": "X_UM", "coverage_name": _v("UM"), "part": "part_1"}]}
    second = {"lob_parts": [{"part_id": "part_1", "lob": "personal_auto", "title": _v(title, 2)}],
              "coverages": [{"coverage_code": "X_COLLISION", "coverage_name": _v("Collision", 2), "part": "part_1"}]}
    merged = _merge(first, second, group="decl")
    served = with_system_fields(merged.extraction, page_count=2, lob="personal_auto")
    assert [(p["part_id"], p["title"]["raw"]) for p in served["lob_parts"]] == [("part_1", "Personal Auto Policy")]
    assert served["policy"]["is_package"] is False
    assert [c["part"] for c in served["coverages"]] == ["part_1", "part_1"]
    assert ("lob_parts.title:merge_conflict" in merged.review_flags) == (title != "Personal Auto Policy")


_TWINS = {"vehicles": [
    {"unit_id": "veh_1", "vin": _v("A"), "year": _v("2020", parsed=2020), "make": _v("Honda"), "model": _v("Civic")},
    {"unit_id": "veh_2", "vin": _v("B"), "year": _v("2020", parsed=2020), "make": _v("Honda"), "model": _v("Civic")}]}


def test_a_fragment_matching_two_vehicles_is_kept_apart_and_flagged():
    """Two 2020 Honda Civics; a later page shows a 2020 Honda Civic's mileage.
    Which one is not for the merge to guess."""
    fragment = {"vehicles": [{"unit_id": "veh_1", "year": _v("2020", 2, 2020), "make": _v("Honda", 2),
                              "model": _v("Civic", 2), "annual_mileage": _v("9000", 2, 9000)}]}
    merged = _merge(_TWINS, fragment)
    vehicles = merged.extraction["vehicles"]
    assert len(vehicles) == 3 and "annual_mileage" not in vehicles[0] and "annual_mileage" not in vehicles[1]
    assert "vehicles:ambiguous_unit" in merged.review_flags


def test_a_keyless_fragment_beside_two_vehicles_is_flagged():
    fragment = {"vehicles": [{"unit_id": "veh_1", "annual_mileage": _v("9000", 2, 9000)}]}
    merged = _merge(_TWINS, fragment)
    assert len(merged.extraction["vehicles"]) == 3
    assert "vehicles:unplaced_fragment" in merged.review_flags


def test_a_fragment_matching_one_vehicle_still_joins_it():
    fragment = {"vehicles": [{"unit_id": "veh_1", "vin": _v("B", 2), "annual_mileage": _v("9000", 2, 9000)}]}
    merged = _merge(_TWINS, fragment)
    assert [v.get("annual_mileage", {}).get("parsed") for v in merged.extraction["vehicles"]] == [None, 9000]
    assert merged.review_flags == []


# --------------------------------------------------------------------------
# The self-contained lines merge exactly as before
# --------------------------------------------------------------------------


_OLD_STYLE_WINDOWS = [
    ("decl", [1], {"policy": {"policy_number": _v("GL-1")},
                   "forms_and_endorsements": [{"form_number": _v("CG 00 01"), "edition_date": _v("04/13")}]}),
    ("lineblk", [2], {"auto": {"vehicles": [{"vin": _v("A", 2), "year": _v("2019", 2, 2019)},
                                            {"vin": _v("B", 2)}]},
                      "forms_and_endorsements": [{"form_number": _v("CG 00 01", 2), "edition_date": _v("04/13", 2)}]}),
    ("lineblk", [3], {"auto": {"vehicles": [{"vin": _v("A", 3), "make": _v("Ford", 3)},
                                            {"annual_mileage": _v("9000", 3, 9000)},
                                            {"vin": _v("B", 3), "year": _v("2021", 3, 2021)}]},
                      "policy": {"policy_number": _v("GL-2", 3)}}),
]

#: What the merge returned for these windows before the common-model fixes.
_OLD_STYLE_MERGED = {
    "policy": {"policy_number": {"raw": "GL-1", "parsed": "GL-1", "page_ref": [1]}},
    "forms_and_endorsements": [{"form_number": {"raw": "CG 00 01", "parsed": "CG 00 01", "page_ref": [1, 2]},
                                "edition_date": {"raw": "04/13", "parsed": "04/13", "page_ref": [1, 2]}}],
    "auto": {"vehicles": [
        {"vin": {"raw": "A", "parsed": "A", "page_ref": [2, 3]},
         "year": {"raw": "2019", "parsed": 2019, "page_ref": [2]},
         "make": {"raw": "Ford", "parsed": "Ford", "page_ref": [3]}},
        {"vin": {"raw": "B", "parsed": "B", "page_ref": [2, 3]},
         "year": {"raw": "2021", "parsed": 2021, "page_ref": [3]}},
        {"annual_mileage": {"raw": "9000", "parsed": 9000, "page_ref": [3]}}]},
}


@pytest.mark.parametrize("lob", [None, "gl", "commercial_auto"])
def test_a_self_contained_line_merges_as_it_did(lob):
    windows = [PolicyWindow(group, pages, copy.deepcopy(answer)) for group, pages, answer in _OLD_STYLE_WINDOWS]
    merged = merge_policy_windows(windows, lob=lob)
    assert merged.extraction == _OLD_STYLE_MERGED
    assert merged.review_flags == ["policy.policy_number:merge_conflict"]
    assert not merged.ambiguous_units and not merged.unplaced_fragments
