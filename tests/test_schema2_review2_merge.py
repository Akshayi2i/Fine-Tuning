"""A common-model line's unit merge, second review: what still depended on page order.

The first review's fixes made a row that joins take in every row it shows to be
one unit, and kept a fragment matching two units apart. Both looked only at
the units merged so far, so the same pages read in another order still gave
another document: a fragment read before two 2020 Honda Civics joined the
first of them silently, and one read after a VIN-B Civic was refused the VIN-A
vehicle its number named. Each test here runs every order of its windows and
asks for one answer.

The last tests are a policy-wide discount or lienholder whose windows each see
some of its vehicles - one row, not one per window - and two buildings one
window wrote, which stay two whatever was joined into either.
"""

from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path

import pytest

from common.canonical import values_view
from common.schema_sections import array_key
from common.structural_ids import deciding_key, descriptive_key, same_unit
from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target
from serving.policy_merge import PolicyWindow, _unit_candidates, merge_policy_windows

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
AUTO = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))


def _v(raw, page=1, parsed=None):
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": [page]}


def _merge(*answers, lob="personal_auto"):
    """One arrays window per answer, on pages 1, 2, 3... in the order given."""
    windows = [PolicyWindow("arrays", [i + 1], copy.deepcopy(a)) for i, a in enumerate(answers)]
    return merge_policy_windows(windows, lob=lob)


def _civic(**values):
    return {"unit_id": "veh_1", "year": _v("2020", parsed=2020), "make": _v("Honda"), "model": _v("Civic"),
            **values}


def _collision():
    return {"coverage_code": "X_COLLISION", "coverage_name": _v("Collision"), "applies_to": ["veh_1"]}


def _outcome(merged):
    """What a reader of the served answer sees, whatever the page numbers and
    ids: each vehicle by its values, the vehicles the coverages are on, and
    the flags."""
    def described(vehicle):
        return tuple(sorted((k, values_view(v)) for k, v in vehicle.items() if k != "unit_id"))

    vehicles = merged.extraction["vehicles"]
    by_id = {v["unit_id"]: described(v) for v in vehicles}
    on = sorted(by_id[i] for c in merged.extraction.get("coverages", []) for i in c.get("applies_to", []))
    return sorted(by_id.values()), on, sorted(set(merged.review_flags))


def _every_order(*answers, lob="personal_auto"):
    """The outcome of every order the answers can be read in, by outcome."""
    outcomes: dict[str, list[tuple[int, ...]]] = {}
    for order in itertools.permutations(range(len(answers))):
        outcome = _outcome(_merge(*(answers[i] for i in order), lob=lob))
        outcomes.setdefault(repr(outcome), []).append(order)
    return outcomes


def _described(**values):
    return tuple(sorted((k, v) for k, v in values.items()))


# --------------------------------------------------------------------------
# Units: one answer whatever the order
# --------------------------------------------------------------------------


_NUMBER_2_CIVIC = {"vehicles": [_civic(vehicle_number=_v("2", parsed=2))], "coverages": [_collision()]}
_VIN_A_CIVIC = {"vehicles": [_civic(vin=_v("VIN-A"))]}
_VIN_B_CIVIC = {"vehicles": [_civic(vin=_v("VIN-B"))]}


def test_a_fragment_beside_two_twins_is_kept_apart_and_flagged_in_every_order():
    """A schedule page prints "Veh 2 2020 Honda Civic" and its Collision, with
    no VIN; two other pages print a VIN-A Civic and a VIN-B Civic. Which of
    them is vehicle 2 is not on any page. Read first, the fragment used to join
    the VIN-A Civic silently - its number and its coverage with it."""
    outcomes = _every_order(_NUMBER_2_CIVIC, _VIN_A_CIVIC, _VIN_B_CIVIC)
    civic = {"year": 2020, "make": "Honda", "model": "Civic"}
    number_2 = _described(vehicle_number=2, **civic)
    assert list(outcomes) == [repr((
        sorted([number_2, _described(vin="VIN-A", **civic), _described(vin="VIN-B", **civic)]),
        [number_2],
        ["vehicles:ambiguous_unit"],
    ))], outcomes


_VIN_A_NUMBER_1 = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-A"), "vehicle_number": _v("1", parsed=1)}]}
_NUMBER_1_CIVIC = {"vehicles": [_civic(vehicle_number=_v("1", parsed=1))], "coverages": [_collision()]}


def test_a_row_its_number_places_joins_that_vehicle_in_every_order():
    """VIN-A is vehicle 1; VIN-B is a 2020 Honda Civic; a page prints "#1 2020
    Honda Civic" and its Collision. The number names VIN-A, and VIN-B - known
    to be another vehicle - matches only on the model. Read after VIN-B, the
    row used to be refused as ambiguous and served as a third vehicle holding
    the coverage."""
    outcomes = _every_order(_VIN_A_NUMBER_1, _VIN_B_CIVIC, _NUMBER_1_CIVIC)
    civic = {"year": 2020, "make": "Honda", "model": "Civic"}
    vin_a = _described(vin="VIN-A", vehicle_number=1, **civic)
    assert list(outcomes) == [repr((sorted([vin_a, _described(vin="VIN-B", **civic)]), [vin_a], []))], outcomes


def test_a_row_its_number_places_joins_that_twin_in_every_order():
    """The same, with VIN-A printed as a 2020 Honda Civic too: true twins."""
    vin_a_civic = {"vehicles": [_civic(vin=_v("VIN-A"), vehicle_number=_v("1", parsed=1))]}
    outcomes = _every_order(vin_a_civic, _VIN_B_CIVIC, _NUMBER_1_CIVIC)
    civic = {"year": 2020, "make": "Honda", "model": "Civic"}
    vin_a = _described(vin="VIN-A", vehicle_number=1, **civic)
    assert list(outcomes) == [repr((sorted([vin_a, _described(vin="VIN-B", **civic)]), [vin_a], []))], outcomes


def test_a_row_stating_both_keys_joins_the_two_halves_in_every_order():
    """A VIN-only row, a number-only row and a row printing both are one
    vehicle, whichever of the three is read first."""
    vin_only = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-A")}]}
    number_only = {"vehicles": [{"unit_id": "veh_1", "vehicle_number": _v("1", parsed=1)}],
                   "coverages": [_collision()]}
    outcomes = _every_order(vin_only, number_only, _VIN_A_NUMBER_1)
    vehicle = _described(vin="VIN-A", vehicle_number=1)
    assert list(outcomes) == [repr(([vehicle], [vehicle], []))], outcomes


def test_a_fragment_matching_one_civic_still_joins_it_in_every_order():
    """With one Civic on the policy, its year, make and model are enough."""
    fragment = {"vehicles": [_civic(annual_mileage=_v("9000", parsed=9000))], "coverages": [_collision()]}
    f150 = {"vehicles": [{"unit_id": "veh_1", "vin": _v("VIN-C"), "year": _v("2018", parsed=2018),
                          "make": _v("Ford"), "model": _v("F-150")}]}
    outcomes = _every_order(fragment, _VIN_A_CIVIC, f150)
    civic = _described(vin="VIN-A", year=2020, make="Honda", model="Civic", annual_mileage=9000)
    ford = _described(vin="VIN-C", year=2018, make="Ford", model="F-150")
    assert list(outcomes) == [repr((sorted([civic, ford]), [civic], []))], outcomes


def test_the_key_a_match_is_decided_on_and_whether_it_only_describes_a_unit():
    vin_a, number_1 = {"vin": _v("A"), "vehicle_number": _v("1", parsed=1)}, {"vehicle_number": _v("1", parsed=1)}
    civic = {"year": _v("2020", parsed=2020), "make": _v("Honda"), "model": _v("Civic")}
    assert deciding_key("vehicles", vin_a, {"vin": _v("A")}, "personal_auto") == 0
    assert deciding_key("vehicles", vin_a, number_1, "personal_auto") == 1
    assert deciding_key("vehicles", {"vin": _v("A"), **civic}, civic, "personal_auto") == 2
    assert deciding_key("vehicles", {"vin": _v("A")}, number_1, "personal_auto") is None
    assert same_unit("vehicles", {"vin": _v("A"), **civic}, {"vin": _v("B"), **civic}, "personal_auto") is False
    assert [descriptive_key("vehicles", i, "personal_auto") for i in range(3)] == [False, False, True]
    assert [descriptive_key("scheduled_items", i, "homeowners") for i in range(2)] == [False, True]
    # A building's place at its location names it, though it is two fields.
    assert descriptive_key("buildings", 0, "homeowners") is False


def test_a_weaker_match_known_to_differ_from_a_stronger_one_is_no_candidate():
    """A row printing VIN-A and number 1 is the VIN-A vehicle, not another
    vehicle numbered 1 that is known to differ from it."""
    vin_a = {"vin": _v("A"), "year": _v("2019", parsed=2019)}
    number_1 = {"vehicle_number": _v("1", parsed=1), "year": _v("2020", parsed=2020)}
    row = {"vin": _v("A"), "vehicle_number": _v("1", parsed=1)}
    candidates = _unit_candidates("vehicles", row, [number_1, vin_a], "personal_auto", descriptive=False)
    assert candidates == [vin_a]


# --------------------------------------------------------------------------
# A row that stands on several units
# --------------------------------------------------------------------------


def _vins(merged, ids):
    by_id = {v["unit_id"]: values_view(v["vin"]) for v in merged.extraction["vehicles"]}
    return sorted(by_id[i] for i in ids)


@pytest.mark.format_profile("standard")
def test_a_discount_and_a_party_on_vehicles_in_two_windows_are_one_row_each(no_window_overlap):
    """One label row each - a multi-car discount and a lienholder on both
    vehicles - with vehicle 1 printed in the first arrays window and vehicle 2
    in the second. Each window is taught a link only to the vehicle it sees;
    merged, the training targets are the label again: one row each, on both."""
    gold = copy.deepcopy(AUTO)
    for page, vehicle in zip((2, 5), gold["vehicles"], strict=True):
        for value in vehicle.values():
            if isinstance(value, dict) and "page_ref" in value:
                value["page_ref"] = [page]
    both = [v["unit_id"] for v in gold["vehicles"]]
    gold["rating_modifiers"] = [{"modifier_type": "other", "description": {**_v("Multi-Car"), "page_ref": [2, 5]},
                                 "applies_to": both}]
    gold["interested_parties"] = [{"role": "lienholder", "name": {**_v("Jane Roe"), "page_ref": [2, 5]},
                                   "applies_to": both}]
    report, windows = TargetReport(), []
    for plan in plan_windows("personal_auto", [1, 2, 3, 4, 5, 6], 1):
        target = window_target(copy.deepcopy(gold), "personal_auto", plan, report)
        windows.append(PolicyWindow(plan.group, list(plan.pages), target))
    arrays = [w for w in windows if w.group == "arrays"]
    assert [w.pages for w in arrays] == [[1, 2, 3], [4, 5, 6]]
    assert all(len(w.extraction["vehicles"]) == 1 for w in arrays)

    merged = merge_policy_windows(windows, lob="personal_auto")
    every_vin = sorted(values_view(v["vin"]) for v in AUTO["vehicles"])
    (modifier,) = merged.extraction["rating_modifiers"]
    assert _vins(merged, modifier["applies_to"]) == every_vin
    parties = [(values_view(p["name"]), _vins(merged, p["applies_to"])) for p in merged.extraction["interested_parties"]]
    assert ("Jane Roe", every_vin) in parties and [name for name, _ in parties].count("Jane Roe") == 1
    assert not merged.conflicts


def test_a_window_that_sees_one_of_the_vehicles_adds_no_second_discount():
    """The first window sees both vehicles and the discount on both; the next
    sees only vehicle B, and links the discount to it alone."""
    first = {"vehicles": [{"unit_id": "veh_1", "vin": _v("A")}, {"unit_id": "veh_2", "vin": _v("B")}],
             "rating_modifiers": [{"modifier_type": "discount", "description": _v("Multi-Car"),
                                   "applies_to": ["veh_1", "veh_2"]}]}
    second = {"vehicles": [{"unit_id": "veh_1", "vin": _v("B", 2)}],
              "rating_modifiers": [{"modifier_type": "discount", "description": _v("Multi-Car", 2),
                                    "applies_to": ["veh_1"]}]}
    merged = _merge(first, second)
    (modifier,) = merged.extraction["rating_modifiers"]
    assert _vins(merged, modifier["applies_to"]) == ["A", "B"] and merged.review_flags == []


def test_a_party_and_a_modifier_are_keyed_without_their_units_on_a_common_model_line_only():
    assert array_key("rating_modifiers", "personal_auto") == ("modifier_type", "description")
    assert array_key("interested_parties", "personal_auto") == ("role", "name")
    assert array_key("interested_parties", "equipment_breakdown") == ("name", "party_type")   # self-contained


# --------------------------------------------------------------------------
# Two buildings one window wrote
# --------------------------------------------------------------------------


def test_two_buildings_one_window_wrote_stay_two_after_one_is_joined_to_another_window():
    """Page 1 shows the dwelling; page 2 shows it again beside a second
    unnumbered structure, with the Other Structures coverage on that. The
    dwelling re-read joins page 1's; the structure is still page 2's second
    building, not more of the dwelling."""
    location = [{"unit_id": "loc_1", "location_number": _v("1", parsed=1)}]
    first = {"locations": location,
             "buildings": [{"unit_id": "bldg_1", "location_ref": "loc_1", "year_built": _v("1995", parsed=1995)}]}
    second = {"locations": location,
              "buildings": [{"unit_id": "bldg_1", "location_ref": "loc_1", "year_built": _v("1995", 2, 1995)},
                            {"unit_id": "bldg_2", "location_ref": "loc_1", "roof_type": _v("Metal", 2)}],
              "coverages": [{"coverage_code": "HO_COV_B", "coverage_name": _v("Other Structures", 2),
                             "applies_to": ["bldg_2"]}]}
    merged = _merge(first, second, lob="homeowners")
    buildings = [(b["unit_id"], values_view({k: v for k, v in b.items() if k not in ("unit_id", "location_ref")}))
                 for b in merged.extraction["buildings"]]
    assert buildings == [("bldg_1", {"year_built": 1995}), ("bldg_2", {"roof_type": "Metal"})]
    (coverage,) = merged.extraction["coverages"]
    assert coverage["applies_to"] == ["bldg_2"]
