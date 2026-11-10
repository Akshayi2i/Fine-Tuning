"""Measures added so every later change is judged honestly (accuracy plan, stage 0.3)."""

from __future__ import annotations

from evaluation.metrics.extraction_faults import score_page_refs
from evaluation.run_eval import _error_summary
from training.vit_gate import ErrorRecord, classify_error, summarise_error_mix


def _value(parsed, pages):
    return {"raw": str(parsed), "parsed": parsed, "page_ref": pages}


def test_an_error_is_classed_by_what_went_wrong():
    gold = {"policy": {"policy_number": "NYHP0000004000"}, "carrier": {"name": "Mercury Casualty Company"}}
    assert classify_error("NYHP0000004000", None, all_expected=gold) == "omission"
    assert classify_error("NYHP0000004000", "NYHP0000004001", all_expected=gold) == "perception"
    # The policy number written where the label holds nothing: read right, placed wrong.
    assert classify_error(None, "NYHP0000004000", all_expected=gold) == "schema_reasoning"
    # A value the label holds nowhere: invented.
    assert classify_error(None, "Debris Removal", all_expected=gold) == "invented"
    assert classify_error(0, 5, all_expected=gold) == "perception"     # a zero is a value, not "nothing"


def test_a_value_the_gold_holds_written_another_way_is_not_called_invented():
    """2026-11-10 review, proven on the real scorer: exact-text search called a value invented beside the gold's
    own copy in another case or form, and called a gold-gap value misplaced because the gold cites that number as a
    page."""
    gold = {"coverages": [{"coverage_name": _value("BUILDING", [3]),
                           "limits": [{"description": _value("OFF PREMISES", [3]), "amount": _value(10000.0, [3])}]}],
            "policy": {"effective_date": _value("08/18/2022", [1, 7])}}
    assert classify_error(None, "Off Premises", all_expected=gold) == "schema_reasoning"
    assert classify_error(None, 10000, all_expected=gold) == "schema_reasoning"
    assert classify_error(None, "$10,000", all_expected=gold) == "schema_reasoning"
    assert classify_error(None, "2022-08-18", all_expected=gold) == "schema_reasoning"
    # Only a page number of the gold: not a value it holds.
    assert classify_error(None, 7, all_expected=gold) == "invented"
    # The expected value in another case is a near-miss on it, not a value from another field.
    assert classify_error("ACCOUNTS RECEIVABLE", "Accounts Receivable.", all_expected=gold) == "perception"
    # A form number stays text: "CG 00 01" is not the amount 1.
    assert classify_error(None, "CG 00 01", all_expected={"x": _value(1, [1])}) == "invented"


def test_invented_values_stay_out_of_the_vision_error_mix():
    errors = [ErrorRecord("d", "a", None, "x", error_class="invented"),
              ErrorRecord("d", "b", "1", "7", error_class="perception")]
    assert summarise_error_mix(errors) == {"perception": 1.0}


def test_right_values_are_checked_for_the_pages_they_cite():
    gold = {"policy": {"policy_number": _value("NYHP0000004000", [1, 2, 3, 4]),
                       "effective_date": _value("05/21/2025", [2])}}
    got = {"policy": {"policy_number": _value("NYHP0000004000", [2]),          # right value, pages missing
                      "effective_date": _value("05/21/2025", [2])}}           # right value, right page
    report = score_page_refs([(gold, got)])
    assert (report.compared, report.exact) == (2, 1)
    assert report.precision == 1.0 and report.recall == 2 / 5


def test_a_wrong_value_says_nothing_about_its_pages():
    gold = {"policy": {"policy_number": _value("NYHP0000004000", [2])}}
    got = {"policy": {"policy_number": _value("ZZ-1", [9])}}
    assert score_page_refs([(gold, got)]).compared == 0


def _home(location: dict, coverages: list[dict]) -> dict:
    return {"lob_parts": [{"lob": "homeowners"}], "locations": [location], "coverages": coverages}


def test_a_link_to_one_location_named_two_ways_is_found():
    from evaluation.metrics.common_model import CommonModelTally

    address = {"street": "2 TOWN RD", "city": "MOUNT MARION", "state": "NY", "postal_code": "12456"}
    label_key = "locations:address=12456|2 town rd|mount marion|ny"
    gold = _home({"address": address}, [{"coverage_name": "A. Dwelling", "coverage_code": "HO_COV_A",
                                         "applies_to": [label_key]}])
    got = _home({"location_number": 2, "address": address},
                [{"coverage_name": "A. Dwelling", "coverage_code": "HO_COV_A",
                  "applies_to": ["locations:location_number=2"]}])
    tally = CommonModelTally()
    tally.add(gold, got, "homeowners")
    assert tally.metrics()["reference_accuracy"] == 1.0          # was 0.0 on exact key strings


def _premises_doc(target, *, number=None, buildings=1):
    from common.structural_ids import comparable_view

    rows = []
    for n in range(1, buildings + 1):
        row = {"unit_id": f"bldg_{n}", "location_ref": "loc_1", "year_built": _value(1990 + n, [1])}
        if number:
            row["building_number"] = _value(n, [1])
        rows.append(row)
    return comparable_view({
        "locations": [{"unit_id": "loc_1", "address": {"street": _value("2 TOWN RD", [1])}}],
        "buildings": rows,
        "coverages": [{"coverage_id": "cov_1", "coverage_code": "HO_COV_A", "coverage_name": _value("Dwelling", [1]),
                       "applies_to": [target]}]}, "homeowners")


def test_a_location_and_its_only_building_are_one_premises():
    from evaluation.metrics.common_model import _links

    # Two links each: the coverage's, and the building's own location_ref. The
    # labels name such a premises by its location in most seeds, its building in others.
    assert _links(_premises_doc("loc_1"), _premises_doc("bldg_1"), "homeowners") == (2, 2)
    assert _links(_premises_doc("bldg_1"), _premises_doc("loc_1"), "homeowners") == (2, 2)
    assert _links(_premises_doc("loc_1", number=True), _premises_doc("bldg_1", number=True), "homeowners") == (2, 2)


def test_a_location_with_two_buildings_is_not_either_building():
    from evaluation.metrics.common_model import _links

    gold, got = _premises_doc("loc_1", number=True, buildings=2), _premises_doc("bldg_1", number=True, buildings=2)
    assert _links(gold, got, "homeowners") == (2, 3)          # both buildings' location_ref, not the coverage


def test_coverage_code_recall_counts_coverages_the_answer_missed():
    from evaluation.metrics.common_model import CommonModelTally

    gold = {"coverages": [{"coverage_name": "A. Dwelling", "coverage_code": "HO_COV_A"},
                          {"coverage_name": "D. Loss Of Use", "coverage_code": "X_LOSS_OF_USE"}]}
    got = {"coverages": [{"coverage_name": "A. Dwelling", "coverage_code": "HO_COV_A"}]}
    tally = CommonModelTally()
    tally.add(gold, got, "homeowners")
    metrics = tally.metrics()
    assert (metrics["coverage_code_accuracy"], metrics["coverage_code_recall"]) == (1.0, 0.5)


def test_errors_are_totalled_by_class_line_and_field():
    records = [
        {"field_path": "coverages[0].coverage_name", "error_class": "invented", "lob": "homeowners"},
        {"field_path": "coverages[3].coverage_name", "error_class": "invented", "lob": "homeowners"},
        {"field_path": "locations[0].location_number", "error_class": "schema_reasoning", "lob": "homeowners"},
        {"field_path": "policy.policy_number", "error_class": "omission", "lob": "personal_auto"},
    ]
    summary = _error_summary(records)
    assert summary["by_class"] == {"invented": 2, "schema_reasoning": 1, "omission": 1}
    assert summary["by_line"]["homeowners"] == {"invented": 2, "schema_reasoning": 1}
    assert summary["top_fields"][0] == {"field": "coverages[].coverage_name", "errors": 2,
                                        "by_class": {"invented": 2}}
    assert _error_summary([]) == {}
