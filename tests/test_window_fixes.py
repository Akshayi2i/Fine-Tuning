"""What reading a policy in windows lost, measured by scripts/diagnose_windowing.py.

* A row fragment with none of its row's identifiers in a window (a premium of
  "1.0" whose label lists 42 pages) is an orphan: no merge can place it, so it
  is not taught.
* Rows inside a line block (vehicles, units, coverages) have no declared key:
  the merge joins a row read across two windows on its identifier, unless a
  field both state disagrees.
* A value with no recorded page gets the one page whose OCR text prints it.
"""

from __future__ import annotations

from data_pipeline.dataset_builder.policy_windows import (
    PolicyWindowPlan,
    TargetReport,
    window_target,
    with_inferred_pages,
)
from serving.policy_merge import PolicyWindow, merge_policy_windows


def _env(value, *pages):
    return {"raw": str(value), "parsed": value, "page_ref": list(pages)}


def _plan(group, pages, index=0):
    return PolicyWindowPlan(group, index, tuple(pages), single=False)


#: A self-contained line's block (auto moved to the common model): a location
#: row, identified by its number, holding its own table of coverages.
PACKAGE = {
    "policy": {"policy_number": _env("CPP-1", 1)},
    "commercial_package": {"per_location_schedule": [{
        "location_number": _env("1", 2),
        "coverages": [
            # The label lists the premium on every page a "1.0" is printed.
            {"coverage_name": _env("Medical Payments", 2), "premium": _env(1.0, 1, 2, 5, 9)},
        ],
    }]},
}


def test_an_orphan_fragment_is_not_taught():
    report = TargetReport()
    far = window_target(PACKAGE, "commercial_package", _plan("lineblk", [9], index=1), report)
    assert not far.get("commercial_package", {}).get("per_location_schedule"), far
    assert any("coverages[0]" in path for path in report.orphaned)


def test_the_window_showing_the_identifier_carries_the_whole_row():
    near = window_target(PACKAGE, "commercial_package", _plan("lineblk", [2]))
    (coverage,) = near["commercial_package"]["per_location_schedule"][0]["coverages"]
    assert coverage["coverage_name"]["raw"] == "Medical Payments" and coverage["premium"]["parsed"] == 1.0


def _vehicle(vin, **fields):
    row = {"vin": _env(vin, 1)} if vin else {}
    row.update({k: _env(v, 1) for k, v in fields.items()})
    return row


def _merged_vehicles(*windows):
    merged = merge_policy_windows([
        PolicyWindow("lineblk", [i + 1], {"auto": {"vehicles": rows}}) for i, rows in enumerate(windows)
    ])
    return merged.extraction["auto"]["vehicles"]


def test_a_vehicle_read_across_two_windows_is_one_row():
    rows = _merged_vehicles([_vehicle("VIN-A", year=2019)], [_vehicle("VIN-A", make="Honda")])
    assert len(rows) == 1 and rows[0]["year"]["parsed"] == 2019 and rows[0]["make"]["raw"] == "Honda"


def test_rows_sharing_an_identifier_but_disagreeing_stay_apart():
    rows = _merged_vehicles([_vehicle("VIN-A", year=2019)], [_vehicle("VIN-A", year=2021)])
    assert len(rows) == 2


def test_different_vehicles_are_never_joined():
    rows = _merged_vehicles([_vehicle("VIN-A", year=2019)], [_vehicle("VIN-B", year=2019)])
    assert len(rows) == 2


PAGES = ["Declarations page one", "Named insured Jane Rivera, policy HO-123456",
         "Premium 1200 and 1200 again", "Policy HO-123456 renewal"]


def _unplaced(value):
    return {"raw": value, "parsed": value, "page_ref": []}


def test_a_value_printed_on_exactly_one_page_gets_that_page():
    report = TargetReport()
    label = {"named_insured": {"name": _unplaced("Jane Rivera")}}
    placed = with_inferred_pages(label, PAGES, report)
    assert placed["named_insured"]["name"]["page_ref"] == [2]
    assert report.inferred == ["named_insured.name"]
    assert label["named_insured"]["name"]["page_ref"] == []      # the stored label is untouched


def test_a_value_on_several_pages_or_none_stays_unplaced():
    label = {"policy": {"policy_number": _unplaced("HO-123456"),       # pages 2 and 4
                        "policy_type": _unplaced("Homeowners 3")}}     # nowhere
    assert with_inferred_pages(label, PAGES) is label


def test_a_short_value_is_never_placed_by_search():
    label = {"premium": {"basic_premium": _unplaced("120")}}
    assert with_inferred_pages(label, PAGES) is label


def test_a_value_is_matched_on_word_boundaries():
    """"1200" inside "11200" is not the value."""
    label = {"premium": {"total_policy_premium": _unplaced("1200")}}
    pages = ["Total 11200", "Basic 1200"]
    assert with_inferred_pages(label, pages)["premium"]["total_policy_premium"]["page_ref"] == [2]


def test_the_golden_eval_reports_the_windowing_ceiling():
    from evaluation.golden_eval import GoldenDocument, windowing_ceiling

    doc = GoldenDocument(source_id="p1", doc_type="policy", golden=PACKAGE, image_keys=["a", "b"],
                         page_texts={1: "", 2: ""}, acord_form=None, lob="commercial_package",
                         is_scanned=False, synthetic=False)
    section = windowing_ceiling([doc])
    assert section["documents"] == 1 and section["value_recall"] == 1.0


def test_rows_sharing_only_a_weak_identifier_are_never_joined():
    """`description` names a kind of row, not one row: two discounts both called
    "Multi-policy" with the second half missing its amount are two discounts."""
    def discount(description, **fields):
        return {"description": _env(description, 1), **{k: _env(v, 1) for k, v in fields.items()}}

    merged = merge_policy_windows([
        PolicyWindow("lineblk", [1], {"auto": {"discounts": [discount("Multi-policy", amount=50)]}}),
        PolicyWindow("lineblk", [2], {"auto": {"discounts": [discount("Multi-policy", applies_to="VEH 2")]}}),
    ])
    assert len(merged.extraction["auto"]["discounts"]) == 2
