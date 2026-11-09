"""An orphaned row piece is lost only when no other window teaches its values (2026-11-09).

data_health on the pod listed up to 46 orphaned pieces per policy as values no training row teaches, and the build
warned they were "printed on pages no window of their group reads". On a common-model line a piece is orphaned only
when its values are printed on another page too; whether that page's window teaches them was never checked. The
window builder now records what each kept window teaches, and an orphan's values are checked against it.
"""

from __future__ import annotations

from data_pipeline.dataset_builder.build_jsonl import _lost_kind, _merge_report
from data_pipeline.dataset_builder.policy_windows import (
    PolicyWindowPlan,
    TargetReport,
    orphans_taught_nowhere,
    window_target,
)


def _env(value, *pages):
    return {"raw": str(value), "parsed": value, "page_ref": list(pages)}


def _plan(group, pages, index=0):
    return PolicyWindowPlan(group, index, tuple(pages), single=False)


#: A location row holding its own coverages; the label lists the premium on every page a "1.0" is printed, so it
#: reaches page 9, which shows no coverage name (tests/test_window_fixes.py).
PACKAGE = {
    "policy": {"policy_number": _env("CPP-1", 1)},
    "commercial_package": {"per_location_schedule": [{
        "location_number": _env("1", 2),
        "coverages": [
            {"coverage_name": _env("Medical Payments", 2), "premium": _env(1.0, 1, 2, 5, 9)},
        ],
    }]},
}
PREMIUM = "commercial_package.per_location_schedule[0].coverages[0].premium"


def _window(pages, index):
    report = TargetReport()
    window_target(PACKAGE, "commercial_package", _plan("lineblk", pages, index), report)
    return report


def test_an_orphan_is_recorded_with_its_values_and_they_are_not_counted_as_taught_there():
    far = _window([9], 1)
    assert any("coverages[0]" in path for path in far.orphaned)
    assert any(path.endswith("coverages[0].premium") for path in far.orphaned_values)
    assert not any(path.endswith("coverages[0].premium") for path in far.taught)


def test_an_orphan_whose_row_another_window_teaches_is_not_lost():
    document = TargetReport()
    for report in (_window([9], 1), _window([2], 0)):
        _merge_report(document, report)
    assert any(path.endswith("coverages[0].premium") for path in document.taught)
    assert orphans_taught_nowhere(document) == []


def test_an_orphan_no_other_window_teaches_is_lost():
    document = TargetReport()
    _merge_report(document, _window([9], 1))           # the window showing the coverage's name is left out
    lost = orphans_taught_nowhere(document)
    assert len(lost) == 1 and lost[0].endswith("coverages[0].premium"), lost
    assert lost[0] == PREMIUM or lost[0].endswith(PREMIUM.split(".", 1)[1])


def test_the_build_warns_only_of_values_no_row_teaches():
    assert _lost_kind("window image_only: unread coverages[3].limits[0].amount") == "unread"
    assert _lost_kind("window image_only: unplaced forms[2].form_number") == "unplaced"
    assert _lost_kind(f"window image_only: orphan taught nowhere {PREMIUM}") == "orphan taught nowhere"
    assert _lost_kind("window image_only: orphaned arrays:coverages[0]") is None      # a piece, its values checked
    assert _lost_kind("window image_only: away from home arrays:coverages[5]") is None  # taught on its home pages
