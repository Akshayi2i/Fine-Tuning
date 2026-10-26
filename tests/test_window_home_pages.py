"""A coverage row is taught on the windows that show its home: where its name
or a figure of its own is printed.

A label lists every page that prints a value, and a limit's caption ("Each
Occurrence Limit") or the word "Included" is printed throughout policy wording
too. A GL schedule names no coverage, so its table has no printed identifier
and the orphan rule never runs: a window of wording was taught a coverage row
holding only that caption, and the smoke-trained model wrote coverages from
wording - or learned that schedules hold rows a page does not show.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from common import schemas as S
from common.canonical import values_view
from data_pipeline.dataset_builder.policy_windows import TargetReport, plan_windows, window_target

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schema2"
AUTO = json.loads((FIXTURES / "personal_auto_6page.json").read_text(encoding="utf-8"))


def _fv(raw, parsed, pages):
    return {"raw": raw, "parsed": parsed, "confidence": {"score": 1.0, "source": "deterministic"},
            "page_ref": pages, "flagged": False}


def _nameless():
    """The fixture as a GL schedule prints it: no coverage named. Vehicle 1's
    rows are on page 4, vehicle 2's on page 5."""
    gold = copy.deepcopy(AUTO)
    for row in gold["coverages"]:
        del row["coverage_name"]
    return gold


def _targets(gold):
    report = TargetReport()
    out = {plan.pages: window_target(gold, "personal_auto", plan, report)
           for plan in plan_windows("personal_auto", [1, 2, 3, 4, 5, 6], 1) if plan.group == "arrays"}
    assert list(S.iter_validation_errors(gold, "policy", None, "personal_auto")) == []
    return out, report


def _rows(target, code, premium="any"):
    """A window's coverage rows of ``code``, by premium (None: rows with none).
    Not by vehicle: the vehicles are read by another group, so the arrays
    windows leave ``applies_to`` out."""
    return [c for c in target.get("coverages") or [] if c["coverage_code"] == code
            and (premium == "any" or values_view(c.get("premium")) == premium)]


@pytest.mark.format_profile("standard")
def test_a_caption_the_wording_prints_does_not_teach_its_row_there(no_window_overlap):
    """Vehicle 2's liability limit is captioned "Each Person" on page 5, and the
    label cites the caption on page 2 as well, where the wording uses it."""
    gold = _nameless()
    gold["coverages"][2]["limits"][0]["description"] = _fv("Each Person", "Each Person", [5, 2])
    targets, report = _targets(gold)
    assert "coverages" not in targets[(1, 2, 3)]
    assert report.away == ["arrays:coverages[2]"]
    (row,) = _rows(targets[(4, 5, 6)], "X_VEHICLE_LIABILITY", 455.0)
    assert values_view(row["limits"][0]["description"]) == "Each Person"
    assert row["limits"][0]["description"]["page_ref"] == [5]
    assert [values_view(limit["amount"]) for limit in row["limits"]] == [100000, 300000, 50000]


@pytest.mark.format_profile("standard")
def test_a_row_of_captions_alone_has_no_home_and_stays_where_it_is_cited(no_window_overlap):
    """A row stating no name and no figure: nothing says where it belongs, so
    it is taught on every window that cites it, as before."""
    gold = _nameless()
    collision = gold["coverages"][1]
    for key in ("premium", "deductibles"):
        del collision[key]
    collision["limits"] = [{"limit_type": "sublimit",
                            "description": _fv("Included", "Included", [2, 4])}]
    targets, report = _targets(gold)
    assert report.away == []
    for pages, page in (((1, 2, 3), 2), ((4, 5, 6), 4)):
        (row,) = _rows(targets[pages], "X_COLLISION", None)
        assert row["limits"][0]["description"]["page_ref"] == [page]


@pytest.mark.format_profile("standard")
def test_a_row_split_at_the_window_boundary_is_at_home_on_both_sides(no_window_overlap):
    """A figure is a home: vehicle 1's liability premium printed on page 3 and
    its limits on page 4 keep a row in each window, as the orphan rule does
    for a named row split the same way."""
    gold = _nameless()
    gold["coverages"][0]["premium"]["page_ref"] = [3]
    targets, report = _targets(gold)
    assert report.away == []
    assert len(_rows(targets[(1, 2, 3)], "X_VEHICLE_LIABILITY", 410.0)) == 1
    (row,) = _rows(targets[(4, 5, 6)], "X_VEHICLE_LIABILITY", None)
    assert [values_view(limit["amount"]) for limit in row["limits"]] == [100000, 300000, 50000]
