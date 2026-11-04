"""One coverage row per coverage, and coverage names OCR broke (the user's labelling rules, 2026-11-04)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from data_pipeline.ingestion.label_rules import merge_coverage_limits, split_name_repairs
from data_pipeline.ingestion.prepare_bundles import _repair_split_coverage_names


def _env(value, pages=(4,)):
    return {"raw": str(value), "parsed": value, "page_ref": list(pages)}


def _row(code, name, limits, identifier="cov_1", **more):
    return {"coverage_code": code, "coverage_id": identifier,
            "coverage_name": _env(name) if name else None,
            "limits": [{"limit_type": kind, "amount": _env(amount)} for kind, amount in limits], **more}


def test_two_limits_of_one_coverage_become_one_row_and_no_printed_word_is_lost():
    gold = {"coverages": [_row("GL_BI_PD", "EACH OCCURRENCE LIMIT", [("per_occurrence", 1000000)], "cov_1"),
                          _row("GL_MED", "MED EXP", [("per_person", 5000)], "cov_2"),
                          _row("GL_BI_PD", "GENERAL AGGREGATE LIMIT", [("aggregate", 2000000)], "cov_5")]}
    notes = merge_coverage_limits(gold)
    assert [row["coverage_id"] for row in gold["coverages"]] == ["cov_1", "cov_2"]
    first = gold["coverages"][0]
    assert first["coverage_name"]["parsed"] == "EACH OCCURRENCE LIMIT"
    assert [limit["limit_type"] for limit in first["limits"]] == ["per_occurrence", "aggregate"]
    assert first["limits"][1]["description"]["parsed"] == "GENERAL AGGREGATE LIMIT"
    assert notes == [("coverage limits merged",
                      "coverages: GL_BI_PD: 'GENERAL AGGREGATE LIMIT' merged into 'EACH OCCURRENCE LIMIT'")]


def test_rows_that_apply_to_different_units_stay_apart():
    gold = {"coverages": [_row("X_MED_PAY", "Cov M", [("per_person", 1)], "a", applies_to=["bldg_1"]),
                          _row("X_MED_PAY", "Cov M", [("per_accident", 2)], "b", applies_to=["bldg_2"])]}
    assert merge_coverage_limits(gold) == [] and len(gold["coverages"]) == 2


def test_rows_of_the_same_applies_to_merge():
    gold = {"coverages": [_row("X_MED_PAY", "Cov M each person", [("per_person", 1)], "a", applies_to=["bldg_1"]),
                          _row("X_MED_PAY", "Cov M each accident", [("per_accident", 2)], "b", applies_to=["bldg_1"])]}
    assert len(merge_coverage_limits(gold)) == 1 and len(gold["coverages"]) == 1


def test_a_limit_type_twice_means_two_coverages():
    gold = {"coverages": [
        _row("X_CYBER", "CYBER LIABILITY", [("per_claim", 1), ("aggregate", 2)], "a"),
        _row("X_CYBER", "FIRST PARTY BREACH", [("per_claim", 3), ("aggregate", 4)], "b")]}
    assert merge_coverage_limits(gold) == [] and len(gold["coverages"]) == 2


def test_a_row_with_no_limit_may_be_anything_so_it_stays():
    gold = {"coverages": [_row("GL_EXT", "Contractors Protector", [], "a"), _row("GL_EXT", "GL Extension", [], "b")]}
    assert merge_coverage_limits(gold) == [] and len(gold["coverages"]) == 2


def test_a_field_on_one_row_moves_and_on_both_leaves_them_apart():
    one = {"coverages": [_row("C", "A", [("per_occurrence", 1)], "a"),
                         _row("C", "B", [("aggregate", 2)], "b", premium=_env(100))]}
    assert len(merge_coverage_limits(one)) == 1 and one["coverages"][0]["premium"]["parsed"] == 100
    both = {"coverages": [_row("C", "A", [("per_occurrence", 1)], "a", premium=_env(100)),
                          _row("C", "B", [("aggregate", 2)], "b", premium=_env(200))]}
    assert merge_coverage_limits(both) == [] and len(both["coverages"]) == 2


def test_a_gold_with_no_coverages_is_left_alone():
    assert merge_coverage_limits({}) == []


def test_a_name_broken_inside_its_words_is_repaired_from_the_same_name_printed_whole():
    repairs = split_name_repairs(["Each Occurrence Limit", "EACH OCCURRENCE LIM IT", "medical ex pen se lim it",
                                  "Medical Expense Limit", "person al an d advertising injury limit",
                                  "personal and advertising injury limit"])
    assert repairs == {"EACH OCCURRENCE LIM IT": "EACH OCCURRENCE LIMIT",
                       "medical ex pen se lim it": "medical expense limit",
                       "person al an d advertising injury limit": "personal and advertising injury limit"}


def test_spacing_around_punctuation_is_how_a_page_prints_it():
    assert split_name_repairs(["products / completed operations", "products/completed operations"]) == {}
    assert split_name_repairs(["pro d u c t s / completed operation s aggregate lim it",
                               "products/completed operations aggregate limit"]) == {
        "pro d u c t s / completed operation s aggregate lim it": "products / completed operations aggregate limit"}


def test_a_name_no_label_prints_whole_stays_as_it_is():
    assert split_name_repairs(["lim it of cover"]) == {}


def test_the_bundle_step_repairs_across_documents_and_never_writes_through_a_link(tmp_path):
    def gold(name):
        return {"coverages": [{"coverage_code": "C", "coverage_name": _env(name)}]}

    for folder, name in (("a", "Each Occurrence Limit"), ("b", "Each Occurrence Lim it")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "golden.json").write_text(json.dumps(gold(name)), encoding="utf-8")
    delivered = tmp_path / "delivered.json"
    delivered.write_text(json.dumps(gold("Each Occurrence Lim it")), encoding="utf-8")
    (tmp_path / "b" / "golden.json").unlink()
    Path(tmp_path / "b" / "golden.json").hardlink_to(delivered)

    report = SimpleNamespace(corrections=Counter(), correction_rows=[])
    _repair_split_coverage_names(tmp_path, report)

    fixed = json.loads((tmp_path / "b" / "golden.json").read_text(encoding="utf-8"))["coverages"][0]["coverage_name"]
    assert fixed["raw"] == fixed["parsed"] == "Each Occurrence Limit"
    assert "Lim it" in delivered.read_text(encoding="utf-8"), "the delivered gold is never changed"
    assert report.corrections == {"coverage name repaired": 1}
    assert report.correction_rows == [("b", "coverage name repaired", "'Each Occurrence Lim it' -> 'Each Occurrence Limit'")]
