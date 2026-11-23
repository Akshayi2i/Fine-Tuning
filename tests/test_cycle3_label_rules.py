"""The cycle 3 labelling rules (user, 2026-11-19): what the label preparation fixes by itself, and what the gold
checker lists for a person."""

from __future__ import annotations

from data_pipeline.ingestion.label_rules import (
    BROKEN_CHARACTER,
    BROKEN_CHARACTER_LEFT,
    apply_cycle3_rules,
    apply_label_rules,
    clear_code_descriptions,
    clear_premiums_without_amount,
    repair_broken_characters,
    strip_form_ref_editions,
)
from scripts.check_golds import check_gold, cycle3_findings, template_form_ref_findings


def _env(value, pages=(1,)):
    return {"raw": None if value is None else str(value), "parsed": value, "page_ref": list(pages)}


def _form(number, edition=None, title="Some Endorsement"):
    return {"form_number": _env(number), "edition_date": _env(edition) if edition else None, "title": _env(title)}


# --- R2: coverage form references written as the forms list's number --------------------------------------------

def test_a_form_reference_loses_the_edition_its_form_prints():
    gold = {"forms_and_endorsements": [_form("SP 30 18", "10 18")],
            "coverages": [{"coverage_name": _env("Hired Auto"), "form_refs": ["SP 30 18 10 18"]}]}
    notes = strip_form_ref_editions(gold)
    assert gold["coverages"][0]["form_refs"] == ["SP 30 18"]
    assert notes == [("form reference edition removed", "coverages[0].form_refs[0] 'SP 30 18 10 18' -> 'SP 30 18'")]


def test_a_form_reference_spelled_with_other_spacing_takes_the_forms_list_spelling():
    gold = {"forms_and_endorsements": [_form("CG 20 18", "04 13")],
            "coverages": [{"form_refs": ["CG20180413", "CG 20 18"]}]}
    strip_form_ref_editions(gold)
    assert gold["coverages"][0]["form_refs"] == ["CG 20 18", "CG 20 18"]


def test_a_reference_already_in_the_forms_list_or_unknown_is_left_alone():
    gold = {"forms_and_endorsements": [_form("CG20180413")],
            "coverages": [{"form_refs": ["CG20180413", "XX 99 99 01 20"]}]}
    assert strip_form_ref_editions(gold) == []
    assert gold["coverages"][0]["form_refs"] == ["CG20180413", "XX 99 99 01 20"]


def test_an_enveloped_reference_is_changed_in_raw_and_parsed():
    ref = _env("SP 30 18 10 18")
    gold = {"forms_and_endorsements": [_form("SP 30 18", "10 18")], "coverages": [{"form_refs": [ref]}]}
    strip_form_ref_editions(gold)
    assert (ref["raw"], ref["parsed"]) == ("SP 30 18", "SP 30 18")


# --- R5, R3 ---------------------------------------------------------------------------------------------------------

def test_a_class_code_written_as_the_description_is_emptied():
    gold = {"rating_exposures": [{"class_code": _env("768606"), "class_description": _env("768606")},
                                 {"class_code": _env("41650"), "class_description": _env("Restaurants")}]}
    notes = clear_code_descriptions(gold)
    assert gold["rating_exposures"][0]["class_description"] is None
    assert gold["rating_exposures"][1]["class_description"]["raw"] == "Restaurants"
    assert len(notes) == 1


def test_a_premium_printed_as_a_word_is_emptied_and_an_amount_kept():
    gold = {"coverages": [{"premium": _env("Included")}, {"premium": _env("$1,250")}, {"premium": None}]}
    notes = clear_premiums_without_amount(gold)
    assert gold["coverages"][0]["premium"] is None
    assert gold["coverages"][1]["premium"]["raw"] == "$1,250"
    assert notes == [("premium without an amount", "coverages[0].premium 'Included' emptied")]


# --- R8: broken characters --------------------------------------------------------------------------------------

def test_a_broken_character_takes_the_one_its_page_prints():
    value = _env(f"Spectrum{BROKEN_CHARACTER} Business Owner{BROKEN_CHARACTER}s Policy", pages=(2,))
    gold = {"policy": {"policy_type": value}}
    notes = repair_broken_characters(gold, {1: "nothing", 2: "SPEC\nSpectrum® Business\nOwner’s Policy"})
    assert value["raw"] == value["parsed"] == "Spectrum® Business Owner’s Policy"
    assert notes[0][0] == "broken character repaired"


def test_a_broken_character_no_text_settles_is_listed_and_left():
    broken = f"Owner{BROKEN_CHARACTER}s"
    gold = {"a": _env(broken), "b": _env(broken)}
    pages = {1: f"Owner{BROKEN_CHARACTER}s"}            # the text layer holds the broken character itself
    notes = repair_broken_characters(gold, pages)
    assert gold["a"]["raw"] == broken
    assert {kind for kind, _ in notes} == {BROKEN_CHARACTER_LEFT}


def test_two_spellings_that_fit_leave_the_value_for_a_person():
    gold = {"a": _env(f"Owner{BROKEN_CHARACTER}s")}
    notes = repair_broken_characters(gold, {1: "Owner's and Owner’s"})
    assert notes[0][0] == BROKEN_CHARACTER_LEFT


# --- where they run ---------------------------------------------------------------------------------------------

def test_the_shared_rules_used_by_serving_and_the_gate_do_not_apply_them():
    gold = {"forms_and_endorsements": [_form("SP 30 18", "10 18")],
            "coverages": [{"coverage_name": _env("Hired Auto"), "form_refs": ["SP 30 18 10 18"]}],
            "rating_exposures": [{"class_code": _env("768606"), "class_description": _env("768606")}]}
    apply_label_rules(gold, {})
    assert gold["coverages"][0]["form_refs"] == ["SP 30 18 10 18"]
    assert gold["rating_exposures"][0]["class_description"]["raw"] == "768606"
    assert {kind for kind, _ in apply_cycle3_rules(gold, {})} == {"form reference edition removed",
                                                                  "class code as description"}


# --- the checker's hints for a person ---------------------------------------------------------------------------

def _rules(findings):
    return {rule.split(":")[0] + ":" + rule.split(":")[1][:20] for rule, _, _ in findings}


def test_the_checker_lists_rows_and_values_a_person_should_look_at():
    gold = {
        "forms_and_endorsements": [_form("IL 00 17"), {"form_number": None, "title": _env("24/7 Claim Reporting")},
                                   {"form_number": None, "title": None}],
        "coverages": [{"coverage_name": None, "coverage_code": "GL_X", "form_refs": ["ZZ 1"]},
                      {"coverage_name": _env("Hired Auto"), "coverage_code": "GL_HA"}],
        "premium": {"items": [{"unit_type": "coverage", "coverage_code": "GL_HA", "amount": _env("$50", (3,))}]},
        "rating_exposures": [{"class_code": _env("41650")}, {}],
        "policy": {}, "document": {},
    }
    pages = {1: "Date Printed: 01/02/2026", 2: "Rates  Pr/Co  All Other", 3: "Countersigned by",
             4: "Name of Officer  Title  Included"}
    rules = [rule for rule, _, _ in cycle3_findings(gold, pages, lob="wc")]
    assert any(r.startswith("R1:") for r in rules)
    assert sum(r.startswith("R9:") for r in rules) == 3      # form, coverage, rating row
    assert any(r.startswith("R2: a coverage form reference not in") for r in rules)
    assert any(r.startswith("R3:") for r in rules)
    assert any(r.startswith("R4:") for r in rules)
    assert any(r.startswith("R6: Pr/Co") for r in rules)
    assert sum(r.startswith("R7:") for r in rules) == 3      # print date, countersignature, officers


def test_a_coverage_premium_is_matched_on_its_vehicle_not_its_code_alone():
    # An auto policy lists each coverage once per vehicle; the premium list holds one amount per vehicle.
    gold = {
        "coverages": [{"coverage_name": _env("Liability"), "coverage_code": "X_LIAB", "applies_to": ["veh_1"]},
                      {"coverage_name": _env("Liability"), "coverage_code": "X_LIAB", "applies_to": ["veh_2"]},
                      {"coverage_name": _env("Collision"), "coverage_code": "X_COLL", "applies_to": ["veh_1"]}],
        "premium": {"items": [
            {"unit_type": "coverage", "coverage_code": "X_LIAB", "applies_to": ["veh_1"], "amount": _env("$44", (21,))},
            {"unit_type": "coverage", "coverage_code": "X_LIAB", "applies_to": ["veh_2"], "amount": _env("$61", (22,))},
            {"unit_type": "coverage", "coverage_code": "X_COLL", "applies_to": ["veh_1"], "amount": _env("$10", (21,))},
            {"unit_type": "coverage", "coverage_code": "X_COLL", "applies_to": ["veh_1"], "amount": _env("$12", (23,))},
        ]},
        "policy": {}, "document": {},
    }
    found = {where: (rule, detail) for rule, where, detail in cycle3_findings(gold, {}, lob="auto")
             if rule.startswith("R3")}
    assert "'$44'" in found["coverages[0]"][1]
    assert "'$61'" in found["coverages[1]"][1]          # vehicle 2's own amount, not vehicle 1's
    assert "check by hand" in found["coverages[2]"][0]  # two amounts for one coverage and vehicle: a person decides


def test_a_policy_minimum_is_flagged_only_when_the_policy_prints_a_minimum_per_coverage_part():
    gold = {"premium": {"minimum": _env("$455", (5,))}, "policy": {}, "document": {}}
    two_parts = {4: "MINIMUM PREMIUM FOR PROPERTY COVERAGE PART: $50",
                 5: "MINIMUM PREMIUM FOR GENERAL LIABILITY COVERAGE PART:\n$455"}
    one_minimum = {3: "MINIMUM PREMIUM $ 345   DEPOSIT PREMIUM $ 995"}
    assert any(r.startswith("R11:") for r, _, _ in cycle3_findings(gold, two_parts, lob="boppr"))
    assert not any(r.startswith("R11:") for r, _, _ in cycle3_findings(gold, one_minimum, lob="wc"))
    empty = {"premium": {}, "policy": {}, "document": {}}
    assert not any(r.startswith("R11:") for r, _, _ in cycle3_findings(empty, two_parts, lob="boppr"))


def test_a_complete_gold_gets_no_hint():
    gold = {"forms_and_endorsements": [_form("IL 00 17")],
            "coverages": [{"coverage_name": _env("Hired Auto"), "form_refs": ["IL 00 17"]}],
            "policy": {"policy_type": _env("Spectrum")}, "document": {}}
    assert cycle3_findings(gold, {1: "Business Owner's Policy"}, lob="gl") == []


def test_the_checker_reports_what_the_preparation_would_fix():
    gold = {"forms_and_endorsements": [_form("SP 30 18", "10 18")],
            "coverages": [{"coverage_name": _env("Hired Auto"), "form_refs": ["SP 30 18 10 18"]}],
            "policy": {"policy_type": _env("Spectrum")}}
    findings = check_gold(gold, {}, lob="gl")
    assert any(f["rule"] == "R2: a coverage form reference carries its edition" for f in findings)
    assert gold["coverages"][0]["form_refs"] == ["SP 30 18 10 18"]          # the checker never edits


def test_one_template_labelled_two_ways_is_flagged_on_the_odd_one():
    def cov(linked):
        return {"coverage_name": _env("x"), "form_refs": ["A 1"] if linked else []}
    golds = {"hartford_bopgl_policy": {"coverages": [cov(False)] * 4},
             "hartford_bopgl_policy_1": {"coverages": [cov(True)] * 4},
             "other_policy": {"coverages": [cov(False)] * 4}}
    flagged = template_form_ref_findings(golds)
    assert [document for document, _ in flagged] == ["hartford_bopgl_policy"]


def test_shares_far_below_the_templates_best_are_flagged():
    def gold(linked, total):
        return {"coverages": [{"form_refs": ["A 1"] if i < linked else []} for i in range(total)]}
    golds = {"hartford_bopgl_policy": gold(0, 65), "hartford_bopgl_policy_1": gold(0, 66),
             "hartford_bopgl_policy_2": gold(27, 64), "hartford_bopgl_policy_7": gold(30, 69)}
    assert sorted(document for document, _ in template_form_ref_findings(golds)) == [
        "hartford_bopgl_policy", "hartford_bopgl_policy_1"]


def test_a_reference_the_preparation_fixes_is_not_listed_as_missing_from_the_forms_list():
    gold = {"forms_and_endorsements": [_form("SP 30 18", "10 18")],
            "coverages": [{"coverage_name": _env("Hired Auto"), "form_refs": ["SP 30 18 10 18", "ZZ 9"]}],
            "policy": {"policy_type": _env("Spectrum")}}
    missing = [f["detail"] for f in check_gold(gold, {}, lob="gl")
               if f["rule"] == "R2: a coverage form reference not in the forms list"]
    assert missing == ["'ZZ 9'"]


# --- Approved 2026-11-20: scoring and edition reading -------------------------------------------------------------

def test_an_underscore_edition_reads_as_month_and_year():
    from data_pipeline.ingestion.label_rules import edition_mm_yy

    assert edition_mm_yy("01_24") == "01/24"
    assert edition_mm_yy("(10_20)") == "10/20"
    assert edition_mm_yy("1.0") is None


def test_a_coverage_form_reference_with_its_edition_on_one_side_only_scores_as_the_same_form():
    from evaluation.metrics.form_editions import FORM_REF_EDITION, align_form_editions

    label = {"coverages": [{"coverage_name": "Hired Auto", "form_refs": ["SP 30 18 10 18", "CG20180413", "ZZ 1"]}]}
    answer = {"coverages": [{"coverage_name": "Hired Auto", "form_refs": ["SP 30 18", "CG 20 18", "ZZ 2"]}]}
    aligned, moved = align_form_editions(label, answer)
    assert aligned["coverages"][0]["form_refs"] == ["SP 30 18", "CG 20 18", "ZZ 1"]
    assert moved[FORM_REF_EDITION] == 2
    assert label["coverages"][0]["form_refs"][0] == "SP 30 18 10 18"          # the caller's label is not changed


def test_another_form_or_edition_in_a_form_reference_is_not_aligned():
    from evaluation.metrics.form_editions import align_form_editions

    label = {"coverages": [{"form_refs": ["SP 30 18 10 18", "SP 30 19"]}]}
    answer = {"coverages": [{"form_refs": ["SP 30 18 10 20", "SP 30 1"]}]}
    aligned, moved = align_form_editions(label, answer)
    assert aligned["coverages"][0]["form_refs"] == ["SP 30 18 10 18", "SP 30 19"] and not moved


def test_the_decoder_lets_a_date_end_after_a_two_digit_year_and_serving_writes_it_whole():
    """Approved 2026-11-20: v4 read Merchants' printed "07/11/25" as "07/11/2555" in every reading - held to four
    digits, the decoder made it add two - and the date, found on no page, was taken out."""
    import json
    import re

    from common.canonical import compact_schema, expand_answer, with_output_dates
    from common.schemas import load_schema

    schema = json.dumps(compact_schema(load_schema("policy", None, "wc")))
    assert "/[0-9]{4}$" not in schema and "/([0-9]{2}|[0-9]{4})$" in schema
    pattern = re.compile("^(0[1-9]|1[0-2])/(0[1-9]|[12][0-9]|3[01])/([0-9]{2}|[0-9]{4})$")
    assert pattern.match("07/11/25") and pattern.match("07/11/2025") and not pattern.match("07/11/255")
    served = with_output_dates(expand_answer({"policy": {"effective_date": {"v": "07/11/25", "p": [3]}}}))
    assert served["policy"]["effective_date"] == {"raw": "07/11/25", "parsed": "07/11/2025", "page_ref": [3]}


def test_training_targets_still_write_four_digit_years():
    from common.canonical import to_compact_answer

    target = to_compact_answer({"effective_date": {"raw": "07/11/25", "parsed": "07/11/2025", "page_ref": [3]}})
    assert target["effective_date"] == {"r": "07/11/25", "v": "07/11/2025", "p": [3]}
