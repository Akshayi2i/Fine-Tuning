"""scripts/sort_error_causes.py: each failing field sorted by whose it is - the model's, the gold's or the scoring's.

One case per cause, built the way a run's files hold them (2026-11-10, before any change to the scorer).
"""

from __future__ import annotations

import json

from scripts.sort_error_causes import (
    ALIGNMENT,
    GOLD_GAP,
    GOLD_NOT_PRINTED,
    MODEL,
    PLACEMENT,
    RULES,
    UNCHECKED,
    Document,
    main,
    sort_record,
)

PAGES = {
    1: "BUSINESSOWNERS DECLARATIONS Policy Number BP-1001 Business Personal Property Limit $110,000 "
       "Damages to premises rented to you $300,000 Forms SP 30 30 10 18",
    2: "PREMIUM CREDIT APPLICATION\nApplicant information\nHired Auto Liability $1,000,000",
    3: "Each Occurrence Limit $1,000,000 General Aggregate $2,000,000",
}


def _env(value, *pages):
    return {"raw": str(value), "parsed": value, "page_ref": list(pages)}


def _coverage(name, code=None, *limits, pages=(1,)):
    row = {"coverage_name": _env(name, *pages), "coverage_code": code}
    if limits:
        row["limits"] = [{"limit_type": kind, "amount": _env(amount, *pages)} for kind, amount in limits]
    return row


def _doc(gold, answer, cut=()):
    return Document(gold, answer, PAGES, set(cut))


def _record(kind, path, expected, got):
    return {"source_id": "policy_0003", "field_path": path, "expected": expected, "got": got, "error_class": kind,
            "modality_mode": "image_only"}


GOLD = {"policy": {"policy_number": _env("BP-1001", 1)},
        "coverages": [_coverage("Business Personal Property", "BOPGL_BPP", ("per_occurrence", 110000.0))]}


def test_a_value_the_document_prints_and_the_gold_lacks_is_a_gold_gap():
    answer = {"coverages": [_coverage("Damages to premises rented to you")]}
    cause, _why = sort_record(_record("invented", "coverages[1].coverage_name", None,
                                      "Damages to premises rented to you"), _doc(GOLD, answer))
    assert cause == GOLD_GAP


def test_a_code_in_a_row_the_gold_lacks_but_the_document_prints_goes_with_its_row():
    answer = {"coverages": [_coverage("Damages to premises rented to you", "BOPGL_DAMAGE_TO_PREMISES_RENTED")]}
    cause, why = sort_record(_record("invented", "coverages[1].coverage_code", None,
                                     "BOPGL_DAMAGE_TO_PREMISES_RENTED"), _doc(GOLD, answer))
    assert cause == GOLD_GAP and "row" in why


def test_the_same_row_under_another_name_or_written_another_way_is_alignment():
    answer = {"coverages": [_coverage("BPP", "BOPGL_BPP", ("per_occurrence", 110000.0))]}
    cause, _why = sort_record(_record("invented", "coverages[0].coverage_name", None, "BPP"), _doc(GOLD, answer))
    assert cause == ALIGNMENT
    cause, _why = sort_record(_record("schema_reasoning", "policy.policy_number", "BP-1001", "bp 1001"),
                              _doc(GOLD, answer))
    assert cause == ALIGNMENT


def test_a_gold_value_the_answer_holds_in_another_field_is_placement():
    answer = {"additional_fields": [{"label": "Policy No", "value": _env("BP-1001", 1)}]}
    cause, why = sort_record(_record("omission", "policy.policy_number", "BP-1001", None), _doc(GOLD, answer))
    assert cause == PLACEMENT and "additional_fields" in why


def test_rows_the_labelling_rules_keep_out_of_a_gold_are_the_rules():
    answer = {"coverages": [_coverage("Hired Auto Liability", None, ("combined_single", 1000000.0), pages=(2,))]}
    cause, why = sort_record(_record("invented", "coverages[1].coverage_name", None, "Hired Auto Liability"),
                             _doc(GOLD, answer))
    assert cause == RULES and "rule 5" in why


def test_a_gold_value_printed_nowhere_is_a_candidate_gold_error():
    gold = {"policy": {"policy_number": _env("PX-999", 1)}}
    cause, _why = sort_record(_record("omission", "policy.policy_number", "PX-999", None), _doc(gold, {}))
    assert cause == GOLD_NOT_PRINTED


def test_the_models_own_errors_say_why():
    cases = [
        (_record("omission", "coverages[0].limits[0].amount", 110000.0, None), {}, (1,), "cut off"),
        (_record("omission", "coverages[0].limits[0].amount", 110000.0, None), {}, (), "missed a printed value"),
        (_record("invented", "coverages[1].coverage_name", None, "Flood Coverage Extension"),
         {"coverages": [_coverage("Flood Coverage Extension")]}, (), "printed nowhere"),
        (_record("schema_reasoning", "policy.policy_number", "BP-1001", "BP-7777"), {}, (), "another value"),
    ]
    for record, answer, cut, why in cases:
        cause, said = sort_record(record, _doc(GOLD, answer, cut))
        assert cause == MODEL and why in said, (record, said)


def test_a_code_that_cannot_be_looked_up_in_a_row_the_document_does_not_print_is_unchecked():
    answer = {"coverages": [_coverage("Flood Coverage Extension", "X_FLOOD")]}
    cause, _why = sort_record(_record("invented", "coverages[1].coverage_code", None, "X_FLOOD"), _doc(GOLD, answer))
    assert cause == UNCHECKED


def test_the_command_reads_a_runs_files_and_reports_by_cause(tmp_path, capsys):
    errors = {"model_version": "v3", "subsets": [
        {"subset": "full", "error_records": [
            _record("invented", "coverages[1].coverage_name", None, "Damages to premises rented to you"),
            _record("omission", "policy.policy_number", "PX-999", None)]},
        {"subset": "image_only", "error_records": [   # the same failures again: counted once
            _record("invented", "coverages[1].coverage_name", None, "Damages to premises rented to you")]}]}
    gold = {"policy": {"policy_number": _env("PX-999", 1)}}
    answer = {"coverages": [_coverage("Damages to premises rented to you")]}
    row = {"source_id": "policy_0003", "modality_mode": "image_only", "doc_type": "policy", "lob": "bopgl"}
    (tmp_path / "e.json").write_text(json.dumps(errors), encoding="utf-8")
    (tmp_path / "a.jsonl").write_text(json.dumps({"row": row, "gold": gold, "answer": answer, "error": None}) + "\n",
                                      encoding="utf-8")
    (tmp_path / "p.json").write_text(json.dumps({"policy_0003": {str(k): v for k, v in PAGES.items()}}),
                                     encoding="utf-8")
    assert main(["--errors", str(tmp_path / "e.json"), "--answers", str(tmp_path / "a.jsonl"),
                 "--pages", str(tmp_path / "p.json"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["failing_fields"] == 2
    assert report["by_class"][GOLD_GAP] == 1 and report["by_class"][GOLD_NOT_PRINTED] == 1
    assert report["by_document"]["policy_0003"] == {GOLD_GAP: 1, GOLD_NOT_PRINTED: 1}


def test_window_rows_are_merged_per_document_and_mode_and_cut_off_pages_are_kept():
    from common.schema_sections import group_names
    from scripts.sort_error_causes import documents

    group = group_names("gl")[0]
    base = {"source_id": "policy_0003", "modality_mode": "image_only", "doc_type": "policy", "lob": "gl",
            "sections": group, "window_leads": False, "task": "policy_declarations"}
    answers = [
        {"row": {**base, "window_index": 0, "window_pages": [1, 2]}, "gold": {"policy": {"policy_number": _env("BP-1001", 1)}},
         "answer": {"policy": {"policy_number": _env("BP-1001", 1)}}, "error": None},
        {"row": {**base, "window_index": 1, "window_pages": [3]}, "gold": {}, "answer": {},
         "error": "stopped at the token limit before the JSON closed", "unparsed_text": '{"policy": {'},
    ]
    docs = documents(answers, {"policy_0003": {str(k): v for k, v in PAGES.items()}})
    doc = docs[("policy_0003", "image_only")]
    assert doc.cut_pages == {3}
    assert doc.gold["policy"]["policy_number"]["parsed"] == "BP-1001"
    assert 2 in doc.application and 1 not in doc.application
