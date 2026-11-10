"""scripts/loop_report.py and scripts/answer_list_lengths.py: the evidence for why v3 loops (2026-11-10). Read-only."""

from __future__ import annotations

import json

from scripts.answer_list_lengths import measure
from scripts.loop_report import exact_loop_start, report_one, row_loop_start


def _entry(label: str) -> str:
    return json.dumps({"label": label, "value": {"v": "1", "p": [2]}})


def test_an_exact_loop_is_found_where_it_starts_and_in_its_list():
    head = '{"coverages": [{"coverage_name": {"v": "Building", "p": [1]}}], "additional_fields": ['
    real = ", ".join(_entry(f"Real {n}") for n in range(5))
    cycle = ", " + ", ".join(_entry(f"Loop {n}") for n in range(3))
    text = head + real + cycle * 40
    loop = report_one({"row": {"source_id": "policy_0018", "task": "policy_schedule", "window_pages": [1, 2]},
                       "gold": {"additional_fields": [{"label": "x"}] * 12}, "unparsed_text": text})
    assert loop["kind"] == "exact" and loop["list"] == "additional_fields" and loop["gold_entries_in_list"] == 12
    assert abs(loop["start"] - len(head + real)) <= loop["period"]          # to within one repeated passage
    assert loop["repeats"] >= 39


def test_a_row_written_again_with_another_field_changing_is_found_at_its_first_repeat():
    forms = [json.dumps({"form_number": {"v": f"AP {n}", "p": [1]}}) for n in range(5)]
    loop = [json.dumps({"form_number": {"v": "APDEC 0919", "p": [1]}, "title": {"v": f"Title {n}", "p": [1]}})
            for n in range(120)]
    text = '{"forms_and_endorsements": [' + ", ".join(forms + loop)
    found = report_one({"row": {"source_id": "policy_0071"}, "gold": {"forms_and_endorsements": [{}, {}]},
                        "unparsed_text": text})
    assert found["kind"] == "row" and found["value"] == "APDEC 0919" and found["list"] == "forms_and_endorsements"
    assert found["start"] == text.index("APDEC 0919") - len('"form_number": {"v": "')
    assert row_loop_start(text, "form_number", "APDEC 0919") == found["start"]


def test_an_answer_that_parsed_is_not_reported():
    assert report_one({"row": {}, "unparsed_text": None}) is None


def test_the_exact_start_is_the_first_character_of_the_repeating_run():
    assert exact_loop_start("xyz" + "ab" * 10, 2) == 3


def _row(answer: dict, source: str = "policy_0001") -> dict:
    return {"source_id": source, "task": "policy_schedule", "window_pages": [1],
            "messages": [{"role": "user", "content": []}, {"role": "assistant", "content": json.dumps(answer)}]}


def test_list_lengths_and_rows_that_name_a_row_twice_are_counted():
    rows = [
        _row({"additional_fields": [{"label": f"L{n}"} for n in range(30)]}),
        _row({"additional_fields": [{"label": "L1"}]}),
        _row({"forms_and_endorsements": [{"form_number": {"v": "CG 00 01"}}, {"form_number": {"v": "CG 00 01"}}]},
             source="policy_0071"),
        {"source_id": "x", "messages": [{"role": "assistant", "content": "not json"}]},
    ]
    report = measure(rows)
    extra = report["lists"]["additional_fields"]
    assert extra["rows"] == 2 and extra["longest"] == 30 and report["unreadable_answers"] == 1
    assert report["rows_naming_a_row_twice"] == 1
    assert report["repeated"][0]["names_twice_or_more"] == {"form_number=CG 00 01": 2}
