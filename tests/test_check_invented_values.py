"""scripts/check_invented_values.py: which of the gate's invented values the document prints."""

from __future__ import annotations

import json

from scripts.check_invented_values import classify, list_sizes, main, render, render_sizes, summarise

PAGES = {"p1": {1: "Commercial General Liability. Forms: CG 00 01 04 13, SL 30 32 06 21. Each Occurrence $1,000,000",
                2: "Damages to premises rented to you"},
         "p2": {1: ""}}


def _r(source, path, got, cls="invented"):
    return {"source_id": source, "field_path": path, "expected": None, "got": got, "error_class": cls}


def test_a_value_the_document_prints_is_not_a_hallucination():
    [row] = classify([_r("p1", "coverages[0].coverage_name", "DAMAGES TO PREMISES RENTED TO YOU")], PAGES)
    assert row["verdict"] == "printed" and row["missing"] == []


def test_a_value_printed_nowhere_is_named_with_what_was_missing():
    [row] = classify([_r("p1", "coverages[0].coverage_name", "Hired Auto Liability")], PAGES)
    assert row["verdict"] == "not printed" and row["missing"] == ["Hired Auto Liability"]


def test_a_list_is_checked_item_by_item_and_any_missing_item_makes_it_not_printed():
    rows = classify([_r("p1", "coverages[0].form_refs", ["CG 00 01 04 13", "ZZ 99 99 99 99"]),
                     _r("p1", "coverages[1].form_refs", ["SL 30 32 06 21"])], PAGES)
    assert [r["verdict"] for r in rows] == ["not printed", "printed"]
    assert rows[0]["missing"] == ["ZZ 99 99 99 99"]


def test_a_figure_is_looked_up_as_a_figure():
    rows = classify([_r("p1", "limits[0].amount", "1000000"), _r("p1", "limits[1].amount", "2000000")], PAGES)
    assert [r["verdict"] for r in rows] == ["printed", "not printed"]


def test_a_document_with_no_text_cannot_be_judged_and_a_tiny_value_is_unchecked():
    rows = classify([_r("p2", "x", "Hartford"), _r("p1", "y", "A"), _r("nowhere", "z", "Hartford")], PAGES)
    assert [r["verdict"] for r in rows] == ["no text", "unchecked", "no text"]


def test_the_summary_counts_by_verdict_field_and_document():
    rows = classify([_r("p1", "coverages[0].coverage_name", "Hired Auto Liability"),
                     _r("p1", "coverages[1].coverage_name", "Damages to premises rented to you")], PAGES)
    summary = summarise(rows)
    assert summary["by_verdict"] == {"not printed": 1, "printed": 1}
    assert summary["by_field"]["coverages[].coverage_name"] == {"not printed": 1, "printed": 1}
    text = render(rows)
    assert "printed nowhere" in text and "Hired Auto Liability" in text


def test_the_script_reads_a_downloaded_report_and_pages(tmp_path, capsys):
    report = {"policy": {"all": {"error_records": [_r("p1", "coverages[0].coverage_name", "Hired Auto Liability"),
                                                   _r("p1", "coverages[0].limit", "5", cls="omission")]}}}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "pages.json").write_text(json.dumps(PAGES), encoding="utf-8")
    assert main(["--file", str(tmp_path / "report.json"), "--pages", str(tmp_path / "pages.json"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["by_verdict"] == {"not printed": 1}


def test_a_gold_list_shorter_than_what_the_document_prints_shows_in_the_sizes():
    pages = {"p1": {1: "Forms CG 00 01 04 13 and SL 30 32 06 21 and IL 00 17 11 98. Coverage Hired Auto"}}
    rows = classify([_r("p1", "forms_and_endorsements[0].form_number", "CG 00 01 04 13"),
                     _r("p1", "forms_and_endorsements[1].form_number", "SL 30 32 06 21"),
                     _r("p1", "forms_and_endorsements[1].form_number", "SL 30 32 06 21"),   # another reading's
                     _r("p1", "forms_and_endorsements[2].form_number", "ZZ 99 99 99 99")], pages)
    sizes = list_sizes(rows, {"p1": {"forms_and_endorsements": [{"form_number": "IL 00 17 11 98"}], "coverages": []}})
    by_section = {r["section"]: r for r in sizes}
    assert by_section["forms_and_endorsements"] == {"source_id": "p1", "section": "forms_and_endorsements",
                                                    "gold_rows": 1, "printed_but_not_in_gold": 2}
    assert by_section["coverages"]["gold_rows"] == 0
    assert "gold rows" in render_sizes(sizes) and "p1" in render_sizes(sizes)
