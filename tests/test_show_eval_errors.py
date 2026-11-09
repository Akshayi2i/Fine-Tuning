"""scripts/show_eval_errors.py: where a release's errors are, from its stored report."""

from __future__ import annotations

import json

from scripts.show_eval_errors import breakdown, collect, field_pattern, invented, main, render

RECORDS = [
    {"source_id": "policy_0012", "field_path": "coverages[0].limit", "expected": "1,000,000", "got": "2,000,000",
     "error_class": "other wrong value", "modality_mode": "image_only", "is_scanned": False, "lob": "gl"},
    {"source_id": "policy_0012", "field_path": "coverages[1].limit", "expected": "500,000", "got": None,
     "error_class": "left empty", "modality_mode": "image_only", "is_scanned": False, "lob": "gl"},
    {"source_id": "policy_0065", "field_path": "policy.cancellation_notice_days", "expected": None, "got": "30",
     "error_class": "invented", "modality_mode": "ocr_plus_image", "is_scanned": True, "lob": "wc"},
]
REPORT = {"policy": {"all": {"metrics": {"field_exact_match": 0.53, "hallucination_rate": 0.01},
                             "error_records": RECORDS}}}


def test_records_and_metrics_are_found_anywhere_in_the_report():
    records, metrics = collect(REPORT)
    assert records == RECORDS
    assert metrics == {"field_exact_match": 0.53, "hallucination_rate": 0.01}


def test_rows_are_unnumbered_so_a_field_is_one_field():
    assert field_pattern("coverages[3].limit") == "coverages[].limit"


def test_the_breakdown_totals_by_class_line_and_document():
    result = breakdown(RECORDS)
    assert result["total"] == 3
    assert result["by_class"] == {"other wrong value": 1, "left empty": 1, "invented": 1}
    assert result["by_line"]["gl"] == {"other wrong value": 1, "left empty": 1}
    assert result["by_scan"]["True"] == {"invented": 1}
    assert result["by_document"]["policy_0012"] == {"other wrong value": 1, "left empty": 1}
    assert result["top_fields"][0] == {"field": "coverages[].limit", "errors": 2,
                                       "by_class": {"other wrong value": 1, "left empty": 1}}


def test_invented_values_are_listed_with_what_the_model_wrote():
    assert [r["source_id"] for r in invented(RECORDS)] == ["policy_0065"]
    text = render(RECORDS, {"hallucination_rate": 0.01})
    assert "Invented values: 1" in text and "policy.cancellation_notice_days" in text and "wrote 30" in text
    assert "hallucination_rate" in text


def test_a_report_without_errors_says_so():
    assert "no error records" in render([], {})


def test_the_script_reads_a_downloaded_report(tmp_path, capsys):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(REPORT), encoding="utf-8")
    assert main(["--file", str(path)]) == 0
    assert "3 failing field(s) in all" in capsys.readouterr().out
    assert main(["--file", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["invented"][0]["source_id"] == "policy_0065"


def test_a_document_in_several_subsets_is_counted_once():
    """2026-11-09: an image-only scanned long policy is in four subsets, each with a copy of its failing fields, and
    every copy was counted - its errors four times, an OCR document's three times."""
    record = {"source_id": "policy_0003", "field_path": "coverages[0].coverage_name", "error_class": "omission"}
    report = {"subsets": [{"subset": name, "error_records": [dict(record)]}
                          for name in ("full", "image_only", "scanned", "long_policy")]}
    records, _metrics = collect(report)
    assert records == [record]
