"""The audit's label-consistency check: each kind of value written one way across
the labels of a folder (decision 2026-10-24).

A field written two ways teaches the model both, and it then writes one at
random. The check reports every split as a warning, never a blocker: the user
chooses the convention.
"""

from __future__ import annotations

import copy
import csv
import json

import pytest

from data_pipeline.audit import (
    LABEL_CONSISTENCY,
    DocumentAudit,
    _kind_of,
    _placeholder,
    label_consistency,
    label_facts,
)


def _value(raw, parsed):
    return {"raw": raw, "parsed": parsed, "confidence": {"score": 1.0, "source": "audit"},
            "page_ref": [1], "flagged": False}


def _found(labels: dict[str, dict], lob="homeowners", sources: dict[str, str] | None = None):
    """label_consistency over labels of one line, by folder name; no PDFs."""
    docs = [DocumentAudit(folder=name, template_id=(sources or {}).get(name), labels=label_facts(label, lob))
            for name, label in labels.items()]
    return label_consistency(docs)


def _by_kind(found) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in found:
        out.setdefault(f.kind, set()).add(f.path)
    return out


# --------------------------------------------------------------------------
# What a field's kind is
# --------------------------------------------------------------------------


def test_a_field_takes_the_kind_calibration_reads_and_its_declared_type():
    assert _kind_of("premium.total", True).kind == "money"                       # a MoneyValue
    assert _kind_of("coverages[].deductibles[].percentage", True).kind == "percent"
    assert _kind_of("buildings[].sprinklered", True).kind == "boolean"           # a BoolValue
    assert _kind_of("document.transaction_type", True).kind == "enum"
    edition = _kind_of("forms_and_endorsements[].edition_date", True)
    assert (edition.kind, edition.declared) == ("date", None)                     # a date by its name
    assert _kind_of("forms_and_endorsements[].form_type", True).kind == "enum"   # *_type
    assert _kind_of("named_insured.primary_name", True).kind is None             # a name: text
    # A self-contained line declares nothing: the field's name decides.
    assert _kind_of("premium.total_policy_premium", False).kind == "money"
    assert _kind_of("policy.policy_type", False).kind == "enum"


@pytest.mark.parametrize("value,held", [
    (" n/a ", True), ("N/A", True), ("NA", True), ("None", True), ("Not Applicable", True), ("-", True),
    ("--", True), ("tbd", True), ("UNKNOWN", True), ("Nil", True),
    ("0", False), ("Nothing", False), ("---", False), (None, False), (0, False),
])
def test_placeholders_are_the_listed_words_trimmed_case_aside(value, held):
    assert _placeholder(value) is held


# --------------------------------------------------------------------------
# 1. One format per field
# --------------------------------------------------------------------------


def test_a_date_kept_as_written_splits_where_one_the_build_rewrites_does_not():
    """An edition date is a plain value: it trains as written ("05/11" and "05 11"
    are two answers). A policy date is a DateValue: written MM/DD/YYYY either way."""
    found = _found({
        "a": {"forms_and_endorsements": [{"edition_date": _value("05/11", "05/11")}],
              "policy": {"effective_date": _value("04/01/2026", "04/01/2026")}},
        "b": {"forms_and_endorsements": [{"edition_date": _value("05 11", "05 11")}],
              "policy": {"effective_date": _value("04/01/2026", "04/01/2026")}},
    })
    [split] = found
    assert (split.kind, split.path, split.folder, split.severity) == (
        "format", "forms_and_endorsements[].edition_date", ".", "warning")
    assert split.check == LABEL_CONSISTENCY
    assert 'text "99/99" in 1 label(s) (e.g. a)' in split.detail
    assert 'text "99 99" in 1 label(s) (e.g. b)' in split.detail


def test_dates_the_build_can_read_train_in_one_format_on_a_self_contained_line():
    found = _found({"a": {"policy": {"effective_date": _value("04/01/2026", "2026-04-01")}},
                    "b": {"policy": {"effective_date": _value("04/01/2026", "04/01/2026")}}},
                   lob="workers_comp")
    assert found == []


def test_whole_numbers_written_two_ways_split_and_a_decimal_beside_them_does_not():
    split = _found({"a": {"premium": {"total": _value("$1,250", 1250)}},
                    "b": {"premium": {"total": _value("$1,250.00", 1250.0)}}})
    assert _by_kind(split) == {"format": {"premium.total"}}
    assert "whole number as 1250 in 1 label(s)" in split[0].detail
    assert "whole number as 1250.0 in 1 label(s)" in split[0].detail
    one_way = _found({"a": {"premium": {"total": _value("$1,250", 1250)}},
                      "b": {"premium": {"total": _value("$1,250.50", 1250.5)}}})
    assert one_way == []


def test_a_number_beside_text_splits_on_a_line_whose_schema_holds_either():
    found = _found({"a": {"premium": {"total_policy_premium": _value("$47,250.00", 47250.0)}},
                    "b": {"premium": {"total_policy_premium": _value("$47,250.00", "47,250.00")}}},
                   lob="workers_comp")
    assert _by_kind(found) == {"format": {"premium.total_policy_premium"}}
    assert 'text "99,999.99"' in found[0].detail


def test_an_enum_value_spelt_two_ways_is_one_split_and_two_values_are_none():
    found = _found({
        "a": {"forms_and_endorsements": [{"form_type": _value("Endorsement", "Endorsement")},
                                         {"form_type": _value("Policy", "policy")}]},
        "b": {"forms_and_endorsements": [{"form_type": _value("ENDORSEMENT", "endorsement")}]},
        "c": {"forms_and_endorsements": [{"form_type": _value("Policy", "policy")}]},
    })
    [split] = found
    assert split.path == "forms_and_endorsements[].form_type"
    assert "spells one value 2 ways" in split.detail
    assert '"Endorsement" in 1 label(s) (e.g. a)' in split.detail
    assert '"endorsement" in 1 label(s) (e.g. b)' in split.detail
    assert "policy" not in split.detail


def test_a_value_another_check_reports_is_left_to_that_report():
    """The schema refuses a MoneyValue written as text and an enum outside its
    list (blockers); formats.csv names an unreadable date and an amount that is
    not a number. None of them is repeated as a split."""
    refused = _found({"a": {"premium": {"total": _value("$1,250", 1250)},
                            "document": {"transaction_type": _value("Renewal", "renewal")}},
                      "b": {"premium": {"total": _value("$1,250", "1,250")},
                            "document": {"transaction_type": _value("Renewal", "Renewal")}}})
    assert refused == []
    formats = _found({"a": {"policy": {"effective_date": _value("04/01/2026", "2026-04-01")},
                            "premium": {"total_policy_premium": _value("$47,250.00", 47250.0)}},
                      "b": {"policy": {"effective_date": _value("sometime", "sometime in April")},
                            "premium": {"total_policy_premium": _value("$47,250.00", "$47,250.00")}}},
                     lob="workers_comp")
    assert formats == []


# --------------------------------------------------------------------------
# 2. Placeholders against nulls
# --------------------------------------------------------------------------


def test_a_placeholder_beside_a_null_where_its_row_is_there():
    found = _found({
        "a": {"coverages": [{"deductibles": [{"amount": _value("N/A", None)}]}]},
        "b": {"coverages": [{"deductibles": [{"amount": _value("n/a", None)}]}]},
        "c": {"coverages": [{"deductibles": [{"deductible_type": _value("Flat", "flat")}]}]},  # no amount
        "d": {"coverages": [{"coverage_name": _value("Dwelling", "Dwelling")}]},           # no deductible rows
        "e": {"coverages": [{"deductibles": [{"amount": _value("$500", 500)}]}]},
    })
    [finding] = [f for f in found if f.kind == "placeholder"]
    assert finding.path == "coverages[].deductibles[].amount" and finding.folder == "."
    assert "a placeholder in 2 label(s) ('N/A' 1, 'n/a' 1; e.g. a, b)" in finding.detail
    assert "left null in 1 (e.g. c)" in finding.detail          # d has no row to leave it in


def test_an_explicit_null_is_a_null_and_a_placeholder_alone_is_no_split():
    found = _found({"a": {"producer": {"producer_code": _value("TBD", "TBD")}},
                    "b": {"producer": {"producer_code": _value(None, None)}}}, lob="workers_comp")
    assert _by_kind(found) == {"placeholder": {"producer.producer_code"}}
    alone = _found({"a": {"producer": {"producer_code": _value("TBD", "TBD")}},
                    "b": {"producer": {"producer_code": _value("TBD", "TBD")}}}, lob="workers_comp")
    assert alone == []


# --------------------------------------------------------------------------
# 3. raw against parsed
# --------------------------------------------------------------------------


def test_raw_that_reads_as_another_date_or_amount_is_reported_per_value():
    found = _found({
        "a": {"policy": {"effective_date": _value("Jan 1, 2026", "02/01/2026")},
              "billing": {"amount_due": _value("$1,250", 125.0)}},
        "b": {"policy": {"effective_date": _value("January 1, 2026", "01/01/2026")},
              "billing": {"amount_due": _value("$1,250.00", 1250)}},
    })
    contradictions = {(f.folder, f.path) for f in found if f.kind == "raw_vs_parsed"}
    assert contradictions == {("a", "policy.effective_date"), ("a", "billing.amount_due")}
    detail = next(f.detail for f in found if f.path == "policy.effective_date")
    assert detail == ("raw_vs_parsed: policy.effective_date: raw 'Jan 1, 2026' reads 2026-01-01, "
                      "parsed '02/01/2026' reads 2026-02-01")


def test_a_row_value_is_named_by_its_index_and_amounts_named_so_are_left_to_formats():
    found = _found({"a": {"coverages": [{"limits": [{"amount": _value("$300,000", 30000.0)}]}],
                          "vehicles": [{"cost_new": _value("$5", 5)}, {"cost_new": _value("$12", 21)}]}},
                   lob="personal_auto")
    # coverages[0].limits[0].amount is named like an amount: formats.csv compares it.
    assert [f.detail for f in found] == [
        "raw_vs_parsed: vehicles[1].cost_new: raw '$12' reads 12.0, parsed is 21"]
    assert found[0].path == "vehicles[].cost_new"


# --------------------------------------------------------------------------
# What is read, and what is named
# --------------------------------------------------------------------------


def test_the_full_text_notes_and_system_fields_are_not_read():
    facts = label_facts({
        "text_sections": {"s1": {"edition_date": _value("05 11", "05 11")}},
        "fideon:provenance": {"note": _value("N/A", "N/A")},
        "document": {"page_count": _value("3", 3)},
        "premium": {"total": _value("$5", 5)},
    }, "homeowners")
    assert set(facts.shapes) == {("premium.total", "money")}
    assert facts.placeholders == {}


def test_examples_come_from_different_sources_first():
    labels = {name: {"premium": {"total": _value("$5", 5)}} for name in ("s1_a", "s1_b", "s1_c", "s2_a")}
    labels["z"] = {"premium": {"total": _value("$5", 5.0)}}
    [split] = _found(labels, sources={"s1_a": "s1", "s1_b": "s1", "s1_c": "s1", "s2_a": "s2"})
    assert "in 4 label(s) (e.g. s1_a, s2_a, s1_b)" in split.detail


def test_a_label_that_is_not_an_object_has_no_part():
    assert label_facts(["not", "a", "label"], "homeowners") is None
    assert _found({}) == []


# --------------------------------------------------------------------------
# The folder: warnings.csv, summary, the report
# --------------------------------------------------------------------------


def test_a_consistent_folder_has_no_label_consistency_warning(tmp_path):
    pytest.importorskip("pymupdf")
    from data_pipeline.audit import audit_folder, render, summary
    from tests.test_data_audit import PAGE_1, _document

    for i in range(2):
        _document(tmp_path, f"doc{i}", pages=[PAGE_1 + [f"copy {i}"]])
    report = audit_folder(tmp_path)
    assert not [f for f in report.findings if f.check == LABEL_CONSISTENCY]
    facts = summary(report)
    assert facts["label_consistency"] == {"format": 0, "placeholder": 0, "raw_vs_parsed": 0}
    text = render(report, facts)
    assert "## Label consistency: 0 warning(s)" in text
    assert "- one format per field: 0 split(s) in 0 field(s)" in text


def test_each_kind_of_split_is_a_warning_in_the_report(tmp_path):
    pytest.importorskip("pymupdf")
    from data_pipeline.audit import audit_folder, write_report
    from tests.test_data_audit import GOLDEN, PAGE_1, _document

    a = copy.deepcopy(GOLDEN)
    a["policy"]["policy_type"] = _value("Standard", "Standard")
    b = copy.deepcopy(GOLDEN)
    b["policy"]["policy_type"] = _value("STANDARD", "STANDARD")             # an enum spelt another way
    b["premium"]["total_policy_premium"]["parsed"] = 47250                   # 47250 beside 47250.0
    b["producer"]["producer_code"] = _value("N/A", "N/A")                   # a has no producer code
    c = copy.deepcopy(GOLDEN)
    c["policy"]["effective_date"]["parsed"] = "2026-01-04"                   # raw says 04/01/2026
    for name, golden in (("a", a), ("b", b), ("c", c)):
        _document(tmp_path / "data", name, golden=golden, pages=[PAGE_1 + [name]])
    report = audit_folder(tmp_path / "data")
    found = [f for f in report.findings if f.check == LABEL_CONSISTENCY]
    assert _by_kind(found) == {
        "format": {"policy.policy_type", "premium.total_policy_premium"},
        "placeholder": {"producer.producer_code"},
        "raw_vs_parsed": {"policy.effective_date"},
    }
    assert all(f.severity == "warning" for f in found)
    assert not report.blockers and all(d.importable for d in report.documents)   # nothing here blocks

    facts = write_report(report, tmp_path / "out")
    assert facts["label_consistency"] == {"format": 2, "placeholder": 1, "raw_vs_parsed": 1}
    assert facts["label_consistency_fields"]["format"] == ["policy.policy_type", "premium.total_policy_premium"]
    with (tmp_path / "out" / "warnings.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["check"] == LABEL_CONSISTENCY]
    assert len(rows) == 4
    assert {"folder": "c", "check": LABEL_CONSISTENCY,
            "detail": "raw_vs_parsed: policy.effective_date: raw '04/01/2026' reads 2026-04-01, "
                      "parsed '2026-01-04' reads 2026-01-04"} in rows
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["label_consistency"] == {
        "format": 2, "placeholder": 1, "raw_vs_parsed": 1}
    text = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Label consistency: 4 warning(s)" in text
    assert "- one format per field: 2 split(s) in 2 field(s): policy.policy_type, premium.total_policy_premium" in text
    assert "- a placeholder beside a null: 1 field(s): producer.producer_code" in text
    assert "- raw against parsed (dates, amounts): 1 value(s) in 1 field(s): policy.effective_date" in text
