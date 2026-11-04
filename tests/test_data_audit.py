"""The pre-upload data audit, on synthetic document folders with real PDFs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data_pipeline.audit import (
    appears,
    application_pages,
    audit_folder,
    main,
    normalise,
    rows_only_on_application_pages,
    summary,
)

pymupdf = pytest.importorskip("pymupdf")

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "golden"
GOLDEN = json.loads((FIXTURES / "policy_0001.golden.json").read_text(encoding="utf-8"))
#: The fixture gold is a workers' comp declarations page written in a
#: self-contained line's shape; workers_comp composes the common model now, so
#: the folder is filed under a line whose schema is still self-contained.
META = {"lob": "equipment_breakdown", "field_provenance": {"policy.policy_number": "Policy No."},
        # Every gold needs a review record to import (decision 2026-10-22).
        "reviewer_id": "a.reviewer", "review_date": "2026-10-20"}

#: What page 1 of the fixture policy prints.
PAGE_1 = [
    "WORKERS COMPENSATION POLICY DECLARATIONS",
    "Producer: Hanover Risk Partners",
    "Policy No. WC-8842317-01   Effective 04/01/2026   Expiration 04/01/2027",
    "Applicant: Rivera Fabrication LLC",
    "1420 Foundry Road, Toledo, OH 43604",
    "Total Premium $47,250.00",
]


def _pdf(path: Path, pages: list[list[str]]) -> None:
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=612, height=792)
        for i, line in enumerate(lines):
            page.insert_text((50, 60 + 16 * i), line, fontsize=9)
    doc.save(path)


def _document(root: Path, name: str, *, pages=None, golden=None, meta=None, pdfs=1) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    for i in range(pdfs):
        _pdf(folder / f"policy{i}.pdf", pages if pages is not None else [PAGE_1, ["endorsements page"] * 3])
    if golden is not False:
        (folder / "golden.json").write_text(json.dumps(golden or GOLDEN), encoding="utf-8")
    if meta is not False:
        (folder / "metadata.json").write_text(json.dumps(meta or META), encoding="utf-8")
    return folder


def _results(report, folder):
    return {v.path: v.result for d in report.documents if d.folder == folder for v in d.values}


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def test_matching_ignores_case_punctuation_and_line_breaks():
    assert normalise("$47,250.00") == "47 250 00"
    assert appears("Rivera Fabrication LLC", "applicant:\nRIVERA   Fabrication, LLC")
    assert appears("1420 Foundry Road", "Foundry Road 1420")          # words across columns
    assert not appears("Rivera Fabrication LLC", "Meridian Holdings")


# --------------------------------------------------------------------------
# A clean document
# --------------------------------------------------------------------------


def test_a_code_is_found_whole_never_from_pieces_printed_apart(tmp_path):
    # Page 1 prints "WC-8842317-01" and, in its dates, "2026": a gold policy
    # number made of those pieces is printed nowhere.
    golden = copy.deepcopy(GOLDEN)
    golden["policy"]["policy_number"]["raw"] = "WC-8842317-2026"
    _document(tmp_path, "pieces", golden=golden)
    assert _results(audit_folder(tmp_path), "pieces")["policy.policy_number"] == "not_found"


def test_a_correct_document_has_no_blockers_and_its_values_are_found(tmp_path):
    _document(tmp_path, "good")
    report = audit_folder(tmp_path)
    assert not report.blockers, [(f.check, f.detail) for f in report.blockers]
    results = _results(report, "good")
    assert results["policy.policy_number"] == "ok"
    assert results["named_insured.primary_name"] == "ok"
    assert results["premium.total_policy_premium"] == "ok"
    assert report.documents[0].digital


# --------------------------------------------------------------------------
# 1. Structure
# --------------------------------------------------------------------------


def test_structural_problems_are_blockers(tmp_path):
    _document(tmp_path, "two_pdfs", pdfs=2)
    _document(tmp_path, "no_golden", golden=False)
    folder = _document(tmp_path, "bad_json")
    (folder / "golden.json").write_text("{not json", encoding="utf-8")
    report = audit_folder(tmp_path)
    details = {name: [f.detail for f in report.blockers if f.folder == name]
               for name in ("two_pdfs", "no_golden", "bad_json")}
    assert details["two_pdfs"] == ["2 PDF files (exactly one per folder)"]   # and nothing else
    assert any("no golden.json" in d for d in details["no_golden"])
    assert any("does not parse" in d for d in details["bad_json"])


def test_a_duplicate_pdf_is_a_blocker(tmp_path):
    import shutil

    _document(tmp_path, "a")
    _document(tmp_path, "b")
    shutil.copy(tmp_path / "a" / "policy0.pdf", tmp_path / "b" / "policy0.pdf")   # the same file
    report = audit_folder(tmp_path)
    assert any("same PDF as a" in f.detail for f in report.blockers if f.folder == "b")


# --------------------------------------------------------------------------
# 2. Label shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("meta,message", [
    ({}, "no lob"),
    ({"lob": "pet_insurance"}, "no canonical schema"),
])
def test_a_missing_or_unknown_line_is_a_blocker(tmp_path, meta, message):
    _document(tmp_path, "doc", meta=meta or {"field_provenance": {}})
    report = audit_folder(tmp_path)
    assert any(message in f.detail for f in report.blockers)


def test_a_gold_with_no_review_record_is_a_blocker_as_at_import(tmp_path):
    unreviewed = {k: v for k, v in META.items() if k not in ("reviewer_id", "review_date")}
    _document(tmp_path, "doc", meta={**unreviewed, "reviewer_id": "pending"})
    report = audit_folder(tmp_path)
    assert any("no review record" in f.detail and "'pending' names nobody" in f.detail for f in report.blockers)
    assert summary(report)["importable"] == 0


def test_every_line_is_audited_by_default(tmp_path):
    """No --lines: a document of any line with a canonical schema is importable
    (the default was personal lines while scopes existed)."""
    _document(tmp_path, "eb")                                      # equipment_breakdown is commercial
    # A personal line too. The fixture gold does not fit its schema, so only
    # the line filter is asserted for it.
    _document(tmp_path, "home", meta={**META, "lob": "homeowners"}, pages=[PAGE_1 + ["home"]])
    report = audit_folder(tmp_path)
    assert report.lines is None
    assert not [f for f in report.blockers if "outside the lines" in f.detail]
    assert not [f for f in report.blockers if f.folder == "eb"], [f.detail for f in report.blockers]
    assert summary(report)["documents_per_line"] == {"equipment_breakdown": 1}


def test_a_line_outside_lines_is_a_blocker(tmp_path):
    _document(tmp_path, "doc")                       # equipment_breakdown is commercial
    report = audit_folder(tmp_path, lines=frozenset({"homeowners", "personal_auto"}))
    outside = [f.detail for f in report.blockers if "is outside the lines audited" in f.detail]
    assert outside and "homeowners" in outside[0] and "personal_auto" in outside[0]
    assert summary(report)["importable"] == 0


def test_a_line_inside_lines_passes_and_a_package_must_be_wholly_inside(tmp_path):
    _document(tmp_path, "eb")
    _document(tmp_path, "package", meta={**META, "lob": ["equipment_breakdown", "gl"]},
              pages=[PAGE_1 + ["package"]])
    report = audit_folder(tmp_path, lines=frozenset({"equipment_breakdown"}))
    blocked = {f.folder for f in report.blockers if "is outside the lines audited" in f.detail}
    assert blocked == {"package"}                    # one of its lines is not asked for


def test_a_page_ref_beyond_the_pdf_is_a_blocker(tmp_path):
    golden = copy.deepcopy(GOLDEN)
    golden["policy"]["policy_number"]["page_ref"] = [9]
    _document(tmp_path, "doc", golden=golden)
    report = audit_folder(tmp_path)
    assert any("page_ref 9 is not a page" in f.detail for f in report.blockers)


# --------------------------------------------------------------------------
# 3. Values against the PDF
# --------------------------------------------------------------------------


def test_a_value_on_another_page_and_a_value_nowhere_are_reported(tmp_path):
    golden = copy.deepcopy(GOLDEN)
    golden["policy"]["policy_number"]["page_ref"] = [2]              # it is printed on page 1
    golden["named_insured"]["primary_name"]["raw"] = "Riviera Fabrications Inc"   # a typo'd value
    _document(tmp_path, "doc", golden=golden)
    results = _results(audit_folder(tmp_path), "doc")
    assert results["policy.policy_number"] == "wrong_page"
    assert results["named_insured.primary_name"] == "not_found"


def test_a_scanned_page_is_left_for_the_check_after_ocr(tmp_path):
    _document(tmp_path, "scan", pages=[[], []])        # no text layer
    report = audit_folder(tmp_path)
    assert set(_results(report, "scan").values()) <= {"needs_ocr", "skipped_short"}
    assert not report.documents[0].digital


# --------------------------------------------------------------------------
# 4. Formats
# --------------------------------------------------------------------------


def test_format_problems_are_reported(tmp_path):
    golden = copy.deepcopy(GOLDEN)
    golden["policy"]["effective_date"]["parsed"] = "sometime in April"
    golden["premium"]["total_policy_premium"]["parsed"] = 4725.0          # raw says $47,250.00
    _document(tmp_path, "doc", golden=golden)
    formats = {f.check for f in audit_folder(tmp_path).formats}
    assert {"date_format", "amount"} <= formats


def test_a_reversed_policy_period_is_reported(tmp_path):
    golden = copy.deepcopy(GOLDEN)
    golden["policy"]["expiration_date"]["parsed"] = "2025-04-01"
    _document(tmp_path, "doc", golden=golden)
    assert any(f.check == "period" for f in audit_folder(tmp_path).formats)


def test_iso_dates_are_fine_the_build_rewrites_them(tmp_path):
    _document(tmp_path, "doc")                          # the fixture's dates are ISO
    assert not [f for f in audit_folder(tmp_path).formats if f.check == "date_format"]


# --------------------------------------------------------------------------
# 5. Totals, the report, the CLI
# --------------------------------------------------------------------------


def test_totals_and_spot_check(tmp_path):
    for i in range(3):
        _document(tmp_path, f"doc{i}", pages=[PAGE_1 + [f"copy {i}"]])
    report = audit_folder(tmp_path)
    facts = summary(report)
    assert facts["importable"] == 3
    assert facts["documents_per_line"] == {"equipment_breakdown": 3}
    assert facts["lines_too_small_to_measure"] == ["equipment_breakdown"]
    assert not facts["test_meets_freeze_minimum"]
    assert facts["spot_check_documents"] == 1


def test_the_report_is_written_and_blockers_fail_the_command(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _document(data, "good", pages=[PAGE_1 + ["one"]])
    _document(data, "broken", golden=False)
    assert main(["--input", str(data), "--out", str(out)]) == 1
    for name in ("report.md", "summary.json", "blockers.csv", "value_checks.csv",
                 "formats.csv", "spot_check.csv", "warnings.csv"):
        assert (out / name).is_file(), name
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Blockers: 1" in report
    assert "Lines: every line · folders: 2" in report


def test_the_command_audits_the_lines_it_is_given(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    _document(data, "good", pages=[PAGE_1 + ["one"]])                     # equipment breakdown
    assert main(["--input", str(data), "--out", str(out), "--lines", "equipment_breakdown"]) == 0
    assert "Lines: equipment_breakdown · folders: 1" in (out / "report.md").read_text(encoding="utf-8")
    assert main(["--input", str(data), "--out", str(out), "--lines", "homeowners,personal_auto"]) == 1
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Lines: homeowners, personal_auto" in report


def test_the_command_refuses_a_line_with_no_schema(tmp_path, capsys):
    """A misspelt --lines would otherwise block every document without a word
    about why."""
    data, out = tmp_path / "data", tmp_path / "out"
    _document(data, "good")
    assert main(["--input", str(data), "--out", str(out), "--lines", "homeowner"]) == 2
    assert "homeowner" in capsys.readouterr().err
    assert not (out / "report.md").exists()


def test_the_audit_output_is_never_committed():
    ignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert "data/audit_report/" in ignore


# --------------------------------------------------------------------------
# Application pages: never a source for forms or coverages
# --------------------------------------------------------------------------
@pytest.mark.parametrize("line", [
    "General Liability Application ID: 3293395", "APPLICANT INFORMATION", "Applicant Information", "APPLICATION AGREEMENT",
    "WORKERS COMPENSATION PREMIUM CREDIT APPLICATION", "APPLICATION FOR INSURANCE", "APPLICATION", "# APPLICATION AGREEMENT",
])
def test_a_page_with_an_applications_wording_reads_as_one(line):
    assert application_pages(["declarations", f"top\n{line}\nbottom"]) == {2}


@pytest.mark.parametrize("line", [
    "Application of Other Exclusions", "(9) Application of Coverage", "dispersal or application of radioactive material,",
    "When the property is subject to the Anti-arson", "Application", "Employee or applicant for employment",
    "Applicant Name (Print)", "Policyholder/Applicant's Signature", "ADDENDUM TO THE APPLICATION",
    "ADDENDUM TO THE DECLARATIONS AND APPLICATION -", "A. \u201cApplication\u201d means:",
    "application for insurance by the insured. This", "IN AN APPLICATION FOR INSURANCE IS GUILTY OF A CRIME AND MAY BE SUBJECT",
    "APPLICATION FOR INSURANCE IS GUILTY OF A CRIME AND MAY BE SUBJECT", "IN THE APPLICATION",
])
def test_policy_wording_and_forms_attached_to_a_policy_do_not(line):
    """Each is a line that really sits on a policy page or an attached form in the cycle 1 PDFs."""
    assert application_pages([f"top\n{line}\nbottom"]) == set()


def test_pages_between_two_application_pages_close_together_are_the_same_application():
    pages = ["x"] * 12
    pages[1], pages[5] = "APPLICANT INFORMATION", "APPLICATION AGREEMENT"
    assert application_pages(pages) == {2, 3, 4, 5, 6}
    pages[5], pages[10] = "x", "APPLICATION AGREEMENT"                     # 9 pages apart: not read as one
    assert application_pages(pages) == {2, 11}


def _row(field_name, value, *pages):
    return {field_name: {"raw": value, "parsed": value, "confidence": {"score": 1.0, "source": "vlm"},
                         "page_ref": list(pages), "flagged": False}}


def test_only_a_form_or_coverage_row_citing_nothing_else_is_counted():
    label = {"forms_and_endorsements": [_row("form_number", "ONLY", 5), _row("form_number", "BOTH", 2, 5),
                                        _row("form_number", "ELSE", 2)],
             "coverages": [_row("coverage_name", "Fire", 5, 6)],
             "named_insured": _row("primary_name", "On the application too", 5)}
    assert rows_only_on_application_pages(label, {5, 6}) == [
        ("forms_and_endorsements[0].form_number", [5]), ("coverages[0].coverage_name", [5, 6])]
    assert rows_only_on_application_pages(label, set()) == []


def test_the_audit_warns_and_never_blocks_on_a_row_read_from_an_application(tmp_path):
    golden = copy.deepcopy(GOLDEN)
    golden["forms_and_endorsements"] = [_row("form_number", "CG 00 01", 2)]
    _document(tmp_path, "from_application", golden=golden,
              pages=[PAGE_1, ["APPLICANT INFORMATION", "Form CG 00 01 requested"]])
    report = audit_folder(tmp_path)
    found = [f for f in report.findings if f.check == "application_pages"]
    assert [f.severity for f in found] == ["warning"]
    assert "forms_and_endorsements[0].form_number (page 2)" in found[0].detail
    assert "CG 00 01" not in found[0].detail                                # the path and page, no value
    assert report.documents[0].application_pages == [2]
    assert summary(report)["application_pages"]["rows_citing_only_them"] == 1


def test_a_clean_document_reports_no_application_rows_and_counts_pages_it_could_not_read(tmp_path):
    _document(tmp_path, "good", pages=[PAGE_1, []])
    facts = summary(audit_folder(tmp_path))["application_pages"]
    assert facts == {"documents": 0, "pages": 0, "rows_citing_only_them": 0, "pages_not_judged_no_text": 1}
