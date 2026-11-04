"""One form written two ways is one form (evaluation.metrics.form_editions)."""

from __future__ import annotations

from evaluation.metrics.form_editions import (
    EDITION_IN_ANSWER_NUMBER,
    EDITION_IN_LABEL_NUMBER,
    align_form_editions,
)
from evaluation.run_eval import build_report


def _env(value, *, pages=(8,)):
    return {"raw": value, "parsed": value, "page_ref": list(pages), "confidence": {"score": 1.0, "source": "vlm"},
            "flagged": False}


def _row(number, edition=None, title="Signature page"):
    return {"form_number": _env(number), "edition_date": _env(edition) if edition else None, "title": _env(title)}


def _doc(*rows):
    return {"forms_and_endorsements": list(rows)}


def _numbers(doc):
    return [(r["form_number"]["parsed"], (r["edition_date"] or {}).get("parsed")) for r in doc["forms_and_endorsements"]]


def test_a_label_holding_the_edition_in_its_number_is_written_as_the_answer_splits_it():
    label = _doc(_row("SSI SF 00 03 0123"))
    aligned, moved = align_form_editions(label, _doc(_row("SSI SF 00 03", "01/23")))
    assert _numbers(aligned) == [("SSI SF 00 03", "01/23")]
    assert moved == {EDITION_IN_LABEL_NUMBER: 1}


def test_the_other_way_round_is_one_form_too_and_counted_apart():
    label = _doc(_row("CG 00 01", "04/13"))
    aligned, moved = align_form_editions(label, _doc(_row("CG 00 01 04 13")))
    assert _numbers(aligned) == [("CG 00 01 04 13", None)]
    assert moved == {EDITION_IN_ANSWER_NUMBER: 1}


def test_the_labels_pages_are_kept_and_the_callers_document_is_not_changed():
    label = _doc(_row("SSI SF 00 03 0123"))
    aligned, _moved = align_form_editions(label, _doc(_row("SSI SF 00 03", "01/23")))
    row = aligned["forms_and_endorsements"][0]
    assert row["form_number"]["page_ref"] == [8] and row["edition_date"]["page_ref"] == [8]
    assert _numbers(label) == [("SSI SF 00 03 0123", None)]


def test_another_edition_of_the_same_form_is_not_the_same_form():
    label = _doc(_row("SSI SF 00 03 0123"))
    aligned, moved = align_form_editions(label, _doc(_row("SSI SF 00 03", "03/18")))
    assert aligned is label and not moved


def test_another_form_and_letters_after_the_number_are_left_alone():
    for number, answer in (("CG 00 01", "CG 00 02"), ("SSI SF 00 03 ABCD", "SSI SF 00 03"), ("0025321173P1", "SSI SF 00 03")):
        label = _doc(_row(number))
        aligned, moved = align_form_editions(label, _doc(_row(answer)))
        assert aligned is label and not moved


def test_a_number_both_hold_alike_is_never_taken_for_another_rows_edition():
    """The answer's `CG 20 10` pairs with the label's own; the label's `CG 20 10 04 13` stays free."""
    label = _doc(_row("CG 20 10"), _row("CG 20 10 04 13"))
    aligned, moved = align_form_editions(label, _doc(_row("CG 20 10")))
    assert aligned is label and not moved


def test_each_label_row_is_taken_once():
    label = _doc(_row("SSI SF 00 03 0123"))
    answer = _doc(_row("SSI SF 00 03", "01/23"), _row("SSI SF 00 03", "01/23"))
    _aligned, moved = align_form_editions(label, answer)
    assert moved == {EDITION_IN_LABEL_NUMBER: 1}


def test_plain_values_work_as_envelopes_do():
    label = {"forms_and_endorsements": [{"form_number": "SS Privacy 10 21", "edition_date": None}]}
    answer = {"forms_and_endorsements": [{"form_number": "SS Privacy", "edition_date": "10/21"}]}
    aligned, moved = align_form_editions(label, answer)
    assert aligned["forms_and_endorsements"][0] == {"form_number": "SS Privacy", "edition_date": "10/21"}
    assert moved == {EDITION_IN_LABEL_NUMBER: 1}


def test_documents_without_a_forms_list_pass_through():
    assert align_form_editions({"policy": {}}, {"policy": {}})[0] == {"policy": {}}
    assert align_form_editions(None, {})[0] is None


def _meta(source="p1"):
    return {"source_id": source, "doc_type": "policy", "lob": "gl", "modality_mode": "image_only"}


def test_the_report_scores_one_form_as_right_and_counts_what_moved():
    gold = _doc(_row("SSI SF 00 03 0123"))
    answer = _doc(_row("SSI SF 00 03", "01/23"))
    report = build_report("v1", [(gold, answer, _meta())])
    assert report.form_editions_aligned == {EDITION_IN_LABEL_NUMBER: 1}
    assert report.as_dict()["form_editions_aligned"] == {EDITION_IN_LABEL_NUMBER: 1}
    records = [r for s in report.full_set() for r in s.error_records]
    assert not [r for r in records if "forms_and_endorsements" in r["field_path"]], records


def test_a_real_extra_form_still_counts_as_invented():
    gold = _doc(_row("SSI SF 00 03 0123"))
    answer = _doc(_row("SSI SF 00 03", "01/23"), _row("0025321173P1", title="Application"))
    report = build_report("v1", [(gold, answer, _meta())])
    records = [r for s in report.full_set() for r in s.error_records]
    assert {"0025321173P1"} <= {r["got"] for r in records}
