"""Near-identical PDFs in the exam: do they get the same answer, and the score with each policy counted once."""

from __future__ import annotations

from types import SimpleNamespace

from evaluation.metrics.near_copy import (
    consistency_section,
    near_copy_agreement,
    near_copy_pairs,
    per_policy_scores,
)


def _page(word):
    return f"{word} " + " ".join(f"line{i}" for i in range(20))


def _doc(source, pages, number="POL-1", **more):
    gold = {"policy": {"policy_number": {"raw": number, "parsed": number}}, **more}
    return SimpleNamespace(source_id=source, page_texts=dict(enumerate(pages, 1)), golden=gold)


def _env(value):
    return {"raw": value, "parsed": value}


def _meta(source, mode="image_only"):
    return {"source_id": source, "modality_mode": mode, "doc_type": "unknown"}


def _pair():
    shared = [_page(f"p{i}") for i in range(19)]
    return _doc("a", [*shared, _page("contact Richard")]), _doc("b", [*shared, _page("contact Cordelia")])


def test_documents_sharing_nine_tenths_of_their_pages_are_a_pair():
    a, b = _pair()
    other = _doc("c", [_page(f"q{i}") for i in range(20)])
    [pair] = near_copy_pairs([a, b, other])
    assert (pair.a, pair.b, pair.share) == ("a", "b", 0.95)


def test_spacing_and_case_do_not_hide_a_copy_and_blank_pages_do_not_make_one():
    a = _doc("a", [_page("x"), "1"])
    b = _doc("b", [_page("x").upper().replace(" ", "  "), "1"])
    assert [p.a for p in near_copy_pairs([a, b])] == ["a"]
    assert near_copy_pairs([_doc("c", ["1", ""]), _doc("d", ["1", ""])]) == []


def test_two_answers_alike_agree_fully():
    a, b = _pair()
    answer = {"name": "Sutton", "number": "POL-1"}
    report = near_copy_agreement([a, b], [({}, answer, _meta("a")), ({}, dict(answer), _meta("b"))])
    assert report["agreement"] == 1.0
    assert report["pairs"][0]["by_mode"]["image_only"]["agreement"] == 1.0


def test_a_difference_in_the_answers_lowers_the_agreement():
    a, b = _pair()
    report = near_copy_agreement([a, b], [({}, {"name": "Sutton", "number": "POL-1"}, _meta("a")),
                                          ({}, {"name": "Sutton", "number": "POL-2"}, _meta("b"))])
    assert report["agreement"] == 0.5


def test_a_value_only_one_gold_holds_is_left_out_because_the_pdfs_really_differ():
    a, b = _pair()
    gold_a, gold_b = {"contact": _env("Richard")}, {"contact": _env("Cordelia")}
    report = near_copy_agreement([a, b], [
        (gold_a, {"name": "Sutton", "contact": "Richard"}, _meta("a")),
        (gold_b, {"name": "Sutton", "contact": "Cordelia"}, _meta("b"))])
    assert report["agreement"] == 1.0
    assert report["pairs"][0]["by_mode"]["image_only"]["fields_compared"] == 1.0


def test_each_mode_is_compared_with_itself():
    a, b = _pair()
    report = near_copy_agreement([a, b], [
        ({}, {"name": "X"}, _meta("a", "image_only")), ({}, {"name": "X"}, _meta("b", "image_only")),
        ({}, {"name": "Y"}, _meta("a", "ocr_plus_image")), ({}, {"name": "Z"}, _meta("b", "ocr_plus_image"))])
    by_mode = report["pairs"][0]["by_mode"]
    assert by_mode["image_only"]["agreement"] == 1.0 and by_mode["ocr_plus_image"]["agreement"] == 0.0


def test_no_pair_in_the_exam_reports_none():
    report = near_copy_agreement([_doc("a", [_page("x")])], [({}, {"name": "X"}, _meta("a"))])
    assert report["pairs"] == [] and report["agreement"] is None


def test_a_policy_on_two_pdfs_counts_once_in_the_per_policy_mean():
    documents = [_doc("a", [], number="POL-1"), _doc("b", [], number="pol 1"), _doc("c", [], number="POL-2")]
    scores = per_policy_scores({"a": 0.2, "b": 0.4, "c": 1.0}, documents)
    assert scores == {"documents": 3, "policies": 2, "mean_per_document": 0.5333, "mean_per_policy": 0.65}


def test_a_document_with_no_policy_number_is_its_own_policy():
    documents = [_doc("a", [], number=""), _doc("b", [], number="")]
    assert per_policy_scores({"a": 0.2, "b": 0.4}, documents)["policies"] == 2
    assert per_policy_scores({}, documents) == {}


def test_a_failure_here_never_costs_a_gate_its_hours():
    section = consistency_section([SimpleNamespace(source_id="a")], [], {})
    assert "error" in section["near_copy_agreement"]
