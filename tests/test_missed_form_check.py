"""scripts/missed_form_check.py: whether a search of the pages could warn about a form an answer left out."""

from __future__ import annotations

from scripts.missed_form_check import candidates, check_document, page_candidates, render, summarise


def _gold(*rows, **more):
    return {"forms_and_endorsements": [{"form_number": number, "edition_date": edition} for number, edition in rows], **more}


def test_every_shape_a_form_number_is_printed_in_is_a_candidate():
    for number in ("CG 00 01 04 13", "IL00171198", "SSI SF 00 03 0123", "PGL 00200 0422", "FS 00 H357 00 1016",
                   "CTB CW SLV 06 20", "WC 00 04 19", "IPP-2000"):
        assert candidates(f"attached form {number} applies") == [number], number


def test_what_only_looks_like_one_is_not():
    for text in ("Old Forge, NY 13420", "PAGE 1 OF 2", "POLICY NUMBER 12345", "Limit $1,000,000", "phone 315 555 1951"):
        assert candidates(text) == [], text


def test_a_number_at_a_pages_head_or_foot_is_counted_apart_from_one_in_its_body():
    text = "\n".join(["CG 20 10 04 13", *[f"line {i}" for i in range(12)], "see form CG 99 99 01 01 for details",
                      *[f"line {i}" for i in range(12)], "Page 1 of 2"])
    edge, body = page_candidates(text)
    assert edge == {"CG 20 10 04 13"} and body == {"CG 99 99 01 01"}


def test_a_gold_form_is_seen_with_its_edition_written_in_the_number_or_apart():
    pages = {1: "CG 00 01 04 13\nSSI SF 00 03 0123"}
    result = check_document(_gold(("CG 00 01", "04/13"), ("SSI SF 00 03 0123", None), ("IL 00 17", None)), pages)
    assert (result["gold_forms"], result["seen"]) == (3, 2)
    assert result["new_edge"] == [] and result["new_body"] == []


def test_a_form_the_gold_does_not_hold_is_a_candidate_and_one_it_holds_elsewhere_is_not():
    pages = {1: "CG 00 01 04 13\nCG 21 47 12 07\nPOL 123456"}
    gold = _gold(("CG 00 01", "04/13"), policy={"policy_number": "POL 123456"})
    assert check_document(gold, pages)["new_edge"] == ["CG 21 47 12 07"]


def test_the_summary_and_the_report_count_the_documents():
    results = {"a": {"gold_forms": 4, "seen": 3, "new_edge": ["X 1234"], "new_body": []}}
    assert summarise(results) == {"documents": 1, "gold_forms": 4, "seen": 3, "seen_share": 0.75, "new_edge": 1,
                                  "new_body": 0}
    assert "1 left out" in render(results, ["b"])
