"""Whether a value is printed, and where: one matcher for every caller (common.grounding)."""

from __future__ import annotations

from calibration.features import ocr_agreement
from common import grounding
from evaluation.metrics.extraction_faults import score_hallucinations


def _value(raw, parsed, pages):
    return {"raw": raw, "parsed": parsed, "page_ref": pages}


PAGES = {1: "Named Insured: Rivera Fabrication LLC", 2: "Coverage A Dwelling $219,000\nDeductible 1,000",
         3: ""}


def test_ground_says_where_a_value_is_printed_against_the_pages_it_cites():
    assert grounding.ground("$219,000", [2], PAGES).status == grounding.ON_CITED_PAGE
    assert grounding.ground("$219,000", [1], PAGES) == grounding.Grounding(grounding.ON_OTHER_PAGE, (2,))
    assert grounding.ground("Loss of Use", [2], PAGES).status == grounding.NOT_PRINTED
    assert grounding.ground("Rivera Fabrication LLC", [], PAGES).status == grounding.ON_CITED_PAGE


def test_what_cannot_be_checked_is_neither_grounded_nor_not():
    assert grounding.ground("NY", [1], PAGES).status == grounding.UNCHECKED          # too short
    # On no page with text, and a page it cites has none: that page may print it.
    assert grounding.ground("Loss of Use", [3], PAGES) == grounding.Grounding(grounding.UNCHECKED, unread=(3,))
    assert grounding.ground("anything", [1], {1: ""}).status == grounding.UNCHECKED  # nothing has text


def test_a_blank_page_among_those_cited_does_not_stop_the_search():
    """H1: a blank page it cites leaves the other pages to settle it."""
    # Cites a page with text and a blank one: printed on neither with text.
    both = grounding.ground("Loss of Use", [2, 3], PAGES)
    assert both.status == grounding.UNCHECKED and both.unread == (3,) and not both.printed
    # Printed on another page: printed, whatever the blank page it cites.
    assert grounding.ground("Rivera Fabrication LLC", [3], PAGES) == grounding.Grounding(grounding.ON_OTHER_PAGE, (1,))
    assert grounding.ground("$219,000", [2, 3], PAGES).status == grounding.ON_CITED_PAGE
    # Every page it cites has text (or it cites one no page is): printed nowhere.
    assert grounding.ground("Loss of Use", [1, 2], PAGES).status == grounding.NOT_PRINTED
    assert grounding.ground("Loss of Use", [9], PAGES).status == grounding.NOT_PRINTED
    assert grounding.ground("Loss of Use", [], PAGES).status == grounding.NOT_PRINTED


def test_words_split_across_lines_or_columns_still_match():
    assert grounding.appears("Rivera Fabrication LLC", "applicant:\nRIVERA   Fabrication, LLC")
    assert grounding.appears("1420 Foundry Road", "Foundry Road 1420")
    assert not grounding.appears("Rivera Fabrication LLC", "Meridian Holdings")


def test_a_code_is_not_found_in_pieces_of_other_values():
    page = "Prior policy HX-0000-0000  Form 0001"
    assert not grounding.appears("HX-0000-0001", page)
    assert grounding.appears("HX-0000-0000", page)


DECLARATIONS = ("POLICY DECLARATIONS\nPolicy Number: PAP-2026-0042\n"
                "Policy Period: 04/01/2025 to 04/01/2026\nNamed Insured: Jon Smith\n"
                "Agency Code 7781\nForm 5250 (rev)\nVeh 1  Year 2019  Honda Accord\n")


def test_a_code_is_one_run_of_words_never_its_pieces_found_apart():
    """H2: a code with letter and number segments is never a bag of words."""
    for invented in ("PAP-2025-0042",     # another segment of the page: the policy period's year
                     "PAP-2026-7781",     # the agency code as its last segment
                     "PAP-0042-2026",     # its segments reordered
                     "PAP-2026-0043"):    # one digit off
        assert grounding.ground(invented, [1], {1: DECLARATIONS}).status == grounding.NOT_PRINTED, invented
    assert grounding.appears("PAP-2026-0042", DECLARATIONS)
    assert grounding.appears("PAP 2026-0042", "Policy PAP-2026-\n0042")         # broken across a line
    # A phrase broken across lines or columns still matches, token by token.
    assert grounding.appears("Honda Accord 2019", DECLARATIONS)


def test_an_amount_matches_only_an_amount_and_a_figure_only_a_whole_one():
    """H3: a figure is whole, and an amount is not a year, a code or a form number."""
    page = {1: DECLARATIONS}
    for invented in ("$2,019.00",   # the vehicle's year
                     "$7,781",      # the agency code
                     "$250.00",     # the end of "Form 5250"
                     "$5,250.00"):  # the form number as an amount
        assert grounding.ground(invented, [1], page).status == grounding.NOT_PRINTED, invented
    assert grounding.ground("2019", [1], page).printed                         # the year itself
    assert grounding.ground("7781", [1], page).printed
    # Never a piece of a longer figure.
    assert not grounding.appears("500", "Deductible $1,500.00")
    assert not grounding.appears("$1.00", "VIN 1HGCV1F30KA000001")


def test_an_amount_matches_the_same_figure_in_another_format():
    assert grounding.appears("$7,579.67", "TOTAL ANNUAL AUTO PREMIUM 7579.67")
    assert grounding.appears(18.0, "Personal Injury $18.00")
    assert grounding.appears("$500", "Deductible $ 500")
    assert grounding.appears("$500", "Deductible 500.00")
    assert grounding.appears("500", "Deductible $500")                           # a bare figure: any form
    assert grounding.appears("$25.00", "On-Water Towing (optional), $25 a year")
    assert not grounding.appears("$7,579.67", "TOTAL 7,579.76")
    assert not grounding.appears("$2.00", "Location: 2 TOWN RD  Page 2 of 3")    # a house number, a page
    # Written as an amount, it needs one: a bare whole number is a count or a code.
    assert not grounding.appears("$500", "Deductible 500")


def test_the_ocr_feature_matches_on_word_boundaries_and_skips_short_values():
    assert ocr_agreement("10", "Policy 10 of 12") is None          # too short to look for
    assert ocr_agreement("2100", "Limit 21000") == 0.0              # no longer a substring hit
    assert ocr_agreement("Fabrication", "RIVERA FABRICATION LLC") == 1.0
    assert ocr_agreement("Fabrication", None) is None


def test_a_value_is_looked_for_as_printed_not_as_reformatted():
    got = {"coverages": [{"limits": [{"amount": _value("$219,000", 219000.0, [2])}]}]}
    report = score_hallucinations([({}, got, PAGES)])
    assert (report.opportunities, report.hits) == (1, 0)       # parsed 219000.0 is not on the page; raw is


def test_an_invented_value_is_counted_and_a_short_one_is_looked_for_as_a_whole_token():
    got = {"coverages": [{"coverage_name": _value("Debris Removal", "Debris Removal", [2]),
                          "premium": _value("18", 18.0, [2]),          # two characters: a whole figure
                          "limit_code": _value("7", "7", [2])}]}         # one: on every page, left out
    report = score_hallucinations([({}, got, PAGES)])
    assert (report.opportunities, report.hits) == (2, 2)
    assert [hit["field"] for hit in report.instances] == ["coverages[0].coverage_name", "coverages[0].premium"]


def test_printed_on_another_page_is_a_wrong_page_not_an_invention():
    got = {"named_insured": {"primary_name": _value("Rivera Fabrication LLC", "Rivera Fabrication LLC", [2])}}
    assert score_hallucinations([({}, got, PAGES)]).hits == 0


def test_extra_fields_are_scored_apart():
    """The rate counts every value the model wrote, as serving's printed-values
    check does (decision 2026-10-24); the extra fields are also reported alone."""
    got = {"policy": {"policy_number": _value("Dwelling", "Dwelling", [2])},
           "additional_fields": [{"label": "Note", "section_hint": "discounts",
                                  "value": _value("Water Leak Detection", "x", [1])}]}
    core = score_hallucinations([({}, got, PAGES)])
    extra = score_hallucinations([({}, got, PAGES)], overflow=True)
    assert (core.name, core.opportunities, core.hits) == ("hallucination_rate", 2, 1)
    # Its label and section hint are bare values of a canonical answer: not looked for.
    assert (extra.name, extra.opportunities, extra.hits) == ("additional_fields_hallucination_rate", 1, 1)


def test_a_flat_answers_bare_values_are_printed_text():
    got = {"insured_name": "Rivera Fabrication LLC", "policy_number": "ZZ-404-9"}
    report = score_hallucinations([({}, got, "Applicant: Rivera Fabrication LLC")])
    assert (report.opportunities, report.hits) == (2, 1)


def test_a_validation_row_keeps_each_pages_text_by_its_number():
    from evaluation.validation_generation import ValidationGeneration

    row = {"modality_mode": "ocr_plus_image", "messages": [{"role": "user", "content": [
        {"type": "image", "image": "p9.png"}, {"type": "text", "text": "<page 9 of 20>\n\nLimit $5,000"},
        {"type": "image", "image": "p14.png"}, {"type": "text", "text": "<page 14 of 20>\n\nForms list"},
    ]}]}
    assert ValidationGeneration(row=row, golden={}).ocr_pages == {9: "Limit $5,000", 14: "Forms list"}
    assert ValidationGeneration(row={**row, "modality_mode": "image_only"}, golden={}).ocr_pages is None


def test_the_agreement_feature_grounds_a_value_on_the_pages_it_cites():
    from calibration.features import build_document_features, page_agreement

    texts = {1: "Named Insured: Rivera Fabrication LLC", 2: "Coverage A Dwelling $219,000"}
    assert page_agreement("$219,000", [2], texts) == 1.0
    assert page_agreement("$219,000", [1], texts) == 0.5          # printed, on another page
    assert page_agreement("$999,000", [2], texts) == 0.0
    assert page_agreement("NY", [1], texts) is None and page_agreement("$219,000", [2], None) is None
    doc = {"coverages": [{"limits": [{"amount": _value("$219,000", 219000.0, [1])}]}]}
    [features] = build_document_features(extraction=doc, spans={}, page_texts=texts)
    assert features.ocr_agreement == 0.5
    [whole] = build_document_features(extraction=doc, spans={}, page_text=texts[2])
    assert whole.ocr_agreement == 1.0                              # without page texts: the text searched whole
