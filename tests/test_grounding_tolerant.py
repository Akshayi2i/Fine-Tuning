"""Whether a value is printed on a scan: OCR's misreads forgiven, inventions not
(common.grounding.ground_tolerant, decision 2026-10-24)."""

from __future__ import annotations

import pytest

from common import grounding
from common.grounding import NOT_PRINTED, ON_CITED_PAGE, ON_OTHER_PAGE, UNCHECKED, ground_tolerant

PAGE = """DECLARATIONS
Policy Number: NIPO115 Effective 0l/01/2026
Named Insured: Acme Manufacturing LLC
lnsurance Company of the West
Premium 537.47   Limit 1.000.000
Deductible $2,500
"""


@pytest.mark.parametrize("raw", [
    "NIP0115",                       # O read for 0 in a code
    "01/01/2026",                    # l read for 1 in a date
    "$37.47",                        # $ read as 5
    "1,000,000",                     # separators misread
    "Insurance Company of the West",  # one letter misread in a long word
    "Acme Manufacturing LLC",        # exact
    "$2,500",                        # exact, as an amount
    2500,                            # a number written as a number
])
def test_what_ocr_misread_is_still_printed(raw):
    assert ground_tolerant(raw, [1], {1: PAGE}).status == ON_CITED_PAGE


@pytest.mark.parametrize("raw", [
    "$25.00",             # $2,500 printed: the decimal point never moves
    "$2,500,000",         # nor the thousands
    "$9,999.00",          # an amount no page prints
    "NIP0116",            # one digit changed is another number
    "Acme Holdings LLC",  # a word no page prints
    "02/01/2026",         # another date
    "SOLD",               # a word without a digit is never folded to digits
])
def test_an_invented_value_is_not_printed(raw):
    assert ground_tolerant(raw, [1], {1: PAGE}).status == NOT_PRINTED


def test_a_leading_5_is_a_misread_dollar_only_where_the_figure_shows_none():
    assert ground_tolerant("$37.47", [1], {1: "Premium 537.47"}).printed
    assert ground_tolerant("$37.47", [1], {1: "Premium $537.47"}).status == NOT_PRINTED


def test_printed_only_on_another_page_is_printed_not_invented():
    found = ground_tolerant("NIP0115", [2], {1: PAGE, 2: "Schedule of forms"})
    assert found.status == ON_OTHER_PAGE and found.printed and found.pages == (1,)


def test_what_the_text_cannot_settle_is_unchecked():
    assert ground_tolerant("NY", [1], {1: PAGE}).status == UNCHECKED            # too short
    assert ground_tolerant(True, [1], {1: PAGE}).status == UNCHECKED            # a reading, not a print
    assert ground_tolerant("$9,999.00", [1], {1: ""}).status == UNCHECKED       # no text at all
    # A page it cites has no text: the scan may print it where OCR read nothing.
    assert ground_tolerant("$9,999.00", [2], {1: PAGE, 2: ""}).status == UNCHECKED


def test_a_code_of_short_pieces_is_not_matched_word_by_word():
    page = "HX-0000-0000 and page 0001"
    assert ground_tolerant("HX-0000-0001", [1], {1: page}).status == NOT_PRINTED


def test_pages_are_prepared_once_for_a_document():
    pages = grounding.tolerant_pages({1: PAGE, 2: "Schedule"})
    assert ground_tolerant("NIP0115", [1], pages).status == ON_CITED_PAGE
    assert ground_tolerant("Acme Holdings LLC", [1], pages).status == NOT_PRINTED


def test_the_strict_matcher_agrees_with_the_exact_one_where_the_text_is_clean():
    clean = "Policy Number HO-1234567 Effective 01/01/2026 Premium $1,250.00"
    for raw in ("HO-1234567", "01/01/2026", "$1,250.00", "1250"):
        assert grounding.ground(raw, [1], {1: clean}).printed
        assert ground_tolerant(raw, [1], {1: clean}).printed
