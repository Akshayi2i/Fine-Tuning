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
    assert ground_tolerant("7", [1], {1: PAGE}).status == UNCHECKED             # one character: on every page
    assert ground_tolerant(True, [1], {1: PAGE}).status == UNCHECKED            # a reading, not a print
    for reading in ("Y", "N", "Yes", "No", "X"):                                # a box marked or not
        assert ground_tolerant(reading, [1], {1: PAGE}).status == UNCHECKED, reading
    assert ground_tolerant("$9,999.00", [1], {1: ""}).status == UNCHECKED       # no text at all
    # A page it cites has no text: the scan may print it where OCR read nothing.
    unread = ground_tolerant("$9,999.00", [2], {1: PAGE, 2: ""})
    assert unread.status == UNCHECKED and unread.unread == (2,)


def test_a_short_value_is_looked_for_as_a_whole_token():
    page = {1: "State NJ  Towing $50  Units 12  Rate 12.50 per day"}
    assert ground_tolerant("NJ", [1], page).status == ON_CITED_PAGE
    assert ground_tolerant("NY", [1], page).status == NOT_PRINTED
    assert ground_tolerant("$50", [1], page).status == ON_CITED_PAGE
    assert ground_tolerant("$5", [1], page).status == NOT_PRINTED
    assert ground_tolerant("12", [1], page).status == ON_CITED_PAGE
    assert ground_tolerant("50", [1], {1: "Rate 12.50 per day"}).status == NOT_PRINTED   # never the cents of 12.50
    assert ground_tolerant("NJ", [1], {1: "NJMX Insurance"}).status == NOT_PRINTED       # never part of a word


def test_a_blank_cited_page_leaves_the_pages_with_text_to_settle_it():
    """H1: an invented value citing a page with text and a blank one is not
    printed on any page with text - unchecked, its blank page named; one found
    on another page is printed."""
    pages = {1: PAGE, 2: "Schedule of forms", 3: ""}
    invented = ground_tolerant("Acme Phantom Insurance Co", [1, 3], pages)
    assert invented.status == UNCHECKED and invented.unread == (3,) and not invented.printed
    assert ground_tolerant("Acme Phantom Insurance Co", [1, 2], pages).status == NOT_PRINTED
    found = ground_tolerant("Schedule of forms", [3], pages)
    assert found.status == ON_OTHER_PAGE and found.pages == (2,)


DECLARATIONS = ("POLICY DECLARATIONS\nPolicy Number: PAP-2026-0042\n"
                "Policy Period: 04/01/2025 to 04/01/2026\nNamed Insured: Jon Smith\n"
                "Agency Code 7781\nForm 5250 (rev)\nVeh 1  Year 2019  Honda Accord\n")


@pytest.mark.parametrize("raw", [
    "PAP-2025-0042",     # another segment of the page: the policy period's year
    "PAP-2026-7781",     # the agency code as its last segment
    "PAP-0042-2026",     # its segments reordered
    "PAP-2026-0043",     # one digit off
    "PAP-2026-OO43",     # one digit off, misread
])
def test_a_code_is_one_run_of_words_never_its_pieces_found_apart(raw):
    """H2: OCR's misreads are forgiven inside a code, never its order."""
    assert ground_tolerant(raw, [1], {1: DECLARATIONS}).status == NOT_PRINTED


def test_a_misread_code_is_still_one_run_of_words():
    assert ground_tolerant("PAP-2026-0042", [1], {1: "Policy PAP-2O26-OO42"}).status == ON_CITED_PAGE
    assert ground_tolerant("Smith-Jones Holdings", [1], {1: "Insured Smlth-Jones Holdings"}).printed
    assert not ground_tolerant("Smith-Jones Holdings", [1], {1: "Jones Smith Holdings"}).printed


@pytest.mark.parametrize("raw", [
    "$250.00",     # "Form 5250": a bare number's leading 5 is not a misread "$"
    "$2,019.00",   # the vehicle's year
    "$7,781",      # the agency code
    "$7781",       # the agency code, as an amount
])
def test_an_amount_is_not_a_year_a_code_or_a_form_number(raw):
    """H3: written as an amount, a figure is only an amount the page prints as one."""
    assert ground_tolerant(raw, [1], {1: DECLARATIONS}).status == NOT_PRINTED


@pytest.mark.parametrize(("raw", "printed"), [
    ("$1,250.00", "Premium S1,250.00"),     # a "$" read as an "S"
    ("$1,250.00", "Premium 51,250.00"),     # a "$" read as a "5"
    ("$1,250.00", "Premium $1,25O.OO"),     # 0s read as Os
    ("$500", "Deductible $ 500"),           # a space after the "$"
])
def test_a_genuine_amount_keeps_its_ocr_tolerance(raw, printed):
    assert ground_tolerant(raw, [1], {1: printed}).status == ON_CITED_PAGE


def test_a_separator_is_never_read_as_moving_the_decimal_point():
    assert ground_tolerant("$2.50", [1], {1: "Deductible $2,500"}).status == NOT_PRINTED
    assert ground_tolerant("$25", [1], {1: "Rating factor 2.5"}).status == NOT_PRINTED


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
