"""The page check keeps a value the page prints in another written form (decision 2026-11-10, A1).

Proven on the real scorer the same day: a right date, form number or policy number was taken out of an answer -
in production too - because only its ``raw`` was looked for, and ``raw`` was not how the page prints it.
"""

from __future__ import annotations

from common.grounding import ground_tolerant, printed_forms


def _printed(envelope: dict, page: str) -> bool:
    return any(ground_tolerant(form, [1], {1: page}).printed for form in printed_forms(envelope))


def _env(raw, parsed=None) -> dict:
    return {"raw": raw, "parsed": raw if parsed is None else parsed, "page_ref": [1]}


def test_a_date_is_found_in_the_form_the_page_prints_it():
    assert _printed(_env("08/18/2022"), "Policy Period 08/18/22 to 08/18/23")
    assert _printed(_env("2022-08-18"), "Effective 8/18/2022")


def test_a_parsed_text_value_is_looked_for_beside_its_raw():
    assert _printed(_env("Aug 18 2022", "08/18/2022"), "Policy Period 08/18/22 to 08/18/23")


def test_a_code_is_found_with_or_without_its_spaces():
    assert _printed(_env("SP3023"), "FORMS SP 30 23 Special Property Coverage")
    assert _printed(_env("GL 123 4567"), "Policy Number GL1234567")
    assert _printed(_env("CG20100413"), "CG 20 10 04 13 Additional Insured")


def test_what_the_page_does_not_print_is_still_not_found():
    assert not _printed(_env("SP3024"), "FORMS SP 30 23 Special Property Coverage")     # one digit off
    assert not _printed(_env("08/19/2022"), "Policy Period 08/18/22 to 08/18/23")
    # Only digits are never joined: two figures side by side are not one number.
    assert not _printed(_env("1018"), "Limits 10 18 Deductible")


def test_a_number_parsed_is_not_looked_for_as_a_bare_figure():
    """A bare 1000 would be found in any year or code the page prints; "$1,000" is looked for as an amount."""
    assert printed_forms({"raw": "$1,000", "parsed": 1000}) == ("$1,000",)
    assert printed_forms({"raw": "08/18/2022", "parsed": "08/18/2022"})[0] == "08/18/2022"   # raw stays first
