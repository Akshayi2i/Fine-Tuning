"""Every page is read; the router only finds the declarations (serving.page_router)."""

from __future__ import annotations

from serving.page_router import plan_pages, score_page

#: Chubb's coverage summary continued, and a Travelers limits page: both scored 0
#: under page selection, and their rows were never read.
CONTINUED = ("Coverage Summary Page 2 Effective date 6/4/21 Policy no. 15162088-01 Homes and Contents "
             "(Continued) Water backup deductible In lieu of the base deductible")
LIMITS = ("PL-50355 PA (05-17) Page D-2 Liability Coverage Section Limit Coverage E - Personal Liability "
          "$500,000 Coverage F - Medical Payments to Others $1,000 Peril Deductible All Perils $1,000")


def test_every_page_is_read_whatever_it_prints():
    """Every page, chunk by chunk, never selected pages (decision 2026-10-19)."""
    texts = {1: "Policy Declarations Named Insured", 2: LIMITS, 3: "Fraud notice", 4: "Privacy notice",
             5: "Your online account", 6: CONTINUED, 7: "Signatures"}
    plan = plan_pages(texts)
    assert plan.pages == [1, 2, 3, 4, 5, 6, 7]
    assert plan.declarations_page == 1


def test_the_declarations_are_found_behind_a_cover_sheet():
    texts = {1: "FAX COVER SHEET", 2: "Billing notice", 3: "Common Policy Declarations - Named Insured"}
    plan = plan_pages(texts)
    assert plan.declarations_page == 3
    assert plan.pages == [1, 2, 3]


def test_a_notice_page_carries_no_signal_and_is_still_read():
    notice = "IMPORTANT NOTICE - DRIVING WHILE IMPAIRED - NEW YORK. Overall 33% Nighttime 59% Weekends 45%"
    assert score_page(notice, 23).score == 0
    assert plan_pages({1: "Declarations", 2: notice}).pages == [1, 2]
