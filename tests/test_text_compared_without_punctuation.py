"""The scorer compares text with punctuation, quote and dash styles left out (decision 2026-11-10, A2).

Proven the same day on the real scorer: "Accounts Receivable." was counted wrong against "ACCOUNTS RECEIVABLE", and
"OWNER’S ... –" against "OWNER'S ... -". Names already compared so (normalize_entity_name).
"""

from __future__ import annotations

from common.normalize import compare_text, normalize_text, values_match


def test_the_same_words_with_other_punctuation_match():
    assert values_match("ACCOUNTS RECEIVABLE", "Accounts Receivable.", field_path="coverage.description")
    assert values_match("OWNER'S AND CONTRACTOR'S - PROTECTIVE", "Owner’s and Contractor’s – Protective",
                        field_path="coverage.description")
    assert values_match(["Building", "Business Income"], ["Business Income.", "building"], field_path="x.y")


def test_punctuation_does_not_join_or_merge_what_differs():
    """Punctuation becomes a space, never nothing: "1.5" is not "15", "A-1" not "A1"."""
    assert not values_match("Limit 1.5 million", "Limit 15 million", field_path="coverage.description")
    assert not values_match("Unit A-1", "Unit A1", field_path="location.description")
    assert not values_match("Accounts Receivable", "Accounts Payable", field_path="coverage.description")


def test_only_the_comparison_changes():
    """Serving's merge, row ids and the page search keep normalize_text."""
    assert normalize_text("Accounts Receivable.") == "accounts receivable."
    assert compare_text("Accounts Receivable.") == "accounts receivable"


def test_dates_numbers_and_identifiers_keep_their_own_rules():
    assert values_match("08/18/2022", "2022-08-18", field_path="policy.effective_date")
    assert not values_match("08/18/2022", "08/19/2022", field_path="policy.effective_date")
    assert values_match("GL-1001", "GL1001", field_path="policy.policy_number")
