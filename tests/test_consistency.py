"""serving/consistency.py: values that cannot all be true together, sent to review.

The printed-values check cannot see a value that is printed but sits in the wrong
field; a rule the document obeys can. Each rule here held on the reviewed golds
of the first commercial set, so a flag is a doubt and not review noise."""

from __future__ import annotations

import pytest

from serving.consistency import CONSISTENCY_FLAGS, consistency_flags
from serving.value_flags import REVIEW_ROUTED, route_served, routed_paths


def fv(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page], "confidence": {"score": 0.9}, "flagged": False}


def policy(**overrides):
    base = {
        "policy": {"policy_number": fv("GL-12345"), "effective_date": fv("07/12/2025"),
                   "expiration_date": fv("07/12/2026")},
        "premium": {"total": fv(500.0), "items": [{"description": fv("Premium"), "amount": fv(500.0)}]},
        "coverages": [{"coverage_code": fv("GL_BI_PD"), "limits": [
            {"limit_type": fv("per_occurrence"), "amount": fv(1000000.0)},
            {"limit_type": fv("general_aggregate"), "amount": fv(2000000.0)}]}],
        "carrier": {"name": fv("Acme Mutual"), "address": {"postal_code": fv("12345"), "state": fv("NY")}},
    }
    for path, value in overrides.items():
        node = base
        *parents, last = path.split(".")
        for part in parents:
            node = node[int(part)] if part.isdigit() else node[part]
        node[int(last) if last.isdigit() else last] = value
    return base


def test_a_consistent_policy_raises_nothing():
    assert consistency_flags(policy()) == []
    assert consistency_flags({}) == [] and consistency_flags(None) == [] and consistency_flags("x") == []


@pytest.mark.parametrize(("overrides", "expected"), [
    ({"policy.expiration_date": fv("07/12/2025")}, "policy.expiration_date:date_order"),          # same day
    ({"policy.expiration_date": fv("06/01/2025")}, "policy.expiration_date:date_order"),          # before
    ({"policy.expiration_date": fv("07/12/2028")}, "policy.expiration_date:term_too_long"),
    ({"policy.effective_date": fv("07/12/1925"), "policy.expiration_date": fv("07/12/1926")},
     "policy.effective_date:implausible_date"),
    ({"policy.policy_number": fv("12")}, "policy.policy_number:bad_policy_no"),
    ({"policy.policy_number": fv("ABCDEFG")}, "policy.policy_number:bad_policy_no"),
    ({"premium.total": fv(300.0), "premium.items": [
        {"description": fv("a"), "amount": fv(200.0)}, {"description": fv("b"), "amount": fv(150.0)}]},
     "premium.total:premium_below_items"),
    ({"coverages.0.limits": [{"limit_type": fv("per_occurrence"), "amount": fv(3000000.0)},
                             {"limit_type": fv("general_aggregate"), "amount": fv(2000000.0)}]},
     "coverages[0].limits[0].amount:limit_order"),
    ({"carrier.address.postal_code": fv("1234")}, "carrier.address.postal_code:bad_postal_code"),
    ({"carrier.address.state": fv("New York")}, "carrier.address.state:bad_state"),
])
def test_each_rule_names_the_value_that_breaks_it(overrides, expected):
    assert consistency_flags(policy(**overrides)) == [expected]


@pytest.mark.parametrize("overrides", [
    {"carrier.address.postal_code": fv("12345-6789")},               # ZIP+4, both spellings
    {"carrier.address.postal_code": fv("123456789")},
    {"premium.total": fv(520.0)},                                    # more than its items: taxes, fees
    {"premium.total": fv(500.4)},                                    # within a cent's rounding
    {"coverages.0.limits": [{"limit_type": fv("per_occurrence"), "amount": fv(1000000.0)}]},    # no aggregate to compare
    {"policy.expiration_date": fv(None)},                            # a rule needs the values it compares
    {"policy.effective_date": fv("2025-07-12"), "policy.expiration_date": fv("2026-07-12")},    # ISO dates
])
def test_a_value_the_rules_allow_is_not_flagged(overrides):
    assert consistency_flags(policy(**overrides)) == []


def test_a_policy_with_nothing_to_compare_says_nothing():
    assert consistency_flags({"policy": {"policy_number": fv("GL-1")}}) == []
    # A total with no itemised amounts, an item with no amount: no sum to hold it to.
    assert consistency_flags({"premium": {"total": fv(10.0), "items": [{"description": fv("x")}]}}) == []


def test_every_rule_sends_its_value_to_review():
    """A flagged value is kept and marked flagged in its envelope: never auto-accepted."""
    assert CONSISTENCY_FLAGS <= REVIEW_ROUTED
    served = policy(**{"policy.expiration_date": fv("06/01/2025")})
    flags = consistency_flags(served)
    assert routed_paths(flags) == {"policy.expiration_date"}
    assert route_served(served, flags) == 1
    assert served["policy"]["expiration_date"]["flagged"] is True and served["policy"]["effective_date"]["flagged"] is False
    assert served["policy"]["expiration_date"]["parsed"] == "06/01/2025"           # kept, not removed


def test_the_pipeline_serves_an_expiration_before_the_effective_date_flagged_and_in_review(monkeypatch):
    """Through extract: the value is kept, its envelope is flagged, and the document says why."""
    import serving.printed_values as printed_values
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from tests.test_policy_windows import _ScriptedBackend, _serve_with

    monkeypatch.setattr(printed_values, "configured_mode", lambda: printed_values.FLAG)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    result = _serve_with(client, _ScriptedBackend(fail=lambda group, pages: "bad_dates" if group == "decl" else None))
    assert "policy.expiration_date:date_order" in result.review_flags
    expiration = result.extraction["policy"]["expiration_date"]
    assert expiration["parsed"] == "06/01/2025" and expiration["flagged"] is True        # kept, and in review
    # Without calibrators every value is flagged, so the routing is read from the flags themselves.
    assert "policy.expiration_date" in routed_paths(result.review_flags)
    assert "policy.effective_date" not in routed_paths(result.review_flags)

    clean = _serve_with(client, _ScriptedBackend())
    assert not any(flag.rpartition(":")[2] in CONSISTENCY_FLAGS for flag in clean.review_flags)
