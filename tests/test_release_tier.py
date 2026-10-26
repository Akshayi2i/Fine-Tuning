"""Production and interim releases (Fideon SPEC_09 §6; evaluation.gating.release_tier).

Every existing floor stays. A release that passes them is PRODUCTION when what
users receive - served_field_match, image_only readings of real documents as
served - is >= 0.92 with its one-sided 95% lower bound >= 0.90 over at least
150 real documents (decision 2026-10-26), and INTERIM otherwise; one under the
0.85 floor is blocked.
"""

from __future__ import annotations

import pytest

from evaluation.gating import GATING_METRICS, PRODUCTION_FIELD_MATCH, promotion_gate, release_tier


def _metrics(field_match: float) -> dict[str, float]:
    base = {name: 0.96 for name in GATING_METRICS}
    base.update(schema_validity_rate=1.0, ece_confidence=0.04, confusable_misattribution_rate=0.02,
                false_null_rate=0.03, auto_accept_error_rate=0.01,
                field_normalized_match=field_match, field_exact_match=field_match,
                served_field_match=field_match)
    return base


def _served(field_match: float, n: int = 160) -> tuple[list[float], list[float]]:
    """Per-document served field match, every document alike: the bound is the point."""
    return [field_match] * n, [1.0] * n


def test_field_match_093_is_a_production_release():
    result = promotion_gate(_metrics(0.93), None, served_documents=_served(0.93))
    assert result.passed and result.tier == "production"


def test_field_match_089_is_interim_not_production():
    result = promotion_gate(_metrics(0.89), None, served_documents=_served(0.89))
    assert result.passed and result.tier == "interim"


def test_field_match_084_is_blocked_with_no_tier():
    result = promotion_gate(_metrics(0.84), None, served_documents=_served(0.84))
    assert not result.passed and result.tier is None


def test_a_small_eval_set_caps_a_production_score_at_interim_with_a_warning():
    """An eval set frozen with --allow-small says too little to call a release
    production, whatever it scores."""
    result = promotion_gate(_metrics(0.95), None, small_eval_set=True, served_documents=_served(0.95))
    assert result.passed and result.tier == "interim"
    assert any("allow-small" in w for w in result.warnings)
    assert "WARNING" in result.report()
    assert result.as_dict()["warnings"] == result.warnings
    assert release_tier({"served_field_match": 0.95}, small_eval_set=True,
                        served_documents=_served(0.95)) == "interim"


def test_the_production_bar_is_092_and_a_missing_value_is_interim():
    assert PRODUCTION_FIELD_MATCH == 0.92
    assert release_tier({"served_field_match": 0.92}, served_documents=_served(0.92)) == "production"
    assert release_tier({}) == "interim"


def test_the_release_bundle_records_the_tier():
    from registry_utils.models import ReleaseBundle

    fields = ReleaseBundle.model_fields
    assert "tier" in fields and fields["tier"].default is None


def test_the_tier_minimum_is_the_frozen_set_minimum():
    from evaluation.freeze_eval_set import MIN_FROZEN_DOCS_PER_TYPE
    from evaluation.gating import PRODUCTION_MIN_REAL_DOCUMENTS

    assert PRODUCTION_MIN_REAL_DOCUMENTS == MIN_FROZEN_DOCS_PER_TYPE == 150


@pytest.mark.parametrize("n", [149, 1])
def test_a_bound_over_fewer_than_150_real_documents_is_interim(n):
    evidence = promotion_gate(_metrics(0.97), None, served_documents=_served(0.97, n)).tier_evidence
    assert evidence.tier == "interim"
    assert any("fewer than 150" in r for r in evidence.reasons), evidence.reasons
