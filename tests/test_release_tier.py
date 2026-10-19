"""Production and interim releases (Fideon SPEC_09 §6; evaluation.gating.release_tier).

Every existing floor stays. A release that passes them is PRODUCTION at field
match >= 0.92 and INTERIM below that; one under the 0.85 floor is blocked.
"""

from __future__ import annotations

from evaluation.gating import GATING_METRICS, PRODUCTION_FIELD_MATCH, promotion_gate, release_tier


def _metrics(field_match: float) -> dict[str, float]:
    base = {name: 0.96 for name in GATING_METRICS}
    base.update(schema_validity_rate=1.0, ece_confidence=0.04, confusable_misattribution_rate=0.02,
                false_null_rate=0.03, auto_accept_error_rate=0.01,
                field_normalized_match=field_match, field_exact_match=field_match)
    return base


def test_field_match_093_is_a_production_release():
    result = promotion_gate(_metrics(0.93), None)
    assert result.passed and result.tier == "production"


def test_field_match_089_is_interim_not_production():
    result = promotion_gate(_metrics(0.89), None)
    assert result.passed and result.tier == "interim"


def test_field_match_084_is_blocked_with_no_tier():
    result = promotion_gate(_metrics(0.84), None)
    assert not result.passed and result.tier is None


def test_a_small_eval_set_caps_a_production_score_at_interim_with_a_warning():
    """An eval set frozen with --allow-small says too little to call a release
    production, whatever it scores."""
    result = promotion_gate(_metrics(0.95), None, small_eval_set=True)
    assert result.passed and result.tier == "interim"
    assert any("allow-small" in w for w in result.warnings)
    assert "WARNING" in result.report()
    assert result.as_dict()["warnings"] == result.warnings
    assert release_tier({"field_normalized_match": 0.95}, small_eval_set=True) == "interim"


def test_the_production_bar_is_092_and_a_missing_value_is_interim():
    assert PRODUCTION_FIELD_MATCH == 0.92
    assert release_tier({"field_normalized_match": 0.92}) == "production"
    assert release_tier({}) == "interim"


def test_the_release_bundle_records_the_tier():
    from registry_utils.models import ReleaseBundle

    fields = ReleaseBundle.model_fields
    assert "tier" in fields and fields["tier"].default is None
