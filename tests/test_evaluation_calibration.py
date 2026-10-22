"""IMPL-08 + IMPL-09 — metrics, the promotion gate, and confidence calibration.

Two guards here matter more than the rest: the gate must block on **any single**
regression with no override, and calibration must **raise** rather than pass raw
overconfident numbers through as if they were calibrated.
"""

from __future__ import annotations

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.apply_calibration import (
    apply_calibration,
    calibrated_output,
    load_calibration,
)
from calibration.fit_calibration import (
    CalibrationError,
    CalibrationParams,
    assert_held_out,
    fit_calibration,
)
from calibration.list_completeness import check_completeness, check_document, merge_review_flags
from calibration.logprob_confidence import (
    ConfidenceError,
    aggregate_span,
    field_confidences,
    overall_confidence,
    unusable_fields,
)
from evaluation.gating import GATING_METRICS, promotion_gate
from evaluation.metrics.confusable import score_alias_accuracy, score_misattribution
from evaluation.metrics.coverage_metrics import (
    expected_calibration_error,
    score_by_mode,
    score_lob,
)
from evaluation.metrics.field_accuracy import score_fields, score_list_field
from inference_core.span_map import FieldSpan

# --------------------------------------------------------------------------
# Field accuracy
# --------------------------------------------------------------------------

def test_normalized_match_beats_string_equality():
    """Insurance fields need it or correct extractions are under-counted."""
    report = score_fields(
        {"effective_date": "2026-04-01", "total_premium": 47250.0},
        {"effective_date": "04/01/2026", "total_premium": "$47,250.00"},
    )
    assert report.normalized_match == 1.0
    assert report.exact_match == 0.0        # neither is string-identical


def test_per_field_accuracy_shows_which_field_fails():
    report = score_fields(
        {"insured_name": "Rivera", "policy_number": "WC-1"},
        {"insured_name": "Rivera", "policy_number": "WC-2"},
    )
    by_field = report.by_field()
    assert by_field["insured_name"] == 1.0
    assert by_field["policy_number"] == 0.0


def test_list_recall_catches_a_dropped_row():
    """A missing row generates no tokens, so recall is the only signal that
    sees it (arch §5)."""
    expected = [{"claim_number": f"C{i}", "paid": i * 100} for i in range(1, 9)]
    got = expected[:6]                       # two claims silently dropped

    report = score_list_field(expected, got, "claims")
    assert report.recall == pytest.approx(6 / 8)
    assert report.precision == 1.0           # every row it DID produce is correct
    assert report.missed_rows == 2


def test_rows_are_matched_by_identity_not_position():
    """Dropping row 2 must not mis-align every row after it."""
    expected = [{"claim_number": f"C{i}"} for i in range(1, 6)]
    got = [row for row in expected if row["claim_number"] != "C2"]
    assert score_list_field(expected, got, "claims").matched_rows == 4


def test_a_missing_list_field_scores_zero_recall():
    from evaluation.metrics.field_accuracy import score_all_list_fields

    reports = score_all_list_fields({"claims": [{"claim_number": "C1"}]}, {})
    assert reports["claims"].recall == 0.0


# --------------------------------------------------------------------------
# Confusable misattribution — the gating metric
# --------------------------------------------------------------------------

def test_misattribution_detects_a_stolen_value():
    """The failure that is fluent, schema-valid, confidently wrong, and caught by
    nothing else in the pipeline."""
    expected = {
        "insured_name": "Rivera Fabrication LLC",
        "certificate_holder": "Meridian Property Group LLC",
    }
    got = {
        "insured_name": "Meridian Property Group LLC",   # the certificate holder
        "certificate_holder": "Meridian Property Group LLC",
    }
    report = score_misattribution(expected, got, "policy", source_id="policy_0001")
    assert report.cases
    case = report.cases[0]
    assert case.field_path == "insured_name"
    assert case.stolen_from == "certificate_holder"


def test_an_ordinary_wrong_value_is_not_misattribution():
    """Only a value that is right for ANOTHER field counts — a merely wrong
    value is ordinary field error with a different fix."""
    expected = {"insured_name": "Rivera Fabrication LLC", "producer": "Hanover"}
    got = {"insured_name": "Completely Unrelated Co", "producer": "Hanover"}
    assert not score_misattribution(expected, got, "policy").cases


def test_a_correct_extraction_has_no_misattribution():
    expected = {"insured_name": "Rivera", "certificate_holder": "Meridian"}
    assert not score_misattribution(expected, dict(expected), "policy").cases


# --------------------------------------------------------------------------
# Alias accuracy — reported, not gating
# --------------------------------------------------------------------------

def test_alias_accuracy_names_the_weak_variant():
    """Turns 'field accuracy is 0.87' into 'we are weak on Applicant'."""
    documents = []
    for _ in range(10):                       # strong on the dominant label
        documents.append((
            {"insured_name": "Rivera"}, {"insured_name": "Rivera"},
            {"insured_name": "Named Insured"},
        ))
    for i in range(4):                        # weak on the rare one
        documents.append((
            {"insured_name": "Rivera"}, {"insured_name": "Wrong" if i < 3 else "Rivera"},
            {"insured_name": "Applicant"},
        ))

    report = score_alias_accuracy(documents)
    assert report.accuracy("insured_name", "Named Insured") == 1.0
    assert report.accuracy("insured_name", "Applicant") == pytest.approx(0.25)

    weakest = report.weakest_variants()
    assert weakest[0][:2] == ("insured_name", "Applicant")
    assert report.spread("insured_name") == pytest.approx(0.75)


def test_alias_accuracy_is_not_a_gating_metric():
    """A rare variant has too little support for a stable threshold; gating on
    it would block good models on noise."""
    assert "alias_accuracy" not in GATING_METRICS
    assert "confusable_misattribution_rate" in GATING_METRICS


# --------------------------------------------------------------------------
# LoB, ECE, mode accuracy
# --------------------------------------------------------------------------

def test_lob_accuracy_is_reported_per_value():
    """A rare class must not hide inside a healthy aggregate (arch §0b)."""
    documents = [({"line_of_business": ["workers_comp"]}, {"line_of_business": ["workers_comp"]})] * 9
    documents.append(({"line_of_business": ["umbrella"]}, {"line_of_business": ["property"]}))

    report = score_lob(documents)
    assert report.overall == pytest.approx(0.9)
    assert report.accuracy_by_value()["workers_comp"] == 1.0
    assert report.accuracy_by_value()["umbrella"] == 0.0     # invisible in the aggregate


def test_unmeasured_lob_values_are_reported():
    """An unmeasured class is not a passing class."""
    report = score_lob([({"line_of_business": ["workers_comp"]}, {"line_of_business": ["workers_comp"]})])
    assert "umbrella" in report.unmeasured_values()


def test_ece_detects_overconfidence():
    """A model 95% confident and 50% correct is badly calibrated whatever its
    accuracy is."""
    overconfident = expected_calibration_error([0.95] * 10, [True] * 5 + [False] * 5)
    well_calibrated = expected_calibration_error([0.5] * 10, [True] * 5 + [False] * 5)
    assert overconfident > 0.4
    assert well_calibrated < 0.05


def test_mode_accuracy_carries_document_counts():
    """The ViT gate needs the counts: below its floor it declines to decide."""
    docs = [({"a": "x"}, {"a": "x"}, "image_only", False) for _ in range(3)]
    docs += [({"a": "x"}, {"a": "y"}, "ocr_plus_image", True) for _ in range(2)]

    report = score_by_mode(docs)
    assert report.accuracy_by_mode["image_only"] == 1.0
    assert report.documents_by_mode["image_only"] == 3
    assert report.scanned_documents == 2
    assert report.error_mix()                       # errors classified for the gate


# --------------------------------------------------------------------------
# The promotion gate
# --------------------------------------------------------------------------

def _metrics(**over) -> dict[str, float]:
    """A metric set that clears every §0d floor, so a test that wants a block
    creates one rather than inheriting it from a fixture that never passed."""
    base = {name: 0.96 for name in GATING_METRICS}
    # Schema validity has the highest floor (0.98) because a structurally
    # invalid extraction is unusable rather than merely inaccurate.
    base["schema_validity_rate"] = 1.0
    base["ece_confidence"] = 0.04                   # lower is better
    base["confusable_misattribution_rate"] = 0.02   # lower is better
    base["false_null_rate"] = 0.03                  # lower is better
    base["auto_accept_error_rate"] = 0.01           # lower is better
    base.update(over)
    return base


def _passing(**over) -> dict[str, float]:
    return _metrics(**over)


def _paired(metric: str, current: float, candidate: float, n: int = 40):
    """Per-document scores that produce a decisive interval.

    Constant within each arm on purpose: the bootstrap then has no within-arm
    variance, so the interval is tight and the test is about the gate's logic
    rather than about sampling luck."""
    return {metric: ([current] * n, [candidate] * n)}


def test_an_improved_candidate_passes():
    result = promotion_gate(_metrics(field_exact_match=0.98), _metrics())
    assert result.passed, result.report()


def test_a_drop_inside_the_noise_margin_no_longer_blocks():
    """THE v1 defect. A flat 0.001 tolerance at pilot volume was 250x finer than
    the eval set could resolve, so a genuinely-equal model was blocked by float
    jitter on one of twelve metrics."""
    result = promotion_gate(_metrics(field_exact_match=0.9595), _metrics())
    assert result.passed, result.report()


def test_a_drop_beyond_the_margin_still_blocks():
    """Not a softening: the margin is per-metric and calibrated to what the eval
    set can resolve, not to what is convenient."""
    result = promotion_gate(_metrics(field_exact_match=0.90), _metrics())
    assert not result.passed
    assert "field_exact_match" in result.failed_gates


def test_a_single_regression_blocks_promotion():
    """Any one metric. That is what 'every gating metric' means."""
    result = promotion_gate(_metrics(list_field_recall=0.85), _metrics())
    assert not result.passed
    assert "list_field_recall" in result.failed_gates


def test_lob_regression_alone_blocks_promotion():
    result = promotion_gate(_metrics(lob_detection_accuracy=0.80), _metrics())
    assert not result.passed


def test_confusable_regression_alone_blocks_promotion():
    """Rising misattribution is a regression even though the number goes UP."""
    result = promotion_gate(_metrics(confusable_misattribution_rate=0.09), _metrics())
    assert not result.passed
    assert "confusable_misattribution_rate" in result.failed_gates


def test_error_rate_metrics_are_scored_in_the_right_direction():
    """Treating ECE like accuracy would promote a model that got worse at
    exactly the thing hardest to notice."""
    improved = promotion_gate(_metrics(ece_confidence=0.02), _metrics())
    assert improved.passed


def test_an_unmeasured_metric_blocks_promotion():
    """A metric that was not measured has not passed."""
    candidate = _metrics()
    del candidate["image_only_accuracy"]
    result = promotion_gate(candidate, _metrics())
    assert not result.passed
    assert "image_only_accuracy" in result.failed_gates


def test_the_first_version_has_nothing_to_regress_against():
    result = promotion_gate(_metrics(), None)
    assert result.passed and result.is_first_version


def test_a_continued_foundation_needs_cross_type_evidence():
    """Continued training compounds drift, and a patch aimed at one type can
    quietly degrade the others (arch §12)."""
    result = promotion_gate(_metrics(), _metrics(), continued_from="foundation-v1")
    assert not result.passed
    assert "cross_type_regression_evidence" in result.failed_gates


def test_a_continued_foundation_passes_with_evidence():
    result = promotion_gate(
        _metrics(), _metrics(), continued_from="foundation-v1",
        cross_type_evidence={
            "acord": {"candidate": _metrics(), "current": _metrics()},
            "lossrun": {"candidate": _metrics(), "current": _metrics()},
        },
    )
    assert result.passed


def test_cross_type_regression_blocks_a_continued_foundation():
    result = promotion_gate(
        _metrics(), _metrics(), continued_from="foundation-v1",
        cross_type_evidence={"acord": {"candidate": _metrics(field_exact_match=0.70),
                                       "current": _metrics()}},
    )
    assert not result.passed
    assert any("acord" in gate for gate in result.failed_gates)


def test_the_gate_has_no_silent_bypass():
    """An override EXISTS under arch v2.1 §15.5 — but there is no parameter that
    forces a pass without attribution. v1 had no override at all, reasoning that
    a waivable gate is a suggestion. That was right about the risk and wrong
    about the remedy: the v1 gate demanded improvement on twelve metrics within
    0.001, which at pilot volume is finer than the eval set can resolve, so the
    rule would have been broken in practice rather than in the open."""
    import inspect

    signature = inspect.signature(promotion_gate)
    for forbidden in ("force", "skip", "ignore_regressions", "bypass"):
        assert forbidden not in signature.parameters


def _span(path: str, value, logprobs: list[float], mapped: bool = True) -> FieldSpan:
    span = FieldSpan(field_path=path, value=value, char_start=0, char_end=1)
    span.token_logprobs = logprobs
    span.mapped = mapped
    if not mapped:
        span.reason = "no token span"
    return span


def test_an_override_must_be_attributable():
    """The waiver worth preventing is the one nobody can trace later."""
    from evaluation.gating import GateError, GateOverrideRecord

    with pytest.raises(GateError, match="name the person"):
        GateOverrideRecord("ci", "x" * 30, ["ece_confidence"]).validate()
    with pytest.raises(GateError, match="written reason"):
        GateOverrideRecord("A. Reviewer", "too busy", ["ece_confidence"]).validate()
    with pytest.raises(GateError, match="name the gates"):
        GateOverrideRecord("A. Reviewer", "y" * 30, []).validate()

    GateOverrideRecord(
        "A. Reviewer",
        "FP8 ECE regressed 0.004 on a 30-document slice; shipping for the pilot",
        ["ece_confidence"],
    ).validate()


def test_an_override_lifts_only_the_gates_it_names():
    """Waiving a gate that passed would make the record say something untrue
    about what the approver decided."""
    from evaluation.gating import GateOverrideRecord

    # Below its §0d floor of 0.95, so the gate genuinely fails on it.
    candidate = {**_passing(), "doc_type_classifier_accuracy": 0.90}
    result = promotion_gate(
        candidate, None,
        override=GateOverrideRecord(
            "A. Reviewer",
            "classifier retrain is tracked in FID-118; shipping the extractor gain now",
            ["doc_type_classifier_accuracy"],
        ),
    )
    assert result.passed, result.report()
    assert result.waived == ["doc_type_classifier_accuracy"]
    assert "doc_type_classifier_accuracy" not in result.failed_gates


def test_an_override_clears_gates_whose_reasons_do_not_start_with_their_name():
    """The waiver matched reasons by their opening words. The cross-type reason
    starts "cross-type evidence ..." / "the continued run ...", never with its gate
    name, so waiving it recorded the waiver and left the release blocked."""
    from evaluation.gating import GateOverrideRecord

    result = promotion_gate(
        _metrics(), _metrics(), continued_from="foundation-v1",
        override=GateOverrideRecord(
            "A. Reviewer",
            "cross-type slices are being relabelled under FID-120; this patch is acord-only",
            ["cross_type_regression_evidence"],
        ),
    )
    assert result.waived == ["cross_type_regression_evidence"]
    assert result.passed, result.report()
    assert not result.blocking_reasons


def test_a_waived_pass_is_not_recorded_as_a_clean_pass():
    """"Passed with a waiver" must never be indistinguishable from "passed"."""
    from evaluation.gating import GateOverrideRecord, apply_to_manifest

    class _Promotion:
        beat_previous_on_all_gates = None
        failed_gates: list = []
        gated_against = None

    class _Manifest:
        promotion = _Promotion()
        status = "trained"

    result = promotion_gate(
        {**_passing(), "doc_type_classifier_accuracy": 0.90}, None,
        override=GateOverrideRecord(
            "A. Reviewer", "classifier retrain tracked in FID-118, shipping anyway",
            ["doc_type_classifier_accuracy"],
        ),
    )
    manifest = apply_to_manifest(result, _Manifest())
    assert manifest.promotion.beat_previous_on_all_gates is False
    assert any("WAIVED" in g for g in manifest.promotion.failed_gates)


def test_min_aggregation_is_sensitive_to_the_weakest_token():
    """Nine confident tokens and one uncertain one is worth checking; mean would
    average that away."""
    logprobs = [-0.01] * 9 + [-2.5]
    assert aggregate_span(logprobs, "min") < 0.1
    assert aggregate_span(logprobs, "mean") > 0.8


def test_geomean_does_not_underflow_on_long_spans():
    """Multiplying probabilities directly would reach zero; log space does not."""
    assert aggregate_span([-3.0] * 200, "geomean") > 0.0


def test_unmapped_fields_get_no_confidence_and_are_surfaced():
    """A field with no confidence must not be indistinguishable from a confident
    one — that is backwards for a risk signal."""
    spans = {
        "carrier": _span("carrier", "Sentinel", [-0.01, -0.02]),
        "policy_number": _span("policy_number", "WC-1", [], mapped=False),
    }
    confidences = field_confidences(spans)
    assert confidences["carrier"].is_usable
    assert not confidences["policy_number"].is_usable
    assert unusable_fields(confidences) == [("policy_number", "no token span")]
    assert overall_confidence(confidences) > 0.9      # the unusable one is excluded


def test_empty_span_cannot_be_aggregated():
    with pytest.raises(ConfidenceError, match="empty logprob span"):
        aggregate_span([], "min")


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def test_temperature_scaling_reduces_ece_on_an_overconfident_model():
    """The failure calibration exists for: a fine-tuned model whose confidence
    barely varies between right and wrong answers."""
    confidences = [0.95] * 60 + [0.92] * 40
    correctness = [True] * 45 + [False] * 15 + [True] * 20 + [False] * 20

    params = fit_calibration(confidences, correctness, doc_type="policy", model_version="v1")
    assert params.improved
    assert params.ece_after < params.ece_before
    assert params.temperature > 1.0             # the overconfident direction


def test_isotonic_also_reduces_ece():
    confidences = [0.9] * 50 + [0.6] * 50
    correctness = [True] * 30 + [False] * 20 + [True] * 40 + [False] * 10

    params = fit_calibration(
        confidences, correctness, doc_type="policy", model_version="v1", method="isotonic"
    )
    assert params.ece_after <= params.ece_before
    assert params.breakpoints


def test_isotonic_breakpoints_are_monotonic():
    """Confidence that falls as raw confidence rises would be uninterpretable."""
    params = fit_calibration(
        [i / 100 for i in range(100)], [i % 3 != 0 for i in range(100)],
        doc_type="policy", model_version="v1", method="isotonic",
    )
    values = [y for _x, y in params.breakpoints]
    assert values == sorted(values)


def test_fitting_on_training_data_is_refused():
    """It measures memorisation, looks excellent, and fails in production."""
    with pytest.raises(CalibrationError, match="BOTH"):
        assert_held_out({"policy_0001", "policy_0002"}, {"policy_0002", "policy_0003"})


def test_missing_calibration_raises_rather_than_passing_raw_through():
    """THE guard. Raw confidence from a fine-tuned model is systematically
    overconfident; returning it unmarked would under-route review."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    with pytest.raises(CalibrationError, match="no calibration exists"):
        load_calibration("v2", "policy", client)


def test_calibration_round_trips_through_blob():
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    params = CalibrationParams(method="temperature", doc_type="policy",
                               model_version="v2", temperature=1.8)
    client.write_json(paths.calibration_params("v2", "policy"), params.as_dict())

    loaded = load_calibration("v2", "policy", client)
    assert loaded.temperature == 1.8 and loaded.method == "temperature"


def test_calibration_flags_fields_below_the_review_threshold():
    spans = {
        "carrier": _span("carrier", "Sentinel", [-0.001]),
        "valuation_date": _span("valuation_date", "2026-03-31", [-3.0]),
    }
    params = CalibrationParams(method="temperature", doc_type="lossrun",
                               model_version="v1", temperature=1.0)

    result = apply_calibration(field_confidences(spans), params, review_threshold=0.70)
    assert not result.fields["carrier"].needs_review
    assert result.fields["valuation_date"].needs_review
    assert "valuation_date:low_confidence" in result.review_flags

    output = calibrated_output(result)
    assert set(output["fields"]["carrier"]) == {"value", "confidence"}


def test_a_field_without_confidence_is_routed_to_review():
    spans = {"policy_number": _span("policy_number", "WC-1", [], mapped=False)}
    params = CalibrationParams(method="temperature", doc_type="policy",
                               model_version="v1", temperature=1.0)
    result = apply_calibration(field_confidences(spans), params)
    assert result.fields["policy_number"].needs_review
    assert "policy_number:no_confidence" in result.review_flags


# --------------------------------------------------------------------------
# List completeness — what logprobs cannot see
# --------------------------------------------------------------------------

def test_a_stated_count_mismatch_flags_the_list():
    """6 of 8 claims: the 2 missing rows produce no tokens, so nothing else
    catches this."""
    signal = check_completeness("claims", extracted_rows=6, stated_count=8)
    assert signal.flagged
    assert signal.missing_rows == 2
    assert signal.confidence < 1.0


def test_a_detected_row_mismatch_flags_the_list_independently():
    """The second cross-check, from MinerU's table row counts."""
    signal = check_completeness("claims", extracted_rows=6, detected_rows=9)
    assert signal.flagged


def test_structure_count_tolerates_a_small_difference():
    """A continuation header or totals row can shift the detected count by one;
    the document's stated count is an assertion and is compared exactly."""
    assert not check_completeness("claims", extracted_rows=8, detected_rows=9).flagged
    assert check_completeness("claims", extracted_rows=8, stated_count=9).flagged


def test_no_cross_check_available_is_not_the_same_as_verified():
    signal = check_completeness("claims", extracted_rows=6)
    assert not signal.flagged
    assert signal.confidence == 0.5             # unverified, not confirmed
    assert "could not be verified" in signal.reasons[0]


def test_a_complete_list_is_not_flagged():
    assert not check_completeness("claims", 8, stated_count=8, detected_rows=8).flagged


def test_document_check_uses_ocr_row_counts_for_the_pages_read():
    """A document is compared against the pages it actually read, not the
    whole file."""
    extraction = {"claims": [{"claim_number": f"C{i}"} for i in range(6)],
                  "total_claims_reported": 8}
    signals = check_document(
        extraction, "lossrun",
        ocr_meta={"table_row_counts": {"1": 5, "2": 4, "3": 99}},
        pages_used=[1, 2],
    )
    assert signals["claims"].flagged
    assert signals["claims"].detected_rows == 9      # page 3 excluded


def test_a_flagged_list_reaches_review_regardless_of_value_confidence():
    """The values present may be perfect while the list is incomplete."""
    signals = {"claims": check_completeness("claims", 6, stated_count=8)}
    flags = merge_review_flags(signals, existing_flags=[])
    assert "claims:row_count_mismatch" in flags


def test_a_hallucinated_list_is_not_maximally_confident():
    """The `or` chain read a stated count of 0 as absent and fell through to the
    extracted count itself, so the pure-hallucination case scored 1.0 — in a
    value that ships in the output contract."""
    from calibration.list_completeness import check_completeness

    signal = check_completeness("claims", 4, stated_count=0)
    assert signal.flagged
    assert signal.confidence == 0.0


def test_over_extraction_is_scored_like_under_extraction():
    """`min(1.0, extracted/reference)` reported a surplus as perfect."""
    from calibration.list_completeness import check_completeness

    short = check_completeness("claims", 4, stated_count=5)
    surplus = check_completeness("claims", 6, stated_count=5)
    assert short.flagged and surplus.flagged
    assert surplus.confidence < 1.0


def test_a_perfect_list_is_fully_confident():
    from calibration.list_completeness import check_completeness

    signal = check_completeness("claims", 4, stated_count=4)
    assert not signal.flagged and signal.confidence == 1.0


def test_a_coincidental_value_collision_is_not_misattribution():
    """The registry filter was computed and never applied, so unrelated
    duplicate values entered a gating metric and could block a promotion."""
    from evaluation.metrics.confusable import score_misattribution

    expected = {"insured_name": "Rivera Fabrication LLC", "producer": "Hanover Risk Partners"}
    got = {"insured_name": "Rivera Fabrication LLC", "producer": "Hanover Risk Partners"}
    assert score_misattribution(expected, got, "policy", source_id="policy_0001").rate == 0.0


def test_the_lob_balance_target_is_reachable():
    """Five values against a 0.20 floor sum to exactly 1.00, so any corpus that
    was not perfectly uniform failed — the flag carried no information."""
    from common.lob import compute_coverage

    coverage = compute_coverage(
        ["workers_comp"] * 21 + ["general_liability"] * 20 + ["commercial_auto"] * 20
        + ["property"] * 20 + ["umbrella"] * 19
    )
    assert coverage.is_balanced, coverage.warning()


# --------------------------------------------------------------------------
# Checkpoint selection (arch v2.1 §11.2)
# --------------------------------------------------------------------------

def _ckpts(*steps):
    return [f"/staging/adapters/foundation/v2/checkpoint-{n}" for n in steps]


def test_selection_reads_generated_field_f1_not_loss():
    """Loss is averaged over every token, so it is dominated by the easy copy
    tokens — schema keys, JSON punctuation, boilerplate — the model gets right
    within a hundred steps. A checkpoint can improve on loss while getting worse
    at the values, which is the only thing the gate reads."""
    from evaluation.checkpoint_eval import select_best

    scores = {"checkpoint-100": 0.70, "checkpoint-200": 0.88, "checkpoint-300": 0.81}

    def scorer(path):
        return {"field_normalized_match": scores[path.rsplit("/", 1)[-1]]}

    report = select_best(_ckpts(100, 200, 300), scorer)
    assert report.selected.endswith("checkpoint-200")
    assert report.margin == pytest.approx(0.07)


def test_the_best_loss_checkpoint_is_scored_even_when_it_is_not_recent():
    """Exactly the interesting case: early stopping liked it and training
    continued past it."""
    from evaluation.checkpoint_eval import select_candidates

    candidates = select_candidates(
        _ckpts(100, 200, 300, 400, 500),
        best_loss="/staging/adapters/foundation/v2/checkpoint-100",
    )
    steps = [c.step for c in candidates]
    assert steps == [100, 300, 400, 500], "the last three plus the best-loss one"
    assert next(c for c in candidates if c.step == 100).is_best_loss


def test_a_disagreement_between_loss_and_f1_is_surfaced():
    """When it fires, validation loss would have shipped a different model. That
    is the whole reason this job exists, so it is recorded rather than buried."""
    from evaluation.checkpoint_eval import select_best

    def scorer(path):
        return {"field_normalized_match": 0.9 if path.endswith("300") else 0.6}

    report = select_best(
        _ckpts(100, 200, 300),
        scorer,
        best_loss="/staging/adapters/foundation/v2/checkpoint-100",
    )
    assert report.selected.endswith("checkpoint-300")
    assert report.loss_and_f1_disagreed is True
    assert report.as_dict()["best_loss_checkpoint"].endswith("checkpoint-100")


def test_a_checkpoint_that_cannot_be_scored_is_skipped_not_zeroed():
    """A scorer that fell over on one candidate says nothing about that
    candidate's quality; scoring it zero would silently remove it from
    contention."""
    from evaluation.checkpoint_eval import select_best

    def scorer(path):
        if path.endswith("200"):
            raise RuntimeError("vLLM OOM")
        return {"field_normalized_match": 0.8}

    report = select_best(_ckpts(100, 200, 300), scorer)
    assert len(report.skipped) == 1 and "OOM" in report.skipped[0][1]
    assert report.selected and not report.selected.endswith("200")


def test_selecting_from_nothing_is_refused():
    """Merging an arbitrary checkpoint would ship a model nobody measured."""
    from evaluation.checkpoint_eval import CheckpointEvalError, select_best

    with pytest.raises(CheckpointEvalError, match="no checkpoints"):
        select_best([], lambda _p: {"field_normalized_match": 1.0})

    with pytest.raises(CheckpointEvalError, match="no checkpoint could be scored"):
        select_best(_ckpts(100), lambda _p: (_ for _ in ()).throw(RuntimeError("boom")))


def test_a_tie_breaks_toward_the_later_checkpoint():
    """Picking the earlier one on a tie would quietly prefer an under-trained
    checkpoint whenever the metric saturates."""
    from evaluation.checkpoint_eval import select_best

    report = select_best(_ckpts(100, 200, 300), lambda _p: {"field_normalized_match": 0.9})
    assert report.selected.endswith("checkpoint-300")
    assert report.margin == 0.0


def test_an_unparseable_checkpoint_name_is_refused():
    """Ordering by step is what 'the last three' means, and a silent 0 would put
    a real checkpoint at the front of the list."""
    from evaluation.checkpoint_eval import CheckpointEvalError, checkpoint_step

    assert checkpoint_step("/a/b/checkpoint-450") == 450
    with pytest.raises(CheckpointEvalError, match="cannot read a step"):
        checkpoint_step("/a/b/final-model")


def test_the_selection_record_says_how_close_it_was():
    """A margin inside ordinary eval noise means the selection was close to
    arbitrary, and a later regression reads better against that than against a
    bare filename."""
    from evaluation.checkpoint_eval import select_best

    def scorer(path):
        return {"field_normalized_match": 0.9001 if path.endswith("300") else 0.9}

    record = select_best(_ckpts(100, 200, 300), scorer).as_dict()
    assert record["selection_metric"] == "field_normalized_match"
    assert 0 < record["margin_over_runner_up"] < 0.005
    assert len(record["candidates"]) == 3


# --------------------------------------------------------------------------
# Paired bootstrap and the statistical gate (arch v2.1 §15.5)
# --------------------------------------------------------------------------

def test_the_interval_is_paired_not_independent():
    """Both models saw the same documents. Drawing separate index sets for each
    side would compare two different document samples and widen every interval
    for nothing — which on a 30-document eval set is the difference between a
    usable gate and an unpassable one."""
    from evaluation.bootstrap import paired_bootstrap

    # Wildly varying documents, but the candidate is uniformly +0.10 on each.
    current = [0.1, 0.9, 0.3, 0.7, 0.5] * 8
    candidate = [c + 0.10 for c in current]

    ci = paired_bootstrap("field_normalized_match", current, candidate)
    assert ci.observed == pytest.approx(0.10, abs=1e-9)
    # The pairing cancels the between-document variance entirely.
    assert ci.width < 1e-6, f"pairing was lost; interval width {ci.width}"
    assert ci.improved


def test_a_genuinely_equal_model_is_non_inferior():
    """The v1 gate blocked this. Twelve metrics each needing not to move down by
    more than 0.001 gave a genuinely-equal model roughly a 0.02% chance of
    passing all twelve."""
    import random

    from evaluation.bootstrap import paired_bootstrap

    rng = random.Random(7)
    current = [rng.random() for _ in range(60)]
    candidate = [c + rng.gauss(0, 0.02) for c in current]

    ci = paired_bootstrap("field_normalized_match", current, candidate)
    assert ci.non_inferior(0.010), ci.describe()
    assert not ci.improved, "noise is not improvement"


def test_a_real_regression_is_not_non_inferior():
    from evaluation.bootstrap import paired_bootstrap

    current = [0.9] * 40
    candidate = [0.7] * 40
    ci = paired_bootstrap("list_field_recall", current, candidate)

    assert not ci.non_inferior(0.010)
    assert ci.upper < 0


def test_a_single_document_is_inconclusive_and_inconclusive_does_not_pass():
    """One document resamples to itself every time, so a naive interval would be
    a point and the gate would read it as certainty. Widened to span the whole
    range instead — which fails non-inferiority, and should: absence of evidence
    is not evidence of absence, the same rule `require_all_measured` applies to
    an unmeasured metric."""
    from evaluation.bootstrap import paired_bootstrap

    ci = paired_bootstrap("field_exact_match", [0.5], [1.0])
    assert ci.lower == -1.0 and ci.upper == 1.0
    assert not ci.improved, "one document cannot establish an improvement"
    assert not ci.non_inferior(0.01), "one document cannot establish non-inferiority either"


def test_mismatched_score_lengths_are_refused():
    """Aligned by index is what the pairing MEANS; a length mismatch is two
    different document sets being compared."""
    from evaluation.bootstrap import paired_bootstrap

    with pytest.raises(ValueError, match="aligned by index"):
        paired_bootstrap("m", [0.1, 0.2], [0.1])


def test_the_interval_is_reproducible():
    """A gate whose verdict changes on re-run is not a gate."""
    from evaluation.bootstrap import paired_bootstrap

    scores = ([0.8, 0.9, 0.7, 0.85] * 10, [0.82, 0.88, 0.75, 0.9] * 10)
    first = paired_bootstrap("m", *scores)
    second = paired_bootstrap("m", *scores)
    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_the_wilson_bound_is_sane_on_a_small_sample():
    """At n=30 with two errors the normal approximation gives nonsense, and n=30
    is the size these eval sets actually are."""
    from evaluation.bootstrap import binomial_upper_bound

    bound = binomial_upper_bound(2, 30)
    assert 0.06 < bound < 0.25, bound
    assert binomial_upper_bound(0, 0) == 1.0, "no trials means no evidence, not zero risk"


def test_a_floor_failure_blocks_the_first_release():
    """v1's gate passed a first version on ZERO measured metrics: with no
    baseline there was nothing to regress against, so nothing blocked."""
    from evaluation.gating import promotion_gate

    assert not promotion_gate({}, None).passed
    below = _metrics(field_normalized_match=0.50)
    result = promotion_gate(below, None)
    assert not result.passed
    assert "field_normalized_match" in result.failed_gates


def test_the_bootstrap_decides_when_per_document_scores_are_supplied():
    """A point comparison cannot tell a real 1pp drop from sampling noise on a
    30-document eval set. The interval can, and it is recorded as the basis."""
    from evaluation.gating import promotion_gate

    current, candidate = _metrics(), _metrics(field_exact_match=0.955)
    result = promotion_gate(
        candidate, current,
        per_document=_paired("field_exact_match", 0.96, 0.955),
    )
    verdict = next(v for v in result.verdicts if v.name == "field_exact_match")
    assert verdict.basis == "paired_bootstrap"
    assert verdict.interval is not None


def test_a_release_that_is_merely_no_worse_does_not_pass():
    """Without an improvement requirement a model could pass forever on
    non-inferiority alone — every release no worse than the last, none better,
    and the cycle producing nothing."""
    from evaluation.gating import promotion_gate

    result = promotion_gate(
        _metrics(), _metrics(),
        per_document={
            **_paired("field_normalized_match", 0.96, 0.96),
            **_paired("list_field_recall", 0.96, 0.96),
        },
    )
    assert not result.passed
    assert "improvement" in result.failed_gates


def test_a_release_that_fixes_a_documented_defect_satisfies_improvement():
    """Some releases exist to fix something rather than to score higher."""
    from evaluation.gating import promotion_gate

    result = promotion_gate(
        _metrics(), _metrics(),
        per_document={
            **_paired("field_normalized_match", 0.96, 0.96),
            **_paired("list_field_recall", 0.96, 0.96),
        },
        fixes_defect="FID-204: claim rows dropped on page boundaries",
    )
    assert result.passed, result.report()


def test_the_improvement_rule_does_not_fire_on_missing_evidence():
    """No per-document scores means "improved" is unknowable rather than false.
    Blocking there would make the rule fire on absent evidence rather than on an
    absent improvement."""
    from evaluation.gating import promotion_gate

    result = promotion_gate(_metrics(), _metrics())
    assert "improvement" not in result.failed_gates


def test_the_improvement_rule_left_unenforced_is_said_in_the_decision():
    """Only logged, a release that passed with the rule unenforced read in its
    decision exactly like one that improved."""
    from evaluation.gating import promotion_gate

    result = promotion_gate(_metrics(), _metrics())
    assert result.passed
    assert any("improvement rule was not enforced" in w for w in result.warnings)
    assert result.as_dict()["warnings"] == result.warnings
    # Nothing to say for a first release, or one that names the defect it fixes.
    assert not promotion_gate(_metrics(), None).warnings
    assert not promotion_gate(_metrics(), _metrics(), fixes_defect="TICKET-12").warnings


def test_a_metric_one_document_measures_is_a_point_estimate_not_a_permanent_block():
    """At n=1 the bootstrap widens the interval to [-1, 1], which fails
    non-inferiority: a frozen set holding one scanned document blocked every
    later candidate on scanned_accuracy, identical scores or not, and the set
    cannot grow once frozen."""
    from evaluation.gating import promotion_gate

    result = promotion_gate(
        _metrics(scanned_accuracy=0.80), _metrics(scanned_accuracy=0.80),
        per_document={
            **_paired("field_normalized_match", 0.90, 0.96),
            **_paired("list_field_recall", 0.96, 0.96),
            "scanned_accuracy": ([0.80], [0.80]),
        },
    )
    assert result.passed, result.report()
    verdict = next(v for v in result.verdicts if v.name == "scanned_accuracy")
    assert verdict.basis == "point_estimate" and verdict.non_inferior and not verdict.improved
    assert any("scanned_accuracy" in w and "1 paired document" in w for w in result.warnings)

    # Still a point comparison against δ: a real drop on that one document blocks.
    dropped = promotion_gate(
        _metrics(scanned_accuracy=0.70), _metrics(scanned_accuracy=0.80),
        per_document={**_paired("field_normalized_match", 0.90, 0.96),
                      "scanned_accuracy": ([0.80], [0.70])},
    )
    assert "scanned_accuracy" in dropped.failed_gates


def test_weights_make_each_side_a_weighted_mean():
    from evaluation.bootstrap import paired_bootstrap

    current, candidate = [0.6, 1.0] * 10, [1.0, 0.85] * 10
    plain = paired_bootstrap("m", current, candidate)
    ones = paired_bootstrap("m", current, candidate, current_weights=[1.0] * 20,
                            candidate_weights=[1.0] * 20)
    assert ones.observed == pytest.approx(plain.observed)
    assert ones.lower == pytest.approx(plain.lower) and ones.upper == pytest.approx(plain.upper)

    weights = [1.0, 4.0] * 10
    weighted = paired_bootstrap("m", current, candidate, current_weights=weights,
                                candidate_weights=weights)
    assert weighted.observed == pytest.approx((1.0 + 4 * 0.85) / 5 - (0.6 + 4 * 1.0) / 5)

    with pytest.raises(ValueError, match="not both"):
        paired_bootstrap("m", current, candidate, current_weights=weights, aggregate=max)
    with pytest.raises(ValueError, match="aligned with the scores"):
        paired_bootstrap("m", current, candidate, current_weights=[1.0], candidate_weights=weights)
    with pytest.raises(ValueError, match="above zero"):
        paired_bootstrap("m", current, candidate, current_weights=weights, candidate_weights=[0.0] * 20)


def test_weights_for_one_side_only_are_refused():
    """A weighted mean against an unweighted one is a difference between two
    different numbers: identical scores bootstrapped as +0.15, improved."""
    from evaluation.bootstrap import paired_bootstrap

    scores = [0.5, 1.0] * 10
    for side in ("current_weights", "candidate_weights"):
        with pytest.raises(ValueError, match="one side only"):
            paired_bootstrap("m", scores, scores, **{side: [1.0, 4.0] * 10})
    same = paired_bootstrap("m", scores, scores, current_weights=[1.0, 4.0] * 10,
                            candidate_weights=[1.0, 4.0] * 10)
    assert same.observed == 0.0 and not same.improved


def test_a_baseline_on_another_yardstick_is_judged_on_floors_alone():
    """Decision 2026-10-22: the promoted release's scores from another frozen
    set, tenant or scoring code are not compared - not as a bootstrap, not as a
    point estimate, not for the improvement rule. The floors still apply."""
    metrics = {name: 0.96 for name in GATING_METRICS}
    metrics.update(ece_confidence=0.04, confusable_misattribution_rate=0.02, false_null_rate=0.03,
                   auto_accept_error_rate=0.01, schema_validity_rate=1.0)
    better = {**metrics, "field_normalized_match": 0.995, "list_field_recall": 0.999}
    compared = promotion_gate(dict(metrics), better)
    assert "field_normalized_match" in compared.failed_gates

    reason = "the set was frozen again"
    result = promotion_gate(dict(metrics), better, baseline_not_comparable=reason,
                            per_document={"list_field_recall": ([1.0] * 20, [0.5] * 20)})
    assert result.passed and not result.is_first_version, result.report()
    assert result.baseline_not_comparable == reason
    assert {v.basis for v in result.verdicts} == {"floor_only"}
    assert any("were not compared" in w and reason in w for w in result.warnings)
    assert "another yardstick" in result.report()

    floor = promotion_gate({**metrics, "list_field_recall": 0.60}, better,
                           baseline_not_comparable=reason)
    assert floor.failed_gates == ["list_field_recall"]


def test_an_override_must_name_real_gates_and_says_what_it_did_not_waive():
    from evaluation.gating import GateError, GateOverrideRecord, is_gate

    reason = "classifier retrain tracked in FID-118, shipping the extractor gain now"
    with pytest.raises(GateError, match="not a gate"):
        GateOverrideRecord("A. Reviewer", reason, ["doc_type_classifier"]).validate()
    assert all(is_gate(g) for g in [*GATING_METRICS, "improvement", "cross_type_regression_evidence",
                                    "cross_type:acord"])
    assert not is_gate("cross_type:invoice") and not is_gate("alias_accuracy")

    result = promotion_gate(
        {**_passing(), "doc_type_classifier_accuracy": 0.90}, None,
        override=GateOverrideRecord(reason=reason, approver="A. Reviewer",
                                    waived_gates=["doc_type_classifier_accuracy", "ece_confidence"]),
    )
    assert result.passed and result.waived == ["doc_type_classifier_accuracy"]
    assert any("['ece_confidence'], which did not fail" in w for w in result.warnings)


@pytest.mark.parametrize("value", [" ", "-", "n/a", "none", "fix", "TBD", ""])
def test_a_blank_or_placeholder_defect_does_not_lift_the_improvement_rule(value):
    """`--fixes-defect " "` lifted the rule: any text at all did."""
    from evaluation.gating import GateError, documented_defect, promotion_gate

    with pytest.raises(GateError, match="names no defect"):
        promotion_gate(_metrics(), _metrics(), fixes_defect=value)
    with pytest.raises(GateError):
        documented_defect(value)


def test_a_ticket_id_names_a_defect():
    from evaluation.gating import documented_defect

    assert documented_defect("  TICKET-12 ") == "TICKET-12"
    assert documented_defect("FID-204: claim rows dropped on page boundaries").startswith("FID-204")


def test_every_pending_metric_names_what_turns_it_on():
    """A metric listed as gating that nothing produces makes the gate
    permanently unpassable — precisely the defect this rewrite exists to fix. So
    pending ones are separated, and each says which phase activates it."""
    from evaluation.gating import GATING_METRICS, PENDING_GATING_METRICS

    assert not set(GATING_METRICS) & set(PENDING_GATING_METRICS)
    for metric, reason in PENDING_GATING_METRICS.items():
        assert reason.strip(), f"{metric} does not say what turns it on"


# --------------------------------------------------------------------------
# Feature-based confidence (arch v2.1 §5.1-5.4)
# --------------------------------------------------------------------------

def _feat(path="policy_number", value="WC-123", logprobs=None, **over):
    from calibration.features import build_features

    return build_features(
        field_path=path, value=value,
        logprobs=logprobs if logprobs is not None else [-0.1, -0.2, -0.05],
        document=over.pop("document", {}), **over,
    )


def test_length_is_always_a_feature_because_the_minimum_is_length_biased():
    """THE v1 defect. The minimum of n draws falls as n grows, so a long correct
    value scored lower than a short wrong one — 'ABC-1234567-01' is nine tokens
    and '2026' is one, and min-logprob called the policy number less trustworthy
    every single time."""
    from calibration.features import FieldFeatures

    short = _feat(value="2026", logprobs=[-0.3])
    long = _feat(value="ABC-1234567-01", logprobs=[-0.05] * 9 + [-0.3])

    assert short.min_logprob == long.min_logprob, "same weakest token by construction"
    assert "log_token_count" in FieldFeatures.feature_names()
    assert long.vector()[3] > short.vector()[3], "length must reach the calibrator"


def test_a_missing_check_is_not_a_failed_check():
    """Encoding "not checked" as 0 would make it indistinguishable from "checked
    and failed", and the calibrator would learn to distrust every field in
    image-only mode — where OCR agreement cannot be computed at all."""
    from calibration.features import FieldFeatures

    names = FieldFeatures.feature_names()
    absent = _feat(page_text=None).vector()
    failed = _feat(value="not-on-the-page", page_text="something else entirely").vector()

    ocr = names.index("ocr_agreement")
    present = names.index("ocr_agreement_present")
    assert absent[ocr] == 0.5 and absent[present] == 0.0
    assert failed[ocr] == 0.0 and failed[present] == 1.0


def test_ocr_agreement_is_none_in_image_only_mode():
    """Scoring it as disagreement would teach the calibrator that every
    image-only field is untrustworthy — a statement about the input mode rather
    than about the extraction."""
    from calibration.features import ocr_agreement

    assert ocr_agreement("WC-123", None) is None
    assert ocr_agreement("WC-123", "policy WC-123 effective") == 1.0
    assert ocr_agreement("WC-999", "policy WC-123 effective") == 0.0


def test_a_null_is_calibrated_as_its_own_class():
    """A null has no tokens, so it has no logprob signal at all — and treating
    the absence as maximum confidence is how a false null ships unreviewed."""
    empty = _feat(value=None, logprobs=[])
    assert empty.is_null
    assert empty.token_count == 0
    assert empty.vector()[FeatureNames().index("is_null")] == 1.0


def FeatureNames():
    from calibration.features import FieldFeatures

    return FieldFeatures.feature_names()


def test_rule_checks_distinguish_not_applicable_from_failed():
    """None means no rule applies; False means a rule applied and the value
    broke it. Collapsing them would make every unchecked field look wrong."""
    from calibration.features import rule_checks

    assert rule_checks("insured_name", "Acme", {}) is None
    assert rule_checks(
        "effective_date", "2026-01-01",
        {"effective_date": "2026-01-01", "expiration_date": "2027-01-01"},
    ) is True
    assert rule_checks(
        "effective_date", "2027-01-01",
        {"effective_date": "2027-01-01", "expiration_date": "2026-01-01"},
    ) is False


def test_a_total_that_does_not_sum_fails_its_rule_check():
    """The confident-but-inconsistent case, which logprobs by construction
    cannot see."""
    from calibration.features import rule_checks

    document = {"claims": [{"incurred": 100.0}, {"incurred": 50.0}]}
    assert rule_checks("total_incurred", 150.0, document) is True
    assert rule_checks("total_incurred", 900.0, document) is False


def test_a_field_type_with_too_little_data_gets_no_calibrator():
    """A calibrator fitted on forty instances produces numbers that look like
    probabilities and are not — and every §5.4 threshold is defined against a
    calibrated score."""
    from calibration.feature_calibrator import CalibratorError, fit_calibrators

    labelled = [(_feat(), i % 4 != 0) for i in range(40)]
    cal = fit_calibrators(labelled, release_id="release-2026.11.1", serving_format="bf16")

    identifier = cal.calibrators["identifier"]
    assert not identifier.enforced
    assert "below the" in (identifier.reason or "")
    assert cal.predict(_feat()) is None, "no number at all, rather than a default one"
    with pytest.raises(CalibratorError, match="no enforced calibrator"):
        identifier.predict(_feat())


def test_a_single_outcome_class_cannot_be_fitted():
    """Any curve fitted on it would report the class prior for every field
    regardless of its features."""
    from calibration.feature_calibrator import fit_calibrators

    labelled = [(_feat(), True) for _ in range(400)]
    cal = fit_calibrators(labelled, release_id="r", serving_format="bf16")
    assert not cal.calibrators["identifier"].enforced
    assert "same outcome" in (cal.calibrators["identifier"].reason or "")


def test_a_fitted_calibrator_separates_correct_from_incorrect():
    import random

    from calibration.feature_calibrator import fit_calibrators

    rng = random.Random(11)
    labelled = []
    for _ in range(600):
        correct = rng.random() < 0.8
        logprobs = [-0.05 - rng.random() * 0.1] * 3 if correct else [-1.5 - rng.random()] * 3
        labelled.append((
            _feat(logprobs=logprobs, page_text="WC-123" if correct else "nothing here"),
            correct,
        ))

    cal = fit_calibrators(labelled, release_id="r", serving_format="bf16")
    assert cal.calibrators["identifier"].enforced

    confident = cal.predict(_feat(logprobs=[-0.05] * 3, page_text="WC-123"))
    doubtful = cal.predict(_feat(logprobs=[-2.5] * 3, page_text="nothing here"))
    assert confident > doubtful, f"{confident} !> {doubtful}"


def test_calibrators_round_trip_through_json():
    """They ship inside a release bundle, so they have to survive the trip."""
    import json
    import random

    from calibration.feature_calibrator import CalibratorSet, fit_calibrators

    rng = random.Random(5)
    labelled = [(_feat(logprobs=[-rng.random()] * 3), rng.random() < 0.8) for _ in range(400)]
    original = fit_calibrators(labelled, release_id="release-2026.11.1", serving_format="fp8")
    restored = CalibratorSet.from_dict(json.loads(original.to_json()))

    probe = _feat()
    assert restored.predict(probe) == pytest.approx(original.predict(probe))


# --------------------------------------------------------------------------
# Risk-controlled thresholds (arch v2.1 §5.4)
# --------------------------------------------------------------------------

def test_the_threshold_is_the_lowest_one_that_meets_the_target():
    """Review is the expensive resource this whole system exists to ration, so a
    higher-than-necessary threshold is a real cost."""
    from calibration.thresholds import choose_threshold

    # 500 clean fields, because a 1% guarantee needs ~400 with zero errors —
    # see the table in calibration/thresholds.py. A perfect run on 200 cannot
    # buy it, and no threshold choice changes that.
    scored = [(0.99, True)] * 300 + [(0.60, True)] * 200 + [(0.55, False)] * 3
    chosen = choose_threshold("identifier", scored)

    assert chosen.enforced, chosen.reason
    assert chosen.threshold <= 0.60, chosen.guarantee()
    assert chosen.achieved_upper_bound <= chosen.target_error_rate


def test_the_bound_is_the_upper_one_not_the_point_estimate():
    """Two errors in forty is a 5% point estimate and an 18% upper bound. The
    promise is about the future, not about those forty."""
    from calibration.thresholds import choose_threshold

    scored = [(0.95, True)] * 38 + [(0.95, False)] * 2
    chosen = choose_threshold("identifier", scored, target_error_rate=0.05)
    assert not chosen.enforced, "a 5% point estimate must not buy a 1% promise"


def test_too_few_accepted_fields_supports_no_promise():
    from calibration.thresholds import choose_threshold

    chosen = choose_threshold("money", [(0.99, True)] * 5)
    assert not chosen.enforced
    assert "accepted fields" in (chosen.reason or "")


def test_free_text_is_never_auto_accepted():
    """Absent from the target table rather than set to 1.0 — a field type with
    no error target is one nobody promised anything about."""
    from calibration.thresholds import choose_threshold

    chosen = choose_threshold("free_text", [(0.99, True)] * 200)
    assert not chosen.enforced
    assert "never auto-accepted" in (chosen.reason or "")


def test_an_absent_confidence_routes_to_review():
    """Absence is never treated as acceptance."""
    from calibration.thresholds import ThresholdSet, choose_threshold

    thresholds = ThresholdSet("r", "bf16", {
        "identifier": choose_threshold("identifier", [(0.99, True)] * 200),
    })
    assert thresholds.needs_review("identifier", None)
    assert thresholds.needs_review("entity", 0.99), "no threshold for the type means review"


def test_the_guarantee_reads_as_a_sentence_somebody_can_be_held_to():
    """v1 used 0.70 for every field, marked 'tunable'. It was never tuned and
    could not be — nothing measured what error rate it bought."""
    from calibration.thresholds import choose_threshold

    chosen = choose_threshold("money", [(0.99, True)] * 500)
    assert chosen.enforced, chosen.reason
    assert "at most" in chosen.guarantee() and "95% confidence" in chosen.guarantee()


def test_a_one_percent_target_needs_about_four_hundred_clean_fields():
    """Not a tuning knob: 0 errors in 200 gives a 1.9% upper bound, so a perfect
    run on 200 cannot buy a 1% promise. At pilot volume the identifier, money and
    date targets are unreachable and those types route everything to review —
    which is the system working, not failing."""
    from calibration.thresholds import choose_threshold

    assert not choose_threshold("identifier", [(0.99, True)] * 200).enforced
    assert choose_threshold("identifier", [(0.99, True)] * 400).enforced


def test_auto_accept_error_rate_counts_only_what_was_accepted():
    """The rate of wrong values that reached a user without a human looking."""
    from calibration.thresholds import ThresholdSet, auto_accept_error_rate, choose_threshold

    scored = [(0.99, True)] * 599 + [(0.99, False)] + [(0.10, False)] * 50
    thresholds = ThresholdSet("r", "bf16", {"identifier": choose_threshold("identifier", scored)})
    rate = auto_accept_error_rate({"identifier": scored}, thresholds)
    assert 0 < rate < 0.01, rate
    assert rate < 50 / len(scored), "the low-confidence errors were correctly not accepted"


# --------------------------------------------------------------------------
# Loss Run totals reconciliation (arch v2.1 §5.5)
# --------------------------------------------------------------------------

def test_reconciliation_is_per_policy_period():
    """A single grand-total check passes whenever two periods' errors cancel —
    which is how a missing claim in 2024 hides behind an invented one in 2025."""
    from calibration.reconciliation import reconcile

    claims = [
        {"policy_period": "2024", "total_incurred": 100.0},
        {"policy_period": "2025", "total_incurred": 300.0},
    ]
    totals = [
        {"row_type": "subtotal", "policy_period": "2024", "total_incurred": 200.0},
        {"row_type": "subtotal", "policy_period": "2025", "total_incurred": 200.0},
    ]
    # The grand total balances exactly: 400 extracted, 400 printed.
    report = reconcile(claims, totals, {"total_incurred": 400.0})

    assert not report.grand_total_mismatches, "the cancelling errors hide from the grand total"
    assert report.flagged, "but not from the per-period check"
    assert [p.period for p in report.by_period if p.mismatches] == ["2024", "2025"]


def test_a_document_with_no_printed_totals_is_unverifiable_not_verified():
    """It has not demonstrated completeness — it has merely not been caught."""
    from calibration.reconciliation import reconcile

    report = reconcile([{"policy_period": "2024", "total_incurred": 100.0}], [])
    assert report.status == "unverifiable"
    assert report.flagged
    assert "cannot be verified" in report.reasons()[0]


def test_a_balanced_loss_run_reconciles():
    from calibration.reconciliation import reconcile

    claims = [
        {"policy_period": "2024", "total_incurred": 100.0, "paid": 60.0},
        {"policy_period": "2024", "total_incurred": 50.0, "paid": 40.0},
    ]
    totals = [{"row_type": "subtotal", "policy_period": "2024",
               "total_incurred": 150.0, "paid": 100.0}]
    report = reconcile(claims, totals, {"total_incurred": 150.0, "paid": 100.0})

    assert report.reconciled and not report.flagged
    assert report.status == "reconciled"


def test_rounding_is_not_a_mismatch():
    """Printed totals round, and a rounding difference is not the omission this
    check exists to catch."""
    from calibration.reconciliation import reconcile

    claims = [{"policy_period": "2024", "total_incurred": 100.004}]
    totals = [{"row_type": "subtotal", "policy_period": "2024", "total_incurred": 100.0}]
    assert reconcile(claims, totals).reconciled


def test_unverifiable_documents_are_excluded_from_the_rate():
    """Letting them drag the metric down would make it a measure of the CORPUS
    rather than of the model."""
    from calibration.reconciliation import reconcile, reconciliation_rate

    good = reconcile(
        [{"policy_period": "2024", "total_incurred": 100.0}],
        [{"row_type": "subtotal", "policy_period": "2024", "total_incurred": 100.0}],
    )
    silent = reconcile([{"policy_period": "2024", "total_incurred": 100.0}], [])

    assert reconciliation_rate([good, silent]) == 1.0
    assert reconciliation_rate([silent]) == 0.0


def test_a_conditional_metric_is_not_applicable_rather_than_unmeasured():
    """An ACORD-only eval set has nothing to say about Loss Run reconciliation,
    and blocking on it would make the gate a statement about the eval set's
    composition rather than about the model."""
    from evaluation.gating import CONDITIONAL_METRICS

    candidate = _metrics()
    del candidate["lossrun_totals_reconciliation_rate"]
    result = promotion_gate(candidate, None)

    assert "lossrun_totals_reconciliation_rate" in CONDITIONAL_METRICS
    assert "lossrun_totals_reconciliation_rate" not in result.failed_gates


def test_a_conditional_metric_still_faces_its_floor_when_present():
    """Exempt ONLY from the was-it-measured requirement. Anything wider would be
    the v1 mistake in reverse: a gate that passes by no longer looking."""
    result = promotion_gate(_metrics(lossrun_totals_reconciliation_rate=0.40), None)
    assert not result.passed
    assert "lossrun_totals_reconciliation_rate" in result.failed_gates


def test_the_reading_regime_metrics_are_not_conditional():
    """They look like subset metrics, but every document is read in both regimes,
    so a run that produced neither has a defective eval set. Exempting them would
    let it quietly stop covering the no-OCR production path while the gate kept
    passing."""
    from evaluation.gating import CONDITIONAL_METRICS

    for metric in ("image_only_accuracy", "ocr_arbitration_accuracy"):
        assert metric not in CONDITIONAL_METRICS


def test_an_eval_set_with_no_scans_is_not_blocked_on_scanned_accuracy():
    """Scans are not always there to hold: a line whose documents all arrive
    digital (the CGL originals) has no scanned document to measure."""
    result = promotion_gate(_without(_metrics(), "scanned_accuracy"), None)
    assert "scanned_accuracy" not in result.failed_gates


def test_scanned_accuracy_the_baseline_measured_is_required():
    """The eval set is frozen, so its documents are the same: a metric the last
    release produced and this one did not is something broken, not a set with
    nothing to measure."""
    result = promotion_gate(_without(_metrics(), "scanned_accuracy"), _metrics())
    assert not result.passed
    assert "scanned_accuracy" in result.failed_gates


def test_scanned_accuracy_still_may_not_regress_when_measured():
    result = promotion_gate(_metrics(scanned_accuracy=0.80), _metrics(scanned_accuracy=0.90))
    assert not result.passed
    assert "scanned_accuracy" in result.failed_gates


# --------------------------------------------------------------------------
# Scope-aware gating (arch v2.1 §4.1, §15.2)
# --------------------------------------------------------------------------


def _without(metrics: dict, *names: str) -> dict:
    return {k: v for k, v in metrics.items() if k not in names}


def test_a_policy_only_run_is_not_blocked_on_loss_run_reconciliation():
    """THE test for this phase. Nothing a policy scope covers prints Loss Run
    totals, so blocking on their absence would make the gate a statement about
    corpus composition — and no policy release could ever pass."""
    from common.scopes import get_scope

    metrics = _without(_passing(), "lossrun_totals_reconciliation_rate")
    result = promotion_gate(metrics, None, scope=get_scope("policy"))

    assert result.passed, result.report()
    verdict = next(v for v in result.verdicts if v.name == "lossrun_totals_reconciliation_rate")
    assert not verdict.applicable
    assert "NOT APPLICABLE" in verdict.describe()


def test_not_applicable_is_reported_apart_from_unmeasured():
    """They are different statements: one is about coverage, the other is a
    missing result. Collapsing them hides which metrics a release actually
    failed to produce."""
    from common.scopes import get_scope

    metrics = _without(
        _passing(), "lossrun_totals_reconciliation_rate", "lob_detection_accuracy",
        "false_null_rate",
    )
    result = promotion_gate(metrics, None, scope=get_scope("policy"))

    assert not result.passed, "a metric the scope CAN produce and did not must block"
    assert "false_null_rate" in result.failed_gates
    assert "lossrun_totals_reconciliation_rate" not in result.failed_gates
    # A canonical policy has no line of business for the model to detect.
    assert "lob_detection_accuracy" not in result.failed_gates

    report = result.report()
    assert "UNMEASURED: false_null_rate" in report
    assert "not applicable: lossrun_totals_reconciliation_rate" in report
    assert set(result.as_dict()["not_applicable"]) >= {
        "lossrun_totals_reconciliation_rate", "lob_detection_accuracy",
    }


def test_a_unified_run_still_blocks_on_the_same_missing_metric():
    """The scope is what changes the verdict, not the metric. A unified run
    covers Loss Runs, so their reconciliation is a result it owes."""
    metrics = _without(_passing(), "lossrun_totals_reconciliation_rate")
    unscoped = promotion_gate(metrics, None)

    # Still exempt as a CONDITIONAL metric (a statement about the eval set), but
    # a metric outside that set is not.
    assert unscoped.passed
    assert not promotion_gate(_without(_passing(), "field_exact_match"), None).passed


def test_a_scope_cannot_waive_a_metric_it_can_produce():
    """The scope's SHAPE decides; its declaration is documentation. This scope is
    built in code, so it never passed the config-load check — and is still not
    excused from a metric it can produce."""
    from dataclasses import replace

    from common.scopes import get_scope
    from evaluation.gating import not_applicable_for

    lying = replace(
        get_scope("unified"),
        not_applicable_metrics=frozenset({"field_exact_match", "lossrun_totals_reconciliation_rate"}),
    )
    assert "field_exact_match" not in not_applicable_for(lying)
    assert not promotion_gate(_without(_passing(), "field_exact_match"), None, scope=lying).passed


def test_a_scope_can_raise_its_own_floors():
    """Floors are per scope because the same number means different things to
    different coverage."""
    from dataclasses import replace

    from common.scopes import get_scope
    from evaluation.gating import floors_for

    strict = replace(get_scope("policy"), floors={"field_exact_match": 0.99})

    assert floors_for(strict)["field_exact_match"] == 0.99
    assert floors_for(get_scope("policy"))["field_exact_match"] == 0.85

    metrics = _without(_passing(field_exact_match=0.97), "lossrun_totals_reconciliation_rate")
    assert promotion_gate(metrics, None, scope=get_scope("policy")).passed
    assert not promotion_gate(metrics, None, scope=strict).passed


def test_the_gate_decision_says_which_scope_it_judged():
    """"v2 passed" means nothing when two scopes share a version."""
    from common.scopes import get_scope

    metrics = _without(_passing(), "lossrun_totals_reconciliation_rate")
    result = promotion_gate(metrics, None, scope=get_scope("policy"))

    assert result.scope == "policy"
    assert result.as_dict()["scope"] == "policy"


def test_an_eval_set_warns_only_about_subsets_its_scope_could_contain():
    """A Loss Run scope has no policies, so `long_policy` is a gap that cannot be
    filled — warning about it teaches everyone to ignore the warning."""
    from common.scopes import get_scope
    from evaluation.run_eval import expected_subsets

    assert "long_policy" in expected_subsets(get_scope("unified"))
    assert "long_policy" in expected_subsets(get_scope("policy"))
    assert "long_policy" not in expected_subsets(get_scope("lossrun"))
    # The modality subsets are required of every corpus (§6), whatever it covers.
    for scope_name in ("unified", "policy", "lossrun"):
        assert {"image_only", "scanned", "noisy_ocr"} <= expected_subsets(get_scope(scope_name))


def test_a_scope_that_forgot_to_declare_a_metric_is_still_not_blocked_by_it():
    """The derivation is what excuses a metric, so a scope that structurally
    cannot produce one is not blocked for ever because nobody wrote it down."""
    from dataclasses import replace

    from common.scopes import get_scope
    from evaluation.gating import not_applicable_for

    forgetful = replace(get_scope("policy"), not_applicable_metrics=frozenset())
    metrics = _without(_passing(), "lossrun_totals_reconciliation_rate")

    assert "lossrun_totals_reconciliation_rate" in not_applicable_for(forgetful)
    assert promotion_gate(metrics, None, scope=forgetful).passed


def test_the_calibrate_stage_reports_how_far_confidence_is_from_accuracy():
    from orchestration.pipeline_dag import _calibration_errors

    honest = [(0.95, True)] * 19 + [(0.95, False)]                  # 95% stated, 95% right
    boastful = [(0.95, True)] * 10 + [(0.95, False)] * 10           # 95% stated, 50% right
    errors = _calibration_errors({"money": honest, "date": boastful})
    assert errors["money"] == 0.0 and errors["date"] == 0.45
    assert 0.0 < errors["all"] < 0.45
