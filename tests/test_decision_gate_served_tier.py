"""The tier measures what users receive, and how sure a pass must be (decisions
#9 and #11 of 2026-10-26; evaluation.gating.tier_evidence).

* served_field_match - field match on the image_only readings of REAL
  documents, as served - is the metric the tier, the improvement rule and
  non-inferiority read. field_normalized_match and the mode accuracies stay
  reported; field_normalized_match keeps its floor.
* New floors of 0.85 on served_field_match and on real_ocr_plus_image_accuracy.
* PRODUCTION needs the gate passed, 150 real documents, served_field_match >=
  0.92 and its one-sided 95% lower bound (document bootstrap, 10,000 resamples,
  seeded) >= 0.90; tier_confidence is ``confirmed`` when the bound >= 0.92.

Synthetic fixtures only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from evaluation.gating import GATING_METRICS, promotion_gate

GOLD = {"policy_number": "HO-1", "insured_name": "Rivera Fabrication LLC",
        "effective_date": "2026-01-01"}
#: One field of three wrong: field match 2/3 on that reading.
WRONG = {**GOLD, "insured_name": "Someone Else"}


def _metrics(**over) -> dict[str, float]:
    """Every gating metric clearing its floor."""
    base = {name: 0.96 for name in GATING_METRICS}
    base.update(schema_validity_rate=1.0, ece_confidence=0.04, confusable_misattribution_rate=0.02,
                false_null_rate=0.03, auto_accept_error_rate=0.01)
    base.update(over)
    return base


def _reading(source: str, mode: str, got: dict, **meta) -> tuple:
    return (GOLD, got, {"source_id": source, "doc_type": "lossrun", "modality_mode": mode,
                        "page_count": 1, **meta})


# --------------------------------------------------------------------------
# The metric: real documents' image_only readings, per reading
# --------------------------------------------------------------------------


def test_served_field_match_is_real_image_only_readings_and_nothing_else():
    from evaluation.run_eval import build_report

    report = build_report("v1", [
        _reading("p1", "ocr_plus_image", GOLD), _reading("p1", "image_only", GOLD),
        _reading("p2", "ocr_plus_image", GOLD), _reading("p2", "image_only", WRONG),
        _reading("p3", "noisy_ocr_image", WRONG),
        # A synthetic twin read image_only: its label came from a generator.
        _reading("s1", "image_only", {"policy_number": "X"}, synthetic=True),
        _reading("s1", "ocr_plus_image", {"policy_number": "X"}, synthetic=True),
    ])
    metrics = report.gate_metrics()

    # The mean over the real image_only readings of the share of fields right.
    assert metrics["served_field_match"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["real_ocr_plus_image_accuracy"] == pytest.approx(1.0)
    # The others stay reported, synthetic readings and all.
    assert metrics["image_only_accuracy"] < metrics["served_field_match"]
    assert "field_normalized_match" in metrics and "ocr_arbitration_accuracy" in metrics
    # Per document, for the paired bootstrap and the tier's bound: real only.
    scores = report.per_document_scores()
    assert scores["served_field_match"] == {"p1": 1.0, "p2": pytest.approx(2 / 3)}
    assert set(scores["real_ocr_plus_image_accuracy"]) == {"p1", "p2"}
    assert report.per_document_weights()["served_field_match"] == {"p1": 1.0, "p2": 1.0}


def test_a_set_of_synthetic_twins_alone_has_no_served_field_match():
    from evaluation.run_eval import build_report

    metrics = build_report("v1", [
        _reading("s1", "image_only", GOLD, synthetic=True),
        _reading("s1", "ocr_plus_image", GOLD, synthetic=True),
    ]).gate_metrics()
    assert "served_field_match" not in metrics and "real_ocr_plus_image_accuracy" not in metrics


# --------------------------------------------------------------------------
# Floors, non-inferiority and the improvement rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["served_field_match", "real_ocr_plus_image_accuracy",
                                    "field_normalized_match"])
def test_each_field_match_floor_is_085(metric):
    from evaluation.gating import PILOT_FLOORS

    assert PILOT_FLOORS[metric] == 0.85
    assert promotion_gate(_metrics(**{metric: 0.85}), None).passed
    blocked = promotion_gate(_metrics(**{metric: 0.84}), None)
    assert blocked.failed_gates == [metric] and blocked.tier is None


def test_non_inferiority_reads_served_field_match_and_only_reports_the_three_mode_match():
    current, candidate = _metrics(field_normalized_match=0.97), _metrics(field_normalized_match=0.90)
    per_document = {"field_normalized_match": ([0.97] * 40, [0.90] * 40),
                    "served_field_match": ([0.96] * 40, [0.96] * 40),
                    "list_field_recall": ([0.90] * 40, [0.96] * 40)}
    result = promotion_gate(candidate, current, per_document=per_document)
    assert result.passed, result.report()
    three_mode = next(v for v in result.verdicts if v.name == "field_normalized_match")
    # Measured, compared and reported - never a block.
    assert three_mode.interval is not None and not three_mode.non_inferior
    assert three_mode.as_dict()["comparison_gated"] is False

    per_document["served_field_match"] = ([0.97] * 40, [0.90] * 40)
    dropped = promotion_gate(_metrics(served_field_match=0.90), _metrics(served_field_match=0.97),
                             per_document=per_document)
    assert dropped.failed_gates == ["served_field_match"]


def test_the_improvement_rule_reads_served_field_match_not_the_three_mode_match():
    from evaluation.gating import PRIMARY_METRICS

    assert PRIMARY_METRICS == ("served_field_match", "list_field_recall")
    same = ([0.96] * 40, [0.96] * 40)
    only_three_mode = promotion_gate(_metrics(), _metrics(), per_document={
        "field_normalized_match": ([0.90] * 40, [0.96] * 40),
        "served_field_match": same, "list_field_recall": same})
    assert only_three_mode.failed_gates == ["improvement"]
    served = promotion_gate(_metrics(), _metrics(), per_document={
        "served_field_match": ([0.90] * 40, [0.96] * 40), "list_field_recall": same})
    assert served.passed and served.improved_metrics == ["served_field_match"]


# --------------------------------------------------------------------------
# The tier and how sure it is
# --------------------------------------------------------------------------


def test_the_tier_thresholds_are_named_constants():
    from evaluation import bootstrap, gating

    assert gating.PRODUCTION_FIELD_MATCH == 0.92
    assert gating.PRODUCTION_LOWER_BOUND == 0.90
    assert gating.CONFIRMED_LOWER_BOUND == 0.92
    assert gating.TIER_CONFIDENCE == 0.95
    assert gating.PRODUCTION_MIN_REAL_DOCUMENTS == 150
    assert bootstrap.RESAMPLES == 10_000 and isinstance(bootstrap.SEED, int)


def _noisy(ones: int, others: int, other: float) -> tuple[list[float], list[float]]:
    scores = [1.0] * ones + [other] * others
    return scores, [1.0] * len(scores)


def test_a_pass_whose_bound_is_0905_is_production_unconfirmed():
    """122 real documents read perfectly and 28 at 0.6: 0.9253, with a one-sided
    95% lower bound near 0.905 - above the 0.90 production bar, below the 0.92
    that would confirm it."""
    served = _noisy(122, 28, 0.6)
    point = sum(served[0]) / len(served[0])
    result = promotion_gate(_metrics(served_field_match=round(point, 4)), None, served_documents=served)
    evidence = result.tier_evidence
    assert 0.90 <= evidence.lower_bound < 0.92, evidence.lower_bound
    assert result.tier == "production" and result.tier_confidence == "unconfirmed"
    assert evidence.documents == 150 and evidence.reasons == []
    recorded = result.as_dict()
    assert recorded["tier"] == "production" and recorded["tier_confidence"] == "unconfirmed"
    assert recorded["tier_evidence"]["lower_bound"] == round(evidence.lower_bound, 6)
    assert "tier production (unconfirmed)" in result.report()


def test_a_bound_reaching_092_is_confirmed():
    served = ([0.95] * 160, [1.0] * 160)
    result = promotion_gate(_metrics(served_field_match=0.95), None, served_documents=served)
    assert (result.tier, result.tier_confidence) == ("production", "confirmed")


def test_a_point_estimate_above_092_with_a_bound_below_090_is_interim():
    served = _noisy(139, 11, 0.0)
    point = sum(served[0]) / len(served[0])
    result = promotion_gate(_metrics(served_field_match=round(point, 4)), None, served_documents=served)
    assert point >= 0.92 and result.tier_evidence.lower_bound < 0.90
    assert result.passed and result.tier == "interim" and result.tier_confidence is None
    assert any("lower bound" in r and "below 0.9" in r for r in result.tier_evidence.reasons)


def test_the_bound_is_reproducible():
    from evaluation.bootstrap import mean_lower_bound

    scores, weights = _noisy(122, 28, 0.6)
    first = mean_lower_bound("served_field_match", scores, weights)
    assert first == mean_lower_bound("served_field_match", scores, weights)
    assert first.lower < first.estimate and first.resamples == 10_000


def test_no_real_image_only_reading_is_interim_with_a_reason_never_production():
    metrics = _metrics()
    del metrics["served_field_match"]
    result = promotion_gate(metrics, None)
    assert result.passed and result.tier == "interim"
    assert any("served_field_match was not measured" in r for r in result.tier_evidence.reasons)
    # Measured but with no per-document scores to bound it: interim too.
    unbounded = promotion_gate(_metrics(served_field_match=0.97), None)
    assert unbounded.tier == "interim"
    assert any("no per-document served_field_match" in r for r in unbounded.tier_evidence.reasons)


def test_a_promoted_release_that_measured_it_makes_it_required():
    metrics = _metrics()
    del metrics["served_field_match"]
    result = promotion_gate(metrics, _metrics())
    assert "served_field_match" in result.failed_gates


def test_a_blocked_release_has_no_tier_but_keeps_its_evidence():
    served = ([0.95] * 160, [1.0] * 160)
    result = promotion_gate(_metrics(served_field_match=0.95, list_field_recall=0.10), None,
                            served_documents=served)
    assert not result.passed and result.tier is None and result.tier_confidence is None
    assert result.tier_evidence.lower_bound == pytest.approx(0.95)


# --------------------------------------------------------------------------
# Where the tier is shown: the eval report, the gate decision, the run summary
# --------------------------------------------------------------------------


def test_the_eval_report_shows_the_bound_and_the_tier_it_would_give():
    from evaluation.run_eval import build_report

    readings = [_reading(f"p{i:03d}", mode, GOLD) for i in range(160)
                for mode in ("ocr_plus_image", "image_only")]
    body = build_report("v1", readings).as_dict()
    evidence = body["tier_evidence"]
    assert evidence["point_estimate"] == 1.0 and evidence["lower_bound"] == 1.0
    assert evidence["documents"] == 160 and evidence["resamples"] == 10_000
    assert evidence["tier_if_gate_passes"] == "production"
    assert evidence["tier_confidence_if_gate_passes"] == "confirmed"
    assert body["gate_metrics"]["served_field_match"] == 1.0


def test_the_gate_decision_and_the_run_summary_show_tier_confidence_and_bound():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation.freeze_eval_set import manifest_key
    from orchestration.pipeline_dag import run_stages, stages_for
    from orchestration.runpod_controller import LocalBackend, RunPodController
    from tests.test_orchestration import PASSING_METRICS, _scored_on, make_context, seed_corpus

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    controller = RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")
    seed_corpus(client)
    client.write_json(manifest_key(), {"frozen_from_corpus": "corpus/v1",
                                       "real_documents_by_doc_type": {"policy": 200}})
    scores = dict(zip((f"policy_{i:04d}" for i in range(150)), _noisy(122, 28, 0.6)[0], strict=True))
    client.write_json(paths.eval_report("v1", scope="policy"), {
        "eval_set": _scored_on(policy=200), "per_document": {"served_field_match": scores},
        "per_document_weights": {"served_field_match": dict.fromkeys(scores, 1.0)}})
    ctx = make_context(client, controller, metrics_provider=lambda _c: {
        **PASSING_METRICS, "served_field_match": 0.9253})
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()

    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["tier"] == "production" and decision["tier_confidence"] == "unconfirmed"
    assert 0.90 <= decision["tier_evidence"]["lower_bound"] < 0.92
    assert decision["tier_evidence"]["documents"] == 150
    gate = next(r for r in report.results if r.name == "evaluation_gate")
    assert gate.data["tier_confidence"] == "unconfirmed"
    assert "tier production (unconfirmed)" in report.render()
    assert "lower bound 0.90" in report.render()

    # The run manifest carries them too (EvalMetrics).
    from registry_utils.query_registry import get as get_manifest

    recorded = get_manifest("policy-v1", client).eval_metrics
    assert recorded.served_field_match == 0.9253
    assert (recorded.tier, recorded.tier_confidence) == ("production", "unconfirmed")
    assert recorded.tier_lower_bound == decision["tier_evidence"]["lower_bound"]


def test_the_manifest_records_the_served_metrics_and_the_tier():
    from evaluation.gating import apply_to_manifest
    from registry_utils.models import EvalMetrics

    fields = EvalMetrics.model_fields
    for name in ("served_field_match", "real_ocr_plus_image_accuracy", "tier", "tier_confidence",
                 "tier_lower_bound"):
        assert name in fields and fields[name].default is None, name

    manifest = SimpleNamespace(promotion=SimpleNamespace(), status="trained", run_id="policy-v1")
    result = promotion_gate(_metrics(served_field_match=0.95, real_ocr_plus_image_accuracy=0.93), None,
                            served_documents=([0.95] * 160, [1.0] * 160))
    apply_to_manifest(result, manifest, metrics=_metrics(served_field_match=0.95,
                                                         real_ocr_plus_image_accuracy=0.93))
    recorded = manifest.eval_metrics
    assert (recorded.served_field_match, recorded.real_ocr_plus_image_accuracy) == (0.95, 0.93)
    assert (recorded.tier, recorded.tier_confidence) == ("production", "confirmed")
    assert recorded.tier_lower_bound == pytest.approx(0.95)
    # The caller's values win: a cascade that blocked the release records no tier.
    apply_to_manifest(result, manifest, metrics={"tier": None, "tier_confidence": None})
    assert manifest.eval_metrics.tier is None and manifest.eval_metrics.tier_lower_bound == pytest.approx(0.95)
