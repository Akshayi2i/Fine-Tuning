"""Calibration, reviewed 2026-10-26: thresholds by learn-then-test; the feature
vector without the never-filled cross-mode pair and with raw-written and
near-tie; calibrators fitted on what varies, and old ones still read; the
production population under the fast profile; and confidence computed again
once the page text arrives (calibration.post_text)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import common.config
from calibration.features import FeatureSet, FieldFeatures, build_document_features
from calibration.thresholds import FieldTypeThreshold, ThresholdSet, choose_threshold


def _pin(monkeypatch, profile: str) -> None:
    monkeypatch.setattr(common.config, "shared_format_config", lambda: {"profile": profile})


# --------------------------------------------------------------------------
# #13 Learn-then-test
# --------------------------------------------------------------------------

def test_the_scan_runs_from_the_strictest_threshold_and_stops_at_the_first_failure():
    """Scanned upward, the first pass was 0.50 - chosen on the same data its
    bound came from, after the stricter tests it never ran. From the top, the
    20 errors at 0.80 fail the test, and the scan stops: 0.81 is chosen."""
    scored = [(0.99, True)] * 500 + [(0.80, False)] * 20 + [(0.60, True)] * 3000
    chosen = choose_threshold("identifier", scored)
    assert chosen.enforced and chosen.threshold == pytest.approx(0.81)
    assert (chosen.accepted, chosen.errors) == (500, 0)
    assert chosen.achieved_upper_bound <= chosen.target_error_rate


def test_a_threshold_that_could_not_pass_with_no_errors_is_skipped_not_failed():
    """Ten fields at the top cannot buy a 1% promise even all right: decided on
    the confidences alone, the threshold is not tested, and the scan goes on."""
    scored = [(0.99, False)] * 10 + [(0.90, True)] * 2000
    chosen = choose_threshold("identifier", scored)
    assert chosen.enforced and chosen.threshold == pytest.approx(0.50)


def test_a_failure_before_any_pass_routes_the_type_to_review_and_says_where():
    scored = [(0.99, True)] * 400 + [(0.99, False)] * 40
    chosen = choose_threshold("identifier", scored)
    assert not chosen.enforced and "first failure (at 0.99)" in chosen.reason


# --------------------------------------------------------------------------
# #22 (b-d) The feature vector
# --------------------------------------------------------------------------

def test_cross_mode_is_out_and_raw_written_and_near_tie_are_in():
    names = FieldFeatures.feature_names()
    assert not any(name.startswith("cross_mode") for name in names)
    assert {"r_written", "near_tie", "near_tie_present"} <= set(names)
    absent = FieldFeatures("a", "identifier", "x")
    values = dict(zip(names, absent.vector(), strict=True))
    assert (values["r_written"], values["near_tie"], values["near_tie_present"]) == (0.5, 0.5, 0.0)
    tie = FieldFeatures("a", "identifier", "x", r_written=False, top2_margin=0.0)
    values = dict(zip(names, tie.vector(), strict=True))
    assert (values["r_written"], values["near_tie"], values["near_tie_present"]) == (0.0, 1.0, 1.0)


def _span(path, logprobs, margin=None):
    return SimpleNamespace(field_path=path, token_logprobs=logprobs, mapped=True, top2_margin=margin)


def test_spans_say_whether_raw_was_written_and_how_close_the_runner_up_came():
    """collapse_spans keys a field to its raw span where the answer wrote one,
    else to its parsed span: the span's own path tells which."""
    extraction = {"policy": {"policy_number": {"raw": "HO-1", "parsed": "HO-1", "page_ref": [1]},
                             "effective_date": {"raw": None, "parsed": "01/01/2026", "page_ref": [1]}}}
    spans = {"policy.policy_number": _span("policy.policy_number.raw", [-0.1, -0.2], margin=0.3),
             "policy.effective_date": _span("policy.effective_date.parsed", [-0.05])}
    by_path = {f.field_path: f for f in build_document_features(extraction=extraction, spans=spans)}
    number, date = by_path["policy.policy_number"], by_path["policy.effective_date"]
    assert (number.r_written, number.top2_margin, number.min_logprob) == (True, 0.3, -0.2)
    assert (date.r_written, date.top2_margin) == (False, None)
    assert number.page_ref == [1] and number.printed_value == "HO-1"
    # Bare logprob lists, as before: the span says neither.
    plain = build_document_features(extraction=extraction, spans={"policy.policy_number": [-0.1]})
    assert {(f.r_written, f.top2_margin) for f in plain} == {(None, None)}


def _features(i: int, *, margin=None) -> FieldFeatures:
    return FieldFeatures("policy.policy_number", "identifier", f"HO-{i}", min_logprob=-(i % 7) / 3,
                         mean_logprob=-(i % 5) / 4, token_count=3 + i % 4, ocr_agreement=float(i % 2),
                         top2_margin=margin)


def test_a_feature_constant_at_fit_time_is_left_out_and_never_read_at_serving():
    from calibration.feature_calibrator import fit_field_type

    rows = FeatureSet("identifier")
    for i in range(320):
        rows.add(_features(i), i % 2 == 1 or i % 7 == 0)
    calibrator = fit_field_type(rows, min_instances=300)
    assert calibrator.enforced
    assert {"near_tie", "near_tie_present", "r_written", "is_null"} <= set(calibrator.constant_features)
    assert "ocr_agreement" in calibrator.feature_names
    # Filled at serving (a span that reports its runner-up): not read, no shift.
    assert calibrator.predict(_features(5)) == calibrator.predict(_features(5, margin=0.01))


def test_a_calibrator_fitted_with_the_cross_mode_pair_still_loads_and_reads_it_as_it_was():
    from calibration.feature_calibrator import FieldTypeCalibrator, _sigmoid

    old = {"field_type": "identifier", "bias": 0.2, "enforced": True,
           "weights": {"min_logprob": 1.0, "ocr_agreement": 2.0, "cross_mode_agreement": 0.4,
                       "cross_mode_present": 3.0}}
    calibrator = FieldTypeCalibrator.from_dict(old)
    features = FieldFeatures("a", "identifier", "x", min_logprob=-0.5, ocr_agreement=1.0)
    assert calibrator.predict(features) == pytest.approx(_sigmoid(0.2 - 0.5 + 2.0 + 0.4 * 0.5 + 3.0 * 0.0))
    with pytest.raises(Exception, match="unknown features"):
        FieldTypeCalibrator.from_dict({**old, "weights": {"min_logprob": 1.0, "no_such": 1.0}})


# --------------------------------------------------------------------------
# #22 (e) Confidence once the text arrives
# --------------------------------------------------------------------------

class _Calibrators:
    """Confident when the value is printed on the page it cites."""

    def predict(self, features):
        return 0.99 if features.ocr_agreement == 1.0 else 0.40


_THRESHOLDS = ThresholdSet("r", "bf16", {"identifier": FieldTypeThreshold("identifier", 0.9, 0.01)})


def _envelope(value, score=0.4, flagged=True):
    return {"raw": value, "parsed": value, "page_ref": [1],
            "confidence": {"score": score, "source": "vlm"}, "flagged": flagged}


def _served(value="HO-1"):
    features = FieldFeatures("policy.policy_number", "identifier", value, printed_value=value, page_ref=[1],
                             modality_mode="image_only")
    return SimpleNamespace(
        source_id="p1",
        extraction={"policy": {"policy_number": _envelope(value)}},
        fields={"policy.policy_number": {"value": value, "confidence": 0.4}},
        review_flags=["policy.policy_number:low_confidence", "coverages:row_count_mismatch"],
        overall_confidence=0.4,
        confidence_features={"policy.policy_number": features},
    )


def test_the_text_arriving_puts_a_printed_value_through_its_calibrator_again():
    from calibration.post_text import recalibrate_after_text

    result = _served()
    recalibrate_after_text(result, {1: "Policy Number HO-1"}, calibrators=_Calibrators(), thresholds=_THRESHOLDS)
    envelope = result.extraction["policy"]["policy_number"]
    assert envelope["confidence"]["score"] == 0.99 and envelope["flagged"] is False
    assert result.fields["policy.policy_number"]["confidence"] == 0.99
    assert result.review_flags == ["coverages:row_count_mismatch"]     # only the confidence flag goes
    assert result.overall_confidence == 0.99
    assert result.confidence_features["policy.policy_number"].ocr_agreement == 1.0


def test_a_value_no_page_prints_stays_flagged():
    from calibration.post_text import recalibrate_after_text

    result = _served()
    recalibrate_after_text(result, {1: "Policy Number ZZ-9"}, calibrators=_Calibrators(), thresholds=_THRESHOLDS)
    assert result.extraction["policy"]["policy_number"]["flagged"] is True
    assert "policy.policy_number:low_confidence" in result.review_flags


def test_without_features_calibrators_or_text_the_result_is_left_as_it_is(caplog):
    from calibration.post_text import recalibrate_after_text

    bare = _served()
    del bare.confidence_features
    with caplog.at_level(logging.WARNING):
        recalibrate_after_text(bare, {1: "HO-1"}, calibrators=_Calibrators(), thresholds=_THRESHOLDS)
    assert "confidence_features" in caplog.text and bare.overall_confidence == 0.4
    for kwargs, texts in (({"calibrators": None}, {1: "HO-1"}), ({"calibrators": _Calibrators()}, {1: " "})):
        result = _served()
        recalibrate_after_text(result, texts, thresholds=_THRESHOLDS, **kwargs)
        assert result.extraction["policy"]["policy_number"]["flagged"] is True


def test_a_value_the_printed_values_hold_took_out_is_not_scored_again():
    from calibration.post_text import recalibrate_after_text

    result = _served()
    result.extraction["policy"]["policy_number"] = {"raw": None, "parsed": None, "page_ref": [],
                                                    "confidence": {"score": 0.0, "source": "system"},
                                                    "flagged": False}
    recalibrate_after_text(result, {1: "HO-1"}, calibrators=_Calibrators(), thresholds=_THRESHOLDS)
    assert result.extraction["policy"]["policy_number"]["confidence"]["source"] == "system"


def test_the_golden_eval_scores_an_image_only_reading_with_its_confidence_on_the_text(monkeypatch):
    """Recomputed after the borrowed hold, as production does once MinerU's
    text arrives: the auto-accept the gate reads is the served one."""
    import serving.pipeline as pipeline
    from evaluation import golden_eval
    from evaluation.golden_eval import GoldenDocument
    from serving.pipeline import ExtractionResult

    _pin(monkeypatch, "fast")

    def extract(request, model, classifier, calibration, **kwargs):
        result = ExtractionResult(source_id="p1", doc_type="policy", model_version="t", mode=request.modality_mode,
                                  schema_valid=True, overall_confidence=0.4,
                                  extraction={"policy": {"policy_number": _envelope("HO-1")}},
                                  fields={"policy.policy_number": {"value": "HO-1", "confidence": 0.4}},
                                  review_flags=["policy.policy_number:low_confidence"],
                                  printed_check=None if request.modality_mode == "image_only" else {
                                      "checked": 1, "not_printed": []})
        result.confidence_features = _served().confidence_features
        return result

    monkeypatch.setattr(pipeline, "extract", extract)
    monkeypatch.setattr("training.stage_data.localize_keys", lambda client, keys, root: {k: k for k in keys})
    doc = GoldenDocument(source_id="p1", doc_type="policy", lob="cyber", image_keys=["k/page_1.png"],
                         golden={"policy": {"policy_number": _envelope("HO-1")}}, page_texts={1: "Policy HO-1"})
    triples = golden_eval.evaluate([doc], object(), object(), "/tmp", calibrators=_Calibrators(),
                                   thresholds=_THRESHOLDS, run_to_run=False, modes=("image_only",))
    [(_gold, served, meta)] = triples
    assert served["policy"]["policy_number"]["flagged"] is False
    assert meta["field_confidence"] == {"policy.policy_number": 0.99}


# --------------------------------------------------------------------------
# #22 (a) The production population
# --------------------------------------------------------------------------

def _generation(source, mode, half, text="Policy Number HO-1"):
    row = {"source_id": source, "modality_mode": mode, "val_half": half}
    return SimpleNamespace(row=row, ocr_pages=None if mode == "image_only" else {1: text})


def _fake_samples(generations):
    halves = {"calibration": [], "threshold": []}
    for g in generations:
        right = not g.row["source_id"].endswith("9")
        features = FieldFeatures("policy.policy_number", "identifier", "HO-1", printed_value="HO-1", page_ref=[1],
                                 min_logprob=-0.1 if right else -2.0, token_count=3,
                                 ocr_agreement=None if g.row["modality_mode"] == "image_only" else 1.0)
        halves[g.row["val_half"]].append((features, right))
    return halves


@pytest.fixture
def generations(monkeypatch):
    import evaluation.validation_generation as validation_generation

    monkeypatch.setattr(validation_generation, "calibration_samples", _fake_samples)
    return [_generation(f"s{i}", mode, "calibration" if i % 2 else "threshold")
            for i in range(20) for mode in ("ocr_plus_image", "noisy_ocr_image", "image_only")]


def test_the_fast_profile_fits_on_image_only_readings_with_their_documents_text(monkeypatch, generations):
    from calibration.population import fitting_modes, production_samples, samples_by_mode

    _pin(monkeypatch, "fast")
    assert fitting_modes() == ("image_only",)
    by_mode = samples_by_mode(generations)
    picked = production_samples(by_mode)
    rows = picked["calibration"] + picked["threshold"]
    assert len(rows) == 20 and {f.modality_mode for f, _ok in rows} == {"image_only"}
    # Borrowed from the ocr_plus_image reading of the same document.
    assert {f.ocr_agreement for f, _ok in rows} == {1.0}


def test_the_standard_profile_pools_every_mode_as_before(monkeypatch, generations):
    from calibration.population import fitting_modes, production_samples, samples_by_mode

    _pin(monkeypatch, "standard")
    assert fitting_modes() is None
    by_mode = samples_by_mode(generations)
    rows = [row for half in production_samples(by_mode).values() for row in half]
    assert len(rows) == 60
    assert {f.ocr_agreement for f, _ok in rows if f.modality_mode == "image_only"} == {None}


def test_the_fit_reports_the_calibration_error_per_mode_and_field_type(monkeypatch, generations):
    from calibration.population import fit_release_calibration, samples_by_mode

    _pin(monkeypatch, "fast")
    fitted = fit_release_calibration(samples_by_mode(generations), release_id="r", serving_format="bf16",
                                     min_instances=5)
    report = fitted.report
    assert report["profile"] == "fast" and report["fitted_on_modes"] == ["image_only"]
    assert report["rows"] == {"calibration": 10, "threshold": 10}
    assert set(report["ece_by_mode"]) == {"image_only", "noisy_ocr_image", "ocr_plus_image"}
    image_only = report["ece_by_mode"]["image_only"]
    assert image_only["fields"] == 10 and set(image_only["by_field_type"]) == {"identifier"}
