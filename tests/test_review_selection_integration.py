"""Integration of selection's outputs with calibration, the report and the gate
(coordinator items after review #27-#40).

1. A validation generation keeps its spans (``FieldSpan``), so calibration
   fitted from validation learns ``r_written`` and ``near_tie``; saved and
   restored answers keep them too.
2. Each reading carries its carrier and whether training saw it.
3. Calibration fits by reading mode (``calibration.population``): image_only
   under fast, every mode under standard, with its report in the stage result.
4. The gate records the validation gate mirror beside the tier, and the run
   manifest carries it.
5. The windowing oracle places pages as the corpus build does.
"""

from __future__ import annotations

import json
import random

import pytest

from artifact_registry import paths
from tests.test_orchestration import make_context, seed_corpus  # noqa: F401 - helpers
from tests.test_review_selection_sharding import DOCUMENTS, answer, model_for, row


@pytest.fixture
def client():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller():
    from orchestration.runpod_controller import LocalBackend, RunPodController

    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


# --------------------------------------------------------------------------
# 1. Spans kept for calibration
# --------------------------------------------------------------------------

def test_a_generation_keeps_its_field_spans_and_calibration_learns_r_written(client):
    from evaluation.validation_generation import calibration_samples, generate_validation
    from inference_core.span_map import FieldSpan

    generations = generate_validation([row(s, "ocr_plus_image") for s in DOCUMENTS], model_for(client))
    first = generations[0]
    assert first.spans and all(isinstance(span, FieldSpan) for span in first.spans.values())
    assert first.spans["policy.policy_number"].field_path.endswith(".raw")
    assert first.logprobs_by_path["policy.policy_number"] == first.spans["policy.policy_number"].token_logprobs
    features = [f for rows in calibration_samples(generations).values() for f, _ok in rows]
    assert features and all(f.r_written is True for f in features)        # a bare list would say None


def test_saved_answers_keep_the_span_details(client):
    from evaluation.validation_generation import generate_validation, generation_record, restored_generation

    generation, = generate_validation([row("d1", "ocr_plus_image")], model_for(client))
    generation.spans["policy.policy_number"].top2_margin = 0.25
    record = json.loads(json.dumps(generation_record(generation)))
    restored = restored_generation(generation.row, record)
    span = restored.spans["policy.policy_number"]
    assert span.top2_margin == 0.25 and span.field_path == generation.spans["policy.policy_number"].field_path
    assert span.token_logprobs == generation.logprobs_by_path["policy.policy_number"] and span.mapped


# --------------------------------------------------------------------------
# 2. Carrier on the metadata
# --------------------------------------------------------------------------

def test_a_reading_carries_its_carrier_and_whether_training_saw_it():
    from evaluation.validation_generation import ValidationGeneration, merged_windows

    r = row("d1", "ocr_plus_image")
    r.update(carrier="Acme Mutual", carrier_seen_in_train=False)
    generation = ValidationGeneration(row=r, golden=answer("HO-1"), extraction=answer("HO-1"))
    assert generation.metadata["carrier"] == "Acme Mutual"
    assert generation.metadata["carrier_seen_in_train"] is False
    document, = merged_windows([generation])
    assert document.metadata["carrier"] == "Acme Mutual"                  # whole documents too


# --------------------------------------------------------------------------
# 3. Calibration by reading mode
# --------------------------------------------------------------------------

def _by_mode(modes=("ocr_plus_image", "image_only"), n=200):
    from calibration.features import build_features

    rng = random.Random(7)

    def half(offset, mode):
        rows = []
        for i in range(n):
            correct = rng.random() < 0.8
            features = build_features(field_path="policy_number", value=f"WC-{i + offset}",
                                      logprobs=[-0.05] * 3 if correct else [-2.0] * 3, document={},
                                      page_text=f"WC-{i + offset}" if correct else "nothing")
            features.modality_mode = mode
            rows.append((features, correct))
        return rows

    return {mode: {"calibration": half(0, mode), "threshold": half(10_000, mode)} for mode in modes}


def _calibrate(client, controller, samples):
    from orchestration.pipeline_dag import stage_calibrate

    return stage_calibrate(make_context(client, controller, calibration_samples={"bf16": samples}))


def test_under_fast_calibration_fits_on_image_only_and_reports_every_mode(client, controller):
    result = _calibrate(client, controller, _by_mode())
    report = result.data["by_format"]["bf16"]["report"]
    assert report["fitted_on_modes"] == ["image_only"]
    assert set(report["ece_by_mode"]) == {"ocr_plus_image", "image_only"}
    assert "bf16 fitted on image_only" in result.detail


@pytest.mark.format_profile("standard")
def test_under_standard_calibration_fits_on_every_mode(client, controller):
    result = _calibrate(client, controller, _by_mode())
    assert result.data["by_format"]["bf16"]["report"]["fitted_on_modes"] == ["image_only", "ocr_plus_image"]


def test_samples_supplied_as_two_halves_are_pooled_as_before(client, controller):
    halves = _by_mode(("ocr_plus_image",))["ocr_plus_image"]
    result = _calibrate(client, controller, halves)
    body = result.data["by_format"]["bf16"]
    assert body["report"]["fitted_on_modes"] == ["supplied"] and "auto_accept_error_rate" in body


def test_collected_samples_are_by_reading_mode(client):
    from orchestration.pipeline_dag import StageContext, collect_calibration_samples

    rows = [row(s, m, half="calibration" if i % 2 else "threshold")
            for i, s in enumerate(DOCUMENTS) for m in ("ocr_plus_image", "image_only")]
    client.write_text(paths.corpus_scope_eval_split("v1", "val", "policy"),
                      "".join(json.dumps(r) + "\n" for r in rows))
    ctx = StageContext(client=client, controller=None, out_version="v1",  # type: ignore[arg-type]
                       formats=["bf16"], serving_model_loader=lambda c, fmt: model_for(c))
    samples = collect_calibration_samples(ctx)
    assert set(samples["bf16"]) == {"ocr_plus_image", "image_only"}
    assert all(f.modality_mode == "image_only" for rows_ in samples["bf16"]["image_only"].values() for f, _ in rows_)


# --------------------------------------------------------------------------
# 4. The gate mirror beside the tier, and on the manifest
# --------------------------------------------------------------------------

def test_the_gate_records_the_validation_gate_mirror_beside_the_tier(client, controller):
    from orchestration.pipeline_dag import run_stages, stages_for
    from registry_utils.models import EvalMetrics

    assert EvalMetrics(validation_gate_mirror_accuracy=0.9).validation_gate_mirror_accuracy == 0.9
    seed_corpus(client)
    client.write_json(paths.checkpoint_selection("v1", scope="policy"),
                      {"selected": None, "validation_gate_mirror_accuracy": 0.9012})
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()
    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["validation_gate_mirror_accuracy"] == 0.9012 and "tier" in decision
    assert ctx.results["evaluation_gate"].data["validation_gate_mirror_accuracy"] == 0.9012
    manifest = ctx.manifests["policy"]
    assert manifest.eval_metrics.validation_gate_mirror_accuracy == 0.9012


def test_no_selection_record_means_no_number(client, controller):
    from orchestration.pipeline_dag import _validation_gate_mirror

    assert _validation_gate_mirror(make_context(client, controller)) is None


# --------------------------------------------------------------------------
# 5. The windowing oracle places pages as the build does
# --------------------------------------------------------------------------

def test_the_oracle_places_a_pageless_value_on_every_page_that_prints_it(monkeypatch):
    """On a common-model line the build gives a value with no page every page
    whose text prints it (with_placed_pages); the oracle measures that target."""
    from data_pipeline.dataset_builder import policy_windows
    from evaluation import windowing_ceiling

    called = []
    real = policy_windows.with_placed_pages

    def placed(label, page_texts, **kwargs):
        called.append(True)
        return real(label, page_texts, **kwargs)

    monkeypatch.setattr(policy_windows, "with_placed_pages", placed)
    monkeypatch.setattr(policy_windows, "with_inferred_pages",
                        lambda *a, **k: pytest.fail("a common-model line is placed by search"))
    label = {"policy": {"policy_number": {"raw": "HO-778812", "parsed": "HO-778812"}}}
    result = windowing_ceiling.oracle("d", label, "homeowners", 2, ["Policy HO-778812", "Policy HO-778812"])
    assert called and result.gold_values == 1 and result.recovered == 1


def test_a_self_contained_line_still_infers_the_page(monkeypatch):
    from data_pipeline.dataset_builder import policy_windows
    from evaluation import windowing_ceiling

    monkeypatch.setattr(policy_windows, "with_placed_pages",
                        lambda *a, **k: pytest.fail("a self-contained line is not placed by search"))
    label = {"policy": {"policy_number": {"raw": "CY-1", "parsed": "CY-1", "page_ref": [1]}}}
    result = windowing_ceiling.oracle("d", label, "cyber", 1, ["Policy CY-1"])
    assert result.gold_values == 1
