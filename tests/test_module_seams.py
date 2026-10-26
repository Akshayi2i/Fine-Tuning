"""Seam tests — what one module emits is what the next module reads.

Every one of the 34 findings from the full-tree review lived at a boundary
between two modules, and the suite stayed green throughout because each module
was tested against a hand-written fixture rather than against what its
neighbour actually produces. Three of those tests passed *because* they encoded
the bug they were meant to guard.

A unit test proves a module is self-consistent. Only a test that runs both
sides proves they agree. Nothing here constructs the data under test by hand.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# The promotion gate reads what the scorer writes
# --------------------------------------------------------------------------


def _scored_documents(n: int = 30):
    golden = {
        "insured_name": "Rivera Fabrication LLC",
        "policy_number": "WC-8842317-01",
        "line_of_business": ["workers_comp"],
        "certificate_holder": "Meridian Property Group LLC",
        "claims": [
            {"claim_number": "CLM-00417", "amount": "12500.00"},
            {"claim_number": "CLM-00418", "amount": "12800.00"},
        ],
    }
    documents = []
    for prefix, mode, scanned, pages in (
        ("a", "ocr_plus_image", False, 2), ("b", "image_only", False, 2),
        ("c", "ocr_plus_image", True, 2), ("d", "noisy_ocr_image", False, 2),
        ("e", "ocr_plus_image", False, 9),
    ):
        for i in range(n):
            documents.append((golden, dict(golden), {
                "source_id": f"{prefix}{i}", "doc_type": "lossrun",
                "modality_mode": mode, "is_scanned": scanned, "page_count": pages,
                "field_confidence": {k: 0.96 for k in golden},
            }))
    return documents


def test_every_gating_metric_is_one_the_scorer_actually_produces():
    """THE showstopper. GATING_METRICS demanded ece_confidence and
    confusable_misattribution_rate; score_subset computed neither, and
    require_all_measured=True blocks on absence — so the gate could not pass
    under any input, for any candidate, ever. score_misattribution had zero
    production callers and ECE had none at all.
    """
    from evaluation.gating import GATING_METRICS
    from evaluation.run_eval import build_report

    report = build_report("v1", _scored_documents(), corpus_version="v1", classifier_scored=True)
    emitted = set(report.gate_metrics())

    # The classifier is scored by IMPL-11, which eval does not run.
    # Conditional metrics only exist when the eval set holds the relevant
    # documents — a Loss Run for reconciliation, a routed Policy for page
    # selection. Their absence is "not applicable", not "not measured", and the
    # gate models that explicitly rather than by exception here.
    from evaluation.gating import CONDITIONAL_METRICS

    required = set(GATING_METRICS) - {"doc_type_classifier_accuracy"} - CONDITIONAL_METRICS
    missing = sorted(required - emitted)
    assert not missing, (
        f"the gate requires {missing}, which the scorer never produces. Every candidate is "
        "blocked as 'not measured', with no override path."
    )


def test_a_flawless_candidate_is_promotable_and_a_regression_is_not():
    """Both directions, against a real EvalReport rather than a hand-built
    metrics dict — building that dict by hand is what hid the bug above.
    """
    from evaluation.gating import promotion_gate
    from evaluation.run_eval import build_report

    metrics = dict(build_report(
        "v1", _scored_documents(), corpus_version="v1", classifier_scored=True,
    ).gate_metrics())
    # Two metrics the synthetic fixture cannot produce honestly: it runs no
    # classifier, and its documents are field dicts rather than schema-valid
    # instances. Stubbed rather than worked around, because this seam is about
    # the gate reading the keys the SCORER emits — not about schema validity.
    metrics["doc_type_classifier_accuracy"] = 0.97
    metrics["schema_validity_rate"] = 1.0
    # The synthetic fixture runs no reconciliation either — it has no Loss Run
    # windows and no printed totals to check against.
    metrics["lossrun_totals_reconciliation_rate"] = 0.95

    assert promotion_gate(metrics, None).passed, "a flawless first version could not be promoted"

    regressed = {**metrics, "field_exact_match": metrics["field_exact_match"] - 0.30}
    assert not promotion_gate(regressed, metrics).passed, "a 30-point drop was promoted"


# --------------------------------------------------------------------------
# Review flags: emitter and consumer
# --------------------------------------------------------------------------


def test_the_review_queue_reads_the_flags_list_completeness_emits():
    """The consumer matched the prefixes list:/rows:/completeness:; the emitter
    has only ever produced "{field}:row_count_mismatch". So the documented
    "row completeness outranks every confidence score" override was dead code
    for every real document.
    """
    from calibration.list_completeness import check_completeness, merge_review_flags
    from data_pipeline.labeling.active_learning import is_row_completeness_flag

    signal = check_completeness("claims", 6, stated_count=8)
    emitted = merge_review_flags({"claims": signal}, [])

    assert emitted, "a 6-of-8 list produced no review flag"
    assert any(is_row_completeness_flag(f) for f in emitted), (
        f"active_learning recognises none of the flags list_completeness emits: {emitted}"
    )


def test_a_flagged_list_is_never_reported_as_completely_confident():
    """A list flagged only by the structural check shipped
    row_completeness_confidence 1.0 — a 3-row gap reported as perfect.
    """
    from calibration.list_completeness import check_completeness

    signal = check_completeness("claims", 6, stated_count=6, detected_rows=9)
    assert signal.flagged
    assert signal.confidence < 1.0, "a flagged list reported perfect completeness"


def test_a_stated_count_is_read_however_the_model_spelled_it():
    """isinstance(stated, int) turned the cross-check off whenever the model
    emitted "8" as a JSON string."""
    from calibration.list_completeness import _as_count

    assert _as_count(8) == 8
    assert _as_count("8") == 8
    assert _as_count(8.0) == 8
    assert _as_count(True) is None
    assert _as_count("many") is None


# --------------------------------------------------------------------------
# Config keys have readers
# --------------------------------------------------------------------------


def test_every_serving_threshold_in_the_yaml_reaches_the_pipeline():
    """Four keys in vllm_serving.yaml were read by nothing: an operator could
    raise review_threshold, redeploy, and change no behaviour at all.
    """
    from common.config import serving_config
    from serving.pipeline import extract
    from serving.vllm_entrypoint import serving_thresholds

    config = serving_config()
    accepted = set(inspect.signature(extract).parameters)
    tuning = serving_thresholds()

    assert tuning, "no serving thresholds were read from the config at all"
    for name in tuning:
        assert name in accepted, f"serving_thresholds emits {name!r}, which extract does not accept"

    threshold = (config.get("confidence") or {}).get("review_threshold")
    if threshold is not None:
        assert tuning.get("review_threshold") == threshold


def test_the_trainer_honours_its_yaml_over_the_helper_defaults():
    """The fixed-one-of-two check. This ordering bug was fixed in train_adapter
    and left standing in train_foundation, exactly as the val-split leak had
    been — twice was a pattern, so it keeps a test now that both have collapsed
    into one trainer.
    """
    from common.config import training_config
    from training import train as T

    swift, _ = T.build_training_config(
        corpus_paths=[f"c/train/epoch_{i}.jsonl" for i in (1, 2, 3, 4)], val_paths=["c/val/val.jsonl"], output_dir="/o")

    evaluation = training_config("unified")["evaluation"]
    for key in ("metric_for_best_model", "load_best_model_at_end"):
        assert swift.args[key] == evaluation[key], (
            f"the trainer emits {key}={swift.args[key]!r} while its YAML says "
            f"{evaluation[key]!r} — the helper's default is winning"
        )
    assert swift.args["greater_is_better"] == evaluation.get("greater_is_better", True)


def test_the_trainer_does_not_train_on_its_validation_split():
    """Kept alongside the test above, for the same reason. Passing both to
    --dataset made ms-swift treat validation as training data and carve its own
    eval split out of the union, so the selected checkpoint was chosen on
    documents the model had memorised."""
    from training.train import build_training_config

    swift, _ = build_training_config(
        corpus_paths=[f"corpus/default/v1/train/epoch_{i}.jsonl" for i in (1, 2, 3, 4)],
        val_paths=["corpus/default/v1/val/val.jsonl"], output_dir="/o")

    assert not set(swift.args["dataset"]) & set(swift.args["val_dataset"])
    assert all("/val/" not in path for path in swift.args["dataset"])
    assert swift.args.get("val_dataset"), "no val_dataset was passed at all"


# --------------------------------------------------------------------------
# Artifact paths never collide
# --------------------------------------------------------------------------


def test_the_gate_verdict_does_not_overwrite_the_scored_eval_report():
    """The gate wrote its thin decision dict over the key EvalReport.as_dict
    writes, destroying by_doc_type and every error record — after which
    vit_gate.evaluate_from_report saw zero image-only and zero scanned
    documents and returned insufficient_data for ever.
    """
    from artifact_registry import paths

    assert paths.gate_decision("v1") != paths.eval_report("v1")


def test_the_double_annotation_sample_is_not_a_subset_of_train():
    """double_annotation_sample reused the corpus splitter's hash and seed
    byte-for-byte, so every sampled document was in train by construction and
    the agreement score never measured noise on the evaluation population.
    """
    from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits
    from data_pipeline.labeling.review_tool.tasks import double_annotation_sample

    source_ids = [f"policy_{i:04d}" for i in range(300)]
    sampled = set(double_annotation_sample(source_ids))
    # One group per document: this seam is about the sampler's hash colliding
    # with the splitter's, not about families.
    assignment = assign_group_splits({
        "policy": [
            GroupRecord(group_id=sid, doc_type="policy", source_ids=[sid])
            for sid in source_ids
        ]
    })
    holdout = {sid for sid, split in assignment.assignment.items() if split != "train"}

    assert sampled, "nothing was sampled"
    assert sampled & holdout, (
        "every double-annotated document is in train, so the agreement score never measures "
        "label noise on the population whose ceiling it is documented to establish"
    )


# --------------------------------------------------------------------------
# Absence is never success
# --------------------------------------------------------------------------


def test_an_unreadable_registry_index_is_not_an_empty_one():
    """Swallowing the read error made a throttled Azure call indistinguishable
    from a first-ever run, so the gate had nothing to regress against and a
    collapsed candidate was promoted.
    """
    from artifact_registry import blob_client as BC
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from registry_utils.query_registry import RegistryQueryError, _index

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    client.write_json("registry/registry_index.json", {"runs": []})

    def unreadable(_key):
        raise BC.BlobError("403 throttled")

    client.read_json = unreadable

    with pytest.raises(RegistryQueryError, match="cannot read the registry index"):
        _index(client)


def test_a_serving_format_that_was_never_scored_cannot_be_promoted():
    """assert_servable checked only blocked_formats; a format nobody measured
    was in neither list, so it passed and shipped with zero measurements.
    """
    from postprocessing.validate_quant import (
        QuantValidationError,
        assert_servable,
        validate_quant,
    )

    reference = {"field_normalized_match": 0.90, "ece_confidence": 0.04,
                 "schema_validity_rate": 1.0}
    report = validate_quant({"bf16": reference}, serving_formats=["bf16", "fp8"])

    with pytest.raises(QuantValidationError, match="never measured"):
        assert_servable(report, ["fp16", "q4_k_m"])


def test_a_diverged_run_cannot_win_a_sweep_phase():
    """Every comparison against NaN is False, so min() never replaced it."""
    from training.sweep import Candidate, CandidateResult, pick_winner

    diverged = CandidateResult(
        Candidate("lr", "foundation", "learning_rate", 2e-4), {"eval_loss": float("nan")})
    real = CandidateResult(
        Candidate("lr", "foundation", "learning_rate", 1e-4), {"eval_loss": 0.20})

    assert pick_winner([diverged, real], "eval_loss", "minimize") is real
    assert pick_winner([diverged], "eval_loss", "minimize") is None


def test_a_baseline_counts_the_documents_it_could_not_extract():
    """field_f1 averaged only the successes, so 8 failures in 10 read as 1.0
    and produced a strong_prior go decision."""
    from pilot.zero_shot_baseline import BaselineReport

    report = BaselineReport()
    report.by_doc_type = {"policy": 1.0}
    report.documents_by_doc_type = {"policy": 2}
    report.documents = 10
    report.failed_documents = [(f"policy_{i}", "unparseable") for i in range(8)]

    assert report.field_f1 == pytest.approx(0.2), (
        f"field F1 is {report.field_f1}, computed over the successes only"
    )
    assert not report.decision.proceed


# --------------------------------------------------------------------------
# Previews and guards
# --------------------------------------------------------------------------


def test_a_dry_run_rollback_does_not_move_the_endpoint():
    """Popping history before the dry_run check meant a preview permanently
    rewrote the recorded live version."""
    from orchestration.runpod_controller import LocalBackend, RunPodController

    controller = RunPodController(
        backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")
    controller._endpoint_versions.extend(["v1", "v2"])

    assert controller.rollback_endpoint(dry_run=True) == "v1"
    assert controller.health_check()["deployed_version"] == "v2"
    assert controller.rollback_endpoint(dry_run=True) == "v1", "the preview consumed history"


def test_the_checkpoint_guard_covers_the_run_ids_this_repo_generates():
    """It matched only undotted single-lineage ids, so foundation-v2.1 and
    every {type}-adapter-v{n} slipped past, trained from base, and recorded a
    lineage that never happened.
    """
    from training.train import TrainingError, assert_checkpoint_path

    for run_id in ("extractor-v3", "extractor-v2.1", "foundation-v3",
                   "policy-adapter-v1", "acord-adapter-v2.3"):
        with pytest.raises(TrainingError, match="checkpoint path"):
            assert_checkpoint_path(run_id)

    assert_checkpoint_path("/runpod-volume/staging/adapters/foundation/v3")


def test_no_gating_metric_is_emitted_under_a_near_miss_name():
    """list_field_recall vs list_recall, and lob_detection_accuracy vs
    lob_accuracy, are how a metric silently stops reaching the gate."""
    from evaluation.gating import GATING_METRICS

    source = (ROOT / "evaluation" / "run_eval.py").read_text(encoding="utf-8")
    emitted = set(re.findall(r'"([a-z_0-9]+)":', source))

    aliases = {name.replace("_field_", "_").replace("_detection_", "_") for name in GATING_METRICS}
    collisions = (aliases & emitted) - set(GATING_METRICS)
    assert not collisions, (
        f"{sorted(collisions)} look like near-misses of a gating metric name; a metric emitted "
        "under the wrong name never reaches the gate"
    )


def test_every_reconciled_column_is_a_loss_run_claim_field():
    """RECONCILED_COLUMNS named "reserve"; the schema says "reserved". A column
    no row carries is skipped without a word, so reserves were never checked."""
    import json

    from calibration.reconciliation import RECONCILED_COLUMNS

    schema = json.loads((ROOT / "schemas" / "lossrun.schema.json").read_text(encoding="utf-8"))
    claim_fields = set(schema["properties"]["claims"]["items"]["properties"])
    unknown = sorted(set(RECONCILED_COLUMNS) - claim_fields)
    assert not unknown, f"reconciliation sums {unknown}, which no Loss Run claim row carries"


def test_a_line_of_business_list_gets_confidence_from_the_spans_the_mapper_emits():
    """The span mapper names list elements line_of_business[0], [1]; the feature
    builder looked up line_of_business, found nothing, and every document with a
    line of business went to review at confidence 0."""
    import json

    from calibration.features import build_document_features
    from inference_core.span_map import map_field_spans

    extraction = {"carrier": "Sentinel", "line_of_business": ["workers_comp", "general_liability"]}
    text = json.dumps(extraction, separators=(",", ":"))
    tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
    spans = map_field_spans(text, tokens, [-0.05] * len(tokens))
    # Exactly the reduction serving/pipeline.py applies before building features.
    logprobs_by_path = {p: s.token_logprobs for p, s in spans.items() if s.mapped}

    features = {
        f.field_path: f
        for f in build_document_features(extraction=extraction, spans=logprobs_by_path)
    }
    lob = features["line_of_business"]
    assert lob.is_usable, lob.reason
    assert lob.token_count > 0

    empty = {
        f.field_path: f
        for f in build_document_features(extraction={"line_of_business": []}, spans={})
    }["line_of_business"]
    assert empty.is_usable and empty.is_null, "an empty list is an answer, not an unmapped field"


def test_claims_invented_for_a_document_with_none_are_not_scored_correct():
    """claims: [] flattened to a field, while the model's invented rows flattened
    to claims[0].* with no claims key, so the lookup found None and [] vs None
    matched."""
    from evaluation.metrics.field_accuracy import score_fields

    expected = {"carrier": "Sentinel", "claims": []}
    invented = {"carrier": "Sentinel", "claims": [{"claim_number": "WC-1", "paid": 100.0}]}
    honest = {"carrier": "Sentinel", "claims": []}

    by_path = {r.field_path: r for r in score_fields(expected, invented).results}
    assert by_path["claims"].correct is False
    assert {r.field_path: r for r in score_fields(expected, honest).results}["claims"].correct
    assert by_path["carrier"].correct
