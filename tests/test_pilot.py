"""IMPL-15 — the pilot validation protocol.

The decision logic is what these tests cover, because it is the part that runs
without a GPU and the part that costs money when it is wrong: a baseline band
misread sends the annotation budget at the wrong problem, and a pilot criterion
that passes when unmeasured hides the one thing the pilot exists to measure.
"""

from __future__ import annotations

import json

import pytest

from pilot import pilot_report, pilot_run, smoke_test, zero_shot_baseline
from pilot.pilot_run import PILOT_CRITERIA, alias_generalization_gap, evaluate_pilot
from pilot.smoke_test import ComponentResult, evaluate_smoke
from pilot.zero_shot_baseline import classify_baseline, summarise_baseline

PASSING_PILOT_METRICS = {
    "field_f1": 0.86,
    "list_field_recall": 0.81,
    "schema_validity_rate": 1.0,
    "ocr_plus_image_field_f1": 0.88,
    "image_only_field_f1": 0.80,
    "row_completeness_detection": 0.85,
    "lob_detection_accuracy": 0.90,
    "alias_generalization_gap": 0.06,
    "confusable_misattribution_rate": 0.03,
}

ALL_COMPONENTS = [
    ComponentResult(name, True) for name in smoke_test.PIPELINE_COMPONENTS
]


# --------------------------------------------------------------------------
# Experiment A — the baseline decision table
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("f1", "band", "proceed"),
    [
        (0.91, "strong_prior", True),
        (0.71, "strong_prior", True),
        (0.70, "moderate_prior", True),
        (0.55, "moderate_prior", True),
        (0.40, "moderate_prior", True),
        (0.39, "insufficient_prior", False),
        (0.0, "insufficient_prior", False),
    ],
)
def test_the_baseline_decision_table(f1, band, proceed):
    decision = classify_baseline(f1)
    assert decision.band == band
    assert decision.proceed is proceed


def test_a_weak_baseline_sends_you_to_the_prompt_not_to_annotation():
    """Annotation is the most expensive way to discover a prompt was the problem."""
    decision = classify_baseline(0.22)
    assert "prompt" in decision.recommendation
    assert not decision.proceed


def test_the_baseline_reports_where_it_fails_not_just_how_much():
    """Schema adherence, list recall, LoB and OCR arbitration have different
    remedies, and one aggregate F1 hides which one you have."""
    expected = {
        "insured_name": "Rivera Fabrication LLC",
        "policy_number": "WC-8842317-01",
        "line_of_business": ["workers_comp"],
    }
    got = {
        "insured_name": "Meridian Property Group",   # a confusable, not a typo
        "policy_number": "WC-8842317-O1",            # OCR-shaped near miss
        "line_of_business": ["workers_comp"],
    }
    report = summarise_baseline([("policy", expected, got, {"schema_valid": True, "document": "p1"})])

    assert report.field_f1 == pytest.approx(1 / 3)
    assert report.failure_modes, "no failure-mode distribution — the useful half of the experiment"
    assert sum(report.failure_modes.values()) == pytest.approx(1.0, abs=0.01)
    assert report.metrics["lob_detection_accuracy"] == 1.0


def test_the_baseline_weights_doc_types_by_document_count():
    """An unweighted mean lets a type with two documents outvote one with ten."""
    report = zero_shot_baseline.BaselineReport(
        by_doc_type={"policy": 0.9, "acord": 0.5},
        documents_by_doc_type={"policy": 9, "acord": 1},
    )
    assert report.field_f1 == pytest.approx(0.86)


def test_thin_doc_types_are_named_rather_than_silently_averaged_in():
    report = zero_shot_baseline.BaselineReport(
        by_doc_type={"policy": 0.9, "acord": 0.9},
        documents_by_doc_type={"policy": 10, "acord": 3},
    )
    assert report.under_sampled_types == ["acord"]


def test_a_document_the_pipeline_refused_counts_against_the_baseline():
    """Dropping unparseable output would flatter the base model — that output is
    the measurement."""
    report = zero_shot_baseline.BaselineReport()
    report.documents = 4
    report.failed_documents = [("policy_0003", "schema invalid")]
    payload = report.as_dict()
    assert payload["failed_documents"]


# --------------------------------------------------------------------------
# Experiment B — the smoke test
# --------------------------------------------------------------------------


def test_a_clean_smoke_run_passes():
    report = evaluate_smoke(
        documents_per_type={"acord": 5, "policy": 5, "lossrun": 5},
        loss_curve=[0.31, 0.04, 0.01],
        train_f1=0.98,
        components=ALL_COMPONENTS,
    )
    assert report.passed, report.failures()


def test_loss_must_fall_by_the_deadline_epoch_not_eventually():
    """A model that needs five epochs to memorise five documents has something
    wrong upstream of any architectural question."""
    report = evaluate_smoke(
        documents_per_type={"policy": 5},
        loss_curve=[0.40, 0.30, 0.10, 0.04],
        train_f1=0.99,
        components=ALL_COMPONENTS,
    )
    assert not report.loss_target_met
    assert not report.passed


def test_train_set_f1_short_of_the_target_fails():
    report = evaluate_smoke(
        documents_per_type={"policy": 5},
        loss_curve=[0.02],
        train_f1=0.80,
        components=ALL_COMPONENTS,
    )
    assert not report.f1_target_met
    assert any("training signal" in reason for reason in report.failures())


def test_an_unexercised_component_is_not_a_passing_component():
    """Absent is not passing — it is the component that breaks in Experiment C."""
    partial = [c for c in ALL_COMPONENTS if c.name != "vllm_multi_adapter_hot_swap"]
    report = evaluate_smoke(
        documents_per_type={"policy": 5},
        loss_curve=[0.01, 0.005],
        train_f1=0.99,
        components=partial,
    )
    assert report.untested_components == ["vllm_multi_adapter_hot_swap"]
    assert not report.passed


def test_every_listed_pipeline_component_must_be_reported_on():
    """The spec names seven; a report covering six is not a smoke test."""
    assert len(smoke_test.PIPELINE_COMPONENTS) == 7
    for required in ("adapter_push_to_blob", "vllm_multi_adapter_hot_swap", "run_manifest_generation"):
        assert required in smoke_test.PIPELINE_COMPONENTS


def test_a_thin_corpus_is_flagged_because_three_variants_are_not_enough():
    """One document expands into three modality variants, which does not exercise
    multi-page, OCR-failure or image-only handling."""
    report = evaluate_smoke(
        documents_per_type={"policy": 2},
        loss_curve=[0.01, 0.005],
        train_f1=0.99,
        components=ALL_COMPONENTS,
    )
    assert not report.passed
    assert any("corpus_size:policy" in reason for reason in report.failures())


def test_the_overfit_config_is_deliberately_wrong_for_production():
    config = smoke_test.overfit_config()
    assert config["val_split"] == 0.0
    assert config["early_stopping"] is False
    assert config["num_train_epochs"] == 5
    assert config["learning_rate"] == pytest.approx(2.0e-4)


def test_the_report_says_what_the_smoke_test_does_not_prove():
    report = evaluate_smoke(
        documents_per_type={"policy": 5}, loss_curve=[0.01, 0.005],
        train_f1=0.99, components=ALL_COMPONENTS,
    )
    assert "generalisation" in report.as_dict()["interpretation"]


# --------------------------------------------------------------------------
# Experiment C — pilot criteria
# --------------------------------------------------------------------------


def test_a_passing_pilot_meets_every_criterion():
    report = evaluate_pilot(PASSING_PILOT_METRICS, documents_by_doc_type={"policy": 28})
    assert report.passed, [r.describe() for r in report.failed]


def test_every_spec_criterion_is_evaluated():
    names = {c.name for c in PILOT_CRITERIA}
    assert names == {
        "field_f1", "list_field_recall", "schema_validity_rate", "image_only_gap",
        "row_completeness_detection", "lob_detection_accuracy",
        "alias_generalization_gap", "confusable_misattribution_rate",
    }


def test_schema_validity_below_one_hundred_percent_fails():
    """Anything short of 100% means the model is improvising structure."""
    report = evaluate_pilot({**PASSING_PILOT_METRICS, "schema_validity_rate": 0.98})
    assert not report.passed
    assert [r.criterion.name for r in report.failed] == ["schema_validity_rate"]


def test_an_unmeasured_criterion_does_not_pass_by_default():
    partial = {k: v for k, v in PASSING_PILOT_METRICS.items() if k != "row_completeness_detection"}
    report = evaluate_pilot(partial)
    assert not report.passed
    failed = [r for r in report.failed if r.criterion.name == "row_completeness_detection"]
    assert failed and failed[0].unmeasured


def test_the_image_only_gap_is_relative_not_absolute():
    """An absolute floor would fail a hard document type and pass an easy one."""
    close = evaluate_pilot({**PASSING_PILOT_METRICS,
                            "ocr_plus_image_field_f1": 0.60, "image_only_field_f1": 0.55})
    gap = next(r for r in close.results if r.criterion.name == "image_only_gap")
    assert gap.met

    wide = evaluate_pilot({**PASSING_PILOT_METRICS,
                           "ocr_plus_image_field_f1": 0.90, "image_only_field_f1": 0.60})
    assert not next(r for r in wide.results if r.criterion.name == "image_only_gap").met


def test_an_image_only_gap_points_at_the_vit_gate_with_its_caveat():
    report = evaluate_pilot({**PASSING_PILOT_METRICS, "image_only_field_f1": 0.50})
    diagnosis = next(r for r in report.failed if r.criterion.name == "image_only_gap").criterion.diagnosis
    assert "vit_gate" in diagnosis
    assert "perception-type" in diagnosis


def test_each_failure_names_a_specific_place_to_look():
    """A failed criterion is a diagnosis, not a verdict."""
    report = evaluate_pilot({
        **PASSING_PILOT_METRICS,
        "lob_detection_accuracy": 0.40,
        "confusable_misattribution_rate": 0.30,
    })
    diagnoses = {r.criterion.name: r.criterion.diagnosis for r in report.failed}
    assert "corpus_manifest" in diagnoses["lob_detection_accuracy"]
    assert "confusable_example_count" in diagnoses["confusable_misattribution_rate"]
    assert "exclusion clause" in diagnoses["confusable_misattribution_rate"]


def test_the_report_carries_the_directional_caveat():
    """At 3-4 test documents per type no single metric is trustworthy alone."""
    payload = evaluate_pilot(PASSING_PILOT_METRICS).as_dict()
    assert "directional" in payload["caveat"]
    assert "production gates" in payload["caveat"]


def test_an_undersized_pilot_corpus_is_named():
    report = evaluate_pilot(PASSING_PILOT_METRICS,
                            documents_by_doc_type={"policy": 28, "acord": 11})
    assert report.under_sized_types == ["acord"]
    assert "indicative only" in pilot_run.render(report)


# --------------------------------------------------------------------------
# Alias generalization — the deliberate hold-out
# --------------------------------------------------------------------------


def test_hold_out_selects_every_document_using_the_label():
    provenance = {
        "policy_0001": {"insured_name": "Applicant"},
        "policy_0002": {"insured_name": "Named Insured"},
        "policy_0003": {"insured_name": "applicant"},   # casing must not matter
        "acord_0001": {"insured_name": "Applicant", "producer": "Agency"},
    }
    held = pilot_run.hold_out_surface_label(
        provenance, field_path="insured_name", surface_label="Applicant"
    )
    assert held == ["acord_0001", "policy_0001", "policy_0003"]


def test_alias_gap_is_measured_against_the_dominant_label():
    from evaluation.metrics.confusable import AliasAccuracyReport

    report = AliasAccuracyReport(counts={
        "insured_name": {"Named Insured": (18, 20), "Applicant": (4, 5)},
    })
    gap = alias_generalization_gap(report, field_path="insured_name", held_out_label="Applicant")
    # dominant 0.90, held-out 0.80 -> an 11% shortfall, inside the 15% tolerance
    assert gap == pytest.approx((0.90 - 0.80) / 0.90)
    assert gap < pilot_run.RELATIVE_TOLERANCE


def test_a_memorised_label_string_shows_up_as_a_wide_gap():
    """This is the one claim no plumbing test can prove: that the model learned
    the semantics rather than the string."""
    from evaluation.metrics.confusable import AliasAccuracyReport

    report = AliasAccuracyReport(counts={
        "insured_name": {"Named Insured": (20, 20), "Applicant": (1, 5)},
    })
    gap = alias_generalization_gap(report, field_path="insured_name", held_out_label="Applicant")
    assert gap == pytest.approx(0.80)
    assert not evaluate_pilot(
        {**PASSING_PILOT_METRICS, "alias_generalization_gap": gap}
    ).passed


def test_a_single_variant_yields_no_gap_rather_than_a_flattering_zero():
    from evaluation.metrics.confusable import AliasAccuracyReport

    report = AliasAccuracyReport(counts={"insured_name": {"Named Insured": (20, 20)}})
    assert alias_generalization_gap(
        report, field_path="insured_name", held_out_label="Applicant"
    ) is None


def test_the_worst_gap_wins_rather_than_the_mean():
    """One field that memorised its labels is the finding; averaging hides it."""
    from evaluation.metrics.confusable import AliasAccuracyReport

    report = AliasAccuracyReport(counts={
        "insured_name": {"Named Insured": (20, 20), "Applicant": (2, 5)},
        "producer": {"Producer": (10, 10), "Agency": (10, 10)},
    })
    worst = pilot_run.derive_alias_metric(
        report, {"insured_name": "Applicant", "producer": "Agency"}
    )
    assert worst == pytest.approx(0.60)


# --------------------------------------------------------------------------
# The go/no-go summary
# --------------------------------------------------------------------------


def _write(tmp_path, name, payload):
    (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


def _baseline_payload(proceed=True):
    report = zero_shot_baseline.BaselineReport(
        documents=30,
        by_doc_type={"policy": 0.80 if proceed else 0.20},
        documents_by_doc_type={"policy": 30},
        failure_modes={"perception": 0.6, "schema_reasoning": 0.4},
    )
    return report.as_dict()


def _smoke_payload(passed=True):
    report = evaluate_smoke(
        documents_per_type={"policy": 5},
        loss_curve=[0.02] if passed else [0.9, 0.8],
        train_f1=0.99 if passed else 0.4,
        components=ALL_COMPONENTS,
    )
    return report.as_dict()


def test_a_missing_experiment_blocks_rather_than_passing(tmp_path):
    """Skipping A and B means a failing C cannot be attributed."""
    _write(tmp_path, "pilot_run", evaluate_pilot(PASSING_PILOT_METRICS).as_dict())
    decision = pilot_report.decide(pilot_report.load_reports(tmp_path))

    assert decision.decision == "incomplete"
    assert set(decision.missing) == {"zero_shot_baseline", "smoke_test"}
    assert not decision.go


def test_all_three_passing_is_a_go(tmp_path):
    _write(tmp_path, "zero_shot_baseline", _baseline_payload())
    _write(tmp_path, "smoke_test", _smoke_payload())
    _write(tmp_path, "pilot_run", evaluate_pilot(PASSING_PILOT_METRICS).as_dict())

    decision = pilot_report.decide(pilot_report.load_reports(tmp_path))
    assert decision.go
    assert "sweep" in decision.rationale, "the go decision is what releases the deferred sweep"
    assert "re-baseline" in decision.rationale.lower()


def test_a_failing_smoke_test_blocks_the_go(tmp_path):
    _write(tmp_path, "zero_shot_baseline", _baseline_payload())
    _write(tmp_path, "smoke_test", _smoke_payload(passed=False))
    _write(tmp_path, "pilot_run", evaluate_pilot(PASSING_PILOT_METRICS).as_dict())

    decision = pilot_report.decide(pilot_report.load_reports(tmp_path))
    assert decision.decision == "no_go"
    assert any("Experiment B" in item for item in decision.blocking)


def test_a_weak_baseline_blocks_the_go(tmp_path):
    _write(tmp_path, "zero_shot_baseline", _baseline_payload(proceed=False))
    _write(tmp_path, "smoke_test", _smoke_payload())
    _write(tmp_path, "pilot_run", evaluate_pilot(PASSING_PILOT_METRICS).as_dict())

    decision = pilot_report.decide(pilot_report.load_reports(tmp_path))
    assert decision.decision == "no_go"
    assert any("Experiment A" in item for item in decision.blocking)


def test_the_summary_carries_each_failure_with_its_diagnosis(tmp_path):
    _write(tmp_path, "zero_shot_baseline", _baseline_payload())
    _write(tmp_path, "smoke_test", _smoke_payload())
    _write(tmp_path, "pilot_run",
           evaluate_pilot({**PASSING_PILOT_METRICS, "list_field_recall": 0.40}).as_dict())

    reports = pilot_report.load_reports(tmp_path)
    text = pilot_report.render(reports)

    assert "DECISION: NO_GO" in text
    assert "list_field_recall" in text
    assert "row-length variety" in text


def test_the_summary_renders_all_three_experiments_side_by_side(tmp_path):
    _write(tmp_path, "zero_shot_baseline", _baseline_payload())
    _write(tmp_path, "smoke_test", _smoke_payload())
    _write(tmp_path, "pilot_run", evaluate_pilot(PASSING_PILOT_METRICS).as_dict())

    text = pilot_report.render(pilot_report.load_reports(tmp_path))
    assert "Experiment A" in text and "Experiment B" in text and "Experiment C" in text


def test_write_summary_emits_both_formats(tmp_path):
    _write(tmp_path, "zero_shot_baseline", _baseline_payload())
    _write(tmp_path, "smoke_test", _smoke_payload())
    _write(tmp_path, "pilot_run", evaluate_pilot(PASSING_PILOT_METRICS).as_dict())

    text_path, json_path = pilot_report.write_summary(pilot_report.load_reports(tmp_path), tmp_path)
    assert text_path.exists() and json_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["decision"]["decision"] == "go"


# --------------------------------------------------------------------------
# Registry discipline
# --------------------------------------------------------------------------


def test_a_pilot_run_without_a_manifest_is_refused():
    """"It was only a pilot" is how untracked training runs become normal."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    with pytest.raises(pilot_run.PilotError, match="no run manifest"):
        pilot_run.assert_manifests_written(["foundation-pilot"], client)


def test_the_pilot_package_is_not_imported_by_the_product():
    """IMPL-15 is executed, not imported — nothing else may depend on it."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    packages = ("common", "serving", "inference_core", "training", "evaluation",
                "calibration", "data_pipeline", "orchestration", "testing",
                "postprocessing", "registry_utils", "artifact_registry")
    for package in packages:
        for path in (root / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = (
                    [a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom)
                    else []
                )
                assert not any(n == "pilot" or n.startswith("pilot.") for n in names), (
                    f"{path.relative_to(root)} imports the pilot package — IMPL-15 is a runbook "
                    "that calls the libraries, never a library the product depends on"
                )


def test_an_empty_report_file_does_not_count_as_a_completed_experiment(tmp_path):
    """A truncated or zero-content report used to count toward the decision, and
    its missing `decision.proceed` then read as non-blocking — so the protocol
    could return `go` on nothing at all."""
    for name in ("zero_shot_baseline", "smoke_test", "pilot_run"):
        _write(tmp_path, name, {})

    decision = pilot_report.decide(pilot_report.load_reports(tmp_path))
    assert decision.decision == "incomplete"
    assert not decision.go
    assert set(decision.missing) == {"zero_shot_baseline", "smoke_test", "pilot_run"}


def test_a_policy_only_pilot_is_not_failed_on_line_detection_it_cannot_measure():
    """A policy takes its line from the request, so a policy-only pilot never
    measures lob_detection_accuracy; counted as missed, it could never pass."""
    without = {k: v for k, v in PASSING_PILOT_METRICS.items() if k != "lob_detection_accuracy"}
    report = evaluate_pilot(without, documents_by_doc_type={"policy": 28})
    assert report.passed, [r.describe() for r in report.failed]
    assert report.not_applicable == ["lob_detection_accuracy"]
    assert report.as_dict()["not_applicable"] == ["lob_detection_accuracy"]


def test_a_pilot_with_acord_still_needs_line_detection():
    without = {k: v for k, v in PASSING_PILOT_METRICS.items() if k != "lob_detection_accuracy"}
    report = evaluate_pilot(without, documents_by_doc_type={"policy": 28, "acord": 27})
    assert not report.passed
    assert [r.criterion.name for r in report.failed] == ["lob_detection_accuracy"]
    # With no document types given, nothing is excused.
    assert not evaluate_pilot(without).passed
