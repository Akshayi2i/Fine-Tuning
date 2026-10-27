"""The code review of the one-adapter change (2026-10-23), one test per finding.

The adapter trains on, is scored on and serves single-line policies of the lines
it trained on; everything else goes to the base model. A release names the
lines it serves, never none. A released version is not packaged again, a dry
run never promotes, and a version number is never shared by two tenants.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration.pipeline_dag import PipelineError
from tests.test_orchestration import controller, make_context  # noqa: F401 - the fixture


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _ctx_with_train_lines(*keys: str) -> SimpleNamespace:
    data_stats = SimpleNamespace(examples_by_line={"train": {key: {} for key in keys}})
    return SimpleNamespace(manifests={"policy": SimpleNamespace(data_stats=data_stats)},
                           out_version="v5", client=None)


# --------------------------------------------------------------------------
# Single-line policies only
# --------------------------------------------------------------------------


def test_a_line_seen_only_inside_a_package_is_not_a_trained_line():
    """The view trains no package, so a property+umbrella package taught nothing
    about a single property policy."""
    from orchestration.pipeline_dag import trained_lines

    ctx = _ctx_with_train_lines("homeowners", "property,umbrella", "unknown")
    assert trained_lines(ctx, {}) == ["homeowners"]


def test_serving_sends_a_package_of_trained_lines_to_the_base_model():
    """Read against _fallback.json, a shape the adapter is not taught."""
    from serving.release_router import ServedRelease, ServingPlan

    plan = ServingPlan(release=ServedRelease("v5", "base", "merged", doc_types=("policy",),
                                             lines=("property", "umbrella")))
    assert plan.route("policy", "property").release is not None
    routed = plan.route("policy", ["property", "umbrella"])
    assert routed.release is None and routed.lob_fallback_used


def test_the_gate_scores_only_the_lines_the_release_serves(client, monkeypatch):
    """Serving sends every other policy - another line, a package, no line - to the
    base model, so scoring them on the adapter measures something nobody is served."""
    import evaluation.golden_eval as golden_eval
    import evaluation.run_eval as run_eval

    lobs = ["homeowners", ["homeowners", "personal_auto"], None, "wc", "workers_comp", "surety"]
    documents = [SimpleNamespace(lob=lob, source_id=f"policy_{i}") for i, lob in enumerate(lobs)]
    monkeypatch.setattr(golden_eval, "_refuse_interrupted_freeze", lambda *a, **k: None)
    monkeypatch.setattr(run_eval, "assert_eval_set_disjoint", lambda *a, **k: None)
    monkeypatch.setattr(golden_eval, "load_golden_set", lambda *a, **k: list(documents))

    class Scored(Exception):
        pass

    def evaluate(scored, *a, **k):
        raise Scored([d.source_id for d in scored])

    monkeypatch.setattr(golden_eval, "evaluate", evaluate)
    with pytest.raises(Scored) as caught:
        golden_eval.evaluate_version(client, object(), version="v1", corpus_version="v1",
                                     lines=["homeowners", "wc"])
    assert caught.value.args[0] == ["policy_0", "policy_3", "policy_4"]
    with pytest.raises(golden_eval.GoldenEvalError, match="lines this release serves"):
        golden_eval.evaluate_version(client, object(), version="v1", corpus_version="v1",
                                     lines=["dwelling_fire"])


def test_a_report_scored_on_other_lines_is_not_reused(client, controller):  # noqa: F811
    from orchestration.pipeline_dag import eval_report_metrics

    client.write_json(paths.corpus_manifest("v1"), {"policy_line_counts": {"homeowners": 3}})
    client.write_json(paths.eval_report("v1", scope="policy"), {
        "gate_metrics": {"field_normalized_match": 0.97}, "lines_scored": None,
        "eval_set": {"tenant_id": paths._tenant(None)}})
    with pytest.raises(PipelineError, match="not reused.*scored on lines all"):
        eval_report_metrics(make_context(client, controller, metrics_provider=None))


# --------------------------------------------------------------------------
# A release names its lines
# --------------------------------------------------------------------------


def test_a_release_with_no_trained_line_is_refused(client, controller):  # noqa: F811
    """An empty list reads as every line at serving."""
    from data_pipeline.corpus_manifest import REVIEW_RULE_VERSION
    from orchestration.pipeline_dag import build_release_bundle

    client.write_json(paths.corpus_manifest("v1"), {"review_rule_version": REVIEW_RULE_VERSION,
                                                    "doc_types": [], "policy_line_counts": {}})
    ctx = make_context(client, controller, dry_run=False)
    with pytest.raises(PipelineError, match="would serve every line"):
        build_release_bundle(ctx)


# --------------------------------------------------------------------------
# Packaging, promotion, numbering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["promoted", "archived"])
def test_a_released_version_is_not_packaged_again(client, controller, status):  # noqa: F811
    """Packaging it again re-ran its gate against the release live now: a pass
    promoted it with no named person, a failure overwrote the decision a
    rollback needs."""
    from orchestration.pipeline_dag import assert_not_released

    client.write_json(paths.release_index(), [{"release_id": "v1", "status": status}])
    with pytest.raises(PipelineError, match=f"already released \\({status}\\).*promote --version v1"):
        assert_not_released(make_context(client, controller))
    client.write_json(paths.release_index(), [{"release_id": "v1", "status": "gated"}])
    assert_not_released(make_context(client, controller))


def test_a_promoted_legacy_release_of_another_model_is_no_baseline(client, controller):  # noqa: F811
    """release-2026.10.1 of extractor-v2 is not v2 of the policy adapter."""
    from orchestration.pipeline_dag import promoted_version

    client.write_json(paths.release_index(), [
        {"release_id": "release-2026.10.1", "status": "promoted", "adapter": "extractor-v2"}])
    assert promoted_version(make_context(client, controller, out_version="v3")) is None
    client.write_json(paths.release_index(), [
        {"release_id": "release-2026.10.1", "status": "promoted", "adapter": "policy-v2"}])
    assert promoted_version(make_context(client, controller, out_version="v3")) == "v2"


def test_an_uncalibrated_baseline_report_is_not_compared(client, controller):  # noqa: F811
    """An ad-hoc run_eval writes the promoted version's report key without
    calibrators: every field flagged, auto_accept_error_rate a trivial 0.0."""
    from orchestration.pipeline_dag import baseline_not_comparable

    client.write_json(paths.release_index(), [{"release_id": "v2", "status": "promoted"}])
    client.write_json(paths.eval_report("v2", scope="policy"), {"gate_metrics": {}, "calibrated": False})
    reason = baseline_not_comparable(make_context(client, controller, out_version="v3"))
    assert reason and "without its calibrators" in reason


# --------------------------------------------------------------------------
# Data in
# --------------------------------------------------------------------------


def test_a_folder_of_other_types_with_no_policy_folder_is_refused(client, controller, tmp_path):  # noqa: F811
    """Read flat, acord/ and lossrun/ would be stored as policies, for good."""
    from orchestration.pipeline_dag import stage_ingestion

    (tmp_path / "acord").mkdir()
    (tmp_path / "lossrun").mkdir()
    (tmp_path / "loose.pdf").write_bytes(b"%PDF-1.7")
    ctx = make_context(client, controller, skip_ingest=False, input_dir=tmp_path)
    with pytest.raises(PipelineError, match="acord/, lossrun/ but no policy/ folder"):
        stage_ingestion(ctx)
    (tmp_path / "policy").mkdir()
    assert stage_ingestion(ctx).status == "completed"


def test_the_audit_blocks_a_folder_teaching_two_output_shapes():
    """Before import: afterwards every policy trains and training refuses the tenant."""
    from data_pipeline.audit import DocumentAudit, output_shape_findings

    one = [DocumentAudit(folder="a", line="homeowners"), DocumentAudit(folder="b", line="gl")]
    assert output_shape_findings(one) == []
    two = [*one, DocumentAudit(folder="c", line="equipment_breakdown"),
           DocumentAudit(folder="d", line="epli+equipment_breakdown")]   # a package does not train
    [finding] = output_shape_findings(two)
    assert finding.severity == "blocker" and "prepare_bundles --lines" in finding.detail


def test_the_preflight_fails_imported_policies_teaching_two_output_shapes():
    from orchestration.preflight import check_output_shape

    results: list = []
    check_output_shape(results, [("policy", "p1", {"lob": "homeowners"}),
                                 ("policy", "p2", {"lob": ["equipment_breakdown", "epli"]})])
    assert results[-1].status == "PASS"
    check_output_shape(results, [("policy", "p1", {"lob": "homeowners"}),
                                 ("policy", "p3", {"lob": "equipment_breakdown"})])
    assert results[-1].status == "FAIL"


def test_lines_match_a_row_by_its_schema_name(tmp_path):
    """--lines wc keeps a row the delivery labels workers_comp."""
    from data_pipeline.ingestion.prepare_bundles import prepare_bundles

    root = tmp_path / "delivery"
    pdf, gold = "Train/pdfs/wc__a__synth_001.pdf", "Train/gold json/wc__a__synth_001.json"
    for rel, body in ((pdf, b"%PDF-1.7"), (gold, b"{}")):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(body)
    row = {"split": "Train", "lob": "workers_comp", "carrier": "Acme", "source": "Acme/wc/a.pdf",
           "sample": 1, "kind": "synthetic", "pdf": pdf, "gold": gold, "pages": 3, "fields": 9,
           "ok": "True", "seconds": 1, "problems": ""}
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    report = prepare_bundles(root, tmp_path / "out", lines=frozenset({"wc"}))
    assert not report.skipped_lines


def test_the_filler_refuses_an_unknown_line_without_a_traceback(tmp_path, capsys):
    from data_pipeline.ingestion.fill_synthetic_labels import main

    assert main(["--input", str(tmp_path), "--lines", "not_a_line"]) == 2
    assert "not_a_line" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Serving config
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["release_pins", "lob_family_threshold"])
def test_a_removed_serving_setting_is_refused_at_cold_start(key):
    """Read by nothing, it would be ignored without a word."""
    from serving.vllm_entrypoint import ColdStartError, assert_no_retired_settings

    with pytest.raises(ColdStartError, match=key):
        assert_no_retired_settings({"routing": {key: {}}})
    assert_no_retired_settings({"routing": {"lob_uncertain_threshold": 0.6}})


def test_the_corpus_view_names_lines_by_their_schema_names():
    from training.corpus_view import _line_of

    assert _line_of({"lob": "workers_comp"}) == "wc"
    assert _line_of({"lob": "classic_auto"}) == "personal_auto"
    assert _line_of({"lob": ["commercial_auto", "workers_comp"]}) == "auto,wc"
    assert _line_of({"lob": None}) == "unknown"
