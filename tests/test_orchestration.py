"""IMPL-13 + IMPL-14 — the command surface, the DAG, and the pod contract.

Four things here guard failures nothing else would notice:

* ``all`` reaching ``package`` after a blocked gate would ship a regression.
* ``all`` invoking extraction would make every build wait on a test run.
* A completed stage re-running would silently duplicate a corpus or a training run.
* A pod launched with no staging volume would write to disk that ceases to exist,
  with every write succeeding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from orchestration import run as cli
from orchestration.pipeline_dag import (
    FINETUNE_STAGES,
    PACKAGE_STAGES,
    STAGES,
    GateBlocked,
    PipelineError,
    StageContext,
    run_stages,
    stages_for,
    stages_from,
)
from orchestration.runpod_controller import (
    GPU_CLASS_BY_STAGE,
    EndpointError,
    LocalBackend,
    PodLaunchError,
    PodSpec,
    RunPodController,
    scrub,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: A candidate that measures every gating metric and beats nothing — the shape a
#: real eval report has, without pretending to be a real score.
PASSING_METRICS = {
    "field_exact_match": 0.90,
    "field_normalized_match": 0.93,
    "list_field_recall": 0.88,
    "field_f1_list_fields": 0.86,
    "schema_validity_rate": 1.0,
    "ocr_arbitration_accuracy": 0.85,
    "image_only_accuracy": 0.78,
    "scanned_accuracy": 0.80,
    "doc_type_classifier_accuracy": 0.97,
    "lob_detection_accuracy": 0.92,
    "ece_confidence": 0.04,
    "confusable_misattribution_rate": 0.02,
    # A value on the page that came back null. Produces no tokens, so §5
    # confidence is blind to it and only this metric sees it (arch v2.1 §15.2).
    "false_null_rate": 0.03,
    # Share of VERIFIABLE Loss Runs whose claims reconciled against printed
    # totals. A missed row produces no tokens, so §5 confidence is blind to it
    # and this is the only signal that sees it (arch v2.1 §5.5).
    "lossrun_totals_reconciliation_rate": 0.92,
}


# --------------------------------------------------------------------------
# Fixtures — a corpus seeded straight into the mock Blob backend
# --------------------------------------------------------------------------


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def client(backend) -> BlobClient:
    return BlobClient(backend=backend, container="main", raw_container="raw")


@pytest.fixture
def raw_client(backend) -> BlobClient:
    """Ingestion-context client — the only one permitted to touch raw-documents/."""
    return BlobClient(backend=backend, container="main", raw_container="raw", context="ingestion")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def ingestion_client(client: BlobClient) -> BlobClient:
    """The same store, seen through the one context allowed near the originals."""
    return BlobClient(
        backend=client._backend, container=client.container,
        raw_container=client.raw_container, context="ingestion",
    )


def seed_corpus(client: BlobClient) -> list[str]:
    """Ingest + OCR + label the fixture set, as stages 1-3 would leave it."""
    raw = ingestion_client(client)
    seeded = []
    for golden_path in sorted(FIXTURES.glob("golden/*.golden.json")):
        source_id = golden_path.name.replace(".golden.json", "")
        doc_type = source_id.rsplit("_", 1)[0]
        metadata = json.loads(
            (FIXTURES / "golden" / f"{source_id}.label_metadata.json").read_text(encoding="utf-8")
        )
        ocr_path = FIXTURES / "ocr" / f"{source_id}_page_1.md"

        raw.write_bytes(paths.raw_pdf(doc_type, source_id), b"%PDF-1.7 fixture")
        raw.write_json(paths.raw_metadata(doc_type, source_id), {
            "source_id": source_id, "doc_type": doc_type, "checksum_sha256": f"sha-{source_id}",
        })
        client.write_text(paths.processed_page(doc_type, source_id, 1, "md"),
                          ocr_path.read_text(encoding="utf-8"))
        client.write_bytes(paths.processed_page(doc_type, source_id, 1, "png"), b"\x89PNG fixture")
        client.write_json(paths.ocr_meta(doc_type, source_id), {
            "source_id": source_id, "doc_type": doc_type, "page_count": 1,
            "mineru_version": "2.0.0", "ocr_device": "cuda", "resolution_cap_px": 1792,
            "source_checksum": f"sha-{source_id}", "table_row_counts": {"1": 0}, "failed_pages": [],
        })
        client.write_json(paths.golden_label(doc_type, source_id),
                          json.loads(golden_path.read_text(encoding="utf-8")))
        client.write_json(paths.label_metadata(doc_type, source_id), metadata)
        seeded.append(source_id)
    return seeded


def make_context(client: BlobClient, controller: RunPodController, **over) -> StageContext:
    defaults = dict(
        client=client,
        raw_client=ingestion_client(client),
        controller=controller,
        out_version="v1",
        release_id="release-2026.9.1",
        tenant_id=None,
        dry_run=True,
        skip_ingest=True,
        min_labels_per_type=1,
        git_commit="abc1234",
        metrics_provider=lambda _ctx: dict(PASSING_METRICS),
    )
    defaults.update(over)
    return StageContext(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The command surface
# --------------------------------------------------------------------------


def test_all_is_finetune_then_package():
    assert stages_for("all") == FINETUNE_STAGES + PACKAGE_STAGES


def test_all_never_includes_extraction():
    """Producing a model and using one are different concerns. Folding extraction
    into the build would make every build wait on a test run (IMPL-13 §1)."""
    names = [s.name for s in stages_for("all")]
    assert "feedback_loop" not in names
    assert not any(s.command == "extract" for s in stages_for("all"))


def test_extract_does_not_map_to_build_stages():
    with pytest.raises(PipelineError, match="extraction routine"):
        stages_for("extract")


def test_every_stage_is_owned_by_exactly_one_command():
    """A stage in two commands would run twice under `all`."""
    finetune = {s.name for s in FINETUNE_STAGES}
    package = {s.name for s in PACKAGE_STAGES}
    assert not finetune & package
    assert len(STAGES) == 13


def test_stage_numbers_match_the_architecture_order():
    assert [s.number for s in STAGES] == list(range(1, 14))


def test_the_stage_order_matches_the_architecture():
    """arch v2.1 §13. Two orderings carry real meaning:

    Selection precedes merge, because the gate scores what ships and what ships
    has to be chosen first (§11.2). And the gate follows merge, quantize AND
    calibrate — under v1 it ran before merge, so it scored the bare adapter
    rather than the merged model in each serving format, and it could not read
    auto_accept_error_rate at all because no threshold had been chosen.
    """
    order = [s.name for s in STAGES]
    assert order.index("training") < order.index("checkpoint_eval")
    assert order.index("checkpoint_eval") < order.index("merge")
    assert order.index("merge") < order.index("quantize")
    assert order.index("quantize") < order.index("calibrate")
    assert order.index("calibrate") < order.index("evaluation_gate")
    assert order.index("evaluation_gate") < order.index("package")


def test_finetune_ends_at_merge_and_package_owns_the_gate():
    """The command boundary moved with the gate (arch v2.1 §13c): `finetune`
    produces artifacts, `package` judges and publishes them."""
    finetune = [s.name for s in STAGES if s.command == "finetune"]
    package = [s.name for s in STAGES if s.command == "package"]

    assert finetune[-1] == "merge"
    assert package == ["quantize", "calibrate", "evaluation_gate", "package"]


def test_from_stage_resumes_mid_pipeline():
    resumed = [s.name for s in stages_from("merge", "finetune")]
    assert resumed == ["merge"]
    assert [s.name for s in stages_from("training", "finetune")][0] == "training"


def test_from_stage_on_all_carries_through_into_package():
    """Resuming at merge must still package afterwards — stopping at the end of
    finetune would leave the artifacts staged and the operator none the wiser."""
    resumed = [s.name for s in stages_from("merge", "all")]
    assert resumed == ["merge", "quantize", "calibrate", "evaluation_gate", "package"]


def test_unknown_stage_names_the_valid_ones():
    with pytest.raises(PipelineError, match="unknown stage"):
        stages_from("trian", "finetune")


# --------------------------------------------------------------------------
# finetune, end to end on fixtures
# --------------------------------------------------------------------------


def test_finetune_runs_ingest_through_merge(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert [r.name for r in report.results] == [s.name for s in FINETUNE_STAGES]
    assert ctx.volume.exists(paths.staging_adapter_dir("foundation", "v1"))
    assert ctx.volume.exists(paths.staging_merged_model_dir("v1", None))


def test_training_reads_exactly_the_files_the_dataset_build_writes(client, controller):
    """The dataset stage wrote {doc_type}/{split}.jsonl while training read
    train/epoch_N.jsonl and val/val.jsonl, which nothing wrote. Every module was
    self-consistent and the suite stayed green; only the real run would fail."""
    from registry_utils.models import DataStats
    from training.train import train

    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.ok, report.render()

    swift, _ = train(
        corpus_version=ctx.corpus, out_version="v9", client=client,
        corpus_manifest=client.read_json(paths.corpus_manifest(ctx.corpus, ctx.tenant_id)),
        data_stats=DataStats(train_examples=1, val_examples=1, test_examples=1),
        dry_run=True, tenant_id=ctx.tenant_id,
    )
    read = list(swift.args["dataset"]) + list(swift.args["val_dataset"])
    missing = [path for path in read if not client.exists(path)]
    assert not missing, f"training reads {missing}, which the dataset build never wrote"


def test_the_dataset_build_assigns_families_before_splitting(client, controller):
    """assign_groups existed and nothing called it, so every document was its own
    group and the group-aware split was a per-document split under another name."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.ok, report.render()

    assert ctx.grouping, "the dataset build recorded no grouping at all"
    for doc_type, grouping in ctx.grouping.items():
        assert grouping["documents"] >= 1, doc_type


def test_finetune_writes_a_blob_manifest_while_weights_stay_staged(client, controller):
    """The volume is working storage with no durability guarantee. Without this
    rule a reclaimed volume means a training run that left no trace (IMPL-13 §3)."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    from registry_utils.query_registry import get, list_runs

    rows = list_runs(client)
    assert rows, "finetune completed without recording a single run manifest"
    foundation = get("extractor-v1", client)
    assert foundation.artifacts.status == "staged"
    assert foundation.artifacts.staging_path


def test_finetune_reports_the_unlabeled_backlog(client, controller):
    """You ingest 500, 200 are labeled, you train on 200 — and the command says
    300 are waiting for a reviewer (IMPL-13 §2)."""
    seed_corpus(client)
    client.write_json(paths.ocr_meta("policy", "policy_0099"),
                      {"source_id": "policy_0099", "page_count": 1})

    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert "policy_0099" in report.unlabeled_backlog["policy"]
    assert "awaiting review" in report.render()


def test_finetune_aborts_below_the_day_zero_floor(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller, min_labels_per_type=25)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert not report.ok
    assert report.failed_at == "labeling"
    assert report.exit_code == 1


def test_finetune_with_no_labels_at_all_stops_before_training(client, controller):
    ctx = make_context(client, controller, doc_types=["policy"])
    report = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert report.failed_at == "labeling"
    assert "human work" in report.results[-1].detail


def test_training_is_one_run_with_no_per_type_fan_out(client, controller):
    """vLLM applies ONE LoRA per request, so a Foundation LoRA and a per-type
    LoRA could never both be active. The v1 topology was not slow or awkward, it
    was unservable — and at 25-30 documents per type a rank-16 adapter memorised
    its own training set anyway (arch v2.1 §4.1)."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    runs = ctx.results["training"].data["runs"]
    assert runs == ["extractor-v1"], f"expected one unified run, got {runs}"
    assert ctx.results["training"].data["run_type"] == "unified"


def test_the_unified_run_produces_one_merged_model(client, controller):
    """One adapter, one merge. v1 produced a merged model per document type
    because each carried its own stacked adapter."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert ctx.volume.exists(paths.staging_merged_model_dir("v1", None))
    for doc_type in ctx.doc_types:
        assert not ctx.volume.exists(paths.staging_merged_model_dir("v1", doc_type)), (
            f"a per-type merged model was produced for {doc_type}; there is one model now"
        )


# --------------------------------------------------------------------------
# The gate — a hard stop with no way round it
# --------------------------------------------------------------------------


def test_a_regression_stops_the_release_not_the_build(client, controller):
    """Under v1 the gate ran before merge, so a regression stopped the build. It
    now runs AFTER merge, quantize and calibrate (arch v2.1 §13), because the
    gate scores the merged model in each serving format — not the bare adapter.

    So merging still happens. What a regression stops is the RELEASE: nothing is
    packaged, nothing is published, and the staged artifacts sit on a volume that
    will be reclaimed."""
    seed_corpus(client)
    regressed = {**PASSING_METRICS, "list_field_recall": 0.60}
    ctx = make_context(
        client, controller,
        baseline_metrics=dict(PASSING_METRICS),
        metrics_provider=lambda _ctx: regressed,
    )
    report = run_stages(ctx, stages_for("all"), command="all")

    assert report.blocked_at == "evaluation_gate"
    assert report.exit_code == 1
    assert "merge" in ctx.results, "the gate scores the merged model, so merge precedes it"
    assert "package" not in ctx.results, "a blocked release is never packaged"


def test_the_block_names_the_metric_and_its_delta(client, controller):
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        baseline_metrics=dict(PASSING_METRICS),
        metrics_provider=lambda _ctx: {**PASSING_METRICS, "lob_detection_accuracy": 0.55},
    )
    report = run_stages(ctx, stages_for("all"), command="all")
    blocked = report.results[-1]

    assert "lob_detection_accuracy" in blocked.detail
    # Both sides of the comparison, so an operator can see the size of the drop
    # without opening the gate decision file.
    assert "0.9200" in blocked.detail and "0.5500" in blocked.detail
    assert blocked.data["failed_gates"] == ["lob_detection_accuracy"]


def test_the_block_records_the_evidence_not_just_the_verdict(client, controller):
    """"Why was this blocked" needs the floor, the margin and the basis — a bare
    delta cannot say whether the drop was outside the noise or inside it."""
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        baseline_metrics=dict(PASSING_METRICS),
        metrics_provider=lambda _ctx: {**PASSING_METRICS, "lob_detection_accuracy": 0.55},
    )
    run_stages(ctx, stages_for("all"), command="all")

    decision = client.read_json(paths.gate_decision("v1"))
    verdict = next(v for v in decision["verdicts"] if v["name"] == "lob_detection_accuracy")
    assert verdict["non_inferior"] is False
    assert verdict["basis"] in ("point_estimate", "paired_bootstrap")
    assert verdict["delta"] > 0


def test_all_never_reaches_package_on_a_failed_gate(client, controller):
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        baseline_metrics=dict(PASSING_METRICS),
        metrics_provider=lambda _ctx: {**PASSING_METRICS, "field_exact_match": 0.10},
    )
    report = run_stages(ctx, stages_for("all"), command="all")

    ran = {r.name for r in report.results}
    # Quantize and calibrate run BEFORE the gate now: the gate reads
    # auto_accept_error_rate, which does not exist until thresholds are chosen.
    assert "quantize" in ran and "calibrate" in ran
    assert "package" not in ran, "nothing is packaged past a failed gate"
    assert report.blocked_at == "evaluation_gate"


def test_an_unmeasured_metric_blocks_rather_than_passing(client, controller):
    """A metric that was not measured has not passed."""
    seed_corpus(client)
    partial = {k: v for k, v in PASSING_METRICS.items() if k != "image_only_accuracy"}
    ctx = make_context(client, controller, baseline_metrics=dict(PASSING_METRICS),
                       metrics_provider=lambda _ctx: partial)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.blocked_at == "evaluation_gate"


def _all_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    options = {opt for action in parser._actions for opt in action.option_strings}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                options |= _all_option_strings(sub)
    return options


def test_no_cli_flag_can_override_the_gate():
    """A flag that exists gets used on the afternoon someone is in a hurry."""
    options = _all_option_strings(cli.build_parser())
    forbidden = {"--force", "--force-deploy", "--skip-gate", "--no-gate", "--ignore-gate"}
    assert not (options & forbidden), f"override flag(s) present: {sorted(options & forbidden)}"


def test_no_stage_function_accepts_an_override_parameter():
    import inspect

    from orchestration import pipeline_dag

    for stage in STAGES:
        params = set(inspect.signature(stage.run).parameters)
        assert params == {"ctx"}, f"{stage.name} takes {params} — a stage takes only the context"

    gate_params = set(inspect.signature(pipeline_dag.stage_evaluation_gate).parameters)
    assert not any("force" in p or "skip" in p for p in gate_params)


def test_the_gate_needs_a_metrics_provider_rather_than_assuming_a_pass(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller, metrics_provider=None)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.failed_at == "evaluation_gate"
    assert "metrics nobody measured" in report.results[-1].detail


def test_the_gate_reads_an_existing_eval_report_when_no_provider_is_given(client, controller):
    """The CLI supplies no provider, so without this default every real
    `finetune` trained a Foundation and three adapters on an A100 and then
    aborted at the gate."""
    from artifact_registry import paths

    seed_corpus(client)
    client.write_json(paths.eval_report("v1"), {"gate_metrics": dict(PASSING_METRICS)})

    ctx = make_context(client, controller, metrics_provider=None)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()


def test_the_baseline_comes_from_the_promoted_version_s_own_report(client, controller):
    """A regression is only detectable against a real baseline, and the CLI
    never supplied one."""
    from artifact_registry import paths
    from orchestration.pipeline_dag import default_baseline_metrics
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import mark_promoted

    seed_corpus(client)
    first = make_context(client, controller, out_version="v1")
    run_stages(first, stages_for("all"), command="all")
    mark_promoted(get("extractor-v1", client), client, promoted_by="test")
    client.write_json(paths.eval_report("v1"), {"gate_metrics": dict(PASSING_METRICS)})

    probe = make_context(client, controller, out_version="v1.1", baseline_metrics=None)
    baseline = default_baseline_metrics(probe)
    assert baseline == PASSING_METRICS


def test_per_document_scores_in_both_reports_gate_on_a_paired_bootstrap(client, controller):
    """With per-document scores in both eval reports, non-inferiority is a paired
    bootstrap and the improvement rule is enforced (arch §15.5): a release no
    better than the last is blocked unless it names the defect it fixes."""
    from artifact_registry import paths
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import mark_promoted

    seed_corpus(client)
    run_stages(make_context(client, controller, out_version="v1"), stages_for("all"), command="all")
    mark_promoted(get("extractor-v1", client), client, promoted_by="test")
    same = {f"doc_{i:02d}": 0.9 for i in range(20)}
    per_document = {"field_normalized_match": same, "list_field_recall": same}
    for version in ("v1", "v1.1", "v1.2"):
        client.write_json(paths.eval_report(version), {
            "gate_metrics": dict(PASSING_METRICS), "per_document": per_document,
            "eval_set": _scored_on(),
        })

    ctx = make_context(client, controller, out_version="v1.1", release_id="release-2026.9.2",
                       metrics_provider=None)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.blocked_at == "evaluation_gate"
    decision = client.read_json(paths.gate_decision("v1.1"))
    assert decision["failed_gates"] == ["improvement"]
    assert decision["paired_documents"]["field_normalized_match"] == 20
    basis = {v["name"]: v["basis"] for v in decision["verdicts"]}
    assert basis["field_normalized_match"] == "paired_bootstrap"
    assert basis["field_exact_match"] == "point_estimate"

    fixed = make_context(client, controller, out_version="v1.2", release_id="release-2026.9.3",
                         metrics_provider=None, fixes_defect="TICKET-12: wrong edition dates")
    assert run_stages(fixed, stages_for("all"), command="all").ok
    assert client.read_json(paths.gate_decision("v1.2"))["fixes_defect"].startswith("TICKET-12")


def test_an_eval_set_accepted_small_caps_the_release_at_interim(client, controller):
    from artifact_registry import paths
    from evaluation.freeze_eval_set import manifest_key

    seed_corpus(client)
    client.write_json(manifest_key(), {
        "small_set_accepted": True, "documents_by_doc_type": {"policy": 9},
        "frozen_from_corpus": "corpus/v1",
    })
    production = {**PASSING_METRICS, "field_normalized_match": 0.97}
    ctx = make_context(client, controller, metrics_provider=lambda _c: dict(production))
    assert run_stages(ctx, stages_for("all"), command="all").ok
    decision = client.read_json(paths.gate_decision("v1"))
    assert decision["tier"] == "interim"
    assert decision["eval_set"]["small_set_accepted"] is True
    assert decision["warnings"]


def _scored_on(tenant: str | None = None, frozen_at: str = "2026-10-01T00:00:00+00:00",
               **documents: int) -> dict:
    """An eval report's record of the frozen set it was scored on
    (``run_eval.eval_set_record``)."""
    return {"tenant_id": paths._tenant(tenant), "frozen_at": frozen_at,
            "frozen_from_corpus": "v1", "documents_by_doc_type": documents}


@pytest.mark.parametrize("case, why", [
    ("refrozen", "different frozen sets"),
    ("other_tenant", "tenant 'other-tenant'"),
    ("no_shared_documents", "share no scored document"),
    ("unrecorded_set", "does not record which frozen eval set"),
])
def test_reports_that_cannot_be_paired_say_so_in_the_decision(client, controller, case, why):
    """Pairing on source_id alone paired documents across two yardsticks: a set
    frozen again with corrected golds reuses its ids, another tenant numbers its
    own, and the report path names neither. Each falls back to a point
    comparison, and the decision says the improvement rule was not enforced and
    why - before, only a log line did, or nothing."""
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import mark_promoted

    seed_corpus(client)
    run_stages(make_context(client, controller, out_version="v1"), stages_for("all"), command="all")
    mark_promoted(get("extractor-v1", client), client, promoted_by="test")
    scores = {f"doc_{i:02d}": 0.9 for i in range(20)}
    promoted = {"gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
                "per_document": {"field_normalized_match": scores, "list_field_recall": scores}}
    if case == "refrozen":
        promoted["eval_set"] = _scored_on(frozen_at="2026-09-01T00:00:00+00:00")
    elif case == "other_tenant":
        promoted["eval_set"] = _scored_on(tenant="other-tenant")
    elif case == "no_shared_documents":
        other = {f"other_{i:02d}": 0.9 for i in range(20)}
        promoted["per_document"] = {"field_normalized_match": other, "list_field_recall": other}
    else:
        del promoted["eval_set"]
    client.write_json(paths.eval_report("v1"), promoted)
    client.write_json(paths.eval_report("v1.1"), {
        "gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
        "per_document": {"field_normalized_match": scores, "list_field_recall": scores},
    })

    ctx = make_context(client, controller, out_version="v1.1", release_id="release-2026.9.2",
                       metrics_provider=None)
    assert run_stages(ctx, stages_for("all"), command="all").ok
    decision = client.read_json(paths.gate_decision("v1.1"))
    assert decision["paired_documents"] == {}
    assert {v["basis"] for v in decision["verdicts"] if v["current"] is not None} == {"point_estimate"}
    assert any("not paired" in w and why in w for w in decision["warnings"]), decision["warnings"]
    assert any("improvement rule was not enforced" in w for w in decision["warnings"])


def test_an_old_report_without_weights_still_pairs(client, controller):
    """A report written before per-document weights were kept pairs each
    document at weight one, as the bootstrap did before."""
    from orchestration.pipeline_dag import paired_document_scores
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import mark_promoted

    seed_corpus(client)
    run_stages(make_context(client, controller, out_version="v1"), stages_for("all"), command="all")
    mark_promoted(get("extractor-v1", client), client, promoted_by="test")
    scores = {"doc_01": 0.5, "doc_02": 1.0}
    client.write_json(paths.eval_report("v1"), {
        "gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
        "per_document": {"list_field_recall": scores},
    })
    client.write_json(paths.eval_report("v1.1"), {
        "gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
        "per_document": {"list_field_recall": scores},
        "per_document_weights": {"list_field_recall": {"doc_01": 1.0, "doc_02": 4.0}},
    })
    warnings: list[str] = []
    paired = paired_document_scores(make_context(client, controller, out_version="v1.1"), warnings)
    assert paired == {"list_field_recall": ([0.5, 1.0], [0.5, 1.0], [1.0, 1.0], [1.0, 4.0])}
    assert warnings == []


def test_the_interim_cap_follows_what_the_scope_was_scored_on(client, controller):
    """small_set_accepted describes the whole set. A GL-only scope scored on 9 GL
    policies of a set frozen with 160 policies reached production; a policy
    scope scored on 200 policies was capped because the set held 9 Loss Runs."""
    from evaluation.freeze_eval_set import MIN_FROZEN_DOCS_PER_TYPE
    from orchestration.pipeline_dag import scope_eval_set_size

    gl = _scoped_ctx(client, controller, "casualty_fleet")
    large_set = {"small_set_accepted": False, "documents_by_doc_type": {"policy": 160},
                 "documents_by_line": {"general_liability": 9, "homeowners": 151}}
    assert scope_eval_set_size(gl, large_set) == (
        {"policy": 9}, "frozen_manifest.documents_by_line")
    # The candidate's own report, when it records them, is what it was scored on.
    client.write_json(paths.eval_report("v1", scope="casualty_fleet"),
                      {"eval_set": _scored_on(policy=9)})
    assert scope_eval_set_size(gl, large_set) == ({"policy": 9}, "eval_report")
    # Another tenant's report at the same path says nothing about this one.
    client.write_json(paths.eval_report("v1", scope="casualty_fleet"),
                      {"eval_set": _scored_on(tenant="other-tenant", policy=400)})
    assert scope_eval_set_size(gl, large_set)[1] == "frozen_manifest.documents_by_line"
    assert scope_eval_set_size(gl, {}) == ({}, "unknown")
    assert MIN_FROZEN_DOCS_PER_TYPE == 150

    from evaluation.freeze_eval_set import manifest_key

    seed_corpus(client)
    client.write_json(manifest_key(), {
        "small_set_accepted": True, "documents_by_doc_type": {"policy": 200, "lossrun": 9},
        "frozen_from_corpus": "corpus/v1",
    })
    client.write_json(paths.eval_report("v1", scope="policy"), {"eval_set": _scored_on(policy=200)})
    policy = _scoped_ctx(client, controller, "policy", metrics_provider=lambda _c: dict(PASSING_METRICS))
    assert run_stages(policy, stages_for("all"), command="all").ok
    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["tier"] == "production"
    assert decision["eval_set"]["scope_documents_by_doc_type"] == {"policy": 200}
    assert decision["eval_set"]["scope_documents_counted_from"] == "eval_report"
    assert decision["eval_set"]["small_for_scope"] is False

    unified = _scoped_ctx(client, controller, "unified")
    assert run_stages(unified, stages_for("all"), command="all").ok
    decision = client.read_json(paths.gate_decision("v1"))
    assert decision["tier"] == "interim"
    assert decision["eval_set"]["scope_documents_by_doc_type"] == {
        "acord": 0, "policy": 200, "lossrun": 9}


def _cli_context(client, controller, argv: list[str], **over):
    args = cli.build_parser().parse_args(argv)
    return cli.build_context(
        args, client=client, controller=controller, raw_client=ingestion_client(client),
        skip_ingest=True, dry_run=True, min_labels_per_type=1, **over,
    )


_REASON = "list recall fell on the relabelled GL schedules only; TICKET-31 re-labels them"


def test_a_written_override_can_be_given_on_the_command_line(client, controller):
    """ctx.gate_override had no flag, so the gate's one exception - a named,
    written override - could not be given at all. The flags build that same
    record, validated as the gate validates it, and it is recorded as a waiver."""
    from evaluation.gating import GateOverrideRecord

    seed_corpus(client)
    regressed = {**PASSING_METRICS, "list_field_recall": 0.60}
    ctx = _cli_context(client, controller, [
        "all", "--input", "./intake", "--out-version", "v1", "--release-id", "release-2026.9.1",
        "--override-approver", "A. Reviewer", "--override-reason", _REASON,
        "--override-gate", "list_field_recall",
    ], baseline_metrics=dict(PASSING_METRICS), metrics_provider=lambda _c: dict(regressed))
    assert isinstance(ctx.gate_override, GateOverrideRecord)
    assert ctx.gate_override.waived_gates == ["list_field_recall"]

    report = cli.run_command("all", ctx)
    assert report.ok, report.render()
    decision = client.read_json(paths.gate_decision("v1"))
    assert decision["waived_gates"] == ["list_field_recall"]
    assert decision["override"]["approver"] == "A. Reviewer"
    bundle = client.read_json(paths.release_bundle("release-2026.9.1"))
    assert bundle["override"]["approver"] == "A. Reviewer"


@pytest.mark.parametrize("flags, match", [
    (["--override-gate", "list_field_recall"], "name the person"),
    (["--override-approver", "A. Reviewer", "--override-gate", "improvement"], "written reason"),
    (["--override-approver", "ci", "--override-reason", _REASON, "--override-gate", "improvement"],
     "name the person"),
    (["--override-approver", "A. Reviewer", "--override-reason", _REASON], "name the gates"),
    (["--override-approver", "A. Reviewer", "--override-reason", "too busy",
      "--override-gate", "improvement"], "written reason"),
])
def test_an_override_flag_alone_is_no_way_round_the_gate(client, controller, flags, match):
    """All three or none: a named person, a reason of 20 characters or more and
    the gates waived. Refused before any work, never a bare bypass."""
    base = ["all", "--out-version", "v1", "--release-id", "release-2026.9.1"]
    with pytest.raises(cli.GateOverrideError, match=match):
        _cli_context(client, controller, base + flags)


def test_one_override_cannot_cover_several_scopes(client, controller):
    with pytest.raises(cli.GateOverrideError, match="one release"):
        _cli_context(client, controller, [
            "all", "--out-version", "v1", "--scope", "policy", "--scope", "lossrun",
            "--release-id", "policy=release-2026.9.1", "--release-id", "lossrun=release-2026.9.2",
            "--override-approver", "A. Reviewer", "--override-reason", _REASON,
            "--override-gate", "improvement",
        ])


def test_fixes_defect_is_checked_and_given_per_scope(client, controller):
    """`--fixes-defect " "` lifted the improvement rule, and one value under
    `all --scope policy --scope lossrun` lifted it for both releases."""
    from common.scopes import get_scope

    policy, lossrun = get_scope("policy"), get_scope("lossrun")
    for blank in (" ", "-", "n/a"):
        with pytest.raises(cli.FixesDefectError, match="names no defect"):
            cli.fixes_defect_for([blank], policy, [policy])
    assert cli.fixes_defect_for(["TICKET-12"], policy, [policy]) == "TICKET-12"
    assert cli.fixes_defect_for([], policy, [policy]) is None
    # A defect's own sentence may hold "=": only a scope's name makes a pair.
    assert cli.fixes_defect_for(["FID-9: rate=0 on endorsements"], policy, [policy]).startswith("FID-9")

    both = [policy, lossrun]
    with pytest.raises(cli.FixesDefectError, match="which release"):
        cli.fixes_defect_for(["TICKET-12"], policy, both)
    with pytest.raises(cli.FixesDefectError, match="does not include"):
        cli.fixes_defect_for(["unified=TICKET-12"], policy, both)

    argv = ["all", "--out-version", "v1", "--scope", "policy", "--scope", "lossrun",
            "--release-id", "policy=release-2026.9.1", "--release-id", "lossrun=release-2026.9.2",
            "--fixes-defect", "policy=TICKET-12"]
    assert _cli_context(client, controller, argv, scope=policy).fixes_defect == "TICKET-12"
    assert _cli_context(client, controller, argv, scope=lossrun).fixes_defect is None


def test_a_model_drafted_label_nobody_reviewed_never_trains(client, controller):
    """Decision 2026-10-19: a gold a model drafted counts only once a person
    reviewed it. The corpus build refuses the run rather than train on it."""
    from orchestration.pipeline_dag import load_labeled_documents

    seeded = seed_corpus(client)
    source_id = seeded[0]
    doc_type = source_id.rsplit("_", 1)[0]
    label = client.read_json(paths.golden_label(doc_type, source_id))
    label["fideon:provenance"] = {"provenance": "Seed gold drafted by Claude Code, not reviewed"}
    client.write_json(paths.golden_label(doc_type, source_id), label)
    metadata = client.read_json(paths.label_metadata(doc_type, source_id))
    unreviewed = {k: v for k, v in metadata.items() if k not in ("reviewer_id", "review_date")}
    client.write_json(paths.label_metadata(doc_type, source_id), unreviewed)

    ctx = make_context(client, controller)
    ctx.doc_types = [doc_type]
    with pytest.raises(PipelineError, match="nobody reviewed"):
        load_labeled_documents(ctx)

    client.write_json(paths.label_metadata(doc_type, source_id),
                      {**unreviewed, "reviewer_id": "a.reviewer", "review_date": "2026-10-20"})
    assert source_id in {d.source_id for d in load_labeled_documents(ctx)}


def test_no_promoted_version_means_no_baseline_not_a_lookup_failure(client, controller):
    from orchestration.pipeline_dag import default_baseline_metrics

    assert default_baseline_metrics(make_context(client, controller)) is None


# --------------------------------------------------------------------------
# Idempotency and resumption
# --------------------------------------------------------------------------


def test_a_completed_stage_re_runs_as_a_no_op(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    first = run_stages(ctx, stages_for("finetune"), command="finetune")
    assert first.ok, first.render()

    again = make_context(client, controller)
    second = run_stages(again, stages_for("finetune"), command="finetune")

    by_name = {r.name: r for r in second.results}
    assert by_name["dataset_build"].status == "skipped"
    assert by_name["training"].status == "skipped"
    assert by_name["merge"].status == "skipped"
    assert "no-op" in by_name["training"].detail


def test_the_gate_is_never_skipped_as_already_complete():
    """An 'already gated' shortcut would let a re-run inherit a pass it did not
    earn. Stage 6 has no completion check, on purpose."""
    gate = next(s for s in STAGES if s.name == "evaluation_gate")
    assert gate.is_complete is None


def test_resuming_at_merge_does_not_retrain(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    resumed = make_context(client, controller)
    report = run_stages(resumed, stages_from("merge", "finetune"), command="finetune")
    assert [r.name for r in report.results] == ["merge"]
    assert "training" not in resumed.results


# --------------------------------------------------------------------------
# package
# --------------------------------------------------------------------------


def _mark_trained(client, version: str) -> None:
    """Flip every run for a version to the status real training would leave.

    `stage_push` publishes only runs that actually finished, so a fixture that
    never launched anything has to say so explicitly rather than relying on a
    default that once let crashed runs be published.
    """
    from registry_utils.query_registry import get, list_runs
    from registry_utils.write_run_manifest import write_manifest

    for row in list_runs(client):
        if row.get("run_id", "").endswith(version):
            manifest = get(row["run_id"], client)
            manifest.status = "trained"
            write_manifest(manifest, client)


def test_package_will_not_publish_a_run_that_never_finished(client, controller):
    """`launch_and_record` marks a crashed run "failed" precisely so the registry
    never claims weights it never wrote. Selecting manifests by run_id alone
    undid that: an adapter that OOM'd at step 40 was flipped to "published" with
    a Blob path serving would then fetch and find empty."""
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import write_manifest

    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    _mark_trained(client, "v1")

    # The run died on the pod.
    crashed = get("extractor-v1", client)
    crashed.status = "failed"
    write_manifest(crashed, client)

    package_ctx = make_context(client, controller)
    run_stages(package_ctx, stages_for("package"), command="package")

    assert get("extractor-v1", client).artifacts.status == "staged",         "a failed run was published, advertising a Blob path serving would find empty"


def test_package_pushes_all_three_artifact_classes_and_publishes(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    # The DAG runs dry by default, so ms-swift is never launched and every
    # manifest is honestly left at status "training". Real training flips it to
    # "trained" via launch_and_record; package refuses anything else, so the
    # fixture has to reflect a run that actually finished.
    _mark_trained(client, "v1")

    package_ctx = make_context(client, controller)
    report = run_stages(package_ctx, stages_for("package"), command="package")
    assert report.ok, report.render()

    pushed = package_ctx.results["package"].data["pushed"]
    assert any(k.startswith("adapter:") for k in pushed)
    assert any(k.startswith("merged:") for k in pushed)
    assert any(k.startswith("quantized:") for k in pushed)

    from registry_utils.query_registry import get

    foundation = get("extractor-v1", client)
    assert foundation.artifacts.status == "published"
    assert foundation.artifacts.staging_path is None
    assert foundation.artifacts.adapter_weights

    # The unified run owns the adapter AND the merged and quantized models —
    # there is one of each under arch v2.1 §4.1. Under v1 it owned only the
    # adapter, because the merged models were produced per document type by the
    # per-type runs stacked on top of it.
    assert foundation.artifacts.merged_model
    # bf16 is the merged model, never a quantized export, so a bf16-only
    # cycle publishes no quantized format.
    assert foundation.artifacts.quantized_formats == []
    assert foundation.artifacts.quantized_model is None


def test_no_per_type_run_is_produced_by_a_default_build(client, controller):
    """A graduated per-type adapter (§4.2) is published by its own run, on its
    own evidence. A default build produces none."""
    from registry_utils.query_registry import RegistryQueryError, get

    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    run_stages(ctx, stages_for("package"), command="package")

    for doc_type in ctx.doc_types:
        with pytest.raises(RegistryQueryError):
            get(f"{doc_type}-adapter-v1", client)


def test_a_default_build_publishes_the_unified_model(client, controller):
    """The unified paths ARE what was built, so the run legitimately owns them."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    run_stages(ctx, stages_for("package"), command="package")

    from registry_utils.query_registry import get

    foundation = get("extractor-v1", client)
    assert foundation.artifacts.merged_model
    # bf16 is the merged model, never a quantized export, so a bf16-only
    # cycle publishes no quantized format.
    assert foundation.artifacts.quantized_formats == []
    assert foundation.artifacts.quantized_model is None


def test_package_clears_staging_after_a_verified_push(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    run_stages(ctx, stages_for("package"), command="package")

    assert ctx.volume.list(paths.staging_root()) == []


def test_keep_staging_retains_the_volume_copy(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller, keep_staging=True)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    run_stages(ctx, stages_for("package"), command="package")

    assert ctx.volume.list(paths.staging_root())


def test_package_fails_loudly_with_remediation_when_nothing_is_staged(client, controller):
    ctx = make_context(client, controller, skip_quantize=True)
    report = run_stages(ctx, stages_for("package"), command="package")

    # The FIRST stage of package, not the last: the remediation is about a
    # reclaimed volume, and discovering it after quantize, calibrate and the gate
    # have run wastes the whole command to say something knowable up front.
    assert report.failed_at == "quantize"
    detail = report.results[-1].detail
    assert "not on the staging volume" in detail
    assert "--from-stage merge" in detail and "--push-adapters" in detail


def test_push_adapters_leaves_only_the_merged_model_staged(client, controller):
    """Recommended when commands 1 and 2 may be separated by more than a day —
    the adapters are tens of MB, the merged model is ~16 GB (IMPL-13 §3)."""
    seed_corpus(client)
    ctx = make_context(client, controller, push_adapters=True)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    # A dry run trained nothing, so there is no adapter to push — and the
    # placeholder note that used to stand in for one is gone. The real upload
    # is covered by test_push_adapters_uploads_the_adapter_itself.
    assert not client.exists(f"{paths.adapter_dir('foundation', 'v1')}/adapter_placeholder.json")


# --------------------------------------------------------------------------
# The pod contract
# --------------------------------------------------------------------------


def test_a_pod_without_the_staging_volume_is_refused(tmp_path):
    """A pod with no volume writes to pod-local disk, terminates, and the work is
    gone — silently, because every write succeeded."""
    # A mount that does not exist, named explicitly: the default /workspace
    # exists on some developer machines (D:\\workspace on Windows), and then the
    # in-process path is allowed and this test asserted the machine, not the code.
    controller = RunPodController(backend=LocalBackend(), volume_id=None, git_commit="abc1234",
                                  volume_mount=str(tmp_path / "no-such-mount"))
    with pytest.raises(PodLaunchError, match="no staging volume attached"):
        controller.launch(controller.spec_for("train_foundation"))


def test_a_pod_leaves_no_process_behind_even_when_the_job_raises(controller):
    with pytest.raises(RuntimeError, match="boom"), controller.pod(controller.spec_for("merge")):
        raise RuntimeError("boom")
    assert controller.active_pods == []


def test_run_job_reports_a_failure_instead_of_leaking_a_pod(controller):
    result = controller.run_job(controller.spec_for("quantize"), lambda _h: 1 / 0)
    assert not result.ok and "division" in (result.error or "")
    assert controller.active_pods == []


def test_every_gpu_stage_runs_on_the_one_h200_pod(controller):
    """The deployment runs every GPU stage on one H200 SXM pod, in process: OCR,
    training, selection, merge, calibration and the gate share the card, and a
    corpus never moves between pods. (The earlier plan put OCR on a cheaper card.)"""
    assert set(GPU_CLASS_BY_STAGE.values()) == {"H200-SXM"}


def test_every_gpu_stage_has_a_declared_class(controller):
    """The keys were v1 lineage names (train_foundation / train_adapter) while
    the DAG asks for "training", so every training pod silently took the default
    card and the A100-80G line described a request nobody made."""
    from orchestration import settings

    for stage in (s for s in STAGES if s.gpu):
        assert stage.name in GPU_CLASS_BY_STAGE, f"{stage.name} has no GPU class"
        assert settings.gpu_class_for(stage.name, "UNSET") != "UNSET", stage.name


def test_a_scope_can_override_the_card_a_stage_runs_on(controller, monkeypatch):
    """Whether a run fits a card is decided by its largest task cap, which is a
    property of the scope: a policy run at 32k against a lossrun run at 20480."""
    from orchestration import settings

    monkeypatch.setattr(
        settings, "pipeline_config",
        lambda: {"gpu_class_by_stage": {"training": "A100-40G"},
                 "gpu_class_by_scope": {"policy": {"training": "A100-80G"}}},
    )
    assert settings.gpu_class_for("training") == "A100-40G"
    assert settings.gpu_class_for("training", scope="policy") == "A100-80G"
    assert settings.gpu_class_for("training", scope="lossrun") == "A100-40G"


def test_finetune_uses_one_pod_for_the_whole_training_fan_out(client, controller):
    """Three launches would mean three cold starts and two Blob round trips of
    the same corpus."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    assert controller.backend._counter == 1  # type: ignore[attr-defined]
    assert controller.active_pods == []


def test_a_pod_will_not_clone_a_moving_branch():
    spec = PodSpec(name="p", stage="merge", gpu_class="L40S", volume_id="vol", git_commit=None)
    with pytest.raises(PodLaunchError, match="no resolvable git commit"):
        spec.clone_command("git@example.com:repo.git")


def test_an_unresolved_commit_is_not_treated_as_pinned():
    """`capture_git_commit` returns 'unknown' off a git tree; a pod cannot clone it."""
    spec = PodSpec(name="p", stage="merge", gpu_class="L40S", volume_id="vol", git_commit="unknown")
    with pytest.raises(PodLaunchError):
        spec.clone_command("git@example.com:repo.git")


def test_logs_are_scrubbed_before_they_leave_the_pod():
    scrubbed = scrub(
        "loss 0.42 for Policy No. WC-8842317-01 reviewer adjuster@carrier.com ssn 123-45-6789"
    )
    assert "WC-8842317-01" not in scrubbed
    assert "adjuster@carrier.com" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "loss 0.42" in scrubbed


def test_a_dry_run_does_not_claim_the_endpoint_moved(controller):
    """History records what was DEPLOYED. Recording a preview made health_check
    report a version that was never live, and the next rollback then "restored"
    the version already running."""
    controller.deploy_endpoint("v2", dry_run=True)
    assert controller.health_check()["deployed_version"] is None
    assert controller.health_check()["history"] == []


def test_rollback_needs_somewhere_to_roll_back_to(controller):
    with pytest.raises(EndpointError, match="nothing to roll back to"):
        controller.rollback_endpoint(dry_run=True)

    # Two real deployments. `deploy_endpoint` refuses to run for real until the
    # RunPod API is wired, so the history is seeded the way that call would.
    controller._endpoint_versions.extend(["v1", "v2"])
    assert controller.health_check()["deployed_version"] == "v2"

    # A preview names the target without moving anything. Asserting the live
    # version had CHANGED after a dry run is what let the bug stand: rollback
    # popped the history before checking dry_run, so health_check reported v1
    # while v2 was still serving and the next real rollback refused with
    # "nothing to roll back to".
    assert controller.rollback_endpoint(dry_run=True) == "v1"
    assert controller.health_check()["deployed_version"] == "v2",         "a dry run moved the endpoint"

    # And it stays repeatable, because nothing was consumed.
    assert controller.rollback_endpoint(dry_run=True) == "v1"


# --------------------------------------------------------------------------
# The CLI shell
# --------------------------------------------------------------------------


def test_finetune_requires_an_out_version():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["finetune", "--input", "./intake"])


def test_all_accepts_the_union_of_both_flag_sets():
    args = cli.build_parser().parse_args([
        "all", "--input", "./intake", "--out-version", "v2",
        "--formats", "bf16", "fp8", "--keep-staging",
    ])
    assert args.out_version == "v2" and args.formats == ["bf16", "fp8"] and args.keep_staging


def test_extract_takes_base_as_a_first_class_model():
    """`base` is the pilot's zero-shot baseline and the day-zero pre-annotation
    path, not a curiosity (IMPL-13 §5)."""
    args = cli.build_parser().parse_args(["extract", "--model", "base", "--source-ids", "policy_0001"])
    assert args.model == "base" and args.command == "extract"


def test_the_cli_holds_no_stage_logic():
    """Stage order, idempotency and the gate live in pipeline_dag, so the CLI
    cannot develop its own opinion about them."""
    import inspect

    source = inspect.getsource(cli)
    assert "promotion_gate" not in source
    assert "train_foundation" not in source


def test_build_context_carries_the_flags_through(client, controller):
    args = cli.build_parser().parse_args([
        "finetune", "--input", "./intake", "--out-version", "v3",
        "--scope", "policy", "--push-adapters", "--dry-run", "--min-labels-per-type", "5",
    ])
    ctx = cli.build_context(args, client=client, controller=controller,
                            raw_client=ingestion_client(client))
    assert ctx.out_version == "v3" and ctx.scope.name == "policy" and ctx.push_adapters
    assert ctx.dry_run and ctx.min_labels_per_type == 5
    assert ctx.corpus == "v3"  # corpus defaults to the output version


def test_all_without_a_release_id_stops_before_training(client, controller):
    """release_id defaulted to "" and nothing set it, so calibrate fitted every
    calibrator and then failed to save them. Under `all` that was after a whole
    training run."""
    seed_corpus(client)
    ctx = make_context(client, controller, release_id="")
    report = run_stages(ctx, stages_for("all"), command="all")

    assert not report.ok
    assert report.failed_at == "ingestion"
    assert [r.name for r in report.results] == ["ingestion"]
    assert "--release-id release-" in report.results[0].detail
    assert not ctx.volume.exists(paths.staging_adapter_dir("foundation", "v1"))


def test_the_suggested_release_id_is_the_next_free_one_this_month():
    used = ["release-2026.9.1", "release-2026.9.3", "release-2026.8.7", "release-2026.10.9"]
    assert paths.next_release_id(used, 2026, 9) == "release-2026.9.4"
    assert paths.next_release_id([], 2026, 9) == "release-2026.9.1"
    assert paths.is_valid_release_id(paths.next_release_id(used, 2026, 9))


def test_run_command_dispatches_all_through_both_commands(client, controller):
    seed_corpus(client)
    args = cli.build_parser().parse_args([
        "all", "--input", "./intake", "--out-version", "v1", "--release-id", "release-2026.9.1",
    ])
    ctx = cli.build_context(
        args, client=client, controller=controller, raw_client=ingestion_client(client),
        skip_ingest=True, dry_run=True, min_labels_per_type=1,
        metrics_provider=lambda _c: dict(PASSING_METRICS),
    )
    report = cli.run_command("all", ctx)

    assert report.ok, report.render()
    assert [r.name for r in report.results] == [s.name for s in stages_for("all")]
    assert report.as_dict()["command"] == "all"


def test_gate_blocked_is_distinguishable_from_a_failure():
    """Both exit non-zero, but only one prints per-metric deltas."""
    assert issubclass(GateBlocked, RuntimeError)
    assert not issubclass(GateBlocked, PipelineError)


# --------------------------------------------------------------------------
# The Foundation-upgrade cascade (arch §12)
# --------------------------------------------------------------------------


def _promote_foundation(client: BlobClient, controller: RunPodController, version: str) -> None:
    """Run a cycle and promote it, so the next one has something to bump from.

    One run under arch v2.1 §4.1: the unified extractor. v1 promoted a Foundation
    plus one adapter per document type, a topology vLLM cannot serve.
    """
    from evaluation.gating import apply_to_manifest, promotion_gate
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import mark_promoted

    ctx = make_context(client, controller, out_version=version)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()

    # Gate first, then promote — the order a real cycle uses. `mark_promoted`
    # requires the affirmative gate result, so a manifest that was never
    # evaluated cannot be promoted at all.
    passed = promotion_gate(dict(PASSING_METRICS), None)
    manifest = apply_to_manifest(passed, get(f"extractor-{version}", client))
    mark_promoted(manifest, client, promoted_by="test")


def _graduate_adapter(client: BlobClient, doc_type: str, foundation_version: str) -> str:
    """Seed a per-type adapter from the §4.2 graduation path.

    The default topology produces none — that is the point of §4.1 — so the
    cascade rule has nothing to act on until a type graduates. These tests are
    about what happens AFTER one does: a graduated adapter trains on the merged
    foundation weights, so a foundation bump invalidates it exactly as before.
    """
    from datetime import UTC, datetime

    from registry_utils.models import (
        Artifacts,
        DataStats,
        Dependencies,
        Promotion,
        RunManifest,
        TrainingConfig,
    )
    from registry_utils.write_run_manifest import write_manifest

    run_id = f"{doc_type}-adapter-{foundation_version}"
    manifest = RunManifest(
        run_id=run_id,
        run_type="per_type_adapter",
        doc_type=doc_type,
        status="promoted",
        dependencies=Dependencies(
            base_model="Qwen/Qwen3-VL-8B-Instruct@abc1234",
            corpus_version=f"corpus/{foundation_version}",
            code_git_commit="abc1234",
            foundation_version=f"extractor-{foundation_version}",
        ),
        training_config=TrainingConfig(
            technique="LoRA", base_quantization="bf16_frozen_base", optimizer="adamw_torch",
            lora_rank=16, lora_alpha=32, learning_rate=7e-5, epochs=4,
            gradient_accumulation_steps=8, effective_batch_size=8,
            target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=24576, seed=42,
        ),
        data_stats=DataStats(train_examples=500, val_examples=80, test_examples=60),
        artifacts=Artifacts(
            status="published",
            adapter_weights=f"adapters/{doc_type}/{foundation_version}/",
        ),
        # Set at construction, not after: the manifest refuses a promoted status
        # with no record of when and by whom, and assignment after the fact
        # cannot satisfy a model validator.
        promotion=Promotion(
            gated_against=None,
            beat_previous_on_all_gates=True,
            promoted_by="test",
            promoted_at=datetime.now(UTC),
        ),
    )
    write_manifest(manifest, client)
    return run_id


def test_a_minor_bump_does_not_trigger_the_cascade(client, controller):
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")

    ctx = make_context(client, controller, out_version="v1.1")
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()


def test_a_major_bump_blocks_until_dependent_adapters_are_revalidated(client, controller):
    """Every per-type adapter was trained on the previous Foundation's weights,
    so promoting a new one without re-validating them ships three models that
    were never evaluated against the base they now sit on."""
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")
    _graduate_adapter(client, "policy", "v1")

    ctx = make_context(client, controller, out_version="v2")
    report = run_stages(ctx, stages_for("all"), command="all")

    assert report.blocked_at == "evaluation_gate"
    detail = report.results[-1].detail
    assert "dependent adapter" in detail
    assert "policy-adapter-v1" in detail


def test_the_cascade_clears_once_every_dependent_has_passed(client, controller):
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")
    _graduate_adapter(client, "policy", "v1")
    _graduate_adapter(client, "lossrun", "v1")

    from orchestration.pipeline_dag import foundation_upgrade_work_list

    probe = make_context(client, controller, out_version="v2")
    dependents = foundation_upgrade_work_list(probe).dependents
    assert dependents, "the graduated adapters should depend on extractor-v1"

    ctx = make_context(client, controller, out_version="v2",
                       revalidation_evidence=dict.fromkeys(dependents, True))
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()


def test_partial_revalidation_still_blocks(client, controller):
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")
    _graduate_adapter(client, "policy", "v1")
    _graduate_adapter(client, "lossrun", "v1")

    from orchestration.pipeline_dag import foundation_upgrade_work_list

    probe = make_context(client, controller, out_version="v2")
    dependents = foundation_upgrade_work_list(probe).dependents
    partial = dict.fromkeys(dependents, True)
    partial[dependents[0]] = False

    ctx = make_context(client, controller, out_version="v2", revalidation_evidence=partial)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.blocked_at == "evaluation_gate"
    assert dependents[0] in report.results[-1].detail


@pytest.mark.parametrize(
    ("previous", "candidate", "expected"),
    [
        ("extractor-v1", "v2", True),
        ("extractor-v1", "v1.1", False),
        ("extractor-v2.3", "v3", True),
        (None, "v1", False),
    ],
)
def test_major_bump_detection(previous, candidate, expected):
    from orchestration.pipeline_dag import is_major_bump

    assert is_major_bump(previous, candidate) is expected


# --------------------------------------------------------------------------
# Retry policy and configuration
# --------------------------------------------------------------------------


def test_a_transient_stage_failure_is_retried(client, controller):
    from orchestration.pipeline_dag import Stage, StageResult

    calls = {"n": 0}

    def flaky(_ctx):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient object-store error")
        return StageResult("flaky", "completed", "second attempt")

    ctx = make_context(client, controller, max_attempts=2)
    report = run_stages(ctx, [Stage(1, "flaky", "finetune", False, flaky)], command="finetune")
    assert report.ok and calls["n"] == 2


def test_every_stage_records_its_wall_time_and_the_run_report_is_written_after_each(client, controller,
                                                                                    monkeypatch, tmp_path):
    """A run's time is measured, not guessed: each stage's wall seconds go into
    its result, and the report so far is written after every stage, so a run
    that dies part-way still says what the finished stages took."""
    import json

    from orchestration.pipeline_dag import PROGRESS_DIR_ENV, Stage, StageResult

    monkeypatch.setenv(PROGRESS_DIR_ENV, str(tmp_path))
    ctx = make_context(client, controller, out_version="v1")
    name = f"run-finetune-{ctx.scope.name}-{paths._tenant(ctx.tenant_id)}-v1.json"
    written = []

    def first(_ctx):
        return StageResult("first", "completed", "done")

    def second(_ctx):
        written.append(json.loads((tmp_path / name).read_text(encoding="utf-8")))
        raise PipelineError("cannot run")

    report = run_stages(ctx, [Stage(1, "first", "finetune", False, first),
                              Stage(2, "second", "finetune", False, second)], command="finetune")
    assert [r.data["wall_seconds"] >= 0 for r in report.results] == [True, True]
    assert [s["stage"] for s in written[0]["stages"]] == ["first"]      # written before stage 2 ran
    final = json.loads((tmp_path / name).read_text(encoding="utf-8"))
    assert final["failed_at"] == "second" and "wall_seconds" in final["stages"][1]


def test_two_scopes_under_one_version_keep_their_own_run_reports(client, controller):
    from common.scopes import get_scope
    from orchestration.pipeline_dag import RunReport, progress_report_name

    report = RunReport(command="finetune", version="v1")
    cgl = make_context(client, controller, scope=get_scope("casualty_fleet"), tenant_id="cgl")
    personal = make_context(client, controller, scope=get_scope("personal_lines"), tenant_id="personal")
    assert progress_report_name(report, cgl) != progress_report_name(report, personal)
    assert progress_report_name(report, cgl) == "run-finetune-casualty_fleet-cgl-v1.json"


def test_a_blocked_gate_is_never_retried(client, controller):
    """Retrying a gate block would be an override path with extra steps."""
    from orchestration.pipeline_dag import Stage

    calls = {"n": 0}

    def blocking(_ctx):
        calls["n"] += 1
        raise GateBlocked("regressed")

    ctx = make_context(client, controller, max_attempts=5)
    report = run_stages(ctx, [Stage(1, "gate", "finetune", False, blocking)], command="finetune")
    assert report.blocked_at == "gate" and calls["n"] == 1


def test_the_config_file_is_actually_read():
    """A config file nothing reads documents a policy the system does not follow."""
    from orchestration import settings

    assert settings.gpu_class_for("training").startswith("H200")
    assert "A100" not in settings.gpu_class_for("preprocessing")
    assert settings.retry_policy()["retry_on_gate_block"] is False
    assert settings.defaults()["formats"] == ["bf16"]


def test_cli_defaults_come_from_the_config_file():
    from orchestration import settings

    args = cli.build_parser().parse_args(["package", "--version", "v2"])
    assert args.formats == settings.defaults()["formats"]
    assert args.dtype == settings.defaults()["dtype"]


def test_the_gpu_flag_actually_reaches_the_training_pod(client, controller):
    """A flag that parses and changes nothing is worse than no flag."""
    seed_corpus(client)
    args = cli.build_parser().parse_args([
        "finetune", "--out-version", "v1", "--gpu", "H100-80G", "--skip-ingest",
    ])
    ctx = cli.build_context(args, client=client, controller=controller,
                            raw_client=ingestion_client(client),
                            min_labels_per_type=1, dry_run=True,
                            metrics_provider=lambda _c: dict(PASSING_METRICS))
    assert ctx.gpu_class == "H100-80G"

    launched: list[str] = []
    original = controller.launch
    controller.launch = lambda spec: (launched.append(spec.gpu_class), original(spec))[1]  # type: ignore[method-assign]
    run_stages(ctx, stages_for("finetune"), command="finetune")
    assert launched == ["H100-80G"]


# --------------------------------------------------------------------------
# Checkpoint selection in the pipeline (arch v2.1 §11.2)
# --------------------------------------------------------------------------

def test_the_selected_checkpoint_is_what_gets_merged(client, controller):
    """The selection is worth nothing if merge ignores it and folds in the
    staged adapter directory anyway."""
    seed_corpus(client)
    chosen = "/runpod-volume/staging/adapters/foundation/v1/checkpoint-300"
    ctx = make_context(
        client, controller,
        checkpoints=[
            "/runpod-volume/staging/adapters/foundation/v1/checkpoint-100",
            "/runpod-volume/staging/adapters/foundation/v1/checkpoint-200",
            chosen,
        ],
        checkpoint_scorer=lambda p: {
            "field_normalized_match": 0.95 if p.endswith("300") else 0.80
        },

    )
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert ctx.results["checkpoint_eval"].data["selected"] == chosen
    assert ctx.results["merge"].data["selected_checkpoint"] == chosen


def test_a_run_with_no_checkpoints_skips_selection_rather_than_guessing(client, controller):
    """A dry run never launched ms-swift, so there is nothing to choose between.
    Merging the staged adapter directory is the honest fallback."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert ctx.results["checkpoint_eval"].status == "skipped"
    assert ctx.results["merge"].data["selected_checkpoint"] is None


def test_the_selection_is_recorded_where_a_later_review_can_read_it(client, controller):
    """The merged weights do not say which checkpoint they came from, and once
    the staging volume is reclaimed nothing else does either."""
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        checkpoints=["/runpod-volume/staging/adapters/foundation/v1/checkpoint-100"],
        checkpoint_scorer=lambda _p: {"field_normalized_match": 0.9},

    )
    run_stages(ctx, stages_for("finetune"), command="finetune")

    record = client.read_json(paths.checkpoint_selection("v1"))
    assert record["selected"].endswith("checkpoint-100")
    assert record["selection_metric"] == "field_normalized_match"


# --------------------------------------------------------------------------
# The calibrate stage (arch v2.1 §13 stage 9)
# --------------------------------------------------------------------------

def _calibration_samples(n: int = 400, correct_rate: float = 0.8):
    """Labelled validation features, split into the two halves of §8.2."""
    import random

    from calibration.features import build_features

    rng = random.Random(17)

    def half(seed_offset: int):
        out = []
        for i in range(n):
            correct = rng.random() < correct_rate
            logprobs = [-0.05 - rng.random() * 0.1] * 3 if correct else [-2.0 - rng.random()] * 3
            out.append((
                build_features(
                    field_path="policy_number", value=f"WC-{i + seed_offset}",
                    logprobs=logprobs, document={},
                    page_text=f"WC-{i + seed_offset}" if correct else "nothing",
                ),
                correct,
            ))
        return out

    return {"calibration": half(0), "threshold": half(10_000)}


def test_calibration_fits_per_serving_format(client, controller):
    """Quantization moves the logprob distribution, so a calibrator fitted on
    bf16 reports confidence for a distribution FP8 does not produce. Sharing one
    is not an optimisation, it is a silently wrong number (arch v2.1 §5.3)."""
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        release_id="release-2026.11.1",
        calibration_samples={"bf16": _calibration_samples()},
    )
    report = run_stages(ctx, stages_for("all"), command="all")

    assert report.ok, report.render()
    assert "bf16" in ctx.calibrators and "bf16" in ctx.thresholds

    stored = client.read_json(paths.release_calibrators("release-2026.11.1", "bf16"))
    assert stored["calibrators"]["serving_format"] == "bf16"
    assert stored["thresholds"]["fitted_on"].startswith("validation threshold half")


def test_no_calibrator_is_shared_across_serving_formats(client, controller):
    """A "*" key used to calibrate every format from one sample set, bypassing
    the per-format rule the stage exists to enforce."""
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        release_id="release-2026.11.1",
        calibration_samples={"*": _calibration_samples()},
    )
    run_stages(ctx, stages_for("all"), command="all")

    assert "bf16" not in ctx.calibrators
    assert not client.exists(paths.release_calibrators("release-2026.11.1", "bf16"))


def test_a_release_with_no_labelled_features_ships_uncalibrated_and_says_so(client, controller):
    """Honest rather than silent: without labelled validation features there is
    nothing to fit, every field routes to review, and an operator should know
    that happened rather than discover it in the review queue."""
    seed_corpus(client)
    ctx = make_context(client, controller, release_id="release-2026.11.1")
    report = run_stages(ctx, stages_for("all"), command="all")

    calibrate = ctx.results["calibrate"]
    assert calibrate.status == "skipped"
    assert "route to review" in calibrate.detail
    assert report.ok, "an uncalibrated release is a degraded one, not a failed one"


def test_the_calibration_stage_reports_its_guarantee(client, controller):
    """v1 used 0.70 for every field, marked "tunable". It was never tuned and
    could not be, because nothing measured what error rate it bought."""
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        release_id="release-2026.11.1",
        calibration_samples={"bf16": _calibration_samples()},
    )
    run_stages(ctx, stages_for("all"), command="all")

    body = ctx.results["calibrate"].data["by_format"]["bf16"]
    assert "auto_accept_error_rate" in body
    assert any("at most" in g or "routed to review" in g for g in body["guarantees"])


# --------------------------------------------------------------------------
# What the reordering exposed
# --------------------------------------------------------------------------

def test_a_failed_run_is_not_resurrected_by_the_gate():
    """The gate runs AFTER training now (arch v2.1 §13), so apply_to_manifest
    was writing "evaluated" over a run recorded as failed — turning a pod that
    OOM'd at step 40 into a publishable release. Whatever was scored in that case
    did not come from those weights, because they were never written."""
    from evaluation.gating import apply_to_manifest, promotion_gate

    class _Promotion:
        beat_previous_on_all_gates = None
        failed_gates: list = []
        gated_against = None

    class _Manifest:
        run_id = "extractor-v1"
        promotion = _Promotion()
        status = "failed"

    manifest = apply_to_manifest(promotion_gate(dict(PASSING_METRICS), None), _Manifest())
    assert manifest.status == "failed", "a run that never wrote weights cannot be evaluated"


def test_the_staging_precondition_survives_skip_quantize(client, controller):
    """`package` operates on what `finetune` staged, so a reclaimed volume is a
    precondition failure for the whole command. Checking it inside the first
    stage meant --skip-quantize turned that stage into a no-op and the check
    disappeared with it."""
    ctx = make_context(client, controller, skip_quantize=True)
    report = run_stages(ctx, stages_for("package"), command="package")

    assert report.failed_at == "quantize"
    assert "not on the staging volume" in report.results[-1].detail


# --------------------------------------------------------------------------
# The release bundle (arch v2.1 §12.3)
# --------------------------------------------------------------------------


def test_package_writes_the_release_bundle_it_pins(client, controller):
    """stage_push was registered as `package` and never built a ReleaseBundle,
    so nothing pinned the prompt, schema, calibrators and OCR version that the
    served weights depend on."""
    seed_corpus(client)
    ctx = make_context(client, controller, release_id="release-2026.11.1")
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()

    bundle = client.read_json(paths.release_bundle("release-2026.11.1"))
    assert bundle["adapter"] == "extractor-v1"
    assert bundle["serving_formats"]["bf16"] == paths.merged_model_dir("v1")
    assert bundle["prompt_hash"] and bundle["vllm_config_hash"] and bundle["lockfile_hash"]
    assert bundle["ocr_pin"]["mineru_version"] == "2.0.0"
    assert "bf16" in bundle["gate_reports"]

    index = client.read_json(paths.release_index())
    assert [row["release_id"] for row in index] == ["release-2026.11.1"]


def test_an_uncalibrated_release_is_gated_not_promoted(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller, release_id="release-2026.11.1")
    run_stages(ctx, stages_for("all"), command="all")

    bundle = client.read_json(paths.release_bundle("release-2026.11.1"))
    assert bundle["status"] == "gated"
    reasons = ctx.results["package"].data["not_promoted_because"]
    assert any("no calibrator" in r for r in reasons)


def test_a_calibrated_gated_bf16_release_is_promoted(client, controller):
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        release_id="release-2026.11.1",
        calibration_samples={"bf16": _calibration_samples()},
    )
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()

    bundle = client.read_json(paths.release_bundle("release-2026.11.1"))
    assert bundle["status"] == "promoted", ctx.results["package"].data["not_promoted_because"]
    assert bundle["calibrators"]["bf16"] == paths.release_calibrators("release-2026.11.1", "bf16")


def test_a_quantized_format_is_not_promoted_on_the_bf16_gate_run(client, controller):
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        release_id="release-2026.11.1",
        formats=["bf16", "fp8"], fp8_verified=True,
        calibration_samples={"bf16": _calibration_samples(), "fp8": _calibration_samples()},
    )
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()

    bundle = client.read_json(paths.release_bundle("release-2026.11.1"))
    assert bundle["status"] == "gated"
    assert set(bundle["gate_reports"]) == {"bf16"}


# --------------------------------------------------------------------------
# Scopes: independent runs sharing one corpus (arch v2.1 §4.1)
# --------------------------------------------------------------------------


def _scoped_ctx(client, controller, scope_name: str, **over):
    from common.scopes import get_scope

    return make_context(
        client, controller,
        scope=get_scope(scope_name),
        release_id=f"release-2026.9.{1 if scope_name == 'unified' else 2}",
        **over,
    )


def test_the_shared_stages_run_once_and_the_rest_run_per_scope(client, controller):
    """Stages 1-4 build the corpus every scope reads. Running them per scope would
    re-ingest and re-split the same documents, and a second split draw is the
    leakage the corpus view exists to avoid."""
    from common.scopes import get_scope
    from orchestration.pipeline_dag import PER_SCOPE_STAGES, SHARED_STAGES, run_multi_scope

    seed_corpus(client)
    contexts = {}

    def build(scope):
        ctx = _scoped_ctx(client, controller, scope.name)
        contexts[scope.name] = ctx
        return ctx

    report = run_multi_scope(
        build, [get_scope("unified"), get_scope("policy")], command="all",
    )

    assert report.ok, report.render()
    assert [r.name for r in report.shared.results] == [s.name for s in SHARED_STAGES]
    for name in ("unified", "policy"):
        assert [r.name for r in report.by_scope[name].results] == [
            s.name for s in PER_SCOPE_STAGES
        ], name


def test_two_scopes_at_one_version_publish_only_their_own_artifacts(client, controller):
    """stage_push matched runs with run_id.endswith(version), so packaging one
    scope published every scope's run at that tag — each against THIS scope's
    paths, advertising prefixes holding another scope's weights or nothing."""
    seed_corpus(client)

    unified = _scoped_ctx(client, controller, "unified")
    run_stages(unified, stages_for("all"), command="all")
    policy = _scoped_ctx(client, controller, "policy")
    run_stages(policy, stages_for("all"), command="all")

    assert unified.results["package"].data["published"] == ["extractor-v1"]
    assert policy.results["package"].data["published"] == ["policy-v1"]

    pushed = policy.results["package"].data["pushed"]
    assert "adapter:policy" in pushed and "adapter:unified" not in pushed
    assert "/scope/policy/" in pushed["merged:policy"]


def test_packaging_one_scope_leaves_another_scopes_staged_model_alone(client, controller):
    """The clear wiped the whole staging root, so packaging `policy` deleted
    `unified`'s staged merged model — the only copy until its own package run."""
    seed_corpus(client)

    unified = _scoped_ctx(client, controller, "unified")
    run_stages(unified, stages_for("finetune"), command="finetune")
    staged_unified = paths.staging_merged_model_dir("v1", scope="unified")
    assert unified.volume.exists(staged_unified)

    # The same controller, so the same staging volume — which is exactly the
    # situation the old blanket clear destroyed.
    policy = _scoped_ctx(client, controller, "policy")
    report = run_stages(policy, stages_for("all"), command="all")

    assert report.ok, report.render()
    assert unified.volume.exists(staged_unified), "packaging policy cleared unified's merge"
    assert not unified.volume.exists(paths.staging_merged_model_dir("v1", scope="policy"))


def test_each_scope_writes_its_own_release_bundle(client, controller):
    seed_corpus(client)
    for name in ("unified", "policy"):
        ctx = _scoped_ctx(client, controller, name)
        assert run_stages(ctx, stages_for("all"), command="all").ok

    unified = client.read_json(paths.release_bundle("release-2026.9.1"))
    policy = client.read_json(paths.release_bundle("release-2026.9.2"))

    assert unified["adapter"] == "extractor-v1" and unified["scope"] == "unified"
    assert unified["doc_types"] == [], "empty means every active type"
    assert policy["adapter"] == "policy-v1" and policy["scope"] == "policy"
    assert policy["doc_types"] == ["policy"], "what this release may serve"


def test_a_gate_block_in_one_scope_does_not_stop_another(client, controller):
    """They are separate adapters: a policy regression says nothing about the
    lossrun model, and stopping the second run would only mean re-running it."""
    from common.scopes import get_scope
    from orchestration.pipeline_dag import run_multi_scope

    seed_corpus(client)
    regressed = {**PASSING_METRICS, "field_exact_match": 0.10}

    def build(scope):
        return _scoped_ctx(
            client, controller, scope.name,
            metrics_provider=(
                (lambda _c: dict(regressed)) if scope.name == "policy"
                else (lambda _c: dict(PASSING_METRICS))
            ),
        )

    report = run_multi_scope(
        build, [get_scope("policy"), get_scope("unified")], command="all",
    )

    assert not report.by_scope["policy"].ok
    assert report.by_scope["unified"].ok, "one scope's block stopped another"
    assert not report.ok, "the aggregate must still fail"
    assert report.exit_code == 1


def test_the_release_id_is_resolved_per_scope():
    """Each scope produces its own release, so sharing one id would write every
    scope's calibrators and gate decision to one prefix."""
    from common.scopes import get_scope

    policy, unified = get_scope("policy"), get_scope("unified")

    assert cli.release_id_for("release-2026.9.1", policy) == "release-2026.9.1"
    pairs = ["unified=release-2026.9.1", "policy=release-2026.9.2"]
    assert cli.release_id_for(pairs, unified) == "release-2026.9.1"
    assert cli.release_id_for(pairs, policy) == "release-2026.9.2"
    # Named for other scopes only: refused later by assert_release_id, rather
    # than silently taking an id meant for a different release.
    assert cli.release_id_for(["unified=release-2026.9.1"], policy) == ""


def test_scope_choices_come_from_the_config_file():
    """A new scope is a YAML entry, never a code change — including at the CLI."""
    from common.scopes import load_scopes

    args = cli.build_parser().parse_args(
        ["all", "--out-version", "v2", "--scope", "policy", "--scope", "unified"]
    )
    assert args.scopes == ["policy", "unified"]
    assert set(load_scopes()) >= {"unified", "policy", "lossrun"}


def test_the_gate_reads_its_own_scopes_eval_report(client, controller):
    """The gate became scope-aware while its INPUT did not, so a policy run was
    judged on the unified model's numbers — or found no report and failed."""
    from orchestration.pipeline_dag import eval_report_metrics

    seed_corpus(client)
    ctx = _scoped_ctx(client, controller, "policy")

    client.write_json(paths.eval_report("v1"), {"gate_metrics": {"field_exact_match": 0.10}})
    client.write_json(
        paths.eval_report("v1", scope="policy"), {"gate_metrics": {"field_exact_match": 0.99}}
    )

    assert eval_report_metrics(ctx)["field_exact_match"] == 0.99


def test_two_scopes_cannot_share_one_release_id():
    """Each scope produces its own release, all addressed by the id, so the
    second bundle would overwrite the first and delete its index row."""
    from common.scopes import get_scope

    with pytest.raises(cli.ReleaseIdError, match="cannot share an id"):
        cli.release_id_for(["release-2026.9.1"], get_scope("policy"), scope_count=2)

    assert cli.release_id_for(
        ["release-2026.9.1"], get_scope("policy"), scope_count=1
    ) == "release-2026.9.1"


# --------------------------------------------------------------------------
# Phase 1: deterministic failures, resumed merges
# --------------------------------------------------------------------------


def test_a_deterministic_training_failure_is_not_retried(client, controller):
    """A missing image or an over-length row fails the same way twice — after
    re-staging and re-measuring the corpus, or a second full training run."""
    from orchestration.pipeline_dag import Stage
    from training.train import TrainingError

    calls = []

    def boom(ctx):
        calls.append(1)
        raise TrainingError("over-length row")

    ctx = make_context(client, controller, max_attempts=3)
    report = run_stages(ctx, [Stage(5, "training", "finetune", True, boom)], command="finetune")
    assert report.failed_at == "training"
    assert len(calls) == 1, "a deterministic failure was retried"


def test_a_transient_failure_is_still_retried(client, controller):
    from orchestration.pipeline_dag import Stage

    calls = []

    def flaky(ctx):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("throttled")
        from orchestration.pipeline_dag import StageResult
        return StageResult("training", "completed")

    ctx = make_context(client, controller, max_attempts=2)
    report = run_stages(ctx, [Stage(5, "training", "finetune", True, flaky)], command="finetune")
    assert report.ok and len(calls) == 2


def test_a_resumed_merge_reads_the_checkpoint_already_selected(client, controller):
    """`--from-stage merge` did not run checkpoint_eval, but an earlier process
    did and wrote its choice down."""
    from orchestration.pipeline_dag import stage_merge

    ctx = make_context(client, controller)
    chosen = "/runpod-volume/staging/adapters/foundation/v1/v0-1/checkpoint-150"
    client.write_json(paths.checkpoint_selection("v1", scope=ctx.scope.name), {"selected": chosen})
    result = stage_merge(ctx)
    assert result.data["selected_checkpoint"] == chosen


def test_package_refuses_a_version_whose_gate_blocked(client, controller):
    """`package` is its own command; it read nothing the gate wrote, so running it
    after a blocked gate published the blocked version anyway."""
    from orchestration.pipeline_dag import stage_push

    seed_corpus(client)
    ctx = make_context(client, controller, release_id="release-2026.11.1")
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()

    key = paths.gate_decision("v1")
    client.write_json(key, {**client.read_json(key), "passed": False, "failed_gates": ["x"]})
    with pytest.raises(PipelineError, match="did not pass its promotion gate"):
        stage_push(ctx)


def test_a_resumed_package_finds_the_calibrators_in_blob(client, controller):
    """Resumed at package, ctx.calibrators is empty; the calibrators the earlier
    process wrote are in Blob and the bundle must still point at them."""
    from orchestration.pipeline_dag import build_release_bundle

    seed_corpus(client)
    first = make_context(
        client, controller, release_id="release-2026.11.1",
        calibration_samples={"bf16": _calibration_samples()},
    )
    assert run_stages(first, stages_for("all"), command="all").ok

    resumed = make_context(client, controller, release_id="release-2026.11.1")
    bundle, reasons = build_release_bundle(resumed)
    assert "bf16" in bundle.calibrators
    assert not any("no calibrator" in r for r in reasons)


def test_a_family_with_any_synthetic_member_stays_in_train():
    """A family moves as a unit. Made splittable by its one real member, it could
    carry its generated labels into val or test, where eval scores the generator."""
    from types import SimpleNamespace

    from orchestration.pipeline_dag import group_records

    def doc(source_id, synthetic):
        return SimpleNamespace(
            source_id=source_id, doc_type="acord", family="fam-1", carrier=None,
            synthetic=synthetic,
        )

    [record] = group_records([doc("real-1", False), doc("synth-1", True)])["acord"]
    assert record.synthetic
    assert record.source_ids == ["real-1", "synth-1"]


def _set_ocr_meta(client, source_id: str, **fields) -> None:
    doc_type = source_id.rsplit("_", 1)[0]
    key = paths.ocr_meta(doc_type, source_id)
    client.write_json(key, {**client.read_json(key), **fields})


def test_a_render_only_document_is_not_trained_as_ocr(client, controller):
    """Rendered for labeling and never OCR'd, it has no page text; built into
    ocr_plus_image rows it would teach that full pages can be blank."""
    from orchestration.pipeline_dag import load_labeled_documents

    seeded = seed_corpus(client)
    _set_ocr_meta(client, seeded[0], render_only=True)
    loaded = {d.source_id for d in load_labeled_documents(make_context(client, controller))}
    assert seeded[0] not in loaded
    assert set(seeded[1:]) <= loaded


def test_a_document_with_no_recorded_pages_is_skipped_not_assumed_one(client, controller):
    from orchestration.pipeline_dag import load_labeled_documents

    seeded = seed_corpus(client)
    _set_ocr_meta(client, seeded[0], page_count=0)
    loaded = {d.source_id for d in load_labeled_documents(make_context(client, controller))}
    assert seeded[0] not in loaded


def test_a_corpus_with_no_training_rows_is_refused(client, controller, monkeypatch):
    from data_pipeline.dataset_builder import build_jsonl
    from orchestration import pipeline_dag

    seed_corpus(client)
    real = build_jsonl.build_corpus

    def no_train(*args, **kwargs):
        built = real(*args, **kwargs)
        built.rows_by_split["train"] = []
        return built

    monkeypatch.setattr(pipeline_dag, "build_corpus", no_train, raising=False)
    monkeypatch.setattr(build_jsonl, "build_corpus", no_train)
    ctx = make_context(client, controller)
    with pytest.raises(PipelineError, match="no train"):
        pipeline_dag.stage_dataset_build(ctx)
    assert not client.exists(paths.corpus_manifest(ctx.corpus, None))


def test_a_corpus_that_set_aside_too_many_documents_is_refused_by_the_build(client, controller, monkeypatch):
    """The chain runs `preflight --skip-corpus`, so the corpus is built once, in
    stage_dataset_build, and preflight's set-aside guard has to hold there too."""
    from data_pipeline.dataset_builder import build_jsonl
    from orchestration import pipeline_dag, preflight

    seed_corpus(client)
    real = build_jsonl.build_corpus

    def one_document_set_aside(*args, **kwargs):
        # Every split filled from the fixture's 4 documents, and the 4th set aside.
        built = real(*args, **kwargs)
        rows = built.all_rows
        ids = sorted({r["source_id"] for r in rows})
        assert len(ids) == 4
        of = {i: [r for r in rows if r["source_id"] == i] for i in ids}
        built.rows_by_split = {"train": of[ids[0]], "val": of[ids[1]], "test": of[ids[2]]}
        return built

    monkeypatch.setattr(pipeline_dag, "build_corpus", one_document_set_aside, raising=False)
    monkeypatch.setattr(build_jsonl, "build_corpus", one_document_set_aside)
    # 1 of the fixture's 4 documents is 25%: lowered so one document crosses it.
    monkeypatch.setattr(preflight, "FAIL_REJECTED_SHARE", 0.2)
    ctx = make_context(client, controller, dry_run=False)
    with pytest.raises(PipelineError, match="set aside 1 of 4"):
        pipeline_dag.stage_dataset_build(ctx)
    assert not client.exists(paths.corpus_manifest(ctx.corpus, None))


def test_in_process_on_the_pod_the_mounted_volume_is_enough(tmp_path):
    """On the pod, LocalBackend runs the stage here and writes to the mounted
    volume; there is no second pod whose disk could vanish, so no volume id."""
    controller = RunPodController(backend=LocalBackend(), volume_id=None, volume_mount=str(tmp_path),
                                  git_commit="abc1234")
    handle = controller.launch(controller.spec_for("training"))
    assert handle.status == "running"
    controller.terminate(handle)


def test_in_process_off_the_pod_is_still_refused(tmp_path):
    controller = RunPodController(backend=LocalBackend(), volume_id=None,
                                  volume_mount=str(tmp_path / "no-such-mount"), git_commit="abc1234")
    with pytest.raises(PodLaunchError, match="no staging volume attached"):
        controller.launch(controller.spec_for("training"))
