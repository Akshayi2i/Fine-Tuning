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
from common.config import served_mineru
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
#: The MinerU every seeded page was read by: the served one, the OCR pin.
SERVED_MINERU = served_mineru()

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
            # Read by the served MinerU, as the OCR pin asks (configs/shared/ocr.yaml).
            "mineru_version": SERVED_MINERU["version"], "mineru_tier": SERVED_MINERU["tier"],
            "ocr_device": "cuda", "resolution_cap_px": 1792,
            "source_checksum": f"sha-{source_id}", "table_row_counts": {"1": 0}, "failed_pages": [],
        })
        client.write_json(paths.golden_label(doc_type, source_id),
                          json.loads(golden_path.read_text(encoding="utf-8")))
        client.write_json(paths.label_metadata(doc_type, source_id), metadata)
        seeded.append(source_id)
    return seeded


def seeded_policies(seeded: list[str]) -> list[str]:
    """The seeded documents the policy adapter trains on. The fixtures also hold
    an ACORD form and a Loss Run, which a policy cycle loads none of."""
    return [source_id for source_id in seeded if source_id.startswith("policy_")]


def seed_policy_copy(client: BlobClient, source_id: str, *, like: str = "policy_0001",
                     own_family: bool = False) -> str:
    """One more labelled policy, a copy of ``like`` under its own id and checksum.

    A copy is a near-duplicate of the same account, so it groups into
    ``like``'s family and moves with it between splits. ``own_family`` names
    another insured (the declared account) and appends words no other copy
    holds (below the near-duplicate threshold), so the copy is a family of its
    own.
    """
    import random

    raw = ingestion_client(client)
    raw.write_bytes(paths.raw_pdf("policy", source_id), b"%PDF-1.7 fixture")
    raw.write_json(paths.raw_metadata("policy", source_id), {
        "source_id": source_id, "doc_type": "policy", "checksum_sha256": f"sha-{source_id}",
    })
    text = client.read_text(paths.processed_page("policy", like, 1, "md"))
    golden = client.read_json(paths.golden_label("policy", like))
    if own_family:
        rng = random.Random(source_id)
        insured = golden["named_insured"]["primary_name"]
        name = f"{source_id.replace('_', ' ').title()} Holdings LLC"
        text = text.replace(insured["raw"], name)
        golden["named_insured"]["primary_name"] = {**insured, "raw": name, "parsed": name}
        words = [f"{rng.choice('bcdfghklmnprstvz')}{rng.choice('aeiou')}{rng.randrange(10**6)}"
                 for _ in range(150)]
        text += "\n\n" + " ".join(words) + "\n"
    client.write_text(paths.processed_page("policy", source_id, 1, "md"), text)
    client.write_bytes(paths.processed_page("policy", source_id, 1, "png"),
                       client.read_bytes(paths.processed_page("policy", like, 1, "png")))
    client.write_json(paths.ocr_meta("policy", source_id), {
        **client.read_json(paths.ocr_meta("policy", like)),
        "source_id": source_id, "source_checksum": f"sha-{source_id}",
    })
    client.write_json(paths.golden_label("policy", source_id), golden)
    client.write_json(paths.label_metadata("policy", source_id),
                      {**client.read_json(paths.label_metadata("policy", like)), "source_id": source_id})
    return source_id


def make_context(client: BlobClient, controller: RunPodController, **over) -> StageContext:
    defaults = dict(
        client=client,
        raw_client=ingestion_client(client),
        controller=controller,
        out_version="v1",
        tenant_id=None,
        dry_run=True,
        skip_ingest=True,
        min_labels_per_type=1,
        git_commit="abc1234",
        metrics_provider=lambda _ctx: dict(PASSING_METRICS),
    )
    defaults.update(over)
    return StageContext(**defaults)  # type: ignore[arg-type]


def promote_in_index(client: BlobClient, version: str, tenant_id: str | None = None,
                     *, created_at: str = "2026-10-01T00:00:00+00:00") -> None:
    """Make ``version`` the tenant's promoted release in its release index, as
    ``package`` or ``promote`` leaves it: the row the gate's baseline is read
    from (``pipeline_dag.promoted_version``). A dry run's bundle is gated (it
    has no calibrator), so the row is written here rather than earned."""
    key = paths.release_index(tenant_id)
    rows = [r for r in (client.read_json(key) if client.exists(key) else [])
            if r.get("release_id") != version]
    rows.append({"release_id": version, "status": "promoted", "adapter": f"policy-{version}",
                 "tenant_scope": paths._tenant(tenant_id), "created_at": created_at})
    client.write_json(key, rows)


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
    assert len(STAGES) == 11


def test_stage_numbers_match_the_architecture_order():
    assert [s.number for s in STAGES] == list(range(1, 12))


def test_the_stage_order_matches_the_architecture():
    """arch v2.1 §13, with nothing merged or quantized (decision 2026-10-23).
    Two orderings carry real meaning:

    Selection precedes calibration and the gate, because the gate scores what
    ships and what ships has to be chosen first (§11.2). And the gate follows
    calibrate: it reads auto_accept_error_rate, which does not exist until a
    threshold has been chosen.
    """
    order = [s.name for s in STAGES]
    assert order.index("training") < order.index("checkpoint_eval")
    assert order.index("checkpoint_eval") < order.index("calibrate")
    assert order.index("calibrate") < order.index("evaluation_gate")
    assert order.index("evaluation_gate") < order.index("package")


def test_no_stage_merges_or_quantizes():
    """The release is its adapter, served as a LoRA on the bf16 base: there is
    no merge or quantize stage to run, resume at or skip (decision 2026-10-23)."""
    from orchestration import pipeline_dag

    assert not {"merge", "quantize"} & {s.name for s in STAGES}
    for gone in ("stage_merge", "_is_merged", "stage_quantize", "_is_quantized"):
        assert not hasattr(pipeline_dag, gone), gone
    with pytest.raises(PipelineError, match="unknown stage 'merge'"):
        stages_from("merge", "all")


def test_finetune_ends_at_checkpoint_selection_and_package_owns_the_gate():
    """The command boundary moved with the gate (arch v2.1 §13c): `finetune`
    produces the adapter and chooses its checkpoint, `package` judges and
    publishes it."""
    finetune = [s.name for s in STAGES if s.command == "finetune"]
    package = [s.name for s in STAGES if s.command == "package"]

    assert finetune[-1] == "checkpoint_eval"
    assert package == ["calibrate", "evaluation_gate", "package"]


def test_from_stage_resumes_mid_pipeline():
    resumed = [s.name for s in stages_from("checkpoint_eval", "finetune")]
    assert resumed == ["checkpoint_eval"]
    assert [s.name for s in stages_from("training", "finetune")][0] == "training"


def test_from_stage_on_all_carries_through_into_package():
    """Resuming at checkpoint selection must still package afterwards — stopping
    at the end of finetune would leave the adapter staged and the operator none
    the wiser."""
    resumed = [s.name for s in stages_from("checkpoint_eval", "all")]
    assert resumed == ["checkpoint_eval", "calibrate", "evaluation_gate", "package"]


def test_unknown_stage_names_the_valid_ones():
    with pytest.raises(PipelineError, match="unknown stage"):
        stages_from("trian", "finetune")


# --------------------------------------------------------------------------
# finetune, end to end on fixtures
# --------------------------------------------------------------------------


def test_finetune_runs_ingest_through_checkpoint_selection(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert [r.name for r in report.results] == [s.name for s in FINETUNE_STAGES]
    assert ctx.volume.exists(paths.scoped_staging_adapter_dir("policy", "v1"))


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
    foundation = get("policy-v1", client)
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
    assert runs == ["policy-v1"], f"expected one policy run, got {runs}"
    assert ctx.results["training"].data["run_type"] == "scoped"


def test_the_policy_run_stages_one_adapter_and_merges_nothing(client, controller):
    """One adapter, staged under its own path, and no merged model at all: the
    release is served as a LoRA on the base (decision 2026-10-23). v1 produced a
    merged model per document type because each carried its own stacked adapter."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert ctx.volume.exists(paths.scoped_staging_adapter_dir("policy", "v1"))
    for scope in ("policy", None):
        assert not ctx.volume.exists(paths.staging_merged_model_dir("v1", scope=scope))
    for doc_type in ctx.doc_types:
        assert not ctx.volume.exists(paths.staging_merged_model_dir("v1", doc_type))


# --------------------------------------------------------------------------
# The gate — a hard stop with no way round it
# --------------------------------------------------------------------------


def test_a_regression_stops_the_release_not_the_build(client, controller):
    """Under v1 the gate ran before merge, so a regression stopped the build. It
    now runs AFTER checkpoint selection and calibrate (arch v2.1 §13), because the
    gate scores what is served: the base with the selected adapter applied.

    So the adapter is still built and chosen. What a regression stops is the
    RELEASE: nothing is packaged, nothing is published, and the staged adapter
    sits on a volume that will be reclaimed."""
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
    assert "checkpoint_eval" in ctx.results and "calibrate" in ctx.results, (
        "the gate scores the selected, calibrated adapter, so both precede it")
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

    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
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
    # Calibrate runs BEFORE the gate now: the gate reads auto_accept_error_rate,
    # which does not exist until thresholds are chosen.
    assert "calibrate" in ran
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
    freeze_as_scored(client)
    client.write_json(paths.eval_report("v1", scope="policy"), {"gate_metrics": dict(PASSING_METRICS),
                                                "eval_set": _scored_on()})

    ctx = make_context(client, controller, metrics_provider=None)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()


@pytest.mark.parametrize("eval_set, why", [
    ("other_tenant", "tenant 'acme''s frozen set"),
    ("unrecorded", "does not record which tenant"),
])
def test_an_eval_report_not_recorded_for_this_tenant_is_scored_again(client, controller, eval_set, why):
    """The report path names no tenant. A report at this version's key scored on
    another tenant's set - or saying nothing of whose - was reused as this
    candidate's metrics, and reached production on another tenant's documents."""
    from orchestration.pipeline_dag import eval_report_metrics

    report = {"gate_metrics": {**PASSING_METRICS, "field_normalized_match": 0.97}}
    if eval_set == "other_tenant":
        report["eval_set"] = _scored_on(tenant="acme", policy=160)
    client.write_json(paths.eval_report("v1", scope="policy"), report)
    # The lines the release serves, from its corpus: the scoring needs them.
    client.write_json(paths.corpus_manifest("v1"), {"policy_line_counts": {"homeowners": 1}})

    with pytest.raises(PipelineError, match=f"not reused.*{why}.*metrics nobody measured"):
        eval_report_metrics(make_context(client, controller, metrics_provider=None))

    class Loaded(Exception):
        pass

    def loader(_ctx, _fmt):
        raise Loaded
    # Off a dry run it is scored again, as when there is no report at all.
    with pytest.raises(Loaded):
        eval_report_metrics(make_context(client, controller, metrics_provider=None, dry_run=False,
                                         serving_model_loader=loader))


def test_the_baseline_comes_from_the_promoted_version_s_own_report(client, controller):
    """A regression is only detectable against a real baseline, and the CLI
    never supplied one."""
    from artifact_registry import paths
    from orchestration.pipeline_dag import default_baseline_metrics

    seed_corpus(client)
    first = make_context(client, controller, out_version="v1")
    run_stages(first, stages_for("all"), command="all")
    promote_in_index(client, "v1")
    client.write_json(paths.eval_report("v1", scope="policy"), {"gate_metrics": dict(PASSING_METRICS)})

    probe = make_context(client, controller, out_version="v1.1", baseline_metrics=None)
    baseline = default_baseline_metrics(probe)
    assert baseline == PASSING_METRICS


def test_per_document_scores_in_both_reports_gate_on_a_paired_bootstrap(client, controller):
    """With per-document scores in both eval reports, non-inferiority is a paired
    bootstrap and the improvement rule is enforced (arch §15.5): a release no
    better than the last is blocked unless it names the defect it fixes."""
    from artifact_registry import paths

    seed_corpus(client)
    run_stages(make_context(client, controller, out_version="v1"), stages_for("all"), command="all")
    promote_in_index(client, "v1")
    freeze_as_scored(client)
    same = {f"doc_{i:02d}": 0.9 for i in range(20)}
    per_document = {"field_normalized_match": same, "list_field_recall": same}
    for version in ("v1", "v1.1", "v1.2"):
        client.write_json(paths.eval_report(version, scope="policy"), {
            "gate_metrics": dict(PASSING_METRICS), "per_document": per_document,
            "eval_set": _scored_on(),
        })

    ctx = make_context(client, controller, out_version="v1.1",
                       metrics_provider=None)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.blocked_at == "evaluation_gate"
    decision = client.read_json(paths.gate_decision("v1.1", scope="policy"))
    assert decision["failed_gates"] == ["improvement"]
    assert decision["paired_documents"]["field_normalized_match"] == 20
    basis = {v["name"]: v["basis"] for v in decision["verdicts"]}
    assert basis["field_normalized_match"] == "paired_bootstrap"
    assert basis["field_exact_match"] == "point_estimate"

    fixed = make_context(client, controller, out_version="v1.2",
                         metrics_provider=None, fixes_defect="TICKET-12: wrong edition dates")
    assert run_stages(fixed, stages_for("all"), command="all").ok
    assert client.read_json(paths.gate_decision("v1.2", scope="policy"))["fixes_defect"].startswith("TICKET-12")


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
    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["tier"] == "interim"
    assert decision["eval_set"]["small_set_accepted"] is True
    assert decision["warnings"]


def _scored_on(tenant: str | None = None, frozen_at: str = "2026-10-01T00:00:00+00:00",
               fingerprint: str | None = None, **documents: int) -> dict:
    """An eval report's record of the yardstick it was scored on
    (``run_eval.eval_set_record``): the frozen set, the scoring code - the
    running code's, unless ``fingerprint`` says otherwise - and the documents
    per type, all of them real."""
    from evaluation.run_eval import scoring_fingerprint

    return {"tenant_id": paths._tenant(tenant), "frozen_at": frozen_at,
            "frozen_from_corpus": "v1", "scoring_fingerprint": fingerprint or scoring_fingerprint(),
            "documents_by_doc_type": documents, "real_documents_by_doc_type": documents}


def freeze_as_scored(client: BlobClient, tenant: str | None = None,
                     frozen_at: str = "2026-10-01T00:00:00+00:00") -> None:
    """The tenant's frozen set as ``_scored_on`` records it. An eval report is
    reused for the gate only when it was scored on the set as it stands now:
    one scored on a set frozen since, or on none, is scored again (H6)."""
    from evaluation.freeze_eval_set import manifest_key

    client.write_json(manifest_key(tenant_id=tenant), {"frozen_at": frozen_at, "frozen_from_corpus": "v1"})


def _promoted_and_candidate(client, controller, promoted: dict, candidate: dict) -> None:
    """Release v1 promoted with ``promoted`` as its eval report, and ``candidate``
    as v1.1's."""
    seed_corpus(client)
    run_stages(make_context(client, controller, out_version="v1"), stages_for("all"), command="all")
    promote_in_index(client, "v1")
    freeze_as_scored(client)
    client.write_json(paths.eval_report("v1", scope="policy"), promoted)
    client.write_json(paths.eval_report("v1.1", scope="policy"), candidate)


@pytest.mark.parametrize("case, why", [
    ("refrozen", "the set was frozen again"),
    ("other_tenant", "tenant 'other-tenant'"),
    ("other_scoring_code", "different scoring code"),
    ("unrecorded_set", "does not record which frozen eval set"),
])
def test_a_promoted_release_on_another_yardstick_is_not_compared(client, controller, case, why):
    """Decision 2026-10-22. The set frozen again with corrected golds, another
    tenant's set, or the same golds read under other label rules: the promoted
    release's numbers say nothing about this candidate's. Compared as a point
    estimate, its 0.99 on the old yardstick blocked a 0.93 on the new one as a
    regression. The candidate is judged on its floors alone, with no
    non-inferiority and no improvement rule, and the decision says why.

    Trained fresh: a cycle that continued v1 is also held to v1 line by line,
    which another yardstick cannot do (tests/test_incremental_training.py)."""
    scores = {f"doc_{i:02d}": 0.9 for i in range(20)}
    per_document = {"field_normalized_match": scores, "list_field_recall": scores}
    promoted = {"gate_metrics": {**PASSING_METRICS, "field_normalized_match": 0.99},
                "eval_set": _scored_on(), "per_document": per_document}
    if case == "refrozen":
        promoted["eval_set"] = _scored_on(frozen_at="2026-09-01T00:00:00+00:00")
    elif case == "other_tenant":
        promoted["eval_set"] = _scored_on(tenant="other-tenant")
    elif case == "other_scoring_code":
        promoted["eval_set"] = _scored_on(fingerprint="0123456789abcdef")
    else:
        del promoted["eval_set"]
    _promoted_and_candidate(client, controller, promoted, {
        "gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(), "per_document": per_document,
    })

    ctx = make_context(client, controller, out_version="v1.1",
                       metrics_provider=None, fresh=True)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()
    decision = client.read_json(paths.gate_decision("v1.1", scope="policy"))
    assert decision["failed_gates"] == [] and decision["paired_documents"] == {}
    assert {v["basis"] for v in decision["verdicts"]} == {"floor_only"}
    assert decision["baseline_metrics"] is None
    assert decision["baseline_metrics_not_compared"]["field_normalized_match"] == 0.99
    assert why in decision["baseline_not_comparable"]
    assert any("were not compared" in w and why in w and "floors alone" in w
               for w in decision["warnings"]), decision["warnings"]
    # Not a first release, and no improvement rule to report as unenforced.
    assert not any("improvement rule was not enforced" in w for w in decision["warnings"])
    gate = next(r for r in report.results if r.name == "evaluation_gate")
    assert "another yardstick" in gate.detail


@pytest.mark.parametrize("case, why", [
    ("no_shared_documents", "share no scored document"),
    ("no_per_document_scores", "has no per-document scores"),
])
def test_reports_on_one_yardstick_that_cannot_be_paired_say_so_in_the_decision(
        client, controller, case, why):
    """Two reports of one yardstick whose per-document scores cannot be paired
    fall back to a point comparison, and the decision says the improvement rule
    was not enforced and why - before, only a log line did, or nothing."""
    scores = {f"doc_{i:02d}": 0.9 for i in range(20)}
    promoted = {"gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
                "per_document": {"field_normalized_match": scores, "list_field_recall": scores}}
    if case == "no_shared_documents":
        other = {f"other_{i:02d}": 0.9 for i in range(20)}
        promoted["per_document"] = {"field_normalized_match": other, "list_field_recall": other}
    else:
        del promoted["per_document"]
    _promoted_and_candidate(client, controller, promoted, {
        "gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
        "per_document": {"field_normalized_match": scores, "list_field_recall": scores},
    })

    ctx = make_context(client, controller, out_version="v1.1",
                       metrics_provider=None)
    assert run_stages(ctx, stages_for("all"), command="all").ok
    decision = client.read_json(paths.gate_decision("v1.1", scope="policy"))
    assert decision["paired_documents"] == {} and decision["baseline_not_comparable"] is None
    assert {v["basis"] for v in decision["verdicts"] if v["current"] is not None} == {"point_estimate"}
    assert any("not paired" in w and why in w for w in decision["warnings"]), decision["warnings"]
    assert any("improvement rule was not enforced" in w for w in decision["warnings"])


def test_a_report_without_weights_pairs_unweighted_on_both_sides(client, controller):
    """A weight missing on one side defaulted to one there only, so the bootstrap
    compared the promoted report's plain mean with the candidate's weighted one:
    identical per-document scores read as +0.15 and satisfied the improvement
    rule. Either side without weights is unweighted on both, and said so."""
    from evaluation.gating import promotion_gate
    from orchestration.pipeline_dag import paired_document_scores

    scores = {f"doc_{i:02d}": (0.5, 1.0)[i % 2] for i in range(20)}
    weights = {f"doc_{i:02d}": (1.0, 4.0)[i % 2] for i in range(20)}
    _promoted_and_candidate(client, controller, {
        "gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
        "per_document": {"list_field_recall": scores},
    }, {
        "gate_metrics": dict(PASSING_METRICS), "eval_set": _scored_on(),
        "per_document": {"list_field_recall": scores},
        "per_document_weights": {"list_field_recall": weights},
    })
    warnings: list[str] = []
    ctx = make_context(client, controller, out_version="v1.1")
    paired = paired_document_scores(ctx, warnings)
    same = [scores[d] for d in sorted(scores)]
    assert paired == {"list_field_recall": (same, same)}
    assert any("promoted report(s) hold no per-document weights" in w and "unweighted" in w
               for w in warnings), warnings
    result = promotion_gate(dict(PASSING_METRICS), dict(PASSING_METRICS), per_document=paired)
    assert result.improved_metrics == [] and "improvement" in result.failed_gates

    # Both sides weighted: each document counts as it counts in the report.
    client.write_json(paths.eval_report("v1", scope="policy"), {
        **client.read_json(paths.eval_report("v1", scope="policy")), "per_document_weights": {"list_field_recall": weights},
    })
    assert paired_document_scores(ctx, []) == {"list_field_recall": (
        same, same, [weights[d] for d in sorted(weights)], [weights[d] for d in sorted(weights)])}


def test_the_interim_cap_follows_what_the_candidate_was_scored_on(client, controller):
    """small_set_accepted describes the whole set, other types included: a policy
    candidate scored on 200 policies was capped because the set held 9 Loss Runs.
    Only the real documents of the adapter's own types count, and lines do not
    narrow them - the one adapter is scored on every frozen policy."""
    from evaluation.freeze_eval_set import MIN_FROZEN_DOCS_PER_TYPE, manifest_key
    from orchestration.pipeline_dag import eval_set_size

    ctx = make_context(client, controller)
    large_set = {"small_set_accepted": True, "documents_by_doc_type": {"policy": 200, "lossrun": 9},
                 "real_documents_by_doc_type": {"policy": 200, "lossrun": 9},
                 "real_policies_by_line": {"general_liability": 9, "homeowners": 191}}
    assert eval_set_size(ctx, large_set) == (
        {"policy": 200}, "frozen_manifest.real_documents_by_doc_type")
    # The candidate's own report, when it records them, is what it was scored on.
    client.write_json(paths.eval_report("v1", scope="policy"), {"eval_set": _scored_on(policy=9)})
    assert eval_set_size(ctx, large_set) == ({"policy": 9}, "eval_report")
    # Another tenant's report at the same path says nothing about this one.
    client.write_json(paths.eval_report("v1", scope="policy"),
                      {"eval_set": _scored_on(tenant="other-tenant", policy=400)})
    assert eval_set_size(ctx, large_set)[1] == "frozen_manifest.real_documents_by_doc_type"
    assert eval_set_size(ctx, {}) == ({}, "unknown")
    assert MIN_FROZEN_DOCS_PER_TYPE == 150

    seed_corpus(client)
    client.write_json(manifest_key(), {**large_set, "frozen_from_corpus": "corpus/v1"})
    # Production also needs what users receive, served_field_match, and its
    # per-document scores to bound it (decision 2026-10-26).
    served = {f"policy_{i:04d}": 0.95 for i in range(200)}
    client.write_json(paths.eval_report("v1", scope="policy"), {
        "eval_set": _scored_on(policy=200), "per_document": {"served_field_match": served},
        "per_document_weights": {"served_field_match": dict.fromkeys(served, 1.0)}})
    policy = make_context(client, controller, metrics_provider=lambda _c: {
        **PASSING_METRICS, "served_field_match": 0.95})
    assert run_stages(policy, stages_for("all"), command="all").ok
    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["tier"] == "production"
    assert decision["eval_set"]["scored_documents_by_doc_type"] == {"policy": 200}
    assert decision["eval_set"]["scored_documents_counted_from"] == "eval_report"
    assert decision["eval_set"]["small"] is False
    # The decision names the adapter it judged; there is no scope to name.
    assert decision["adapter"] == "policy-v1" and decision["doc_types"] == ["policy"]
    assert "scope" not in decision


def test_synthetic_twins_do_not_count_toward_the_production_minimum(client, controller):
    """Decision 2026-10-22: the 150 documents per type that decide production
    count real documents only. A policy candidate scored on 160 policies, 60 of them
    twins, was scored on 100 real ones: interim. A manifest frozen before it
    counted real documents cannot say, so its count is unknown: interim, and
    the decision says why."""
    from types import SimpleNamespace

    from evaluation.freeze_eval_set import manifest_key
    from evaluation.run_eval import eval_set_record
    from orchestration.pipeline_dag import eval_set_size

    scored = ([SimpleNamespace(doc_type="policy", synthetic=False)] * 100
              + [SimpleNamespace(doc_type="policy", synthetic=True)] * 60)
    record = eval_set_record(client, None, scored)
    assert record["documents_by_doc_type"] == {"policy": 160}
    assert record["real_documents_by_doc_type"] == {"policy": 100}

    seed_corpus(client)
    client.write_json(paths.eval_report("v1", scope="policy"), {"eval_set": record})
    production = {**PASSING_METRICS, "field_normalized_match": 0.97}
    policy = make_context(client, controller, metrics_provider=lambda _c: dict(production))
    assert run_stages(policy, stages_for("all"), command="all").ok
    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["eval_set"]["scored_documents_by_doc_type"] == {"policy": 100}
    assert decision["tier"] == "interim"

    # The report records no real counts, and the manifest predates them.
    old_manifest = {"frozen_at": "2026-10-01T00:00:00+00:00", "frozen_from_corpus": "v1",
                    "documents_by_doc_type": {"policy": 400}}
    client.write_json(manifest_key(), old_manifest)
    client.write_json(paths.eval_report("v1", scope="policy"),
                      {"eval_set": {**record, "real_documents_by_doc_type": None}})
    warnings: list[str] = []
    assert eval_set_size(policy, old_manifest, warnings) == ({}, "unknown")
    assert any("no real_documents_by_doc_type" in w and "interim" in w for w in warnings), warnings


def test_the_baseline_is_this_tenant_s_promoted_release(client, controller):
    """The newest promoted run of the lineage, whatever its tenant, was once the
    baseline, so one tenant's candidate was gated against another's release. The
    baseline is now read from the tenant's own release index."""
    from orchestration.pipeline_dag import _promoted_eval_report_key, promoted_version

    promote_in_index(client, "v1")
    promote_in_index(client, "v2", "acme", created_at="2026-10-02T00:00:00+00:00")
    client.write_json(paths.eval_report("v1", scope="policy"), {"gate_metrics": dict(PASSING_METRICS)})
    client.write_json(paths.eval_report("v2", scope="policy"), {"gate_metrics": dict(PASSING_METRICS)})

    probe = make_context(client, controller, out_version="v3")
    assert promoted_version(probe) == "v1"
    assert _promoted_eval_report_key(probe) == paths.eval_report("v1", scope="policy")
    assert promoted_version(make_context(client, controller, out_version="v3", tenant_id="acme")) == "v2"


def test_the_promoted_release_s_eval_report_is_the_gate_baseline(client, controller):
    """Release v2 promoted in the tenant's index, with its eval report at the
    policy adapter's path, is what candidate v3 is gated against."""
    from orchestration.pipeline_dag import default_baseline_metrics, promoted_version

    baseline = {**PASSING_METRICS, "field_normalized_match": 0.95}
    promote_in_index(client, "v2")
    client.write_json(paths.eval_report("v2", scope="policy"), {"gate_metrics": baseline})

    candidate = make_context(client, controller, out_version="v3")
    assert promoted_version(candidate) == "v2"
    assert default_baseline_metrics(candidate) == baseline


def test_the_candidate_s_own_release_is_never_its_baseline(client, controller):
    """A resumed package may find its own release already promoted; comparing a
    candidate with itself would pass every regression check by construction."""
    from orchestration.pipeline_dag import promoted_version

    promote_in_index(client, "v2")
    promote_in_index(client, "v3", created_at="2026-10-05T00:00:00+00:00")
    assert promoted_version(make_context(client, controller, out_version="v3")) == "v2"

    only = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    promote_in_index(only, "v3")
    assert promoted_version(make_context(only, controller, out_version="v3")) is None


def test_a_run_manifest_marked_promoted_is_not_a_baseline(client, controller):
    """The baseline was read from run-manifest status, which no real run ever
    set, so every candidate was gated as a first version. Only a promoted
    release in the index counts; with none, the candidate is a first version."""
    from evaluation.gating import apply_to_manifest, promotion_gate
    from orchestration.pipeline_dag import default_baseline_metrics, promoted_version
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import mark_promoted

    seed_corpus(client)
    run_stages(make_context(client, controller), stages_for("finetune"), command="finetune")
    passed = apply_to_manifest(promotion_gate(dict(PASSING_METRICS), None), get("policy-v1", client))
    assert mark_promoted(passed, client, promoted_by="test").status == "promoted"
    client.write_json(paths.eval_report("v1", scope="policy"), {"gate_metrics": dict(PASSING_METRICS)})

    candidate = make_context(client, controller, out_version="v2")
    assert promoted_version(candidate) is None
    assert default_baseline_metrics(candidate) is None


def test_a_legacy_release_id_gives_its_adapter_s_version(client, controller):
    """An index written before releases were named by version holds
    release-YYYY.M.N ids: the baseline is the version of the adapter it served."""
    from orchestration.pipeline_dag import promoted_version

    client.write_json(paths.release_index(), [
        {"release_id": "release-2026.9.1", "status": "promoted", "adapter": "policy-v2",
         "created_at": "2026-09-01T00:00:00+00:00"},
    ])
    assert promoted_version(make_context(client, controller, out_version="v3")) == "v2"


def test_paired_documents_records_only_what_was_bootstrapped(client, controller):
    """A metric paired on one document is compared as a point estimate, and the
    decision recorded it as paired on 1."""
    ctx = make_context(client, controller, baseline_metrics=dict(PASSING_METRICS), per_document={
        "field_normalized_match": ([0.9] * 20, [0.95] * 20),
        "scanned_accuracy": ([0.8], [0.8]),
    })
    seed_corpus(client)
    assert run_stages(ctx, stages_for("all"), command="all").ok
    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["paired_documents"] == {"field_normalized_match": 20}
    basis = {v["name"]: v["basis"] for v in decision["verdicts"]}
    assert basis["scanned_accuracy"] == "point_estimate"


def test_an_unreviewed_frozen_gold_is_refused_before_the_model_loads_and_once(client, controller):
    """The gate loaded the bf16 model, then learned from the frozen set
    that a gold had no review record - and the stage was retried, loading the
    model a second time to learn it again."""
    from evaluation.freeze_eval_set import manifest_key
    from orchestration.pipeline_dag import Stage, stage_evaluation_gate

    root = paths.golden_eval_set_dir()
    client.write_json(manifest_key(), {"frozen_at": "2026-10-01T00:00:00+00:00",
                                       "frozen_from_corpus": "v1", "source_ids": ["policy_0001"]})
    client.write_json(f"{root}/policy_0001/metadata.json", {"doc_type": "policy", "lob": "homeowners"})
    client.write_json(f"{root}/policy_0001/golden.json", {})
    client.write_bytes(f"{root}/policy_0001/page_1.png", b"png")
    loads: list[str] = []
    attempts: list[int] = []

    def gate(stage_ctx):
        attempts.append(1)
        return stage_evaluation_gate(stage_ctx)

    ctx = make_context(client, controller, metrics_provider=None, dry_run=False, max_attempts=2,
                       retry_backoff_seconds=0, serving_model_loader=lambda _c, fmt: loads.append(fmt))
    report = run_stages(ctx, [Stage(10, "evaluation_gate", "finetune", True, gate)], command="finetune")
    assert report.failed_at == "evaluation_gate"
    assert "no review record" in report.results[-1].detail
    assert loads == [], "refused before the model was loaded"
    assert attempts == [1], "an unreviewed gold is not retried"


def test_the_gate_s_own_refusals_are_not_retried():
    from evaluation.freeze_eval_set import FreezeError
    from evaluation.gating import GateError
    from evaluation.golden_eval import GoldenEvalError
    from evaluation.run_eval import EvalSetLeakage
    from orchestration.pipeline_dag import _deterministic_errors

    for error in (GoldenEvalError, EvalSetLeakage, FreezeError, GateError):
        assert issubclass(error, _deterministic_errors())


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
        "all", "--input", "./intake", "--version", "v1",
        "--override-approver", "A. Reviewer", "--override-reason", _REASON,
        "--override-gate", "list_field_recall",
    ], baseline_metrics=dict(PASSING_METRICS), metrics_provider=lambda _c: dict(regressed))
    assert isinstance(ctx.gate_override, GateOverrideRecord)
    assert ctx.gate_override.waived_gates == ["list_field_recall"]

    report = cli.run_command("all", ctx)
    assert report.ok, report.render()
    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["waived_gates"] == ["list_field_recall"]
    assert decision["override"]["approver"] == "A. Reviewer"
    bundle = client.read_json(paths.release_bundle("v1"))
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
    base = ["all", "--version", "v1"]
    with pytest.raises(cli.GateOverrideError, match=match):
        _cli_context(client, controller, base + flags)


def test_an_override_names_real_gates_and_says_when_it_waived_nothing(client, controller):
    """A misspelt gate was accepted, waived nothing, and left the release
    blocked; a waiver of a gate that passed showed only in the log."""
    base = ["all", "--input", "./intake", "--version", "v1",
            "--override-approver", "A. Reviewer", "--override-reason", _REASON]
    with pytest.raises(cli.GateOverrideError, match=r"\['list_recall'\], which is not a gate"):
        _cli_context(client, controller, base + ["--override-gate", "list_recall"])
    for gate in ("improvement", "line_regression_evidence", "line:homeowners", "table_f1"):
        assert _cli_context(client, controller, base + ["--override-gate", gate]).gate_override
    # The cross-type gates went with continued Foundations (decision 2026-10-23).
    for gone in ("cross_type_regression_evidence", "cross_type:lossrun", "line:not_a_line"):
        with pytest.raises(cli.GateOverrideError, match="which is not a gate"):
            _cli_context(client, controller, base + ["--override-gate", gone])

    seed_corpus(client)
    ctx = _cli_context(client, controller, base + ["--override-gate", "list_field_recall"],
                       metrics_provider=lambda _c: dict(PASSING_METRICS))
    assert cli.run_command("all", ctx).ok
    decision = client.read_json(paths.gate_decision("v1", scope="policy"))
    assert decision["waived_gates"] == []
    assert any("['list_field_recall'], which did not fail" in w for w in decision["warnings"])


def test_flags_are_refused_before_the_command_detaches_on_the_pod(monkeypatch, capsys):
    """On the pod, main() detached before build_context checked --fixes-defect and
    --override-*: the operator's command exited 0 and the refusal surfaced only
    inside tmux. Off the pod it was a traceback."""
    import orchestration.detach as detach

    started: list[object] = []
    monkeypatch.setattr(detach, "should_detach", lambda: True)
    monkeypatch.setattr(detach, "detach", lambda command, hint: started.append(command) or "all-test")
    base = ["all", "--input", "./intake", "--version", "v9"]
    for flags, said in ((["--fixes-defect", " "], "names no defect"),
                        (["--override-gate", "list_field_recall"], "name the person"),
                        (["--override-approver", "A. Reviewer", "--override-reason", _REASON,
                          "--override-gate", "list_recall"], "not a gate")):
        with pytest.raises(SystemExit) as exited:
            cli.main(base + flags)
        assert exited.value.code == 2 and said in capsys.readouterr().err
    assert started == [], "nothing was detached"

    for flags in (["--mark-held-out", "--corpus", "v1"], ["--mark-reviewed", "--allow-small"], []):
        with pytest.raises(SystemExit):
            cli.main(["freeze-eval-set", "--tenant", "t", *flags])
    assert started == []


def test_mark_held_out_and_mark_reviewed_both_run(monkeypatch, capsys):
    """Given together, --mark-reviewed was skipped without a word."""
    import evaluation.freeze_eval_set as freeze
    import orchestration.detach as detach

    ran: list[str] = []
    monkeypatch.setattr(detach, "should_detach", lambda: False)
    monkeypatch.setattr(cli, "BlobClient", lambda: None)
    monkeypatch.setattr(freeze, "mark_held_out", lambda _c, tenant_id: ran.append("held_out") or ["p1"])
    monkeypatch.setattr(freeze, "mark_reviewed", lambda _c, tenant_id: ran.append("reviewed") or {})
    assert cli.main(["freeze-eval-set", "--tenant", "t", "--mark-held-out", "--mark-reviewed"]) == 0
    assert ran == ["held_out", "reviewed"]
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"held_out_source_ids": ["p1"], "still_without_review_record": {}}


def test_fixes_defect_is_checked_before_any_work(client, controller):
    """`--fixes-defect " "` lifted the improvement rule. A blank or placeholder
    value is refused; a defect's own sentence is taken as written."""
    for blank in (" ", "-", "n/a"):
        with pytest.raises(cli.FixesDefectError, match="names no defect"):
            cli.fixes_defect_for(blank)
    assert cli.fixes_defect_for("TICKET-12") == "TICKET-12"
    assert cli.fixes_defect_for(None) is None
    # A sentence holding "=" is a sentence: there is no scope=defect pair any more.
    assert cli.fixes_defect_for("FID-9: rate=0 on endorsements").startswith("FID-9")

    argv = ["all", "--version", "v1", "--fixes-defect", "TICKET-12"]
    assert _cli_context(client, controller, argv).fixes_defect == "TICKET-12"
    assert _cli_context(client, controller, ["all", "--version", "v1"]).fixes_defect is None


def test_a_gold_with_no_review_record_never_trains(client, controller):
    """Decision 2026-10-22: every gold needs a review record. The corpus build
    refuses the run rather than train on one without it."""
    from orchestration.pipeline_dag import load_labeled_documents

    seeded = seed_corpus(client)
    source_id = seeded_policies(seeded)[0]
    doc_type = source_id.rsplit("_", 1)[0]
    label = client.read_json(paths.golden_label(doc_type, source_id))
    label["fideon:provenance"] = {"provenance": "Seed gold drafted by Claude Code, not reviewed"}
    client.write_json(paths.golden_label(doc_type, source_id), label)
    metadata = client.read_json(paths.label_metadata(doc_type, source_id))
    unreviewed = {k: v for k, v in metadata.items() if k not in ("reviewer_id", "review_date")}
    client.write_json(paths.label_metadata(doc_type, source_id), unreviewed)

    ctx = make_context(client, controller)
    ctx.doc_types = [doc_type]
    with pytest.raises(PipelineError, match="no review record"):
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
    assert "no-op" in by_name["training"].detail


def test_the_gate_is_never_skipped_as_already_complete():
    """An 'already gated' shortcut would let a re-run inherit a pass it did not
    earn. Stage 6 has no completion check, on purpose."""
    gate = next(s for s in STAGES if s.name == "evaluation_gate")
    assert gate.is_complete is None


def test_resuming_at_checkpoint_selection_does_not_retrain(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    resumed = make_context(client, controller)
    report = run_stages(resumed, stages_from("checkpoint_eval", "finetune"), command="finetune")
    assert [r.name for r in report.results] == ["checkpoint_eval"]
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
    crashed = get("policy-v1", client)
    crashed.status = "failed"
    write_manifest(crashed, client)

    package_ctx = make_context(client, controller)
    run_stages(package_ctx, stages_for("package"), command="package")

    assert get("policy-v1", client).artifacts.status == "staged",         "a failed run was published, advertising a Blob path serving would find empty"


def test_package_pushes_the_adapter_and_publishes(client, controller):
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

    # The adapter alone: it is the release, served as a LoRA on the base, so
    # nothing merged or quantized is pushed (decision 2026-10-23).
    pushed = package_ctx.results["package"].data["pushed"]
    assert pushed == {"adapter:policy": paths.scoped_adapter_dir("policy", "v1")}

    from registry_utils.query_registry import get

    foundation = get("policy-v1", client)
    assert foundation.artifacts.status == "published"
    assert foundation.artifacts.staging_path is None
    assert foundation.artifacts.adapter_weights == paths.scoped_adapter_dir("policy", "v1")
    assert foundation.artifacts.merged_model is None
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


def test_a_default_build_publishes_the_adapter_alone(client, controller):
    """The adapter's path IS what was built, so the run owns it, and nothing
    else: no merged or quantized model exists to publish."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")
    run_stages(ctx, stages_for("package"), command="package")

    from registry_utils.query_registry import get

    foundation = get("policy-v1", client)
    assert foundation.artifacts.adapter_weights == paths.scoped_adapter_dir("policy", "v1")
    assert foundation.artifacts.merged_model is None
    assert foundation.artifacts.quantized_formats == []
    assert foundation.artifacts.quantized_model is None
    # A dry run's placeholder, where the adapter's own config would be.
    assert client.exists(f"{paths.scoped_adapter_dir('policy', 'v1')}/adapter_config.json")


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
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("package"), command="package")

    # The FIRST stage of package, not the last: the remediation is about a
    # reclaimed volume, and discovering it after calibrate and the gate have run
    # wastes the whole command to say something knowable up front.
    assert report.failed_at == "calibrate"
    detail = report.results[-1].detail
    assert "not on the staging volume" in detail
    assert "--from-stage training" in detail and "--push-adapters" in detail


def test_push_adapters_in_a_dry_run_pushes_no_placeholder(client, controller):
    """Recommended when commands 1 and 2 may be separated by more than a day:
    the adapter is tens of MB, and pushed early a reclaimed volume loses nothing
    (IMPL-13 §3)."""
    seed_corpus(client)
    ctx = make_context(client, controller, push_adapters=True)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    # A dry run trained nothing, so there is no adapter to push — and the
    # placeholder note that used to stand in for one is gone. The real upload
    # is covered by test_push_adapters_uploads_the_adapter_itself.
    assert not client.exists(f"{paths.scoped_adapter_dir('policy', 'v1')}/adapter_placeholder.json")


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
    with pytest.raises(RuntimeError, match="boom"), controller.pod(controller.spec_for("calibrate")):
        raise RuntimeError("boom")
    assert controller.active_pods == []


def test_run_job_reports_a_failure_instead_of_leaking_a_pod(controller):
    result = controller.run_job(controller.spec_for("evaluation_gate"), lambda _h: 1 / 0)
    assert not result.ok and "division" in (result.error or "")
    assert controller.active_pods == []


def test_every_gpu_stage_runs_on_the_one_h200_pod(controller):
    """The deployment runs every GPU stage on one H200 SXM pod, in process: OCR,
    training, selection, calibration and the gate share the card, and a
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


def test_the_card_a_stage_runs_on_is_set_per_stage_only(controller, monkeypatch):
    """There is one adapter, so there is no per-scope card to choose: a stage's
    card comes from gpu_class_by_stage, else the fallback."""
    import inspect

    from orchestration import settings

    monkeypatch.setattr(
        settings, "pipeline_config",
        lambda: {"gpu_class_by_stage": {"training": "A100-40G"}},
    )
    assert settings.gpu_class_for("training") == "A100-40G"
    assert settings.gpu_class_for("calibrate", "L40S") == "L40S"
    assert "scope" not in inspect.signature(settings.gpu_class_for).parameters
    monkeypatch.undo()
    assert "gpu_class_by_scope" not in settings.pipeline_config()


def test_finetune_uses_one_pod_for_the_whole_training_fan_out(client, controller):
    """Three launches would mean three cold starts and two Blob round trips of
    the same corpus."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("finetune"), command="finetune")

    assert controller.backend._counter == 1  # type: ignore[attr-defined]
    assert controller.active_pods == []


def test_a_pod_will_not_clone_a_moving_branch():
    spec = PodSpec(name="p", stage="calibrate", gpu_class="L40S", volume_id="vol", git_commit=None)
    with pytest.raises(PodLaunchError, match="no resolvable git commit"):
        spec.clone_command("git@example.com:repo.git")


def test_an_unresolved_commit_is_not_treated_as_pinned():
    """`capture_git_commit` returns 'unknown' off a git tree; a pod cannot clone it."""
    spec = PodSpec(name="p", stage="calibrate", gpu_class="L40S", volume_id="vol", git_commit="unknown")
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


def test_a_fresh_finetune_names_no_version_and_package_must():
    """A fresh cycle takes the next version itself (``cycle_version``); package
    publishes one cycle, so it names it."""
    assert cli.build_parser().parse_args(["finetune", "--input", "./intake"]).version is None
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["package"])
    for gone in ("--out-version", "--release-id", "--scope", "--doc-types"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["all", "--input", "./intake", gone, "x"])


def test_all_accepts_the_union_of_both_flag_sets():
    args = cli.build_parser().parse_args([
        "all", "--input", "./intake", "--version", "v2",
        "--fresh", "--keep-staging",
    ])
    assert args.version == "v2" and args.fresh and args.keep_staging


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
        "finetune", "--input", "./intake", "--version", "v3",
        "--push-adapters", "--dry-run", "--min-labels-per-type", "5",
    ])
    ctx = cli.build_context(args, client=client, controller=controller,
                            raw_client=ingestion_client(client))
    assert ctx.out_version == "v3" and ctx.push_adapters
    assert ctx.dry_run and ctx.min_labels_per_type == 5
    assert ctx.corpus == "v3"  # corpus defaults to the output version
    # One number names the run, the adapter and the release.
    assert ctx.release_id == "v3"
    assert ctx.doc_types == ["policy"]


def test_the_release_id_is_the_cycle_s_version(client, controller):
    """A release is named by its version: there is no separate id to pass, and
    so none to get wrong between a failed run and its resume."""
    ctx = make_context(client, controller, out_version="v4")
    assert ctx.release_id == "v4"
    assert paths.is_valid_release_id(ctx.release_id)
    # The release-YYYY.M.N form an older index holds still reads.
    assert paths.is_valid_release_id("release-2026.9.1")
    assert not paths.is_valid_release_id("policy-v4")
    with pytest.raises(TypeError):
        make_context(client, controller, release_id="v9")


def test_the_next_version_is_one_more_than_any_used():
    assert paths.next_version([]) == "v1"
    assert paths.next_version(["v1", "v3", "v2.1"]) == "v4"
    assert paths.next_version(["v2", "v10", "release-2026.9.1", "policy-v40", "latest"]) == "v11"
    assert paths.version_number("v3") == 3 and paths.version_number("v2.1") == 2
    assert paths.version_number("release-2026.9.1") is None


def test_run_command_dispatches_all_through_both_commands(client, controller):
    seed_corpus(client)
    args = cli.build_parser().parse_args([
        "all", "--input", "./intake", "--version", "v1",
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
    """Run a cycle and promote its release, so the next one has something to bump from.

    One run per cycle: the policy adapter (``common.adapter``). v1 promoted a
    Foundation plus one adapter per document type, a topology vLLM cannot serve.
    The cascade reads the promoted release from the tenant's release index.
    """
    ctx = make_context(client, controller, out_version=version)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()
    promote_in_index(client, version)


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
            foundation_version=f"policy-{foundation_version}",
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

    ctx = make_context(client, controller, out_version="v1.1", fresh=True)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()


def test_a_major_bump_blocks_until_dependent_adapters_are_revalidated(client, controller):
    """Every per-type adapter was trained on the previous Foundation's weights,
    so promoting a new one without re-validating them ships three models that
    were never evaluated against the base they now sit on."""
    seed_corpus(client)
    _promote_foundation(client, controller, "v1")
    _graduate_adapter(client, "policy", "v1")

    ctx = make_context(client, controller, out_version="v2", fresh=True)
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
    assert dependents, "the graduated adapters should depend on policy-v1"

    ctx = make_context(client, controller, out_version="v2", fresh=True,
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

    ctx = make_context(client, controller, out_version="v2", fresh=True, revalidation_evidence=partial)
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
    name = f"run-finetune-{paths._tenant(ctx.tenant_id)}-v1.json"
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


def test_two_tenants_under_one_version_keep_their_own_run_reports(client, controller):
    from orchestration.pipeline_dag import RunReport, progress_report_name

    report = RunReport(command="finetune", version="v1")
    cgl = make_context(client, controller, tenant_id="cgl")
    personal = make_context(client, controller, tenant_id="personal")
    assert progress_report_name(report, cgl) != progress_report_name(report, personal)
    assert progress_report_name(report, cgl) == "run-finetune-cgl-v1.json"


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
    assert settings.defaults()["min_labels_per_type"] == 25
    # Nothing is merged or quantized, so there is no format or merge dtype to
    # configure (decision 2026-10-23).
    assert "formats" not in settings.defaults() and "dtype" not in settings.defaults()


def test_cli_defaults_come_from_the_config_file():
    from orchestration import settings

    args = cli.build_parser().parse_args(["finetune", "--input", "./intake"])
    assert args.min_labels_per_type == settings.defaults()["min_labels_per_type"]


def test_the_gpu_flag_actually_reaches_the_training_pod(client, controller):
    """A flag that parses and changes nothing is worse than no flag."""
    seed_corpus(client)
    args = cli.build_parser().parse_args([
        "finetune", "--version", "v1", "--gpu", "H100-80G", "--skip-ingest",
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

def test_the_selected_checkpoint_is_what_gets_served(client, controller):
    """The selection is worth nothing if calibration, the gate and package
    ignore it and take the staged adapter directory anyway."""
    from orchestration.pipeline_dag import selected_checkpoint

    seed_corpus(client)
    chosen = "/runpod-volume/staging/adapters/scope/policy/v1/checkpoint-300"
    ctx = make_context(
        client, controller,
        checkpoints=[
            "/runpod-volume/staging/adapters/scope/policy/v1/checkpoint-100",
            "/runpod-volume/staging/adapters/scope/policy/v1/checkpoint-200",
            chosen,
        ],
        checkpoint_scorer=lambda p: {
            "field_normalized_match": 0.95 if p.endswith("300") else 0.80
        },

    )
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert ctx.results["checkpoint_eval"].data["selected"] == chosen
    assert selected_checkpoint(ctx) == chosen


def test_a_run_with_no_checkpoints_skips_selection_rather_than_guessing(client, controller):
    """A dry run never launched ms-swift, so there is nothing to choose between.
    Serving the staged adapter directory is the honest fallback."""
    from orchestration.pipeline_dag import selected_checkpoint

    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("finetune"), command="finetune")

    assert report.ok, report.render()
    assert ctx.results["checkpoint_eval"].status == "skipped"
    assert selected_checkpoint(ctx) is None


def test_the_selection_is_recorded_where_a_later_review_can_read_it(client, controller):
    """The adapter's weights do not say which checkpoint they came from, and
    once the staging volume is reclaimed nothing else does either."""
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        checkpoints=["/runpod-volume/staging/adapters/scope/policy/v1/checkpoint-100"],
        checkpoint_scorer=lambda _p: {"field_normalized_match": 0.9},

    )
    run_stages(ctx, stages_for("finetune"), command="finetune")

    record = client.read_json(paths.checkpoint_selection("v1", scope="policy"))
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
        calibration_samples={"bf16": _calibration_samples()},
    )
    report = run_stages(ctx, stages_for("all"), command="all")

    assert report.ok, report.render()
    assert "bf16" in ctx.calibrators and "bf16" in ctx.thresholds

    stored = client.read_json(paths.release_calibrators("v1", "bf16"))
    assert stored["calibrators"]["serving_format"] == "bf16"
    assert stored["thresholds"]["fitted_on"].startswith("validation threshold half")


def test_no_calibrator_is_shared_across_serving_formats(client, controller):
    """A "*" key used to calibrate every format from one sample set, bypassing
    the per-format rule the stage exists to enforce."""
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        calibration_samples={"*": _calibration_samples()},
    )
    run_stages(ctx, stages_for("all"), command="all")

    assert "bf16" not in ctx.calibrators
    assert not client.exists(paths.release_calibrators("v1", "bf16"))


def test_a_release_with_no_labelled_features_ships_uncalibrated_and_says_so(client, controller):
    """Honest rather than silent: without labelled validation features there is
    nothing to fit, every field routes to review, and an operator should know
    that happened rather than discover it in the review queue."""
    seed_corpus(client)
    ctx = make_context(client, controller)
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


# --------------------------------------------------------------------------
# The release bundle (arch v2.1 §12.3)
# --------------------------------------------------------------------------


def test_package_writes_the_release_bundle_it_pins(client, controller):
    """stage_push was registered as `package` and never built a ReleaseBundle,
    so nothing pinned the prompt, schema, calibrators and OCR version that the
    served weights depend on."""
    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()

    bundle = client.read_json(paths.release_bundle("v1"))
    assert bundle["adapter"] == "policy-v1"
    # The release IS its adapter, served as a LoRA on the bf16 base: nothing is
    # merged, so bf16's artifact is the adapter's directory (decision 2026-10-23).
    assert bundle["served_as"] == "lora_on_base" and bundle["merged_model"] == ""
    assert bundle["serving_formats"] == {"bf16": paths.scoped_adapter_dir("policy", "v1")}
    # What it may serve: policies. Which lines is trained_lines' to say.
    assert bundle["doc_types"] == ["policy"]
    assert bundle["prompt_hash"] and bundle["vllm_config_hash"] and bundle["lockfile_hash"]
    assert bundle["ocr_pin"]["mineru_version"] == SERVED_MINERU["version"]
    assert bundle["ocr_pin"]["mineru_tier"] == SERVED_MINERU["tier"]
    assert "bf16" in bundle["gate_reports"]

    index = client.read_json(paths.release_index())
    assert [row["release_id"] for row in index] == ["v1"]


def test_the_release_serves_the_lines_the_corpus_trained(client, controller):
    """The bundle records the lines the adapter trained on, as canonical schema
    names; serving sends a policy of any other line to the base model. With no
    train view on the run manifest, the corpus's own line counts say."""
    from orchestration.pipeline_dag import trained_lines

    ctx = make_context(client, controller)
    corpus = {"policy_line_counts": {"wc": 1, "auto": 2, "homeowners": 0, "not_a_line": 4}}
    assert trained_lines(ctx, corpus) == ["auto", "wc"]
    assert trained_lines(ctx, {}) == []


def test_the_lines_of_the_run_s_own_train_view_are_canonical(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    assert run_stages(ctx, stages_for("all"), command="all").ok
    # The view keys lines by their schema names (labels say commercial_auto,
    # workers_comp), as the release and the data mix name them.
    train = ctx.manifests["policy"].data_stats.examples_by_line["train"]
    assert set(train) == {"auto", "wc"}
    assert client.read_json(paths.release_bundle("v1"))["lines"] == ["auto", "wc"]


def test_an_uncalibrated_release_is_gated_not_promoted(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, stages_for("all"), command="all")

    bundle = client.read_json(paths.release_bundle("v1"))
    assert bundle["status"] == "gated"
    reasons = ctx.results["package"].data["not_promoted_because"]
    assert any("no calibrator" in r for r in reasons)


def _package_for_real(ctx) -> None:
    """The release step as a real `package` ends: the bundle written, gated -
    package never promotes (decision 2026-10-26, #44)."""
    from orchestration.pipeline_dag import build_release_bundle, write_release_bundle

    ctx.dry_run = False
    bundle, _reasons = build_release_bundle(ctx)
    write_release_bundle(ctx, bundle)


def test_a_calibrated_gated_bf16_release_is_left_gated_for_the_shadow_check(client, controller):
    """Decision 2026-10-26 (#44): package auto-promoted a calibrated release
    whose gate passed, interim tier included. It writes it ``gated`` now, a
    dry run and a real one alike; a person promotes it after the shadow check
    (tests/test_decision_promotion.py)."""
    seed_corpus(client)
    ctx = make_context(
        client, controller,
        calibration_samples={"bf16": _calibration_samples()},
    )
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()
    assert client.read_json(paths.release_bundle("v1"))["status"] == "gated"
    _package_for_real(ctx)

    bundle = client.read_json(paths.release_bundle("v1"))
    assert bundle["status"] == "gated", ctx.results["package"].data["not_promoted_because"]
    assert bundle["calibrators"]["bf16"] == paths.release_calibrators("v1", "bf16")
    assert bundle.get("promotions", []) == []
    # The index row says so too: serving reads no promoted release.
    [row] = client.read_json(paths.release_index())
    assert row["release_id"] == "v1" and row["status"] == "gated"


def test_packaging_a_release_never_archives_the_promoted_one(client, controller):
    """Packaging a calibrated v2 used to promote it and archive v1. Now v1
    stays the promoted release - in the index, in its bundle, and as the next
    gate's baseline - and v2 waits, gated, for its shadow check and a person's
    promotion. Each from the base: a v2 continuing v1 needs v1's per-line
    scores, which is tests/test_incremental_training.py's subject."""
    from orchestration.pipeline_dag import promoted_version
    from registry_utils.releases import promote_release

    seed_corpus(client)
    for version in ("v1", "v2"):
        ctx = make_context(client, controller, out_version=version, fresh=True,
                           calibration_samples={"bf16": _calibration_samples()})
        report = run_stages(ctx, stages_for("all"), command="all")
        assert report.ok, report.render()
        _package_for_real(ctx)
        if version == "v1":
            promote_release(client, "v1", promoted_by="A. Reviewer")   # as `promote` would

    status = {row["release_id"]: row["status"] for row in client.read_json(paths.release_index())}
    assert status == {"v1": "promoted", "v2": "gated"}
    assert client.read_json(paths.release_bundle("v1"))["status"] == "promoted"
    assert client.read_json(paths.release_bundle("v2"))["status"] == "gated"
    assert promoted_version(make_context(client, controller, out_version="v3")) == "v1"


# --------------------------------------------------------------------------
# Versions: each cycle its own run, model and release (decision 2026-10-22)
# --------------------------------------------------------------------------


def test_packaging_one_version_publishes_only_its_own_run(client, controller):
    """stage_push matched runs with run_id.endswith(version), so packaging one
    run published every run at that tag, against THIS run's paths. Two cycles
    trained on one volume: packaging v2 publishes policy-v2 and nothing else."""
    from registry_utils.query_registry import get

    seed_corpus(client)
    for version in ("v1", "v2"):
        run_stages(make_context(client, controller, out_version=version),
                   stages_for("finetune"), command="finetune")
    v2 = make_context(client, controller, out_version="v2")
    assert run_stages(v2, stages_for("package"), command="package").ok

    assert v2.results["package"].data["published"] == ["policy-v2"]
    pushed = v2.results["package"].data["pushed"]
    assert set(pushed) == {"adapter:policy"}
    assert "/scope/policy/v2" in pushed["adapter:policy"]
    assert get("policy-v1", client).artifacts.status == "staged"


def test_packaging_one_version_leaves_another_version_s_staged_adapter_alone(client, controller):
    """The clear wiped the whole staging root, so packaging one run deleted
    another's staged weights - the only copy until its own package run."""
    seed_corpus(client)

    first = make_context(client, controller, out_version="v1")
    run_stages(first, stages_for("finetune"), command="finetune")
    staged_v1 = paths.scoped_staging_adapter_dir("policy", "v1")
    assert first.volume.exists(staged_v1)

    # The same controller, so the same staging volume - which is exactly the
    # situation the old blanket clear destroyed.
    second = make_context(client, controller, out_version="v2")
    report = run_stages(second, stages_for("all"), command="all")

    assert report.ok, report.render()
    assert first.volume.exists(staged_v1), "packaging v2 cleared v1's adapter"
    assert not first.volume.exists(paths.scoped_staging_adapter_dir("policy", "v2"))


def test_the_gate_reads_the_policy_adapter_s_eval_report(client, controller):
    """The gate reads the adapter's own report, where stage_push records it on
    the manifest - not the unscoped key the unified extractor used."""
    from orchestration.pipeline_dag import eval_report_metrics

    seed_corpus(client)
    freeze_as_scored(client)
    ctx = make_context(client, controller)

    client.write_json(paths.eval_report("v1"), {"gate_metrics": {"field_exact_match": 0.10},
                                                "eval_set": _scored_on()})
    client.write_json(
        paths.eval_report("v1", scope="policy"), {"gate_metrics": {"field_exact_match": 0.99},
                                                  "eval_set": _scored_on()}
    )

    assert eval_report_metrics(ctx)["field_exact_match"] == 0.99


# --------------------------------------------------------------------------
# Phase 1: deterministic failures, resumed packages
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


def test_a_resumed_package_reads_the_checkpoint_already_selected(client, controller):
    """`--from-stage calibrate` (or `package`) did not run checkpoint_eval, but an
    earlier process did and wrote its choice down: calibration, the gate and the
    push read it there."""
    from orchestration.pipeline_dag import selected_checkpoint

    ctx = make_context(client, controller)
    chosen = "/runpod-volume/staging/adapters/scope/policy/v1/v0-1/checkpoint-150"
    client.write_json(paths.checkpoint_selection("v1", scope="policy"), {"selected": chosen})
    assert selected_checkpoint(ctx) == chosen


def test_package_refuses_a_version_whose_gate_blocked(client, controller):
    """`package` is its own command; it read nothing the gate wrote, so running it
    after a blocked gate published the blocked version anyway."""
    from orchestration.pipeline_dag import stage_push

    seed_corpus(client)
    ctx = make_context(client, controller)
    report = run_stages(ctx, stages_for("all"), command="all")
    assert report.ok, report.render()

    key = paths.gate_decision("v1", scope="policy")
    client.write_json(key, {**client.read_json(key), "passed": False, "failed_gates": ["x"]})
    with pytest.raises(PipelineError, match="did not pass its promotion gate"):
        stage_push(ctx)


def test_a_resumed_package_finds_the_calibrators_in_blob(client, controller):
    """Resumed at package, ctx.calibrators is empty; the calibrators the earlier
    process wrote are in Blob and the bundle must still point at them."""
    from orchestration.pipeline_dag import build_release_bundle

    seed_corpus(client)
    first = make_context(
        client, controller,
        calibration_samples={"bf16": _calibration_samples()},
    )
    assert run_stages(first, stages_for("all"), command="all").ok

    resumed = make_context(client, controller)
    bundle, reasons = build_release_bundle(resumed)
    assert "bf16" in bundle.calibrators
    assert not any("no calibrator" in r for r in reasons)


def test_a_policy_corpus_built_with_another_window_plan_neither_trains_nor_packages(client, controller):
    """The prompt hash is computed at packaging, from the packaging code, so it
    cannot tell that the adapter was trained on windows this code no longer plans."""
    from orchestration.pipeline_dag import build_release_bundle, stage_training

    seed_corpus(client)
    ctx = make_context(client, controller)
    assert run_stages(ctx, stages_for("all"), command="all").ok
    key = paths.corpus_manifest(ctx.corpus, ctx.tenant_id)
    manifest = client.read_json(key)
    assert "policy" in manifest["doc_types"]

    client.write_json(key, {**manifest, "window_plan_version": "1"})
    with pytest.raises(PipelineError, match="Rebuild the corpus and retrain"):
        stage_training(ctx)
    with pytest.raises(PipelineError, match="release v1 rests on a corpus built with window plan 1"):
        build_release_bundle(ctx)


def test_a_corpus_built_before_the_review_rule_neither_trains_nor_packages(client, controller):
    """Reused with --corpus-version, a corpus skips the dataset build and its
    check of every gold's review record, so one built before the rule would
    train its unreviewed golds."""
    from data_pipeline.corpus_manifest import REVIEW_RULE_VERSION
    from data_pipeline.dataset_builder.policy_windows import window_plan_id
    from orchestration.pipeline_dag import build_release_bundle, stage_training

    seed_corpus(client)
    ctx = make_context(client, controller)
    assert run_stages(ctx, stages_for("all"), command="all").ok
    bundle, _reasons = build_release_bundle(ctx)
    assert bundle.window_plan_version == window_plan_id()             # serving checks it
    key = paths.corpus_manifest(ctx.corpus, ctx.tenant_id)
    manifest = client.read_json(key)
    assert manifest["review_rule_version"] == REVIEW_RULE_VERSION

    client.write_json(key, {k: v for k, v in manifest.items() if k != "review_rule_version"})
    with pytest.raises(PipelineError, match="review rule unrecorded"):
        stage_training(ctx)
    with pytest.raises(PipelineError, match="review rule unrecorded"):
        build_release_bundle(ctx)


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

    first, *others = seeded_policies(seed_corpus(client))
    _set_ocr_meta(client, first, render_only=True)
    loaded = {d.source_id for d in load_labeled_documents(make_context(client, controller))}
    assert first not in loaded
    assert set(others) <= loaded and others


def test_a_document_with_no_recorded_pages_is_skipped_not_assumed_one(client, controller):
    from orchestration.pipeline_dag import load_labeled_documents

    first, *others = seeded_policies(seed_corpus(client))
    _set_ocr_meta(client, first, page_count=0)
    loaded = {d.source_id for d in load_labeled_documents(make_context(client, controller))}
    assert first not in loaded
    assert set(others) <= loaded and others


def test_a_policy_cycle_loads_no_other_document_type(client, controller):
    """The corpus is built over the adapter's types: policies. The fixtures'
    ACORD form and Loss Run are labelled and OCR'd, and never loaded."""
    from orchestration.pipeline_dag import load_labeled_documents

    seeded = seed_corpus(client)
    ctx = make_context(client, controller)
    assert ctx.doc_types == ["policy"]
    assert {d.source_id for d in load_labeled_documents(ctx)} == set(seeded_policies(seeded))


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

    # Four policies: the fixtures' two and two copies.
    seed_corpus(client)
    seed_policy_copy(client, "policy_0003", like="policy_0001")
    seed_policy_copy(client, "policy_0004", like="policy_0002")
    real = build_jsonl.build_corpus

    def one_document_set_aside(*args, **kwargs):
        # Every split filled from the 4 policies, and the 4th set aside.
        built = real(*args, **kwargs)
        rows = built.all_rows
        ids = sorted({r["source_id"] for r in rows})
        assert len(ids) == 4
        of = {i: [r for r in rows if r["source_id"] == i] for i in ids}
        built.rows_by_split = {"train": of[ids[0]], "val": of[ids[1]], "test": of[ids[2]]}
        return built

    monkeypatch.setattr(pipeline_dag, "build_corpus", one_document_set_aside, raising=False)
    monkeypatch.setattr(build_jsonl, "build_corpus", one_document_set_aside)
    # 1 of 4 documents is 25%: lowered so one document crosses it.
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
