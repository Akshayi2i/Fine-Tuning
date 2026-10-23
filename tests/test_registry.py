"""IMPL-02 — Blob layout, access rules, and the run registry.

Runs entirely against :class:`InMemoryBackend`: no Azure account, no network, no
GPU. The in-memory backend enforces the same write-once and container-isolation
rules as the real one, so passing here means passing the actual contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import (
    AccessDeniedError,
    BlobClient,
    BlobError,
    ImmutableBlobError,
    InMemoryBackend,
)
from registry_utils import query_registry as Q
from registry_utils import write_run_manifest as W
from registry_utils.models import (
    Artifacts,
    DataStats,
    Dependencies,
    Promotion,
    RunManifest,
    TrainingConfig,
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def ingestion_client(client: BlobClient) -> BlobClient:
    return BlobClient(
        backend=client._backend, container="main", raw_container="raw", context="ingestion"
    )


def _deps(**over) -> Dependencies:
    base = dict(
        base_model="qwen3-vl-8b-instruct@a1b2c3d",
        corpus_version="corpus/v3",
        code_git_commit="deadbee",
        mineru_version="1.4.2",
        schema_version="1.0.0",
        prompt_template_version="1.0.0",
    )
    base.update(over)
    return Dependencies(**base)


def _tc(**over) -> TrainingConfig:
    base = dict(
        technique="LoRA", base_quantization="bf16_frozen_base", optimizer="adamw_torch",
        lora_rank=16, lora_alpha=32, learning_rate=7e-5, epochs=4,
        gradient_accumulation_steps=16, effective_batch_size=16,
        target_modules=["q_proj"], resolution_cap_px=1792, max_seq_len=8192, seed=42,
    )
    base.update(over)
    return TrainingConfig(**base)


_DS = DataStats(train_examples=100, val_examples=20, test_examples=20)


def _adapter(run_id: str, doc_type: str, foundation: str) -> RunManifest:
    return RunManifest(
        run_id=run_id, run_type="per_type_adapter", doc_type=doc_type,
        dependencies=_deps(foundation_version=foundation),
        training_config=_tc(), data_stats=_DS,
    )


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def test_tenant_prefix_applied_only_to_document_layers():
    assert paths.is_tenant_scoped(paths.corpus_split("v2", "policy", "train"))
    assert paths.is_tenant_scoped(paths.raw_pdf("acord", "acord_0001"))
    # Shared artifacts carry no tenant data and must never be prefixed.
    assert not paths.is_tenant_scoped(paths.adapter_dir("foundation", "v2"))
    assert not paths.is_tenant_scoped(paths.registry_index())
    assert not paths.TENANT_SCOPED & paths.SHARED


def test_each_tenant_has_its_own_golden_eval_set():
    """Copied from one tenant's test split, so it is that tenant's data; and
    source ids are numbered per tenant, so one shared set would gate every
    tenant on whichever froze first."""
    assert "golden-eval-set" in paths.TENANT_SCOPED and "golden-eval-set" not in paths.SHARED
    assert paths.golden_eval_set_dir("broker_a") == "golden-eval-set/broker_a"
    assert paths.golden_eval_set_dir() == f"golden-eval-set/{paths.default_tenant()}"
    assert paths.tenant_of(f"{paths.golden_eval_set_dir('broker_a')}/policy_0001/golden.json") == "broker_a"
    assert paths.golden_eval_set_dir("broker_a") != paths.golden_eval_set_dir("broker_b")


def test_tenant_defaults_but_is_still_extractable():
    """Single-tenant build: reserved prefix, not a required argument."""
    assert paths.tenant_of(paths.corpus_split("v2", "policy", "train", "broker_a")) == "broker_a"
    assert paths.tenant_of(paths.adapter_dir("foundation", "v2")) is None


def test_staging_paths_are_absolute():
    """They are a real mount point, not blob keys. A relative path would write to
    the working directory and silently miss the volume."""
    assert paths.staging_adapter_dir("foundation", "v2").startswith("/")
    assert paths.staging_merged_model_dir("v2").startswith("/")


def test_staging_mirrors_the_blob_layout():
    """So ``package`` copies rather than translates."""
    assert paths.staging_adapter_dir("foundation", "v2").endswith(paths.adapter_dir("foundation", "v2"))


def test_invalid_inputs_are_rejected_early():
    with pytest.raises(paths.PathError, match="invalid version"):
        paths.adapter_dir("foundation", "2")           # missing the v
    with pytest.raises(paths.PathError, match="unknown doc_type"):
        paths.corpus_split("v1", "invoice", "train")
    with pytest.raises(paths.PathError, match="unknown quantization format"):
        paths.quantized_model_dir("v1", "q3_k_s")
    with pytest.raises(paths.PathError, match="1-based"):
        paths.processed_page("policy", "policy_0001", 0, "png")
    with pytest.raises(paths.PathError, match="not per-doc_type"):
        paths.adapter_dir("foundation", "v1", "policy")


# --------------------------------------------------------------------------
# Access rules
# --------------------------------------------------------------------------

def test_raw_documents_unreachable_from_training_context(client, ingestion_client):
    """raw-documents/ is the only unredacted-PII layer (arch §18a)."""
    key = paths.raw_pdf("acord", "acord_0001")
    ingestion_client.write_bytes(key, b"%PDF fake")
    with pytest.raises(AccessDeniedError, match="may not access"):
        client.read_bytes(key)


def test_raw_documents_are_write_once(ingestion_client):
    """A corrected document becomes a NEW source_id, so historical runs stay
    reproducible against the bytes they actually trained on."""
    key = paths.raw_pdf("acord", "acord_0001")
    ingestion_client.write_bytes(key, b"%PDF original")
    with pytest.raises(ImmutableBlobError, match="write-once"):
        ingestion_client.write_bytes(key, b"%PDF corrected")


def test_raw_documents_cannot_be_casually_deleted(ingestion_client):
    key = paths.raw_pdf("acord", "acord_0001")
    ingestion_client.write_bytes(key, b"%PDF original")
    with pytest.raises(ImmutableBlobError, match="compliance action"):
        ingestion_client.delete(key)


def test_raw_and_main_containers_are_actually_separate(client, ingestion_client):
    key = paths.raw_pdf("acord", "acord_0001")
    ingestion_client.write_bytes(key, b"x")
    backend = client._backend
    assert backend.exists("raw", key)
    assert not backend.exists("main", key)


def test_corpus_artifacts_round_trip(client):
    key = paths.corpus_manifest("v1")
    client.write_json(key, {"mineru_version": "1.4.2", "schema_version": "1.0.0"})
    assert client.read_json(key)["mineru_version"] == "1.4.2"


# --------------------------------------------------------------------------
# Manifest validation
# --------------------------------------------------------------------------

def test_manifest_requires_a_pinned_base_revision():
    with pytest.raises(ValueError, match="pin a revision"):
        _deps(base_model="qwen3-vl-8b-instruct")


def test_per_type_adapter_must_record_its_foundation():
    """Otherwise the Foundation-bump work list becomes a manual audit (arch §12)."""
    with pytest.raises(ValueError, match="foundation_version"):
        RunManifest(
            run_id="lossrun-adapter-v1", run_type="per_type_adapter", doc_type="lossrun",
            dependencies=_deps(), training_config=_tc(), data_stats=_DS,
        )


def test_vit_training_must_be_lora_never_full_fine_tune():
    """arch §3 — full FT risks the pretrained OCR ability image-only depends on."""
    with pytest.raises(ValueError, match="add a ViT LoRA"):
        _tc(vit_trainable=True, vit_method="frozen")


def test_published_artifacts_must_point_somewhere():
    with pytest.raises(ValueError, match="points nowhere|no Blob path"):
        Artifacts(status="published")


def test_promoted_run_must_record_who_and_when():
    with pytest.raises(ValueError, match="promoted"):
        RunManifest(
            run_id="foundation-v1", run_type="foundation", status="promoted",
            dependencies=_deps(), training_config=_tc(), data_stats=_DS,
        )


# --------------------------------------------------------------------------
# Registry behaviour
# --------------------------------------------------------------------------

def test_manifest_written_to_blob_even_while_weights_are_staged(client):
    """The rule that stops a reclaimed volume erasing a training run (master §12a)."""
    manifest = _adapter("lossrun-adapter-v2", "lossrun", "foundation-v2")
    assert manifest.artifacts.status == "staged"
    key = W.write_manifest(manifest, client)
    assert client.exists(key)
    assert client.read_json(key)["artifacts"]["status"] == "staged"


def test_index_replaces_rather_than_duplicates_a_run(client):
    """A run's row changes as it moves trained -> evaluated -> promoted. Two rows
    would make the index ambiguous exactly when answering 'what is live?'."""
    manifest = _adapter("lossrun-adapter-v2", "lossrun", "foundation-v2")
    W.write_manifest(manifest, client)
    manifest.status = "evaluated"
    W.write_manifest(manifest, client)
    rows = [r for r in Q.list_runs(client) if r["run_id"] == "lossrun-adapter-v2"]
    assert len(rows) == 1
    assert rows[0]["status"] == "evaluated"


def test_adapters_depending_on_returns_the_revalidation_work_list(client):
    for dt in ("acord", "policy", "lossrun"):
        W.write_manifest(_adapter(f"{dt}-adapter-v2", dt, "foundation-v2"), client)
    W.write_manifest(_adapter("lossrun-adapter-v1", "lossrun", "foundation-v1"), client)

    assert Q.adapters_depending_on("foundation-v2", client) == [
        "acord-adapter-v2", "lossrun-adapter-v2", "policy-adapter-v2",
    ]
    assert Q.adapters_depending_on("foundation-v1", client) == ["lossrun-adapter-v1"]


def test_resolve_base_returns_the_pinned_model_with_no_adapter(client):
    """`base` is the shared path for the pilot baseline, day-zero pre-annotation,
    and `extract --model base` (IMPL-13)."""
    resolved = Q.resolve_model_version("base", client)
    assert resolved["kind"] == "base"
    assert resolved["foundation_adapter"] is None
    # Something a loader can open — the pod's local copy or the Hub id — not the
    # "model_id@revision" identity string, which vLLM cannot load.
    from common.config import base_model_source

    assert resolved["base_model"] == base_model_source()
    assert "@" not in resolved["base_model"]


def test_staged_version_resolves_to_the_volume_not_blob(client):
    """So a model can be spot-checked with `extract` before `package` runs."""
    W.write_manifest(
        RunManifest(run_id="foundation-v2", run_type="foundation",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS),
        client,
    )
    resolved = Q.resolve_model_version("v2", client, doc_type="lossrun")
    assert resolved["from_staging"] is True
    assert resolved["foundation_adapter"].startswith("/")


def test_published_version_resolves_to_blob(client):
    manifest = RunManifest(
        run_id="foundation-v3", run_type="foundation",
        dependencies=_deps(), training_config=_tc(), data_stats=_DS,
        artifacts=Artifacts(status="published", adapter_weights="adapters/foundation/v3"),
    )
    W.write_manifest(manifest, client)
    resolved = Q.resolve_model_version("v3", client)
    assert resolved["from_staging"] is False
    assert not resolved["foundation_adapter"].startswith("/")


def test_quantized_artifact_refused_while_staged(client):
    W.write_manifest(
        RunManifest(run_id="foundation-v2", run_type="foundation",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS),
        client,
    )
    with pytest.raises(Q.RegistryQueryError, match="still staged"):
        Q.resolve_model_version("v2", client, quant_format="q5_k_m")


def test_unknown_version_fails_loudly(client):
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"]), client)
    with pytest.raises(Q.RegistryQueryError, match="no run of the policy adapter found") as refused:
        Q.resolve_model_version("v99", client)
    assert "policy-v3" in str(refused.value)            # it names the runs it knows


def test_a_unified_run_resolves_by_tag(client):
    """The trainer writes run_type="unified" (arch v2.1 §4.1) while this resolver
    looked only for "foundation", so NO run produced by the current code could be
    resolved by tag — `extract --model v4` could not find what training had just
    written."""
    W.write_manifest(
        RunManifest(run_id="extractor-v4", run_type="unified",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS),
        client,
    )
    resolved = Q.resolve_model_version("v4", client)
    assert resolved["tag"] == "v4"
    assert resolved["foundation_adapter"]


def test_diff_surfaces_what_would_explain_a_regression(client):
    """arch §12 — did the corpus change? the Foundation? MinerU? a hyperparameter?"""
    W.write_manifest(
        RunManifest(run_id="lossrun-adapter-v1", run_type="per_type_adapter", doc_type="lossrun",
                    dependencies=_deps(foundation_version="foundation-v1",
                                       corpus_version="corpus/v2", mineru_version="1.3.0"),
                    training_config=_tc(epochs=3), data_stats=_DS),
        client,
    )
    W.write_manifest(_adapter("lossrun-adapter-v2", "lossrun", "foundation-v2"), client)

    diff = Q.diff_manifests("lossrun-adapter-v1", "lossrun-adapter-v2", client)
    assert diff["dependencies.corpus_version"] == {"a": "corpus/v2", "b": "corpus/v3"}
    assert diff["dependencies.foundation_version"]["b"] == "foundation-v2"
    assert diff["dependencies.mineru_version"]["b"] == "1.4.2"
    assert diff["training_config.epochs"] == {"a": 3, "b": 4}

    # Dependencies first: a corpus or Foundation change explains a regression far
    # more often than a metric delta does.
    assert Q.summarize_diff(diff)[0].startswith("dependencies.")


def test_a_failed_gate_cannot_be_promoted(client):
    """The gate has no override path by design (arch §13)."""
    manifest = _adapter("lossrun-adapter-v2", "lossrun", "foundation-v2")
    manifest.promotion = Promotion(beat_previous_on_all_gates=False, failed_gates=["list_field_recall"])
    W.write_manifest(manifest, client)
    with pytest.raises(W.RegistryError, match="no override path"):
        W.mark_promoted(manifest, client, promoted_by="someone")


def test_mark_published_flips_status_and_clears_staging(client):
    manifest = _adapter("lossrun-adapter-v2", "lossrun", "foundation-v2")
    manifest.artifacts.staging_path = "/runpod-volume/staging/adapters/lossrun/v2"
    W.write_manifest(manifest, client)

    published = W.mark_published(manifest, client, adapter_weights="adapters/lossrun/v2")
    assert published.artifacts.status == "published"
    assert published.artifacts.staging_path is None


def test_sweep_runs_are_recorded_but_excluded_from_default_listings(client):
    """Recorded so they are first-class (arch §11a); hidden so they do not swamp
    a listing. Sweeps are deferred until after the pilot."""
    manifest = RunManifest(
        run_id="sweep-lr-01", run_type="per_type_adapter", doc_type="lossrun",
        dependencies=_deps(foundation_version="foundation-v2"),
        training_config=_tc(), data_stats=_DS,
        is_sweep_run=True, sweep_id="phase1_lr", sweep_phase="lr",
    )
    W.write_manifest(manifest, client)

    assert "sweep-lr-01" not in [r["run_id"] for r in Q.list_runs(client)]
    assert "sweep-lr-01" in [r["run_id"] for r in Q.list_runs(client, include_sweeps=True)]


def test_latest_promoted_finds_the_serving_version(client):
    W.write_manifest(
        RunManifest(run_id="foundation-v2", run_type="foundation", status="promoted",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS,
                    promotion=Promotion(promoted_by="ci", promoted_at=datetime.now(UTC))),
        client,
    )
    assert Q.latest_promoted(client, "foundation") == "foundation-v2"


# --------------------------------------------------------------------------
# Release bundles (arch v2.1 §12.3)
# --------------------------------------------------------------------------

def _bundle(**over):
    """A release of the policy adapter's v3, as `package` writes it."""
    from registry_utils.models import ReleaseBundle

    base = dict(
        release_id="v3",
        tenant_scope="tenant_default",
        scope="policy",
        doc_types=["policy"],
        base_model="Qwen/Qwen3-VL-8B-Instruct@abc1234",
        adapter="policy-v3",
        merged_model=paths.merged_model_dir("v3", scope="policy"),
        serving_formats={"bf16": paths.merged_model_dir("v3", scope="policy")},
        prompt_hash="sha256:aaa",
        vision_config_hash="sha256:bbb",
        vllm_config_hash="sha256:ccc",
        lockfile_hash="sha256:ddd",
    )
    base.update(over)
    return ReleaseBundle(**base)


def test_a_release_without_the_bf16_reference_is_refused():
    """Every §13b quantization threshold is an absolute margin against bf16, so a
    release that ships only a quantized format has nothing to measure its own
    drop from."""
    with pytest.raises(ValueError, match="bf16 reference"):
        _bundle(serving_formats={"fp8": "models/.../fp8/"})


def test_a_calibrator_cannot_name_a_format_the_release_does_not_serve():
    with pytest.raises(ValueError, match="does not serve"):
        _bundle(calibrators={"fp8": "calib-fp8"})


def test_promoting_a_format_without_its_own_calibrator_is_refused():
    """A calibrator fitted on bf16 is wrong for FP8 — quantization moves the
    logprob distribution, and every review threshold downstream is defined
    against a calibrated score."""
    with pytest.raises(ValueError, match="no calibrator"):
        _bundle(
            status="promoted",
            serving_formats={"bf16": "a/", "fp8": "b/"},
            calibrators={"bf16": "calib-bf16"},
            gate_reports={"bf16": "g1", "fp8": "g2"},
        )


def test_promoting_a_format_without_its_own_gate_run_is_refused():
    """Quantization degrades exactly what was fine-tuned in, so a format inherits
    nothing from bf16's gate result."""
    with pytest.raises(ValueError, match="without a gate run"):
        _bundle(
            status="promoted",
            serving_formats={"bf16": "a/", "fp8": "b/"},
            calibrators={"bf16": "c1", "fp8": "c2"},
            gate_reports={"bf16": "g1"},
        )


def test_an_override_must_name_a_person_not_a_service_account():
    """v1 had no override, reasoning that a waivable gate is not a guarantee.
    That was right about the risk and wrong about the remedy — the v1 gate could
    not be passed at all, so the rule would have been broken in practice rather
    than in the open. The remedy is attribution, not absence."""
    from registry_utils.models import GateOverride

    with pytest.raises(ValueError, match="names the person|name the person"):
        GateOverride(approver="ci", reason="x" * 30, waived_gates=["ece_confidence"])


def test_an_override_must_say_which_gates_it_waives():
    from registry_utils.models import GateOverride

    with pytest.raises(ValueError):
        GateOverride(approver="A. Reviewer", reason="y" * 30, waived_gates=[])


def test_an_overridden_release_records_who_approved_it_in_the_index():
    from registry_utils.models import GateOverride

    bundle = _bundle(
        status="promoted",
        serving_formats={"bf16": "a/", "fp8": "b/"},
        calibrators={"bf16": "c1", "fp8": "c2"},
        gate_reports={"bf16": "g1"},
        override=GateOverride(
            approver="A. Reviewer",
            reason="FP8 ECE regressed 0.004 on a 30-document slice; shipping for the pilot",
            waived_gates=["ece_confidence"],
        ),
    )
    row = bundle.index_row()
    assert row["overridden"] is True
    assert row["override_approver"] == "A. Reviewer"


def test_release_paths_are_tenant_scoped_and_per_format():
    """A model trained on a tenant's documents IS that tenant's data, so the
    deletion cascade has to reach the release too (arch v2.1 §8b)."""
    bundle = paths.release_bundle("v3", "acme")
    assert paths.is_tenant_scoped(bundle) and paths.tenant_of(bundle) == "acme"

    bf16 = paths.release_gate_decision("v3", "bf16", "acme")
    fp8 = paths.release_gate_decision("v3", "fp8", "acme")
    assert bf16 != fp8, "each serving format needs its own gate decision"


def test_a_malformed_release_id_is_refused():
    """A release id is what --model resolves and what the endpoint pulls; a typo
    is a deploy that serves the wrong weights or nothing at all."""
    for bad in ("release-v2", "2026.11.1", "v", "v3a", "V3", "release-2026.11", "policy-v3"):
        with pytest.raises(paths.PathError, match="invalid release id"):
            paths.release_bundle(bad)


def test_a_release_is_named_by_its_cycles_version():
    """One number names a cycle's run, adapter and release; the older
    release-YYYY.M.N form still reads, so an index written before loads."""
    assert paths.release_bundle("v3") == f"releases/{paths.default_tenant()}/v3/bundle.json"
    assert paths.release_bundle("v2.1", "acme") == "releases/acme/v2.1/bundle.json"
    assert paths.release_bundle("release-2026.11.1").endswith("/release-2026.11.1/bundle.json")
    assert not hasattr(paths, "next_release_id")


@pytest.mark.parametrize("tag,number", [
    ("v3", 3), ("v10", 10), ("v2.1", 2), (" v4 ", 4),
    ("release-2026.11.1", None), ("policy-v3", None), ("3", None), ("", None),
])
def test_a_version_number_is_the_whole_number_of_a_version_tag(tag, number):
    assert paths.version_number(tag) == number


def test_the_next_version_is_one_more_than_the_highest_used():
    assert paths.next_version([]) == "v1"
    assert paths.next_version(["v1", "v2"]) == "v3"
    # By number, not by string: v10 is after v9.
    assert paths.next_version(["v9", "v10", "v2"]) == "v11"
    # A minor version counts as its whole number; anything else is ignored.
    assert paths.next_version(["v2.1", "release-2026.11.1", "extractor-v7", "corpus"]) == "v3"
    assert paths.next_version(iter(["v4"])) == "v5"


def test_every_corpus_of_a_tenant_sits_under_one_root():
    assert paths.corpus_root() == f"corpus/{paths.default_tenant()}"
    assert paths.corpus_dir("v3", "acme").startswith(paths.corpus_root("acme") + "/")
    assert paths.is_tenant_scoped(paths.corpus_root("acme"))


def test_gguf_formats_are_not_serving_formats():
    """GGUF targets llama.cpp, not vLLM. Asking for a GGUF format on the serving
    path is a category error, not a preference (arch v2.1 §13a)."""
    with pytest.raises(paths.PathError, match="unknown quantization format"):
        paths.release_gate_decision("v3", "q4_k_m")


def test_the_corpus_materializes_exactly_four_epoch_files():
    """Mode is sampled per document per epoch, so each epoch is a different draw.
    Materializing them makes a run reproducible from the corpus alone rather than
    depending on a sampler running identically at training time."""
    assert paths.corpus_epoch_file("v2", 1).endswith("train/epoch_1.jsonl")
    with pytest.raises(paths.PathError, match="outside 1-4"):
        paths.corpus_epoch_file("v2", 5)


def test_everything_a_release_owns_sits_under_one_tenant_prefix():
    """A tenant deletion has to remove the calibrators fitted on that tenant's
    fields as well as the bundle. One prefix makes that one delete instead of a
    checklist (arch v2.1 §8b)."""
    root = paths.release_dir("v3", "acme")
    for path in (
        paths.release_bundle("v3", "acme"),
        paths.release_calibrators("v3", "fp8", "acme"),
        paths.release_gate_decision("v3", "bf16", "acme"),
    ):
        assert path.startswith(root), f"{path} escapes the release prefix"
        assert paths.tenant_of(path) == "acme"


def test_the_v1_unscoped_trees_are_not_misread_as_tenant_scoped():
    """`eval-reports/v2/...` has no tenant segment. Adding that prefix to
    TENANT_SCOPED would make tenant_of() return the version tag as a tenant id —
    which is why release artifacts live under `releases/` instead of being
    scattered into the v1 trees."""
    assert paths.tenant_of(paths.eval_report("v2")) is None
    assert paths.tenant_of(paths.calibration_params("v2", "acord")) is None


# --------------------------------------------------------------------------
# Scoped runs (arch v2.1 §4.1) — additive, and the old guarantees kept
# --------------------------------------------------------------------------


def _scoped(run_id: str, scope: str, doc_types: list[str], **over) -> RunManifest:
    # On the configured base, as training records it: a version resolves only on
    # the base it trained on (query_registry.base_model_refusal).
    body = dict(
        run_id=run_id, run_type="scoped", scope=scope, doc_types=doc_types,
        dependencies=_deps(base_model=Q.configured_base_model()), training_config=_tc(), data_stats=_DS,
    )
    body.update(over)
    return RunManifest(**body)


def test_a_manifest_written_before_scopes_existed_still_loads(client):
    """Every manifest in Blob predates the scope fields. They are optional
    precisely so those files keep loading — pydantic forbids extras, so the
    reverse (old code, new manifest) is the direction that breaks, which is why
    this lands one deploy early."""
    W.write_manifest(
        RunManifest(run_id="extractor-v1", run_type="unified",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS),
        client,
    )
    manifest = Q.get("extractor-v1", client)

    assert manifest.scope is None, "absence must stay absence, not be backfilled"
    assert manifest.doc_types == []
    assert manifest.index_row()["scope"] == "unified", "absence READS as unified"


def test_a_scoped_run_must_say_what_it_covers():
    """What it covers is the one thing a scoped run exists to state, and the gate
    reads it to decide which metrics are not applicable."""
    with pytest.raises(ValueError, match="records no doc_types"):
        _scoped("policy-v2", "policy", [])
    with pytest.raises(ValueError, match="must name its scope"):
        RunManifest(run_id="policy-v2", run_type="scoped", doc_types=["policy"],
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS)


def test_a_unified_run_still_cannot_name_a_doc_type():
    """The v1 guarantee, kept: a run claiming to span everything must not
    secretly be one type. A narrower run is run_type='scoped', which says so."""
    with pytest.raises(ValueError, match="must not name one"):
        RunManifest(run_id="extractor-v2", run_type="unified", doc_type="policy",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS)
    with pytest.raises(ValueError, match="cannot name scope"):
        RunManifest(run_id="extractor-v2", run_type="unified", scope="policy",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS)


def test_a_scoped_run_records_coverage_not_a_graduated_lineage():
    """doc_type means "the §4.2 adapter this IS"; doc_types means "what this
    covers". Naming both makes the lineage ambiguous."""
    with pytest.raises(ValueError, match="doc_types .* not doc_type"):
        _scoped("policy-v2", "policy", ["policy"], doc_type="policy")


def test_two_scopes_at_one_version_resolve_to_their_own_artifacts(client):
    """The collision this phase exists to prevent: policy-v2 and lossrun-v2 are
    the same version tag. Resolving by tag alone would hand one scope the
    other's weights."""
    W.write_manifest(_scoped("policy-v2", "policy", ["policy"]), client)
    W.write_manifest(_scoped("lossrun-v2", "lossrun", ["lossrun"]), client)

    assert Q.get("policy-v2", client).scope == "policy"
    assert Q.get("lossrun-v2", client).scope == "lossrun"
    assert Q._find_run(client, "scoped", "v2", scope="policy") == "policy-v2"
    assert Q._find_run(client, "scoped", "v2", scope="lossrun") == "lossrun-v2"


def test_scoped_manifests_are_filed_apart_from_the_foundation_prefix(client):
    """A unified run and a scoped run can share a version, so filing them
    together would make "which manifest is v2's" ambiguous."""
    unified = paths.run_manifest("extractor-v2", "unified")
    scoped = paths.run_manifest("policy-v2", "scoped", scope="policy")

    assert unified != scoped
    assert "/scope/policy/" in scoped
    with pytest.raises(paths.PathError, match="needs the scope"):
        paths.run_manifest("policy-v2", "scoped")


def test_listing_by_scope_reads_an_absent_scope_as_unified(client):
    W.write_manifest(
        RunManifest(run_id="extractor-v1", run_type="unified",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS),
        client,
    )
    W.write_manifest(_scoped("policy-v1", "policy", ["policy"]), client)

    assert [r["run_id"] for r in Q.list_runs(client, scope="unified")] == ["extractor-v1"]
    assert [r["run_id"] for r in Q.list_runs(client, scope="policy")] == ["policy-v1"]


# --------------------------------------------------------------------------
# A version tag resolves to the policy adapter's run (common.adapter)
# --------------------------------------------------------------------------


def test_a_version_resolves_to_the_policy_adapters_run(client):
    """Every cycle trains the policy adapter, as run_type "scoped". Looking only
    for unified and Foundation runs, `extract --model v3` could not find the
    model the cycle had just trained."""
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"]), client)
    # Staged, it is the checkpoint its selection chose, not the ms-swift output root.
    chosen = f"{paths.scoped_staging_adapter_dir('policy', 'v3')}/v0-1/checkpoint-150"
    client.write_json(paths.checkpoint_selection("v3", scope="policy"), {"selected": chosen})

    # Served as it is released: the base with the version's adapter on every
    # request, never merged (decision 2026-10-23).
    resolved = Q.resolve_model_version("v3", client)
    assert resolved["tag"] == "v3" and resolved["kind"] == "base_plus_adapter"
    assert resolved["from_staging"] is True and resolved["artifact_status"] == "staged"
    assert resolved["foundation_adapter"] == chosen
    assert resolved["merged_model"] is None and resolved["quantized_model"] is None
    assert resolved["type_adapter"] is None


def test_a_published_version_of_the_adapter_resolves_to_blob(client):
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"], artifacts=Artifacts(
        status="published", adapter_weights=paths.scoped_adapter_dir("policy", "v3"))), client)

    resolved = Q.resolve_model_version("v3", client)
    assert resolved["kind"] == "base_plus_adapter" and resolved["from_staging"] is False
    assert resolved["foundation_adapter"] == paths.scoped_adapter_dir("policy", "v3")
    assert resolved["merged_model"] is None
    assert Q.resolve_model_version("v3", client, quant_format="bf16") == resolved


@pytest.mark.parametrize("status", ["staged", "published"])
def test_a_version_of_the_adapter_has_no_quantized_format(client, status):
    """Nothing is merged, so nothing is quantized: asking for fp8 is refused,
    staged or published, rather than pointed at a model that does not exist."""
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"], artifacts=Artifacts(
        status=status, adapter_weights=paths.scoped_adapter_dir("policy", "v3"),
        staging_path=paths.scoped_staging_adapter_dir("policy", "v3") if status == "staged" else None,
    )), client)
    with pytest.raises(Q.RegistryQueryError, match="LoRA on the bf16 base: nothing is merged"):
        Q.resolve_model_version("v3", client, quant_format="fp8")


def test_the_adapter_answers_for_its_version_before_an_older_unified_run(client):
    W.write_manifest(
        RunManifest(run_id="extractor-v3", run_type="unified",
                    dependencies=_deps(), training_config=_tc(), data_stats=_DS),
        client,
    )
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"]), client)
    client.write_json(paths.checkpoint_selection("v3", scope="policy"), {
        "selected": f"{paths.scoped_staging_adapter_dir('policy', 'v3')}/v0-1/checkpoint-150"})
    assert "/scope/policy/" in Q.resolve_model_version("v3", client)["foundation_adapter"]
    # Another scope's run of the same version is not the adapter.
    W.write_manifest(_scoped("lossrun-v4", "lossrun", ["lossrun"]), client)
    with pytest.raises(Q.RegistryQueryError, match="no run of the policy adapter"):
        Q.resolve_model_version("v4", client)


def test_the_index_row_records_whose_corpus_trained_it_and_its_tier():
    """The version table's columns (registry_utils.releases.version_table)."""
    manifest = _scoped("policy-v3", "policy", ["policy"], tenant_id="acme",
                       promotion=Promotion(tier="interim"))
    row = manifest.index_row()
    assert (row["tenant_id"], row["tier"]) == ("acme", "interim")
    assert _scoped("policy-v4", "policy", ["policy"]).index_row()["tier"] is None


def test_a_release_bundle_records_the_lines_it_trained_on():
    from registry_utils.models import ReleaseBundle

    assert ReleaseBundle.model_fields["lines"].default_factory() == []
    assert _bundle(lines=["homeowners"]).lines == ["homeowners"]


# --------------------------------------------------------------------------
# Storage login: a SAS token before the account key
# --------------------------------------------------------------------------


class _RecordingBackend:
    def __init__(self, connection_string=None, *, account_url=None, credential=None,
                 raw_container=None, raw_credential=None):
        self.connection_string, self.account_url, self.credential = (
            connection_string, account_url, credential)
        self.raw_container, self.raw_credential = raw_container, raw_credential


@pytest.fixture
def azure_env(monkeypatch):
    from artifact_registry import blob_client

    for name in ("AZURE_STORAGE_ACCOUNT_URL", "AZURE_STORAGE_SAS_TOKEN",
                 "AZURE_STORAGE_CONNECTION_STRING", "AZURE_RAW_SAS_TOKEN",
                 "AZURE_BLOB_CONTAINER", "AZURE_RAW_CONTAINER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(blob_client, "AzureBackend", _RecordingBackend)
    return monkeypatch


def test_a_sas_login_is_preferred_to_the_connection_string(azure_env):
    """The connection string carries the account key; a SAS is short-lived and
    limited to its permissions."""
    from artifact_registry.blob_client import azure_from_environment

    azure_env.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://acct.blob.core.windows.net")
    azure_env.setenv("AZURE_STORAGE_SAS_TOKEN", "?sv=2024&sig=abc")
    azure_env.setenv("AZURE_STORAGE_CONNECTION_STRING", "AccountName=acct;AccountKey=k")
    backend = azure_from_environment()
    assert backend.account_url == "https://acct.blob.core.windows.net"
    assert backend.credential == "sv=2024&sig=abc"
    assert backend.connection_string is None


def test_the_connection_string_still_works_alone(azure_env):
    from artifact_registry.blob_client import azure_from_environment

    azure_env.setenv("AZURE_STORAGE_CONNECTION_STRING", "AccountName=acct;AccountKey=k")
    assert azure_from_environment().connection_string == "AccountName=acct;AccountKey=k"


def test_half_a_sas_login_is_refused_without_naming_the_token(azure_env):
    from artifact_registry.blob_client import azure_from_environment

    azure_env.setenv("AZURE_STORAGE_SAS_TOKEN", "sv=2024&sig=secret")
    azure_env.setenv("AZURE_STORAGE_CONNECTION_STRING", "AccountName=acct;AccountKey=k")
    with pytest.raises(BlobError, match="AZURE_STORAGE_ACCOUNT_URL is not set") as caught:
        azure_from_environment()
    assert "secret" not in str(caught.value)


def test_no_storage_login_at_all_is_refused(azure_env):
    from artifact_registry.blob_client import azure_from_environment

    with pytest.raises(BlobError, match="no storage login"):
        azure_from_environment()


def _sas_login(env):
    env.setenv("AZURE_STORAGE_ACCOUNT_URL", "https://acct.blob.core.windows.net")
    env.setenv("AZURE_STORAGE_SAS_TOKEN", "sv=2024&sig=main-fake")


def test_the_raw_container_can_have_its_own_token(azure_env):
    """A container SAS reaches one container. With one token for both, the
    main container's token got 403 on the raw one, and the only token that
    worked was an account SAS that gave every pod the raw PDFs."""
    _sas_login(azure_env)
    azure_env.setenv("AZURE_BLOB_CONTAINER", "main-c")
    azure_env.setenv("AZURE_RAW_CONTAINER", "raw-c")
    azure_env.setenv("AZURE_RAW_SAS_TOKEN", "?sv=2024&sig=raw-fake")
    backend = BlobClient()._backend
    assert backend.credential == "sv=2024&sig=main-fake"
    assert (backend.raw_container, backend.raw_credential) == ("raw-c", "sv=2024&sig=raw-fake")


def test_without_a_raw_token_one_token_serves_both_containers(azure_env):
    _sas_login(azure_env)
    azure_env.setenv("AZURE_RAW_CONTAINER", "raw-c")
    backend = BlobClient()._backend
    assert backend.credential == "sv=2024&sig=main-fake"
    assert backend.raw_credential is None and backend.raw_container is None


def test_half_a_raw_login_is_refused_without_naming_a_token(azure_env):
    """The raw token joins a SAS login: beside a connection string alone it is
    half of one, and with no separate raw container it has nothing to log in to."""
    azure_env.setenv("AZURE_RAW_SAS_TOKEN", "sv=2024&sig=raw-secret")
    azure_env.setenv("AZURE_STORAGE_CONNECTION_STRING", "AccountName=acct;AccountKey=k")
    with pytest.raises(BlobError, match="AZURE_STORAGE_ACCOUNT_URL and AZURE_STORAGE_SAS_TOKEN are "
                                        "not set") as caught:
        BlobClient(raw_container="raw-c")
    assert "raw-secret" not in str(caught.value)

    _sas_login(azure_env)
    with pytest.raises(BlobError, match="no separate raw container") as caught:
        BlobClient()                      # AZURE_RAW_CONTAINER unset: raw is the main container
    assert "raw-secret" not in str(caught.value) and "main-fake" not in str(caught.value)


def test_raw_container_calls_use_the_raw_token(monkeypatch):
    """The backend picks the login per call, by the container the client routed to."""
    from artifact_registry.blob_client import AzureBackend

    sdk = pytest.importorskip("azure.storage.blob")
    made = []

    class FakeServiceClient:
        """Stands in for the SDK's client: says which token each call would carry."""

        def __init__(self, account_url, credential):
            made.append(credential)
            self.credential = credential

        def get_blob_client(self, container, blob):
            return (self.credential, container, blob)

        def get_container_client(self, container):
            return (self.credential, container)

    monkeypatch.setattr(sdk, "BlobServiceClient", FakeServiceClient)
    backend = AzureBackend(account_url="https://acct.blob.core.windows.net", credential="main-fake",
                           raw_container="raw-c", raw_credential="raw-fake")
    assert made == ["main-fake", "raw-fake"]
    assert backend._client("raw-c", "intake/b/d/x.pdf")[0] == "raw-fake"
    assert backend._client("main-c", "corpus/t/v1/manifest.json")[0] == "main-fake"
    assert backend._service_for("raw-c").get_container_client("raw-c")[0] == "raw-fake"

    single = AzureBackend(account_url="https://acct.blob.core.windows.net", credential="main-fake")
    assert single._client("raw-c", "raw-documents/t/policy/p/original.pdf")[0] == "main-fake"


def test_a_client_sends_raw_layer_calls_on_the_raw_login_and_the_rest_on_the_main_one(monkeypatch):
    """Through BlobClient, as every stage calls it, logged in from the
    environment: raw-documents/, intake/ and exports/ go to the raw container
    on its own SAS, every other key to the main container on the main SAS. Each
    call - read, exists, list - is checked against a fake SDK that records the
    token and container it was made with."""
    from types import SimpleNamespace

    sdk = pytest.importorskip("azure.storage.blob")
    calls = []

    class FakeBlobClient:
        def __init__(self, credential, container, blob):
            self.made_with = (credential, container)
            self.blob = blob

        def download_blob(self):
            calls.append(("read", *self.made_with))
            return SimpleNamespace(readall=lambda: b"{}")

        def exists(self):
            calls.append(("exists", *self.made_with))
            return True

    class FakeContainerClient:
        def __init__(self, credential, container):
            self.made_with = (credential, container)

        def list_blobs(self, name_starts_with):
            calls.append(("list", *self.made_with))
            return [SimpleNamespace(name=f"{name_starts_with}/x.json")]

    class FakeServiceClient:
        def __init__(self, account_url, credential):
            self.credential = credential

        def get_blob_client(self, container, blob):
            return FakeBlobClient(self.credential, container, blob)

        def get_container_client(self, container):
            return FakeContainerClient(self.credential, container)

    monkeypatch.setattr(sdk, "BlobServiceClient", FakeServiceClient)
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    for name, value in (("AZURE_STORAGE_ACCOUNT_URL", "https://acct.blob.core.windows.net"),
                        ("AZURE_STORAGE_SAS_TOKEN", "sv=2024&sig=main-fake"),
                        ("AZURE_RAW_SAS_TOKEN", "sv=2024&sig=raw-fake"),
                        ("AZURE_BLOB_CONTAINER", "main-c"), ("AZURE_RAW_CONTAINER", "raw-c")):
        monkeypatch.setenv(name, value)
    client = BlobClient(context="ingestion")

    raw = [paths.raw_pdf("policy", "policy_0001", "acme"), "intake/personal-v1/HO_1/golden.json",
           f"{paths.export_dir('comparisons', 'run-1', 'acme')}/summary.xlsx"]
    main = [paths.corpus_manifest("v1", "acme"), paths.registry_index(),
            paths.label_metadata("policy", "policy_0001", "acme")]
    for key in raw + main:
        calls.clear()
        client.read_bytes(key)
        client.exists(key)
        client.list(key.rsplit("/", 1)[0])
        login = ("sv=2024&sig=raw-fake", "raw-c") if key in raw else ("sv=2024&sig=main-fake", "main-c")
        assert calls == [(op, *login) for op in ("read", "exists", "list")], key
