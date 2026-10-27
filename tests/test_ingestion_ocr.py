"""IMPL-03 — ingestion, MinerU version/device pinning, and OCR output.

Runs against :class:`InMemoryBackend` and a stub OCR engine, so no Azure, no
MinerU install, and no CUDA are needed. The stub implements the same
:class:`~data_pipeline.ocr.run_mineru.OcrEngine` protocol the real engine does,
which is the point of that interface existing.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, BlobError, InMemoryBackend
from data_pipeline.ingestion import pull_raw_pdfs as ingest
from data_pipeline.ocr import mineru_version as mv
from data_pipeline.ocr import run_mineru
from data_pipeline.ocr.run_mineru import PageOutput

FIXTURE_OCR = Path(__file__).resolve().parent / "fixtures" / "ocr"


class StubEngine:
    """Deterministic stand-in for MinerU, driven by the OCR fixtures."""

    def __init__(self, markdown: str = "# STUB\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |\n"):
        self.markdown = markdown
        self.calls: list[dict] = []

    def process(self, pdf_bytes: bytes, *, device: str, max_long_side_px: int) -> list[PageOutput]:
        self.calls.append({"device": device, "cap": max_long_side_px})
        return [
            PageOutput(
                page_number=1,
                markdown=self.markdown,
                image_bytes=b"\x89PNG stub",
                table_row_count=run_mineru.count_table_rows(self.markdown),
            )
        ]


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def ocr_client(backend) -> BlobClient:
    return BlobClient(backend=backend, container="main", raw_container="raw", context="ocr")


@pytest.fixture
def ingestion_client(backend) -> BlobClient:
    return BlobClient(backend=backend, container="main", raw_container="raw", context="ingestion")


def _seed_raw_document(client: BlobClient, doc_type="policy", source_id="policy_0001", checksum="abc123"):
    client.write_bytes(paths.raw_pdf(doc_type, source_id), b"%PDF-1.7 stub")
    client.write_json(paths.raw_metadata(doc_type, source_id), {
        "source_id": source_id, "doc_type": doc_type,
        "checksum_sha256": checksum, "page_count": 1, "is_scanned": False,
    })


@pytest.fixture(autouse=True)
def _on_a_gpu_pod(monkeypatch):
    """OCR and rendering are GPU-only, so CI must simulate a GPU pod.

    Autouse rather than opt-in: every test in this module exercises a path that
    now refuses to run without CUDA, and that refusal is the point — a silent
    CPU fallback would write markdown from a different distribution.
    """
    from data_pipeline.ocr import mineru_version

    monkeypatch.setattr(mineru_version, "cuda_available", lambda: (True, "NVIDIA L4"))



# --------------------------------------------------------------------------
# Version and device pinning (arch §8a, §14)
# --------------------------------------------------------------------------

def test_version_mismatch_is_a_regression_trigger():
    """The model learns HOW MinerU formats output, so this is distribution shift."""
    env = mv.OcrEnvironment(mineru_version="1.4.2", device="cuda")
    with pytest.raises(mv.MinerUVersionError, match="distribution shift"):
        mv.assert_version_matches({"mineru_version": "1.3.0", "ocr_device": "cuda"}, env)


def test_device_mismatch_is_caught_too():
    env = mv.OcrEnvironment(mineru_version="1.4.2", device="cpu")
    with pytest.raises(mv.MinerUVersionError, match="device mismatch"):
        mv.assert_version_matches({"mineru_version": "1.4.2", "ocr_device": "cuda"}, env)


def test_device_check_relaxes_when_output_proves_identical():
    """If the Phase 0 spike shows GPU and CPU output match, the check can relax."""
    env = mv.OcrEnvironment(mineru_version="1.4.2", device="cpu")
    mv.assert_version_matches(
        {"mineru_version": "1.4.2", "ocr_device": "cuda"}, env, check_device=False
    )


def test_matching_environment_passes():
    env = mv.OcrEnvironment(mineru_version="1.4.2", device="cuda")
    mv.assert_version_matches({"mineru_version": "1.4.2", "ocr_device": "cuda"}, env)


def test_no_gpu_fails_rather_than_falling_back(monkeypatch):
    """Silent fallback is the failure worth preventing: the job would finish,
    write markdown from a different distribution than the corpus was built on,
    and report success. Failing here is the only way that becomes visible."""
    monkeypatch.setattr(mv, "cuda_available", lambda: (False, None))
    with pytest.raises(mv.MinerUVersionError, match="no CUDA device is visible"):
        mv.resolve_device("cuda", strict=True)


def test_gpu_is_the_only_device(monkeypatch):
    """Not a default with an opt-out — the only device there is.

    MinerU's CPU path uses lighter model variants and produces *different
    markdown* from the same PDF: different table splits, different cell
    boundaries. The model learns how MinerU formats its output (arch §8a), so a
    corpus built across both devices is built from two distributions. That is a
    correctness question, not a speed one, which is why there is no opt-out.
    """
    monkeypatch.setattr(mv, "cuda_available", lambda: (True, "NVIDIA L4"))
    assert mv.resolve_device(None) == "cuda"
    assert mv.resolve_device("cuda") == "cuda"

    with pytest.raises(mv.MinerUVersionError, match="GPU only"):
        mv.resolve_device("cpu")


def test_a_cpu_corpus_cannot_be_extended(monkeypatch):
    """Readable, not extendable: adding GPU documents to a CPU-built corpus puts
    two markdown formats in one training set."""
    with pytest.raises(mv.MinerUVersionError, match="GPU-only"):
        mv.assert_gpu_only("cpu")
    mv.assert_gpu_only("cuda")      # does not raise


def test_there_is_no_cpu_option_on_the_ocr_cli():
    """An option that accepts one value is a way to be surprised later.

    Asserted against argparse's real option list rather than the source text,
    which would also match the comment explaining why the flag is gone.
    """
    import pytest as _pytest

    from data_pipeline.ocr import run_mineru

    with _pytest.raises(SystemExit):        # argparse rejects an unknown option
        run_mineru.main(["--doc-type", "policy", "--source-ids", "x", "--device", "cpu"])


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------

def test_ingest_writes_pdf_and_metadata(tmp_path, ingestion_client, monkeypatch):
    """pypdf is an optional extra, so page inspection is stubbed — the ingestion
    contract being tested is the checksum, paths, and metadata, not PDF parsing."""
    monkeypatch.setattr(
        ingest, "inspect_pdf",
        lambda path: {"page_count": 2, "is_scanned": False, "text_chars_per_sampled_page": 900.0},
    )
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.7 sample")

    source_id, status = ingest.ingest_pdf(pdf, "policy", ingestion_client, known_checksums={})
    assert status == "ingested"
    assert source_id == "policy_0001"

    assert ingestion_client.exists(paths.raw_pdf("policy", source_id))
    meta = ingestion_client.read_json(paths.raw_metadata("policy", source_id))
    assert meta["checksum_sha256"] == ingest.sha256_file(pdf)
    assert meta["original_filename"] == "sample.pdf"
    assert meta["page_count"] == 2
    assert meta["is_scanned"] is False
    # Never omitted: absence must not be mistakable for "no PII".
    assert meta["pii_flags"]["status"] == "not_assessed"


def test_identical_content_is_deduplicated(tmp_path, ingestion_client, monkeypatch):
    """Loss Runs and renewal policies are re-submitted constantly. De-duping
    prevents paying to label the same document twice."""
    monkeypatch.setattr(
        ingest, "inspect_pdf",
        lambda path: {"page_count": 1, "is_scanned": False, "text_chars_per_sampled_page": 900.0},
    )
    first, second = tmp_path / "a.pdf", tmp_path / "b.pdf"
    first.write_bytes(b"%PDF identical bytes")
    second.write_bytes(b"%PDF identical bytes")

    known: dict[str, str] = {}
    sid_a, status_a = ingest.ingest_pdf(first, "policy", ingestion_client, known_checksums=known)
    sid_b, status_b = ingest.ingest_pdf(second, "policy", ingestion_client, known_checksums=known)

    assert status_a == "ingested"
    assert status_b == "duplicate"
    assert sid_b == sid_a, "the duplicate must resolve to the already-ingested source_id"


def test_a_corrected_document_becomes_a_new_source_id(tmp_path, ingestion_client, monkeypatch):
    """Raw documents are write-once, so a correction is a new source_id — that is
    what keeps historical runs reproducible against the bytes they trained on."""
    monkeypatch.setattr(
        ingest, "inspect_pdf",
        lambda path: {"page_count": 1, "is_scanned": False, "text_chars_per_sampled_page": 900.0},
    )
    original, corrected = tmp_path / "v1.pdf", tmp_path / "v2.pdf"
    original.write_bytes(b"%PDF original")
    corrected.write_bytes(b"%PDF corrected")

    known: dict[str, str] = {}
    sid_1, _ = ingest.ingest_pdf(original, "policy", ingestion_client, known_checksums=known)
    sid_2, status = ingest.ingest_pdf(corrected, "policy", ingestion_client, known_checksums=known)

    assert status == "ingested"
    assert sid_2 != sid_1
    assert ingestion_client.exists(paths.raw_pdf("policy", sid_1))
    assert ingestion_client.exists(paths.raw_pdf("policy", sid_2))


def test_scanned_flag_drives_the_eval_subset(tmp_path, ingestion_client, monkeypatch):
    """Detected at ingest because it is cheap here and expensive later, and it
    defines the scanned eval subset feeding the ViT gate (arch §3, §15)."""
    monkeypatch.setattr(
        ingest, "inspect_pdf",
        lambda path: {"page_count": 1, "is_scanned": True, "text_chars_per_sampled_page": 3.0},
    )
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF scanned")
    source_id, _ = ingest.ingest_pdf(pdf, "policy", ingestion_client, known_checksums={})
    assert ingestion_client.read_json(paths.raw_metadata("policy", source_id))["is_scanned"] is True


def test_checksum_is_stable_and_content_addressed(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "b.pdf"
    a.write_bytes(b"%PDF identical")
    b.write_bytes(b"%PDF identical")
    assert ingest.sha256_file(a) == ingest.sha256_file(b)

    b.write_bytes(b"%PDF different")
    assert ingest.sha256_file(a) != ingest.sha256_file(b)


def test_purge_plan_reports_the_full_cascade_without_deleting():
    """Retaining derived data from a purged document is a real audit gap, but
    corpus files are shared, so a purge is a rebuild rather than a delete."""
    plan = ingest.purge_plan("policy_0001", "policy", "broker_a")
    assert plan["raw"] and plan["processed"] and plan["golden_labels"]
    assert "rebuild" in plan["corpus_action"][0]
    assert "policy_0001" in plan["corpus_action"][0]
    # The frozen eval set is a copy of the document; a purge must reach it.
    assert plan["golden_eval_set"] == ["golden-eval-set/broker_a/policy_0001/"]
    assert plan["exports"] == ["exports/broker_a/"]


def _trained_run(client, run_id, tenant, corpus):
    from registry_utils.models import RunManifest
    from registry_utils.write_run_manifest import write_manifest
    from tests.test_registry import _DS, _deps, _tc

    manifest = RunManifest(
        run_id=run_id, run_type="foundation", tenant_id=tenant,
        dependencies=_deps(corpus_version=f"corpus/{corpus}"), training_config=_tc(), data_stats=_DS,
    )
    manifest.artifacts.adapter_weights = f"adapters/{run_id}"
    write_manifest(manifest, client)


def test_purge_plan_names_every_model_trained_on_the_document():
    """A model trained on the document is the tenant's data too (arch §8b): the
    plan names its weights and the releases built on it."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    client.write_json(paths.corpus_manifest("v1", "broker_a"),
                      {"source_ids_by_split": {"train": ["policy_0001"], "val": ["policy_0002"]}})
    client.write_json(paths.corpus_manifest("v2", "broker_a"),
                      {"source_ids_by_split": {"train": ["policy_0003"]}})
    _trained_run(client, "extractor-v1", "broker_a", "v1")
    _trained_run(client, "extractor-v2", "broker_a", "v2")
    _trained_run(client, "extractor-v9", "broker_b", "v1")
    client.write_json(paths.release_index("broker_a"),
                      [{"release_id": "release-2026.9.1", "adapter": "extractor-v1"}])

    plan = ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)
    [run] = plan["trained_weights"]
    assert run["run_id"] == "extractor-v1"
    assert run["adapter_weights"] == "adapters/extractor-v1"
    assert run["releases"] == ["release-2026.9.1"]
    assert run["action"] == "needs retrain or delete"


def test_a_tenant_purge_reaches_every_tenant_scoped_prefix_and_its_runs():
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    _trained_run(client, "extractor-v1", "broker_a", "v1")
    _trained_run(client, "extractor-v9", "broker_b", "v1")

    plan = ingest.tenant_purge_plan("broker_a", client)
    assert plan["prefixes"] == sorted(f"{p}/broker_a/" for p in paths.TENANT_SCOPED)
    assert [r["run_id"] for r in plan["trained_weights"]] == ["extractor-v1"]


def _keys_under(prefixes, keys):
    return {key for key in keys for prefix in prefixes if key.startswith(prefix)}


def test_every_purge_prefix_stops_at_a_folder_boundary():
    """Without the trailing '/', policy_1's eval-set prefix also matched
    policy_10's documents, and tenant acme's corpus prefix matched acme_west's:
    a purge run from the plan deleted someone else's data."""
    plan = ingest.purge_plan("policy_1", "policy", "acme")
    prefixes = [p for k in ("raw", "processed", "golden_labels", "golden_eval_set", "exports")
                for p in plan[k]]
    assert prefixes and all(p.endswith("/") for p in prefixes)
    neighbours = ["golden-eval-set/acme/policy_10/metadata.json",
                  "processed/acme/policy/policy_10/page_1.png",
                  "exports/acme_west/compare/x/report.json"]
    assert _keys_under(prefixes, neighbours) == set()
    assert _keys_under(prefixes, ["golden-eval-set/acme/policy_1/golden.json"])

    tenant = ingest.tenant_purge_plan("acme")
    assert all(p.endswith("/") for p in tenant["prefixes"])
    assert _keys_under(tenant["prefixes"], ["corpus/acme_west/v1/manifest.json",
                                            "releases/acme-2/release_index.json"]) == set()


def _labeled_import(client, folder, source_id="policy_0001", tenant="broker_a"):
    client.write_json(paths.label_metadata("policy", source_id, tenant),
                      {"source_id": source_id, "imported_from": folder})


def test_the_purge_plan_finds_the_delivered_folder_in_intake():
    """intake/{batch}/{document}/ holds the PDF and golden.json as delivered, and
    nothing deletes it. The import records the folder, not the batch."""
    backend = InMemoryBackend()
    client = BlobClient(backend=backend, container="main", raw_container="raw", context="ingestion")
    _labeled_import(client, "HO_1")
    for key in ("intake/personal-v1/HO_1/HO_1.pdf", "intake/personal-v1/HO_1/golden.json",
                "intake/personal-v2/HO_1/HO_1.pdf", "intake/personal-v1/HO_10/HO_10.pdf",
                "intake/personal-v1/AU_3/AU_3.pdf"):
        client.write_bytes(key, b"x")

    plan = ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)
    assert plan["intake"] == ["intake/personal-v1/HO_1/", "intake/personal-v2/HO_1/"]
    assert "confirm each is this document" in plan["intake_action"][0]
    # The pod's pulled copies, in every batch: the folder name is known.
    assert plan["pod_volume"][0].endswith("/*/HO_1/")


def test_without_raw_access_the_intake_check_is_left_to_the_operator():
    """A general client may not list the raw container: the plan says so and
    names the folder to look for, rather than reporting nothing to delete."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    _labeled_import(client, "HO_1")
    plan = ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)
    assert plan["intake"] == []
    assert "by hand" in plan["intake_action"][0] and "'HO_1'" in plan["intake_action"][0]

    unrecorded = ingest.purge_plan("policy_0002", "policy", "broker_a", client=client)
    assert "nothing records which batch" in unrecorded["intake_action"][0]
    assert "check each batch by hand" in unrecorded["pod_volume_action"][0]


def _reached(patterns):
    return {Path(found).as_posix() for pattern in patterns
            for found in glob.glob(pattern + "**", recursive=True) if Path(found).is_file()}


def test_the_purge_plan_names_the_copies_on_the_pods_volume(tmp_path, monkeypatch):
    """Every copy the code writes of a document's pages or values, written here
    by the calls that write them on the pod and then found from the plan's
    globs: training's page cache, the golden eval's (its keys are the frozen
    set's, under golden-{version}), the testing harness's pages and results,
    and the adapter cache. The golden eval's and the harness's were missed."""
    from inference_core import model_runner
    from serving.pipeline import ExtractionResult
    from testing import run_extraction
    from training.stage_data import localize_keys

    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT", (tmp_path / "volume").as_posix())
    monkeypatch.setattr(run_extraction, "TESTING_ROOT", tmp_path / "testing")
    monkeypatch.setattr(run_extraction, "PAGE_CACHE", tmp_path / "testing" / "ocr_cache")
    monkeypatch.setattr(model_runner, "ADAPTER_CACHE", tmp_path / "adapters")
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")

    def write_copies(source_id):
        page = paths.processed_page("policy", source_id, 1, "png", "broker_a")
        frozen = f"{paths.golden_eval_set_dir('broker_a')}/{source_id}/page_1.png"
        for key in (page, frozen):
            client.write_bytes(key, b"png")
        # The calls training (pipeline_dag), the golden eval and the testing harness make.
        localize_keys(client, [page], paths.staging_train_images_dir("v3", "broker_a"))
        localize_keys(client, [frozen], paths.staging_train_images_dir("golden-v3", "broker_a"))
        localize_keys(client, [page], run_extraction.PAGE_CACHE)
        result = ExtractionResult(source_id, "policy", "extractor-v3", "ocr_plus_image", True, 0.9)
        run_extraction.write_outputs(result, {}, root=run_extraction.TESTING_ROOT)
        return {p.as_posix() for p in tmp_path.rglob("*") if p.is_file()}

    written = write_copies("policy_0001")
    neighbour = write_copies("policy_0002") - written

    plan = ingest.purge_plan("policy_0001", "policy", "broker_a")
    volume = plan["pod_volume"]
    assert written <= _reached(volume), sorted(written - _reached(volume))
    assert not neighbour & _reached(volume), "another document's copies were named"
    images = paths.staging_train_images_dir("*", "broker_a")
    assert f"{images}/processed/broker_a/policy/policy_0001/" in volume
    assert f"{paths.staging_train_data_dir('*', '*')}/" in volume
    assert any(p.startswith("/workspace/ocr_check") for p in volume)
    assert f"{(tmp_path / 'adapters').as_posix()}/" in volume
    # Each path comes with what it holds and what to do with it.
    actions = plan["pod_volume_action"]
    assert all(any(action.startswith(f"{path}: ") for action in actions) for path in volume)
    results = next(a for a in actions if a.startswith(f"{(tmp_path / 'testing').as_posix()}/results/*/"))
    assert "not by tenant" in results

    tenant = ingest.tenant_purge_plan("broker_a")
    assert (written | neighbour) <= _reached(tenant["pod_volume"])
    assert f"{images}/" in tenant["pod_volume"]
    assert f"{(tmp_path / 'adapters').as_posix()}/" in tenant["pod_volume"]
    assert "not filed by tenant" in tenant["intake_action"][0]


def test_entries_that_reach_other_documents_or_tenants_say_so():
    """A result set holds many documents' outputs, and the pod's intake folder
    every tenant's batches: neither is the document's or the tenant's to delete
    whole, and the plan said nothing of it."""
    from data_pipeline.ingestion.pull_intake import default_intake_dir

    plan = ingest.purge_plan("policy_0001", "policy", "broker_a")
    [action] = plan["exports_action"]
    assert action.startswith("exports/broker_a/: ") and "many documents' outputs" in action
    assert "check each set for this document" in action and "not the prefix" in action

    tenant = ingest.tenant_purge_plan("broker_a")
    assert tenant["pod_volume"][0] == f"{default_intake_dir().as_posix()}/"
    assert "holds every tenant's batches: check each one by hand" in tenant["pod_volume_action"][0]


def test_the_purge_plan_looks_for_every_folder_the_document_was_imported_from():
    """The same PDF delivered twice is one document imported twice. The import
    keeps both folders (with the batch when it pulled one), and the plan looks
    for each: the last folder alone missed the first delivery."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw", context="ingestion")
    client.write_json(paths.label_metadata("policy", "policy_0001", "broker_a"), {
        "source_id": "policy_0001", "imported_from": "HO_0001",
        "imported_from_history": [{"folder": "HO_1", "batch": "personal-v1"},
                                  {"folder": "HO_0001", "batch": None}]})
    for key in ("intake/personal-v1/HO_1/HO_1.pdf", "intake/personal-v2/HO_0001/HO_0001.pdf",
                "intake/personal-v2/HO_10/HO_10.pdf"):
        client.write_bytes(key, b"x")

    plan = ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)
    assert plan["intake"] == ["intake/personal-v1/HO_1/", "intake/personal-v2/HO_0001/"]
    assert "'HO_1', 'HO_0001'" in plan["intake_action"][0]
    assert "the record names intake/personal-v1/HO_1/" in plan["intake_action"][0]
    pulled = [p for p in plan["pod_volume"] if p.endswith(("/*/HO_1/", "/*/HO_0001/"))]
    assert len(pulled) == 2 and plan["pod_volume"][:2] == pulled


def test_an_intake_lookup_that_finds_nothing_is_no_all_clear():
    """A record from before the history was kept names only the last folder; a
    first delivery under another name is not found by it. An empty result says
    so, rather than reading as nothing to delete."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw", context="ingestion")
    _labeled_import(client, "HO_0001")
    client.write_bytes("intake/personal-v1/HO_1/HO_1.pdf", b"the same bytes")

    plan = ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)
    assert plan["intake"] == []
    action = plan["intake_action"][0]
    assert action.startswith("none found by name: ") and "'HO_0001'" in action and "Check by hand" in action


def test_each_run_names_its_staged_weights_eval_report_and_calibrator():
    """Eval reports and calibrators sit under shared prefixes, so no tenant
    prefix reaches them: the report's error_records hold the expected values."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import write_manifest

    _trained_run(client, "extractor-v1", "broker_a", "v1")
    manifest = get("extractor-v1", client)
    manifest.artifacts.staging_path = "/workspace/staging/adapters/foundation/v1"
    manifest.artifacts.eval_report = paths.eval_report("v1")
    manifest.artifacts.calibration_params = paths.calibration_params("v1", "policy")
    write_manifest(manifest, client)

    [run] = ingest.tenant_purge_plan("broker_a", client)["trained_weights"]
    assert run["staging_path"] == "/workspace/staging/adapters/foundation/v1"
    assert run["eval_report"] == "eval-reports/v1/summary.json"
    assert run["calibration_params"] == "calibration/v1/policy.json"


def test_an_unreadable_manifest_is_listed_not_fatal():
    """Manifests forbid unknown fields, the index can hold a row no path is
    built from, and a blob can hold something that is not JSON. Each is that
    run's problem: the plan lists it to check by hand and goes on."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    _trained_run(client, "extractor-v1", "broker_a", "v1")
    _trained_run(client, "extractor-v2", "broker_a", "v1")
    _trained_run(client, "extractor-v3", "broker_a", "v1")
    key_v2 = paths.run_manifest("extractor-v2", "unified")
    client.write_json(key_v2, {**client.read_json(key_v2), "field_from_a_newer_build": 1})
    client.write_text(paths.run_manifest("extractor-v3", "unified"), "{not json")
    index = client.read_json(paths.registry_index())
    index["runs"].append({"run_id": "extractor-v4", "run_type": "no_such_type"})
    client.write_json(paths.registry_index(), index)

    runs = {r["run_id"]: r["action"] for r in ingest.tenant_purge_plan("broker_a", client)["trained_weights"]}
    assert runs["extractor-v1"] == "needs retrain or delete"
    assert runs["extractor-v2"] == "check by hand: the manifest cannot be read (ValidationError)"
    assert runs["extractor-v3"] == "check by hand: the manifest cannot be read (JSONDecodeError)"
    assert runs["extractor-v4"] == "check by hand: the manifest cannot be read (PathError)"


def test_lineage_lists_continued_runs_and_adapters_on_a_listed_foundation():
    """A run continued from a checkpoint carries what earlier corpora taught
    it; a per-type adapter sits on its Foundation's weights. Neither shows the
    document in its own corpus manifest."""
    from registry_utils.models import RunManifest
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import write_manifest
    from tests.test_registry import _DS, _deps, _tc

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    client.write_json(paths.corpus_manifest("v1", "broker_a"),
                      {"source_ids_by_split": {"train": ["policy_0001"]}})
    client.write_json(paths.corpus_manifest("v2", "broker_a"),
                      {"source_ids_by_split": {"train": ["policy_0003"]}})
    _trained_run(client, "extractor-v1", "broker_a", "v1")
    _trained_run(client, "extractor-v2", "broker_a", "v2")
    _trained_run(client, "extractor-v3", "broker_a", "v2")
    continued = get("extractor-v3", client)
    continued.continued_from = "/workspace/staging/adapters/foundation/v1/checkpoint-280"
    write_manifest(continued, client)
    for run_id, foundation in (("policy-adapter-v1", "extractor-v1"), ("policy-adapter-v2", "extractor-v2")):
        write_manifest(RunManifest(
            run_id=run_id, run_type="per_type_adapter", doc_type="policy", tenant_id="broker_a",
            dependencies=_deps(corpus_version="corpus/v2", foundation_version=foundation),
            training_config=_tc(), data_stats=_DS,
        ), client)

    runs = {r["run_id"]: r["action"] for r in
            ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)["trained_weights"]}
    assert runs["extractor-v1"] == "needs retrain or delete"
    assert runs["extractor-v3"] == (
        "check by hand: continued from /workspace/staging/adapters/foundation/v1/checkpoint-280, "
        "which may carry earlier corpora")
    assert runs["policy-adapter-v1"] == "built on extractor-v1, listed here: needs retrain or delete"
    # Neither trained on the document nor built on a run that did.
    assert "extractor-v2" not in runs and "policy-adapter-v2" not in runs


def test_every_quantized_format_the_run_published_is_named():
    """publish exports each format in quantized_formats to its own directory;
    quantized_model names the first one only, and the plan named that alone."""
    from registry_utils.query_registry import get
    from registry_utils.write_run_manifest import write_manifest
    from tests.test_registry import _scoped

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    _trained_run(client, "extractor-v2", "broker_a", "v2")
    write_manifest(_scoped("policy-v2", "policy", ["policy"], tenant_id="broker_a"), client)
    for run_id, scope in (("extractor-v2", None), ("policy-v2", "policy")):
        manifest = get(run_id, client)
        manifest.artifacts.quantized_model = paths.quantized_model_dir("v2", "fp8", scope=scope)
        manifest.artifacts.quantized_formats = ["fp8", "awq_int4"]
        write_manifest(manifest, client)

    runs = {r["run_id"]: r for r in ingest.tenant_purge_plan("broker_a", client)["trained_weights"]}
    assert runs["extractor-v2"]["quantized_models"] == [
        "quantized-models/unified/v2/vllm/fp8", "quantized-models/unified/v2/vllm/awq_int4"]
    assert runs["policy-v2"]["quantized_models"] == [
        "quantized-models/scope/policy/v2/vllm/fp8", "quantized-models/scope/policy/v2/vllm/awq_int4"]
    assert runs["extractor-v2"]["quantized_model"] == "quantized-models/unified/v2/vllm/fp8"


class _FailingBackend(InMemoryBackend):
    """exists() fails on one key, as AzureBackend's does on a 503 or a timeout -
    with a message quoting the URL, as the SDK's can."""

    def __init__(self):
        super().__init__()
        self.failing = None

    def exists(self, container, key):
        if key == self.failing:
            raise BlobError(f"failed checking https://acct.blob.core.windows.net/{container}/{key}"
                            "?sv=1&sig=FAKESIG: timed out")
        return super().exists(container, key)


@pytest.mark.parametrize("failing", ["corpus manifest", "release index", "run registry"])
def test_a_failed_storage_call_is_listed_to_check_by_hand_and_the_plan_goes_on(failing):
    """One storage call that failed ended the whole plan, every other run and
    copy unreported. Each is now its own entry's to check by hand, named by the
    error's type and never its message."""
    backend = _FailingBackend()
    client = BlobClient(backend=backend, container="main", raw_container="raw")
    for corpus in ("v1", "v2"):
        client.write_json(paths.corpus_manifest(corpus, "broker_a"),
                          {"source_ids_by_split": {"train": ["policy_0001"]}})
    _trained_run(client, "extractor-v1", "broker_a", "v1")
    _trained_run(client, "extractor-v2", "broker_a", "v2")
    backend.failing = {"corpus manifest": paths.corpus_manifest("v2", "broker_a"),
                       "release index": paths.release_index("broker_a"),
                       "run registry": paths.registry_index()}[failing]

    plan = ingest.purge_plan("policy_0001", "policy", "broker_a", client=client)
    runs = plan["trained_weights"]
    assert plan["raw"] and plan["pod_volume"] and plan["intake_action"]
    assert "FAKESIG" not in json.dumps(plan)
    if failing == "corpus manifest":
        assert {r["run_id"]: r["action"] for r in runs} == {
            "extractor-v1": "needs retrain or delete",
            "extractor-v2": "check by hand: the run's corpus manifest cannot be read (BlobError)"}
    elif failing == "release index":
        assert [r["action"] for r in runs] == ["needs retrain or delete"] * 2
        assert all(r["releases"] == ["check by hand: the release index cannot be read (BlobError)"]
                   for r in runs)
    else:
        assert runs == [{"run_id": None, "action": "check by hand: the run registry cannot be read "
                                                   "(BlobError), so no run trained on the data is listed"}]


def test_unknown_doc_type_is_refused(tmp_path, ingestion_client):
    with pytest.raises(ingest.IngestionError, match="unknown doc_type"):
        ingest.ingest_directory(tmp_path, "invoice", ingestion_client)


def test_unclassified_bucket_is_allowed(tmp_path, ingestion_client):
    """Forcing a type guess at ingest would bake a mistake into the source_id,
    which is the identifier everything downstream joins on."""
    result = ingest.ingest_directory(tmp_path, ingest.UNCLASSIFIED, ingestion_client)
    assert result.ingested == []


# --------------------------------------------------------------------------
# OCR output
# --------------------------------------------------------------------------

def test_table_row_count_discounts_header_and_separator():
    """Feeds the row-completeness cross-check: the model extracting 6 rows from a
    page MinerU saw 8 on is a recall failure per-field confidence cannot see."""
    markdown = (
        "| Claim | Paid |\n| --- | --- |\n"
        "| A | 1 |\n| B | 2 |\n| C | 3 |\n"
    )
    assert run_mineru.count_table_rows(markdown) == 3


def test_table_row_count_on_a_real_fixture():
    markdown = (FIXTURE_OCR / "lossrun_0001_page_1.md").read_text(encoding="utf-8")
    # The fixture Loss Run has exactly three claim rows.
    assert run_mineru.count_table_rows(markdown) == 3


def test_table_row_count_is_zero_without_a_table():
    assert run_mineru.count_table_rows("# Heading\n\nSome prose.\n") == 0


def test_process_document_records_version_device_and_cap(ocr_client, monkeypatch):
    monkeypatch.setattr(
        run_mineru, "current_environment",
        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda", "NVIDIA L4"),
    )
    _seed_raw_document(ocr_client)
    meta = run_mineru.process_document("policy", "policy_0001", ocr_client, StubEngine())

    assert meta["mineru_version"] == "1.4.2"
    assert meta["ocr_device"] == "cuda"
    assert meta["resolution_cap_px"] == 1792
    assert meta["page_count"] == 1
    assert meta["table_row_counts"] == {"1": 2}
    assert meta["source_checksum"] == "abc123"


def test_process_document_writes_page_image_and_markdown(ocr_client, monkeypatch):
    monkeypatch.setattr(
        run_mineru, "current_environment",
        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"),
    )
    _seed_raw_document(ocr_client)
    run_mineru.process_document("policy", "policy_0001", ocr_client, StubEngine())

    assert ocr_client.exists(paths.processed_page("policy", "policy_0001", 1, "png"))
    assert ocr_client.exists(paths.processed_page("policy", "policy_0001", 1, "md"))


def test_reprocessing_is_skipped_when_nothing_changed(ocr_client, monkeypatch):
    monkeypatch.setattr(
        run_mineru, "current_environment",
        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"),
    )
    _seed_raw_document(ocr_client)
    engine = StubEngine()
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 1, "second run should have been skipped"


def test_a_mineru_version_change_forces_reprocessing(ocr_client, monkeypatch):
    """Each of version, device and checksum changes the input distribution the
    model was trained to arbitrate against."""
    _seed_raw_document(ocr_client)
    engine = StubEngine()

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.5.0", "cuda"))
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 2


def test_a_device_change_forces_reprocessing(ocr_client, monkeypatch):
    _seed_raw_document(ocr_client)
    engine = StubEngine()

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cpu"))
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 2


def test_engine_receives_the_configured_resolution_cap(ocr_client, monkeypatch):
    """The cap must be identical here and at inference, or the model is served a
    distribution it never trained on (arch §11)."""
    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    _seed_raw_document(ocr_client)
    engine = StubEngine()
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert engine.calls[0]["cap"] == 1792


def test_find_unprocessed_lists_only_documents_without_ocr_meta(ocr_client, monkeypatch):
    # The in-process engine's readings, from before every page was read by OCR:
    # held to version and tier, as then (test_ocr_every_page holds the setting).
    monkeypatch.setattr("common.config.ocr_every_page", lambda: False)
    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    _seed_raw_document(ocr_client, source_id="policy_0001")
    _seed_raw_document(ocr_client, source_id="policy_0002")
    assert run_mineru.find_unprocessed(ocr_client, "policy") == ["policy_0001", "policy_0002"]

    run_mineru.process_document("policy", "policy_0001", ocr_client, StubEngine())
    assert run_mineru.find_unprocessed(ocr_client, "policy") == ["policy_0002"]


def test_a_document_another_mineru_read_is_unprocessed_for_this_one(ocr_client, monkeypatch):
    """The model learns how MinerU formats its output: a tenant read partly by
    MinerU 1.x and partly by 3.x is two distributions, so --all-unprocessed reads
    the 1.x documents again."""
    # The in-process engine's readings, from before every page was read by OCR:
    # held to version and tier, as then (test_ocr_every_page holds the setting).
    monkeypatch.setattr("common.config.ocr_every_page", lambda: False)
    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.3.12", "cuda"))
    _seed_raw_document(ocr_client, source_id="policy_0001")
    run_mineru.process_document("policy", "policy_0001", ocr_client, StubEngine())

    assert run_mineru.find_unprocessed(ocr_client, "policy") == []
    assert run_mineru.find_unprocessed(ocr_client, "policy", mineru_version="1.3.12") == []
    assert run_mineru.find_unprocessed(ocr_client, "policy", mineru_version="3.4.5") == ["policy_0001"]


def test_a_corpus_read_by_two_minerus_is_refused():
    """The manifest records one MinerU version, taken from one document: a mix
    would train on two input distributions and not even show."""
    from orchestration.pipeline_dag import PipelineError, assert_one_mineru_version

    assert_one_mineru_version({"3.4.5": 103})
    with pytest.raises(PipelineError, match=r"1\.3\.12 \(40\), 3\.4\.5 \(63\).*--all-unprocessed"):
        assert_one_mineru_version({"3.4.5": 63, "1.3.12": 40})


def test_ocr_engine_protocol_is_satisfied_by_the_stub():
    """The swappable interface is what lets the pipeline be verified without
    MinerU, and what lets the Phase 0 spike replace the engine outright."""
    engine: run_mineru.OcrEngine = StubEngine()
    pages = engine.process(b"%PDF", device="cpu", max_long_side_px=1792)
    assert pages and pages[0].page_number == 1


# --------------------------------------------------------------------------
# Near-duplicate detection and grouping (arch v2.1 §8.2)
# --------------------------------------------------------------------------

def _fp(source_id, **over):
    from data_pipeline.ingestion.dedup_and_group import DocumentFingerprint

    base = dict(
        source_id=source_id,
        doc_type="policy",
        content_sha256=f"sha-{source_id}",
    )
    base.update(over)
    return DocumentFingerprint(**base)


def test_the_same_file_twice_is_grouped_not_dropped():
    """A carrier that issues four hundred near-identical certificates is a real
    part of the distribution. Removing them trains on a corpus that does not look
    like production; what must not happen is those four hundred spanning a split."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([_fp("a", content_sha256="same"), _fp("b", content_sha256="same")])

    assert report.group_of["a"] == report.group_of["b"]
    assert report.exact_duplicates == {"b": "a"}
    assert set(report.group_of) == {"a", "b"}, "neither document was dropped"


def test_a_renewal_groups_with_its_prior_year():
    """Same carrier, same template, same account — every value differs and the
    family does not. This is the leak the source-document split could not see."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([
        _fp("2025", carrier="Acme Ins. Co.", template_id="ACORD25", account="Rivera Fabrication"),
        _fp("2026", carrier="ACME INS CO", template_id="acord25", account="rivera fabrication"),
    ])
    assert report.group_of["2025"] == report.group_of["2026"], (
        "carrier and account names differing only in case and punctuation are one family"
    )


def test_an_identical_page_one_layout_is_one_template():
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([
        _fp("a", layout_phash="ffff0000"),
        _fp("b", layout_phash="ffff0000"),
        _fp("c", layout_phash="0000ffff"),
    ])
    assert report.group_of["a"] == report.group_of["b"]
    assert report.group_of["c"] != report.group_of["a"]


def test_near_duplicate_text_merges_groups():
    from data_pipeline.ingestion.dedup_and_group import assign_groups, minhash

    shared = " ".join(f"policy clause number {i} applies to the named insured" for i in range(40))
    report = assign_groups([
        _fp("a", minhash=minhash(shared)),
        _fp("b", minhash=minhash(shared + " one extra trailing clause")),
        _fp("c", minhash=minhash("completely unrelated loss run claim table headings only")),
    ])
    assert report.group_of["a"] == report.group_of["b"]
    assert report.group_of["c"] != report.group_of["a"]
    assert report.near_duplicate_pairs


def test_grouping_is_transitive():
    """A chain of renewals is one account. Letting the ends of the chain split
    leaks exactly what grouping prevents, even where the ends are not themselves
    similar enough to merge directly."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([
        _fp("a", layout_phash="1111"),
        _fp("b", layout_phash="1111", account="shared"),
        _fp("c", account="shared"),
    ])
    assert report.group_of["a"] == report.group_of["c"]
    assert report.group_count == 1


def test_documents_naming_no_carrier_are_reported():
    """They can still be grouped by layout and text, but the held-out-carrier
    slice cannot use them — so the count is worth surfacing."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([_fp("a"), _fp("b", carrier="Acme")])
    assert report.unattributed == ["a"]


def test_an_unrelated_document_is_its_own_family():
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    report = assign_groups([_fp("a"), _fp("b"), _fp("c")])
    assert report.group_count == 3


def test_word_shingles_survive_a_single_ocr_error():
    """Character shingles break on every shingle overlapping a corrupted
    character; a word shingle loses only those containing that word."""
    from data_pipeline.ingestion.dedup_and_group import estimated_jaccard, minhash

    clean = " ".join(f"the named insured shown in declarations item {i}" for i in range(30))
    ocr_error = clean.replace("insured", "1nsured", 1)
    assert estimated_jaccard(minhash(clean), minhash(ocr_error)) > 0.85


def test_grouping_is_deterministic():
    """Corpus composition must be reproducible from the manifest alone."""
    from data_pipeline.ingestion.dedup_and_group import assign_groups

    docs = [_fp(f"d{i}", layout_phash=f"h{i % 3}") for i in range(9)]
    assert assign_groups(docs).group_of == assign_groups(list(reversed(docs))).group_of


def test_a_reading_from_before_the_drawn_figure_check_is_redone_only_where_figures_are_drawn(
        ocr_client, monkeypatch):
    """Documents OCR'd from their text layer before the check existed: a drawn one
    is read again through OCR; the rest keep their reading."""
    from data_pipeline.ocr import modality

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    _seed_raw_document(ocr_client)
    engine = StubEngine()
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    meta_key = paths.ocr_meta("policy", "policy_0001")
    old = {k: v for k, v in ocr_client.read_json(meta_key).items()
           if k not in ("drawn_check", "read_by_ocr", "ocr_rules")}
    ocr_client.write_json(meta_key, old)                                   # as an older version wrote it

    checks = []
    monkeypatch.setattr(modality, "detect_modality", lambda pdf: checks.append(1) or modality.NATIVE)
    monkeypatch.setattr(modality, "data_is_drawn", lambda pdf: False)
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 1                                          # not drawn: kept
    assert ocr_client.read_json(meta_key)["ocr_rules"] == run_mineru.OCR_RULES_VERSION   # and recorded
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 1 and len(checks) == 1                     # so not checked again

    ocr_client.write_json(meta_key, old)
    monkeypatch.setattr(modality, "data_is_drawn", lambda pdf: True)
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 2                                          # drawn: read again
    assert ocr_client.read_json(meta_key)["drawn_check"] is True
    assert ocr_client.read_json(meta_key)["ocr_rules"] == run_mineru.OCR_RULES_VERSION
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 2                                          # and not again after that


def test_a_reading_from_before_the_visible_text_rule_is_redone_where_it_is_a_scan(ocr_client, monkeypatch):
    """Read from its text layer under the drawn-figure rules, a searchable scan
    passed for digital: its invisible OCR layer counted. It is read again
    through OCR; a document still digital keeps its reading."""
    from data_pipeline.ocr import modality

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    _seed_raw_document(ocr_client)
    engine = StubEngine()
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    meta_key = paths.ocr_meta("policy", "policy_0001")
    rules_1 = {k: v for k, v in ocr_client.read_json(meta_key).items() if k != "ocr_rules"}
    rules_1.update(drawn_check=True, read_by_ocr=False, is_scanned=False)   # as the drawn-figure version wrote it
    ocr_client.write_json(meta_key, rules_1)

    monkeypatch.setattr(modality, "data_is_drawn", lambda pdf: True)        # not re-checked under rules 1
    monkeypatch.setattr(modality, "detect_modality", lambda pdf: modality.NATIVE)
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 1                                          # still digital: kept
    assert ocr_client.read_json(meta_key)["ocr_rules"] == run_mineru.OCR_RULES_VERSION

    ocr_client.write_json(meta_key, rules_1)
    monkeypatch.setattr(modality, "detect_modality", lambda pdf: modality.SCANNED)
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 2                                          # a scan after all: read again
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 2


def test_a_reading_whose_recheck_cannot_run_is_kept_and_checked_again_next_time(ocr_client, monkeypatch):
    from data_pipeline.ocr import modality

    monkeypatch.setattr(run_mineru, "current_environment",
                        lambda device=None, strict=True: mv.OcrEnvironment("1.4.2", "cuda"))
    _seed_raw_document(ocr_client)
    engine = StubEngine()
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    meta_key = paths.ocr_meta("policy", "policy_0001")
    rules_1 = {k: v for k, v in ocr_client.read_json(meta_key).items() if k != "ocr_rules"}
    rules_1.update(drawn_check=True, read_by_ocr=False, is_scanned=False)
    ocr_client.write_json(meta_key, rules_1)

    def broken(pdf):
        raise RuntimeError("cannot read it")

    monkeypatch.setattr(modality, "detect_modality", broken)
    run_mineru.process_document("policy", "policy_0001", ocr_client, engine)
    assert len(engine.calls) == 1                                          # kept
    assert "ocr_rules" not in ocr_client.read_json(meta_key)              # not recorded as checked


class _ServedStub(StubEngine):
    """A served MinerU as the OCR stage sees one (``ServedMinerUEngine``): it says
    its version and records its tier."""

    def __init__(self, tier: str):
        super().__init__()
        self.tier = tier

    def environment(self) -> mv.OcrEnvironment:
        return mv.OcrEnvironment("4.0.12", "cuda")

    def meta(self) -> dict:
        return {"mineru_backend": "api", "mineru_tier": self.tier}


def test_a_reading_at_another_tier_of_the_served_minerU_is_read_again(ocr_client, monkeypatch):
    """The tier is part of the pin: another tier reads a page with other models,
    so a document read at it is another distribution, not an unchanged one."""
    # The in-process engine's readings, from before every page was read by OCR:
    # held to version and tier, as then (test_ocr_every_page holds the setting).
    monkeypatch.setattr("common.config.ocr_every_page", lambda: False)
    _seed_raw_document(ocr_client)
    run_mineru.process_document("policy", "policy_0001", ocr_client, _ServedStub("flash"))
    assert run_mineru.find_unprocessed(ocr_client, "policy", mineru_version="4.0.12") == ["policy_0001"]
    assert run_mineru.find_unprocessed(ocr_client, "policy", mineru_version="4.0.12", mineru_tier="flash") == []

    pinned = _ServedStub("basic")                                          # the pinned tier
    meta = run_mineru.process_document("policy", "policy_0001", ocr_client, pinned)
    assert meta["_reprocessed"] is True and meta["mineru_tier"] == "basic" and len(pinned.calls) == 1
    assert run_mineru.find_unprocessed(ocr_client, "policy", mineru_version="4.0.12") == []
    run_mineru.process_document("policy", "policy_0001", ocr_client, pinned)
    assert len(pinned.calls) == 1                                          # the same tier: skipped
