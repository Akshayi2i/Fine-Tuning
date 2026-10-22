"""The second review's findings (evaluation, calibration, orchestration, import, OCR)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend


def _env(value, page=1):
    return {"raw": str(value), "parsed": value, "page_ref": [page]}


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


# 1 - an override file written before the merge still loads
def test_a_classic_auto_override_is_read_as_personal_auto(tmp_path):
    from data_pipeline.ingestion.prepare_bundles import read_lob_overrides

    path = tmp_path / "overrides.csv"
    path.write_text("source_prefix,lob,reason\nHagerty Insurance/personal_auto/,classic_auto,x\n",
                    encoding="utf-8")
    assert read_lob_overrides(path) == [("Hagerty Insurance/personal_auto/", "personal_auto")]


# 2 - the fill records which values it computed
def test_the_fill_records_the_values_it_added():
    from data_pipeline.ingestion.fill_synthetic_labels import fill

    source = {"carrier": {"company_name": _env("Northfield Mutual"), "naic_code": _env("12345")},
              "policy": {"policy_number": _env("HO-1")}}
    twin = {"carrier": {"company_name": _env("Northfield Mutual")},
            "policy": {"policy_number": _env("HO-9")}}
    merged, _stats = fill(source, twin)
    added = merged["fideon:filled"]["paths"]
    assert "carrier.naic_code" in added
    # The twin's own values were read off its page: never recorded as added.
    assert "carrier.company_name" not in added and "policy.policy_number" not in added


def test_added_paths_are_the_merged_labels_own_and_survive_a_second_fill():
    from data_pipeline.dataset_builder.label_verification import _at, _steps
    from data_pipeline.ingestion.fill_synthetic_labels import fill

    source = {"forms_and_endorsements": [
        {"form_number": _env("HO 00 03"), "form_title": _env("Special Form")},
        {"form_number": _env("HO 04 90"), "form_title": _env("Personal Property Replacement")}]}
    twin = {"forms_and_endorsements": [{"form_number": _env("HO 04 90")}]}
    merged, _ = fill(source, twin)
    for path in merged["fideon:filled"]["paths"]:
        assert _at(merged, _steps(path)) is not None, path
    again, _ = fill(source, merged)
    assert set(merged["fideon:filled"]["paths"]) <= set(again["fideon:filled"]["paths"])


# 3 - package knows the corpus its version trained on
def _ctx(client, **over):
    scope = SimpleNamespace(name="personal_lines", doc_types=("policy",),
                            run_id=lambda version: f"personal_lines-{version}")
    base = dict(client=client, corpus_version="", out_version="v1", dry_run=False, tenant_id=None,
                scope=scope, from_blob=False, release_id="release-2026.10.1",
                calibrators={}, thresholds={})
    base.update(over)
    return SimpleNamespace(**base)


def test_package_takes_its_corpus_from_the_run_manifest(client, monkeypatch):
    from orchestration import pipeline_dag
    from registry_utils import query_registry

    manifest = SimpleNamespace(dependencies=SimpleNamespace(corpus_version="corpus/c7"))
    monkeypatch.setattr(query_registry, "get", lambda run_id, c: manifest)
    ctx = _ctx(client)
    pipeline_dag.resolve_corpus_version(ctx)
    assert ctx.corpus_version == "c7"


def test_package_refuses_when_no_corpus_is_known(client, monkeypatch):
    from orchestration import pipeline_dag
    from registry_utils import query_registry

    def missing(run_id, c):
        raise KeyError(run_id)

    monkeypatch.setattr(query_registry, "get", missing)
    with pytest.raises(pipeline_dag.PipelineError, match="--corpus-version"):
        pipeline_dag.resolve_corpus_version(_ctx(client))
    # A corpus named after the version is the old convention, still accepted.
    client.write_json(paths.corpus_manifest("v1", None), {"rows": 1})
    pipeline_dag.resolve_corpus_version(_ctx(client))


# 4 - a version on the disk is staged, whatever this process recorded
def test_a_version_on_the_mount_counts_as_staged(client, tmp_path, monkeypatch):
    from orchestration import pipeline_dag

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    monkeypatch.setattr(paths, "scoped_staging_adapter_dir", lambda scope, version: str(adapter))
    volume = SimpleNamespace(exists=lambda path: False)
    pipeline_dag.assert_staged(_ctx(client, volume=volume))                 # does not raise
    adapter.rmdir()
    with pytest.raises(pipeline_dag.PipelineError, match="not on the staging volume"):
        pipeline_dag.assert_staged(_ctx(client, volume=volume))


def test_clearing_staging_removes_real_files_only_inside_the_staging_root(tmp_path, monkeypatch):
    from orchestration import pipeline_dag

    root = tmp_path / "staging"
    (root / "adapters" / "v1").mkdir(parents=True)
    (root / "adapters" / "v1" / "adapter_model.safetensors").write_bytes(b"x")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setattr(paths, "staging_root", lambda: str(root))
    assert pipeline_dag._remove_from_mount(str(root / "adapters" / "v1")) == 1
    assert not (root / "adapters" / "v1").exists()
    assert pipeline_dag._remove_from_mount(str(outside)) == 0 and outside.exists()
    assert pipeline_dag._remove_from_mount(str(root)) == 0 and root.exists()


# 5 - a gate in a new process uses the release's saved calibrators
def test_the_gate_loads_the_releases_calibrators_from_where_calibrate_saved_them(client):
    from calibration.feature_calibrator import CalibratorSet
    from calibration.thresholds import ThresholdSet
    from orchestration.pipeline_dag import release_calibration

    ctx = _ctx(client)
    assert release_calibration(ctx, "bf16") == (None, None)
    client.write_json(paths.release_calibrators(ctx.release_id, "bf16", None), {
        "calibrators": CalibratorSet(release_id=ctx.release_id, serving_format="bf16").as_dict(),
        "thresholds": ThresholdSet(release_id=ctx.release_id, serving_format="bf16").as_dict(),
    })
    calibrators, thresholds = release_calibration(ctx, "bf16")
    assert isinstance(calibrators, CalibratorSet) and isinstance(thresholds, ThresholdSet)
    assert ctx.calibrators["bf16"] is calibrators


# 6 - a two-digit year far in the future is the last century
def test_two_digit_years_resolve_to_the_nearer_century():
    from common.normalize import normalize_date

    assert normalize_date("07/04/85") == "1985-07-04"          # a date of birth
    assert normalize_date("01/01/27") == "2027-01-01"          # an expiration
    assert normalize_date("03/15/2085") == "2085-03-15"        # four digits are taken as written


# 7 - both import paths draw ids from every place one is in use
def test_the_next_id_skips_ids_used_in_any_layer(client):
    from common.ids import next_source_id
    from data_pipeline.ingestion.pull_raw_pdfs import taken_source_ids

    client.write_json(paths.golden_label("policy", "policy_0001", None), {})
    client.write_json(paths.ocr_meta("policy", "policy_0002", None), {})
    taken = taken_source_ids(client, "policy", None)
    assert {"policy_0001", "policy_0002"} <= set(taken)
    assert next_source_id(taken, "policy") == "policy_0003"


# 8 - a flat folder cannot say which type its PDFs are
def test_a_flat_folder_with_several_types_is_refused(tmp_path, client):
    from orchestration.pipeline_dag import PipelineError, stage_ingestion

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    ctx = SimpleNamespace(skip_ingest=False, input_dir=tmp_path, raw=client, tenant_id=None,
                          doc_types=["acord", "policy", "lossrun"])
    with pytest.raises(PipelineError, match="which document type"):
        stage_ingestion(ctx)


# 10 - re-rendering keeps the scanned flag
def test_rerendering_an_ocrd_document_keeps_its_scanned_flag(monkeypatch):
    from artifact_registry.blob_client import for_ocr
    from data_pipeline.ocr import render_only

    client = for_ocr(backend=InMemoryBackend(), container="main", raw_container="raw")
    key = paths.ocr_meta("policy", "policy_0001", None)
    client.write_json(key, {"render_only": False, "is_scanned": True, "resolution_cap_px": 800})
    client.write_json(paths.raw_metadata("policy", "policy_0001", None), {"checksum_sha256": "x"})
    client.write_bytes(paths.raw_pdf("policy", "policy_0001", None), b"%PDF")
    monkeypatch.setattr(render_only, "render_pdf_pages", lambda pdf, cap: [b"png"])
    monkeypatch.setattr(render_only, "current_environment", lambda device, strict=True: SimpleNamespace(
        mineru_version="1.3.12", device="cpu", gpu_name=None))
    meta = render_only.render_document("policy", "policy_0001", client, force=True)
    assert meta["is_scanned"] is True


# 12 - calibration labels pair rows by identifier, not position
def test_rows_are_aligned_to_the_models_order_before_comparing():
    from evaluation.metrics.field_accuracy import flatten_scalars, rows_aligned_to

    gold = {"vehicles": [{"vin": "A", "year": 2019}, {"vin": "B", "year": 2021}, {"vin": "C", "year": 2023}]}
    got = {"vehicles": [{"vin": "B", "year": 2021}, {"vin": "C", "year": 2023}]}   # the first omitted
    expected = flatten_scalars(rows_aligned_to(gold, got))
    assert expected["vehicles[0].year"] == 2021 and expected["vehicles[1].year"] == 2023


def test_a_row_the_label_does_not_hold_pairs_with_nothing():
    from evaluation.metrics.field_accuracy import flatten_scalars, rows_aligned_to

    gold = {"vehicles": [{"vin": "A", "year": 2019}]}
    got = {"vehicles": [{"vin": "Z", "year": 1999}, {"vin": "A", "year": 2019}]}
    expected = flatten_scalars(rows_aligned_to(gold, got))
    # Nothing to compare the invented row with: its values read as wrong.
    assert expected.get("vehicles[0].year") is None and expected["vehicles[1].year"] == 2019


# 13 - the oracle names the reason by the row it found
def test_a_matched_rows_lost_value_is_a_merge_loss_not_an_unmatched_row(monkeypatch):
    from evaluation import windowing_ceiling
    from serving import policy_merge

    label = {"policy": {"policy_number": _env("PA-1")},
             "auto": {"vehicles": [{"vin": _env("1HGCM82633A004352"), "year": _env(2019)}]}}

    def drop_year(windows, lob=None):
        merged = policy_merge.MergedPolicy()
        merged.extraction = {"auto": {"vehicles": [{"vin": _env("1HGCM82633A004352")}]},
                             "policy": {"policy_number": _env("PA-1")}}
        return merged

    monkeypatch.setattr(policy_merge, "merge_policy_windows", drop_year)
    result = windowing_ceiling.oracle("d", label, "commercial_auto", 1, None)
    assert dict(result.lost) == {"merge": 1}


def test_a_cached_eval_report_made_without_calibrators_is_not_reused(client, monkeypatch):
    """The gate reuses a report only when it was made with calibrators (5)."""
    from orchestration import pipeline_dag

    ctx = _ctx(client, serving_model_loader=lambda c, fmt: object(), corpus="v1")
    key = paths.eval_report("v1", scope="personal_lines")
    # Recorded for this tenant, so the missing calibrators are what refuse it.
    client.write_json(key, {"gate_metrics": {"auto_accept_error_rate": 0.0}, "calibrated": False,
                            "eval_set": {"tenant_id": paths._tenant(None)}})
    monkeypatch.setattr(pipeline_dag, "release_calibration", lambda c, fmt: ("cal", "thr"))
    rescored = {}

    def evaluate_version(client, model, **kwargs):
        rescored.update(kwargs)
        return {"gate_metrics": {"auto_accept_error_rate": 0.012}}

    import evaluation.golden_eval as golden_eval
    import inference_core.model_runner as model_runner

    monkeypatch.setattr(golden_eval, "evaluate_version", evaluate_version)
    monkeypatch.setattr(model_runner, "release_model", lambda model: None)
    monkeypatch.setattr(pipeline_dag, "assert_eval_set_disjoint", lambda *a, **k: None, raising=False)
    metrics = pipeline_dag.eval_report_metrics(ctx)
    assert metrics["auto_accept_error_rate"] == 0.012 and rescored["calibrators"] == "cal"
