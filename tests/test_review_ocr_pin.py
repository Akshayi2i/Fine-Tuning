"""Review 2026-10-26 #4: the OCR pin is held before a release can be promoted.

A corpus read by another MinerU than the served one (configs/shared/ocr.yaml)
could be trained, gated and auto-promoted, and the Azure worker then refused to
start: an outage. It is now refused at each step - the OCR stage's completion
check, preflight, the corpus build, and promotion (``serving_refusal``) - with
the remedy that reads it through the served MinerU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.config import served_mineru
from orchestration.pipeline_dag import PipelineError
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import make_context, seed_corpus, seeded_policies

PIN = served_mineru()


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def _read_by(client: BlobClient, source_id: str, version: str | None, tier: str | None = None) -> None:
    """Record ``source_id`` as read by MinerU ``version`` (at ``tier``)."""
    key = paths.ocr_meta("policy", source_id)
    meta = {k: v for k, v in client.read_json(key).items() if k not in ("mineru_version", "mineru_tier")}
    if version is not None:
        meta["mineru_version"] = version
    if tier is not None:
        meta["mineru_tier"] = tier
    client.write_json(key, meta)


# --------------------------------------------------------------------------
# The release: promotion refuses what the worker refuses
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("ocr_pin", "refused"), [
    ({"mineru_version": PIN["version"], "mineru_tier": PIN["tier"]}, None),
    # An in-process reading at the pinned version records no tier: held to its version.
    ({"mineru_version": PIN["version"]}, None),
    ({"mineru_version": "3.4.5"}, "MinerU 3.4.5"),
    ({}, "not recorded"),
    ({"mineru_version": PIN["version"], "mineru_tier": "pro"}, "tier pro"),
])
def test_the_ocr_pin_refusal_is_the_workers(ocr_pin, refused):
    from registry_utils.releases import ocr_pin_refusal

    why = ocr_pin_refusal(ocr_pin)
    if refused is None:
        assert why is None
    else:
        assert why and refused in why


def test_a_release_trained_on_another_minerus_text_is_not_promoted(client):
    from registry_utils.releases import ReleaseError, promote_release, serving_refusal
    from tests.test_releases import _release

    _release(client, "v2", ocr_pin={"mineru_version": "3.4.5", "ocr_device": "cuda"})
    assert "MinerU 3.4.5" in serving_refusal(client.read_json(paths.release_bundle("v2")))
    with pytest.raises(ReleaseError, match="serving would refuse it at cold start"):
        promote_release(client, "v2", promoted_by="package v2")
    assert not client.exists(paths.release_index()) or all(
        row["status"] != "promoted" for row in client.read_json(paths.release_index()))


def test_a_release_on_the_served_minerus_text_is_promoted(client):
    from registry_utils.releases import promote_release
    from tests.test_releases import _release

    _release(client, "v2", ocr_pin={"mineru_version": PIN["version"], "mineru_tier": PIN["tier"]})
    assert promote_release(client, "v2", promoted_by="package v2") == []


# --------------------------------------------------------------------------
# The corpus build and the OCR stage
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("readings", "refused"), [
    ({(PIN["version"], PIN["tier"]): 3}, False),
    ({(PIN["version"], None): 3}, False),
    ({("3.4.5", None): 2}, True),
    ({("unknown", None): 1}, True),
    ({(PIN["version"], "pro"): 1}, True),
])
def test_the_corpus_build_holds_every_reading_to_the_pin(readings, refused):
    from orchestration.pipeline_dag import assert_corpus_ocr_pin

    if not refused:
        assert_corpus_ocr_pin(readings)
        return
    with pytest.raises(PipelineError) as err:
        assert_corpus_ocr_pin(readings)
    assert "--mineru-url" in str(err.value) and PIN["version"] in str(err.value)


def test_a_corpus_another_minerus_text_is_in_is_not_built(client, controller):
    from orchestration.pipeline_dag import plan_corpus

    policies = seeded_policies(seed_corpus(client))
    # One of them: two MinerUs, refused with the served pin's remedy.
    _read_by(client, policies[0], "3.4.5")
    with pytest.raises(PipelineError, match="more than one MinerU.*--mineru-url"):
        plan_corpus(make_context(client, controller))
    # All of them, by one MinerU - not the served one.
    for source_id in policies:
        _read_by(client, source_id, "3.4.5")
    with pytest.raises(PipelineError, match=rf"read by another MinerU than the served one.*3\.4\.5 \({len(policies)}\)"):
        plan_corpus(make_context(client, controller))


def test_the_corpus_records_the_tier_and_the_release_carries_it(client, controller):
    from orchestration.pipeline_dag import run_stages, stages_for

    seed_corpus(client)
    report = run_stages(make_context(client, controller), stages_for("finetune"), command="finetune")
    assert report.results[3].name == "dataset_build" and report.results[3].status == "completed"
    manifest = client.read_json(paths.corpus_manifest("v1"))
    assert manifest["mineru_version"] == PIN["version"] and manifest["mineru_tier"] == PIN["tier"]


def test_without_a_served_mineru_named_a_reading_by_another_is_still_unread(client, controller, monkeypatch):
    """MINERU_URL unset (the training pod): the completion check holds each
    reading to the pin, not to nothing - 3.4.5's corpus is not 'preprocessed'."""
    from orchestration.pipeline_dag import _is_preprocessed

    monkeypatch.delenv("MINERU_URL", raising=False)
    seed_corpus(client)
    ctx = make_context(client, controller)
    assert _is_preprocessed(ctx)
    _read_by(client, "policy_0001", "3.4.5")
    ctx.ocr_meta_cache.clear()
    assert not _is_preprocessed(ctx)


def test_the_ocr_stage_refuses_an_installed_mineru_that_is_not_the_pin(monkeypatch):
    from data_pipeline.ocr import mineru_version, run_mineru
    from orchestration import pipeline_dag

    monkeypatch.delenv("MINERU_URL", raising=False)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
    monkeypatch.setattr(mineru_version, "get_mineru_version", lambda: "3.4.5")
    monkeypatch.setattr(run_mineru, "find_unprocessed", lambda raw, dt, tenant, **held: ["d1"])
    ctx = SimpleNamespace(raw=object(), tenant_id="t", doc_types=["policy"], ocr_engine=None)
    with pytest.raises(PipelineError) as err:
        pipeline_dag.stage_preprocessing(ctx)
    message = str(err.value)
    assert "installed here is 3.4.5" in message and "--mineru-url" in message and "--tenant t" in message


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def test_preflight_fails_a_document_another_mineru_read_and_names_the_served_one(client, controller,
                                                                                  monkeypatch):
    from artifact_registry import blob_client
    from orchestration.preflight import check_ocr_done
    from tests.test_orchestration import ingestion_client

    seed_corpus(client)
    monkeypatch.setattr(blob_client, "for_ocr", lambda: ingestion_client(client))
    ctx = make_context(client, controller)

    results = []
    check_ocr_done(ctx, results)
    assert [r.status for r in results] == ["PASS"]

    _read_by(client, "policy_0001", "3.4.5")
    results = []
    check_ocr_done(ctx, results)
    [check] = results
    assert check.status == "FAIL" and "1 not read by the served MinerU" in check.detail
    assert "--all-unprocessed --mineru-url" in check.fix


def test_the_smoke_run_reads_through_the_served_mineru_when_one_is_named(tmp_path, monkeypatch):
    from orchestration.smoke_run import OCR_PYTHON, commands

    def ocr():
        return commands(batch_dir=tmp_path, subset_dir=tmp_path / "s", tenant="smoke", version="v0",
                        check_out=tmp_path / "c")["ocr"]

    monkeypatch.setenv("MINERU_URL", "https://mineru.example.test")
    served = ocr()
    assert served[0] != str(OCR_PYTHON)
    assert served[served.index("--mineru-url") + 1] == "https://mineru.example.test"
    assert served[served.index("--workers") + 1] == "4" and served[served.index("--tenant") + 1] == "smoke"
    monkeypatch.delenv("MINERU_URL")
    assert ocr()[0] == str(OCR_PYTHON)
