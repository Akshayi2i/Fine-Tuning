"""The Azure worker's cold start (azure_service/runtime.py): the promoted release, the
checks serving's endpoint makes, the served model's name and the release's calibrators."""

from __future__ import annotations

import json
import logging

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from azure_service.runtime import (
    BASE_FALLBACK_ROUTE,
    RELEASE_ROUTE,
    JobRefused,
    load_runtime,
    openai_models,
    served_model,
)
from azure_service.settings import Settings
from inference_core.model_runner import EchoBackend
from serving.vllm_entrypoint import ColdStartError

ADAPTER = "adapters/scope/policy/v3/run-1/final"
CALIBRATORS = "releases/default/v3/calibration/bf16/calibrators.json"


@pytest.fixture
def registry() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw", context="serving")


def promote(registry: BlobClient, **changes) -> dict:
    """A release as `package` leaves one since 2026-10-23: a LoRA on the base."""
    from common.config import served_mineru
    from common.prompts import prompt_hash
    from data_pipeline.dataset_builder.policy_windows import window_plan_id
    from registry_utils.query_registry import configured_base_model

    bundle = {
        "release_id": "v3", "status": "promoted", "scope": "policy", "doc_types": ["policy"],
        "lines": ["personal_auto"], "created_at": "2026-10-23T00:00:00+00:00",
        "base_model": configured_base_model(), "adapter": "policy-v3", "served_as": "lora_on_base",
        "serving_formats": {"bf16": ADAPTER}, "calibrators": {"bf16": CALIBRATORS},
        "prompt_hash": prompt_hash(), "window_plan_version": window_plan_id(),
        "ocr_pin": {"mineru_version": served_mineru()["version"]},
        **changes,
    }
    registry.write_json(paths.release_bundle("v3"), bundle)
    registry.write_json(paths.release_index(), [{"release_id": "v3", "status": "promoted", "scope": "policy",
                                                 "doc_types": ["policy"], "created_at": bundle["created_at"]}])
    return bundle


def calibrate(registry: BlobClient) -> None:
    registry.write_json(CALIBRATORS, {
        "calibrators": {"release_id": "v3", "serving_format": "bf16", "calibrators": {}},
        "thresholds": {"release_id": "v3", "serving_format": "bf16", "thresholds": {}},
    })


class Models:
    """vLLM's served names, answered by the echo model; records what was asked for."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def __call__(self, name: str):
        self.asked.append(name)
        return served_model(EchoBackend("{}"), name if name.startswith("policy-") else "base")


def settings(**changes) -> Settings:
    return Settings(**{"tenants": ("default",), **changes})


def test_the_promoted_release_is_asked_of_vllm_by_name_with_its_calibrators(registry):
    promote(registry)
    calibrate(registry)
    models = Models()
    runtime = load_runtime(settings(), registry, models=models, reader=object())
    tenant = runtime.tenant("default")
    assert models.asked == ["policy-v3"] and tenant.release_model.tag == "policy-v3"
    assert tenant.release_runtime.calibrators is not None and tenant.release_runtime.adapter is None
    reading = tenant.reading("personal_auto")
    assert reading.route == RELEASE_ROUTE and reading.release_id == "v3" and reading.calibrators is not None
    assert reading.extract_options({"review_threshold": 0.9})["calibrators"] is reading.calibrators
    assert runtime.describe() == {"default": {"release": "v3", "model": "policy-v3", "calibrated": True,
                                              "base_fallback": False}}
    with pytest.raises(JobRefused, match="ALLOW_BASE_FALLBACK"):
        tenant.reading("homeowners")                               # a line the release did not train on
    with pytest.raises(JobRefused, match="serves tenants"):
        runtime.tenant("acme")


@pytest.mark.parametrize("changes, why", [
    ({"prompt_hash": "0" * 64}, "prompt hash"),
    ({"window_plan_version": "old-plan"}, "window plan"),
    ({"base_model": "qwen3-vl-8b-instruct@0000000"}, "base"),
    ({"ocr_pin": {"mineru_version": "3.4.5"}}, "MinerU 3.4.5"),
    ({"served_as": "merged", "merged_model": "merged/v3"}, "lora_on_base"),
])
def test_a_release_this_code_cannot_serve_refuses_the_cold_start(registry, changes, why):
    promote(registry, **changes)
    models = Models()
    with pytest.raises(ColdStartError, match=why):
        load_runtime(settings(), registry, models=models, reader=object())
    assert models.asked == []                                       # refused before vLLM is asked


def test_with_nothing_promoted_only_a_configured_base_model_reads(registry):
    with pytest.raises(ColdStartError, match="no release is promoted"):
        load_runtime(settings(), registry, models=Models(), reader=object())
    models = Models()
    runtime = load_runtime(settings(allow_base_fallback=True, base_model_name="qwen3-vl-8b"), registry,
                           models=models, reader=object())
    reading = runtime.tenant("default").reading("personal_auto")
    assert models.asked == ["qwen3-vl-8b"] and reading.route == BASE_FALLBACK_ROUTE
    assert reading.model.tag == "base" and reading.release_id is None and reading.calibrators is None


def test_a_release_without_calibrators_is_served_with_raw_confidence_and_said_so(registry, caplog):
    promote(registry)
    with caplog.at_level(logging.WARNING):
        runtime = load_runtime(settings(), registry, models=Models(), reader=object())
    assert runtime.tenant("default").release_runtime.calibrators is None
    assert "no calibrators" in caplog.text


def test_one_backend_per_served_name_with_the_configured_server(monkeypatch):
    import inference_core.openai_backend as openai_backend

    made = []

    class FakeBackend(EchoBackend):
        def __init__(self, url, token, model=None, **options):
            super().__init__("{}")
            made.append((url, token, model, options["images"], options["max_concurrency"]))

    monkeypatch.setattr(openai_backend, "OpenAIChatBackend", FakeBackend)
    models = openai_models(settings(model_url="https://pod-8000.proxy.runpod.net", model_token="t",
                                    base_model_name="qwen3-vl-8b", max_concurrency=64))
    first, again, base = models("policy-v3"), models("policy-v3"), models("qwen3-vl-8b")
    assert first is again and first.tag == "policy-v3" and base.tag == "base"
    assert made == [("https://pod-8000.proxy.runpod.net", "t", "policy-v3", "inline", 64),
                    ("https://pod-8000.proxy.runpod.net", "t", "qwen3-vl-8b", "inline", 64)]


def test_the_release_calibrators_reach_extract_once_read_fields_passes_them_on(tmp_path, monkeypatch):
    """Phase 1 hands the release's calibrators to extract as soon as
    serving.fields_first.read_fields takes ``extract_options``; until then it
    reads with raw confidence and says so in the job's record."""
    import serving.fields_first as fields_first
    from tests.test_azure_service import Service

    original = fields_first.read_fields
    seen = []

    def read_fields(pdf, *, extract_options=None, **kwargs):
        seen.append(extract_options)
        return original(pdf, **kwargs)

    service = Service(tmp_path)
    job_id = service.submit()["job_id"]
    service.runtime.tenants["default"].release_runtime.calibrators = calibrators = object()
    monkeypatch.setattr(fields_first, "read_fields", read_fields)
    service.worker().poll_once()
    assert seen and seen[0]["calibrators"] is calibrators
    assert service.status(job_id)["calibration"] == "release"
    assert json.loads(json.dumps(service.status(job_id)))["status"] == "fields_ready"
