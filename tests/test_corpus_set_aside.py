"""Why a document did not reach the corpus is written where the corpus is described, and a vLLM call says where its time went."""

from __future__ import annotations

import logging

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from inference_core import model_runner
from orchestration.pipeline_dag import run_stages, stages_for
from orchestration.runpod_controller import LocalBackend, RunPodController
from tests.test_orchestration import make_context, seed_corpus


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


@pytest.fixture
def controller() -> RunPodController:
    return RunPodController(backend=LocalBackend(), volume_id="vol-test", git_commit="abc1234")


def test_the_manifest_names_each_document_set_aside_and_why(client, controller, monkeypatch, caplog):
    import data_pipeline.dataset_builder.build_jsonl as build

    seed_corpus(client)
    real = build._refusal
    refused: list[str] = []

    def refuse_one(source_id, rows, cap_report):
        if not refused or source_id == refused[0]:
            refused[:1] = [source_id]
            return "over_budget", f"{source_id}: its rows are over their task budget"
        return real(source_id, rows, cap_report)

    monkeypatch.setattr(build, "_refusal", refuse_one)
    ctx = make_context(client, controller)
    with caplog.at_level(logging.WARNING):
        run_stages(ctx, [s for s in stages_for("finetune") if s.number <= 4], command="finetune")

    manifest = client.read_json(paths.corpus_manifest(ctx.corpus))
    assert [row["source_id"] for row in manifest["set_aside"]] == refused
    assert "over their task budget" in manifest["set_aside"][0]["reason"]
    assert sum(manifest["documents_set_aside"].values()) == 1
    assert any(f"{refused[0]} was set aside" in r.getMessage() for r in caplog.records)


def test_a_corpus_that_kept_everything_says_so(client, controller):
    seed_corpus(client)
    ctx = make_context(client, controller)
    run_stages(ctx, [s for s in stages_for("finetune") if s.number <= 4], command="finetune")
    manifest = client.read_json(paths.corpus_manifest(ctx.corpus))
    assert manifest["set_aside"] == [] and manifest["documents_set_aside"] == {}


def test_a_vllm_call_says_how_long_it_spent_reading_pages_and_in_the_engine(monkeypatch, caplog):
    class Engine:
        def chat(self, requests, params, lora_request=None):
            return [object() for _ in requests]

    backend = object.__new__(model_runner.VLLMBackend)
    monkeypatch.setattr(backend, "_load", lambda _config: Engine(), raising=False)
    monkeypatch.setattr(backend, "_sampling_params", lambda _config: None, raising=False)
    monkeypatch.setattr(backend, "_lora", lambda _adapter: None, raising=False)
    monkeypatch.setattr(backend, "_to_generation", lambda output, config, latency: "ok", raising=False)
    monkeypatch.setattr(model_runner, "to_vllm_messages", lambda m: m)

    with caplog.at_level(logging.INFO, logger="inference_core.model_runner"):
        assert backend.generate_batch([[{"role": "user", "content": "x"}]], [object()]) == ["ok"]
    assert any("reading pages" in r.getMessage() and "in the engine" in r.getMessage() for r in caplog.records)
