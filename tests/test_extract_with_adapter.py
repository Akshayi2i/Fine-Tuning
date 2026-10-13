"""Extraction with the base model and a LoRA adapter applied per request, not merged.

``extract --model base --adapter DIR`` loads the base weights and applies the
adapter on every request; the serving pipeline runs unchanged. The extraction
command itself (testing/run_extraction.py) reads documents already imported and
OCR'd from the store and writes one JSON per document plus a summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from inference_core.model_runner import (
    EchoBackend,
    ModelRunnerError,
    adapter_label,
    generate,
    generate_batch,
    load_model,
)


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _adapter(tmp_path: Path, rank: int = 64, name: str = "checkpoint-280") -> Path:
    folder = tmp_path / "v1-20261005-120000" / name
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": 2 * rank}), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"x")
    return folder


def test_the_adapter_is_applied_to_every_request(client, tmp_path):
    echo = EchoBackend('{"a": 1}')
    model = load_model("base", client, backend_impl=echo, adapter=str(_adapter(tmp_path)))
    generate(model, [{"role": "user", "content": "x"}])
    generate_batch(model, [([{"role": "user", "content": "y"}], None)])
    assert {call["adapter"] for call in echo.calls} == {model.default_adapter}
    assert model.default_adapter.endswith("checkpoint-280")
    assert model.resolved["kind"] == "base_plus_adapter" and model.resolved.get("merged_model") is None


def test_a_requests_own_adapter_still_wins(client, tmp_path):
    echo = EchoBackend('{"a": 1}')
    model = load_model("base", client, backend_impl=echo, adapter=str(_adapter(tmp_path)))
    generate(model, [{"role": "user", "content": "x"}], adapter="/adapters/lossrun")
    assert echo.calls[-1]["adapter"] == "/adapters/lossrun"


def test_without_an_adapter_nothing_changes(client):
    echo = EchoBackend('{"a": 1}')
    model = load_model("base", client, backend_impl=echo)
    generate(model, [{"role": "user", "content": "x"}])
    assert echo.calls[-1]["adapter"] is None and model.default_adapter is None


def test_results_are_filed_under_the_run_and_checkpoint(client, tmp_path):
    model = load_model("base", client, backend_impl=EchoBackend(), adapter=str(_adapter(tmp_path)))
    assert model.tag == "base+v1-20261005-120000-checkpoint-280"
    assert adapter_label("/x/run-a/checkpoint-1") == "base+run-a-checkpoint-1"
    labelled = load_model("base", client, backend_impl=EchoBackend(), adapter=str(_adapter(tmp_path, name="c2")),
                          label="smoke-adapter")
    assert labelled.tag == "smoke-adapter"


def test_a_folder_that_is_not_an_adapter_is_refused(client, tmp_path):
    with pytest.raises(ModelRunnerError, match="adapter_config.json"):
        load_model("base", client, backend_impl=EchoBackend(), adapter=str(tmp_path))


def test_an_adapter_above_the_engines_rank_is_refused(client, tmp_path):
    with pytest.raises(ModelRunnerError, match="max_lora_rank"):
        load_model("base", client, backend_impl=EchoBackend(), adapter=str(_adapter(tmp_path, rank=128)))


def test_an_adapter_with_a_version_other_than_base_is_refused(client, tmp_path):
    with pytest.raises(ModelRunnerError, match="--model base"):
        load_model("v1", client, backend_impl=EchoBackend(), adapter=str(_adapter(tmp_path)))


def test_an_adapter_named_by_a_blob_prefix_is_copied_down(client, tmp_path, monkeypatch):
    from inference_core import model_runner

    monkeypatch.setattr(model_runner, "ADAPTER_CACHE", tmp_path / "cache")
    prefix = "adapters/personal_lines/v1"
    client.write_json(f"{prefix}/adapter_config.json", {"r": 64})
    client.write_bytes(f"{prefix}/adapter_model.safetensors", b"x")
    model = load_model("base", client, backend_impl=EchoBackend(), adapter=prefix)
    assert Path(model.default_adapter, "adapter_config.json").is_file()


# --------------------------------------------------------------------------
# The extraction command, end to end on an imported document
# --------------------------------------------------------------------------

ANSWER = '{"policy": {"policy_number": {"raw": "HO-1", "parsed": "HO-1", "page_ref": [1]}}}'


def _imported_document(client, source_id="policy_0001", tenant="t1"):
    client.write_json(paths.ocr_meta("policy", source_id, tenant), {"page_count": 2})
    for page in (1, 2):
        client.write_bytes(paths.processed_page("policy", source_id, page, "png", tenant), b"\x89PNG fake")
        client.write_text(paths.processed_page("policy", source_id, page, "md", tenant), f"page {page} HO-1")
    client.write_json(paths.label_metadata("policy", source_id, tenant), {"lob": "homeowners"})
    client.write_json(paths.golden_label("policy", source_id, tenant),
                      {"policy": {"policy_number": {"raw": "HO-1", "parsed": "HO-1", "page_ref": [1]}}})


def test_the_extraction_command_runs_each_document_through_serving_with_the_adapter(client, tmp_path):
    from testing.run_extraction import run_batch

    _imported_document(client)
    echo = EchoBackend(ANSWER)
    model = load_model("base", client, backend_impl=echo, adapter=str(_adapter(tmp_path)))
    summary = run_batch(model, client, ["policy_0001", "policy_9999"], tenant_id="t1",
                        images_root=tmp_path / "pages", root=tmp_path / "out")

    assert summary.documents == 2 and [sid for sid, _ in summary.failed] == ["policy_9999"]
    assert echo.calls and {call["adapter"] for call in echo.calls} == {model.default_adapter}
    result = json.loads((tmp_path / "out" / "results" / model.tag / "policy_0001.json").read_text(encoding="utf-8"))
    extraction = result["extraction"]
    assert extraction["policy"]["policy_number"]["raw"] == "HO-1"
    assert "effective_date" in extraction["policy"]              # every schema key, null where not found
    metrics = json.loads((tmp_path / "out" / "metrics" / model.tag / "policy_0001.metrics.json")
                         .read_text(encoding="utf-8"))
    assert metrics["fields"]["policy.policy_number"]["correct"] is True
    assert (tmp_path / "out" / "metrics" / model.tag / "summary.json").is_file()
    # Every field of the schema, graded against the gold, in the JSON a reader opens.
    every = result["all_fields"]["fields"]
    assert every["policy.policy_number"]["result"] == "correct"
    assert (every["policy.effective_date"]["status"], every["policy.effective_date"]["result"]) == ("null", "empty")
    assert "all_fields" not in metrics
    # Each written output is traced to what produced it; the failed document wrote none.
    registry = json.loads((tmp_path / "out" / "extraction_registry.json").read_text(encoding="utf-8"))
    assert [(e["document"], e["model_version"]) for e in registry["extractions"]] == [("policy_0001", model.tag)]


def test_the_extract_subcommand_passes_the_adapter_through(monkeypatch):
    from orchestration import run as cli
    from testing import run_extraction

    seen = {}
    monkeypatch.setattr(run_extraction, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    args = cli.build_parser().parse_args(["extract", "--model", "base", "--adapter", "/a/checkpoint-1",
                                          "--split", "test", "--corpus", "v1", "--tenant", "t1"])
    cli.run_extract(args)
    argv = seen["argv"]
    assert argv[argv.index("--adapter") + 1] == "/a/checkpoint-1"
    assert argv[argv.index("--split") + 1] == "test" and argv[argv.index("--corpus") + 1] == "v1"
