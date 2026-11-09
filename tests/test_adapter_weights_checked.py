"""An adapter with no weights is refused before the engine starts, and test 2 keeps what it found (2026-11-09).

On the pod, checkpoint-857 held its adapter_config.json and a half-copied adapter_model.safetensors.part: the engine
started, and ten minutes in every row failed alone with "doesn't contain tensors". The Blob copy was skipped whenever
its cache folder held a config, so a copy cut short would have been reused too. And the training-documents score kept
only its totals, so the question it raised (where does v3 go wrong on its own documents?) needed another GPU run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from inference_core.model_runner import EchoBackend, ModelRunnerError, load_model


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _config_only(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "adapter_config.json").write_text(json.dumps({"r": 64}), encoding="utf-8")
    return folder


def test_a_checkpoint_whose_weights_never_finished_copying_is_refused_at_once(client, tmp_path):
    folder = _config_only(tmp_path / "checkpoint-857")
    (folder / "adapter_model.safetensors.part").write_bytes(b"half")
    with pytest.raises(ModelRunnerError, match=r"no adapter weights.*adapter_model\.safetensors\.part"):
        load_model("base", client, backend_impl=EchoBackend(), adapter=str(folder))


def test_an_empty_weights_file_is_no_weights(client, tmp_path):
    folder = _config_only(tmp_path / "checkpoint-1")
    (folder / "adapter_model.safetensors").write_bytes(b"")
    with pytest.raises(ModelRunnerError, match="no adapter weights"):
        load_model("base", client, backend_impl=EchoBackend(), adapter=str(folder))


def test_a_blob_copy_cut_short_is_copied_again(client, tmp_path, monkeypatch):
    from inference_core import model_runner

    monkeypatch.setattr(model_runner, "ADAPTER_CACHE", tmp_path / "cache")
    prefix = "adapters/scope/policy/v3"
    client.write_json(f"{prefix}/adapter_config.json", {"r": 64})
    client.write_bytes(f"{prefix}/adapter_model.safetensors", b"weights")
    _config_only(tmp_path / "cache" / "adapters_scope_policy_v3")       # an earlier copy that stopped after the config
    model = load_model("base", client, backend_impl=EchoBackend(), adapter=prefix)
    assert Path(model.default_adapter, "adapter_model.safetensors").read_bytes() == b"weights"


def test_the_training_documents_score_keeps_its_errors_and_answers(tmp_path):
    """What score_on_training_docs writes beside --out, and that show_eval_errors.py reads its failing fields."""
    from evaluation.validation_generation import ValidationGeneration, score_generations
    from scripts.score_on_training_docs import errors_path, write_details
    from scripts.show_eval_errors import collect

    golden = json.loads((Path(__file__).resolve().parent / "fixtures" / "golden" / "policy_0001.golden.json")
                        .read_text(encoding="utf-8"))
    row = {"source_id": "policy_0003", "doc_type": "policy", "messages": [{"role": "user", "content": "x"}]}
    generations = [ValidationGeneration(row=row, golden=golden, extraction={"carrier": "WRONG"})]
    reports: dict = {}
    score_generations(generations, reports=reports)
    assert set(reports) == {"windows"}                                  # flat rows: no whole-document report

    out = tmp_path / "v3_on_training_docs.json"
    written = write_details(out, generations, reports)
    assert out.with_name("v3_on_training_docs.answers.jsonl") in written
    (answer,) = [json.loads(line) for line in out.with_name("v3_on_training_docs.answers.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert answer["answer"] == {"carrier": "WRONG"} and answer["gold"] == golden
    assert "messages" not in answer["row"] and answer["row"]["source_id"] == "policy_0003"

    windows = json.loads(out.with_name("v3_on_training_docs.window_errors.json").read_text(encoding="utf-8"))
    records, _metrics = collect(windows)
    assert records and all(record["source_id"] == "policy_0003" for record in records)
    assert errors_path(out).name == "v3_on_training_docs.errors.json"


def _cut_off(text: str):
    from types import SimpleNamespace

    return SimpleNamespace(text=text, tokens=[0] * 12288, token_logprobs=[], finish_reason="length",
                           truncated=lambda: True)


def test_an_answer_cut_off_at_the_limit_keeps_its_whole_text():
    """2026-11-09: cut-off answers ran to 29,000+ characters against golds of 199-16,503, and only their last 60
    characters were kept - not enough to tell a loop from a long list of real values."""
    from evaluation.validation_generation import ValidationGeneration, _finish

    text = '{"additional_fields": [' + ", ".join(['{"label": "Debris Removal Limit", "value": {"v": "1"}}'] * 50)
    entry = ValidationGeneration(row={"source_id": "policy_0003"}, golden={})
    _finish(entry, _cut_off(text))
    assert entry.error and "token limit" in entry.error and entry.failure_kind == "output"
    assert entry.unparsed_text == text and not entry.extraction      # scored as an empty answer


def test_the_repeat_count_tells_a_loop_from_a_long_list(tmp_path):
    from evaluation.validation_generation import ValidationGeneration
    from scripts.score_on_training_docs import runaway, write_details

    loop = '{"additional_fields": [' + ", ".join(['{"label": "SF-2 (Cancellation)"}'] * 40)
    listing = '{"additional_fields": [' + ", ".join(f'{{"label": "Limit {i}"}}' for i in range(40))
    rows = [ValidationGeneration(row={"source_id": source, "task": "policy_schedule", "window_pages": [1, 2]},
                                 golden={"coverages": []}, error="stopped at the token limit", unparsed_text=text)
            for source, text in (("policy_0018", loop), ("policy_0003", listing))]
    lines = runaway([*rows, ValidationGeneration(row={"source_id": "policy_0008"}, golden={}, extraction={})])
    assert len(lines) == 2
    assert "policy_0018" in lines[0] and "40 additional_fields label(s), 1 different" in lines[0]
    assert "40 additional_fields label(s), 40 different" in lines[1]

    out = tmp_path / "result.json"
    write_details(out, rows, {})
    saved = [json.loads(line) for line in out.with_name("result.answers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry["unparsed_text"] for entry in saved] == [loop, listing]
