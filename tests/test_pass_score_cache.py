"""A selection that stopped part-way resumes from the checkpoints it had scored."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from evaluation import checkpoint_eval as C
from inference_core.model_runner import EchoBackend, load_model

DOCUMENTS = {"d1": "HO-1", "d2": "HO-2", "d3": "HO-3"}


def env(value):
    return {"raw": str(value), "parsed": value, "page_ref": [1]}


def answer(number):
    return {"policy": {"policy_number": env(number), "effective_date": env("01/01/2026")}}


def row(source, mode="ocr_plus_image"):
    marker = "<page 1 of 1>" if mode == "image_only" else f"<page 1 of 1>\n\nDeclarations Policy {DOCUMENTS[source]}"
    return {"source_id": source, "doc_type": "policy", "lob": ["cyber"], "modality_mode": mode,
            "sections": "decl", "window_pages": [1], "window_index": 0, "synthetic": False,
            "val_half": "calibration",
            "messages": [{"role": "user", "content": [{"type": "image", "image": f"{source}/page_1.png"},
                                                      {"type": "text", "text": marker}]},
                         {"role": "assistant", "content": json.dumps(answer(DOCUMENTS[source]))}]}


def rows(sources=("d1", "d2", "d3")):
    return [row(source) for source in sources]


class CountingBackend(EchoBackend):
    """Answers each row by its document, and counts the calls made."""

    def __init__(self):
        super().__init__()
        self.batch_calls = 0

    def generate(self, messages, config, adapter=None):
        parts = [p for m in messages if isinstance(m.get("content"), list) for p in m["content"]]
        source = next(p["image"] for p in parts if p.get("type") == "image").split("/")[0]
        self.response = json.dumps(answer(DOCUMENTS[source]), separators=(",", ":"))
        return super().generate(messages, config, adapter)

    def generate_batch(self, messages_list, configs, adapter=None):
        self.batch_calls += 1
        return super().generate_batch(messages_list, configs, adapter)


@pytest.fixture
def backend():
    return CountingBackend()


@pytest.fixture
def model(backend):
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    return load_model("base", client, backend_impl=backend)


@pytest.fixture
def checkpoint(tmp_path):
    folder = tmp_path / "checkpoint-10"
    folder.mkdir()
    (folder / "adapter_model.safetensors").write_bytes(b"weights-v1")
    (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    return str(folder)


def test_a_second_run_reuses_the_score_and_generates_nothing(model, backend, checkpoint, tmp_path):
    first = C.generation_scorer(rows(), model, cache_dir=tmp_path / "cache")
    metrics = first(checkpoint)
    made = backend.batch_calls
    assert made > 0 and list((tmp_path / "cache").glob("checkpoint-10-*.json"))

    again = C.generation_scorer(rows(), model, cache_dir=tmp_path / "cache")
    assert again(checkpoint) == metrics
    assert backend.batch_calls == made                       # nothing generated
    assert again.passes[0]["cached"] is True and C.generation_cost(again)["rows"] == 0


def test_other_weights_are_scored_again(model, backend, checkpoint, tmp_path):
    cache = tmp_path / "cache"
    C.generation_scorer(rows(), model, cache_dir=cache)(checkpoint)
    made = backend.batch_calls
    # The same folder retrained: other bytes in the adapter.
    (tmp_path / "checkpoint-10" / "adapter_model.safetensors").write_bytes(b"weights-v2-longer")
    C.generation_scorer(rows(), model, cache_dir=cache)(checkpoint)
    assert backend.batch_calls > made


def test_other_rows_or_other_readings_are_scored_again(model, backend, checkpoint, tmp_path):
    cache = tmp_path / "cache"
    C.generation_scorer(rows(), model, cache_dir=cache)(checkpoint)
    made = backend.batch_calls
    C.generation_scorer(rows(("d1", "d2")), model, cache_dir=cache)(checkpoint)
    assert backend.batch_calls > made
    made = backend.batch_calls
    C.generation_scorer(rows(), model, repeat=True, cache_dir=cache)(checkpoint)
    assert backend.batch_calls > made


def test_a_finalists_pass_restores_where_its_answers_were_saved(model, backend, checkpoint, tmp_path):
    cache, answers = tmp_path / "cache", tmp_path / "answers"
    first = C.generation_scorer(rows(), model, finalists=True, save_dir=answers, cache_dir=cache)
    first(checkpoint)
    made = backend.batch_calls
    assert first.saved[checkpoint]

    again = C.generation_scorer(rows(), model, finalists=True, save_dir=answers, cache_dir=cache)
    again(checkpoint)
    assert backend.batch_calls == made and again.saved[checkpoint] == first.saved[checkpoint]


def test_answers_that_are_gone_mean_the_pass_is_generated_again(model, backend, checkpoint, tmp_path):
    cache, answers = tmp_path / "cache", tmp_path / "answers"
    first = C.generation_scorer(rows(), model, finalists=True, save_dir=answers, cache_dir=cache)
    first(checkpoint)
    made = backend.batch_calls
    for path in first.saved[checkpoint].values():
        Path(path).unlink()

    C.generation_scorer(rows(), model, finalists=True, save_dir=answers, cache_dir=cache)(checkpoint)
    assert backend.batch_calls > made


def test_an_unreadable_or_foreign_cache_file_is_not_a_score(model, backend, checkpoint, tmp_path):
    cache = tmp_path / "cache"
    C.generation_scorer(rows(), model, cache_dir=cache)(checkpoint)
    made = backend.batch_calls
    for file in cache.glob("*.json"):
        file.write_text("{ not json", encoding="utf-8")
    C.generation_scorer(rows(), model, cache_dir=cache)(checkpoint)
    assert backend.batch_calls > made
    for file in cache.glob("*.json"):
        file.write_text(json.dumps({"key": "another", "metrics": {"field_normalized_match": 1.0}}), encoding="utf-8")
    made = backend.batch_calls
    C.generation_scorer(rows(), model, cache_dir=cache)(checkpoint)
    assert backend.batch_calls > made


def test_without_a_cache_directory_nothing_is_saved_or_reused(model, backend, checkpoint, tmp_path):
    C.generation_scorer(rows(), model)(checkpoint)
    made = backend.batch_calls
    C.generation_scorer(rows(), model)(checkpoint)
    assert backend.batch_calls == 2 * made
    assert not list(tmp_path.glob("**/*.tmp"))
