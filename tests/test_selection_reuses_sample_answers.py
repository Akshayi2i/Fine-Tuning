"""Checkpoint selection reads a finalist's sample rows once (2026-11-10 review: about 44 minutes of cycle 1's
selection went to generating the 400 sample rows a second time for each finalist's whole pass)."""

from __future__ import annotations

import json

import pytest

from evaluation import checkpoint_eval as C


def _row(n: int, mode: str = "image_only") -> dict:
    return {"source_id": f"policy_{n // 4:04d}", "modality_mode": mode, "sections": "decl", "window_index": n % 4,
            "window_pages": [n % 4 + 1], "doc_type": "policy"}


@pytest.fixture
def generated(monkeypatch):
    """What generate_validation was asked for: one list of rows per call."""
    from evaluation import validation_generation

    calls: list[list[dict]] = []

    def fake(rows, model, *, adapter=None, stats=None, **_):
        calls.append(list(rows))
        if stats is not None:
            stats.update(rows=len(rows), generated=len(rows), calls=1, seconds=1.0, halves_read=0)
        return [{"answer_for": (adapter, row["source_id"], row["window_index"])} for row in rows]

    monkeypatch.setattr(validation_generation, "generate_validation", fake)
    return calls


def test_a_whole_pass_after_the_sample_generates_only_the_rows_the_sample_did_not(generated):
    rows = [_row(n) for n in range(12)]
    sample = rows[:4]
    answers: dict = {}
    first = C._generated_or_reused(sample, None, "/ck/checkpoint-800", {}, answers)
    stats: dict = {}
    whole = C._generated_or_reused(rows, None, "/ck/checkpoint-800", stats, answers)
    assert [len(call) for call in generated] == [4, 8]                  # the 4 sample rows were not asked again
    assert whole[:4] == first and stats["reused"] == 4 and stats["generated"] == 8
    assert [g["answer_for"][2] for g in whole] == [r["window_index"] for r in rows]   # in row order


def test_another_checkpoints_readings_are_never_taken(generated):
    rows = [_row(n) for n in range(4)]
    answers: dict = {}
    C._generated_or_reused(rows, None, "/ck/checkpoint-800", {}, answers)
    again = C._generated_or_reused(rows, None, "/ck/checkpoint-857", {}, answers)
    assert [len(call) for call in generated] == [4, 4]
    assert {g["answer_for"][0] for g in again} == {"/ck/checkpoint-857"}


def test_a_pass_whose_rows_were_all_read_generates_nothing(generated):
    rows = [_row(n) for n in range(4)]
    answers: dict = {}
    C._generated_or_reused(rows, None, "/ck/checkpoint-800", {}, answers)
    stats: dict = {}
    C._generated_or_reused(rows, None, "/ck/checkpoint-800", stats, answers)
    assert len(generated) == 1 and stats == {"rows": 0, "generated": 0, "calls": 0, "seconds": 0.0,
                                              "halves_read": 0, "reused": 4}


def test_without_a_store_every_row_is_generated(generated):
    rows = [_row(n) for n in range(4)]
    C._generated_or_reused(rows, None, "/ck/checkpoint-800", {}, None)
    C._generated_or_reused(rows, None, "/ck/checkpoint-800", {}, None)
    assert [len(call) for call in generated] == [4, 4]


def test_only_the_finalists_readings_are_kept_for_their_whole_pass(generated):
    """Code review 2026-11-10: every checkpoint's sample readings stayed in memory, though only the finalists'
    are ever reused."""
    rows = [_row(n) for n in range(4)]
    scorer = C.generation_scorer(rows, object())
    answers: dict = {}
    scorer.configure(answers=answers)
    for checkpoint in ("/ck/checkpoint-200", "/ck/checkpoint-400", "/ck/checkpoint-600"):
        C._generated_or_reused(rows, None, checkpoint, {}, answers)
    scorer.keep_answers(["/ck/checkpoint-400"])
    assert list(answers) == ["/ck/checkpoint-400"]


def test_the_sample_and_the_whole_pass_share_one_store():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend

    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="main")
    rows = [_row(n, mode) for n in range(200) for mode in ("image_only", "ocr_plus_image")]
    client.write_text("corpus/val.jsonl", "\n".join(json.dumps(r) for r in rows))
    stores = []
    original = C.generation_scorer

    def recording(rows_, model, **kwargs):
        made = original(rows_, model, **kwargs)
        real = made.configure

        def configure(**changes):
            if "answers" in changes:
                stores.append(changes["answers"])
            real(**changes)

        made.configure = configure
        return made

    C.generation_scorer = recording
    try:
        C.vllm_scorer(client=client, val_path="corpus/val.jsonl", model=object(), sample_rows=40, repeat=True)
    finally:
        C.generation_scorer = original
    assert len(stores) == 2 and stores[0] is stores[1]
