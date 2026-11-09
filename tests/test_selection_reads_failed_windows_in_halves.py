"""Selection reads a failed window again in halves, as serving and the gate do (2026-11-10).

v3 looped on 5 of 242 windows of its own training documents (2026-11-09), and those windows held about half the
values it missed. The gate reads through serving, which reads such a window again as two halves; selection scored
it as empty, so a checkpoint could win or lose on where it happened to loop.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.schema_sections import group_names
from evaluation.validation_generation import (
    ValidationGeneration,
    generate_validation,
    generation_record,
    merged_windows,
    messages_for_pages,
    restored_generation,
)
from inference_core.input_builder import build_messages, build_training_row, shown_pages
from inference_core.model_runner import EchoBackend, Generation, load_model

LOB = "gl"
GROUP = group_names(LOB)[0]
PAGES = [f"processed/default/policy/policy_0001/page_{n}.png" for n in range(1, 5)]
TEXTS = [f"Page {n} text" for n in range(1, 5)]


def _row(golden: dict | None = None) -> dict:
    row = build_training_row("policy", "policy_0001", PAGES, TEXTS, "ocr_plus_image", json.dumps(golden or {}),
                             lob=LOB, sections=GROUP, split="val", page_numbers=[1, 2, 3, 4], total_pages=4)
    row.update(window_pages=[1, 2, 3, 4], window_index=0, window_leads=False, task="policy_declarations")
    return row


def test_a_half_of_a_window_prompt_is_the_prompt_serving_builds_for_that_half():
    whole = build_messages("policy", PAGES, TEXTS, "ocr_plus_image", lob=LOB, sections=GROUP,
                           page_numbers=[1, 2, 3, 4], total_pages=4).messages
    for pages in ([1, 2], [3, 4]):
        served = build_messages("policy", PAGES[pages[0] - 1:pages[-1]], TEXTS[pages[0] - 1:pages[-1]],
                                "ocr_plus_image", lob=LOB, sections=GROUP, page_numbers=pages,
                                total_pages=4).messages
        assert messages_for_pages(whole, pages) == served
    assert messages_for_pages(whole, [5]) is None                     # a page the prompt does not hold


class _LoopsOnTheWholeWindow(EchoBackend):
    """Loops to the token limit on a four-page window; answers its halves."""

    def generate(self, messages, config, adapter=None):
        self.calls.append(sorted(shown_pages(messages)))
        if len(shown_pages(messages)) > 2:
            return Generation(text='{"policy": {"policy_number": {"v": "P-1", "p": [1]}, "pol', tokens=["{"],
                              token_logprobs=[-0.1], finish_reason="length")
        return Generation(text="{}", tokens=["{}"], token_logprobs=[-0.01])


@pytest.fixture
def model():
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    return load_model("base", client, backend_impl=_LoopsOnTheWholeWindow())


def test_a_window_cut_off_at_the_limit_is_read_again_in_halves(model):
    stats: dict = {}
    (window,) = generate_validation([_row()], model, stats=stats)
    assert window.cut_off and window.failure_kind == "output"
    assert [h.row["window_pages"] for h in window.halves] == [[1, 2], [3, 4]]
    assert [h.row["window_half"] for h in window.halves] == ["0", "1"]
    assert all(h.extraction == {} and not h.error for h in window.halves)
    assert model.backend.calls == [[1, 2, 3, 4], [1, 2], [3, 4]] and stats["halves_read"] == 2


def test_the_halves_travel_in_the_saved_record(model):
    (window,) = generate_validation([_row()], model)
    record = json.loads(json.dumps(generation_record(window)))
    restored = restored_generation(_row(), record)
    assert restored.cut_off and [h.row["window_pages"] for h in restored.halves] == [[1, 2], [3, 4]]
    assert all(h.golden == {} for h in restored.halves)


def test_the_document_merge_takes_the_halves_answers_in_the_windows_place():
    row = _row()
    value = {"raw": "P-1", "parsed": "P-1", "page_ref": [3]}
    halves = [ValidationGeneration(row={**row, "window_pages": [1, 2], "window_half": "0"}, golden={}, extraction={}),
              ValidationGeneration(row={**row, "window_pages": [3, 4], "window_half": "1"}, golden={},
                                   extraction={"policy": {"policy_number": value}})]
    window = ValidationGeneration(row=row, golden={"policy": {"policy_number": value}}, extraction={},
                                  error="stopped at the token limit", failure_kind="output", cut_off=True,
                                  halves=halves)
    (document,) = merged_windows([window])
    assert document.answer["policy"]["policy_number"]["parsed"] == "P-1"
    (without,) = merged_windows([ValidationGeneration(row=row, golden=window.golden, extraction={},
                                                      failure_kind="output", cut_off=True)])
    assert not (without.answer.get("policy") or {}).get("policy_number")
