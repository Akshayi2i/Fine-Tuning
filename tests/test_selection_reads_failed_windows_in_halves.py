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


# -- review 2026-11-10 ------------------------------------------------------------------------------------------


def test_a_prompt_whose_pages_are_one_string_is_not_split_into_itself():
    """Finding 4: passed through whole, a "half" would read every page again."""
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "<page 1 of 2>\n\ntext\n\n<page 2 of 2>\n\ntext"}]
    assert messages_for_pages(messages, [1]) is None


class _FailsTheRequestOfTheWholeWindow(_LoopsOnTheWholeWindow):
    """The four-page window's request fails outright; its halves answer."""

    def generate(self, messages, config, adapter=None):
        if len(shown_pages(messages)) > 2:
            self.calls.append(sorted(shown_pages(messages)))
            raise RuntimeError("the engine refused the request")
        return super().generate(messages, config, adapter)


def test_a_window_whose_request_failed_is_read_again_in_halves_as_serving_does():
    """Finding 3: serving reads any failed window again in halves, a failed request included."""
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = load_model("base", client, backend_impl=_FailsTheRequestOfTheWholeWindow())
    (window,) = generate_validation([_row()], model)
    assert window.failure_kind == "setup" and [h.row["window_pages"] for h in window.halves] == [[1, 2], [3, 4]]


def test_an_answer_with_a_section_in_the_wrong_shape_is_unusable_and_read_again():
    """Finding 3: serving refuses a section given in another JSON type than its schema's."""
    from evaluation.validation_generation import _wrong_section_shape

    row = _row()
    assert _wrong_section_shape(row, {"policy": "WC-1"})
    assert _wrong_section_shape(row, {"policy": {}}) is None


def test_a_batch_of_halves_that_fails_together_is_retried_one_by_one(monkeypatch):
    """Finding 6: one bad page image refuses a whole batched call; the halves loop now retries it as the main loop."""
    from evaluation import validation_generation
    from inference_core import model_runner

    calls = []

    def generate_batch(model, requests, adapter=None):
        calls.append(len(requests))
        return [RuntimeError("one image could not be read")] * len(requests)

    def generate(model, messages, adapter=None, json_schema=None, max_new_tokens=None):
        calls.append(1)
        return Generation(text="{}", tokens=["{}"], token_logprobs=[-0.01])

    monkeypatch.setattr(model_runner, "generate_batch", generate_batch)
    monkeypatch.setattr(model_runner, "generate", generate)
    results = validation_generation._call_with_retry(object(), [([], None, 10)] * 4, None)
    assert [type(r).__name__ for r in results] == ["Generation"] * 4 and calls == [4, 2, 1, 1, 2, 1, 1]


def test_the_selection_report_shows_how_many_halves_each_checkpoint_needed():
    """Finding 7: how often a checkpoint looped, which its score alone does not show."""
    from types import SimpleNamespace

    from evaluation.checkpoint_eval import generation_cost

    scorer = SimpleNamespace(passes=[
        {"checkpoint": "ck-1", "generated": 10, "seconds": 5, "halves_read": 4, "repeat": {"halves_read": 2}},
        {"checkpoint": "ck-2", "generated": 10, "seconds": 5, "halves_read": 0}])
    cost = generation_cost(scorer)
    assert cost["halves_read"] == 6 and cost["halves_read_by_checkpoint"] == {"ck-1": 6}
