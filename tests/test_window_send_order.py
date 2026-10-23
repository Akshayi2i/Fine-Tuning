"""A policy's windows go to the model longest answers first, in calls of up to
MAX_WINDOWS_PER_BATCH, and come back in planned order.

The sending order is a speed choice only. What each window is asked, and what
the merge is given, must not change with it: a call ends when its longest
answer does, so a long answer started first runs beside the short ones instead
of keeping the GPU waiting on it alone at the end of the call.
"""

from __future__ import annotations

import json
import re

import pytest

import serving.pipeline as pipeline
import serving.policy_merge as policy_merge
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.fit_calibration import CalibrationParams
from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
from inference_core.model_runner import EchoBackend, load_model
from serving.doc_type_classifier import StaticClassifier
from serving.pipeline import ExtractionRequest, _batch_in_halves, _send_order, extract

#: A self-contained line, as tests/test_policy_windows.py reads.
LOB = "property"
TOTAL = 30
TEXTS = (["Common Policy Declarations. Named Insured: Rivera Fabrication LLC. Policy Number WC-1."]
         + ["Standard conditions and definitions."] * (TOTAL - 1))


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _pages(messages) -> list[int]:
    return [int(n) for n in re.findall(r"<page (\d+) of \d+>", json.dumps(messages))]


class PageEcho(EchoBackend):
    """Answers every window with the pages it was shown, and records each call:
    the pages of its windows and their answer limits, in the order sent."""

    def __init__(self) -> None:
        super().__init__()
        self.sent: list[list[tuple[list[int], int]]] = []

    def generate_batch(self, messages_list, configs, adapter=None):
        self.sent.append([(_pages(m), c.max_new_tokens) for m, c in zip(messages_list, configs, strict=True)])
        return super().generate_batch(messages_list, configs, adapter=adapter)

    def generate(self, messages, config, adapter=None):
        pages = _pages(messages)
        shown = "-".join(map(str, pages))
        self.response = json.dumps(
            {"policy": {"policy_number": {"raw": shown, "parsed": shown, "page_ref": pages}}})
        return super().generate(messages, config, adapter)


def _extract(client, backend) -> None:
    extract(
        ExtractionRequest(
            source_id="policy_0001",
            image_paths=[f"processed/default/policy/policy_0001/page_{p}.png" for p in range(1, TOTAL + 1)],
            page_texts={p: t for p, t in enumerate(TEXTS, start=1)},
            known_doc_type="policy", known_lob=LOB,
        ),
        load_model("base", client, backend_impl=backend), StaticClassifier("policy"),
        CalibrationParams(method="temperature", doc_type="policy", model_version="v1", temperature=1.0),
        strict_schema=False,
    )


def test_the_longest_answers_are_sent_first_and_otherwise_as_planned():
    pending = [("decl", [1, 2, 3, 4, 5]), ("arrays", [1, 2, 3]), ("arrays", [3, 4]),
               ("lineblk", [1, 2, 3, 4, 5]), ("decl", [6, 7])]
    # A schedule reserves 12,288 answer tokens and the declarations 4,096; then
    # more pages first, and equal windows keep their planned order.
    assert _send_order(pending, "policy") == [3, 1, 2, 0, 4]


def test_a_policy_goes_in_one_call_longest_first_and_each_answer_keeps_its_window(client, monkeypatch):
    merged: list[tuple[str, list[int], dict]] = []
    real_merge = policy_merge.merge_policy_windows

    def capture(windows, lob=None):
        merged.extend((w.group, list(w.pages), json.loads(json.dumps(w.extraction))) for w in windows)
        return real_merge(windows, lob=lob)

    monkeypatch.setattr(policy_merge, "merge_policy_windows", capture)
    backend = PageEcho()
    _extract(client, backend)

    planned = [(plan.group, list(plan.pages))
               for plan in plan_windows(LOB, *pages_to_read(TEXTS, TOTAL), TEXTS)]
    assert 1 < len(planned) <= pipeline.MAX_WINDOWS_PER_BATCH
    assert len(backend.sent) == 1                                          # one call
    caps = [cap for _pages, cap in backend.sent[0]]
    assert caps == sorted(caps, reverse=True) and caps[0] > caps[-1]       # schedules first
    assert sorted(pages for pages, _cap in backend.sent[0]) == sorted(pages for _g, pages in planned)
    # Read back in planned order, each answer with the window it was asked of.
    assert [(group, pages) for group, pages, _answer in merged] == planned
    assert all(answer["policy"]["policy_number"]["page_ref"] == pages for _g, pages, answer in merged)


def test_a_refused_call_is_read_again_in_halves_until_the_bad_window_is_alone(monkeypatch):
    calls: list[list[str]] = []

    def generate_batch(model, requests, adapter=None):
        calls.append(list(requests))
        if "bad" in requests:
            return [RuntimeError("the decoder prompt is longer than the model length")] * len(requests)
        return [f"ok:{r}" for r in requests]

    monkeypatch.setattr(pipeline, "generate_batch", generate_batch)
    requests = [f"w{i}" for i in range(16)]
    requests[5] = "bad"
    results, times = _batch_in_halves(object(), requests, None, "policy_0001")

    assert [isinstance(r, Exception) for r in results] == [i == 5 for i in range(16)]
    assert results[0] == "ok:w0" and results[15] == "ok:w15"               # in order
    assert len(calls) == len(times) == 9                                   # 1 + 2 log2(16), not 1 + 16
    assert ["bad"] in calls


def test_windows_failing_each_on_their_own_are_not_read_again(monkeypatch):
    """Different errors are the windows', not a refused call: the round's own
    retry splits each window's pages, and re-sending the call would only repeat it."""
    calls: list[list[str]] = []

    def generate_batch(model, requests, adapter=None):
        calls.append(list(requests))
        return [RuntimeError(f"failed: {r}") for r in requests]

    monkeypatch.setattr(pipeline, "generate_batch", generate_batch)
    results, times = _batch_in_halves(object(), ["a", "b", "c"], None, "policy_0001")
    assert len(calls) == len(times) == 1 and all(isinstance(r, Exception) for r in results)
