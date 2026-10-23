"""A served model's windows are taken as each is answered, and a failed window's
halves go out at once - without changing what the merge is given.

The in-process engine runs a call to its end, so its windows go in rounds: a
window cut off in round 1 is retried only after round 1's slowest window. A
served model answers each window on its own (``can_submit``), and the pipeline
uses that: the halves of a failed window are sent the moment it fails. The
order the merge sees stays the rounds' order, whichever window finished first.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import serving.policy_merge as policy_merge
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.fit_calibration import CalibrationParams
from inference_core.model_runner import Generation, ModelBackend, load_model
from serving.doc_type_classifier import StaticClassifier
from serving.pipeline import ExtractionRequest, extract

#: A self-contained line, as tests/test_window_send_order.py reads.
LOB = "property"
TOTAL = 30
TEXTS = (["Common Policy Declarations. Named Insured: Rivera Fabrication LLC. Policy Number WC-1."]
         + ["Standard conditions and definitions."] * (TOTAL - 1))
#: The merge itself, before any test wraps it.
REAL_MERGE = policy_merge.merge_policy_windows


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _pages(messages) -> list[int]:
    return [int(n) for n in re.findall(r"<page (\d+) of \d+>", json.dumps(messages))]


class PageAnswers(ModelBackend):
    """Answers each window with the pages it was shown; a schedule window of
    more than one page fails, as a dense schedule cut off at its limit does."""

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config, adapter=None) -> Generation:
        pages = _pages(messages)
        if self._is_schedule(config) and len(pages) > 1:
            raise RuntimeError(f"cut off over pages {pages}")
        shown = "-".join(map(str, pages))
        text = json.dumps({"policy": {"policy_number": {"raw": shown, "parsed": shown, "page_ref": pages}}})
        tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
        return Generation(text=text, tokens=tokens, token_logprobs=[-0.01] * len(tokens),
                          generation_fingerprint=config.fingerprint(), prompt_tokens=100 * len(pages))

    @staticmethod
    def _is_schedule(config) -> bool:
        # A schedule's answer limit is the largest (common.config.answer_cap).
        return config.max_new_tokens >= 12288


class AnsweredOneByOne(PageAnswers):
    """The same answers, each on its own thread, finishing in a scrambled order."""

    def __init__(self) -> None:
        self.pool = ThreadPoolExecutor(max_workers=64)
        self.sent: list[list[int]] = []

    def supports_submit(self) -> bool:
        return True

    def submit(self, messages, config, adapter=None):
        pages = _pages(messages)
        self.sent.append(pages)

        def answer():
            time.sleep(((sum(pages) * 7) % 5) / 200)        # finish out of order
            try:
                return self.generate(messages, config, adapter)
            except Exception as exc:  # noqa: BLE001 - returned, as a backend does
                return exc
        return self.pool.submit(answer)


def _extract(client, backend, monkeypatch):
    merged: list[tuple[str, list[int], str]] = []

    def capture(windows, lob=None):
        merged.extend((w.group, list(w.pages), json.dumps(w.extraction)) for w in windows)
        return REAL_MERGE(windows, lob=lob)

    monkeypatch.setattr(policy_merge, "merge_policy_windows", capture)
    result = extract(
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
    return result, merged


def test_windows_answered_one_by_one_merge_exactly_as_rounds_do(client, monkeypatch):
    in_rounds, merged_in_rounds = _extract(client, PageAnswers(), monkeypatch)
    answered, merged_answered = _extract(client, AnsweredOneByOne(), monkeypatch)

    assert merged_answered == merged_in_rounds                          # same windows, same order
    assert answered.extraction == in_rounds.extraction
    assert sorted(answered.review_flags) == sorted(in_rounds.review_flags)
    # Every schedule window was split down to single pages, over several rounds.
    assert {s["outcome"] for s in answered.window_stats} == {"read", "split"}
    assert max(s["attempt"] for s in answered.window_stats) > 1
    # One entry for the whole reading, against one per call in rounds.
    assert len(answered.window_latencies_ms) == 1 < len(in_rounds.window_latencies_ms)
    read = [s for s in answered.window_stats if s["outcome"] == "read"]
    assert all(s["prompt_tokens"] == 100 * len(s["pages"]) and s["answer_tokens"] > 0 for s in read)
    assert all(s["seconds"] is not None for s in read)

    def untimed(stats):
        return [{k: v for k, v in s.items() if k != "seconds"} for s in stats]

    # The same reads, recorded in the same (merge) order.
    assert untimed(answered.window_stats) == untimed(in_rounds.window_stats)


class SlowDeclarations(AnsweredOneByOne):
    """A declarations window that does not finish until the failed schedule
    windows' halves have been sent: in rounds, that would never happen."""

    def __init__(self) -> None:
        super().__init__()
        self.halves_sent = threading.Event()
        self.waited: list[bool] = []

    def submit(self, messages, config, adapter=None):
        pages = _pages(messages)
        if self._is_schedule(config) and len(pages) == 1:
            self.halves_sent.set()
        if not self._is_schedule(config) and 1 in pages:
            self.sent.append(pages)

            def slow():
                self.waited.append(self.halves_sent.wait(timeout=10))
                return self.generate(messages, config, adapter)
            return self.pool.submit(slow)
        return super().submit(messages, config, adapter)


def test_a_failed_windows_halves_are_sent_before_the_slowest_window_ends(client, monkeypatch):
    backend = SlowDeclarations()
    result, _merged = _extract(client, backend, monkeypatch)
    assert backend.waited and all(backend.waited)                        # never waited out
    assert result.schema_valid is not None and not any("window_failed" in f for f in result.review_flags)
