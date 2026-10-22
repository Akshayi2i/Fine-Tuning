"""Windows sized by their pages' own text, and values that record no page.

Every page is read (decision 2026-10-19), so the declarations group reads policy
wording too - and a window's page count is sized for a typical page. A page of
dense wording costs twice that: each window is split by what its pages' text
actually costs (decision 2026-10-22), the same way in training and serving.
"""

from __future__ import annotations

import json
import random
import re

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.fit_calibration import CalibrationParams
from data_pipeline.dataset_builder.build_jsonl import SourceDocument, build_corpus, expand_document
from data_pipeline.dataset_builder.expand_tasks import (
    fit_windows_to_text,
    page_tokens,
    window_page_room,
)
from data_pipeline.dataset_builder.policy_windows import (
    WINDOW_PLAN_VERSION,
    assert_window_plan,
    pages_to_read,
    plan_windows,
)
from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits
from inference_core.model_runner import EchoBackend, load_model
from serving.doc_type_classifier import StaticClassifier
from serving.pipeline import ExtractionRequest, extract

#: A self-contained line, as tests/test_policy_windows.py reads.
LOB = "property"
TOTAL = 30

_WORDS = ["the", "insured", "we", "our", "you", "your", "coverage", "property", "damage",
          "bodily", "injury", "occurrence", "limit", "condition", "exclusion", "section",
          "policy", "period", "loss", "payment", "means", "includes"]


def _wording(chars: int, seed: int) -> str:
    rng = random.Random(seed)
    words: list[str] = []
    while sum(len(w) + 1 for w in words) < chars:
        words.append(rng.choice(_WORDS))
    return " ".join(words)


def _texts(chars: int) -> list[str]:
    """A short declarations block on pages 1-5, then policy-form wording."""
    return (["Common Policy Declarations. Named Insured: Rivera Fabrication LLC. Policy Number WC-1.",
             "Schedule of Locations 1 Toledo"] + ["Premium summary"] * 3
            + [_wording(chars, page) for page in range(6, TOTAL + 1)])


def _fv(value, pages):
    return {"raw": value, "parsed": value, "confidence": {"score": 1.0, "source": "audit"},
            "page_ref": pages, "flagged": False}


LABEL = {
    "carrier": {"company_name": _fv("Granite Mutual", [1])},
    "named_insured": {"primary_name": _fv("Rivera Fabrication LLC", [1])},
    "policy": {"policy_number": _fv("WC-1", [1])},
    "locations": [{"location_number": _fv("1", [2]), "address": {"city": _fv("Toledo", [2])}}],
}


def _document(source_id: str, texts: list[str], label=LABEL) -> SourceDocument:
    return SourceDocument(
        source_id=source_id, doc_type="policy", golden_label=label, ocr_pages=list(texts),
        image_paths=[f"processed/default/policy/{source_id}/page_{p}.png" for p in range(1, len(texts) + 1)],
        lob=LOB, tenant_id="default",
    )


def test_dense_wording_splits_a_window_rather_than_setting_the_document_aside():
    """5,500 characters a page of wording: a declarations window of them needed
    ~34,400 tokens against a 32,768 cap, and all three documents were set aside."""
    texts = _texts(5500)
    docs = [_document(f"policy_{i:04d}", texts) for i in range(1, 4)]
    groups = {"policy": [GroupRecord(group_id=d.source_id, doc_type="policy", source_ids=[d.source_id])
                         for d in docs]}
    built = build_corpus(docs, assign_group_splits(groups, seed=42))
    assert not built.skipped
    assert built.all_rows

    pages, declarations = pages_to_read(texts, TOTAL)
    plans = plan_windows(LOB, pages, declarations, texts)
    for group in {plan.group for plan in plans}:
        windows = [plan.pages for plan in plans if plan.group == group]
        assert {p for w in windows for p in w} == set(pages), group
        room = window_page_room("policy", None, LOB, group)
        for window in windows:
            cost = sum(page_tokens(texts[p - 1], p, TOTAL, group) for p in window)
            assert cost <= room or len(window) == 1, (group, window, cost, room)


def test_a_typical_page_keeps_the_windows_its_page_count_plans():
    """A window that fits is left as planned: only a text-heavy one is split."""
    texts = ["Declarations"] + ["Standard conditions and definitions."] * 14
    pages, declarations = pages_to_read(texts, len(texts))
    assert plan_windows(LOB, pages, declarations, texts) == plan_windows(LOB, pages, declarations)


def test_a_window_is_split_evenly_and_a_page_too_big_to_fit_reads_alone():
    cost = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 50}.__getitem__
    # Filled in order this is 4 + 2; the even split of the same count fits too.
    assert fit_windows_to_text([[1, 2, 3, 4, 5, 6]], cost, 40) == [[1, 2, 3], [4, 5, 6]]
    assert fit_windows_to_text([[6, 7]], cost, 40) == [[6], [7]]
    assert fit_windows_to_text([[1, 2], [3, 4]], cost, 40) == [[1, 2], [3, 4]]


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _served_windows(client, texts: list[str], backend_type=EchoBackend) -> list[list[int]]:
    response = {
        "carrier": {"company_name": {"raw": "Granite Mutual", "parsed": "Granite Mutual", "page_ref": [1]}},
        "named_insured": {"primary_name": {"raw": "Rivera", "parsed": "Rivera", "page_ref": [1]}},
        "policy": {"policy_number": {"raw": "WC-1", "parsed": "WC-1", "page_ref": [1]}},
    }
    backend = backend_type(json.dumps(response))
    model = load_model("base", client, backend_impl=backend)
    extract(
        ExtractionRequest(
            source_id="policy_0001",
            image_paths=[f"processed/default/policy/policy_0001/page_{p}.png"
                         for p in range(1, len(texts) + 1)],
            page_texts={p: t for p, t in enumerate(texts, start=1)},
            known_doc_type="policy", known_lob=LOB,
        ),
        model, StaticClassifier("policy"),
        CalibrationParams(method="temperature", doc_type="policy", model_version="v1", temperature=1.0),
    )
    windows = []
    for call in backend.calls:
        text = " ".join(b.get("text", "") for b in call["messages"][1]["content"] if b.get("type") == "text")
        windows.append([int(n) for n in re.findall(r"<page (\d+) of \d+>", text)])
    return windows


def test_serving_sizes_dense_windows_as_training_did(client):
    """The windows a model is served are the ones it trained on - dense pages too."""
    texts = _texts(7000)
    rows, _ = expand_document(_document("policy_0001", texts), "train", modes=("ocr_plus_image",))
    trained = [r["window_pages"] for r in rows]
    assert len(trained) > len(plan_windows(LOB, *pages_to_read(texts, TOTAL)))
    assert _served_windows(client, texts) == trained


def test_a_long_policy_goes_to_the_model_in_bounded_batches(client, monkeypatch):
    """Some 190 windows for a 200-page policy: in one batch, every page image is
    decoded at once and one refused prompt fails them all."""
    import serving.pipeline as pipeline

    monkeypatch.setattr(pipeline, "MAX_WINDOWS_PER_BATCH", 3)
    sizes: list[int] = []

    class Counting(EchoBackend):
        def generate_batch(self, messages_list, configs, adapter=None):
            sizes.append(len(messages_list))
            return super().generate_batch(messages_list, configs, adapter=adapter)

    texts = _texts(2000)
    served = _served_windows(client, texts, Counting)
    assert len(served) == len(plan_windows(LOB, *pages_to_read(texts, TOTAL), texts)) > 3
    assert sizes and max(sizes) <= 3 and sum(sizes) == len(served)


def test_a_value_with_no_page_is_taught_in_the_window_the_declarations_lead():
    """The label rules could not place it on a page: a policy-level value, which
    the declarations print. Read over several windows, the declarations group
    had left it out of all of them."""
    texts = [("Common Policy Declarations. " if p == 1 else "Standard conditions and definitions. ")
             + "Policy Number WC-12345 Named Insured Rivera Fabrication LLC" for p in range(1, 16)]
    label = {
        "carrier": {"company_name": _fv("Granite Mutual", [1])},
        "named_insured": {"primary_name": _fv("Rivera Fabrication LLC", [])},
        "policy": {"policy_number": _fv("WC-12345", [])},
        "locations": [{"location_number": _fv("1", [7]), "address": {"city": _fv("Toledo", [7])}}],
    }
    rows, details = expand_document(_document("p1", texts, label), "train", modes=("ocr_plus_image",))
    decl = [(r["window_pages"], json.loads(r["messages"][-1]["content"]))
            for r in rows if r["sections"] == "decl"]
    assert len(decl) > 1
    first_pages, first = decl[0]
    assert first_pages[0] == 1
    assert first["policy"]["policy_number"]["raw"] == "WC-12345"
    assert first["policy"]["policy_number"]["page_ref"] == []
    assert first["named_insured"]["primary_name"]["raw"] == "Rivera Fabrication LLC"
    assert all("policy_number" not in (target.get("policy") or {}) for _pages, target in decl[1:])
    assert not [d for d in details if "unplaced" in d]


def test_a_corpus_built_with_another_window_plan_is_refused():
    assert_window_plan({"window_plan_version": WINDOW_PLAN_VERSION})
    for manifest in ({}, {"window_plan_version": "1"}):
        with pytest.raises(ValueError, match="Rebuild the corpus and retrain"):
            assert_window_plan(manifest)
