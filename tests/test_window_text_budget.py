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
    _home_window,
    assert_window_plan,
    pages_to_read,
    plan_windows,
    window_plan_id,
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
    # Sent longest answers first (serving.pipeline._send_order): the same
    # windows, in another order (tests/test_window_send_order.py).
    assert sorted(_served_windows(client, texts)) == sorted(trained)


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


@pytest.mark.format_profile("standard")
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
    assert_window_plan({"window_plan_version": window_plan_id()})
    for manifest in ({}, {"window_plan_version": "1"}, {"window_plan_version": "3"}):
        with pytest.raises(ValueError, match="Rebuild the corpus and retrain"):
            assert_window_plan(manifest)


def test_the_window_plan_follows_the_budgets_the_planner_reads(monkeypatch):
    """Raising the declarations answer from 4,096 to 6,144 tokens turned four
    declarations windows into seven with the version and prompt hash unchanged."""
    import common.config as config
    from common.schema_sections import task_for

    before = window_plan_id()
    real = config.sequence_for_task

    def larger_answer(task, doc_type=None):
        budget = dict(real(task, doc_type))
        if task == task_for("decl"):
            budget["max_output_tokens"] = int(budget["max_output_tokens"]) + 2048
        return budget

    monkeypatch.setattr(config, "sequence_for_task", larger_answer)
    assert window_plan_id() != before


def test_serving_refuses_a_policy_release_cut_with_another_window_plan():
    from serving.release_router import ServedRelease, ServingPlan
    from serving.vllm_entrypoint import ColdStartError, assert_window_plan_served

    def plan(doc_types, version):
        return ServingPlan(ServedRelease("v3", "base@abc1234", "merged", doc_types=tuple(doc_types),
                                         window_plan_version=version))

    assert_window_plan_served(plan(["policy"], window_plan_id()))
    assert_window_plan_served(plan(["acord"], None))                   # serves no policy
    assert_window_plan_served(ServingPlan())                           # nothing promoted
    for stale in (plan(["policy"], "2"), plan([], None)):              # [] is every type
        with pytest.raises(ColdStartError, match="window plan"):
            assert_window_plan_served(stale)


# --------------------------------------------------------------------------
# A split keeps the rules a planned window keeps
# --------------------------------------------------------------------------


DECL_PAGE = 9


def _notices_first(total=20):
    """Dense notices and a policy jacket first, the declarations on page 9."""
    return [("Policy Declarations. Named Insured: Rivera Fabrication LLC. Carrier Granite Mutual"
             if p == DECL_PAGE else _wording(5500, p)) for p in range(1, total + 1)]


def _taught(rows, key):
    """The window pages of the declarations rows whose target holds policy.``key``."""
    return [r["window_pages"] for r in rows if r["sections"] == "decl"
            and key in (json.loads(r["messages"][-1]["content"]).get("policy") or {})]


@pytest.mark.format_profile("standard")
def test_the_declarations_led_window_is_the_one_holding_the_declarations_page():
    assert _home_window([[1, 2, 3], [4, 5, 9], [6, 7, 8]], [1, 2, 3, 4, 5, 9], 9) == 1
    assert _home_window([[1, 2, 3], [4, 5]], [1, 2, 3, 4, 5], 4) == 1
    assert _home_window([[1, 2, 3], [4, 5]], [1, 2, 3, 4, 5], None) == 0
    assert _home_window([[1, 2], [3, 4]], [], None) is None

    texts = _notices_first()
    pages, declarations = pages_to_read(texts, len(texts))
    assert declarations == DECL_PAGE
    decl = [plan for plan in plan_windows(LOB, pages, declarations, texts) if plan.group == "decl"]
    assert DECL_PAGE not in decl[0].pages                               # the lead was split
    (home,) = [plan for plan in decl if plan.leads]
    assert DECL_PAGE in home.pages and all(plan.homed for plan in decl)


def test_a_value_with_no_page_is_taught_where_the_declarations_page_is_read():
    """Pages [1, 2, 3] were taught a policy number the declarations print on
    page 9, and the window showing page 9 was taught to leave it out."""
    texts = _notices_first()
    label = {
        "carrier": {"company_name": _fv("Granite Mutual", [DECL_PAGE])},
        "named_insured": {"primary_name": _fv("Rivera Fabrication LLC", [DECL_PAGE])},
        "policy": {"policy_number": _fv("WC-99999", [])},              # in no page's text
        "locations": [{"location_number": _fv("1", [12]), "address": {"city": _fv("Toledo", [12])}}],
    }
    rows, details = expand_document(_document("p1", texts, label), "train", modes=("ocr_plus_image",))
    taught = _taught(rows, "policy_number")
    assert len(taught) == 1 and DECL_PAGE in taught[0]
    assert not [d for d in details if "unplaced" in d]


def test_a_declarations_value_printed_only_on_a_later_page_is_taught_there():
    texts = ["Standard conditions and definitions."] * 15
    texts[0] = "Common Policy Declarations. Named Insured Rivera Fabrication LLC"
    texts[11] = "Endorsement. This endorsement changes Policy Number WC-77881."
    label = {
        "carrier": {"company_name": _fv("Granite Mutual", [1])},
        "named_insured": {"primary_name": _fv("Rivera Fabrication LLC", [1])},
        "policy": {"policy_number": _fv("WC-77881", [])},
        "locations": [{"location_number": _fv("1", [7]), "address": {"city": _fv("Toledo", [7])}}],
    }
    rows, _ = expand_document(_document("p1", texts, label), "train", modes=("ocr_plus_image",))
    taught = _taught(rows, "policy_number")
    assert taught and all(12 in pages for pages in taught)


def test_a_value_no_page_prints_does_not_cost_the_declarations_window():
    texts = (["Common Policy Declarations. Named Insured Rivera Fabrication LLC"]
             + ["Standard conditions and definitions."] * 14)
    label = {**LABEL, "policy": {"policy_number": _fv("WC-1", [1]), "policy_term_months": _fv("12", [])}}
    document = _document("p1", texts, label)
    document.unprinted_values = ["policy.policy_term_months"]
    rows, details = expand_document(document, "train", modes=("ocr_plus_image",))
    decl = [r for r in rows if r["sections"] == "decl"]
    assert decl and decl[0]["window_pages"][0] == 1
    assert not _taught(rows, "policy_term_months")
    assert not [d for d in details if "left out" in d]


def test_a_dense_run_keeps_the_pages_its_windows_share():
    """Fitted window by window, a dense personal-auto schedule became twelve
    calls where seven were planned, five page breaks read by no window."""
    cost = dict.fromkeys(range(1, 8), 10).__getitem__
    assert fit_windows_to_text([[1, 2, 3], [3, 4, 5], [5, 6, 7]], cost, 25, overlap=1) == [
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]

    texts = ["Personal Auto Declarations. Named Insured: Rivera."] + [_wording(5500, p) for p in range(2, 15)]
    pages, declarations = pages_to_read(texts, len(texts))
    arrays = [list(plan.pages) for plan in plan_windows("personal_auto", pages, declarations, texts)
              if plan.group == "arrays"]
    assert {p for w in arrays for p in w} == set(pages)
    assert not [w for w in arrays if any(set(w) < set(other) for other in arrays)]
    spanned = {(a, a + 1) for w in arrays for a in w if a + 1 in w}
    assert {(a, a + 1) for a in range(min(pages), max(pages))} <= spanned


def test_one_refused_prompt_fails_only_its_own_window(client):
    """vLLM fails every prompt of a batch when it refuses one: innocent windows
    were split into shapes never trained, or dropped and flagged."""
    texts = (["Common Policy Declarations. Named Insured Rivera Fabrication LLC"]
             + ["Standard conditions and definitions."] * 11)

    class RefusesPageSeven(EchoBackend):
        def generate_batch(self, messages_list, configs, adapter=None):
            if any("<page 7 of" in json.dumps(m) for m in messages_list):
                return [RuntimeError("the decoder prompt is longer than the model length")] * len(messages_list)
            return super().generate_batch(messages_list, configs, adapter=adapter)

    response = {
        "carrier": {"company_name": {"raw": "Granite Mutual", "parsed": "Granite Mutual", "page_ref": [1]}},
        "named_insured": {"primary_name": {"raw": "Rivera", "parsed": "Rivera", "page_ref": [1]}},
        "policy": {"policy_number": {"raw": "WC-1", "parsed": "WC-1", "page_ref": [1]}},
    }
    backend = RefusesPageSeven(json.dumps(response))
    result = extract(
        ExtractionRequest(
            source_id="policy_0001",
            image_paths=[f"processed/default/policy/policy_0001/page_{p}.png" for p in range(1, 13)],
            page_texts={p: t for p, t in enumerate(texts, start=1)},
            known_doc_type="policy", known_lob=LOB,
        ),
        load_model("base", client, backend_impl=backend), StaticClassifier("policy"),
        CalibrationParams(method="temperature", doc_type="policy", model_version="v1", temperature=1.0),
    )
    failed = [flag for flag in result.review_flags if flag.startswith("window_failed")]
    assert failed and all(flag.endswith(":p7") for flag in failed)
    planned = [list(plan.pages) for plan in plan_windows(LOB, *pages_to_read(texts, 12), texts)]
    read = [[int(n) for n in re.findall(r"<page (\d+) of \d+>", json.dumps(call["messages"]))]
            for call in backend.calls]
    innocent = [w for w in planned if 7 not in w]
    assert all(w in read for w in innocent)                            # read whole, never split
