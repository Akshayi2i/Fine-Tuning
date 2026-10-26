"""A split declarations window keeps its lead; a window's answer is held to its
own pages and its schema's shapes; a request is refused before it is misread;
a run split by its text shares its pages - the same way in training and serving.

Synthetic documents only: every text, label and answer here is made up.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.fit_calibration import CalibrationParams
from common.schema_sections import groups_for
from common.schemas import resolved_schema
from data_pipeline.dataset_builder.expand_tasks import _fit_overlapping, fit_windows_to_text
from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
from inference_core.model_runner import Generation, ModelBackend, load_model
from serving.doc_type_classifier import StaticClassifier
from serving.pipeline import (
    PAGE_REF_OUTSIDE_WINDOW_FLAG,
    ExtractionRequest,
    PipelineError,
    _check_section_shapes,
    _cite_shown_pages,
    _halves,
    _Planned,
    _section_shapes,
    extract,
)
from serving.policy_merge import PolicyWindow, merge_policy_windows

CALIBRATION = CalibrationParams(method="temperature", doc_type="policy", model_version="v1", temperature=1.0)
#: A self-contained line, and a common-model line (overlapping runs, ids).
LINES = ("property", "personal_auto")

DENSE = " ".join(["Notice of privacy practices and cancellation wording."] * 900)
DECLARATIONS = ("COMMON POLICY DECLARATIONS named insured policy number policy period\n"
                "Named Insured: Rivera Fabrication LLC\nGranite Mutual Insurance Co\nPolicy Number WC-NEW")


def _fv(raw, *pages):
    return {"raw": raw, "parsed": raw, "page_ref": list(pages)}


def _pages(messages) -> list[int]:
    return [int(n) for n in re.findall(r"<page (\d+) of \d+>", json.dumps(messages))]


def _group(config, lob) -> str | None:
    """The section group a window asks for, from its decoding schema's keys."""
    keys = set((config.json_schema or {}).get("properties", {}))
    return next((group for group in groups_for(lob)
                 if set(resolved_schema("policy", None, lob, group).get("properties", {})) == keys), None)


class Scripted(ModelBackend):
    """Answers each window with ``answer(group, pages)``: a JSON string, or a
    Generation as it is (one cut off at the token limit, say)."""

    def __init__(self, answer, lob):
        self.answer, self.lob = answer, lob
        self.calls: list[tuple[str | None, list[int]]] = []
        self.texts: list[str] = []

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config, adapter=None) -> Generation:
        group, pages = _group(config, self.lob), _pages(messages)
        self.calls.append((group, pages))
        self.texts.append(json.dumps(messages[1]["content"]))
        out = self.answer(group, pages)
        return out if isinstance(out, Generation) else _generation(out)


class AnsweredOneByOne(Scripted):
    """The same answers, each on its own thread: the served-model path."""

    def __init__(self, answer, lob):
        super().__init__(answer, lob)
        self.pool = ThreadPoolExecutor(max_workers=32)

    def supports_submit(self) -> bool:
        return True

    def submit(self, messages, config, adapter=None):
        def answer():
            time.sleep(((sum(_pages(messages)) * 7) % 5) / 400)      # finish out of order
            return self.generate(messages, config, adapter)
        return self.pool.submit(answer)


def _generation(text: str, finish: str = "stop") -> Generation:
    tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
    return Generation(text=text, tokens=tokens, token_logprobs=[-0.01] * len(tokens), finish_reason=finish)


def _request(texts: dict, lob, *, images: int | None = None, **over) -> ExtractionRequest:
    count = len(texts) if images is None else images
    return ExtractionRequest(
        source_id="qa_doc", known_doc_type="policy", known_lob=lob, page_texts=texts,
        image_paths=[f"processed/default/policy/qa_doc/page_{p}.png" for p in range(1, count + 1)],
        **over,
    )


def _extract(backend, request, *, strict=False, **kwargs):
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    return extract(request, load_model("base", client, backend_impl=backend), StaticClassifier("policy"),
                   CALIBRATION, strict_schema=strict, **kwargs)


# --------------------------------------------------------------------------
# The declarations window split or cut off keeps its lead
# --------------------------------------------------------------------------

def _renewal_texts() -> dict[int, str]:
    """Page 1 a renewal notice printing the PRIOR policy number; the
    declarations, with the policy's own, on page 4."""
    texts = {p: f"Page {p} wording." for p in range(1, 9)}
    texts[1] = "Renewal notice. Your prior policy WC-OLD expires soon."
    texts[4] = DECLARATIONS
    return texts


def _policy_number_answer(cut_off: set[tuple[int, ...]] | None = None):
    """WC-NEW from any window showing page 4, WC-OLD from one showing page 1
    alone; a decl window in ``cut_off`` is cut off at the token limit once."""
    seen: set[tuple[int, ...]] = set()

    def answer(group, pages):
        if group != "decl":
            return "{}"
        if 4 in pages:
            out = {"policy": {"policy_number": _fv("WC-NEW", 4)}}
        elif 1 in pages:
            out = {"policy": {"policy_number": _fv("WC-OLD", 1)}}
        else:
            out = {}
        text = json.dumps(out)
        if cut_off and tuple(pages) in cut_off and tuple(pages) not in seen:
            seen.add(tuple(pages))
            return _generation(text[:20], finish="length")
        return text
    return answer


@pytest.mark.parametrize("lob", LINES)
def test_a_declarations_window_split_by_its_text_serves_the_declarations_value(lob):
    """Dense notices on pages 1-3 split the declarations window. Training
    teaches the policy-level values in the piece holding the declarations page;
    the merge took the piece holding page 1 first, and served WC-OLD."""
    texts = _renewal_texts()
    texts[1] = texts[1] + " " + DENSE
    texts[2] = texts[3] = DENSE
    every, declarations = pages_to_read(texts, len(texts))
    decl = [plan for plan in plan_windows(lob, every, declarations, texts) if plan.group == "decl"]
    led = [plan for plan in decl if plan.leads]
    assert declarations == 4 and len(led) == 1 and 4 in led[0].pages and 1 not in led[0].pages
    assert any(1 in plan.pages for plan in decl if not plan.leads)            # the split happened

    result = _extract(Scripted(_policy_number_answer(), lob), _request(texts, lob))
    leaf = result.extraction["policy"]["policy_number"]
    assert (leaf["raw"], leaf["page_ref"]) == ("WC-NEW", [4])
    assert "policy.policy_number:merge_conflict" in result.review_flags      # the other reading, shown


@pytest.mark.parametrize("lob", LINES)
@pytest.mark.parametrize("backend_type", [Scripted, AnsweredOneByOne])
def test_a_declarations_window_retried_in_halves_keeps_its_lead(lob, backend_type):
    """[1..5] cut off at the token limit is retried as [1, 2] + [3, 4, 5]: the
    half holding the declarations page (4) leads, in rounds and as answered."""
    backend = backend_type(_policy_number_answer(cut_off={(1, 2, 3, 4, 5)}), lob)
    result = _extract(backend, _request(_renewal_texts(), lob))
    decl_calls = [pages for group, pages in backend.calls if group == "decl"]
    assert [1, 2, 3, 4, 5] in decl_calls and [1, 2] in decl_calls and [3, 4, 5] in decl_calls
    leaf = result.extraction["policy"]["policy_number"]
    assert (leaf["raw"], leaf["page_ref"]) == ("WC-NEW", [4])
    assert "policy.policy_number:merge_conflict" in result.review_flags


def test_the_half_holding_the_declarations_page_inherits_the_lead():
    led = _Planned("decl", [1, 2, 3, 4, 5], leads=True)
    assert _halves(led, 4) == [_Planned("decl", [1, 2], False), _Planned("decl", [3, 4, 5], True)]
    assert _halves(led, 1) == [_Planned("decl", [1, 2], True), _Planned("decl", [3, 4, 5], False)]
    # No declarations page found, or not in this window: the first half, as
    # policy_windows._home_window leads the first piece.
    assert [half.leads for half in _halves(led, None)] == [True, False]
    assert [half.leads for half in _halves(led, 9)] == [True, False]
    # A window its declarations do not lead leads neither half.
    assert not any(half.leads for half in _halves(_Planned("decl", [6, 7, 8]), 7))


@pytest.mark.parametrize("lob", LINES)
def test_the_merge_takes_the_declarations_led_window_first_in_its_group(lob):
    def windows(order):
        made = {
            "notice": PolicyWindow("decl", [1], {"policy": {"policy_number": _fv("WC-OLD", 1)}}),
            "decl": PolicyWindow("decl", [3, 4, 5], {"policy": {"policy_number": _fv("WC-NEW", 4)}},
                                 leads=True),
            "later": PolicyWindow("decl", [6, 7], {"policy": {"policy_number": _fv("WC-9", 6)}}),
        }
        return [made[name] for name in order]

    for order in (["notice", "decl", "later"], ["later", "notice", "decl"]):
        merged = merge_policy_windows(windows(order), lob=lob)
        assert merged.extraction["policy"]["policy_number"]["raw"] == "WC-NEW"
        assert "policy.policy_number:merge_conflict" in merged.review_flags
    # Led by none, the windows merge in page order, as before.
    plain = [PolicyWindow(w.group, w.pages, w.extraction) for w in windows(["decl", "notice"])]
    assert merge_policy_windows(plain, lob=lob).extraction["policy"]["policy_number"]["raw"] == "WC-OLD"


# --------------------------------------------------------------------------
# A window's answer: its own pages, its schema's shapes
# --------------------------------------------------------------------------

def test_a_page_reference_is_held_to_the_windows_pages():
    answer = {
        "named_insured": {"primary_name": _fv("Rivera", 99)},
        "policy": {"policy_number": {"raw": "WC-1", "parsed": "WC-1", "page_ref": [0, 2]},
                   "policy_type": {"raw": "Special", "parsed": "Special", "page_ref": "12"},
                   "effective_date": {"raw": "04/01/2026", "parsed": "04/01/2026", "page_ref": 3}},
        "locations": [{"location_number": {"raw": "1", "parsed": "1", "page_ref": ["2", "x", True]}}],
        "carrier": {"company_name": _fv("Granite", 2, 3)},
    }
    outside = _cite_shown_pages(answer, [2, 3, 12])
    assert answer["named_insured"]["primary_name"]["page_ref"] == []
    assert answer["policy"]["policy_number"]["page_ref"] == [2]
    assert answer["policy"]["policy_type"]["page_ref"] == [12]                # never [1, 2]
    assert answer["policy"]["effective_date"]["page_ref"] == [3]
    assert answer["locations"][0]["location_number"]["page_ref"] == [2]
    assert answer["carrier"]["company_name"]["page_ref"] == [2, 3]            # untouched
    assert outside == ["locations[].location_number", "named_insured.primary_name", "policy.policy_number"]


@pytest.mark.parametrize("bad", [99, 0, 7])
def test_a_value_citing_a_page_its_window_was_not_shown_is_flagged(bad):
    texts = {p: f"Page {p} wording." for p in range(1, 9)}
    texts[1] = "Common Policy Declarations. Named Insured: Rivera Fabrication LLC\nPolicy Number WC-1"

    def answer(group, pages):
        if group != "decl" or 1 not in pages:
            return "{}"
        return json.dumps({"named_insured": {"primary_name": _fv("Rivera Fabrication LLC", bad)},
                           "policy": {"policy_number": _fv("WC-1", 1)}})

    backend = Scripted(answer, "property")
    result = _extract(backend, _request(texts, "property"))
    shown = next(pages for group, pages in backend.calls if group == "decl" and 1 in pages)
    assert bad not in shown
    assert bad not in result.extraction["named_insured"]["primary_name"]["page_ref"]
    assert f"named_insured.primary_name:{PAGE_REF_OUTSIDE_WINDOW_FLAG}" in result.review_flags
    assert not [f for f in result.review_flags if f.startswith("policy.") and PAGE_REF_OUTSIDE_WINDOW_FLAG in f]


@pytest.mark.parametrize("lob", LINES)
def test_a_window_schema_says_which_sections_are_objects_and_which_lists(lob):
    decl, arrays = _section_shapes(_route(), lob, "decl"), _section_shapes(_route(), lob, "arrays")
    assert decl["policy"] == decl["named_insured"] == "object"
    assert set(arrays.values()) == {"array"}


def _route():
    from serving.adapter_router import Route

    return Route(doc_type="policy", acord_form=None, adapter=None, schema_doc_type="policy",
                 schema_acord_form=None)


def test_a_section_in_the_wrong_shape_is_refused():
    shapes = {"policy": "object", "locations": "array"}
    _check_section_shapes({"policy": {}, "locations": [], "unknown": "kept for the verdict"}, shapes, _route())
    _check_section_shapes({"policy": None}, shapes, _route())                   # nothing found
    for wrong in ({"policy": "WC-1"}, {"policy": []}, {"locations": {"a": 1}}, {"policy": 5}):
        with pytest.raises(PipelineError, match="shape their schema does not allow"):
            _check_section_shapes(wrong, shapes, _route())


def test_a_common_model_section_answered_as_a_bare_string_is_a_window_failure():
    """``"policy": "WC-1"`` crashed the request with a TypeError where the
    system fields are written into the policy section."""
    texts = {1: "Declarations. Policy Number WC-1", 2: "x", 3: "y"}

    def answer(group, pages):
        if group == "decl" and 1 in pages:
            return json.dumps({"document": {"title": _fv("Declarations", 1)}, "policy": "WC-1"})
        return "{}"

    with pytest.raises(PipelineError):
        _extract(Scripted(answer, "personal_auto"), _request(texts, "personal_auto"), strict=True)
    result = _extract(Scripted(answer, "personal_auto"), _request(texts, "personal_auto"))
    assert "window_failed:decl:p1" in result.review_flags
    assert "schema:invalid" in result.review_flags


# --------------------------------------------------------------------------
# Refused before it is misread
# --------------------------------------------------------------------------

def _decl_only(group, pages):
    if group == "decl" and 1 in pages:
        return json.dumps({"policy": {"policy_number": _fv("PA-1", 1)}})
    return "{}"


TEXTS3 = {1: "Declarations\nGranite Mutual Insurance Co\nPolicy Number PA-1", 2: "Schedule", 3: "Forms"}


@pytest.mark.parametrize("line", ["persnal_auto", "nonsense_line", ["personal_auto", "nonsense_line"]])
def test_a_line_no_schema_has_is_refused_without_a_serving_plan(line):
    with pytest.raises(PipelineError, match="allow_lob_fallback"):
        _extract(Scripted(_decl_only, "personal_auto"), _request(TEXTS3, line))


def test_a_line_no_schema_has_is_read_against_the_fallback_flagged_when_asked():
    from serving.pipeline import LOB_FALLBACK_FLAG

    result = _extract(Scripted(_decl_only, None), _request(TEXTS3, "persnal_auto", allow_lob_fallback=True))
    assert result.route_info["lob_fallback_used"] is True
    assert LOB_FALLBACK_FLAG in result.review_flags
    # A known line is read as it was: no fallback, no flag.
    known = _extract(Scripted(_decl_only, "personal_auto"), _request(TEXTS3, "personal_auto"))
    assert known.route_info["lob_fallback_used"] is False and LOB_FALLBACK_FLAG not in known.review_flags


@pytest.mark.parametrize(("texts", "images", "message"), [
    ({1: "Declarations", 2: "a", 3: "b"}, 4, r"covers pages \[1, 2, 3\] but 4 page image"),
    ({0: "Declarations", 1: "a", 2: "b"}, 3, r"covers pages \[0, 1, 2\]"),
    ({}, 0, "supplies no page images"),
    ({1: "Declarations", 2.0: "a"}, 2, "is not a page number"),
])
def test_page_texts_not_numbered_as_the_images_are_refused(texts, images, message):
    with pytest.raises(PipelineError, match=message):
        _extract(Scripted(_decl_only, "property"), _request(texts, "property", images=images))


def test_page_text_keys_written_as_digits_are_read_as_page_numbers():
    backend = Scripted(_decl_only, "property")
    result = _extract(backend, _request({"1": "Declarations Policy Number PA-1", "2": "Schedule page"},
                                        "property"))
    assert result.pages_used == [1, 2]
    assert any("Schedule page" in text for text in backend.texts)              # not read as blank


# --------------------------------------------------------------------------
# A run split by its text shares its pages
# --------------------------------------------------------------------------

def test_a_window_with_no_room_beyond_the_shared_page_starts_after_it():
    cost = {1: 1, 2: 6, 3: 6, 4: 6}.__getitem__
    assert _fit_overlapping([1, 2, 3, 4], cost, 10, 1) == [[1, 2], [3], [4]]
    cost = dict.fromkeys(range(1, 6), 4).__getitem__
    assert _fit_overlapping([1, 2, 3, 4, 5], cost, 10, 1) == [[1, 2], [2, 3], [3, 4], [4, 5]]


def test_a_run_split_by_its_text_alone_shares_its_pages():
    cost = dict.fromkeys(range(1, 12), 6).__getitem__
    # Kept whole by its page count, split by its text: shared, as one page longer is.
    assert fit_windows_to_text([[1, 2, 3]], cost, 13, overlap=1) == [[1, 2], [2, 3]]
    assert fit_windows_to_text([[1, 2, 3], [3, 4]], cost, 13, overlap=1) == [[1, 2], [2, 3], [3, 4]]
    # Two runs packed into one window share nothing where they meet.
    assert fit_windows_to_text([[5, 9, 10]], cost, 13, overlap=1) == [[5, 9], [9, 10]]
    assert fit_windows_to_text([[5, 6, 10]], cost, 13, overlap=1) == [[5, 6], [10]]
    # No overlap on the line: split evenly, as before.
    assert fit_windows_to_text([[1, 2, 3]], cost, 13) == [[1, 2], [3]]


@pytest.mark.format_profile("standard")
def test_training_and_serving_plan_the_same_windows():
    """The dataset build's rows and the windows serving sends, on one synthetic
    personal-auto policy whose dense pages split windows by their text."""
    from data_pipeline.dataset_builder.build_jsonl import SourceDocument, expand_document
    from tests.test_schema2_serving import _compact_auto

    lob, total = "personal_auto", 9
    texts = (["Personal Auto Declarations. Named Insured: JANE RIVERA. Policy Number PA-1."]
             + [f"Schedule page {p}. " + " ".join(["VIN 1HGCM82633A004352 Liability 100/300 Premium"] * 150)
                for p in range(2, total + 1)])
    images = [f"processed/default/policy/pa_1/page_{p}.png" for p in range(1, total + 1)]
    document = SourceDocument(source_id="pa_1", doc_type="policy", golden_label=_compact_auto(),
                              ocr_pages=texts, image_paths=images, lob=lob, tenant_id="default")
    rows, _ = expand_document(document, "train", modes=("ocr_plus_image",))
    trained = sorted((row["sections"], tuple(row["window_pages"])) for row in rows)

    backend = Scripted(lambda group, pages: "{}", lob)
    _extract(backend, ExtractionRequest(source_id="pa_1", image_paths=images,
                                        page_texts=dict(enumerate(texts, start=1)),
                                        known_doc_type="policy", known_lob=lob))
    assert sorted((group, tuple(pages)) for group, pages in backend.calls) == trained

    # Split by their text, with shared pages - a run its page count kept whole
    # too (forms' [6..9]) - and no window of shared pages alone: the cases this guards.
    every, declarations = pages_to_read(texts, total)
    by_count = sorted((p.group, p.pages) for p in plan_windows(lob, every, declarations))
    assert trained != by_count
    assert ("decl", (6, 7, 8, 9)) in by_count and {("decl", (6, 7, 8)), ("decl", (8, 9))} <= set(trained)
    for group in {group for group, _pages in trained}:
        windows = [set(pages) for name, pages in trained if name == group]
        assert set().union(*windows) == set(every)
        assert not [w for w in windows if any(w < other for other in windows)]


@pytest.mark.format_profile("standard")
def test_a_window_plan_cut_before_these_rules_is_refused(monkeypatch):
    """The windows changed for the same input, so the plan's version did: a
    corpus, or a release, cut with the version-3 plan is refused."""
    import data_pipeline.dataset_builder.policy_windows as policy_windows
    from serving.release_router import ServedRelease, ServingPlan
    from serving.vllm_entrypoint import ColdStartError, assert_window_plan_served

    current = policy_windows.window_plan_id()
    monkeypatch.setattr(policy_windows, "WINDOW_PLAN_VERSION", "3")
    previous = policy_windows.window_plan_id()
    monkeypatch.undo()
    assert current.startswith("4:") and previous.startswith("3:") and previous != current

    with pytest.raises(ValueError, match="Rebuild the corpus"):
        policy_windows.assert_window_plan({"window_plan_version": previous})
    release = ServedRelease("v3", "base@abc1234", "merged", doc_types=("policy",), window_plan_version=previous)
    with pytest.raises(ColdStartError, match="window plan"):
        assert_window_plan_served(ServingPlan(release))
