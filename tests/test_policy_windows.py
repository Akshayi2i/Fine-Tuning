"""Policy windows — how a canonical policy is read, in training and in serving.

A policy is read as a cross product of section groups and page windows. These
tests pin the properties that make that safe: a window's target is only what its
own pages show, page numbers stay the document's, nothing reachable is lost, and
the windows a model is trained on are exactly the windows it is served.
"""

from __future__ import annotations

import json
import re

from types import SimpleNamespace

import pytest

from artifact_registry.blob_client import BlobClient, InMemoryBackend
from calibration.fit_calibration import CalibrationParams
from data_pipeline.dataset_builder.build_jsonl import (
    SourceDocument,
    build_corpus,
    expand_document,
    train_rows_by_epoch,
)
from data_pipeline.dataset_builder.policy_windows import (
    PolicyWindowPlan,
    TargetReport,
    pages_to_read,
    plan_windows,
    unread_values,
    window_target,
)
from data_pipeline.dataset_builder.split_groups import GroupRecord, assign_group_splits
from inference_core.model_runner import EchoBackend, Generation, ModelBackend, load_model
from serving.doc_type_classifier import StaticClassifier
from serving.pipeline import ExtractionRequest, extract

TOTAL = 15
#: A 15-page policy: declarations on page 1, a location schedule on 7-8, an
#: endorsement on 12, boilerplate everywhere else.
PAGE_TEXT = {
    1: "Common Policy Declarations. Named Insured: Rivera Fabrication LLC. Policy Number WC-1.",
    7: "Schedule of Locations",
    8: "Schedule of Locations (continued)",
    12: "Endorsement - Additional Insured",
}
TEXTS = [PAGE_TEXT.get(page, "Standard conditions and definitions.") for page in range(1, TOTAL + 1)]

#: The line these windows are read for: a self-contained one, the shape LABEL
#: is written in (equipment breakdown keeps the layout property had before it
#: moved). Not the fallback (no line), and not property, general liability, auto,
#: wc or umbrella: those compose the common model, whose windows
#: tests/test_schema2_*.py pin.
LOB = "equipment_breakdown"


def _fv(value, page):
    return {"raw": value, "parsed": value, "confidence": {"score": 1.0, "source": "audit"},
            "page_ref": [page], "flagged": False}


LABEL = {
    "carrier": {"company_name": _fv("Granite Mutual", 1)},
    "named_insured": {"primary_name": _fv("Rivera Fabrication LLC", 1)},
    "policy": {"policy_number": _fv("WC-1", 1)},
    "locations": [
        {"location_number": _fv("1", 7), "address": {"city": _fv("Toledo", 7)}},
        # One row printed across a page break: its city is on the next page.
        {"location_number": _fv("2", 7), "address": {"city": _fv("Dayton", 8)}},
    ],
    "forms_and_endorsements": [{"form_number": _fv("CG 20 10", 12)}],
    # A value printed on a page no section signal marks: read all the same.
    "state_notices": [{"state": _fv("OH", 14)}],
}


def _plans(lob=LOB):
    pages, declarations = pages_to_read(TEXTS, TOTAL)
    return pages, plan_windows(lob, pages, declarations)


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------

def test_every_group_is_asked_over_every_page():
    """Every page, chunk by chunk, never selected pages (decision 2026-10-19)."""
    pages, plans = _plans()
    assert pages == list(range(1, TOTAL + 1))
    by_group: dict[str, set[int]] = {}
    for plan in plans:
        by_group.setdefault(plan.group, set()).update(plan.pages)
    assert by_group and all(read == set(pages) for read in by_group.values()), by_group
    # The declarations group's first window: pages 1-5, read together.
    assert next(p for p in plans if p.group == "decl").pages == (1, 2, 3, 4, 5)


def test_a_declarations_page_behind_a_cover_sheet_joins_the_first_window():
    texts = ["FAX COVER SHEET"] * 7 + ["Common Policy Declarations"] + ["Conditions"] * 4
    pages, declarations = pages_to_read(texts, len(texts))
    plans = plan_windows(LOB, pages, declarations)
    assert declarations == 8
    assert next(p for p in plans if p.group == "decl").pages == (1, 2, 3, 4, 5, 8)
    assert {page for p in plans if p.group == "decl" for page in p.pages} == set(pages)


def test_a_short_policy_is_windowed_too():
    """Always, whatever the length: a threshold computed from prompt length would
    move between corpus builds and silently re-shape documents."""
    pages, declarations = pages_to_read(["Declarations"], 1)
    groups = {plan.group for plan in plan_windows("homeowners", pages, declarations)}
    assert groups == {"decl", "arrays", "forms"}


def test_image_only_reads_every_page():
    assert pages_to_read(None, 4) == ([1, 2, 3, 4], None)


# --------------------------------------------------------------------------
# The gold slicer
# --------------------------------------------------------------------------

def _plan(group, pages, *, single=False, index=0):
    return PolicyWindowPlan(group, index, tuple(pages), single=single)


def test_a_window_target_holds_only_its_sections_on_its_pages():
    target = window_target(LABEL, LOB, _plan("arrays", [12]))
    assert set(target) == {"forms_and_endorsements"}
    assert "carrier" not in target, "another group's section leaked into the slice"


def test_page_ref_keeps_the_documents_page_numbers():
    """A window-relative number would train the model to emit 1..n per window,
    and every page_ref in production would be off by an offset nobody sees."""
    target = window_target(LABEL, LOB, _plan("arrays", [12]))
    assert target["forms_and_endorsements"][0]["form_number"]["page_ref"] == [12]


def test_a_row_across_a_window_boundary_is_in_both_with_what_each_can_see():
    first = window_target(LABEL, LOB, _plan("arrays", [7]))
    second = window_target(LABEL, LOB, _plan("arrays", [8], index=1))
    rows_first = first["locations"]
    assert [r["location_number"]["raw"] for r in rows_first] == ["1", "2"]
    assert "address" not in rows_first[1], "page 8's city is not visible in the page-7 window"
    (row_second,) = second["locations"]
    assert row_second["address"]["city"]["raw"] == "Dayton"


def test_the_decl_target_keeps_its_required_objects():
    target = window_target(LABEL, LOB, _plan("decl", [1], single=True))
    assert {"carrier", "named_insured", "policy"} <= set(target)
    assert set(target["carrier"]["company_name"]) == {"raw", "parsed", "page_ref"}


def test_a_value_with_no_page_is_placed_only_when_there_is_one_window():
    label = {"policy": {"policy_number": {**_fv("WC-1", 1), "page_ref": []}},
             "carrier": {}, "named_insured": {}}
    single = window_target(label, LOB, _plan("decl", [1], single=True))
    assert single["policy"]["policy_number"]["raw"] == "WC-1"

    report = TargetReport()
    split = window_target(label, LOB, _plan("decl", [1], single=False), report)
    assert "policy_number" not in split["policy"]
    assert report.unplaced == ["decl:policy.policy_number"]


def test_no_value_is_left_on_a_page_no_window_reads():
    """Page 14 carries no section signal and is read all the same: a value
    there was unreachable under page selection, at training and at serving."""
    _pages, plans = _plans()
    assert unread_values(LABEL, LOB, plans) == []
    # Still counted when a plan leaves a page out.
    assert "state_notices[0].state" in unread_values(LABEL, LOB, [p for p in plans if 14 not in p.pages])


# --------------------------------------------------------------------------
# The corpus build
# --------------------------------------------------------------------------

def _document(source_id="policy_0001", lob=LOB, label=LABEL):
    return SourceDocument(
        source_id=source_id, doc_type="policy", golden_label=label, ocr_pages=list(TEXTS),
        image_paths=[f"processed/default/policy/{source_id}/page_{p}.png" for p in range(1, TOTAL + 1)],
        lob=lob, tenant_id="default",
    )


def test_a_policy_becomes_one_row_per_window_per_mode():
    rows, details = expand_document(_document(), "train", modes=("ocr_plus_image",))
    _routed, plans = _plans()
    assert [(r["sections"], r["window_pages"]) for r in rows] == [
        (p.group, list(p.pages)) for p in plans
    ]
    assert all(r["task"] for r in rows)
    assert not any("unread" in d for d in details)


def test_a_row_carries_the_documents_page_markers():
    rows, _ = expand_document(_document(), "train", modes=("ocr_plus_image",))
    arrays = next(r for r in rows if r["sections"] == "arrays" and 12 in r["window_pages"])
    text = json.dumps(arrays["messages"][1])
    assert f"<page 12 of {TOTAL}>" in text
    assert "<page 1 of 1>" not in text


@pytest.mark.format_profile("standard")
def test_a_row_prompts_for_its_slice_only():
    rows, _ = expand_document(_document(), "train", modes=("ocr_plus_image",))
    decl = next(r for r in rows if r["sections"] == "decl")["messages"][0]["content"]
    arrays = next(r for r in rows if r["sections"] == "arrays")["messages"][0]["content"]
    assert '"company_name"' in decl and '"company_name"' not in arrays
    assert '"locations"' in arrays and '"locations"' not in decl


def test_a_flat_policy_label_is_skipped_rather_than_trained_as_empty():
    """A flat label has none of the canonical sections: every window's target
    would be `{}`, teaching "this policy states nothing" with no error at all."""
    flat = _document(label={"insured_name": "X", "line_of_business": ["workers_comp"]})
    groups = {"policy": [GroupRecord(group_id=flat.source_id, doc_type="policy",
                                     source_ids=[flat.source_id])]}
    built = build_corpus([flat], assign_group_splits(groups, seed=42))
    assert not built.all_rows
    assert built.skipped and "pre-canonical" in built.skipped[0][1]


def test_every_epoch_holds_every_windowed_policy():
    docs = [_document(f"policy_{i:04d}") for i in range(1, 21)]
    groups = {"policy": [GroupRecord(group_id=d.source_id, doc_type="policy",
                                     source_ids=[d.source_id]) for d in docs]}
    built = build_corpus(docs, assign_group_splits(groups, seed=42))
    by_epoch = train_rows_by_epoch(built)
    train_ids = {r["source_id"] for r in built.rows_by_split["train"]}
    for rows in by_epoch.values():
        assert {r["source_id"] for r in rows} == train_ids
        assert len(rows) > len(train_ids), "a policy is several windows, so several rows"


# --------------------------------------------------------------------------
# Training and serving read the same windows
# --------------------------------------------------------------------------

@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _serve(client, response, **request):
    backend = EchoBackend(json.dumps(response))
    model = load_model("base", client, backend_impl=backend)
    result = extract(
        ExtractionRequest(
            source_id="policy_0001",
            image_paths=[f"processed/default/policy/policy_0001/page_{p}.png"
                         for p in range(1, TOTAL + 1)],
            page_texts={p: t for p, t in enumerate(TEXTS, start=1)},
            known_doc_type="policy", **request,
        ),
        model, StaticClassifier("policy"),
        CalibrationParams(method="temperature", doc_type="policy",
                          model_version="v1", temperature=1.0),
    )
    return result, backend


def _decl_response():
    return {
        "carrier": {"company_name": {"raw": "Granite Mutual", "parsed": "Granite Mutual",
                                     "page_ref": [1]}},
        "named_insured": {"primary_name": {"raw": "Rivera", "parsed": "Rivera", "page_ref": [1]}},
        "policy": {"policy_number": {"raw": "WC-1", "parsed": "WC-1", "page_ref": [1]}},
    }


@pytest.mark.format_profile("standard")
def test_serving_asks_the_windows_training_built(client, monkeypatch):
    """The one property the whole design rests on. A served window that no
    training row had is a shape the model was never taught."""
    import serving.pipeline as pipeline

    # Sent in planned order, so each call pairs with its training row; the
    # order windows are sent in has tests of its own (test_window_send_order).
    monkeypatch.setattr(pipeline, "_send_order", lambda pending, doc_type: list(range(len(pending))))
    rows, _ = expand_document(_document(), "train", modes=("ocr_plus_image",))
    trained = [(r["sections"], r["window_pages"]) for r in rows]

    _result, backend = _serve(client, _decl_response(), known_lob=LOB)
    served_schemas = [call["json_schema"] for call in backend.calls]

    assert len(backend.calls) == len(trained)
    from common.schemas import resolved_schema, with_page_bounds
    from inference_core.input_builder import page_total

    # Each constrained to its slice, with page_ref bounded to the pages its
    # training row showed.
    assert served_schemas == [
        with_page_bounds(resolved_schema("policy", None, LOB, group),
                         page_total(call["messages"]), pages=list(pages))
        for (group, pages), call in zip(trained, backend.calls, strict=True)
    ]
    served_prompts = [call["messages"][0]["content"] for call in backend.calls]
    trained_prompts = [r["messages"][0]["content"] for r in rows]
    assert served_prompts == trained_prompts, "a served prompt differs from its training row's"


def _fallback_response():
    """The same declarations as the fallback's common-model view writes them
    (the carrier's name is `name` there), and a coverage: its id is one the
    pipeline supplies, never the model, and the client's schema requires it."""
    response = _decl_response()
    response["carrier"] = {"name": response["carrier"]["company_name"]}
    response["coverages"] = [{"coverage_code": "X_TERRORISM",
                              "coverage_name": {"raw": "Terrorism", "parsed": "Terrorism", "page_ref": [12]}}]
    return response


@pytest.mark.parametrize("lob,response,carrier_name", [
    (LOB, _decl_response(), "company_name"),
    # No line: the fallback, which composes the common model. Every policy
    # served without a line is read against it.
    (None, _fallback_response(), "name"),
], ids=["line", "fallback"])
def test_serving_returns_one_canonical_document(client, lob, response, carrier_name):
    result, _ = _serve(client, response, known_lob=lob)
    assert result.schema_valid
    leaf = result.extraction["carrier"][carrier_name]
    assert set(leaf) == {"raw", "parsed", "confidence", "page_ref", "flagged"}
    assert result.pages_used == list(range(1, TOTAL + 1)), "every page is read"
    # Every window echoed the same declarations: one value, not several.
    assert not any(f.endswith(":merge_conflict") for f in result.review_flags)


# --------------------------------------------------------------------------
# Concurrency, and a bad window not losing the document
# --------------------------------------------------------------------------

class _ScriptedBackend(ModelBackend):
    """Answers each window by a rule over (group, pages), and records batches.

    ``fail(group, pages)`` returns ``"garbage"``, ``"truncated"``, ``"unknown_key"``
    (the decl answer carries a key no schema names), ``"transport"``
    (the server could not be reached), ``"bad_request"`` (a 4xx) or ``None``.
    """

    def __init__(self, fail=lambda group, pages: None):
        self.fail = fail
        self.batches: list[int] = []
        self.windows: list[tuple[str, list[int]]] = []

    def supports_logprobs(self) -> bool:
        return True

    @staticmethod
    def _window(messages, config):
        properties = (config.json_schema or {}).get("properties", {})
        group = "decl" if "carrier" in properties else "arrays" if "locations" in properties else "other"
        text = json.dumps(messages[1]["content"])
        pages = [int(n) for n in re.findall(r"<page (\d+) of", text)]
        return group, pages

    def generate_batch(self, messages_list, configs, adapter=None):
        self.batches.append(len(messages_list))
        return super().generate_batch(messages_list, configs, adapter)

    def generate(self, messages, config, adapter=None) -> Generation:
        group, pages = self._window(messages, config)
        self.windows.append((group, pages))
        mode = self.fail(group, pages)
        if mode == "transport":
            from inference_core.model_runner import ModelTransportError

            raise ModelTransportError("502 from /v1/chat/completions: an HTML error page (Waiting for service)")
        if mode == "bad_request":
            from inference_core.model_runner import ModelRunnerError

            raise ModelRunnerError("400 from /v1/chat/completions: maximum context length exceeded")
        body = _decl_response() if group == "decl" else {}
        if mode == "bad_dates" and group == "decl":
            body = {**body, "policy": {**body["policy"], "effective_date": _fv_("07/12/2025"),
                                       "expiration_date": _fv_("06/01/2025")}}
        if mode == "unknown_key" and group == "decl":
            body = {**body, "made_up_section": {"x": _fv_("1")},
                    "carrier": {**body.get("carrier", {}), "invented_rating": _fv_("A++")}}
        text = "{\"carrier\": {\"company" if mode == "garbage" else json.dumps(body)
        tokens = [text[i:i + 4] for i in range(0, len(text), 4)]
        return Generation(
            text=text, tokens=tokens, token_logprobs=[-0.01] * len(tokens),
            finish_reason="length" if mode == "truncated" else "stop",
        )


def _serve_with(client, backend):
    from inference_core.model_runner import load_model as _load

    model = _load("base", client, backend_impl=backend)
    return extract(
        ExtractionRequest(
            source_id="policy_0001",
            image_paths=[f"processed/default/policy/policy_0001/page_{p}.png"
                         for p in range(1, TOTAL + 1)],
            page_texts={p: t for p, t in enumerate(TEXTS, start=1)},
            known_doc_type="policy", known_lob=LOB,
        ),
        model, StaticClassifier("policy"),
        CalibrationParams(method="temperature", doc_type="policy",
                          model_version="v1", temperature=1.0),
    )


def test_every_window_goes_to_the_model_in_one_batch(client):
    """vLLM schedules a batch together, so a policy costs about its slowest
    window, not the sum of all of them."""
    backend = _ScriptedBackend()
    _serve_with(client, backend)
    _routed, plans = _plans()
    assert backend.batches == [len(plans)]


def test_a_failed_window_is_split_and_retried_over_fewer_pages(client):
    """Greedy decoding would write the same thing again, so the retry is a
    smaller window — less to write — not the same one."""
    backend = _ScriptedBackend(
        fail=lambda group, pages: "truncated" if group == "arrays" and len(pages) > 1 else None
    )
    result = _serve_with(client, backend)

    assert len(backend.batches) >= 2, "the failed window was never retried"
    retried = [pages for group, pages in backend.windows if group == "arrays" and len(pages) == 1]
    assert {p for pages in retried for p in pages} >= {7, 8, 12}
    assert not any(f.startswith("window_failed") for f in result.review_flags)
    assert result.schema_valid


def test_a_document_with_a_window_lost_to_the_transport_is_not_served_as_read(client):
    """A pod that goes down mid-document answers 502 to every window after it. Those
    windows were never read, so the document is refused (the worker tries it again)
    - not returned, flagged, as a finished job with holes in it."""
    from serving.pipeline import PipelineError

    backend = _ScriptedBackend(fail=lambda group, pages: "transport" if group == "arrays" and 12 in pages else None)
    with pytest.raises(PipelineError, match=r"never reached the model.*arrays:p12.*not served"):
        _serve_with(client, backend)


def test_a_window_the_server_refused_as_a_bad_request_is_still_flagged_not_refused(client, monkeypatch):
    """A 4xx will fail again however often it is sent: the document is returned
    without that window, flagged, as before."""
    import serving.printed_values as printed_values

    monkeypatch.setattr(printed_values, "configured_mode", lambda: printed_values.FLAG)
    backend = _ScriptedBackend(fail=lambda group, pages: "bad_request" if group == "arrays" and 12 in pages else None)
    result = _serve_with(client, backend)
    assert "window_failed:arrays:p12" in result.review_flags


def test_a_window_that_fails_alone_is_flagged_and_the_rest_returned(client, monkeypatch):
    import serving.printed_values as printed_values

    # Its page text prints none of the answer's values; the printed-values
    # check is not what this tests (tests/test_printed_values.py).
    monkeypatch.setattr(printed_values, "configured_mode", lambda: printed_values.FLAG)
    backend = _ScriptedBackend(
        fail=lambda group, pages: "garbage" if group == "arrays" and 12 in pages else None
    )
    result = _serve_with(client, backend)

    assert "window_failed:arrays:p12" in result.review_flags
    assert result.extraction["carrier"]["company_name"]["parsed"] == "Granite Mutual"
    assert result.schema_valid


def test_a_document_with_no_readable_window_is_an_error(client):
    from serving.pipeline import PipelineError

    backend = _ScriptedBackend(fail=lambda group, pages: "garbage")
    with pytest.raises(PipelineError, match="no window of policy_0001 could be read"):
        _serve_with(client, backend)


def test_a_batch_reports_one_failure_without_losing_the_others(client):
    from inference_core.model_runner import ModelRunnerError, generate_batch
    from inference_core.model_runner import load_model as _load

    class _Half(_ScriptedBackend):
        def generate(self, messages, config, adapter=None):
            if messages[0]["content"] == "boom":
                raise RuntimeError("backend fell over")
            return super().generate(messages, config, adapter)

    model = _load("base", client, backend_impl=_Half())
    ok = [{"role": "system", "content": "x"}, {"role": "user", "content": [{"type": "text", "text": "<page 1 of 1>"}]}]
    bad = [{"role": "system", "content": "boom"}, ok[1]]
    results = generate_batch(model, [(ok, None), (bad, None), (ok, None)])

    assert isinstance(results[1], ModelRunnerError) and "fell over" in str(results[1])
    assert not isinstance(results[0], Exception) and not isinstance(results[2], Exception)


# --------------------------------------------------------------------------
# What the served-model backend calls a transport failure
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("status", "transport"), [
    (502, True), (520, True), (503, True), (429, True), (408, True), (500, True),
    (400, False), (413, False), (422, False), (401, False), (404, False),
])
def test_only_a_server_side_http_status_is_a_transport_failure(status, transport):
    import io
    import urllib.error

    from inference_core.model_runner import ModelRunnerError, ModelTransportError
    from inference_core.openai_backend import OpenAIChatBackend

    backend = OpenAIChatBackend("https://pod.example.test", "tok", model="m")
    error = urllib.error.HTTPError("https://pod.example.test/v1/chat/completions", status, "x", {},
                                   io.BytesIO(b"<!DOCTYPE html><html><head><title>Waiting for service to "
                                              b"respond - RunPod</title></head><body>" + b"x" * 5000 + b"</body></html>"))

    from unittest import mock

    with mock.patch("common.transport.open_url", side_effect=error):
        with pytest.raises(ModelRunnerError) as raised:
            backend._request("/v1/chat/completions", b"{}")
    assert isinstance(raised.value, ModelTransportError) is transport
    # A proxy's HTML page is reduced to its title, never the whole page.
    assert "Waiting for service to respond - RunPod" in str(raised.value) and len(str(raised.value)) < 200


def test_an_unreachable_server_and_a_broken_off_answer_are_transport_failures():
    import urllib.error
    from unittest import mock

    from inference_core.model_runner import ModelTransportError
    from inference_core.openai_backend import OpenAIChatBackend

    backend = OpenAIChatBackend("https://pod.example.test", "tok", model="m")
    for failure in (urllib.error.URLError("[Errno 11001] getaddrinfo failed"), ConnectionResetError("reset")):
        with mock.patch("common.transport.open_url", side_effect=failure):
            with pytest.raises(ModelTransportError):
                backend._request("/v1/chat/completions", b"{}")


# --------------------------------------------------------------------------
# No document content in a failure message or a log line
# --------------------------------------------------------------------------

SECRET = "Jane Q. Insured 123-45-6789"


def test_an_answer_that_does_not_parse_is_never_quoted_in_its_error(client):
    """The error is logged at WARNING, kept in the window's stats and returned in a
    job's error: quoted, it carried the first 160 characters of the answer."""
    from inference_core.model_runner import Generation
    from serving.pipeline import PipelineError, _parse_generation

    text = '{"named_insured": {"name": {"raw": "' + SECRET + '", "page_ref": [1]}}'     # cut off: no closing braces
    result = Generation(text=text, tokens=[text], token_logprobs=[-0.1], finish_reason="stop")
    route_ = SimpleNamespace(schema_doc_type="policy")
    with pytest.raises(PipelineError) as raised:
        _parse_generation(result, route_, refuse_truncated=True, compact=False)
    assert SECRET not in str(raised.value) and "Jane" not in str(raised.value)
    assert f"{len(text)} characters" in str(raised.value)


def test_a_classifier_answer_that_is_not_json_is_never_quoted():
    from serving.doc_type_classifier import ClassificationError, parse_classification

    with pytest.raises(ClassificationError) as raised:
        parse_classification("Named insured " + SECRET + " is a restaurant")
    assert SECRET not in str(raised.value)


# --------------------------------------------------------------------------
# A whole business-owners policy: told which part to send, never guessed
# --------------------------------------------------------------------------

def _release_of(*lines):
    from serving.release_router import ServedRelease

    return ServedRelease(release_id="policy-v1", base_model="b", merged_model=None, doc_types=("policy",), adapter="a",
                         adapter_dir=None, calibrators=None, prompt_hash="", window_plan_version=None, ocr_pin={},
                         created_at="", lines=lines, served_as="lora_on_base")


@pytest.mark.parametrize("alias", ["cgl", "general_liability", "workers_comp"])
def test_a_line_sent_under_its_other_name_reaches_the_release(alias):
    from serving.release_router import ServingPlan

    plan = ServingPlan(_release_of("gl", "wc", "boppr", "bopgl"))
    assert plan.route("policy", alias, allow_fallback=True).release is not None


def test_a_business_owners_policy_is_not_guessed_into_a_part():
    """business_owners goes to the base model (no part chosen), bop to a refusal;
    boppr and bopgl, the parts the adapter trained on, to the release."""
    from serving.release_router import ServingPlan, UnservedDocType, business_owners_hint

    plan = ServingPlan(_release_of("gl", "wc", "boppr", "bopgl"))
    assert plan.route("policy", "boppr").release is not None and plan.route("policy", "bopgl").release is not None
    assert plan.route("policy", "business_owners", allow_fallback=True).release is None
    with pytest.raises(UnservedDocType, match="lob=boppr .* lob=bopgl"):
        plan.release_for("policy", "business_owners")
    assert business_owners_hint("BOP") and business_owners_hint(["business_owners"]) and not business_owners_hint("gl")


def test_a_request_naming_bop_is_told_which_part_to_send():
    from serving.vllm_entrypoint import ServingError, request_lob

    with pytest.raises(ServingError, match="lob=boppr .* lob=bopgl"):
        request_lob({"lob": "bop"})
    with pytest.raises(ServingError) as raised:
        request_lob({"lob": "no_such_line"})
    assert "lob=boppr" not in str(raised.value)                  # only for a business-owners name
    assert request_lob({"lob": "boppr"}) == "boppr"



# --------------------------------------------------------------------------
# No key the schema has no place for
# --------------------------------------------------------------------------

def _fv_(value, page=1):
    return {"raw": value, "parsed": value, "page_ref": [page]}


@pytest.mark.parametrize("line", ["gl", "wc", "bopgl", "boppr"])
def test_every_object_the_model_is_decoded_against_is_closed(line):
    """The grammar can only refuse a key the schema leaves out if the schema says
    so: every object of every field group's slice is additionalProperties false."""
    from common.schema_sections import group_names
    from common.schemas import resolved_schema

    open_objects = []

    def walk(node, where):
        if isinstance(node, dict):
            if (node.get("type") == "object" or "properties" in node) and node.get("additionalProperties", None) is not False:
                open_objects.append(where)
            for key, value in node.items():
                walk(value, f"{where}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")

    for group in group_names(line):
        walk(resolved_schema("policy", None, line, sections=group), f"{line}:{group}")
    assert not open_objects, open_objects[:5]


def test_a_key_no_schema_names_is_never_served_and_is_said(client, monkeypatch):
    """Whatever the backend: an answer carrying a key the client's schema has no place
    for is served without it, flagged where it was, and the key's confidence goes too."""
    import serving.printed_values as printed_values

    monkeypatch.setattr(printed_values, "configured_mode", lambda: printed_values.FLAG)
    result = _serve_with(client, _ScriptedBackend(fail=lambda group, pages: "unknown_key" if group == "decl" else None))
    flags = set(result.review_flags)
    assert "made_up_section:unknown_key_removed" in flags and "carrier.invented_rating:unknown_key_removed" in flags
    assert "made_up_section" not in result.extraction and "invented_rating" not in result.extraction["carrier"]
    assert not any("invented_rating" in path for path in result.fields)
    assert result.extraction["carrier"]["company_name"]["parsed"] == "Granite Mutual"      # the rest is untouched
    assert not any("made_up_section" in error or "invented_rating" in error for error in result.validation_errors)


def test_a_clean_answer_has_nothing_taken_out(client, monkeypatch):
    import serving.printed_values as printed_values

    monkeypatch.setattr(printed_values, "configured_mode", lambda: printed_values.FLAG)
    result = _serve_with(client, _ScriptedBackend())
    assert not any(flag.endswith("unknown_key_removed") for flag in result.review_flags)
