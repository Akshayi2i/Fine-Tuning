"""Which release answers for which document (arch v2.1 §12.3; decision 2026-10-22).

One promoted release per tenant: the newest promoted row of the tenant's release
index is served. It answers for the types it was trained on and refuses the rest,
rather than letting a model read a type it never saw. A policy of a line it
trained on goes to it; a policy of any other known line is read by the base
model with no LoRA against ``_fallback.json`` (Fideon SPEC_06 §9a, handoff item
4); a policy with no line, or a line with no schema, is refused unless the
caller asks for that fallback - or an older release that records no lines is
promoted, which reads any line, as it was trained to.
"""

from __future__ import annotations

import json
import logging

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from common.schemas import SNAPSHOT_LINES
from serving.release_router import (
    Routed,
    ServedRelease,
    ServingPlan,
    UnservedDocType,
    build_serving_plan,
    promoted_rows,
)

BASE = "qwen3-vl-8b-instruct@a1b2c3d"


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def promote(
    client: BlobClient,
    release_id: str = "v3",
    *,
    lines: tuple[str, ...] | list[str] = (),
    doc_types: tuple[str, ...] | list[str] = ("policy",),
    scope: str = "policy",
    created_at: str = "2026-10-01T00:00:00+00:00",
    base_model: str = BASE,
    status: str = "promoted",
    adapter: str | None = None,
    window_plan_version: str | None = None,
    tenant_id: str | None = None,
) -> dict:
    """Write a release bundle and its index row, as `package` leaves them.

    Defaults to a release of the policy adapter (``scope="policy"``, policies
    only). ``scope="unified", doc_types=()`` is an older release, written before
    types and lines were recorded."""
    version = release_id if paths.version_number(release_id) is not None else "v1"
    merged = paths.merged_model_dir(version, scope=scope)
    bundle = {
        "release_id": release_id,
        "status": status,
        "scope": scope,
        "doc_types": list(doc_types),
        "lines": list(lines),
        "tenant_scope": tenant_id or paths.default_tenant(),
        "created_at": created_at,
        "base_model": base_model,
        "adapter": adapter or f"{'extractor' if scope == 'unified' else scope}-{version}",
        "merged_model": merged,
        "serving_formats": {"bf16": merged},
        "calibrators": {"bf16": paths.release_calibrators(release_id, "bf16", tenant_id)},
        "prompt_hash": "abc123",
        "window_plan_version": window_plan_version,
        "ocr_pin": {"mineru_version": "2.0.0"},
    }
    client.write_json(paths.release_bundle(release_id, tenant_id), bundle)

    index_key = paths.release_index(tenant_id)
    rows = client.read_json(index_key) if client.exists(index_key) else []
    rows = [r for r in rows if r.get("release_id") != release_id]
    rows.append({"release_id": release_id, "status": status, "scope": scope,
                 "doc_types": list(doc_types), "created_at": created_at})
    client.write_json(index_key, rows)
    return bundle


def _personal(client, release_id="v3", **kw):
    """The policy adapter trained on the personal lines."""
    return promote(client, release_id, lines=SNAPSHOT_LINES, **kw)


# --------------------------------------------------------------------------
# The plan, from the tenant's release index
# --------------------------------------------------------------------------


def test_the_promoted_release_is_what_serving_reads(client):
    _personal(client)
    plan = build_serving_plan(client)

    release = plan.release
    assert release is not None and plan.served == [release]
    assert release.release_id == "v3" and release.base_model == BASE
    assert release.adapter == "policy-v3"
    assert release.adapter_dir == paths.scoped_adapter_dir("policy", "v3")
    assert release.merged_model == paths.merged_model_dir("v3", scope="policy")
    assert release.calibrators == paths.release_calibrators("v3", "bf16")
    assert release.lines == SNAPSHOT_LINES
    assert plan.served_doc_types == ("policy",)
    assert plan.unserved_doc_types == ("acord", "lossrun")


def test_the_newest_promoted_release_answers_and_a_warning_says_there_were_several(client, caplog):
    """`promote` leaves one promoted; an index edited by hand may hold two. The
    newest answers, whatever order the rows were written in."""
    _personal(client, "v4", created_at="2026-11-01T00:00:00+00:00")
    _personal(client, "v3", created_at="2026-10-01T00:00:00+00:00")

    with caplog.at_level(logging.WARNING, logger="serving.release_router"):
        plan = build_serving_plan(client)

    assert plan.release.release_id == "v4"
    assert [r["release_id"] for r in promoted_rows(client)] == ["v4", "v3"]
    assert "2 releases are promoted" in caplog.text


def test_only_a_promoted_release_serves(client):
    """A gated release is one that did not clear the gate. Serving it would make
    the gate advisory."""
    _personal(client, "v3", status="gated")
    promote(client, "v2", status="archived", created_at="2026-09-01T00:00:00+00:00")

    plan = build_serving_plan(client)
    assert plan.release is None and plan.served == []
    assert plan.unserved_doc_types == ("acord", "lossrun", "policy")
    assert plan.describe() == "nothing promoted"


def test_an_indexed_release_with_no_bundle_is_skipped_for_the_next(client, caplog):
    _personal(client, "v3", created_at="2026-10-01T00:00:00+00:00")
    rows = client.read_json(paths.release_index())
    rows.append({"release_id": "v4", "status": "promoted", "created_at": "2026-11-01T00:00:00+00:00"})
    client.write_json(paths.release_index(), rows)

    with caplog.at_level(logging.WARNING, logger="serving.release_router"):
        plan = build_serving_plan(client)

    assert plan.release.release_id == "v3"
    assert "v4 is indexed as promoted but has no bundle" in caplog.text


def test_a_release_with_nothing_to_serve_in_the_format_is_skipped(client):
    _personal(client, "v4", created_at="2026-11-01T00:00:00+00:00")
    empty = client.read_json(paths.release_bundle("v4"))
    empty.update(merged_model="", serving_formats={})
    client.write_json(paths.release_bundle("v4"), empty)
    _personal(client, "v3")

    assert build_serving_plan(client).release.release_id == "v3"


def test_the_served_format_brings_its_own_weights_and_calibrators(client):
    _personal(client)
    bundle = client.read_json(paths.release_bundle("v3"))
    bundle["serving_formats"]["fp8"] = paths.quantized_model_dir("v3", "fp8", scope="policy")
    bundle["calibrators"]["fp8"] = paths.release_calibrators("v3", "fp8")
    client.write_json(paths.release_bundle("v3"), bundle)

    release = build_serving_plan(client, fmt="fp8").release
    assert release.merged_model == paths.quantized_model_dir("v3", "fp8", scope="policy")
    assert release.calibrators == paths.release_calibrators("v3", "fp8")


def test_each_tenant_serves_its_own_promoted_release(client):
    _personal(client, "v3", tenant_id="acme")

    assert build_serving_plan(client).release is None
    assert build_serving_plan(client, tenant_id="acme").release.release_id == "v3"


def test_an_empty_release_index_serves_nothing_rather_than_guessing(client):
    plan = build_serving_plan(client)
    with pytest.raises(UnservedDocType):
        plan.release_for("policy", "homeowners")


def test_an_old_bundle_with_no_doc_types_covers_every_type(client):
    """Every bundle written before types were recorded carries an empty list,
    and it means every type - so nothing is backfilled."""
    promote(client, "release-2026.9.1", scope="unified", doc_types=())

    plan = build_serving_plan(client)
    assert plan.release.release_id == "release-2026.9.1"
    assert set(plan.served_doc_types) == {"acord", "policy", "lossrun"}
    assert plan.release.adapter_dir == paths.scoped_adapter_dir(None, "v1")    # the unified prefix


def test_the_plan_names_the_release_and_its_lines():
    release = ServedRelease("v3", BASE, "m", doc_types=("policy",), lines=("homeowners",))
    assert ServingPlan(release).describe() == "v3 covering ['policy'], lines ['homeowners']"
    assert "lines" not in ServedRelease("v1", BASE, "m").describe()


# --------------------------------------------------------------------------
# What the release packaged with: prompt hash and window plan, checked on a
# real plan
# --------------------------------------------------------------------------


def test_the_served_release_carries_the_window_plan_its_corpus_was_cut_with(client):
    """It was dropped on the way from the bundle, so the cold-start check read an
    attribute the served release did not have."""
    from data_pipeline.dataset_builder.policy_windows import window_plan_id
    from serving.vllm_entrypoint import assert_window_plan_served

    _personal(client, window_plan_version=window_plan_id())
    plan = build_serving_plan(client)
    assert plan.release.window_plan_version == window_plan_id()
    assert_window_plan_served(plan)


@pytest.mark.parametrize("stale", [None, "2"])
def test_a_policy_release_cut_with_another_window_plan_is_refused_at_cold_start(client, stale):
    from serving.vllm_entrypoint import ColdStartError, assert_window_plan_served

    _personal(client, window_plan_version=stale)
    with pytest.raises(ColdStartError, match="window plan"):
        assert_window_plan_served(build_serving_plan(client))


def test_nothing_promoted_has_no_window_plan_to_check(client):
    from serving.vllm_entrypoint import assert_prompt_hash, assert_window_plan_served

    plan = build_serving_plan(client)
    assert_window_plan_served(plan)
    assert_prompt_hash(plan)


def test_a_release_packaged_on_other_prompts_is_refused_on_a_real_plan(client):
    from common.prompts import prompt_hash
    from serving.vllm_entrypoint import ColdStartError, assert_prompt_hash

    _personal(client)
    bundle = client.read_json(paths.release_bundle("v3"))
    bundle["prompt_hash"] = prompt_hash()
    client.write_json(paths.release_bundle("v3"), bundle)
    assert_prompt_hash(build_serving_plan(client))

    bundle["prompt_hash"] = "0" * 64
    client.write_json(paths.release_bundle("v3"), bundle)
    with pytest.raises(ColdStartError, match="prompt hash"):
        assert_prompt_hash(build_serving_plan(client))


# --------------------------------------------------------------------------
# release_for: the release, or a refusal
# --------------------------------------------------------------------------


def test_a_type_the_release_does_not_cover_is_refused_by_name(client):
    _personal(client)
    plan = build_serving_plan(client)

    with pytest.raises(UnservedDocType, match="no promoted release covers 'lossrun'"):
        plan.release_for("lossrun")


def test_a_line_the_release_did_not_train_on_is_refused(client):
    _personal(client)
    plan = build_serving_plan(client)

    assert plan.release_for("policy", "homeowners").release_id == "v3"
    for lob in ("gl", "wc", None):
        with pytest.raises(UnservedDocType, match="serves policies of"):
            plan.release_for("policy", lob)


@pytest.mark.parametrize("lob,covered", [
    ("homeowners", True),
    ("watercraft", True),                         # L1's code for ocean marine
    ("classic_auto", True),                       # read as personal auto
    # A package of common-model lines is read against _fallback.json, another
    # shape than the adapter learned (common.lob.common_model_package).
    (["homeowners", "personal_auto"], False),
    (["homeowners", "gl"], False),                # one line outside
    ("flood", False),                             # not trained
    ("workers_comp", False),
    (None, False),                                # no line: cannot be shown to be trained
    ([], False),
])
def test_a_release_covers_only_the_lines_it_trained_on(lob, covered):
    release = ServedRelease("v3", BASE, "m", doc_types=("policy",), lines=SNAPSHOT_LINES)
    assert release.covers_lob(lob) is covered


def test_a_release_recording_no_lines_covers_every_line():
    release = ServedRelease("v1", BASE, "m")
    assert release.covers_lob("wc") and release.covers_lob(None)
    assert ServingPlan(release).release_for("policy", ["homeowners", "personal_auto"]) is release


# --------------------------------------------------------------------------
# route: the release, the base model, or a refusal
# --------------------------------------------------------------------------


def test_a_policy_of_a_line_the_release_trained_on_goes_to_it(client):
    _personal(client)
    routed = build_serving_plan(client).route("policy", "personal_auto")
    assert routed.release.release_id == "v3" and not routed.lob_fallback_used


def test_a_known_line_the_release_did_not_train_on_goes_to_the_base_model(client):
    _personal(client)
    plan = build_serving_plan(client)
    for lob in ("agriculture_farm", "gl", ["homeowners", "gl"], ["homeowners", "personal_auto"]):
        assert plan.route("policy", lob) == Routed(None, lob_fallback_used=True), lob


def test_a_policy_with_no_line_or_no_schema_is_refused_unless_the_fallback_is_chosen(client):
    _personal(client)
    plan = build_serving_plan(client)
    for lob in ("title", None):
        with pytest.raises(UnservedDocType, match="allow_lob_fallback"):
            plan.route("policy", lob)
        assert plan.route("policy", lob, allow_fallback=True) == Routed(None, lob_fallback_used=True)


def test_while_nothing_is_promoted_every_known_line_goes_to_the_base_model(client):
    plan = build_serving_plan(client)
    assert plan.route("policy", "homeowners") == Routed(None, lob_fallback_used=True)
    with pytest.raises(UnservedDocType, match="allow_lob_fallback"):
        plan.route("policy", None)
    with pytest.raises(UnservedDocType, match="no promoted release covers 'acord'"):
        plan.route("acord")


def test_an_older_release_recording_no_lines_reads_any_policy(client):
    """Until it is replaced, it reads a lineless policy, a line with no schema
    and a package against the fallback, as it was trained to - and the other
    types as before."""
    promote(client, "release-2026.9.1", scope="unified", doc_types=())
    plan = build_serving_plan(client)
    for lob in (None, "title", "gl", ["homeowners", "personal_auto"]):
        routed = plan.route("policy", lob)
        assert routed.release.release_id == "release-2026.9.1" and not routed.lob_fallback_used, lob
    assert plan.route("acord").release.release_id == "release-2026.9.1"


def test_other_types_route_as_release_for_does(client):
    _personal(client)
    with pytest.raises(UnservedDocType, match="no promoted release covers 'lossrun'"):
        build_serving_plan(client).route("lossrun")


def test_l1_watercraft_is_the_ocean_marine_line(client):
    """lob_schema_map.yaml (D-B): L1 sends a personal boat policy as watercraft."""
    from common.lob import merge_line
    from common.schemas import schema_key
    from serving.vllm_entrypoint import request_lob

    assert merge_line("watercraft") == "ocean_marine"
    assert schema_key("policy", None, "watercraft") == "policy:ocean_marine"
    assert request_lob({"lob": "watercraft"}) == "ocean_marine"
    _personal(client)
    assert build_serving_plan(client).route("policy", "watercraft").release.release_id == "v3"


# --------------------------------------------------------------------------
# The pipeline and the endpoint
# --------------------------------------------------------------------------


def _policy_request(lob, allow=False):
    from serving.pipeline import ExtractionRequest

    return ExtractionRequest(source_id="policy_0001",
                             image_paths=["processed/default/policy/policy_0001/page_1.png"],
                             page_texts={1: "Declarations"}, known_doc_type="policy", known_lob=lob,
                             allow_lob_fallback=allow)


def test_the_pipeline_refuses_an_unserved_type_before_generating(client):
    """Before generation, deliberately: refusing costs nothing, answering costs a
    full extraction and returns something nobody can tell from a real result."""
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, PipelineError, extract

    _personal(client)
    plan = build_serving_plan(client)

    backend = EchoBackend(json.dumps({"carrier": "Sentinel"}))
    model = load_model("base", client, backend_impl=backend)
    request = ExtractionRequest(
        source_id="lossrun_0001",
        image_paths=["processed/default/lossrun/lossrun_0001/page_1.png"],
        ocr_text="claims",
        known_doc_type="lossrun",
    )

    with pytest.raises(PipelineError, match="no promoted release covers"):
        extract(request, model, StaticClassifier("lossrun"), None, plan=plan)

    assert not backend.calls, "the model was asked to extract a type nobody serves"


def test_serving_reads_a_fallback_policy_with_the_base_model_against_the_fallback_schema(client):
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import LOB_FALLBACK_FLAG, PipelineError, extract

    _personal(client)
    plan = build_serving_plan(client)
    backend = EchoBackend(json.dumps({}))
    model = load_model("base", client, backend_impl=backend)

    result = extract(_policy_request("agriculture_farm"), model, StaticClassifier("policy"), None,
                     plan=plan, strict_schema=False)
    assert result.route_info["lob_fallback_used"] and result.route_info["adapter"] is None
    assert "layout_family" not in result.route_info and LOB_FALLBACK_FLAG in result.review_flags
    assert all(call["adapter"] is None for call in backend.calls)
    with pytest.raises(PipelineError, match="allow_lob_fallback"):
        extract(_policy_request("title"), model, StaticClassifier("policy"), None, plan=plan)
    assert extract(_policy_request("title", allow=True), model, StaticClassifier("policy"), None,
                   plan=plan, strict_schema=False).route_info["lob_fallback_used"]


def test_the_fallback_reads_the_fallback_schema_with_no_adapter_at_all(client):
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import extract

    _personal(client)
    plan = build_serving_plan(client)
    backend = EchoBackend(json.dumps({}))
    model = load_model("base", client, backend_impl=backend)
    extract(_policy_request("agriculture_farm"), model, StaticClassifier("policy"), None,
            plan=plan, strict_schema=False)
    served = [call["json_schema"] for call in backend.calls]
    backend.calls.clear()
    extract(_policy_request(None, allow=True), model, StaticClassifier("policy"), None,
            plan=plan, strict_schema=False)
    assert served == [call["json_schema"] for call in backend.calls]         # _fallback.json, as with no line


def test_no_adapter_means_none_even_when_the_model_has_a_default():
    import dataclasses

    from inference_core.model_runner import NO_ADAPTER, EchoBackend, generate, load_model

    backend = EchoBackend(json.dumps({}))
    model = dataclasses.replace(load_model("base", BlobClient(backend=InMemoryBackend(), container="m",
                                                                raw_container="r"), backend_impl=backend),
                                default_adapter="/adapters/policy")
    generate(model, [{"role": "user", "content": "x"}], want_logprobs=False)
    generate(model, [{"role": "user", "content": "x"}], adapter=NO_ADAPTER, want_logprobs=False)
    assert [call["adapter"] for call in backend.calls] == ["/adapters/policy", None]


def test_the_fallback_never_fails_on_calibration_the_endpoint_does_not_hold(client, monkeypatch):
    import serving.printed_values as printed_values
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import extract
    from tests.test_policy_windows import _decl_response

    # Its page text prints none of the answer's values; the printed-values
    # check is not what this tests (tests/test_printed_values.py).
    monkeypatch.setattr(printed_values, "configured_mode", lambda: printed_values.FLAG)
    _personal(client)
    model = load_model("base", client, backend_impl=EchoBackend(json.dumps(_decl_response())))
    result = extract(_policy_request("agriculture_farm"), model, StaticClassifier("policy"), {},
                     plan=build_serving_plan(client), strict_schema=False)
    assert result.route_info["lob_fallback_used"]
    assert result.fields and all(f"{path}:uncalibrated" in result.review_flags for path in result.fields)


def test_an_engine_holding_a_merged_release_refuses_the_fallback_rather_than_mislabel_it(client):
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import PipelineError, extract

    _personal(client)
    model = load_model("base", client, backend_impl=EchoBackend(json.dumps({})))
    with pytest.raises(PipelineError, match="merged model, not the base"):
        extract(_policy_request("agriculture_farm"), model, StaticClassifier("policy"), None,
                plan=build_serving_plan(client), base_model_loaded=False)


def test_a_package_of_personal_lines_is_read_by_the_base_model(client):
    """It is read against _fallback.json; the adapter learned the common model."""
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import extract

    _personal(client)
    backend = EchoBackend(json.dumps({}))
    model = load_model("base", client, backend_impl=backend)
    result = extract(_policy_request(["homeowners", "personal_auto"]), model, StaticClassifier("policy"),
                     None, plan=build_serving_plan(client), strict_schema=False)
    assert result.route_info["lob_fallback_used"] and all(c["adapter"] is None for c in backend.calls)


def test_the_endpoint_loads_the_base_whenever_a_line_would_fall_back(client):
    from common.lob import known_lines
    from serving.vllm_entrypoint import serves_base_with_loras

    assert not serves_base_with_loras(build_serving_plan(client))            # nothing promoted
    promote(client, "release-2026.9.1", scope="unified", doc_types=())
    assert not serves_base_with_loras(build_serving_plan(client))            # it reads every line

    every = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    promote(every, "v3", lines=sorted(known_lines()))
    assert not serves_base_with_loras(build_serving_plan(every))             # trained on every line

    personal = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    _personal(personal)
    assert serves_base_with_loras(build_serving_plan(personal))              # commercial lines need the base


def test_the_endpoint_passes_the_callers_choice_through():
    from serving.vllm_entrypoint import build_request

    payload = {"source_id": "p1", "image_paths": ["page_1.png"], "doc_type": "policy", "allow_lob_fallback": True}
    assert build_request(payload).allow_lob_fallback is True
    assert build_request({**payload, "allow_lob_fallback": False}).allow_lob_fallback is False


def test_a_caller_who_chose_the_fallback_may_name_a_line_with_no_schema():
    from serving.vllm_entrypoint import ServingError, build_request

    payload = {"source_id": "p1", "image_paths": ["page_1.png"], "doc_type": "policy", "lob": "title"}
    with pytest.raises(ServingError, match="no canonical policy schema"):
        build_request(payload)
    assert build_request({**payload, "allow_lob_fallback": True}).known_lob == "title"


def test_the_endpoint_hands_the_serving_plan_to_every_extraction(client, monkeypatch):
    """The plan was built at cold start and passed to nothing, so the refusal it
    exists for never ran in the real endpoint — only in tests that passed
    `plan=` themselves."""
    from serving import vllm_entrypoint as entry

    _personal(client)
    plan = build_serving_plan(client)
    seen: dict = {}

    def fake_extract(request, model, classifier, calibration, **kwargs):
        seen.update(kwargs)
        raise entry.ServingError("stop here — the arguments are what this asserts")

    monkeypatch.setattr(entry, "extract", fake_extract)
    state = entry.EndpointState(
        model_version="v3", ready=True, calibration=object(), plan=plan,
    )
    entry.handler({"input": {
        "source_id": "policy_0001",
        "image_paths": ["processed/default/policy/policy_0001/page_1.png"],
        "ocr_text": "Named Insured",
    }}, state)

    assert seen.get("plan") is plan, "the serving plan never reached extract"
