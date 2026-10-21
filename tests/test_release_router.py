"""Which promoted release answers for which document type (arch v2.1 §12.3).

Once scopes exist, several releases can be promoted at once. The rule this file
pins is the one the operator was promised: **a policy release takes policies
while an older unified release keeps ACORDs and Loss Runs**, and a type nothing
covers is refused rather than extracted by a model that never trained on it.
"""

from __future__ import annotations

import json

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from serving.release_router import (
    ServingPlanError,
    UnservedDocType,
    assert_one_base_model,
    build_serving_plan,
)

BASE = "qwen3-vl-8b-instruct@a1b2c3d"


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def promote(
    client: BlobClient,
    release_id: str,
    *,
    scope: str = "unified",
    doc_types: list[str] | None = None,
    created_at: str = "2026-09-01T00:00:00+00:00",
    base_model: str = BASE,
    status: str = "promoted",
    adapter: str | None = None,
) -> None:
    """Write a release bundle and its index row, as `package` leaves them."""
    bundle = {
        "release_id": release_id,
        "status": status,
        "scope": scope,
        "doc_types": doc_types or [],
        "tenant_scope": "default",
        "created_at": created_at,
        "base_model": base_model,
        "adapter": adapter or f"{scope}-v1",
        "merged_model": f"merged-models/{scope}/v1",
        "serving_formats": {"bf16": f"merged-models/{scope}/v1"},
        "calibrators": {"bf16": f"releases/default/{release_id}/calibrators/bf16.json"},
        "prompt_hash": "abc123",
        "ocr_pin": {"mineru_version": "2.0.0"},
    }
    client.write_json(paths.release_bundle(release_id), bundle)

    index_key = paths.release_index()
    rows = client.read_json(index_key) if client.exists(index_key) else []
    rows = [r for r in rows if r.get("release_id") != release_id]
    rows.append({"release_id": release_id, "status": status, "scope": scope,
                 "doc_types": doc_types or [], "created_at": created_at})
    client.write_json(index_key, rows)


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


def test_a_narrower_release_takes_its_types_and_leaves_the_rest(client):
    """The whole point: train policy on its own, serve it for policies, and keep
    serving everything else from the release that was gated on it."""
    promote(client, "release-2026.9.1", scope="unified")
    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"],
            created_at="2026-10-01T00:00:00+00:00")

    plan = build_serving_plan(client)

    assert plan.release_for("policy").release_id == "release-2026.10.1"
    assert plan.release_for("acord").release_id == "release-2026.9.1"
    assert plan.release_for("lossrun").release_id == "release-2026.9.1"


def test_an_older_narrow_release_still_wins_over_a_newer_broad_one(client):
    """Narrowest first, not newest first: a release naming a type was trained and
    gated on that type specifically, while a unified release covers it as one of
    many. A new unified release does not silently take policies back."""
    promote(client, "release-2026.9.1", scope="policy", doc_types=["policy"])
    promote(client, "release-2026.11.1", scope="unified",
            created_at="2026-11-01T00:00:00+00:00")

    plan = build_serving_plan(client)

    assert plan.release_for("policy").release_id == "release-2026.9.1"
    assert plan.release_for("acord").release_id == "release-2026.11.1"


def test_the_newer_release_wins_between_two_of_equal_breadth(client):
    promote(client, "release-2026.9.1", scope="policy", doc_types=["policy"])
    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"],
            created_at="2026-10-01T00:00:00+00:00")

    assert build_serving_plan(client).release_for("policy").release_id == "release-2026.10.1"


def test_an_old_bundle_with_no_doc_types_covers_everything(client):
    """Every bundle written before scopes existed carries an empty list, and it
    means the unified release — so nothing is backfilled."""
    promote(client, "release-2026.9.1", scope="unified", doc_types=[])

    plan = build_serving_plan(client)
    assert set(plan.served_doc_types) == {"acord", "policy", "lossrun"}


def test_only_promoted_releases_serve(client):
    """A gated release is one that did not clear the gate. Serving it would make
    the gate advisory."""
    promote(client, "release-2026.9.1", scope="unified", status="gated")

    plan = build_serving_plan(client)
    assert not plan.by_doc_type
    assert plan.unserved_doc_types == ("acord", "lossrun", "policy")


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------


def test_a_type_no_release_covers_is_refused_by_name(client):
    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"])

    plan = build_serving_plan(client)

    assert plan.served_doc_types == ("policy",)
    with pytest.raises(UnservedDocType, match="no promoted release covers 'lossrun'"):
        plan.release_for("lossrun")


def test_an_empty_release_index_serves_nothing_rather_than_guessing(client):
    plan = build_serving_plan(client)
    with pytest.raises(UnservedDocType):
        plan.release_for("policy")


def test_the_pipeline_refuses_an_unserved_type_before_generating(client):
    """Before generation, deliberately: refusing costs nothing, answering costs a
    full extraction and returns something nobody can tell from a real result."""
    from inference_core.model_runner import EchoBackend, load_model
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, PipelineError, extract

    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"])
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


# --------------------------------------------------------------------------
# Pins and the shared base
# --------------------------------------------------------------------------


def test_a_pin_overrides_the_precedence_rule(client):
    """For a rollback, or to hold one type on an older release while a new one is
    watched."""
    promote(client, "release-2026.9.1", scope="unified")
    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"],
            created_at="2026-10-01T00:00:00+00:00")

    plan = build_serving_plan(client, pins={"policy": "release-2026.9.1"})
    assert plan.release_for("policy").release_id == "release-2026.9.1"


def test_a_pin_to_a_release_that_does_not_cover_the_type_is_an_error(client):
    """A pin is an explicit instruction about where a document goes. Ignoring one
    would serve a type from the release the operator routed away from."""
    promote(client, "release-2026.9.1", scope="unified")
    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"])

    with pytest.raises(ServingPlanError, match="which covers"):
        build_serving_plan(client, pins={"lossrun": "release-2026.10.1"})

    with pytest.raises(ServingPlanError, match="not a promoted release"):
        build_serving_plan(client, pins={"policy": "release-2027.1.1"})


def test_releases_on_different_base_models_cannot_be_served_together(client):
    """vLLM loads one base and applies a LoRA per request, so this is caught at
    cold start rather than as a load failure on the first request routed to the
    odd one out."""
    promote(client, "release-2026.9.1", scope="unified")
    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"],
            base_model="qwen3-vl-8b-instruct@different")

    with pytest.raises(ServingPlanError, match="different base models"):
        build_serving_plan(client)


def test_one_base_model_is_checked_on_the_served_set_only(client):
    """A promoted release nothing routes to cannot break the endpoint."""
    promote(client, "release-2026.9.1", scope="unified")
    plan = build_serving_plan(client)
    assert_one_base_model(plan)


def test_the_endpoint_hands_the_serving_plan_to_every_extraction(client, monkeypatch):
    """The plan was built at cold start and passed to nothing, so the refusal it
    exists for never ran in the real endpoint — only in tests that passed
    `plan=` themselves."""
    from serving import vllm_entrypoint as entry

    promote(client, "release-2026.10.1", scope="policy", doc_types=["policy"])
    plan = build_serving_plan(client)
    seen: dict = {}

    def fake_extract(request, model, classifier, calibration, **kwargs):
        seen.update(kwargs)
        raise entry.ServingError("stop here — the arguments are what this asserts")

    monkeypatch.setattr(entry, "extract", fake_extract)
    state = entry.EndpointState(
        model_version="v1", ready=True, calibration=object(), plan=plan,
    )
    entry.handler({"input": {
        "source_id": "policy_0001",
        "image_paths": ["processed/default/policy/policy_0001/page_1.png"],
        "ocr_text": "Named Insured",
    }}, state)

    assert seen.get("plan") is plan, "the serving plan never reached extract"


