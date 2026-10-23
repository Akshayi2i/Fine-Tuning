"""One promoted release per tenant: promote, archive, roll back (registry_utils.releases).

``package`` promotes a cycle's release when its gate passed; ``promote --version``
puts an older one back. Either way the tenant ends with ONE promoted release -
the one serving reads - and the one it replaced is archived, its weights and
gate decision left where they are so a later promotion can bring it back. The
run manifests follow, so the registry shows which version is live.
"""

from __future__ import annotations

import json
import logging

import pytest

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, InMemoryBackend
from registry_utils import query_registry as Q
from registry_utils import write_run_manifest as W
from registry_utils.releases import (
    ReleaseError,
    is_named_person,
    next_cycle_version,
    promote_release,
    used_versions,
    version_table,
    write_index,
)
from serving.release_router import build_serving_plan
from tests.test_registry import _bundle, _scoped


def _served_pins() -> dict:
    """What this code serves with: a release packaged by it carries both."""
    from common.prompts import prompt_hash
    from data_pipeline.dataset_builder.policy_windows import window_plan_id

    return {"prompt_hash": prompt_hash(), "window_plan_version": window_plan_id()}


@pytest.fixture
def client() -> BlobClient:
    return BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")


def _release(client, version, *, tenant_id=None, passed=True, calibrated=True, run=True,
             created_at="2026-10-01T00:00:00+00:00", **over):
    """A candidate release of the adapter's ``version``: its bundle, its index
    row, its bf16 gate decision and (``run``) the run manifest behind it."""
    merged = paths.merged_model_dir(version, scope="policy")
    bundle = _bundle(
        release_id=version, adapter=f"policy-{version}", tenant_scope=tenant_id or paths.default_tenant(),
        merged_model=merged, serving_formats={"bf16": merged}, created_at=created_at,
        calibrators={"bf16": paths.release_calibrators(version, "bf16", tenant_id)} if calibrated else {},
        gate_reports={"bf16": paths.release_gate_decision(version, "bf16", tenant_id)},
        **{"lines": ["homeowners"], **_served_pins(), **over},
    )
    client.write_json(paths.release_bundle(version, tenant_id), json.loads(bundle.model_dump_json()))
    index_key = paths.release_index(tenant_id)
    rows = client.read_json(index_key) if client.exists(index_key) else []
    write_index(client, [r for r in rows if r["release_id"] != version] + [bundle.index_row()], tenant_id)
    if passed is not None:
        client.write_json(paths.release_gate_decision(version, "bf16", tenant_id), {
            "passed": passed, "failed_gates": [] if passed else ["field_normalized_match"]})
    if run:
        W.write_manifest(_scoped(f"policy-{version}", "policy", ["policy"]), client)
    return bundle


def _statuses(client, tenant_id=None):
    return {r["release_id"]: r["status"] for r in client.read_json(paths.release_index(tenant_id))}


# --------------------------------------------------------------------------
# Promotion leaves one promoted release
# --------------------------------------------------------------------------


def test_promoting_a_release_archives_the_one_it_replaces(client):
    _release(client, "v2")
    assert promote_release(client, "v2", promoted_by="package v2") == []
    _release(client, "v3")

    archived = promote_release(client, "v3", promoted_by="package v3")

    assert archived == ["v2"]
    assert _statuses(client) == {"v2": "archived", "v3": "promoted"}
    assert client.read_json(paths.release_bundle("v2"))["status"] == "archived"
    assert client.read_json(paths.release_bundle("v3"))["status"] == "promoted"
    assert build_serving_plan(client).release.release_id == "v3"
    # The runs follow their releases.
    live = Q.get("policy-v3", client)
    assert live.status == "promoted" and live.promotion.promoted_by == "package v3"
    assert live.promotion.promoted_at is not None
    assert Q.get("policy-v2", client).status == "archived"


def test_a_rollback_promotes_an_older_version_back(client):
    for version in ("v2", "v3"):
        _release(client, version)
        promote_release(client, version, promoted_by="A. Reviewer")

    assert promote_release(client, "v2", promoted_by="A. Reviewer") == ["v3"]
    assert _statuses(client) == {"v2": "promoted", "v3": "archived"}
    assert build_serving_plan(client).release.release_id == "v2"
    # Its weights and gate decision were left where they were.
    assert client.exists(paths.release_gate_decision("v3", "bf16"))


def test_a_release_whose_gate_did_not_pass_is_not_promoted(client):
    _release(client, "v2")
    promote_release(client, "v2", promoted_by="package v2")
    _release(client, "v3", passed=False)

    with pytest.raises(ReleaseError, match=r"did not pass its promotion gate .*field_normalized_match"):
        promote_release(client, "v3", promoted_by="package v3")
    assert _statuses(client) == {"v2": "promoted", "v3": "candidate"}

    _release(client, "v4", passed=None)                                  # no decision written at all
    with pytest.raises(ReleaseError, match="no decision"):
        promote_release(client, "v4", promoted_by="package v4")


def test_a_release_missing_its_calibrators_is_not_promoted(client):
    """Promoted is a state the bundle must validate in: every served format with
    its own calibrator and gate run (ReleaseBundle)."""
    _release(client, "v2")
    promote_release(client, "v2", promoted_by="package v2")
    _release(client, "v3", calibrated=False)

    with pytest.raises(ReleaseError, match=r"(?s)v3 cannot be promoted: .*no calibrator"):
        promote_release(client, "v3", promoted_by="package v3")
    assert _statuses(client) == {"v2": "promoted", "v3": "candidate"}
    assert client.read_json(paths.release_bundle("v2"))["status"] == "promoted"


def test_a_release_that_does_not_exist_cannot_be_promoted(client):
    with pytest.raises(ReleaseError, match="there is no release v9"):
        promote_release(client, "v9", promoted_by="A. Reviewer")


def test_a_promotion_archives_nothing_of_another_tenant(client):
    _release(client, "v2")
    promote_release(client, "v2", promoted_by="package v2")
    _release(client, "v3", tenant_id="acme")

    assert promote_release(client, "v3", "acme", promoted_by="package v3") == []
    assert _statuses(client) == {"v2": "promoted"}
    assert _statuses(client, "acme") == {"v3": "promoted"}


def test_a_run_manifest_that_cannot_be_read_does_not_stop_the_promotion(client, caplog):
    """The release index is what serving reads; the run's status only mirrors it."""
    _release(client, "v3", run=False)
    with caplog.at_level(logging.WARNING, logger="registry_utils.releases"):
        promote_release(client, "v3", promoted_by="package v3")
    assert _statuses(client) == {"v3": "promoted"}
    assert "policy-v3" in caplog.text and "cannot be read" in caplog.text


def test_the_release_index_is_kept_in_version_order(client):
    rows = [{"release_id": rid, "status": "archived"} for rid in ("v10", "release-2026.9.1", "v2", "v3")]
    write_index(client, rows, None)
    assert [r["release_id"] for r in client.read_json(paths.release_index())] == [
        "v2", "v3", "v10", "release-2026.9.1"]


@pytest.mark.parametrize("name,person", [
    ("A. Reviewer", True), ("akshay", True),
    ("ci", False), (" System ", False), ("automated", False), ("", False), ("  ", False), (None, False),
])
def test_only_a_named_person_is_accountable_for_a_promotion(name, person):
    assert is_named_person(name) is person


# --------------------------------------------------------------------------
# The next cycle's version, and the version table
# --------------------------------------------------------------------------


def test_the_next_cycle_takes_one_more_than_any_version_used(client):
    """Runs of the adapter and EVERY tenant's corpora and releases count: a cycle
    that failed before training, or is still building its corpus, left one, and
    the run and its weights are not kept per tenant - two tenants on one number
    would write one run's artifacts over the other's."""
    assert next_cycle_version(client) == "v1"
    W.write_manifest(_scoped("policy-v1", "policy", ["policy"]), client)
    W.write_manifest(_scoped("policy-v2", "policy", ["policy"]), client)
    W.write_manifest(_scoped("lossrun-v9", "lossrun", ["lossrun"]), client)      # not the adapter
    client.write_json(paths.corpus_manifest("v4"), {})                            # failed before training
    client.write_json(paths.release_bundle("v3"), {})
    client.write_json(paths.corpus_manifest("v8", "acme"), {})                    # another tenant's

    assert used_versions(client) == {"v1", "v2", "v3", "v4", "v8"}
    assert next_cycle_version(client) == "v9"
    assert next_cycle_version(client, "acme") == "v9"


def test_a_cycle_still_building_its_corpus_keeps_its_number_from_another_tenant(client):
    """acme's v4 has a corpus and no run yet: the default tenant's next cycle is v5."""
    W.write_manifest(_scoped("policy-v3", "policy", ["policy"]), client)
    client.write_json(paths.corpus_manifest("v4", "acme"), {})
    assert next_cycle_version(client) == "v5"


def test_the_version_table_shows_which_version_is_live(client):
    _release(client, "v2", created_at="2026-09-01T00:00:00+00:00")
    promote_release(client, "v2", promoted_by="package v2")
    _release(client, "v3")
    promote_release(client, "v3", promoted_by="package v3")
    W.write_manifest(_scoped("policy-v1", "policy", ["policy"], tenant_id="acme"), client)  # never released
    W.write_manifest(_scoped("policy-v10", "policy", ["policy"], status="failed"), client)
    W.write_manifest(_scoped("lossrun-v3", "lossrun", ["lossrun"]), client)                # not the adapter

    table = version_table(client)

    assert [row["version"] for row in table] == ["v1", "v2", "v3", "v10"]
    by_version = {row["version"]: row for row in table}
    assert (by_version["v3"]["run_status"], by_version["v3"]["release_status"]) == ("promoted", "promoted")
    assert (by_version["v2"]["run_status"], by_version["v2"]["release_status"]) == ("archived", "archived")
    assert by_version["v1"]["release_status"] is None and by_version["v1"]["tenant_id"] == "acme"
    assert by_version["v10"]["run_status"] == "failed"
    assert set(by_version["v3"]) == {"version", "run_status", "tenant_id", "created_at",
                                     "release_status", "tier"}


def test_a_release_with_no_run_in_the_registry_still_has_a_row(client):
    _release(client, "v4", run=False)
    (row,) = version_table(client)
    assert (row["version"], row["run_status"], row["release_status"]) == ("v4", None, "candidate")


# --------------------------------------------------------------------------
# What promotion refuses
# --------------------------------------------------------------------------


def test_a_release_serving_would_refuse_at_cold_start_is_not_promoted(client):
    """Packaged with another prompt hash, or another window plan: the endpoint
    refuses it at cold start, so promoting it would archive the release that works."""
    _release(client, "v2")
    promote_release(client, "v2", promoted_by="package v2")
    _release(client, "v3", prompt_hash="sha256:other")
    with pytest.raises(ReleaseError, match="serving would refuse it at cold start"):
        promote_release(client, "v3", promoted_by="Akshay")
    _release(client, "v4", window_plan_version="3:old")
    with pytest.raises(ReleaseError, match="window plan"):
        promote_release(client, "v4", promoted_by="Akshay")
    assert _statuses(client)["v2"] == "promoted"


def test_a_release_with_no_lines_is_not_promoted(client):
    """An empty list reads as every line at serving."""
    _release(client, "v2", lines=[])
    with pytest.raises(ReleaseError, match="no line of business"):
        promote_release(client, "v2", promoted_by="Akshay")


def test_promoting_the_promoted_release_changes_nothing(client):
    """Its run keeps who promoted it, and when."""
    _release(client, "v2")
    promote_release(client, "v2", promoted_by="package v2")
    before = Q.get("policy-v2", client).promotion
    assert promote_release(client, "v2", promoted_by="Someone Else") == []
    after = Q.get("policy-v2", client).promotion
    assert (after.promoted_by, after.promoted_at) == (before.promoted_by, before.promoted_at)
