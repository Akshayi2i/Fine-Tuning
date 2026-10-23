"""One promoted release per tenant: promote, archive, roll back (decision 2026-10-22).

Every fine-tuning cycle takes the next version (``v1``, ``v2``, ...), and its
release is named by it. ``package`` promotes the release when its gate passed;
``python -m orchestration.run promote --version vK`` promotes an older one back
(a rollback). Either way the tenant ends with ONE promoted release - the one
serving reads (:mod:`serving.release_router`) and the gate compares the next
candidate with - and every other release that was promoted is archived. Its
weights, calibrators and gate decision stay where they are, so a later
``promote`` can bring it back.

The run manifests follow the releases: the promoted version's run is marked
``promoted`` and an archived one's ``archived``, so the registry index shows
which version is live.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)


class ReleaseError(RuntimeError):
    """Raised when a release cannot be promoted."""


#: Names that are a system, not a person (as a gate override refuses them).
_NOT_A_PERSON = frozenset({"system", "ci", "automated", "n/a", "none"})


def is_named_person(name: str | None) -> bool:
    """Whether ``name`` could be the person accountable for a promotion."""
    return bool(name and name.strip() and name.strip().lower() not in _NOT_A_PERSON)


def _rows(client: BlobClient, tenant_id: str | None) -> list[dict[str, Any]]:
    key = paths.release_index(tenant_id)
    return list(client.read_json(key) or []) if client.exists(key) else []


def write_index(client: BlobClient, rows: list[dict[str, Any]], tenant_id: str | None) -> None:
    """The release index, in version order (``v2`` before ``v10``)."""
    def order(row: dict[str, Any]) -> tuple:
        release_id = str(row.get("release_id", ""))
        number = paths.version_number(release_id)
        return (number is None, number or 0, release_id)

    client.write_json(paths.release_index(tenant_id), sorted(rows, key=order))


def _set_bundle_status(client: BlobClient, release_id: str, tenant_id: str | None,
                       status: str) -> dict[str, Any] | None:
    key = paths.release_bundle(release_id, tenant_id)
    if not client.exists(key):
        return None
    body = client.read_json(key)
    body["status"] = status
    client.write_json(key, body)
    return body


def _mark_run(client: BlobClient, run_id: str | None, status: str, *, promoted_by: str) -> None:
    """Mirror a release's status on its run manifest. A run that cannot be read
    is warned about: the release index is what serving reads."""
    if not run_id:
        return
    from registry_utils.query_registry import RegistryQueryError, get
    from registry_utils.write_run_manifest import write_manifest

    try:
        manifest = get(run_id, client)
    except (RegistryQueryError, KeyError, FileNotFoundError) as exc:
        log.warning("release of %s: its run manifest cannot be read (%s); the registry index "
                    "does not show its status", run_id, type(exc).__name__)
        return
    if status == "promoted":
        manifest.promotion.promoted_by = promoted_by
        manifest.promotion.promoted_at = datetime.now(UTC)
    manifest.status = status  # type: ignore[assignment]
    write_manifest(manifest, client)


def promote_release(
    client: BlobClient,
    release_id: str,
    tenant_id: str | None = None,
    *,
    promoted_by: str,
) -> list[str]:
    """Make ``release_id`` the tenant's one promoted release. Returns the ids archived.

    Refused unless its bf16 gate decision passed (an override is folded into
    ``passed``), the bundle is valid as promoted - every serving format with
    its own calibrator and gate run (``ReleaseBundle``) - a release of the policy
    adapter records the lines it serves, and serving would start with it: the
    prompt hash and window plan it was packaged with are this code's, and a
    release served as a LoRA on the base trained on the configured base
    (:func:`serving_refusal`). Promoting a release the endpoint then refuses at
    cold start would archive the one that works.

    Promoting the release already promoted changes nothing.
    """
    from common.adapter import ADAPTER
    from registry_utils.models import ReleaseBundle

    bundle_key = paths.release_bundle(release_id, tenant_id)
    if not client.exists(bundle_key):
        raise ReleaseError(f"there is no release {release_id} at {bundle_key}")
    decision_key = paths.release_gate_decision(release_id, "bf16", tenant_id)
    decision = client.read_json(decision_key) if client.exists(decision_key) else {}
    if not decision.get("passed"):
        raise ReleaseError(
            f"{release_id} did not pass its promotion gate ({decision_key}: "
            f"{'failed ' + str(decision.get('failed_gates')) if decision else 'no decision'}), "
            "so it is not promoted."
        )
    body = client.read_json(bundle_key)
    try:
        bundle = ReleaseBundle.model_validate({**body, "status": "promoted"})
    except ValueError as exc:
        raise ReleaseError(f"{release_id} cannot be promoted: {exc}") from exc
    if bundle.scope == ADAPTER.name and not bundle.lines:
        raise ReleaseError(
            f"{release_id} records no line of business it trained on, and serving reads that as "
            "every line: it is not promoted."
        )
    refusal = serving_refusal(body)
    if refusal:
        raise ReleaseError(
            f"{release_id} is not promoted: serving would refuse it at cold start ({refusal}). "
            "Promote it from the code it was packaged with, the code serving will run."
        )

    every = _rows(client, tenant_id)
    promoted_now = [r.get("release_id") for r in every if r.get("status") == "promoted"]
    if promoted_now == [release_id]:
        log.info("%s is already the promoted release for tenant %s; nothing changed",
                 release_id, paths._tenant(tenant_id))
        return []
    rows = [r for r in every if r.get("release_id") != release_id]
    archived = []
    for row in rows:
        if row.get("status") == "promoted":
            row["status"] = "archived"
            archived_body = _set_bundle_status(client, str(row["release_id"]), tenant_id, "archived")
            _mark_run(client, (archived_body or row).get("adapter"), "archived", promoted_by=promoted_by)
            archived.append(str(row["release_id"]))
    client.write_json(bundle_key, {**body, "status": "promoted"})
    write_index(client, [*rows, bundle.index_row()], tenant_id)
    _mark_run(client, bundle.adapter, "promoted", promoted_by=promoted_by)
    log.info("promoted %s for tenant %s%s", release_id, paths._tenant(tenant_id),
             f"; archived {', '.join(archived)}" if archived else "")
    return archived


def serving_refusal(bundle: dict[str, Any]) -> str | None:
    """Why the serving endpoint would refuse this bundle at cold start, or None.

    The same checks cold start runs (``serving.vllm_entrypoint.assert_prompt_hash``,
    ``assert_window_plan_served`` and ``assert_base_model_served``), on a plan
    holding this release alone.
    """
    from serving.release_router import ServingPlan, _as_release
    from serving.vllm_entrypoint import (
        ColdStartError,
        assert_base_model_served,
        assert_prompt_hash,
        assert_window_plan_served,
    )

    plan = ServingPlan(release=_as_release(bundle, "bf16"))
    try:
        assert_prompt_hash(plan)
        assert_window_plan_served(plan)
        assert_base_model_served(plan)
    except ColdStartError as exc:
        return str(exc)
    return None


# --------------------------------------------------------------------------
# The next cycle's version
# --------------------------------------------------------------------------


def used_versions(client: BlobClient, tenant_id: str | None = None) -> set[str]:
    """Every version a cycle has already taken, by ANY tenant: the adapter's
    runs (the registry index), and every tenant's corpora and releases.

    One sequence for every tenant, because the run (``policy-vN``), its weights
    and its eval reports are not kept per tenant: two tenants on one number
    would write one run's artifacts over the other's. A corpus counts even
    without a run - a cycle still building its corpus, or one that failed
    before training - so another tenant's cycle cannot take its number.
    ``tenant_id`` is accepted for the callers' symmetry and does not narrow it.
    """
    from common.adapter import ADAPTER
    from common.run_ids import RunIdError, parse_run_id
    from registry_utils.query_registry import list_runs

    used: set[str] = set()
    for row in list_runs(client, include_sweeps=True):
        try:
            parsed = parse_run_id(str(row.get("run_id", "")))
        except RunIdError:
            continue
        if parsed.lineage == ADAPTER.name:
            used.add(parsed.version)
    for root in ("corpus", "releases"):
        # {root}/{tenant}/{version}/...
        for key in client.list(root + "/"):
            parts = key.strip("/").split("/")
            if len(parts) > 2 and paths.version_number(parts[2]) is not None:
                used.add(parts[2])
    return used



def next_cycle_version(client: BlobClient, tenant_id: str | None = None) -> str:
    """The version the next fine-tuning cycle takes: one more than any used."""
    return paths.next_version(used_versions(client, tenant_id))


def version_table(client: BlobClient, tenant_id: str | None = None) -> list[dict[str, Any]]:
    """One row per version of the adapter: its run's status and its release's.

    The registry at a glance: which version is live (``release_status``
    promoted), which were archived by a later promotion, and which never got
    past training or the gate.
    """
    from common.adapter import ADAPTER
    from common.run_ids import RunIdError, parse_run_id
    from registry_utils.query_registry import list_runs

    rows: dict[str, dict[str, Any]] = {}
    for run in list_runs(client):
        try:
            parsed = parse_run_id(str(run.get("run_id", "")))
        except RunIdError:
            continue
        if parsed.lineage != ADAPTER.name:
            continue
        rows[parsed.version] = {
            "version": parsed.version, "run_status": run.get("status"),
            "tenant_id": run.get("tenant_id"), "created_at": run.get("created_at"),
            "release_status": None, "tier": run.get("tier"),
        }
    for release in _rows(client, tenant_id):
        version = str(release.get("release_id", ""))
        if paths.version_number(version) is None:
            continue
        row = rows.setdefault(version, {"version": version, "run_status": None,
                                        "tenant_id": release.get("tenant_scope"),
                                        "created_at": release.get("created_at"), "tier": None})
        row["release_status"] = release.get("status")
    return sorted(rows.values(), key=lambda r: paths.version_number(r["version"]) or 0)
