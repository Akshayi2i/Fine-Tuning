"""One promoted release per tenant: promote, archive, roll back (decisions 2026-10-22, 2026-10-26).

Every fine-tuning cycle takes the next version (``v1``, ``v2``, ...), and its
release is named by it. ``package`` never promotes (decision 2026-10-26, #44):
a release that passed its gate is written ``gated``. A person promotes it with
``python -m orchestration.run promote --version vN --by NAME`` (:func:`promote`),
which asks for a passing shadow check of vN on the real server path
(``orchestration.run shadow``, :mod:`orchestration.shadow`) - or a written
reason to skip it, recorded with the person's name - and, for an interim-tier
release, ``--accept-interim``. Promoting an older release back (a rollback to
one promoted before, now archived) needs no shadow.

Either way the tenant ends with ONE promoted release - the one serving reads
(:mod:`serving.release_router`) and the gate compares the next candidate with -
and every other release that was promoted is archived. Its weights,
calibrators and gate decision stay where they are, so a later ``promote`` can
bring it back. Each promotion is recorded on the bundle
(``ReleaseBundle.promotions``): who, when, the shadow report or the skip
reason, and the gate's tier and tier confidence.

The run manifests follow the releases: the promoted version's run is marked
``promoted`` and an archived one's ``archived``, so the registry index shows
which version is live.
"""

from __future__ import annotations

import json
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

#: The shortest written reason that promotes a release without a passing shadow
#: check (``promote --skip-shadow --reason``): as long as a gate override's.
SKIP_SHADOW_MIN_CHARS = 20

#: Release statuses a release is promoted from with a shadow check: ``gated``
#: (every release ``package`` writes since 2026-10-26) and ``candidate`` (a
#: bundle written before releases were gated).
SHADOWED_STATUSES = frozenset({"gated", "candidate"})


def shadow_report_key(release_id: str, tenant_id: str | None = None) -> str:
    """``releases/{tenant}/{vN}/shadow/shadow_report.json``: the release's
    shadow check (:mod:`orchestration.shadow`), under the release's own prefix
    so a tenant deletion takes it with the bundle."""
    return f"{paths.release_dir(release_id, tenant_id)}/shadow/shadow_report.json"


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


def _promotable(client: BlobClient, release_id: str,
                tenant_id: str | None) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """``(bundle body, bundle as promoted, bf16 gate decision)``, or the reason
    ``release_id`` cannot be promoted at all (:func:`promote_release`)."""
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
    return body, bundle, decision


def promotable_bundle(client: BlobClient, release_id: str, tenant_id: str | None = None) -> dict[str, Any]:
    """``release_id``'s bundle when nothing but the shadow check (and, for an
    interim tier, ``--accept-interim``) stands between it and a promotion;
    else the reason, as :func:`promote_release` refuses it. Read by the shadow
    check before it reads a document: a release promote would refuse anyway is
    not worth the GPU time."""
    return _promotable(client, release_id, tenant_id)[0]


def gate_tier(client: BlobClient, release_id: str, tenant_id: str | None = None,
              body: dict[str, Any] | None = None) -> tuple[str | None, Any]:
    """The gate's tier of ``release_id`` (production or interim) and its tier
    confidence, where recorded: what a promotion records."""
    decision_key = paths.release_gate_decision(release_id, "bf16", tenant_id)
    decision = client.read_json(decision_key) if client.exists(decision_key) else {}
    if body is None:
        key = paths.release_bundle(release_id, tenant_id)
        body = client.read_json(key) if client.exists(key) else {}
    return _gate_tier(decision, body)


def _gate_tier(decision: dict[str, Any], body: dict[str, Any]) -> tuple[str | None, Any]:
    """The gate's tier (production or interim) and its tier confidence, where recorded."""
    return decision.get("tier") or body.get("tier"), decision.get("tier_confidence")


def previously_promoted(body: dict[str, Any]) -> bool:
    """Whether a release was promoted before: archived by a later promotion, or
    carrying a promotion record. Promoting it again is a rollback."""
    return body.get("status") == "archived" or bool(body.get("promotions"))


def promoted_release_id(client: BlobClient, tenant_id: str | None = None,
                        *, excluding: str | None = None) -> str | None:
    """The tenant's promoted release other than ``excluding``; ``None`` when none is."""
    rows = [str(r.get("release_id")) for r in _rows(client, tenant_id)
            if r.get("status") == "promoted" and r.get("release_id") != excluding]
    return rows[0] if rows else None


def promote_release(
    client: BlobClient,
    release_id: str,
    tenant_id: str | None = None,
    *,
    promoted_by: str,
    record: dict[str, Any] | None = None,
) -> list[str]:
    """Make ``release_id`` the tenant's one promoted release. Returns the ids archived.

    The mechanics of a promotion. The operator's way in is :func:`promote`
    (``orchestration.run promote``), which first asks for a passing shadow
    check, or a written reason to skip it, and ``--accept-interim`` for an
    interim-tier release; ``record`` is what it found (:func:`promotion_clearance`).
    ``package`` never promotes (decision 2026-10-26, #44).

    Refused unless its bf16 gate decision passed (an override is folded into
    ``passed``), the bundle is valid as promoted - every serving format with
    its own calibrator and gate run (``ReleaseBundle``) - a release of the policy
    adapter records the lines it serves, and serving would start with it: the
    prompt hash and window plan it was packaged with are this code's, and a
    release served as a LoRA on the base trained on the configured base
    (:func:`serving_refusal`, which holds the OCR pin too: :func:`ocr_pin_refusal`).
    Promoting a release the endpoint then refuses at cold start would archive
    the one that works.

    Each promotion is appended to the bundle's ``promotions``
    (``ReleasePromotion``): who and when, ``record``'s evidence, and the gate's
    tier and tier confidence. Promoting the release already promoted changes nothing.
    """
    from registry_utils.models import ReleasePromotion

    body, bundle, decision = _promotable(client, release_id, tenant_id)
    every = _rows(client, tenant_id)
    promoted_now = [r.get("release_id") for r in every if r.get("status") == "promoted"]
    if promoted_now == [release_id]:
        log.info("%s is already the promoted release for tenant %s; nothing changed",
                 release_id, paths._tenant(tenant_id))
        return []
    tier, tier_confidence = _gate_tier(decision, body)
    try:
        promotion = ReleasePromotion(**{"promoted_by": promoted_by, "rollback": previously_promoted(body),
                                        "tier": tier, "tier_confidence": tier_confidence, **(record or {})})
    except ValueError as exc:
        raise ReleaseError(f"{release_id} is not promoted: its promotion record is invalid ({exc})") from exc
    rows = [r for r in every if r.get("release_id") != release_id]
    archived = []
    for row in rows:
        if row.get("status") == "promoted":
            row["status"] = "archived"
            archived_body = _set_bundle_status(client, str(row["release_id"]), tenant_id, "archived")
            _mark_run(client, (archived_body or row).get("adapter"), "archived", promoted_by=promoted_by)
            archived.append(str(row["release_id"]))
    client.write_json(paths.release_bundle(release_id, tenant_id), {
        **body, "status": "promoted",
        "promotions": [*(body.get("promotions") or []), json.loads(promotion.model_dump_json())],
    })
    write_index(client, [*rows, bundle.index_row()], tenant_id)
    _mark_run(client, bundle.adapter, "promoted", promoted_by=promoted_by)
    log.info("promoted %s for tenant %s by %s%s", release_id, paths._tenant(tenant_id), promotion.promoted_by,
             f"; archived {', '.join(archived)}" if archived else "")
    return archived


def shadow_problem(client: BlobClient, release_id: str, tenant_id: str | None = None,
                   body: dict[str, Any] | None = None) -> str | None:
    """Why ``release_id``'s shadow report does not let it be promoted; ``None`` when it does.

    The report (:func:`shadow_report_key`, written by ``orchestration.run
    shadow``) must be this release's, of the bundle as it is now packaged
    (``bundle_created_at``: a release packaged again is read again), passed,
    and compared with the release promoted now - the one this promotion
    archives - or with nothing, when nothing is promoted.
    """
    key = shadow_report_key(release_id, tenant_id)
    if not client.exists(key):
        return f"there is no shadow report of {release_id} at {key}"
    if body is None:
        body = client.read_json(paths.release_bundle(release_id, tenant_id))
    report = client.read_json(key) or {}
    if report.get("release_id") != release_id:
        return f"the shadow report at {key} is {report.get('release_id')!r}'s, not {release_id}'s"
    if report.get("bundle_created_at") != body.get("created_at"):
        return (f"the shadow report at {key} read {release_id} as packaged at {report.get('bundle_created_at')}, "
                f"and it was packaged again at {body.get('created_at')}: run the shadow check again")
    if report.get("passed") is not True:
        reasons = "; ".join(str(r) for r in report.get("reasons") or []) or "see the report"
        return f"the shadow check of {release_id} failed ({reasons}; {key})"
    promoted_now = promoted_release_id(client, tenant_id, excluding=release_id)
    compared = report.get("promoted_release_id")
    if compared != promoted_now:
        return (f"the shadow check compared {release_id} with {compared or 'nothing promoted'}, and "
                f"{promoted_now or 'nothing'} is promoted now: run the shadow check again")
    return None


def promotion_clearance(
    client: BlobClient,
    release_id: str,
    tenant_id: str | None = None,
    *,
    body: dict[str, Any] | None = None,
    skip_shadow_reason: str | None = None,
    accept_interim: bool = False,
) -> dict[str, Any]:
    """What lets ``release_id`` be promoted, as its promotion records it, or
    the reason it may not be (decision 2026-10-26, #44).

    * A release promoted before (archived) is a rollback: no shadow and no
      ``accept_interim`` are asked - it was accepted once, and a rollback is
      usually urgent.
    * An interim-tier release (the gate's tier) needs ``accept_interim``.
    * A gated release needs a passing shadow report (:func:`shadow_problem`),
      unless ``skip_shadow_reason`` - written, at least
      :data:`SKIP_SHADOW_MIN_CHARS` characters - says why not; the reason is
      recorded with what was wrong with the shadow.
    """
    if body is None:
        body, _bundle, decision = _promotable(client, release_id, tenant_id)
    else:
        decision_key = paths.release_gate_decision(release_id, "bf16", tenant_id)
        decision = client.read_json(decision_key) if client.exists(decision_key) else {}
    tier, tier_confidence = _gate_tier(decision, body)
    record: dict[str, Any] = {}
    if previously_promoted(body):
        return {"rollback": True}
    if tier == "interim":
        if not accept_interim:
            confidence = f", tier confidence {tier_confidence}" if tier_confidence is not None else ""
            raise ReleaseError(
                f"{release_id} is an interim-tier release (the gate's tier{confidence}): its golden field "
                "match is below the production bar, or was measured on an eval set accepted small. "
                "Promote it knowingly with --accept-interim."
            )
        record["accepted_interim"] = True
    status = body.get("status")
    if status not in SHADOWED_STATUSES:
        raise ReleaseError(
            f"{release_id} is {status!r}: a release is promoted from gated, after its shadow check, or "
            "rolled back to once it was promoted before."
        )
    problem = shadow_problem(client, release_id, tenant_id, body)
    if problem is None:
        return {**record, "shadow_report": shadow_report_key(release_id, tenant_id)}
    reason = (skip_shadow_reason or "").strip()
    if not reason:
        tenant = f" --tenant {tenant_id}" if tenant_id else ""
        raise ReleaseError(
            f"{release_id} is not promoted: {problem}. Read it on the real server path first: "
            f"python -m orchestration.run shadow --version {release_id} --model-url <vLLM URL> "
            f"--mineru-url <MinerU URL>{tenant}, then promote again. To promote it without a passing "
            f"shadow check: --skip-shadow --reason \"<why, {SKIP_SHADOW_MIN_CHARS} characters or more>\", "
            "recorded with your name."
        )
    if len(reason) < SKIP_SHADOW_MIN_CHARS:
        raise ReleaseError(
            f"--reason must say why {release_id} is promoted without its shadow check, in "
            f"{SKIP_SHADOW_MIN_CHARS} characters or more (got {len(reason)})"
        )
    log.warning("%s is promoted without a passing shadow check (%s): %s", release_id, problem, reason)
    return {**record, "shadow_skipped_reason": reason, "shadow_problem": problem}


def promote(
    client: BlobClient,
    release_id: str,
    tenant_id: str | None = None,
    *,
    promoted_by: str,
    skip_shadow_reason: str | None = None,
    accept_interim: bool = False,
) -> list[str]:
    """``python -m orchestration.run promote``: make ``release_id`` the tenant's
    promoted release, by a named person, on the evidence
    :func:`promotion_clearance` asks for. Returns the ids archived.

    What :func:`promote_release` refuses is refused first (the gate, the bundle,
    the lines, :func:`serving_refusal` and :func:`ocr_pin_refusal`), then what
    the clearance asks; the release already promoted changes nothing.
    """
    if not is_named_person(promoted_by):
        raise ReleaseError("a promotion must name the person accountable for it, not a system identity")
    body, _bundle, _decision = _promotable(client, release_id, tenant_id)
    if [r.get("release_id") for r in _rows(client, tenant_id) if r.get("status") == "promoted"] == [release_id]:
        log.info("%s is already the promoted release for tenant %s; nothing changed",
                 release_id, paths._tenant(tenant_id))
        return []
    record = promotion_clearance(client, release_id, tenant_id, body=body,
                                 skip_shadow_reason=skip_shadow_reason, accept_interim=accept_interim)
    return promote_release(client, release_id, tenant_id, promoted_by=promoted_by.strip(), record=record)


def serving_refusal(bundle: dict[str, Any]) -> str | None:
    """Why the serving endpoint would refuse this bundle at cold start, or None.

    The same checks cold start runs (``serving.vllm_entrypoint.assert_prompt_hash``,
    ``assert_window_plan_served`` and ``assert_base_model_served``), on a plan
    holding this release alone - and the OCR pin the Azure worker and
    ``scripts.served_adapter`` hold every release to (:func:`ocr_pin_refusal`):
    a release trained on another MinerU's text, promoted, stops the worker at
    start-up, an outage.
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
    return ocr_pin_refusal(bundle.get("ocr_pin"))


def ocr_pin_refusal(ocr_pin: dict[str, Any] | None, pin: dict[str, str] | None = None) -> str | None:
    """Why serving would refuse a release whose corpus ``ocr_pin`` records, or None.

    ``scripts.served_adapter.assert_ocr_pin`` - what the Azure worker
    (``azure_service.runtime.check_release``) and ``served_adapter`` check at
    start-up: the MinerU version that read the corpus is the served one
    (``common.config.served_mineru``, ``configs/shared/ocr.yaml``); one not
    recorded is refused too. And the tier, where the corpus recorded one: a
    reading at another tier was made by other models (an in-process reading
    records none and is held to its version alone, as ``find_unprocessed``
    holds it).
    """
    from common.config import served_mineru
    from scripts.served_adapter import ServedAdapterError, assert_ocr_pin

    pin = pin or served_mineru()
    recorded = dict(ocr_pin or {})
    try:
        assert_ocr_pin(recorded, pin)
    except ServedAdapterError as exc:
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
