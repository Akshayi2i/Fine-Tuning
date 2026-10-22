"""Querying the run registry — including the two questions it exists to answer.

``resolve_model_version``
    Turns an operator's tag (``base``, ``v2``) into concrete artifact paths, so
    nobody using the CLI has to know where anything lives.

``adapters_depending_on``
    Turns the arch §12 dependency-upgrade rule into a query. When Foundation
    moves to a new major version, every dependent per-type adapter must be
    re-validated first — this produces that work list, instead of it being
    reconstructed by hand from memory.

``diff_manifests`` supports the third: when a document type regresses, diff the
new manifest against the last-good one and read off whether the corpus, the
Foundation version, the MinerU version, or a hyperparameter moved.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, BlobError
from common.constants import ACTIVE_DOC_TYPES
from common.run_ids import is_valid_run_id, lineage_of, version_of
from registry_utils.models import RunManifest, RunStatus, RunType


class RegistryQueryError(RuntimeError):
    """Raised when a run or model version cannot be resolved."""


class ResolvedModel(dict):
    """Concrete artifact locations for a model tag.

    Keys: ``tag``, ``kind`` (``base`` | ``adapter_stack`` | ``merged`` | ``quantized``),
    ``base_model``, ``foundation_adapter``, ``type_adapter``, ``merged_model``,
    ``quantized_model``, ``artifact_status``, ``from_staging``.
    """


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _index(client: BlobClient) -> list[dict[str, Any]]:
    """Every run in the registry index.

    An empty list means "the registry has no runs". It must never mean "the
    index could not be read": swallowing a BlobError here made a throttled or
    403'd Azure read indistinguishable from a first-ever run, so
    `latest_promoted` returned None, the gate set is_first_version=True, and a
    candidate whose accuracy had collapsed was promoted with nothing to regress
    against. The same swallow silently cleared the Foundation cascade block.
    """
    key = paths.registry_index()
    if not client.exists(key):
        return []
    try:
        return list(client.read_json(key).get("runs", []))
    except BlobError as exc:
        raise RegistryQueryError(
            f"cannot read the registry index at {key}: {exc}. Refusing to continue, because an "
            "unreadable index is not an empty one — treating it as empty turns the promotion gate "
            "into 'first version, nothing to regress against' and lets any regression through."
        ) from exc


def get(run_id: str, client: BlobClient) -> RunManifest:
    """Load one manifest. Searches both lineages, since ``run_id`` alone does not
    say which."""
    row = next((r for r in _index(client) if r.get("run_id") == run_id), None)
    candidates = (
        [paths.run_manifest(
            run_id, row["run_type"], row.get("doc_type"), scope=row.get("scope"),
        )]
        if row
        # "unified" and "foundation" resolve to the same prefix, so this one
        # candidate covers both lineages when the index has no row yet.
        else [paths.run_manifest(run_id, "unified")]
        # A scoped run is filed under its own lineage, which the run id names.
        # A per-type lineage like `acord-adapter` is not a scope name, so it is
        # skipped rather than probed — its own candidate is built below.
        + ([paths.run_manifest(run_id, "scoped", scope=lineage_of(run_id))]
           if is_valid_run_id(run_id) and paths.is_valid_scope(lineage_of(run_id)) else [])
        # ACTIVE_DOC_TYPES rather than a second copy of the same tuple: a
        # hand-written list here stops covering a type the moment one is added.
        + [
            paths.run_manifest(run_id, "per_type_adapter", dt)
            for dt in ACTIVE_DOC_TYPES
        ]
    )
    for key in candidates:
        if client.exists(key):
            return RunManifest.model_validate(client.read_json(key))
    raise RegistryQueryError(f"no manifest found for run_id {run_id!r}")


def list_runs(
    client: BlobClient,
    *,
    run_type: RunType | None = None,
    doc_type: str | None = None,
    scope: str | None = None,
    status: RunStatus | None = None,
    include_sweeps: bool = False,
) -> list[dict[str, Any]]:
    """Index rows, filtered. Sweep runs are excluded unless asked for — they are
    recorded but would otherwise swamp a listing (arch §11a)."""
    rows = _index(client)
    if not include_sweeps:
        rows = [r for r in rows if not r.get("is_sweep_run")]
    if run_type:
        rows = [r for r in rows if r.get("run_type") == run_type]
    if doc_type:
        rows = [r for r in rows if r.get("doc_type") == doc_type]
    if scope:
        # Rows written before scopes existed carry no `scope` — they are the
        # unified run, which is what absence means.
        rows = [r for r in rows if (r.get("scope") or "unified") == scope]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return rows


def adapters_depending_on(foundation_version: str, client: BlobClient) -> list[str]:
    """Every adapter built on a Foundation version.

    The arch §12 rule — when Foundation moves, all dependent adapters must be
    re-validated and likely retrained **before** the new Foundation becomes
    production — is a dependency upgrade, not an automatic cascade. This is the
    work list it produces.
    """
    return sorted(
        r["run_id"]
        for r in _index(client)
        if r.get("foundation_version") == foundation_version and r.get("run_type") == "per_type_adapter"
    )


def latest_promoted(
    client: BlobClient,
    run_type: RunType,
    doc_type: str | None = None,
    *,
    scope: str | None = None,
    tenant: str | None = None,
) -> str | None:
    """The run currently serving for this lineage, or ``None``.

    ``scope`` matters once more than one adapter is in flight: the baseline a
    policy run is gated against is the previous POLICY run, not whichever run
    happens to be newest.

    ``tenant`` (a tenant id, ``paths._tenant(...)``; ``None`` narrows nothing)
    narrows it to that tenant's runs, read from each run's manifest - the index
    row does not carry the tenant. The baseline a tenant's candidate is gated
    against is that tenant's previous release, not another tenant's newer one,
    trained and gated on other documents. A manifest naming no tenant (written
    before training recorded it) is kept: the gate also checks the tenant its
    eval report was scored on.
    """
    rows = [
        r
        for r in list_runs(
            client, run_type=run_type, doc_type=doc_type, scope=scope, status="promoted"
        )
    ]
    if tenant is not None:
        rows = [r for r in rows if (_run_tenant(r["run_id"], client) or tenant) == tenant]
    if not rows:
        return None
    return max(rows, key=lambda r: r.get("created_at") or "")["run_id"]


def _run_tenant(run_id: str, client: BlobClient) -> str | None:
    """The tenant a run's manifest records, or ``None`` when it records none."""
    tenant = get(run_id, client).tenant_id
    return paths._tenant(tenant) if tenant else None


# --------------------------------------------------------------------------
# Version resolution — what the CLI actually calls
# --------------------------------------------------------------------------

def resolve_model_version(
    tag: str,
    client: BlobClient,
    *,
    doc_type: str | None = None,
    quant_format: str | None = None,
) -> ResolvedModel:
    """Resolve an operator tag to concrete artifact paths.

    ``base`` is a first-class tag, not a curiosity: it is the zero-shot baseline
    of the pilot protocol (IMPL-15), the day-zero pre-annotation path (IMPL-04),
    and ``extract --model base``. Supporting it here means those three share one
    implementation rather than drifting apart.

    A ``staged`` version resolves to **staging-volume** paths, so a model can be
    spot-checked with ``extract`` before ``package`` has published it.
    """
    tag = tag.strip().lower()
    from common.config import base_model_source

    # What vLLM and HF LOAD: the pod's local copy, else the Hub id. It used to
    # be "model_id@revision" — an identity string no loader accepts as a path.
    base_ref = base_model_source()

    if tag == "base":
        return ResolvedModel(
            tag="base",
            kind="base",
            base_model=base_ref,
            foundation_adapter=None,
            type_adapter=None,
            merged_model=None,
            quantized_model=None,
            artifact_status="published",
            from_staging=False,
        )

    # BOTH run types. The trainer writes `run_type="unified"` (arch v2.1 §4.1)
    # while this looked only for `"foundation"`, so no run produced by the
    # current code was resolvable by tag at all — `extract --model v2` could not
    # find the model it had just trained.
    lineage_types: tuple[RunType, ...] = ("unified", "foundation")
    foundation_run = _find_run(client, lineage_types, tag)
    if foundation_run is None:
        known = sorted({
            str(r.get("run_id", ""))
            for one in lineage_types
            for r in list_runs(client, run_type=one)
        })
        raise RegistryQueryError(
            f"no unified or Foundation run found for tag {tag!r}. Known versions: {known}"
        )

    manifest = get(foundation_run, client)
    staged = manifest.artifacts.status == "staged"

    def _adapter(kind: paths.AdapterKind, dt: str | None) -> str:
        return (
            paths.staging_adapter_dir(kind, tag, dt) if staged else paths.adapter_dir(kind, tag, dt)
        )

    resolved = ResolvedModel(
        tag=tag,
        kind="adapter_stack",
        base_model=base_ref,
        foundation_adapter=_adapter("foundation", None),
        type_adapter=_adapter("doc_type", doc_type) if doc_type else None,
        merged_model=(
            paths.staging_merged_model_dir(tag, doc_type) if staged else paths.merged_model_dir(tag, doc_type)
        ),
        quantized_model=None,
        artifact_status=manifest.artifacts.status,
        from_staging=staged,
    )

    if quant_format:
        if staged:
            raise RegistryQueryError(
                f"version {tag} is still staged, so no quantized artifact exists yet. "
                "Run `orchestration.run package` first."
            )
        resolved["kind"] = "quantized"
        resolved["quantized_model"] = paths.quantized_model_dir(tag, quant_format, doc_type)
    return resolved


def _tag_of(run_id: str) -> str:
    """The version tag inside a run_id: ``foundation-v2`` -> ``v2``.

    Parsed by :mod:`common.run_ids` rather than by ``rsplit("-", 1)``, which is
    right only by luck: it reads ``extractor-v2.1`` correctly but has no idea
    whether what it returned is a version at all, so a malformed id resolved to
    a tag that matched nothing and reported "no run found" instead of "that is
    not a run id".
    """
    return version_of(run_id) if is_valid_run_id(run_id) else run_id


def _find_run(
    client: BlobClient,
    run_type: RunType | tuple[RunType, ...],
    tag: str,
    *,
    scope: str | None = None,
) -> str | None:
    """Find the run_id whose version tag matches, preferring a promoted one.

    Takes several run types because one lineage is written under more than one:
    the trainer records ``unified`` while v1 recorded ``foundation``, and both
    resolve to the same artifacts.
    """
    wanted = (run_type,) if isinstance(run_type, str) else tuple(run_type)
    rows = [row for one in wanted for row in list_runs(client, run_type=one, scope=scope)]
    # Exact tag match on the run_id's final segment. `tag in run_id` made "v1"
    # match "foundation-v10" and "v1.1", so resolve_model_version("v1") could
    # pick v10, read ITS artifact status, and return a staging path built from
    # the v1 tag — a volume path for a model published in Blob.
    matches = [r for r in rows if _tag_of(str(r.get("run_id", ""))) == tag]
    if not matches:
        return None
    promoted = [r for r in matches if r.get("status") == "promoted"]
    pool = promoted or matches
    return max(pool, key=lambda r: r.get("created_at") or "")["run_id"]


# --------------------------------------------------------------------------
# Regression debugging
# --------------------------------------------------------------------------

def diff_manifests(run_a: str, run_b: str, client: BlobClient) -> dict[str, dict[str, Any]]:
    """Field-by-field delta between two runs.

    The arch §12 regression workflow: when a document type regresses, this says
    whether the corpus, the Foundation version, the MinerU version, or a
    hyperparameter moved — instead of it being reconstructed from memory.
    """
    a = get(run_a, client).model_dump(mode="json")
    b = get(run_b, client).model_dump(mode="json")

    diff: dict[str, dict[str, Any]] = {}

    def walk(x: Any, y: Any, prefix: str = "") -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(x.get(key), y.get(key), f"{prefix}{key}.")
        elif x != y:
            diff[prefix.rstrip(".")] = {"a": x, "b": y}

    walk(a, b)
    return diff


def summarize_diff(diff: dict[str, dict[str, Any]]) -> list[str]:
    """Human-readable diff lines, most decision-relevant first.

    Ordering is deliberate: a corpus or Foundation change explains a regression
    far more often than a metric delta does, so those surface at the top.
    """
    priority = ("dependencies.", "training_config.", "data_stats.", "eval_metrics.", "artifacts.")
    ordered = sorted(
        diff.items(),
        key=lambda kv: next((i for i, p in enumerate(priority) if kv[0].startswith(p)), len(priority)),
    )
    return [f"{field}: {vals['a']!r} -> {vals['b']!r}" for field, vals in ordered]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Query the training run registry")
    sub = parser.add_subparsers(dest="command", required=True)

    p_get = sub.add_parser("get", help="print one run manifest")
    p_get.add_argument("run_id")

    p_list = sub.add_parser("list", help="list runs")
    p_list.add_argument("--run-type", choices=["foundation", "per_type_adapter"])
    p_list.add_argument("--doc-type")
    p_list.add_argument("--status")
    p_list.add_argument("--include-sweeps", action="store_true")

    p_dep = sub.add_parser("depends-on", help="adapters built on a Foundation version")
    p_dep.add_argument("foundation_version")

    p_res = sub.add_parser("resolve", help="resolve a model tag to artifact paths")
    p_res.add_argument("tag")
    p_res.add_argument("--doc-type")
    p_res.add_argument("--format", dest="quant_format")

    p_diff = sub.add_parser("diff", help="diff two runs")
    p_diff.add_argument("run_a")
    p_diff.add_argument("run_b")

    args = parser.parse_args(list(argv) if argv is not None else None)
    client = BlobClient()

    if args.command == "get":
        print(json.dumps(get(args.run_id, client).model_dump(mode="json"), indent=2))
    elif args.command == "list":
        for row in list_runs(
            client, run_type=args.run_type, doc_type=args.doc_type,
            status=args.status, include_sweeps=args.include_sweeps,
        ):
            print(f"{row['run_id']:<32} {row['status']:<10} {row.get('created_at', '')}")
    elif args.command == "depends-on":
        for run_id in adapters_depending_on(args.foundation_version, client):
            print(run_id)
    elif args.command == "resolve":
        print(json.dumps(resolve_model_version(
            args.tag, client, doc_type=args.doc_type, quant_format=args.quant_format), indent=2))
    elif args.command == "diff":
        for line in summarize_diff(diff_manifests(args.run_a, args.run_b, client)):
            print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
