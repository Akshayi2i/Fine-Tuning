"""Writing run manifests, and keeping the flat index in step.

Every training, evaluation, and quantization job writes a manifest — **no silent
runs, sweeps included** (arch §12). The manifest goes to Azure Blob even when the
weights are only staged on the RunPod volume: staging is working storage with no
durability guarantee, so without this a reclaimed volume would leave a training
run that happened and left no trace (master §12a).

``registry_index.json`` is maintained alongside as a flat table, so the whole
history is visible without opening manifests one at a time.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, BlobError
from registry_utils.models import RunManifest

log = logging.getLogger(__name__)

#: Retries for the shared index blob. Small on purpose: this is contention
#: detection, not a locking scheme — a store with conditional writes is the real
#: fix, and the error below says so rather than retrying forever.
_INDEX_WRITE_ATTEMPTS = 3


class RegistryError(RuntimeError):
    """Raised when a manifest cannot be written or the index cannot be updated."""


def capture_git_commit(short: bool = True) -> str:
    """The exact repo commit that ran this job.

    Returns ``"unknown"`` rather than raising when git is unavailable — a pod
    that clones a tarball can still train — but the manifest then records that
    the run is not reproducible, which is the honest answer.
    """
    try:
        # Built directly rather than filtered. `args.index(a)` returns the FIRST
        # match, so with short=False both "HEAD" entries were dropped and the
        # command became a bare `git rev-parse` — which git rejects, so
        # code_git_commit recorded "unknown" inside a healthy worktree.
        args = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
        out = subprocess.run(args, capture_output=True, text=True, timeout=5, check=True)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def is_dirty_worktree() -> bool:
    """Whether uncommitted changes exist.

    A dirty tree means ``code_git_commit`` does not fully describe what ran, so
    callers warn rather than silently recording a commit that is not the truth.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=True
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def write_manifest(
    manifest: RunManifest,
    client: BlobClient,
    *,
    update_index: bool = True,
) -> str:
    """Write the manifest to Blob and refresh the index. Returns the blob key.

    Always writes to **Blob**, never only to staging, regardless of whether the
    weights are staged or published.
    """
    key = paths.run_manifest(
        manifest.run_id, manifest.run_type, manifest.doc_type, scope=manifest.scope
    )
    payload = manifest.model_dump(mode="json")
    try:
        client.write_json(key, payload)
    except BlobError as exc:
        raise RegistryError(f"failed writing manifest for {manifest.run_id}: {exc}") from exc

    # Rewritten with the manifest, so the card always says what it says. A card
    # that fails to write is a missing document, not a failed run.
    from registry_utils.model_card import write_model_card

    try:
        write_model_card(manifest, client)
    except Exception as exc:  # noqa: BLE001 - the manifest above is the record
        log.warning("model card for %s not written: %s", manifest.run_id, exc)

    if update_index:
        update_registry_index(manifest, client)
    # Mirrored to the tracker with the stage times so far: a no-op unless
    # MLFLOW_TRACKING_URI is set, and never a reason for the write to fail.
    log_to_tracker(manifest, stage_seconds=_STAGE_SECONDS.get(manifest.run_id))
    return key


def update_registry_index(manifest: RunManifest, client: BlobClient) -> None:
    """Append or replace this run's row in ``registry_index.json``.

    Replace-by-``run_id`` rather than append-only: a run's row changes as it moves
    from trained to evaluated to promoted, and two rows for one run would make the
    index ambiguous exactly when it is being used to answer "what is live?".
    """
    index_key = paths.registry_index()
    row = manifest.index_row()

    # Read-modify-write against a shared blob, re-read and re-checked before it
    # is accepted. Two adapter runs finishing together each read the same N rows
    # and each wrote N+1, so the later write dropped the earlier run's row — and
    # the index is exactly what adapters_depending_on and latest_promoted read,
    # so a lost row makes a run invisible to the queries the registry exists for.
    # The manifest blob itself is per-run and never contended.
    for attempt in range(_INDEX_WRITE_ATTEMPTS):
        try:
            index: dict[str, Any] = (
                client.read_json(index_key) if client.exists(index_key) else {"runs": []}
            )
        except BlobError:
            index = {"runs": []}

        rows = [r for r in index.get("runs", []) if r.get("run_id") != manifest.run_id]
        rows.append(row)
        rows.sort(key=lambda r: (r.get("created_at") or "", r.get("run_id") or ""))

        index["runs"] = rows
        index["updated_at"] = datetime.now(UTC).isoformat()
        client.write_json(index_key, index)

        # Confirm nothing that was there before this write has gone missing.
        try:
            written = {r.get("run_id") for r in client.read_json(index_key).get("runs", [])}
        except BlobError:
            return
        if {r.get("run_id") for r in rows} <= written:
            return
        log.warning(
            "registry index lost rows during a concurrent write (attempt %d of %d); retrying",
            attempt + 1, _INDEX_WRITE_ATTEMPTS,
        )

    raise RegistryError(
        f"could not record {manifest.run_id} in the registry index after "
        f"{_INDEX_WRITE_ATTEMPTS} attempts — concurrent writers keep overwriting each other. "
        "The run manifest itself is written and safe; the index is what adapters_depending_on "
        "and latest_promoted read, so serialise the writers or move it to a store with "
        "conditional writes before running jobs in parallel."
    )


def mark_promoted(
    manifest: RunManifest,
    client: BlobClient,
    *,
    promoted_by: str,
    gated_against: str | None = None,
) -> RunManifest:
    """Record a promotion. Only ever called by the gate, never by hand.

    The gate is a hard stop: a candidate is promoted only if it matches or beats
    the current production version on every gating metric (arch §13). There is
    deliberately no ``force`` parameter here — adding one would make every other
    guarantee in the pipeline advisory.
    """
    # Requires the affirmative. `is False` let a manifest that was never gated
    # through — the field defaults to None, and only the gate ever sets it — so
    # skipping evaluation promoted cleanly and the registry index could not tell
    # that run from one that actually passed.
    if manifest.promotion.beat_previous_on_all_gates is not True:
        raise RegistryError(
            f"{manifest.run_id} has not passed the evaluation gate "
            f"(beat_previous_on_all_gates={manifest.promotion.beat_previous_on_all_gates!r}, "
            f"failed gates {manifest.promotion.failed_gates}) and cannot be promoted. A run that "
            "was never gated has not passed one; the gate has no override path by design "
            "(arch §13)."
        )
    manifest.promotion.promoted_by = promoted_by
    manifest.promotion.promoted_at = datetime.now(UTC)
    if gated_against:
        manifest.promotion.gated_against = gated_against
    manifest.status = "promoted"
    write_manifest(manifest, client)
    return manifest


def mark_published(
    manifest: RunManifest,
    client: BlobClient,
    *,
    adapter_weights: str | None = None,
    merged_model: str | None = None,
    quantized_model: str | None = None,
    quantized_formats: list[str] | None = None,
) -> RunManifest:
    """Flip a staged manifest to published once ``package`` has pushed to Blob."""
    artifacts = manifest.artifacts
    if adapter_weights:
        artifacts.adapter_weights = adapter_weights
    if merged_model:
        artifacts.merged_model = merged_model
    if quantized_model:
        artifacts.quantized_model = quantized_model
    if quantized_formats:
        artifacts.quantized_formats = list(quantized_formats)
    artifacts.status = "published"
    artifacts.staging_path = None
    manifest.artifacts = artifacts  # re-run the model validators
    write_manifest(manifest, client)
    return manifest


#: Each run's stage wall times so far, ``{run id: {stage: seconds}}``: recorded as
#: each stage of ``pipeline_dag.run_stages`` finishes (``orchestration.notify``),
#: logged with every manifest write after it.
_STAGE_SECONDS: dict[str, dict[str, float]] = {}

#: The tag an MLflow run carries the run id in, so later writes find it again.
TRACKER_RUN_TAG = "fideon_run_id"


def record_stage_seconds(run_id: str, stage: str, seconds: float) -> None:
    """A stage of ``run_id`` took ``seconds``: kept for its next manifest write,
    and logged to the tracker now (a no-op without one)."""
    _STAGE_SECONDS.setdefault(run_id, {})[stage] = float(seconds)
    log_to_tracker(None, run_id=run_id, stage_seconds={stage: float(seconds)})


def log_to_tracker(manifest: RunManifest | None, *, run_id: str | None = None,
                   stage_seconds: dict[str, float] | None = None) -> None:
    """Mirror the manifest - and each stage's wall time - into MLflow when configured.

    A no-op unless ``MLFLOW_TRACKING_URI`` is set. The tracker adds a queryable
    UI; the Blob manifest is the durable record that travels with the artifacts
    (arch §12), so the tracker is never the source of truth and never blocks a
    run: every failure here is logged and swallowed.

    One MLflow run per run id, found again by its :data:`TRACKER_RUN_TAG` tag:
    every manifest write and every stage logs into the same run, rather than
    one MLflow run per write. ``stage_seconds`` go in as ``stage_seconds.<stage>``.
    """
    import os

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    run_id = run_id or (manifest.run_id if manifest is not None else None)
    if not run_id:
        return
    try:
        import mlflow
    except ImportError:
        log.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed; %s is not tracked", run_id)
        return
    try:
        found = mlflow.search_runs(filter_string=f"tags.{TRACKER_RUN_TAG} = '{run_id}'",
                                   max_results=1, output_format="list")
        existing = found[0].info.run_id if found else None
        with (mlflow.start_run(run_id=existing) if existing
              else mlflow.start_run(run_name=run_id, tags={TRACKER_RUN_TAG: run_id})):
            if manifest is not None:
                try:
                    mlflow.log_params(manifest.training_config.model_dump(mode="json"))
                except Exception as exc:  # noqa: BLE001 - a param logged before with another value
                    log.debug("tracker: params of %s not logged again (%s)", run_id, exc)
                metrics = {
                    k: v
                    for k, v in manifest.eval_metrics.model_dump(mode="json").items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                if metrics:
                    mlflow.log_metrics(metrics)
                mlflow.set_tag("status", str(manifest.status))
            if stage_seconds:
                mlflow.log_metrics({f"stage_seconds.{stage}": float(seconds)
                                    for stage, seconds in stage_seconds.items()})
    except Exception as exc:  # noqa: BLE001 - tracking must never fail a run
        log.warning("tracker: %s not logged (%s: %s)", run_id, type(exc).__name__, exc)
