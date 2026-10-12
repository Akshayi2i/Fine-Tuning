"""Path-aware push/pull helpers (IMPL-02 §3).

``BlobClient`` moves bytes; ``paths`` says where they go. This module is the join:
one named function per artifact class, so **no caller ever hand-builds a Blob
path**.

That is not tidiness. Every path a caller assembles inline is a place the layout
can be got wrong silently — and it was: ``stage_push`` built its own destinations
and published a Foundation run against the ``doc_type=None`` "unified" merged and
quantized paths, which only exist under ``--foundation-only``. The manifest then
pointed at artifacts nobody had built. A named ``push_merged_model(doc_type=...)``
makes that mistake impossible to express.

Split by direction for readability, but kept in one module because a push and its
pull must agree on the path, and two files are two places for them to drift.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)


class TransferError(RuntimeError):
    """Raised when an artifact cannot be moved."""


def _require_dir(local: str | Path, what: str) -> Path:
    path = Path(local)
    if not path.is_dir():
        raise TransferError(
            f"{what} directory {path} does not exist. Pushing an absent artifact would create an "
            "empty Blob prefix that later reads as a published model."
        )
    return path


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


def push_adapter(
    local_dir: str | Path,
    kind: paths.AdapterKind,
    version: str,
    doc_type: str | None = None,
    *,
    client: BlobClient,
) -> str:
    """Push a LoRA adapter to ``adapters/foundation/`` or ``adapters/{doc_type}/``."""
    if kind == "doc_type" and not doc_type:
        raise TransferError("a per-type adapter must name its doc_type; the path depends on it")
    prefix = paths.adapter_dir(kind, version, doc_type)
    count = client.upload_dir(_require_dir(local_dir, "adapter"), prefix)
    log.info("pushed %d adapter file(s) -> %s", count, prefix)
    return prefix


#: What serving, merging and the release read from an adapter folder. A
#: training checkpoint also holds the optimizer and scheduler state a resume
#: needs (~1.5 GB of a ~2 GB rank-64 checkpoint), which no reader of a published
#: adapter opens.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def push_scoped_adapter(
    local_dir: str | Path, scope: str | None, version: str, *, client: BlobClient
) -> str:
    """Push one scope's adapter to :func:`paths.scoped_adapter_dir` — where the
    run manifest and the release bundle say it is.

    The adapter files only (:data:`ADAPTER_FILES`), and none whose bytes Blob
    already holds: a re-push of the same release sends nothing.
    """
    prefix = paths.scoped_adapter_dir(scope, version)
    root = _require_dir(local_dir, "adapter")
    missing = [name for name in ADAPTER_FILES if not (root / name).is_file()]
    if missing:
        raise TransferError(f"{root} holds no {', '.join(missing)}: it is not an adapter to publish")
    count = client.upload_dir(root, prefix, names=ADAPTER_FILES, skip_identical=True)
    log.info("pushed %d adapter file(s) -> %s", count, prefix)
    return prefix


def pull_adapter(
    kind: paths.AdapterKind,
    version: str,
    local_dir: str | Path,
    doc_type: str | None = None,
    *,
    client: BlobClient,
) -> Path:
    prefix = paths.adapter_dir(kind, version, doc_type)
    client.download_dir(prefix, local_dir)
    return Path(local_dir)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def push_merged_model(
    local_dir: str | Path, version: str, doc_type: str | None = None, *, client: BlobClient,
    scope: str | None = None,
) -> str:
    """Push a merged model. ``doc_type=None`` is the **unified** model, which only
    exists when the cycle was built ``--foundation-only``. ``scope`` addresses a
    scoped run's own model."""
    prefix = paths.merged_model_dir(version, doc_type, scope=scope)
    # ~18 GB for an 8B bf16 model: a shard Blob already holds is not sent again.
    count = client.upload_dir(_require_dir(local_dir, "merged model"), prefix, skip_identical=True)
    log.info("pushed %d merged-model file(s) -> %s", count, prefix)
    return prefix


def pull_merged_model(
    version: str, local_dir: str | Path, doc_type: str | None = None, *, client: BlobClient,
    scope: str | None = None,
) -> Path:
    client.download_dir(paths.merged_model_dir(version, doc_type, scope=scope), local_dir)
    return Path(local_dir)


def push_quantized(
    local_dir: str | Path, version: str, fmt: str, doc_type: str | None = None, *,
    client: BlobClient, scope: str | None = None,
) -> str:
    prefix = paths.quantized_model_dir(version, fmt, doc_type, scope=scope)
    count = client.upload_dir(_require_dir(local_dir, "quantized model"), prefix)
    log.info("pushed %d %s file(s) -> %s", count, fmt, prefix)
    return prefix


def pull_quantized(
    version: str, fmt: str, local_dir: str | Path, doc_type: str | None = None, *,
    client: BlobClient, scope: str | None = None,
) -> Path:
    client.download_dir(paths.quantized_model_dir(version, fmt, doc_type, scope=scope), local_dir)
    return Path(local_dir)


def pull_base_model(local_dir: str | Path, *, client: BlobClient, allow_hf: bool = True) -> Path:
    """The pinned base model, from Blob — or from Hugging Face and then cached.

    Caching matters on RunPod: pods are ephemeral, so an uncached base model is
    re-downloaded from HF on every launch, and HF is the one dependency in this
    pipeline that is neither pinned to your infrastructure nor rate-limit free.
    """
    prefix = paths.base_model_dir()
    if client.list(prefix):
        client.download_dir(prefix, local_dir)
        return Path(local_dir)

    if not allow_hf:
        raise TransferError(
            f"the base model is not cached at {prefix} and Hugging Face access was not permitted. "
            "Cache it once with pull_base_model(allow_hf=True) so pods stop depending on HF."
        )
    raise NotImplementedError(
        "wire the Hugging Face download here, at the revision pinned in configs/base_model.yaml, "
        "then upload_dir it to base-models/ so the next pod reads it from Blob. Pinning the "
        "revision is the point: a floating one makes a regression impossible to attribute."
    )


# --------------------------------------------------------------------------
# Corpus, calibration, eval
# --------------------------------------------------------------------------


def push_corpus_version(
    local_dir: str | Path, version: str, *, client: BlobClient, tenant_id: str | None = None
) -> str:
    prefix = paths.corpus_dir(version, tenant_id)
    count = client.upload_dir(_require_dir(local_dir, "corpus"), prefix)
    log.info("pushed %d corpus file(s) -> %s", count, prefix)
    return prefix


def pull_corpus_version(
    version: str, local_dir: str | Path, *, client: BlobClient, tenant_id: str | None = None
) -> Path:
    client.download_dir(paths.corpus_dir(version, tenant_id), local_dir)
    return Path(local_dir)


def push_calibration(
    version: str, doc_type: str, params: dict[str, Any], *, client: BlobClient
) -> str:
    key = paths.calibration_params(version, doc_type)
    client.write_json(key, params)
    return key


def pull_calibration(version: str, doc_type: str, *, client: BlobClient) -> dict[str, Any]:
    key = paths.calibration_params(version, doc_type)
    if not client.exists(key):
        raise TransferError(
            f"no calibration at {key}. Serving raw logprob confidence as if calibrated would make "
            "every downstream review threshold meaningless (IMPL-09), so this raises rather than "
            "falling back."
        )
    return dict(client.read_json(key))


def push_eval_report(
    report: dict[str, Any], version: str, doc_type: str | None = None, *, client: BlobClient
) -> str:
    key = paths.eval_report(version, doc_type)
    client.write_json(key, report)
    return key


def pull_eval_report(
    version: str, doc_type: str | None = None, *, client: BlobClient
) -> dict[str, Any] | None:
    key = paths.eval_report(version, doc_type)
    return dict(client.read_json(key)) if client.exists(key) else None


def push_golden_eval_set(
    local_dir: str | Path, *, client: BlobClient, tenant_id: str | None = None
) -> str:
    """Push one tenant's frozen eval set.

    Frozen and versioned separately from the corpus, and held constant across
    corpus versions — that constancy is the only reason model versions are
    comparable over time (arch §8). Never trained on. Per tenant: each tenant
    is gated on its own set, never on another's.
    """
    prefix = paths.golden_eval_set_dir(tenant_id)
    count = client.upload_dir(_require_dir(local_dir, "golden eval set"), prefix)
    log.info("pushed %d golden eval file(s) -> %s", count, prefix)
    return prefix


def pull_golden_eval_set(
    local_dir: str | Path, *, client: BlobClient, tenant_id: str | None = None
) -> Path:
    client.download_dir(paths.golden_eval_set_dir(tenant_id), local_dir)
    return Path(local_dir)
