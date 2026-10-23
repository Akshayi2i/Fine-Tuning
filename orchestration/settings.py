"""Loader for ``orchestration/config/pipeline.yaml`` (IMPL-13 §9).

A config file nothing reads is worse than no config file: it documents a policy
the system does not follow. Everything in ``pipeline.yaml`` is loaded here and
used — GPU class per stage, retry policy, and the CLI's own defaults — so
changing a value there changes what runs.

**Secrets are not here.** The file names the environment variables that carry
them; it never carries them itself.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

from common.config import load_yaml

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "pipeline.yaml"


@functools.lru_cache(maxsize=1)
def pipeline_config(path: Path | None = None) -> dict[str, Any]:
    """The parsed pipeline config. Cached — it does not change mid-run."""
    return load_yaml(path or CONFIG_PATH)


def gpu_class_for(stage: str, fallback: str = "H200-SXM") -> str:
    """The GPU class a stage runs on."""
    return str(pipeline_config().get("gpu_class_by_stage", {}).get(stage, fallback))


def retry_policy() -> dict[str, Any]:
    """How many times a *failed* stage is retried.

    A blocked gate is never retried and the config says so explicitly: retrying
    a gate block would be an override path with extra steps.
    """
    policy = dict(pipeline_config().get("retry", {}))
    policy.setdefault("max_attempts", 1)
    policy.setdefault("retry_on_gate_block", False)
    return policy


def backoff_seconds() -> int:
    """Seconds to wait between stage retries.

    Declared in ``pipeline.yaml`` and read by nothing until now, so a retry fired
    within milliseconds of the failure — inside the same throttle window that
    caused it, which makes the retry a guaranteed second failure.
    """
    return int(retry_policy().get("backoff_seconds", 0) or 0)


def defaults() -> dict[str, Any]:
    """Values the CLI uses when a flag is not given."""
    return dict(pipeline_config().get("defaults", {}))


def staging_mount() -> str:
    volume = pipeline_config().get("staging_volume", {})
    return str(volume.get("mount_default", "/runpod-volume"))


def pod_image() -> str:
    return str(pipeline_config().get("pod", {}).get("image", ""))
