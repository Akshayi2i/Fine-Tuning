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


#: ``continued_training`` rules and their values when unset: every rule off.
CONTINUED_TRAINING_DEFAULTS: dict[str, Any] = {
    "fresh_when_new_share_above": None,
    "fresh_every_n_cycles": None,
    "fresh_when_golds_changed": False,
}


class SettingsError(ValueError):
    """A ``pipeline.yaml`` value that would not do what it says."""


def continued_training_rules(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """When a cycle starts from the base instead of continuing the promoted adapter
    (``pipeline_dag.fresh_by_rule``), checked: every rule off unless set.

    Refused rather than read loosely: a share of 40 (meant as 40%) would never
    fire, a cycle count of 0 would fire every cycle, and a key spelt otherwise
    would be silently unused.
    """
    block = (config if config is not None else pipeline_config()).get("continued_training") or {}
    if not isinstance(block, dict):
        raise SettingsError(f"continued_training in pipeline.yaml must be a mapping, got {block!r}")
    unknown = sorted(set(block) - set(CONTINUED_TRAINING_DEFAULTS))
    if unknown:
        raise SettingsError(f"continued_training in pipeline.yaml has unknown key(s) {unknown}; "
                            f"expected {sorted(CONTINUED_TRAINING_DEFAULTS)}")
    rules = {**CONTINUED_TRAINING_DEFAULTS, **{k: v for k, v in block.items() if v is not None}}
    share = rules["fresh_when_new_share_above"]
    if share is not None and (isinstance(share, bool) or not isinstance(share, (int, float))
                              or not 0 <= share < 1):
        raise SettingsError("continued_training.fresh_when_new_share_above is a share of the documents, "
                            f"0 to below 1 (e.g. 0.4), or null; got {share!r}")
    every = rules["fresh_every_n_cycles"]
    if every is not None and (isinstance(every, bool) or not isinstance(every, int) or every < 1):
        raise SettingsError(f"continued_training.fresh_every_n_cycles is a whole number of cycles, 1 or more "
                            f"(e.g. 4), or null; got {every!r}")
    if not isinstance(rules["fresh_when_golds_changed"], bool):
        raise SettingsError("continued_training.fresh_when_golds_changed is true or false; "
                            f"got {rules['fresh_when_golds_changed']!r}")
    return rules


def notification_settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Which run events post to the webhook (``orchestration.notify``)."""
    block = (config if config is not None else pipeline_config()).get("notifications") or {}
    return {
        "webhook_url_env": str(block.get("webhook_url_env") or "FIDEON_NOTIFY_WEBHOOK_URL"),
        "on_stage_failure": bool(block.get("on_stage_failure", True)),
        "on_gate_block": bool(block.get("on_gate_block", True)),
        "on_run_complete": bool(block.get("on_run_complete", True)),
        "timeout_seconds": float(block.get("timeout_seconds", 10) or 10),
    }


def staging_mount() -> str:
    volume = pipeline_config().get("staging_volume", {})
    return str(volume.get("mount_default", "/runpod-volume"))


def pod_image() -> str:
    return str(pipeline_config().get("pod", {}).get("image", ""))
