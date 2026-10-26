"""What a cycle tells people while it runs: a webhook post, and the experiment tracker.

**The webhook** (``notifications`` in ``orchestration/config/pipeline.yaml``):
a JSON POST on a stage that failed, a gate that blocked, and a command that
finished, to the URL in the environment variable the config names
(``FIDEON_NOTIFY_WEBHOOK_URL``) - HTTPS only (``common.transport.assert_encrypted``).
Unset, nothing is posted: the events are logged, as before. The body says which
run, version, stage, status and tier, and a short reason - the first line of the
stage's message, cut short - never a document's content: no page text, no
extracted value, no gold. A post that fails is logged and never fails the run.

**The tracker** (``registry_utils.write_run_manifest.log_to_tracker``): each
stage's wall time, as it finishes, beside what each manifest write logs. A
no-op unless ``MLFLOW_TRACKING_URI`` is set.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

#: Longest reason sent: enough to say what failed, too short to carry a document.
MAX_REASON_CHARS = 200

#: Events, and the ``notifications`` switch each obeys.
EVENT_SWITCH = {
    "stage_failure": "on_stage_failure",
    "gate_block": "on_gate_block",
    "run_complete": "on_run_complete",
}


def short_reason(text: Any) -> str:
    """The first line of ``text``, at most :data:`MAX_REASON_CHARS` characters."""
    line = next((part.strip() for part in str(text or "").splitlines() if part.strip()), "")
    return line if len(line) <= MAX_REASON_CHARS else line[: MAX_REASON_CHARS - 3].rstrip() + "..."


def webhook_url(settings: dict[str, Any] | None = None) -> str | None:
    """The webhook's URL from the environment, or None when none is set."""
    from orchestration.settings import notification_settings

    settings = settings or notification_settings()
    return (os.environ.get(settings["webhook_url_env"]) or "").strip() or None


def event_body(event: str, *, run_id: str | None, version: str | None, stage: str | None,
               status: str | None, tier: str | None, reason: Any = None,
               command: str | None = None, tenant: str | None = None) -> dict[str, Any]:
    """The JSON a webhook is sent: identifiers and outcome only."""
    return {
        "source": "fideon-finetuning",
        "event": event,
        "run_id": run_id,
        "version": version,
        "command": command,
        "tenant": tenant,
        "stage": stage,
        "status": status,
        "tier": tier,
        "reason": short_reason(reason) if reason else None,
    }


def notify(event: str, body: dict[str, Any], *, settings: dict[str, Any] | None = None,
           post: Any = None) -> bool:
    """Post ``body`` for ``event`` when the config asks and a URL is set. True when posted.

    ``post(url, data, timeout)`` is injectable for tests; the default is urllib.
    Never raises: a notification is never a reason for a run to fail.
    """
    from orchestration.settings import notification_settings

    try:
        settings = settings or notification_settings()
        if not settings.get(EVENT_SWITCH.get(event, ""), False):
            return False
        url = webhook_url(settings)
        log.info("notification %s: %s", event, json.dumps(body, ensure_ascii=False))
        if not url:
            return False
        from common.transport import assert_encrypted

        assert_encrypted(url, "the notification webhook")
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        (post or _post)(url, data, float(settings.get("timeout_seconds") or 10))
        return True
    except Exception as exc:  # noqa: BLE001 - a notification never fails a run
        log.warning("notification %s not sent: %s", event, short_reason(f"{type(exc).__name__}: {exc}"))
        return False


def _post(url: str, data: bytes, timeout: float) -> None:
    import urllib.request

    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - HTTPS checked above
        response.read()


def release_tier(ctx: Any) -> str | None:
    """The tier the cycle's release was gated at (its bf16 gate decision), or None."""
    from artifact_registry import paths

    try:
        key = paths.release_gate_decision(ctx.release_id, "bf16", ctx.tenant_id)
        return ctx.client.read_json(key).get("tier") if ctx.client.exists(key) else None
    except Exception:  # noqa: BLE001 - no decision readable: no tier to report
        return None


def after_stage(report: Any, ctx: Any) -> None:
    """After a stage finished, failed or blocked: its wall time to the tracker,
    and a webhook post for a failure or a block. Never raises."""
    if not report.results:
        return
    result = report.results[-1]
    try:
        from common.adapter import ADAPTER
        from registry_utils.write_run_manifest import record_stage_seconds

        run_id = ADAPTER.run_id(ctx.out_version) if ctx.out_version else None
        seconds = result.data.get("wall_seconds")
        if run_id and isinstance(seconds, (int, float)):
            record_stage_seconds(run_id, result.name, float(seconds))
    except Exception as exc:  # noqa: BLE001 - tracking never fails a run
        log.debug("stage time not tracked: %s", exc)
        run_id = None
    event = {"failed": "stage_failure", "blocked": "gate_block"}.get(result.status)
    if event:
        notify(event, event_body(
            event, run_id=run_id, version=report.version, stage=result.name, status=result.status,
            tier=release_tier(ctx) if event == "gate_block" else None, reason=result.detail,
            command=report.command, tenant=getattr(ctx, "tenant_id", None)))


def run_complete(report: Any, ctx: Any) -> None:
    """A command finished - completed, failed or blocked: one post saying how."""
    from common.adapter import ADAPTER

    last = report.results[-1] if report.results else None
    status = "completed" if report.ok else ("blocked" if report.blocked_at else "failed")
    notify("run_complete", event_body(
        "run_complete", run_id=ADAPTER.run_id(report.version) if report.version else None,
        version=report.version, stage=report.blocked_at or report.failed_at or (last.name if last else None),
        status=status, tier=release_tier(ctx),
        reason=None if report.ok else (last.detail if last else None),
        command=report.command, tenant=getattr(ctx, "tenant_id", None)))
