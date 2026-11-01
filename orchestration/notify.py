"""What a cycle tells people while it runs: a webhook post, and the experiment tracker.

**The webhook** (``notifications`` in ``orchestration/config/pipeline.yaml``):
a JSON POST on a stage that failed, a gate that blocked, and a command that
finished, to the URL in the environment variable the config names
(``FIDEON_NOTIFY_WEBHOOK_URL``) - HTTPS only (``common.transport.assert_encrypted``).
Unset, nothing is posted: the events are logged, as before. The body says which
run, version, stage, status and tier, and a short reason - the first line of the
stage's message, cut short - never a document's content: no page text, no
extracted value, no gold. A post that fails is logged and never fails the run.

**Text messages** (e-mail, and Telegram which is kept but off): what a person following the run wants to
read - the run started, each phase finished and the next one starting, a phase that failed or was
blocked, a stall, and the end with the gate's result. Which channels are on is ``notifications.channels``
in ``orchestration/config/pipeline.yaml`` (e-mail only for now). A channel needs its settings in the
environment (``.env`` on the pod; never in the repo):

* E-mail: ``FIDEON_NOTIFY_EMAIL_TO``, ``FIDEON_NOTIFY_SMTP_HOST``, ``FIDEON_NOTIFY_SMTP_USER``,
  ``FIDEON_NOTIFY_SMTP_PASSWORD``; optionally ``FIDEON_NOTIFY_SMTP_PORT`` (587; 465 is SSL) and
  ``FIDEON_NOTIFY_EMAIL_FROM``.
* Telegram (off until ``telegram`` is added to ``channels``): ``FIDEON_NOTIFY_TELEGRAM_TOKEN`` (from
  @BotFather) and ``FIDEON_NOTIFY_TELEGRAM_CHAT_ID`` (``python -m orchestration.notify
  --telegram-chat-id`` finds it).
* Webhook, which needs no mailbox password (``webhook`` in ``channels``): each message is posted as
  ``{"source", "subject", "text"}`` JSON to ``FIDEON_NOTIFY_WEBHOOK_URL`` (HTTPS), for example a Power
  Automate flow "When an HTTP request is received" that sends the e-mail or a Teams message under the
  signed-in user's own login. The URL is a secret.

A channel that is off or not set up sends nothing. A message carries a version, a phase name, a time and a short reason - never
a document's content - and a failed send is logged and never fails the run.
``python -m orchestration.notify --test`` sends a test message through whatever is set.

**The tracker** (``registry_utils.write_run_manifest.log_to_tracker``): each
stage's wall time, as it finishes, beside what each manifest write logs. A
no-op unless ``MLFLOW_TRACKING_URI`` is set.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

#: What a phase is called in a message, by stage name.
STAGE_LABELS = {
    "ingestion": "Ingestion", "preprocessing": "OCR and preprocessing", "labeling": "Labeling",
    "dataset_build": "Dataset build", "training": "Training", "checkpoint_eval": "Checkpoint selection",
    "calibrate": "Calibration", "evaluation_gate": "Evaluation gate", "package": "Packaging",
    "serving": "Serving deploy", "feedback_loop": "Feedback loop",
}

#: Longest message sent: a status line, never a document.
MAX_MESSAGE_CHARS = 1000

#: The subject of every e-mail (and the webhook's subject field): the same line each time, so the inbox
#: shows one thread of updates and the status itself is in the body.
EMAIL_SUBJECT = "Fideon Fine-Tuning update"

#: Written, in the log folder, when the run itself has sent its closing message (a failure, a block, the
#: end), so scripts/pod_run.sh does not send a second one for the same exit.
NOTIFIED_MARKER_ENV = "FIDEON_RUN_NAME"

#: The run in progress, for the messages and for a stall to say where it was.
_RUN: dict[str, Any] = {"version": None, "command": None, "stages": [], "current": None}

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
        if "webhook" in settings.get("channels", []):
            # The same facts reach the webhook as a text message (send_text), once, not twice.
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


TELEGRAM_TOKEN_ENV = "FIDEON_NOTIFY_TELEGRAM_TOKEN"
TELEGRAM_CHAT_ENV = "FIDEON_NOTIFY_TELEGRAM_CHAT_ID"
EMAIL_TO_ENV = "FIDEON_NOTIFY_EMAIL_TO"
EMAIL_FROM_ENV = "FIDEON_NOTIFY_EMAIL_FROM"
SMTP_HOST_ENV = "FIDEON_NOTIFY_SMTP_HOST"
SMTP_PORT_ENV = "FIDEON_NOTIFY_SMTP_PORT"
SMTP_USER_ENV = "FIDEON_NOTIFY_SMTP_USER"
SMTP_PASSWORD_ENV = "FIDEON_NOTIFY_SMTP_PASSWORD"


def enabled_channels() -> list[str]:
    """The channels ``notifications.channels`` turns on: e-mail only unless the config adds more."""
    from orchestration.settings import notification_settings

    try:
        return list(notification_settings()["channels"])
    except Exception:  # noqa: BLE001 - an unreadable config falls back to the default, not to silence
        return ["email"]


def text_channels() -> list[str]:
    """The channels that are both turned on (:func:`enabled_channels`) and set up in the environment."""
    on = enabled_channels()
    channels = []
    if "telegram" in on and os.environ.get(TELEGRAM_TOKEN_ENV) and os.environ.get(TELEGRAM_CHAT_ENV):
        channels.append("telegram")
    if "email" in on and os.environ.get(EMAIL_TO_ENV) and os.environ.get(SMTP_HOST_ENV):
        channels.append("email")
    if "webhook" in on and webhook_url():
        channels.append("webhook")
    return channels


def _redact(text: str) -> str:
    """``text`` without any credential the environment holds: a failed request's message can carry the
    bot token, since Telegram puts it in the URL."""
    for name in (TELEGRAM_TOKEN_ENV, SMTP_PASSWORD_ENV, "FIDEON_NOTIFY_WEBHOOK_URL"):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, "***")
    return text


def _telegram_call(method: str, payload: dict[str, Any], timeout: float, http: Any = None) -> dict[str, Any]:
    token = os.environ[TELEGRAM_TOKEN_ENV]
    url = f"https://api.telegram.org/bot{token}/{method}"
    if http is not None:
        return http(url, payload, timeout)
    import urllib.request

    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https host
        return json.loads(response.read() or b"{}")


def _send_email(subject: str, text: str, timeout: float, smtp_factory: Any = None) -> None:
    from email.message import EmailMessage

    host, to = os.environ[SMTP_HOST_ENV], os.environ[EMAIL_TO_ENV]
    port = int(os.environ.get(SMTP_PORT_ENV) or 587)
    user, password = os.environ.get(SMTP_USER_ENV), os.environ.get(SMTP_PASSWORD_ENV)
    message = EmailMessage()
    message["Subject"], message["To"] = subject, to
    message["From"] = os.environ.get(EMAIL_FROM_ENV) or user or to
    message.set_content(text)
    if smtp_factory is not None:
        server = smtp_factory(host, port, timeout)
    else:
        import smtplib

        server = smtplib.SMTP_SSL(host, port, timeout=timeout) if port == 465 else smtplib.SMTP(host, port, timeout=timeout)
    import smtplib

    with server:
        if port != 465 and smtp_factory is None:
            server.starttls()
        if user and password:
            try:
                server.login(user, password)
            except smtplib.SMTPServerDisconnected as exc:
                # What smtplib reports when a server drops the connection after refusing the login
                # (Gmail does, for an account that is not its own): not a network fault.
                raise RuntimeError(
                    f"{host} closed the connection while logging in: wrong host, user or password for this "
                    "mailbox (Outlook needs smtp.office365.com)") from exc
        server.send_message(message)


def send_text(text: str, *, subject: str | None = None, timeout: float = 10.0, http: Any = None,
              smtp_factory: Any = None) -> dict[str, str]:
    """Send ``text`` on every text channel that is set up: ``{channel: "sent" | the reason it was not}``.

    Never raises: a message that does not arrive is no reason for a run to fail. Empty when no channel is
    set up. Credentials are redacted from the reasons.
    """
    text = text if len(text) <= MAX_MESSAGE_CHARS else text[: MAX_MESSAGE_CHARS - 3].rstrip() + "..."
    outcome: dict[str, str] = {}
    for channel in text_channels():
        try:
            if channel == "telegram":
                reply = _telegram_call("sendMessage", {"chat_id": os.environ[TELEGRAM_CHAT_ENV], "text": text},
                                       timeout, http)
                if reply.get("ok") is False:
                    raise RuntimeError(str(reply.get("description") or "Telegram refused the message"))
            elif channel == "webhook":
                from common.transport import assert_encrypted

                url = webhook_url()
                assert_encrypted(url, "the notification webhook")
                body = json.dumps({"source": "fideon-finetuning", "subject": subject or EMAIL_SUBJECT,
                                   "text": text}, ensure_ascii=False).encode("utf-8")
                (http or _post)(url, body, timeout)
            else:
                _send_email(subject or EMAIL_SUBJECT, text, timeout, smtp_factory)
            outcome[channel] = "sent"
        except Exception as exc:  # noqa: BLE001 - a message never fails a run
            outcome[channel] = _redact(short_reason(f"{type(exc).__name__}: {exc}"))
            log.warning("text message not sent on %s: %s", channel, outcome[channel])
    return outcome


def _enabled(key: str) -> bool:
    from orchestration.settings import notification_settings

    try:
        return bool(notification_settings().get(key, True))
    except Exception:  # noqa: BLE001 - an unreadable config does not silence a failure message
        return True


def _span(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "an unknown time"
    minutes, rest = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {rest:02d}s" if hours else f"{minutes}m {rest:02d}s"


def phase_name(number: int, name: str) -> str:
    return f"Phase {number} '{STAGE_LABELS.get(name, name)}'"


def run_started(report: Any, ctx: Any, stages: Any) -> None:
    """A command is about to run ``stages``: remember them, and say the run started."""
    _RUN.update(version=report.version, command=report.command,
                stages=[(stage.number, stage.name) for stage in stages], current=None)
    _clear_notified()
    if _RUN["stages"] and _enabled("on_run_start"):
        number, name = _RUN["stages"][0]
        send_text(f"Fine-tuning {report.version} started ({report.command}). Starting {phase_name(number, name)}.")


def stage_started(stage: Any) -> None:
    """The stage now running, so a stall can say where it was."""
    _RUN["current"] = (stage.number, stage.name)


def _following(name: str) -> tuple[int, str] | None:
    stages = _RUN["stages"]
    for index, (_number, stage_name) in enumerate(stages):
        if stage_name == name:
            return stages[index + 1] if index + 1 < len(stages) else None
    return None


def _notified_marker() -> str | None:
    run = os.environ.get(NOTIFIED_MARKER_ENV)
    folder = os.environ.get("FIDEON_LOG_DIR") or "/workspace/logs"
    return os.path.join(folder, f"{run}.notified") if run and os.path.isdir(folder) else None


def _mark_notified() -> None:
    marker = _notified_marker()
    if marker:
        with contextlib.suppress(OSError):
            open(marker, "w", encoding="utf-8").close()


def _clear_notified() -> None:
    marker = _notified_marker()
    if marker and os.path.exists(marker):
        with contextlib.suppress(OSError):
            os.remove(marker)


def _phase_events(report: Any, result: Any) -> None:
    """The text messages for a stage that finished, failed or was blocked."""
    version = report.version
    number = next((n for n, name in _RUN["stages"] if name == result.name), None)
    phase = phase_name(number, result.name) if number is not None else f"stage '{result.name}'"
    if result.status == "completed":
        following = _following(result.name)
        if following and _enabled("on_phase_change"):
            send_text(f"Fine-tuning {version}: {phase} completed in {_span(result.data.get('wall_seconds'))}. "
                      f"Now starting {phase_name(*following)}.")
    elif result.status == "failed":
        send_text(f"Fine-tuning {version} STOPPED: {phase} failed. {short_reason(result.detail)}")
        _mark_notified()
    elif result.status == "blocked":
        verdict = "Gate is not passed (blocked). " if result.name == "evaluation_gate" else ""
        send_text(f"Fine-tuning {version} STOPPED: {phase} was blocked. {verdict}{short_reason(result.detail)}")
        _mark_notified()


def stalled(reason: str) -> None:
    """A call ran past its limit and the process is about to stop (inference_core.stall_guard)."""
    if not _enabled("on_stall"):
        return
    version, current = _RUN["version"], _RUN["current"]
    where = phase_name(*current) if current else "a generation call"
    run = f"Fine-tuning {version}" if version else "Fine-tuning"
    send_text(f"{run} STALLED in {where}: {short_reason(reason)} The process is stopping; "
              "resume it with --from-stage.", timeout=5.0)
    _mark_notified()


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
    try:
        _phase_events(report, result)
    except Exception as exc:  # noqa: BLE001 - a message never fails a run
        log.warning("phase message not sent: %s", short_reason(f"{type(exc).__name__}: {exc}"))
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
    if report.ok:
        try:
            send_text(_closing_text(report, ctx))
            _mark_notified()
        except Exception as exc:  # noqa: BLE001 - a message never fails a run
            log.warning("closing message not sent: %s", short_reason(f"{type(exc).__name__}: {exc}"))


def _closing_text(report: Any, ctx: Any) -> str:
    """The message for a command that ended well: everything done, with the gate's result, or the phases
    this command covered and what comes next."""
    version = report.version
    packaged = any(r.name == "package" and r.status == "completed" for r in report.results)
    if packaged:
        gate = getattr(ctx, "results", {}).get("evaluation_gate")
        gate_line = (gate.data.get("gate_message") if gate is not None else None) or "Gate result: see the gate decision"
        tier = release_tier(ctx)
        return (f"Fine-tuning {version} is completed, all phases done successfully. {gate_line}."
                + (f" Release tier: {tier}." if tier else ""))
    stages = _RUN["stages"]
    covered = f"phases {stages[0][0]} to {stages[-1][0]}" if stages else "its phases"
    return (f"Fine-tuning {version}: {covered} of the {report.command} command are done. "
            "Next: the package command (calibration, evaluation gate, packaging).")


def main(argv: list[str] | None = None) -> int:
    """``python -m orchestration.notify --test | --text MESSAGE | --telegram-chat-id``."""
    import argparse

    parser = argparse.ArgumentParser(description="Send a fine-tuning status message, or check the setup.")
    parser.add_argument("--test", action="store_true", help="send a test message on every channel set up")
    parser.add_argument("--text", help="send this message (used by scripts/pod_run.sh)")
    parser.add_argument("--telegram-chat-id", action="store_true",
                        help="list the chats that have written to the bot: send it any message first")
    args = parser.parse_args(argv)

    if args.telegram_chat_id:
        token = os.environ.get(TELEGRAM_TOKEN_ENV)
        if not token:
            print(f"{TELEGRAM_TOKEN_ENV} is not set.")
            return 1
        try:
            reply = _telegram_call("getUpdates", {}, 10.0)
        except Exception as exc:  # noqa: BLE001
            print(f"Telegram did not answer: {_redact(short_reason(f'{type(exc).__name__}: {exc}'))}")
            return 1
        chats = {}
        for update in reply.get("result") or []:
            chat = (update.get("message") or update.get("channel_post") or {}).get("chat") or {}
            if chat.get("id") is not None:
                chats[chat["id"]] = chat.get("first_name") or chat.get("title") or chat.get("username") or ""
        if not chats:
            print("No chat has written to the bot yet: open the bot in Telegram, press Start, send any message, "
                  "and run this again.")
            return 1
        for chat_id, name in chats.items():
            print(f"{TELEGRAM_CHAT_ENV}={chat_id}   ({name})")
        return 0

    if args.text:
        send_text(args.text)       # a script's message: its failure is not the script's failure
        return 0

    if args.test:
        channels = text_channels()
        if not channels:
            print(f"No channel is on and set up (on: {', '.join(enabled_channels()) or 'none'}). E-mail needs "
                  f"{EMAIL_TO_ENV} and {SMTP_HOST_ENV} (with {SMTP_USER_ENV} and {SMTP_PASSWORD_ENV} for a "
                  "server that logs in). Telegram is off until notifications.channels in "
                  "orchestration/config/pipeline.yaml lists it.")
            return 1
        outcome = send_text("Fideon fine-tuning: this is a test message. If you can read it, the status "
                            "messages are set up.")
        for channel, result in outcome.items():
            print(f"{channel}: {result}")
        return 0 if all(result == "sent" for result in outcome.values()) else 1

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
