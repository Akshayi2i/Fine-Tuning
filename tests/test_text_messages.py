"""The status messages a person following a run reads: Telegram and e-mail, never a document's content."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_core import stall_guard
from orchestration import notify as N

TOKEN, CHAT = "123456:SECRET-BOT-TOKEN", "98765"
PASSWORD = "app-password-secret"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for name in (N.TELEGRAM_TOKEN_ENV, N.TELEGRAM_CHAT_ENV, N.EMAIL_TO_ENV, N.EMAIL_FROM_ENV, N.SMTP_HOST_ENV,
                 N.SMTP_PORT_ENV, N.SMTP_USER_ENV, N.SMTP_PASSWORD_ENV, N.NOTIFIED_MARKER_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FIDEON_LOG_DIR", str(tmp_path))
    N._RUN.update(version=None, command=None, stages=[], current=None)


@pytest.fixture
def telegram(monkeypatch):
    """Telegram switched on (it is off by default) and set up, with the call to Telegram recorded."""
    monkeypatch.setenv(N.TELEGRAM_TOKEN_ENV, TOKEN)
    monkeypatch.setenv(N.TELEGRAM_CHAT_ENV, CHAT)
    monkeypatch.setattr(N, "enabled_channels", lambda: ["telegram", "email"])
    sent = []

    def http(url, payload, timeout):
        sent.append((url, payload))
        return {"ok": True}

    monkeypatch.setattr(N, "_telegram_call", lambda method, payload, timeout, http_=None: http(
        f"https://api.telegram.org/bot{TOKEN}/{method}", payload, timeout))
    return sent


def stages(*names):
    numbers = {"ingestion": 1, "dataset_build": 4, "training": 5, "checkpoint_eval": 6, "calibrate": 7,
               "evaluation_gate": 8, "package": 9}
    return [SimpleNamespace(number=numbers[name], name=name) for name in names]


def result(name, status="completed", detail="", **data):
    return SimpleNamespace(name=name, status=status, detail=detail, data=data)


def report(*results, command="finetune", version="v3", ok=True):
    return SimpleNamespace(version=version, command=command, results=list(results), ok=ok,
                           blocked_at=None, failed_at=None)


def messages(sent):
    return [payload["text"] for _url, payload in sent]


def test_nothing_is_sent_and_nothing_fails_when_no_channel_is_set_up():
    assert N.text_channels() == [] and N.send_text("hello") == {}
    N.run_started(report(), SimpleNamespace(), stages("training"))     # no channel: a silent no-op


def test_a_run_says_it_started_and_names_the_first_phase(telegram):
    N.run_started(report(command="finetune"), SimpleNamespace(), stages("training", "checkpoint_eval"))
    assert messages(telegram) == ["Fine-tuning v3 started (finetune). Starting Phase 5 'Training'."]


def test_a_finished_phase_says_the_next_one_is_starting(telegram):
    N.run_started(report(), SimpleNamespace(), stages("training", "checkpoint_eval"))
    telegram.clear()
    N._phase_events(report(), result("training", wall_seconds=28052.0))
    assert messages(telegram) == [
        "Fine-tuning v3: Phase 5 'Training' completed in 7h 47m 32s. Now starting Phase 6 'Checkpoint selection'."]


def test_the_last_phase_of_a_command_hands_over_to_the_closing_message_not_a_second_one(telegram):
    N.run_started(report(), SimpleNamespace(), stages("training", "checkpoint_eval"))
    telegram.clear()
    N._phase_events(report(), result("checkpoint_eval", wall_seconds=60.0))
    assert telegram == []


def test_a_skipped_phase_sends_nothing(telegram):
    N.run_started(report(), SimpleNamespace(), stages("ingestion", "training"))
    telegram.clear()
    N._phase_events(report(), result("ingestion", status="skipped"))
    assert telegram == []


def test_a_failed_phase_is_told_with_its_reason_and_marked_as_told(telegram, monkeypatch):
    monkeypatch.setenv(N.NOTIFIED_MARKER_ENV, "cycle1")
    N.run_started(report(), SimpleNamespace(), stages("training"))
    telegram.clear()
    N._phase_events(report(), result("training", status="failed", detail="CUDA out of memory\nlong trace..."))
    assert messages(telegram) == ["Fine-tuning v3 STOPPED: Phase 5 'Training' failed. CUDA out of memory"]
    import os
    assert os.path.exists(N._notified_marker())          # pod_run.sh will not report this exit again


def test_a_blocked_gate_is_told_as_not_passed(telegram):
    N.run_started(report(), SimpleNamespace(), stages("evaluation_gate"))
    telegram.clear()
    N._phase_events(report(), result("evaluation_gate", status="blocked", detail="promotion gate blocked v3"))
    text, = messages(telegram)
    assert "Phase 8 'Evaluation gate' was blocked. Gate is not passed (blocked)." in text


def test_the_end_of_a_whole_cycle_says_all_phases_are_done_and_whether_the_gate_was_passed(telegram):
    ctx = SimpleNamespace(results={"evaluation_gate": result("evaluation_gate", gate_message="Gate is passed")},
                          release_id="r1", tenant_id=None, client=None)
    done = report(result("package"), command="package")
    assert N._closing_text(done, ctx) == "Fine-tuning v3 is completed, all phases done successfully. Gate is passed."

    ctx.results["evaluation_gate"] = result(
        "evaluation_gate", gate_message="Gate is not passed: 1 gate(s) failed (list_field_recall) and were waived "
                                        "by override, so the release can still be promoted")
    text = N._closing_text(done, ctx)
    assert text.startswith("Fine-tuning v3 is completed, all phases done successfully. Gate is not passed:")


def test_a_finetune_command_that_ends_at_phase_six_says_what_comes_next(telegram):
    N.run_started(report(), SimpleNamespace(), stages("training", "checkpoint_eval"))
    text = N._closing_text(report(result("training"), result("checkpoint_eval")), SimpleNamespace(results={}))
    assert "phases 5 to 6" in text and "package command" in text and "all phases done" not in text


def test_a_stall_is_told_with_the_phase_it_was_in(telegram):
    N.run_started(report(), SimpleNamespace(), stages("checkpoint_eval"))
    N.stage_started(SimpleNamespace(number=6, name="checkpoint_eval"))
    telegram.clear()
    N.stalled("a vLLM call of 96 request(s) has run for 1800 s without returning")
    text, = messages(telegram)
    assert text.startswith("Fine-tuning v3 STALLED in Phase 6 'Checkpoint selection':") and "--from-stage" in text


def test_the_stall_guard_tells_before_it_ends_the_process(monkeypatch):
    told = []
    monkeypatch.setattr(N, "stalled", told.append)
    monkeypatch.setattr(stall_guard.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
    with pytest.raises(SystemExit) as stopped:
        stall_guard.abort("a vLLM call has run for 1800 s")
    assert stopped.value.code == stall_guard.EXIT_CODE and told == ["a vLLM call has run for 1800 s"]


def test_a_failed_send_is_reported_without_the_token_and_never_raises(monkeypatch):
    monkeypatch.setenv(N.TELEGRAM_TOKEN_ENV, TOKEN)
    monkeypatch.setenv(N.TELEGRAM_CHAT_ENV, CHAT)
    monkeypatch.setattr(N, "enabled_channels", lambda: ["telegram"])

    def refuse(method, payload, timeout, http=None):
        raise OSError(f"HTTP Error 401 for https://api.telegram.org/bot{TOKEN}/sendMessage")

    monkeypatch.setattr(N, "_telegram_call", refuse)
    outcome = N.send_text("hello")
    assert set(outcome) == {"telegram"} and outcome["telegram"] != "sent"
    assert TOKEN not in outcome["telegram"] and "***" in outcome["telegram"]


def test_an_email_goes_through_the_smtp_server_set_up(monkeypatch):
    monkeypatch.setenv(N.EMAIL_TO_ENV, "me@example.com")
    monkeypatch.setenv(N.SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(N.SMTP_USER_ENV, "sender@example.com")
    monkeypatch.setenv(N.SMTP_PASSWORD_ENV, PASSWORD)
    seen = {}

    class Server:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            seen["login"] = (user, password)

        def send_message(self, message):
            seen["to"], seen["subject"], seen["body"] = message["To"], message["Subject"], message.get_content()

    outcome = N.send_text("Fine-tuning v3 started.\nmore", smtp_factory=lambda host, port, timeout: Server())
    assert outcome == {"email": "sent"} and seen["login"] == ("sender@example.com", PASSWORD)
    # The same subject on every e-mail; the status is in the body.
    assert seen["to"] == "me@example.com" and seen["subject"] == "Fideon Fine-Tuning update"
    assert seen["body"].startswith("Fine-tuning v3 started.\nmore")


def test_a_server_that_hangs_up_on_the_login_is_explained_not_called_a_network_fault(monkeypatch):
    """Gmail drops the connection after refusing an Outlook account's login; smtplib calls that
    "connection unexpectedly closed", which sent the first test looking at the network."""
    import smtplib

    monkeypatch.setenv(N.EMAIL_TO_ENV, "me@example.com")
    monkeypatch.setenv(N.SMTP_HOST_ENV, "smtp.gmail.com")
    monkeypatch.setenv(N.SMTP_USER_ENV, "me@example.com")
    monkeypatch.setenv(N.SMTP_PASSWORD_ENV, PASSWORD)

    class Server:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")

    outcome = N.send_text("hello", smtp_factory=lambda host, port, timeout: Server())
    assert "closed the connection while logging in" in outcome["email"] and "smtp.office365.com" in outcome["email"]
    assert PASSWORD not in outcome["email"]


def test_a_long_message_is_cut_so_it_cannot_carry_a_document(telegram):
    N.send_text("x" * 5000)
    assert len(messages(telegram)[0]) <= N.MAX_MESSAGE_CHARS


def test_the_test_command_says_what_is_missing_or_what_was_sent(telegram, capsys):
    assert N.main(["--test"]) == 0 and "telegram: sent" in capsys.readouterr().out


def test_the_test_command_with_no_channel_explains_the_setup(capsys):
    assert N.main(["--test"]) == 1
    said = capsys.readouterr().out
    assert N.EMAIL_TO_ENV in said and N.SMTP_HOST_ENV in said and "Telegram is off" in said


def test_the_settings_switch_the_chatty_messages_and_never_the_failure_ones():
    from orchestration.settings import notification_settings

    settings = notification_settings()
    assert settings["on_run_start"] and settings["on_phase_change"] and settings["on_stall"]
    assert notification_settings({"notifications": {"on_phase_change": False}})["on_phase_change"] is False


def test_telegram_is_off_by_default_even_when_it_is_set_up(monkeypatch):
    """The code is kept; it does nothing until notifications.channels lists telegram."""
    from orchestration.settings import notification_settings

    assert notification_settings()["channels"] == ["email"] and N.enabled_channels() == ["email"]
    monkeypatch.setenv(N.TELEGRAM_TOKEN_ENV, TOKEN)
    monkeypatch.setenv(N.TELEGRAM_CHAT_ENV, CHAT)
    called = []
    monkeypatch.setattr(N, "_telegram_call", lambda *a, **k: called.append(a))
    assert N.text_channels() == [] and N.send_text("hello") == {} and called == []


def test_email_alone_is_on_by_default(monkeypatch):
    monkeypatch.setenv(N.EMAIL_TO_ENV, "me@example.com")
    monkeypatch.setenv(N.SMTP_HOST_ENV, "smtp.example.com")
    assert N.text_channels() == ["email"]


def test_telegram_turns_on_when_the_config_lists_it_and_it_is_set_up(monkeypatch):
    from orchestration.settings import notification_settings

    on = notification_settings({"notifications": {"channels": ["email", "telegram"]}})["channels"]
    assert on == ["email", "telegram"]
    monkeypatch.setattr(N, "enabled_channels", lambda: on)
    monkeypatch.setenv(N.TELEGRAM_TOKEN_ENV, TOKEN)
    monkeypatch.setenv(N.TELEGRAM_CHAT_ENV, CHAT)
    assert N.text_channels() == ["telegram"]


def test_a_misspelt_channel_is_refused():
    from orchestration.settings import SettingsError, notification_settings

    with pytest.raises(SettingsError, match="whatsapp"):
        notification_settings({"notifications": {"channels": ["email", "whatsapp"]}})


def test_the_webhook_channel_posts_the_text_with_no_mailbox_password(monkeypatch):
    """A Power Automate flow sends the e-mail under the signed-in user's own login: the pipeline holds
    only the URL."""
    import json

    monkeypatch.setenv("FIDEON_NOTIFY_WEBHOOK_URL", "https://prod-1.example.logic.azure.com/workflows/abc?sig=SECRET")
    monkeypatch.setattr(N, "enabled_channels", lambda: ["webhook"])
    posted = []
    outcome = N.send_text("Fine-tuning v3 started.\nmore", http=lambda url, body, timeout: posted.append((url, body)))
    assert outcome == {"webhook": "sent"} and N.text_channels() == ["webhook"]
    (url, body), = posted
    assert "sig=SECRET" in url
    assert json.loads(body) == {"source": "fideon-finetuning", "subject": "Fideon Fine-Tuning update",
                                "text": "Fine-tuning v3 started.\nmore"}


def test_the_webhook_channel_is_off_by_default_refuses_plain_http_and_hides_its_url(monkeypatch):
    monkeypatch.setenv("FIDEON_NOTIFY_WEBHOOK_URL", "http://insecure.example/hook?sig=SECRET")
    assert N.text_channels() == []                                        # not listed in channels
    monkeypatch.setattr(N, "enabled_channels", lambda: ["webhook"])
    outcome = N.send_text("hello", http=lambda *a: None)
    assert outcome["webhook"] != "sent" and "SECRET" not in outcome["webhook"]


def test_with_the_webhook_channel_on_the_structured_event_post_is_not_sent_as_well(monkeypatch):
    monkeypatch.setenv("FIDEON_NOTIFY_WEBHOOK_URL", "https://prod-1.example.logic.azure.com/hook")
    posted = []
    settings = {"channels": ["webhook"], "on_run_complete": True, "webhook_url_env": "FIDEON_NOTIFY_WEBHOOK_URL",
                "timeout_seconds": 5}
    assert N.notify("run_complete", {"x": 1}, settings=settings, post=lambda *a: posted.append(a)) is False
    assert posted == []
