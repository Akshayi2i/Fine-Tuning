"""The pod launcher keeps runs alive across disconnects and never ends one itself."""

from __future__ import annotations

import re
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent / "scripts" / "pod_run.sh").read_text(
    encoding="utf-8"
)


def test_runs_start_detached_in_tmux():
    assert re.search(r"tmux new-session -d -s", SCRIPT)


def test_nothing_stops_the_pod_or_kills_a_run_unasked():
    assert "runpodctl" not in SCRIPT
    kills = [line for line in SCRIPT.splitlines() if "kill-session" in line]
    assert len(kills) == 1   # only inside `stop`, after a typed confirmation
    assert "Type the run name to confirm" in SCRIPT


def test_the_log_and_exit_code_live_on_the_volume():
    assert 'LOG_DIR="${FIDEON_LOG_DIR:-/workspace/logs}"' in SCRIPT
    assert "PIPESTATUS[0]" in SCRIPT   # the pipeline's exit, not tee's


def test_the_script_has_unix_line_endings():
    raw = (Path(__file__).resolve().parent.parent / "scripts" / "pod_run.sh").read_bytes()
    assert b"\r\n" not in raw


def test_a_job_started_without_a_virtualenv_runs_in_the_training_venv():
    """The first smoke run died on `No module named 'azure'`: the system python has none."""
    block = SCRIPT[SCRIPT.index('if [ -z "${VIRTUAL_ENV:-}" ]'):]
    block = block[:block.index("  fi")]
    assert "/venv/bin/python" in block and "export PATH=" in block and "export VIRTUAL_ENV=" in block


def test_a_silent_log_and_an_unreported_exit_are_told_by_the_launcher():
    """A hard-killed process cannot say it died or hung: the launcher watches the log and, when the run
    exits badly without its own closing message, sends one (orchestration.notify, a no-op without a channel)."""
    assert "POD_RUN_STALE_MINUTES" in SCRIPT and "may be stuck" in SCRIPT
    assert "orchestration.notify --text" in SCRIPT
    assert "FIDEON_RUN_NAME" in SCRIPT and ".notified" in SCRIPT   # the pipeline's own message is not repeated


def test_a_run_stopped_by_a_signal_says_so_in_its_log_and_exit():
    """A run vanished on 2026-11-07 with an empty log and no exit; the cause stayed unknown. A hang-up, Ctrl-C or
    kill now writes a line to the log and 'signal X' as the exit, and status reports it as stopped."""
    assert "on_signal() {" in SCRIPT and 'echo "signal $1" >' in SCRIPT
    for signal, number in (("HUP", 1), ("TERM", 15)):
        assert f"trap 'on_signal {signal} {number}' {signal}" in SCRIPT
    # Keys typed in an attached session never stop a run (user, 2026-11-09): ignored, and the run in its own session.
    assert "trap '' INT QUIT TSTP" in SCRIPT and 'own_session="setsid -w"' in SCRIPT
    assert '$own_session bash -c "$job_command" </dev/null &' in SCRIPT
    # Set before the command runs, so a signal at any point of it is caught.
    assert SCRIPT.index("trap 'on_signal HUP 1' HUP") < SCRIPT.index("echo 'attempt=1'")
    assert "STOPPED by" in SCRIPT and "grep -q '^signal'" in SCRIPT
