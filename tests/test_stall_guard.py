"""A generation call that never returns ends the process with its stacks written, and a run can resume."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from evaluation import validation_generation
from inference_core import stall_guard
from inference_core.runner_config import RunnerConfig, load_runner_config

ROOT = Path(__file__).resolve().parent.parent
POD_RUN = (ROOT / "scripts" / "pod_run.sh").read_text(encoding="utf-8")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, capture_output=True,
                          text=True, timeout=60, check=False)


def test_a_call_past_its_limit_ends_the_process_with_every_stack_written():
    done = _run("""
        import time
        from inference_core.stall_guard import stall_guard
        with stall_guard(1, "a test call"):
            time.sleep(30)
        print("not reached")
    """)
    assert done.returncode == stall_guard.EXIT_CODE
    assert "not reached" not in done.stdout
    assert stall_guard.MARKER in done.stderr and "a test call has run for 1 s" in done.stderr
    assert "most recent call first" in done.stderr   # faulthandler's stack of the stuck thread


def test_a_call_inside_its_limit_is_untouched_and_the_timers_are_cancelled():
    done = _run("""
        import time
        from inference_core.stall_guard import stall_guard
        with stall_guard(5, "a quick call"):
            pass
        time.sleep(6)           # past the limit: a timer left running would end the process here
        print("still here")
    """)
    assert done.returncode == 0 and "still here" in done.stdout


@pytest.mark.parametrize("seconds", [None, 0])
def test_no_limit_means_no_guard(seconds):
    with stall_guard.stall_guard(seconds, "anything"):
        pass


def test_the_serving_config_gives_every_engine_call_a_limit_and_a_bare_config_gives_none():
    assert load_runner_config().call_timeout_seconds == 1800.0
    assert RunnerConfig().call_timeout_seconds is None


def test_a_whole_call_that_failed_on_the_image_cache_stops_the_run(monkeypatch):
    stopped = []
    monkeypatch.setattr(stall_guard, "abort", stopped.append)
    cache = RuntimeError("generation failed for base: Expected a cached item for mm_hash='5cf3bf'")
    validation_generation._stop_if_image_cache_out_of_step([cache, cache])
    assert len(stopped) == 1 and "image cache" in stopped[0]


def test_other_whole_call_failures_still_go_through_the_retry(monkeypatch):
    stopped = []
    monkeypatch.setattr(stall_guard, "abort", stopped.append)
    validation_generation._stop_if_image_cache_out_of_step([RuntimeError("one unreadable page image")])
    assert stopped == []


def test_pod_run_starts_a_stalled_run_again_only_when_asked_and_only_for_a_resume():
    assert "POD_RUN_RETRY_STALLED" in POD_RUN
    assert 'MARKER' not in POD_RUN and "STALL-GUARD:" in POD_RUN   # the line the guard writes
    assert stall_guard.MARKER in POD_RUN
    assert '" --from-stage "' in POD_RUN   # refused for a command that would start a new cycle
