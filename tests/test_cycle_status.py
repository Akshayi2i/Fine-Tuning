"""scripts/cycle_status.sh: a cycle's status in plain words, read from made-up logs of each situation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _git_bash() -> str | None:
    found = shutil.which("bash")
    return None if not found or "system32" in found.lower() else found


pytestmark = pytest.mark.skipif(_git_bash() is None, reason="needs bash")


def _status(tmp_path: Path, logs: dict[str, str], exits: dict[str, str], running: set[str]) -> str:
    (tmp_path / "repo" / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "bin").mkdir(exist_ok=True)
    shutil.copy(ROOT / "scripts" / "cycle_status.sh", tmp_path / "repo" / "scripts")
    for name, text in logs.items():
        (tmp_path / "logs" / f"{name}.log").write_text(text, encoding="utf-8")
    for name, code in exits.items():
        (tmp_path / "logs" / f"{name}.exit").write_text(code + "\n", encoding="utf-8")
    sessions = " ".join(f"=fideon-{r}" for r in running)
    (tmp_path / "bin" / "tmux").write_bytes(
        f'#!/usr/bin/env bash\nfor s in {sessions}; do [ "$3" = "$s" ] && exit 0; done; exit 1\n'.encode())
    (tmp_path / "drive.sh").write_bytes(b"""#!/usr/bin/env bash
S="$(pwd)"; chmod +x "$S/bin/tmux"; export PATH="$S/bin:$PATH" FIDEON_LOG_DIR="$S/logs"
bash "$S/repo/scripts/cycle_status.sh"
""")
    done = subprocess.run([_git_bash(), "drive.sh"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout


IMPORTING = "\n".join(f"2026-11-14 10:5{i} INFO x ingested document.pdf as policy_0{100 + i} (9 pages, digital)"
                      for i in range(5))
READING = "\n".join(f"2026-11-14 11:0{i} INFO x processed policy_0{100 + i}: 9 pages on cuda at 1792px"
                    for i in range(3))
WAITING = "2026-11-14 11:38:32 UTC full_cycle: waiting for 'importc2' to finish\n"


def test_now_importing_and_reading_and_the_cycle_waiting(tmp_path):
    out = _status(tmp_path, {"importc2": IMPORTING, "ocrloop": READING, "cycle2": WAITING}, {},
                  {"importc2", "ocrloop", "cycle2"})
    assert "NOW:      Importing documents (5 of 836) and reading their pages (3 read)" in out
    assert "[running] 1. Import documents            (5 of 836)" in out
    assert "[running] 2. Read pages (OCR)            (3 documents read)" in out
    assert "HEALTH:   OK - no errors found" in out


def test_a_cycle_waiting_quietly_is_healthy_but_a_quiet_import_is_not(tmp_path):
    """The waiting cycle writes nothing for hours (2026-11-14: a "may be stuck" mail for exactly that)."""
    import os
    import time

    _status(tmp_path, {"importc2": IMPORTING, "ocrloop": READING, "cycle2": WAITING}, {}, set())
    old = time.time() - 3600
    for name in ("cycle2", "importc2"):
        os.utime(tmp_path / "logs" / f"{name}.log", (old, old))
    out = _status(tmp_path, {}, {}, {"importc2", "ocrloop", "cycle2"})
    assert "'cycle2' has written nothing" not in out
    assert "'importc2' has written nothing for 60 minutes" in out


def test_mid_training_shows_the_step_and_time_left(tmp_path):
    cycle = WAITING + (
        "2026-11-14 14:00:00 UTC full_cycle: this cycle is v4: finetune --skip-ingest --fresh\n"
        "2026-11-14 14:00:05 INFO orchestration.pipeline_dag stage 2 preprocessing: already complete, skipping\n"
        "2026-11-14 14:00:06 INFO orchestration.pipeline_dag stage 4 dataset_build [GPU]\n"
        "2026-11-14 14:20:06 INFO orchestration.pipeline_dag stage 4 dataset_build: completed in 20m 00s\n"
        "2026-11-14 14:20:07 INFO orchestration.pipeline_dag stage 5 training [GPU]\n"
        "{'loss': 0.2, 'global_step/max_steps': '2371/9486', 'percentage': '25.00%', 'remaining_time': '20h 51m 3s'}\n"
    )
    out = _status(tmp_path, {"importc2": IMPORTING, "ocrloop": READING, "cycle2": cycle},
                  {"importc2": "0", "ocrloop": "0"}, {"cycle2"})
    assert "NOW:      Training the model" in out and "VERSION:  v4" in out
    assert "TRAINING: step 2371 of 9486 (24%), about 20h 51m 3s left" in out
    assert "- Build training set: done (20m 00s)" in out and "- Train the model: running" in out
    assert "[done]    1. Import documents" in out
    assert "HEALTH:   OK" in out


def test_a_failure_says_needs_attention_and_shows_the_error(tmp_path):
    cycle = WAITING + (
        "2026-11-14 14:20:07 INFO orchestration.pipeline_dag stage 5 training [GPU]\n"
        "2026-11-14 15:00:00 ERROR orchestration.pipeline_dag stage training failed after 1 attempt(s): CUDA out of"
        " memory\n"
        "2026-11-14 15:00:01 UTC full_cycle: finetune stopped. Resume it: ...\n"
    )
    out = _status(tmp_path, {"importc2": IMPORTING, "ocrloop": READING, "cycle2": cycle},
                  {"importc2": "0", "ocrloop": "0", "cycle2": "1"}, set())
    assert "HEALTH:   NEEDS ATTENTION" in out
    assert "'cycle2' ended with an error (exit 1)" in out
    assert "CUDA out of memory" in out and "finetune stopped" in out
    assert "[FAILED]  3. Training run, then gate" in out


def test_a_run_whose_session_vanished_is_flagged(tmp_path):
    out = _status(tmp_path, {"importc2": IMPORTING, "ocrloop": READING, "cycle2": WAITING}, {},
                  {"importc2", "ocrloop"})                  # cycle2's session is gone, no exit recorded
    assert "[STOPPED] 3. Training run" in out and "is not running and recorded no exit" in out


def test_finished(tmp_path):
    cycle = WAITING + (
        "2026-11-14 14:00:00 UTC full_cycle: this cycle is v4: finetune --skip-ingest --fresh\n"
        "2026-11-16 02:00:00 UTC full_cycle: package v4\n"
        "2026-11-16 04:00:00 INFO orchestration.pipeline_dag stage 9 package: completed in 1m 00s\n"
        "2026-11-16 04:00:01 UTC full_cycle: done: v4 is packaged and gated, not promoted\n"
    )
    out = _status(tmp_path, {"importc2": IMPORTING, "ocrloop": READING, "cycle2": cycle},
                  {"importc2": "0", "ocrloop": "0", "cycle2": "0"}, set())
    assert "NOW:      FINISHED - v4 is packaged and gated (not promoted)" in out
    assert "HEALTH:   OK" in out
