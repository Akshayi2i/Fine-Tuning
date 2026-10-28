"""A job keeps the Python environment it was started from, wherever it runs.

A tmux session takes its environment from the tmux server — started by whichever
job came first — not from the shell that starts a run. Without care, `finetune`
started from the training venv could not find that venv's `swift`, and the
bootstrap's spike could not either.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
POD_RUN = (ROOT / "scripts" / "pod_run.sh").read_text(encoding="utf-8")
BOOTSTRAP = (ROOT / "scripts" / "pod_bootstrap.sh").read_text(encoding="utf-8")


def test_each_run_records_the_callers_path_and_venv():
    assert "printf 'export PATH=%q\\n' \"$PATH\"" in POD_RUN
    assert "export VIRTUAL_ENV=%q" in POD_RUN
    # recorded before the command line is written
    assert POD_RUN.index("export PATH=%q") < POD_RUN.index("printf 'python -m orchestration.run'")


def test_swift_is_found_next_to_this_python_even_off_path(tmp_path, monkeypatch):
    from training import train as T

    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    swift = bin_dir / ("swift.exe" if os.name == "nt" else "swift")
    swift.write_text("#!/bin/sh\n", encoding="utf-8")
    swift.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, check, env: calls.append((argv, env)))

    T.launch(T.SwiftConfig(args={"model": "m"}))
    [(argv, env)] = calls
    assert Path(argv[0]) == swift and argv[1] == "sft"
    assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)   # swift's own workers resolve here too


def test_no_swift_anywhere_is_a_clear_error(tmp_path, monkeypatch):
    from training import train as T

    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(T.TrainingError, match="swift"):
        T.launch(T.SwiftConfig(args={}))


def test_the_bootstraps_spike_runs_with_its_venv_on_path():
    assert 'PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV"' in BOOTSTRAP


def test_the_bootstrap_builds_venvs_with_python_311_or_newer():
    assert "sys.version_info < (3, 11)" in BOOTSTRAP
    assert '"$PYTHON" -m venv "$venv"' in BOOTSTRAP
    assert "python3 -m venv" not in BOOTSTRAP


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_recorded_path_survives_quoting(tmp_path):
    """A PATH with spaces (Windows-style, or odd venv names) must come back intact."""
    odd = f"{tmp_path}/with space/bin:/usr/bin"
    script = f"printf 'export PATH=%q\\n' \"$P\" > {tmp_path.as_posix()}/r.sh; . {tmp_path.as_posix()}/r.sh; printf %s \"$PATH\""
    out = subprocess.run(["bash", "-c", script], env={**os.environ, "P": odd},
                         capture_output=True, text=True, check=True).stdout
    assert out == odd
