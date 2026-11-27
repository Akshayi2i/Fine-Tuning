"""scripts/full_cycle.sh: a whole cycle on the pod in one command, run here with stand-ins for Python and MinerU.

The order is the point: the runs it waits for, then the GPUs freed, then finetune, the freeze, package - and
nothing after a step that failed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _git_bash() -> str | None:
    found = shutil.which("bash")
    # Windows' System32 bash is WSL, a different machine's paths; the pod and Git Bash are what this runs on.
    return None if not found or "system32" in found.lower() else found


pytestmark = pytest.mark.skipif(_git_bash() is None, reason="needs bash")

#: Python, as the script calls it: answers the two questions it asks with -c, and records every command.
FAKE_PYTHON = b"""#!/usr/bin/env bash
echo "python $*" | tr '\\n' ' ' >> "$S/calls"; echo >> "$S/calls"
if [ "$1" = "-c" ]; then
  case "$2" in
    *next_cycle_version*) echo "INFO some library line"; echo "v4" ;;
    *is_frozen*) echo "${FROZEN:-False}" ;;
  esac
  exit 0
fi
case "$3" in
  finetune) case "$*" in
      *--from-stage*) exit "${TRAIN_EXIT:-0}" ;;   # the training half of the split
      *) exit "${FINETUNE_EXIT:-0}" ;;             # the corpus-build half (--to-stage)
    esac ;;
  freeze-eval-set) case "$*" in *--mark-reviewed*) exit "${REVIEWED_EXIT:-0}" ;; *) exit 0 ;; esac ;;
  package) exit "${PACKAGE_EXIT:-0}" ;;
esac
"""
FAKE_SERVER = b"""#!/usr/bin/env bash
echo "mineru_server $*" >> "$S/calls"
"""


def _run(tmp_path: Path, args: str, env: str = "", exits: dict[str, str] | None = None,
         before: str = "") -> tuple[int, list[str], str]:
    (tmp_path / "repo" / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    shutil.copy(ROOT / "scripts" / "full_cycle.sh", tmp_path / "repo" / "scripts")
    (tmp_path / "repo" / "scripts" / "mineru_server.sh").write_bytes(FAKE_SERVER)
    (tmp_path / "python").write_bytes(FAKE_PYTHON)
    for name, code in (exits or {}).items():
        (tmp_path / "logs" / f"{name}.exit").write_text(code + "\n", encoding="utf-8")
    (tmp_path / "drive.sh").write_bytes(f"""#!/usr/bin/env bash
export S="$(pwd)"; chmod +x "$S/python"
export FIDEON_PYTHON="$S/python" FIDEON_LOG_DIR="$S/logs" FULL_CYCLE_POLL_S=0 {env}
{before}
bash "$S/repo/scripts/full_cycle.sh" {args}
""".encode())
    done = subprocess.run([_git_bash(), "drive.sh"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    calls_file = tmp_path / "calls"
    calls = calls_file.read_text(encoding="utf-8").splitlines() if calls_file.exists() else []
    return done.returncode, [c.strip() for c in calls if c.strip()], done.stdout + done.stderr


def _steps(calls: list[str]) -> list[str]:
    named = []
    for call in calls:
        if call.startswith("mineru_server"):
            named.append("stop servers")
        elif "next_cycle_version" in call:
            named.append("version")
        elif "is_frozen" in call:
            named.append("frozen?")
        elif "--mark-reviewed" in call:
            named.append("mark reviewed")
        elif any(step in call for step in ("run finetune", "freeze-eval-set --corpus", "run package")):
            named.append(call.split("orchestration.run ", 1)[1])
    return named


def test_the_whole_cycle_in_order(tmp_path):
    code, calls, out = _run(tmp_path, "--after importc2 --after ocrloop -- --skip-ingest --fresh",
                            exits={"importc2": "0", "ocrloop": "0"})
    assert code == 0, out
    steps = _steps(calls)
    # The freeze runs in the background during training, so filter it out: the main thread's order is
    # build the corpus (--to-stage), check frozen, train (--from-stage), package.
    main = [s for s in steps if not (s.startswith("freeze-eval-set --corpus") or s == "mark reviewed")]
    assert main == [
        "stop servers", "version", "finetune --version v4 --to-stage dataset_build --skip-ingest --fresh",
        "frozen?", "finetune --version v4 --from-stage training --skip-ingest --fresh",
        "package --version v4 --keep-staging",
    ]
    # the eval set is frozen (in parallel) and its review records copied, after the corpus is built
    assert "freeze-eval-set --corpus v4" in steps and "mark reviewed" in steps
    assert steps.index("frozen?") < steps.index("freeze-eval-set --corpus v4")
    assert "this cycle is v4" in out and "done: v4 is packaged" in out


def test_it_waits_for_a_run_still_going_and_starts_nothing_until_it_ends(tmp_path):
    """Started while the reading loop still runs: nothing - not even the servers' stop - until its marker lands."""
    code, calls, out = _run(tmp_path, "--after ocrloop -- --fresh", env="FULL_CYCLE_POLL_S=1",
                            before='(sleep 3; [ -s "$S/calls" ] && echo early > "$S/started_early"; '
                                   'echo 0 > "$S/logs/ocrloop.exit") &')
    assert code == 0, out
    assert not (tmp_path / "started_early").exists()
    assert _steps(calls)[0] == "stop servers" and _steps(calls)[-1] == "package --version v4 --keep-staging"


def test_a_run_waited_for_that_failed_stops_everything(tmp_path):
    code, calls, out = _run(tmp_path, "--after importc2 -- --fresh", exits={"importc2": "1"})
    assert code == 1 and calls == []
    assert "'importc2' ended with '1', not 0: stopping before any training" in out


def test_a_corpus_build_that_failed_freezes_nothing_and_packages_nothing(tmp_path):
    code, calls, out = _run(tmp_path, "-- --fresh", env="FINETUNE_EXIT=1")
    assert code == 1
    assert _steps(calls) == ["stop servers", "version",
                             "finetune --version v4 --to-stage dataset_build --fresh"]
    assert "--version v4 --from-stage" in out                     # how to resume it


def test_a_training_that_failed_packages_nothing(tmp_path):
    """The corpus built and the freeze may run beside it, but a failed training stops before the gate."""
    code, calls, out = _run(tmp_path, "-- --fresh", env="TRAIN_EXIT=1")
    assert code == 1
    steps = _steps(calls)
    assert "finetune --version v4 --from-stage training --fresh" in steps
    assert not any(s.startswith("package") for s in steps)
    assert "--version v4 --from-stage" in out                     # how to resume it


def test_a_frozen_set_is_scored_as_it_is_never_frozen_again(tmp_path):
    code, calls, _out = _run(tmp_path, "-- --fresh", env="FROZEN=True")
    assert code == 0
    assert not any(step.startswith("freeze-eval-set --corpus") for step in _steps(calls))
    assert "mark reviewed" in _steps(calls)


def test_a_gold_without_a_review_record_stops_before_package(tmp_path):
    code, calls, out = _run(tmp_path, "-- --fresh", env="REVIEWED_EXIT=1")
    assert code == 1 and not any(step.startswith("package") for step in _steps(calls))
    assert "stopping before package" in out


def test_a_tenant_goes_to_every_step(tmp_path):
    code, calls, _out = _run(tmp_path, "--tenant acme -- --fresh")
    assert code == 0
    run_calls = [c for c in calls if "orchestration.run" in c]
    assert run_calls and all("--tenant acme" in c for c in run_calls)


def test_an_unknown_option_is_refused_before_anything_runs(tmp_path):
    code, calls, out = _run(tmp_path, "--fresh")
    assert code == 2 and calls == [] and "finetune's options go after --" in out
