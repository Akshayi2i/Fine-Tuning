"""On the pod, every long job runs detached in tmux; nothing needs remembering.

A job in an SSH terminal dies when the laptop shuts or the Wi-Fi drops. Each
long entry point re-launches itself through scripts/pod_run.sh instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from orchestration import detach

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def launched(monkeypatch):
    calls = []
    monkeypatch.setattr(detach.subprocess, "run", lambda cmd, check: calls.append(cmd))
    for var in ("RUNPOD_POD_ID", "TMUX", detach.DETACHED_ENV, detach.NO_DETACH_ENV):
        monkeypatch.delenv(var, raising=False)
    # This machine is not the pod whatever it happens to have: pod signals come
    # only from what each test sets.
    monkeypatch.setattr(detach, "RUNPOD_ENV_FILE", Path("/nonexistent/rp_environment"))
    monkeypatch.setattr(detach, "_is_linux", lambda: False)
    return calls


def test_off_the_pod_nothing_detaches(launched):
    assert not detach.detach_module_if_needed("training.train", ["--corpus", "v1"])
    assert launched == []


def test_on_the_pod_a_job_relaunches_itself_in_tmux(launched, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    assert detach.detach_module_if_needed("training.train", ["--corpus", "v1"])
    [cmd] = launched
    assert cmd[:3] == ["bash", str(detach.POD_RUN), "start"]
    assert cmd[3].startswith("train-")
    assert cmd[4] == "--" and cmd[-3:] == ["training.train", "--corpus", "v1"]


@pytest.mark.parametrize("var,value", [
    (detach.DETACHED_ENV, "1"),   # already inside a pod_run.sh run: no second copy
    ("TMUX", "/tmp/tmux-0/default,1,0"),   # the operator's own tmux is already safe
    (detach.NO_DETACH_ENV, "1"),   # explicit opt-out
])
def test_a_job_that_is_already_safe_runs_in_place(launched, monkeypatch, var, value):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    monkeypatch.setenv(var, value)
    assert not detach.detach_module_if_needed("training.train", [])
    assert launched == []


def test_the_pipeline_cli_detaches_on_the_pod_and_runs_nothing_here(launched, monkeypatch):
    from orchestration import run as cli

    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    monkeypatch.setattr(cli, "run_scopes", lambda *a, **k: pytest.fail("ran in the foreground"))
    assert cli.main(["finetune", "--out-version", "v1", "--dry-run"]) == 0
    [cmd] = launched
    assert cmd[-4:] == ["finetune", "--out-version", "v1", "--dry-run"]


def test_switching_the_endpoint_stays_in_the_foreground(launched, monkeypatch):
    from orchestration import run as cli

    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    switched = []
    monkeypatch.setattr(cli, "RunPodController", lambda: type(
        "C", (), {"deploy_endpoint": lambda self, m, dry_run: switched.append(m)})())
    assert cli.main(["deploy-endpoint", "--model", "v1", "--dry-run"]) == 0
    assert switched == ["v1"] and launched == []


#: Entry points that finish in seconds, or only print a pointer and exit.
QUICK = {
    "registry_utils/query_registry.py", "scripts/combine_specs.py",
    "scripts/derive_aliases_from_canonical.py", "testing/render_prompts.py",
    "pilot/pilot_report.py", "pilot/zero_shot_baseline.py",
    "data_pipeline/ocr/mineru_config.py", "scripts/propose_field_types.py",
    "scripts/diagnose_windowing.py", "scripts/check_alias_coverage.py",
    "scripts/check_section_builder.py",
}


def _entry_points() -> list[Path]:
    found = []
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts or any(p.startswith(".") for p in path.parts):
            continue
        if '__name__ == "__main__"' in path.read_text(encoding="utf-8"):
            found.append(path)
    return found


def test_every_entry_point_is_either_detached_or_known_to_be_quick():
    """A new long job added without the guard would run attached and die with the
    laptop. This fails until it is given the guard or listed as quick."""
    unguarded = []
    for path in _entry_points():
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        if rel in QUICK:
            continue
        if not any(name in text for name in ("detach_module_if_needed", "detach_script_if_needed")):
            unguarded.append(rel)
    assert not unguarded, f"long-running entry points without the tmux guard: {unguarded}"


def test_the_guard_comes_after_argument_parsing():
    """So --help and bad arguments answer at once instead of starting a session."""
    for path in _entry_points():
        text = path.read_text(encoding="utf-8")
        if "detach_" not in text or path.name == "detach.py":
            continue
        tree = ast.parse(text)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        source = ast.get_source_segment(text, main)
        assert source.index("parse_args") < source.index("detach_"), path.name


def test_setup_detaches_on_the_pod():
    text = (ROOT / "scripts" / "setup_pod.sh").read_text(encoding="utf-8")
    assert "RUNPOD_POD_ID" in text and "pod_run.sh start" in text
