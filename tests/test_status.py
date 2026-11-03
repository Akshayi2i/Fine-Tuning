"""orchestration.status: what a run is doing, from its log and its run report."""

from __future__ import annotations

import json
import os
import time

from orchestration.status import find_progress, main, read_status, render, run_names


def _log(folder, name, lines, *, age=0.0, exit_code=None):
    path = folder / f"{name}.log"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stamp = time.time() - age
    os.utime(path, (stamp, stamp))
    if exit_code is not None:
        (folder / f"{name}.exit").write_text(str(exit_code), encoding="utf-8")
    return path


def test_a_log_written_a_moment_ago_is_running_with_its_last_line(tmp_path):
    _log(tmp_path, "finetune-v3", ["2026-11-03 10:00:00 INFO a starting", "2026-11-03 10:00:05 INFO a working"])
    status = read_status("finetune-v3", tmp_path)
    assert status.state == "RUNNING" and status.last_line.endswith("working")
    assert "last log line" in render(status)


def test_a_quiet_log_is_reported_as_possibly_stuck(tmp_path):
    _log(tmp_path, "finetune-v3", ["2026-11-03 10:00:00 INFO a working"], age=40 * 60)
    status = read_status("finetune-v3", tmp_path)
    assert status.state == "SILENT" and "may be stuck" in render(status)


def test_a_finished_run_reports_its_exit_code(tmp_path):
    _log(tmp_path, "package-v3", ["done"], age=3600, exit_code=75)
    status = read_status("package-v3", tmp_path)
    assert status.state == "FINISHED" and status.exit_code == 75 and "exit 75" in render(status)


def test_row_progress_gets_an_estimate_from_the_rate_between_timestamped_lines():
    lines = [
        "2026-11-03 10:00:00 INFO evaluation.validation_generation: validation generation: 96/400 row(s) in 330 s",
        "2026-11-03 10:11:00 INFO evaluation.validation_generation: validation generation: 192/400 row(s) in 330 s",
        "2026-11-03 10:22:00 INFO evaluation.validation_generation: validation generation: 288/400 row(s) in 330 s",
    ]
    progress = find_progress(lines)
    assert (progress.done, progress.total) == (288, 400)
    assert round(progress.eta_seconds) == round((400 - 288) / (192 / (22 * 60)))      # 96 rows per 11 min


def test_training_progress_uses_the_trainers_own_remaining_time():
    lines = ["{'loss': 0.4, 'global_step/max_steps': '120/857', 'remaining_time': '6:41:10'}"]
    progress = find_progress(lines)
    assert (progress.name, progress.done, progress.total) == ("training step", 120, 857)
    assert progress.eta_seconds == 6 * 3600 + 41 * 60 + 10 and "trainer" in progress.eta_source


def test_no_estimate_is_made_up_from_one_line_or_untimed_lines():
    one = ["2026-11-03 10:00:00 INFO validation generation: 96/400 row(s) in 330 s"]
    assert find_progress(one).eta_seconds is None
    assert find_progress(["nothing to see"]) is None


def test_the_run_report_stages_are_listed(tmp_path):
    _log(tmp_path, "finetune-v3", ["x"])
    (tmp_path / "run-finetune-default-v3.json").write_text(json.dumps({"stages": [
        {"stage": "training", "status": "completed", "detail": "857 steps"},
        {"stage": "checkpoint_eval", "status": "failed", "detail": "stalled"}]}), encoding="utf-8")
    text = render(read_status("finetune-v3", tmp_path))
    assert "[completed] training: 857 steps" in text and "[failed   ] checkpoint_eval" in text


def test_the_newest_run_is_the_default_and_list_names_them_all(tmp_path, capsys):
    _log(tmp_path, "old", ["a"], age=7200, exit_code=0)
    _log(tmp_path, "new", ["b"])
    assert run_names(tmp_path) == ["new", "old"]
    assert main(["--log-dir", str(tmp_path)]) == 0 and "run 'new'" in capsys.readouterr().out
    assert main(["--log-dir", str(tmp_path), "--list"]) == 0
    assert main(["--log-dir", str(tmp_path), "--name", "missing"]) == 1
    assert main(["--log-dir", str(tmp_path / "nowhere")]) == 1
