"""The per-module coverage floors CI holds (review R8)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.check_module_coverage import FLOORS, main, module_coverage, problems


def _report(percent: float, *, leave_out: str | None = None) -> dict:
    return {"files": {f"{module.replace('.', '/')}.py": {"summary": {"percent_covered": percent}}
                      for module in FLOORS if module != leave_out}}


def test_every_module_at_or_above_its_floor_passes() -> None:
    assert problems(_report(100.0)) == []


def test_a_module_below_its_floor_fails() -> None:
    found = problems(_report(10.0))
    assert any(problem.startswith("training.data_collator: 10.0% covered") for problem in found)


def test_a_module_missing_from_the_report_fails_rather_than_passes() -> None:
    found = problems(_report(100.0, leave_out="serving.pipeline"))
    assert found == ["serving.pipeline: not in the coverage report (is it passed to --cov?)"]


def test_a_report_with_absolute_paths_is_matched_and_main_passes(tmp_path: Path) -> None:
    report = {"files": {f"/home/runner/work/repo/{module.replace('.', '/')}.py": {"summary": {"percent_covered": 99.0}}
                        for module in FLOORS}}
    assert set(module_coverage(report).values()) == {99.0}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(_report(100.0)), encoding="utf-8")
    assert main([str(path)]) == 0
