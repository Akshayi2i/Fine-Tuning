"""The frozen type-error counts CI holds (scripts/check_type_errors.py)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from scripts.check_type_errors import BASELINE, NotAMypyRun, errors_per_file, main, problems

RUN = """serving\\pipeline.py:10: error: Incompatible types  [assignment]
serving\\pipeline.py:20:5: error: Missing return  [return]
common/prompts.py:3: error: Name "x" is not defined  [name-defined]
common/prompts.py:4: note: See https://mypy.readthedocs.io
Found 3 errors in 2 files (checked 9 source files)
"""


def test_errors_are_counted_per_file_with_one_spelling_of_the_path() -> None:
    assert errors_per_file(RUN) == Counter({"serving/pipeline.py": 2, "common/prompts.py": 1})
    assert errors_per_file("Success: no issues found in 9 source files\n") == Counter()


@pytest.mark.parametrize("output", ["", "mypy: error: unrecognized arguments\n",
                                    RUN.replace("Found 3 errors", "Found 4 errors")])
def test_output_that_is_not_a_finished_run_is_refused(output: str) -> None:
    with pytest.raises(NotAMypyRun):
        errors_per_file(output)


def test_a_new_error_fails_and_a_fixed_one_is_named() -> None:
    """Seen to fail on purpose: one error more than the file was allowed."""
    counts = errors_per_file(RUN)
    failures, lower = problems(counts, {"serving/pipeline.py": 1, "common/prompts.py": 2})
    assert failures == ["serving/pipeline.py: 2 type error(s), 1 allowed - fix the new one(s)"]
    assert lower == ["common/prompts.py: 1 type error(s), 2 allowed - run with --update to lower it"]
    failures, _ = problems(Counter({"new_module.py": 1}), {})
    assert failures == ["new_module.py: 1 type error(s), 0 allowed - fix the new one(s)"]


def test_update_lowers_the_baseline_but_never_raises_it(tmp_path: Path) -> None:
    output, baseline = tmp_path / "mypy.txt", tmp_path / "baseline.json"
    output.write_text(RUN, encoding="utf-8")
    baseline.write_text(json.dumps({"serving/pipeline.py": 3, "common/prompts.py": 1}), encoding="utf-8")
    assert main([str(output), "--baseline", str(baseline), "--update"]) == 0
    assert json.loads(baseline.read_text(encoding="utf-8")) == {"common/prompts.py": 1, "serving/pipeline.py": 2}
    baseline.write_text(json.dumps({"serving/pipeline.py": 1, "common/prompts.py": 1}), encoding="utf-8")
    assert main([str(output), "--baseline", str(baseline), "--update"]) == 1
    assert json.loads(baseline.read_text(encoding="utf-8"))["serving/pipeline.py"] == 1


def test_the_baseline_names_files_that_exist() -> None:
    root = BASELINE.parent.parent
    counts = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert counts and all((root / file).is_file() for file in counts)
    assert all(isinstance(n, int) and n > 0 for n in counts.values())
