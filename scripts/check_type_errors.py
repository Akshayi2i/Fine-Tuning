"""Hold the type errors to what they were: no file may gain one (CI).

mypy found 271 errors in 62 files on 24 November 2026, all older than the check
that would have stopped them. Fixing all of them at once was not chosen; this
freezes them instead. Each file may have at most the errors
``scripts/type_error_baseline.json`` records for it, and a file it does not list
none. A file that has fewer than its count passes and is named, so its count can
be lowered (``--update``): the numbers only go down.

Reads mypy's output, so CI runs mypy once:

    mypy <packages> > mypy.txt || true
    python -m scripts.check_type_errors mypy.txt            # the check
    python -m scripts.check_type_errors mypy.txt --update   # after fixing errors

Output that is not a finished mypy run (a crash, a config error: no "Found N
errors" or "Success" line, or a count that does not add up) fails: never a pass
on a check that did not run.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "type_error_baseline.json"

_ERROR = re.compile(r"^(?P<file>[^:\n]+\.pyi?):\d+(?::\d+)?: error:", re.M)
_FOUND = re.compile(r"^Found (\d+) errors? in \d+ files?", re.M)
_SUCCESS = re.compile(r"^Success: no issues found", re.M)


class NotAMypyRun(ValueError):
    """The text is not the output of a finished mypy run."""


def errors_per_file(output: str) -> Counter[str]:
    counts = Counter(Path(match["file"]).as_posix() for match in _ERROR.finditer(output))
    found = _FOUND.search(output)
    if found is None and not _SUCCESS.search(output):
        raise NotAMypyRun("no 'Found N errors' or 'Success' line: mypy did not finish")
    total = int(found.group(1)) if found else 0
    if sum(counts.values()) != total:
        raise NotAMypyRun(f"mypy reports {total} errors but {sum(counts.values())} error lines were read")
    return counts


def problems(counts: Counter[str], baseline: dict[str, int]) -> tuple[list[str], list[str]]:
    """(failures, files that can lower their count)."""
    failures, lower = [], []
    for file in sorted(set(counts) | set(baseline)):
        now, allowed = counts.get(file, 0), baseline.get(file, 0)
        if now > allowed:
            failures.append(f"{file}: {now} type error(s), {allowed} allowed - fix the new one(s)")
        elif now < allowed:
            lower.append(f"{file}: {now} type error(s), {allowed} allowed - run with --update to lower it")
    return failures, lower


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output", type=Path, help="mypy's output, saved to a file")
    parser.add_argument("--update", action="store_true",
                        help="write the counts to the baseline (only lowers: a rise still fails)")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    try:
        counts = errors_per_file(args.output.read_text(encoding="utf-8", errors="replace"))
    except NotAMypyRun as exc:
        print(f"::error::{exc}")
        return 1
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline.exists() else {}
    failures, lower = problems(counts, baseline)
    for line in failures:
        print(f"::error::{line}")
    for line in lower:
        print(line)
    if args.update and not failures:
        args.baseline.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {sum(counts.values())} errors in {len(counts)} files")
    print(f"{sum(counts.values())} type errors in {len(counts)} files; "
          f"{sum(baseline.values())} allowed in {len(baseline)} files")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
