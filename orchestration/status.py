"""What a run is doing now, from its log and its run report: no GPU, no Azure, nothing is touched.

    python -m orchestration.status                   # the newest run on this pod
    python -m orchestration.status --name finetune-v3
    python -m orchestration.status --list

It reads ``$FIDEON_LOG_DIR`` (default /workspace/logs): the ``<name>.log`` and ``<name>.exit`` files that
``scripts/pod_run.sh`` writes and the ``run-*.json`` report the pipeline rewrites after every stage. It says
whether the run finished, how long the log has been quiet, the last progress it printed (training steps, rows
generated) and, when the log carries timestamps, how long that progress still needs at the rate so far.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

#: A log quiet for longer than this, with the run not finished, is reported as possibly stuck.
SILENT_MINUTES = 15
TAIL_BYTES = 200_000

STAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\b")
#: Progress the pipeline prints: (name, pattern with done and total).
PROGRESS = (
    ("training step", re.compile(r"global_step/max_steps['\"]?\s*:\s*['\"]?(\d+)/(\d+)")),
    ("gate readings", re.compile(r"golden eval: (\d+)/(\d+) reading")),
    ("validation rows", re.compile(r"validation generation: (\d+)/(\d+) row")),
    ("training step", re.compile(r"\bstep (\d+)/(\d+)\b")),
)
REMAINING = re.compile(r"remaining_time['\"]?\s*:\s*['\"]?([0-9:]+)")


@dataclass
class Progress:
    name: str
    done: int
    total: int
    eta_seconds: float | None = None
    eta_source: str = ""


@dataclass
class Status:
    name: str
    state: str                        # RUNNING / SILENT / FINISHED / STOPPED
    exit_code: int | None = None
    quiet_seconds: float = 0.0
    last_line: str = ""
    progress: Progress | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    report: str = ""


def log_dir() -> Path:
    return Path(os.environ.get("FIDEON_LOG_DIR") or "/workspace/logs")


def _tail_lines(path: Path) -> list[str]:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - TAIL_BYTES))
        data = handle.read()
    return data.decode("utf-8", errors="replace").splitlines()


def _stamp(line: str) -> datetime | None:
    match = STAMP.match(line)
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S") if match else None


def _clock_seconds(text: str) -> float | None:
    parts = text.split(":")
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        return None
    seconds = 0
    for value in numbers:
        seconds = seconds * 60 + value
    return float(seconds)


def find_progress(lines: list[str]) -> Progress | None:
    """The last progress the log printed, with an estimate of what is left.

    The estimate is the program's own ``remaining_time`` when the line has one, otherwise the rate between the
    first and last timestamped progress lines of the same kind; with neither, none: a number is never made up.
    """
    for name, pattern in PROGRESS:
        hits = [(index, pattern.search(line)) for index, line in enumerate(lines)]
        hits = [(index, m) for index, m in hits if m]
        if not hits:
            continue
        last_index, last = hits[-1]
        done, total = int(last.group(1)), int(last.group(2))
        progress = Progress(name, done, total)
        remaining = REMAINING.search(lines[last_index])
        if remaining and (seconds := _clock_seconds(remaining.group(1))) is not None:
            progress.eta_seconds, progress.eta_source = seconds, "the trainer's own estimate"
            return progress
        first_index, first = hits[0]
        began, now = _stamp(lines[first_index]), _stamp(lines[last_index])
        first_done = int(first.group(1))
        if began and now and now > began and done > first_done and total >= done:
            rate = (done - first_done) / (now - began).total_seconds()
            progress.eta_seconds = (total - done) / rate
            progress.eta_source = f"the rate over the last {len(hits)} progress line(s)"
        return progress
    return None


def read_status(name: str, folder: Path, *, now: float | None = None) -> Status:
    now = time.time() if now is None else now
    log = folder / f"{name}.log"
    exit_file = folder / f"{name}.exit"
    lines = _tail_lines(log) if log.is_file() else []
    quiet = now - log.stat().st_mtime if log.is_file() else 0.0
    exit_code = None
    if exit_file.is_file():
        try:
            exit_code = int(exit_file.read_text(encoding="utf-8").strip())
        except ValueError:
            exit_code = -1
    if exit_code is not None:
        state = "FINISHED"
    elif quiet > SILENT_MINUTES * 60:
        state = "SILENT"
    else:
        state = "RUNNING"
    last = next((line for line in reversed(lines) if line.strip()), "")
    status = Status(name=name, state=state, exit_code=exit_code, quiet_seconds=quiet, last_line=last[:300],
                    progress=find_progress(lines))
    report = newest_report(folder, name)
    if report is not None:
        status.report = report.name
        try:
            status.stages = json.loads(report.read_text(encoding="utf-8")).get("stages", [])
        except (OSError, ValueError):
            status.stages = []
    return status


def newest_report(folder: Path, name: str) -> Path | None:
    """The run report written most recently; the one whose version the run's name mentions when several match."""
    reports = sorted(folder.glob("run-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for report in reports:
        version = report.stem.rsplit("-", 1)[-1]
        if version and version in name:
            return report
    return reports[0] if reports else None


def run_names(folder: Path) -> list[str]:
    """Runs this pod knows, the newest log first."""
    logs = [p for p in folder.glob("*.log") if not p.name.endswith(".notified")]
    return [p.stem for p in sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)]


def _duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def render(status: Status) -> str:
    out = [f"run '{status.name}': {status.state}" + (f" (exit {status.exit_code})" if status.exit_code is not None
                                                  else "")]
    if status.state == "SILENT":
        out.append(f"  the log has been quiet for {_duration(status.quiet_seconds)}: it may be stuck. "
                   "Check the last line below and `nvidia-smi`; resume with --from-stage if it is.")
    elif status.state == "RUNNING":
        out.append(f"  last log line {_duration(status.quiet_seconds)} ago")
    if status.progress:
        p = status.progress
        text = f"  {p.name}: {p.done}/{p.total} ({100 * p.done / p.total:.0f}%)" if p.total else f"  {p.name}: {p.done}"
        if p.eta_seconds is not None:
            text += f"; about {_duration(p.eta_seconds)} left (from {p.eta_source})"
        out.append(text)
    if status.stages:
        out.append(f"  stages so far ({status.report}):")
        out += [f"    [{s.get('status', '?'):9}] {s.get('stage', '?')}: {str(s.get('detail', ''))[:100]}"
                for s in status.stages]
    if status.last_line:
        out.append(f"  last line: {status.last_line}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=None, help="a run started with pod_run.sh; default: the newest")
    parser.add_argument("--list", action="store_true", help="every run this pod has a log for")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args(argv)
    folder = Path(args.log_dir) if args.log_dir else log_dir()
    if not folder.is_dir():
        print(f"no log folder at {folder}", file=sys.stderr)
        return 1
    names = run_names(folder)
    if args.list:
        for name in names:
            print(render(read_status(name, folder)).splitlines()[0])
        return 0
    name = args.name or (names[0] if names else None)
    if not name:
        print(f"no run logs in {folder}", file=sys.stderr)
        return 1
    if not (folder / f"{name}.log").is_file():
        print(f"no log for run '{name}' in {folder}; runs: {', '.join(names) or 'none'}", file=sys.stderr)
        return 1
    print(render(read_status(name, folder)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
