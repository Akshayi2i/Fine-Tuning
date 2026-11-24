"""Hold each correctness-critical module to its own coverage floor (CI).

CI ran the whole suite twice: once for the coverage report, once more for the
gate on the critical modules. This reads the one run's ``coverage.json``
(``pytest --cov ... --cov-report=json``) instead, so the suite runs once, and
checks each module below against its floor.

The floors: 85% for the modules the gate always held; for the serving modules
added on 24 November 2026 (review R8), the coverage measured that day, rounded
down, so it can only go up. Never a number nobody measured.

    python -m scripts.check_module_coverage coverage.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

#: module -> the lowest line coverage it may have, in percent.
FLOORS: dict[str, float] = {
    # Silent failures: broken masking does not show in a loss curve, a leaked
    # split inflates every number, a drifting prompt degrades with no symptom.
    "training.data_collator": 85,
    "data_pipeline.dataset_builder.split_groups": 85,
    "inference_core.input_builder": 85,
    "inference_core.span_map": 85,
    "common.prompts": 85,
    "common.normalize": 85,
    "evaluation.gating": 85,
    "calibration.apply_calibration": 85,
    "registry_utils.models": 85,
    # The served path (review R8): measured on 2026-11-24, rounded down.
    "serving.pipeline": 97,
    "serving.policy_merge": 93,
    "serving.printed_values": 97,
    "serving.fields_first": 97,
}


def _file_of(module: str) -> str:
    return module.replace(".", "/") + ".py"


def module_coverage(report: dict) -> dict[str, float | None]:
    """Each module of :data:`FLOORS`: its line coverage in percent, None when
    the report does not hold it (not measured: a failure, never a pass)."""
    files = {Path(name).as_posix(): data for name, data in (report.get("files") or {}).items()}
    out: dict[str, float | None] = {}
    for module in FLOORS:
        wanted = _file_of(module)
        data = next((value for name, value in files.items() if name == wanted or name.endswith("/" + wanted)), None)
        out[module] = None if data is None else float(data["summary"]["percent_covered"])
    return out


def problems(report: dict) -> list[str]:
    found = []
    for module, percent in module_coverage(report).items():
        if percent is None:
            found.append(f"{module}: not in the coverage report (is it passed to --cov?)")
        elif percent < FLOORS[module]:
            found.append(f"{module}: {percent:.1f}% covered, below its floor of {FLOORS[module]}%")
    return found


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0] if args else "coverage.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    for module, percent in module_coverage(report).items():
        shown = "not measured" if percent is None else f"{percent:.1f}%"
        print(f"{module:48s} {shown:>13s}  (floor {FLOORS[module]}%)")
    found = problems(report)
    for problem in found:
        print(f"::error::{problem}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
