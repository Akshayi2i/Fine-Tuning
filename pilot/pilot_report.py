"""The go/no-go summary across all three experiments (IMPL-15 §4).

Puts baseline, smoke test and pilot run side by side with their thresholds,
actuals and pass/fail, and states which criteria failed along with the diagnosis
path for each.

**This is the gate the deferred work waits on.** The hyperparameter sweep
(IMPL-06 ``sweep.py``, arch §11a) runs *after* this protocol passes and *before*
production-scale training: sweeping against 25–30 documents per type measures
noise, not signal, so running it earlier buys a confidently wrong config.

The order matters and is enforced here rather than trusted: a pilot report with
no smoke test in front of it is a pilot whose failures cannot be attributed
between a code bug and an architectural limit.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pilot import pilot_run, smoke_test, zero_shot_baseline

log = logging.getLogger(__name__)

PILOT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PILOT_ROOT / "reports"

EXPERIMENT_ORDER: tuple[tuple[str, str], ...] = (
    ("A", "zero_shot_baseline"),
    ("B", "smoke_test"),
    ("C", "pilot_run"),
)


@dataclass
class GoNoGo:
    """The decision, and what it rests on."""

    decision: str  # "go" | "no_go" | "incomplete"
    completed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def go(self) -> bool:
        return self.decision == "go"

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "completed_experiments": self.completed,
            "missing_experiments": self.missing,
            "blocking": self.blocking,
            "rationale": self.rationale,
        }


def load_reports(root: Path = REPORTS_DIR) -> dict[str, dict[str, Any]]:
    """Whatever has been run so far. A missing report is absence, not failure."""
    found = {}
    for _letter, name in EXPERIMENT_ORDER:
        path = root / f"{name}.json"
        if path.exists():
            found[name] = json.loads(path.read_text(encoding="utf-8"))
    return found


def decide(reports: dict[str, dict[str, Any]]) -> GoNoGo:
    """Go/no-go across the three experiments, in order.

    An experiment that was never run **blocks**; it does not pass by default.
    That is the whole reason the protocol is ordered: skipping A and B means a
    failing C cannot be attributed, and "we'll come back to it" never does.
    """
    # An empty or truncated report file is not a completed experiment. Treating
    # `{}` as present let a zero-content zero_shot_baseline.json count toward the
    # decision, and its missing `decision.proceed` then read as non-blocking — so
    # the protocol could return `go` on nothing at all.
    completed = [name for _letter, name in EXPERIMENT_ORDER if reports.get(name)]
    missing = [name for _letter, name in EXPERIMENT_ORDER if not reports.get(name)]
    blocking: list[str] = []

    baseline = reports.get("zero_shot_baseline")
    if baseline and not baseline.get("decision", {}).get("proceed", False):
        blocking.append(
            "Experiment A: " + baseline.get("decision", {}).get("rationale", "baseline below the floor")
        )

    smoke = reports.get("smoke_test")
    if smoke and not smoke.get("passed", False):
        blocking += [f"Experiment B: {reason}" for reason in smoke.get("failures", [])]

    pilot = reports.get("pilot_run")
    if pilot and not pilot.get("passed", False):
        blocking += [
            f"Experiment C: {c['criterion']} = {c['value']} (target {c['threshold']}) — {c['diagnosis']}"
            for c in pilot.get("criteria", [])
            if not c.get("met")
        ]

    if missing:
        return GoNoGo(
            decision="incomplete",
            completed=completed,
            missing=missing,
            blocking=blocking,
            rationale=(
                f"{', '.join(missing)} {'has' if len(missing) == 1 else 'have'} not been run. "
                "The experiments are ordered because each "
                "one attributes the next one's failures: without B, a failing C cannot be told "
                "apart from a code bug."
            ),
        )
    if blocking:
        return GoNoGo(
            decision="no_go",
            completed=completed,
            blocking=blocking,
            rationale=(
                "One or more criteria failed. Each has a named diagnosis path — expand corpus "
                "coverage for the failing case and re-run the pilot rather than abandoning the "
                "architecture or lowering the threshold."
            ),
        )
    return GoNoGo(
        decision="go",
        completed=completed,
        rationale=(
            "All three experiments passed. Scale annotation, and run the deferred hyperparameter "
            "sweep (IMPL-06) now — after this protocol and before production-scale training. "
            "Re-baseline at real scale: these pilot thresholds are directional and must not be "
            "carried forward as production gates."
        ),
    )


def render(reports: dict[str, dict[str, Any]], decision: GoNoGo | None = None) -> str:
    """The human-readable go/no-go document."""
    decision = decision or decide(reports)
    lines = [
        "PILOT VALIDATION PROTOCOL — go/no-go",
        f"generated {datetime.now(UTC).isoformat()}",
        "",
    ]

    for letter, name in EXPERIMENT_ORDER:
        payload = reports.get(name)
        if payload is None:
            lines += [f"Experiment {letter} — {name}: NOT RUN", ""]
            continue
        lines.append(_render_one(name, payload))
        lines.append("")

    lines.append(f"DECISION: {decision.decision.upper()}")
    lines.append(f"  {decision.rationale}")
    if decision.blocking:
        lines.append("  blocking:")
        lines += [f"    - {item}" for item in decision.blocking]
    return "\n".join(lines)


def _render_one(name: str, payload: dict[str, Any]) -> str:
    """Re-render a stored report through its own module's renderer.

    Rebuilding the dataclass rather than formatting the JSON keeps one
    description of each experiment: a summary that formatted the dict separately
    would drift from the report it summarises.
    """
    if name == "zero_shot_baseline":
        report = zero_shot_baseline.BaselineReport(
            documents=payload.get("documents", 0),
            by_doc_type=payload.get("field_f1_by_doc_type", {}),
            documents_by_doc_type=payload.get("documents_by_doc_type", {}),
            metrics=payload.get("metrics", {}),
            failure_modes=payload.get("failure_modes", {}),
            weakest_fields=[(str(row[0]), float(row[1])) for row in payload.get("weakest_fields", [])],
        )
        return zero_shot_baseline.render(report)

    if name == "smoke_test":
        report_b = smoke_test.SmokeReport(
            documents_per_type=payload.get("documents_per_type", {}),
            loss_curve=payload.get("loss_curve", []),
            train_f1=payload.get("train_field_f1"),
            components=[
                smoke_test.ComponentResult(c["component"], c["passed"], c.get("detail", ""))
                for c in payload.get("components", [])
            ],
        )
        return smoke_test.render(report_b)

    report_c = pilot_run.PilotReport(
        documents_by_doc_type=payload.get("documents_by_doc_type", {}),
        held_out_labels=payload.get("held_out_surface_labels", {}),
        run_ids=payload.get("run_ids", []),
        results=[
            pilot_run.CriterionResult(pilot_run.CRITERION_BY_NAME[c["criterion"]], c["value"])
            for c in payload.get("criteria", [])
            if c["criterion"] in pilot_run.CRITERION_BY_NAME
        ],
    )
    return pilot_run.render(report_c)


def write_summary(reports: dict[str, dict[str, Any]], root: Path = REPORTS_DIR) -> tuple[Path, Path]:
    decision = decide(reports)
    text_path = root / "pilot_summary.txt"
    json_path = root / "pilot_summary.json"
    root.mkdir(parents=True, exist_ok=True)
    text_path.write_text(render(reports, decision), encoding="utf-8")
    json_path.write_text(
        json.dumps({"decision": decision.as_dict(), "experiments": reports}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return text_path, json_path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pilot go/no-go summary")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--write", action="store_true", help="also write pilot_summary.{txt,json}")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    reports = load_reports(args.reports)
    decision = decide(reports)
    print(render(reports, decision))
    if args.write:
        text_path, _json_path = write_summary(reports, args.reports)
        log.info("wrote %s", text_path)
    return 0 if decision.go else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
