"""A release's score under the scoring code as it stands now: no model, no GPU, nothing is overwritten.

Re-scores the answers the gate stored (``evaluation.rescore.rescore_report``) against the frozen golds and prints the
stored metrics beside the new ones, with each floor. The stored report in Blob is left as it is; the new report is
written to ``--out``, which ``scripts/sort_gate_errors.py --file`` can read.

    python scripts/rescore_version.py --version v3 --out /workspace/logs/v3_rescored.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: The metrics printed, the gate's floors first.
SHOWN = (
    "field_exact_match", "field_normalized_match", "served_field_match", "real_ocr_plus_image_accuracy",
    "worst_line_field_match", "list_field_recall", "schema_validity_rate", "confusable_misattribution_rate",
    "field_precision", "field_recall", "field_f1", "field_f1_list_fields", "image_only_accuracy",
    "held_out_carrier_match", "hallucination_rate", "false_null_rate", "coverage_code_recall", "reference_accuracy",
    "additional_fields_precision", "additional_fields_recall",
)


def _cell(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def comparison(before: dict[str, Any], after: dict[str, Any], floors: dict[str, Any]) -> str:
    """The stored metrics beside the new ones, each with its floor and the change."""
    out = [f"{'metric':32} {'floor':>7} {'stored':>8} {'re-scored':>10} {'change':>8}"]
    for name in SHOWN:
        if name not in before and name not in after:
            continue
        old, new = before.get(name), after.get(name)
        both = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (old, new))
        change = f"{new - old:+.4f}" if both else "-"
        out.append(f"{name:32} {_cell(floors.get(name)):>7} {_cell(old):>8} {_cell(new):>10} {change:>8}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True, help="the release whose stored answers to re-score, e.g. v3")
    parser.add_argument("--out", required=True, help="where to write the re-scored report (JSON)")
    parser.add_argument("--tenant", default=None)
    args = parser.parse_args(argv)

    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from common.adapter import ADAPTER
    from evaluation.gating import PILOT_FLOORS
    from evaluation.rescore import rescore_report

    client = BlobClient()
    stored = client.read_json(paths.eval_report(args.version, scope=ADAPTER.name))
    report, why = rescore_report(client, args.version, args.tenant)
    if report is None:
        print(why, file=sys.stderr)
        return 1
    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(why)
    print(comparison(stored.get("gate_metrics") or stored.get("gate_metrics_partial") or {},
                     report.get("gate_metrics") or {}, PILOT_FLOORS))
    print(f"\nForm rows aligned: {report.get('form_editions_aligned') or {}}")
    print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
