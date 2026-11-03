"""What v3 got wrong, from the gate's stored report: no GPU, no model, nothing is re-run.

The eval report keeps every failing field (``error_records``: the document, the field, the gold value, the
served value, the error class, the reading mode, whether it is a scan, the line and the carrier). This reads
it and prints where the errors are, and the values the model invented, so the next fix is chosen on evidence.

    python scripts/show_eval_errors.py --version v3            # from Blob (needs the AZURE_* settings)
    python scripts/show_eval_errors.py --file report.json      # from a downloaded report

Error classes are whatever the gate recorded (left empty, invented, misread, value of another field, other
wrong value); the script does not rename them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

#: Metrics worth printing beside the errors, when the report holds them.
METRICS = (
    "field_exact_match", "field_normalized_match", "served_field_match", "real_ocr_plus_image_accuracy",
    "worst_line_field_match", "hallucination_rate", "false_null_rate", "list_field_recall",
    "schema_validity_rate", "auto_accept_error_rate", "run_to_run_agreement",
)

#: Invented values listed in full, at most.
INVENTED_SHOWN = 25


def collect(obj: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(error_records, metrics)`` found anywhere in a report: the records of every part, and the first
    value of each metric of interest."""
    records: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            found = node.get("error_records")
            if isinstance(found, list):
                records.extend(r for r in found if isinstance(r, dict))
            body = node.get("metrics")
            if isinstance(body, dict):
                for name in METRICS:
                    if name in body and name not in metrics:
                        metrics[name] = body[name]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(obj)
    return records, metrics


def field_pattern(path: str) -> str:
    """A field's path with its rows unnumbered: ``coverages[3].limit`` is ``coverages[].limit``."""
    return re.sub(r"\[[^\]]*\]", "[]", path or "")


def breakdown(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The records totalled by class, and by class within line, mode, scan, document and field."""
    def split(key: str) -> dict[str, dict[str, int]]:
        out: dict[str, Counter] = {}
        for r in records:
            out.setdefault(str(r.get(key) if r.get(key) not in (None, "") else "unknown"), Counter())[
                r.get("error_class") or "unknown"] += 1
        return {name: dict(counts.most_common()) for name, counts in sorted(out.items())}

    by_field: dict[str, Counter] = {}
    for r in records:
        by_field.setdefault(field_pattern(str(r.get("field_path", ""))), Counter())[
            r.get("error_class") or "unknown"] += 1
    top_fields = sorted(by_field.items(), key=lambda item: -sum(item[1].values()))[:20]
    return {
        "total": len(records),
        "by_class": dict(Counter(r.get("error_class") or "unknown" for r in records).most_common()),
        "by_line": split("lob"),
        "by_mode": split("modality_mode"),
        "by_scan": split("is_scanned"),
        "by_document": split("source_id"),
        "top_fields": [{"field": name, "errors": sum(c.values()), "by_class": dict(c.most_common())}
                       for name, c in top_fields],
    }


def invented(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The records whose class names an invented value, in the order the gate wrote them."""
    return [r for r in records if "invent" in str(r.get("error_class") or "").lower()]


def _cell(value: Any, width: int = 48) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def render(records: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    out: list[str] = []
    if metrics:
        out.append("Metrics in the report")
        out += [f"  {name:32} {metrics[name]}" for name in METRICS if name in metrics]
        out.append("")
    if not records:
        out.append("The report holds no error records (a perfect score, or a report from before they were kept).")
        return "\n".join(out)
    result = breakdown(records)
    out.append(f"{result['total']} failing field(s) in all")
    out.append("  by class: " + ", ".join(f"{k} {v}" for k, v in result["by_class"].items()))
    for title, key in (("line", "by_line"), ("reading mode", "by_mode"), ("scan", "by_scan"),
                       ("document", "by_document")):
        out.append(f"\nBy {title}")
        for name, counts in result[key].items():
            out.append(f"  {name:36} {sum(counts.values()):5}   " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    out.append("\nFields that fail most")
    for row in result["top_fields"]:
        out.append(f"  {row['field']:52} {row['errors']:4}   "
                   + ", ".join(f"{k} {v}" for k, v in row["by_class"].items()))
    made_up = invented(records)
    out.append(f"\nInvented values: {len(made_up)} (the gold states nothing, the model wrote a value)")
    for r in made_up[:INVENTED_SHOWN]:
        out.append(f"  {str(r.get('source_id', '')):28} {field_pattern(str(r.get('field_path', ''))):44} "
                   f"wrote {_cell(r.get('got'))}")
    if len(made_up) > INVENTED_SHOWN:
        out.append(f"  … and {len(made_up) - INVENTED_SHOWN} more")
    return "\n".join(out)


def _from_blob(version: str, tenant: str | None) -> Any:
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient

    client = BlobClient()
    keys = [paths.eval_report(version, "policy", scope="policy"), paths.eval_report(version, scope="policy"),
            paths.eval_report(version, "policy"), paths.eval_report(version)]
    found = [client.read_json(key) for key in dict.fromkeys(keys) if client.exists(key)]
    if not found:
        raise SystemExit(f"no eval report for {version} at any of: " + ", ".join(dict.fromkeys(keys)))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", help="the release whose stored eval report to read, e.g. v3")
    source.add_argument("--file", help="a downloaded eval report (JSON)")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--json", action="store_true", help="print the breakdown as JSON")
    args = parser.parse_args(argv)

    report = json.loads(Path(args.file).read_text(encoding="utf-8")) if args.file else _from_blob(
        args.version, args.tenant)
    records, metrics = collect(report)
    if args.json:
        print(json.dumps({"metrics": metrics, **breakdown(records), "invented": invented(records)},
                         indent=2, ensure_ascii=False))
    else:
        print(render(records, metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
