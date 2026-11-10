"""Does every training answer fit the schema production decodes with? Read-only: no model, no GPU.

Production decodes each window against its schema slice, its page references held to the window's pages, its
envelopes compact (``serving.pipeline._build_request``: ``resolved_schema`` -> ``with_page_bounds`` ->
``compact_schema``). The corpus build checks each target against the full schema view only
(``policy_windows._check_writable``), and only three synthetic rows were ever checked against the exact decoding
schema (2026-11-10 review, D4). An answer that does not fit is one the decoder can never write: the model is taught
it and production refuses it, token by token.

    python -m scripts.check_answers_fit_decoder --version v3            # the corpus v3 trained on (reads Blob)
    python -m scripts.check_answers_fit_decoder --file epoch_1.jsonl    # a corpus file on disk
    python -m scripts.check_answers_fit_decoder --bundles data/bundles_c1   # local golds, through the target build

``--bundles`` builds each document's window targets from its ``golden.json`` with the build's own
``window_target`` on an image-only plan (no page text: no page inference, no search placement), so it checks the
golds' shape, not the exact corpus; the corpus is the real check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

#: The page markers every request carries: page number and the document's page count.
_MARKER = re.compile(r"<page (\d+) of (\d+)>")
SHOWN = 20


def decoding_schema(lob: Any, sections: str, pages: list[int], total_pages: int, compact: bool) -> dict[str, Any]:
    """The schema production decodes one window with (``serving.pipeline._build_request``)."""
    from common.canonical import compact_schema
    from common.schemas import resolved_schema, with_page_bounds

    schema = with_page_bounds(resolved_schema("policy", None, lob, sections), total_pages, pages=pages)
    return compact_schema(schema) if compact else schema


def _pages_of(row: dict[str, Any]) -> tuple[list[int], int]:
    pages, total = [], 0
    for message in row.get("messages") or []:
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "text":
                for number, count in _MARKER.findall(part.get("text") or ""):
                    pages.append(int(number))
                    total = max(total, int(count))
    return sorted(set(pages)), total


def _answer(row: dict[str, Any]) -> Any:
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return None
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return json.loads(content or "null")


def check_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each policy window row's answer against its decoding schema: the rows that do not fit, and why."""
    from jsonschema import Draft202012Validator

    validators: dict[str, Any] = {}
    checked, misfits = 0, []
    reasons: Counter = Counter()
    for row in rows:
        if row.get("doc_type") != "policy" or not row.get("sections"):
            continue
        pages, total = _pages_of(row)
        if not pages:
            continue
        compact = bool(row.get("answer_format"))
        key = json.dumps([row.get("lob"), row["sections"], pages, total, compact])
        if key not in validators:
            validators[key] = Draft202012Validator(decoding_schema(row.get("lob"), row["sections"], pages, total,
                                                                   compact))
        checked += 1
        errors = sorted(validators[key].iter_errors(_answer(row)), key=lambda e: list(e.path))
        if errors:
            first = errors[0]
            reason = f"{'/'.join(str(p) for p in first.path) or '<root>'}: {first.message[:140]}"
            reasons[re.sub(r"\d+", "N", reason)] += 1
            misfits.append({"source_id": row.get("source_id"), "sections": row["sections"], "pages": pages,
                            "errors": len(errors), "first": reason})
    return {"checked": checked, "misfit": len(misfits), "reasons": dict(reasons.most_common(SHOWN)),
            "rows": misfits[:SHOWN]}


def rows_from_bundles(folder: Path) -> list[dict[str, Any]]:
    """Window rows built from local golds the way the corpus build cuts them (image-only plan, no page text)."""
    import pymupdf

    from common.prompts import fast_layout
    from data_pipeline.dataset_builder.policy_windows import plan_windows, window_target
    from inference_core.input_builder import build_training_row

    rows: list[dict[str, Any]] = []
    for bundle in sorted(p for p in folder.iterdir() if (p / "golden.json").is_file()):
        meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        lob = meta.get("lob")
        label = json.loads((bundle / "golden.json").read_text(encoding="utf-8"))
        with pymupdf.open(bundle / "document.pdf") as pdf:
            total = pdf.page_count
        for plan in plan_windows(lob, list(range(1, total + 1)), None, None):
            try:
                target = window_target(label, lob, plan)
            except Exception as exc:  # noqa: BLE001 - reported as the row's misfit
                rows.append({"doc_type": "policy", "source_id": bundle.name, "sections": plan.group,
                             "build_error": f"{type(exc).__name__}: {str(exc)[:160]}"})
                continue
            row = build_training_row("policy", bundle.name, [f"p{n}.png" for n in plan.pages], None, "image_only",
                                     json.dumps(target, ensure_ascii=False), lob=lob, sections=plan.group,
                                     split="train", page_numbers=list(plan.pages), total_pages=total)
            row["fast_layout"] = fast_layout("policy", None, lob, plan.group)
            rows.append(row)
    return rows


def render(report: dict[str, Any], build_errors: list[dict[str, Any]]) -> str:
    out = [f"{report['checked']} window answer(s) checked against production's decoding schema; "
           f"{report['misfit']} do not fit"]
    if build_errors:
        out.append(f"{len(build_errors)} window(s) the target build itself refused (first: "
                   f"{build_errors[0]['source_id']} {build_errors[0]['sections']}: {build_errors[0]['build_error']})")
    if report["reasons"]:
        out.append("\nwhy (first error per answer, numbers as N):")
        out += [f"  {count:>5}  {reason}" for reason, count in report["reasons"].items()]
        out.append("\nexamples:")
        out += [f"  {r['source_id']} {r['sections']} pages {r['pages'][:6]}: {r['errors']} error(s); {r['first']}"
                for r in report["rows"][:10]]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", help="the release whose training corpus to read (its first epoch file)")
    source.add_argument("--file", help="a corpus file (JSON lines) on disk")
    source.add_argument("--bundles", help="a folder of local bundles (golden.json, metadata.json, document.pdf)")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.file:
        rows = [json.loads(line) for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.bundles:
        rows = rows_from_bundles(Path(args.bundles))
    else:
        from scripts.answer_list_lengths import _rows_from_blob

        rows = _rows_from_blob(args.version, args.tenant)
    build_errors = [row for row in rows if "build_error" in row]
    report = check_rows([row for row in rows if "build_error" not in row])
    report["build_errors"] = len(build_errors)
    print(json.dumps(report, indent=1, default=str) if args.json else render(report, build_errors))
    return 1 if report["misfit"] else 0


if __name__ == "__main__":
    sys.exit(main())
