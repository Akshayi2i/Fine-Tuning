"""Of the values the gate calls invented, which are printed in the document and which are printed nowhere.

The gate's "invented" class means the gold states nothing for a field and the model wrote a value that no other
field of the gold holds. It does not say the document lacks the value: a gold that missed a row, or a row the model
listed that the gold lists differently, is counted the same as a value made up out of nothing. This looks each such
value up in the document's own page text (the frozen set's ``page_N.md``), with the same page-text search the
serving check uses (``common.grounding``), and splits them:

* **printed** - the document holds the value somewhere: the gold missed it or the rows do not line up. Not a
  hallucination; a data or matching problem.
* **not printed** - no page holds it: the model wrote something the document does not say. These are the
  inventions to fix.
* **unchecked** - too short or not a word or figure to look for (a flag, a one-letter code).

    python scripts/check_invented_values.py --version v3        # reads Blob (needs the AZURE_* settings)
    python scripts/check_invented_values.py --file report.json --pages pages.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.show_eval_errors import _cell, collect, field_pattern, invented

NOT_PRINTED_SHOWN = 40


def _values(got: Any) -> list[Any]:
    """The scalar values a served answer holds: the list's items, the object's values, or the value itself."""
    if isinstance(got, (list, tuple)):
        return [v for item in got for v in _values(item)]
    if isinstance(got, dict):
        return [v for item in got.values() for v in _values(item)]
    return [] if got is None or got == "" else [got]


def classify(records: list[dict[str, Any]], pages_by_document: dict[str, dict[Any, str]]) -> list[dict[str, Any]]:
    """Each invented record with ``verdict`` (printed / not printed / unchecked / no text) and the values not found.

    A value is looked for on every page of its document; the same value of the same document is looked up once
    however many readings wrote it.
    """
    from common.grounding import PageText, checkable, on_page

    pages: dict[str, list[Any]] = {}
    seen: dict[tuple[str, str], bool | None] = {}
    out: list[dict[str, Any]] = []
    for record in records:
        source = str(record.get("source_id", ""))
        if source not in pages:
            pages[source] = [PageText(text) for _page, text in sorted(pages_by_document.get(source, {}).items(),
                                                                      key=lambda item: str(item[0]))]
        if not any(page.has_text for page in pages[source]):
            out.append({**record, "verdict": "no text", "missing": []})
            continue
        missing: list[Any] = []
        checked = 0
        for value in _values(record.get("got")):
            if not checkable(value):
                continue
            checked += 1
            key = (source, json.dumps(value, sort_keys=True, default=str))
            if key not in seen:
                seen[key] = any(on_page(value, page) for page in pages[source])
            if not seen[key]:
                missing.append(value)
        verdict = "unchecked" if not checked else ("not printed" if missing else "printed")
        out.append({**record, "verdict": verdict, "missing": missing})
    return out


def summarise(classified: list[dict[str, Any]]) -> dict[str, Any]:
    by_field: dict[str, Counter] = {}
    by_document: dict[str, Counter] = {}
    for r in classified:
        by_field.setdefault(field_pattern(str(r.get("field_path", ""))), Counter())[r["verdict"]] += 1
        by_document.setdefault(str(r.get("source_id", "")), Counter())[r["verdict"]] += 1
    return {
        "total": len(classified),
        "by_verdict": dict(Counter(r["verdict"] for r in classified).most_common()),
        "by_field": {k: dict(v.most_common()) for k, v in sorted(by_field.items(), key=lambda i: -sum(i[1].values()))},
        "by_document": {k: dict(v.most_common()) for k, v in sorted(by_document.items())},
    }


def render(classified: list[dict[str, Any]]) -> str:
    summary = summarise(classified)
    total = summary["total"] or 1
    out = [f"{summary['total']} invented value(s) in the gate's report, looked up in the documents' own page text",
           "  " + ", ".join(f"{k} {v} ({100 * v / total:.0f}%)" for k, v in summary["by_verdict"].items()),
           "\nBy document"]
    out += [f"  {name:28} " + ", ".join(f"{k} {v}" for k, v in counts.items())
            for name, counts in summary["by_document"].items()]
    out.append("\nBy field (most first)")
    out += [f"  {name:50} " + ", ".join(f"{k} {v}" for k, v in counts.items())
            for name, counts in list(summary["by_field"].items())[:20]]
    seen: set[tuple[str, str, str]] = set()
    lines: list[str] = []
    for r in classified:
        if r["verdict"] != "not printed":
            continue
        key = (str(r.get("source_id")), field_pattern(str(r.get("field_path"))), json.dumps(r["missing"], default=str))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {key[0]:24} {key[1]:44} wrote {_cell(r['missing'])}")
    out.append(f"\nWritten but printed nowhere in the document: {len(lines)} distinct value(s)")
    out += lines[:NOT_PRINTED_SHOWN]
    if len(lines) > NOT_PRINTED_SHOWN:
        out.append(f"  ... and {len(lines) - NOT_PRINTED_SHOWN} more")
    return "\n".join(out)


def _pages_from_blob(tenant: str | None) -> dict[str, dict[Any, str]]:
    from artifact_registry.blob_client import BlobClient
    from evaluation.golden_eval import load_golden_set

    return {d.source_id: dict(d.page_texts) for d in load_golden_set(BlobClient(), None, tenant_id=tenant)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", help="the release whose stored eval report to read, e.g. v3")
    source.add_argument("--file", help="a downloaded eval report (JSON)")
    parser.add_argument("--pages", help="with --file: JSON {source_id: {page: text}} instead of reading Blob")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.file:
        report = json.loads(Path(args.file).read_text(encoding="utf-8"))
    else:
        from scripts.show_eval_errors import _from_blob

        report = _from_blob(args.version, args.tenant)
    records, _metrics = collect(report)
    pages = (json.loads(Path(args.pages).read_text(encoding="utf-8")) if args.pages
             else _pages_from_blob(args.tenant))
    classified = classify(invented(records), pages)
    print(json.dumps(summarise(classified), indent=2, ensure_ascii=False) if args.json else render(classified))
    return 0


if __name__ == "__main__":
    sys.exit(main())
