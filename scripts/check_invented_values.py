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
    python -m scripts.check_invented_values --file report.json --pages pages.json --review-csv review.csv

``--review-csv`` writes the **printed** ones as a list for whoever fixes the labels: per document, field and value,
the pages that print it and the text around it - a value the gold missed, or a row the gold lists another way. A
review file holds label values: keep it out of git, as golds and audit files are.
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


#: The list sections whose size the gold is compared with: (section, the field naming one row).
LIST_SECTIONS = (("forms_and_endorsements", "forms_and_endorsements[].form_number"),
                 ("coverages", "coverages[].coverage_name"))


def list_sizes(classified: list[dict[str, Any]], golds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per document and list section: the rows the gold lists, and the distinct rows the model wrote that the gold
    does not hold but the document prints.

    A gold that lists 12 forms beside 90 printed forms the model also found is an incomplete gold, and every one
    of those 90 scores against the model as an invented value.
    """
    out: list[dict[str, Any]] = []
    for source in sorted(golds):
        for section, field in LIST_SECTIONS:
            rows = golds[source].get(section)
            extra = {json.dumps(r["missing"] or _values(r.get("got")), default=str)
                     for r in classified if r.get("source_id") == source and r["verdict"] == "printed"
                     and field_pattern(str(r.get("field_path"))) == field}
            out.append({"source_id": source, "section": section,
                        "gold_rows": len(rows) if isinstance(rows, list) else 0,
                        "printed_but_not_in_gold": len(extra)})
    return out


def render_sizes(sizes: list[dict[str, Any]]) -> str:
    out = ["\nGold list size against what the document prints that the gold does not hold",
           f"  {'document':26} {'section':26} {'gold rows':>9} {'printed, not in gold':>21}"]
    out += [f"  {r['source_id']:26} {r['section']:26} {r['gold_rows']:9} {r['printed_but_not_in_gold']:21}"
            for r in sizes]
    return "\n".join(out)


def _plain(value: Any) -> Any:
    """A gold's values without the envelopes around them (``{"value": ..., "page": ...}`` becomes the value)."""
    from common.canonical import values_view

    return values_view(value)


def _snippet(value: Any, pages: dict[Any, str], width: int = 55) -> tuple[list[Any], str]:
    """The pages whose text holds ``value`` (its words in order, any spacing) and a little text around the first."""
    import re

    words = [re.escape(w) for w in str(value).split() if w]
    if not words:
        return [], ""
    pattern = re.compile(r"\s+".join(words), re.IGNORECASE)
    found: list[Any] = []
    snippet = ""
    for page, text in sorted(pages.items(), key=lambda item: str(item[0])):
        match = pattern.search(text or "")
        if match:
            found.append(page)
            if not snippet:
                snippet = " ".join(text[max(0, match.start() - width):match.end() + width].split())
    return found, snippet


def show_document(source_id: str, classified: list[dict[str, Any]], golds: dict[str, dict[str, Any]],
                  pages_by_document: dict[str, dict[Any, str]], shown: int = 30) -> str:
    """One document, side by side: the rows the gold lists, and the values the document prints that the gold
    does not hold, each with the pages it is on and the text around it. Whether those are forms the policy
    carries, or only mentioned in passing, is for a person to read."""
    out = [source_id]
    pages = pages_by_document.get(source_id, {})
    plain = _plain(golds.get(source_id, {}))
    for section, field in LIST_SECTIONS:
        leaf = field.rsplit(".", 1)[-1]
        rows = plain.get(section) if isinstance(plain, dict) else None
        gold_values = [str(r.get(leaf)) for r in rows if isinstance(r, dict) and r.get(leaf)] if rows else []
        extra: dict[str, None] = {}
        for r in classified:
            if (r.get("source_id") == source_id and r["verdict"] == "printed"
                    and field_pattern(str(r.get("field_path"))) == field):
                for value in _values(r.get("got")):
                    extra[str(value)] = None
        out.append(f"\n{section}: the gold lists {len(rows or [])} row(s)")
        out.append("  gold: " + "; ".join(gold_values[:shown]) + ("; ..." if len(gold_values) > shown else ""))
        out.append(f"  printed in the document, not in the gold ({len(extra)}):")
        for value in list(extra)[:shown]:
            found, text = _snippet(value, pages)
            where = ",".join(str(p) for p in found[:6]) + ("..." if len(found) > 6 else "")
            out.append(f"    {value}   [p. {where or '?'}]   ...{text}...")
    return "\n".join(out)


REVIEW_COLUMNS = ("source_id", "field", "value", "pages", "around", "readings")


def review_rows(classified: list[dict[str, Any]], pages_by_document: dict[str, dict[Any, str]]) -> list[dict[str, Any]]:
    """The printed values the gold lacks, one row per document, field and value, for a label review
    (2026-11-10: 77 of 102 values v3 was marked as inventing on its own training documents were printed)."""
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in classified:
        if record.get("verdict") != "printed":
            continue
        source = str(record.get("source_id", ""))
        field = field_pattern(str(record.get("field_path", "")))
        for value in _values(record.get("got")):
            key = (source, field, str(value))
            if key in rows:
                rows[key]["readings"] += 1
                continue
            found, around = _snippet(value, pages_by_document.get(source, {}))
            rows[key] = {"source_id": source, "field": field, "value": str(value),
                         "pages": " ".join(str(p) for p in found), "around": around, "readings": 1}
    return sorted(rows.values(), key=lambda r: (r["source_id"], r["field"], r["value"]))


def write_review(path: str, rows: list[dict[str, Any]]) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _golds_and_pages(tenant: str | None) -> tuple[dict[str, dict[Any, str]], dict[str, dict[str, Any]]]:
    from artifact_registry.blob_client import BlobClient
    from evaluation.golden_eval import load_golden_set

    documents = load_golden_set(BlobClient(), None, tenant_id=tenant)
    return ({d.source_id: dict(d.page_texts) for d in documents}, {d.source_id: d.golden for d in documents})


def _pages_from_blob(tenant: str | None) -> dict[str, dict[Any, str]]:
    return _golds_and_pages(tenant)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", help="the release whose stored eval report to read, e.g. v3")
    source.add_argument("--file", help="a downloaded eval report (JSON)")
    parser.add_argument("--pages", help="with --file: JSON {source_id: {page: text}} instead of reading Blob")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--review-csv", dest="review_csv", default=None,
                        help="also write the printed values the gold lacks to this CSV, for a label review "
                             "(keep it out of git)")
    parser.add_argument("--show", metavar="SOURCE_ID", default=None,
                        help="one document side by side: the gold's rows, and what the document prints that "
                             "the gold does not hold, with the text around each")
    args = parser.parse_args(argv)

    if args.file:
        report = json.loads(Path(args.file).read_text(encoding="utf-8"))
    else:
        from scripts.show_eval_errors import _from_blob

        report = _from_blob(args.version, args.tenant)
    records, _metrics = collect(report)
    golds: dict[str, dict[str, Any]] = {}
    if args.pages:
        pages = json.loads(Path(args.pages).read_text(encoding="utf-8"))
    else:
        pages, golds = _golds_and_pages(args.tenant)
    classified = classify(invented(records), pages)
    if args.review_csv:
        rows = review_rows(classified, pages)
        write_review(args.review_csv, rows)
        print(f"{len(rows)} printed value(s) the gold lacks written to {args.review_csv}", file=sys.stderr)
    if args.show:
        print(show_document(args.show, classified, golds, pages))
        return 0
    sizes = list_sizes(classified, golds) if golds else []
    if args.json:
        print(json.dumps({**summarise(classified), "list_sizes": sizes}, indent=2, ensure_ascii=False))
    else:
        print(render(classified) + (render_sizes(sizes) if sizes else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
