"""How long the lists are in the training answers, and whether any training answer repeats a row. Read-only.

v3 looped on five windows of its own training documents (2026-11-09) - one form number 166 times, extra fields in
cycles of 14 to 24 - while their golds held 0 to 33 extra fields and the form twice. Whether training taught long
lists, or rows written twice, is not known; this counts it, over every training row of a corpus:

* per list (``additional_fields``, ``coverages``, ``forms_and_endorsements``, ... at any depth): rows holding it,
  and its length - median, 90th and 99th percentile, the longest - and the rows with the longest;
* rows whose answer names one row twice or more in a list (the same form number, coverage name or extra-field
  label): what a model would learn to repeat from.

    python -m scripts.answer_list_lengths --version v3              # the corpus v3 trained on (reads Blob)
    python -m scripts.answer_list_lengths --file epoch_1.jsonl      # a corpus file on disk
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

#: The keys that name a row in a list (as the loop guard reads them, ``openai_backend.LOOP_IDENTIFIER_KEYS``).
NAME_KEYS = ("form_number", "coverage_name", "label", "name", "vin")
#: Rows shown per list, longest first.
SHOWN = 10


def _answer(row: dict[str, Any]) -> Any:
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return None
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    try:
        return json.loads(content or "")
    except ValueError:
        return None


def _name(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    for key in NAME_KEYS:
        value = entry.get(key)
        if isinstance(value, dict):
            value = value.get("v", value.get("parsed", value.get("r", value.get("raw"))))
        if value not in (None, ""):
            return f"{key}={value}"
    return None


def lists_in(node: Any, path: str = "") -> list[tuple[str, list[Any]]]:
    """Every list of objects in an answer, as ``(path with rows unnumbered, the list)``."""
    out: list[tuple[str, list[Any]]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out += lists_in(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        if any(isinstance(item, dict) for item in node):
            out.append((path, node))
        for item in node:
            out += lists_in(item, path + "[]")
    return out


def _percentile(values: list[int], share: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))] if ordered else 0


def measure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths: dict[str, list[tuple[int, str]]] = defaultdict(list)
    repeated: list[dict[str, Any]] = []
    unreadable = 0
    for row in rows:
        answer = _answer(row)
        if answer is None:
            unreadable += 1
            continue
        where = f"{row.get('source_id')} {row.get('task')} pages {row.get('window_pages')}"
        for path, items in lists_in(answer):
            lengths[path].append((len(items), where))
            names = Counter(name for item in items if (name := _name(item)))
            twice = {name: n for name, n in names.items() if n >= 2}
            if twice:
                repeated.append({"row": where, "list": path, "names_twice_or_more": dict(sorted(
                    twice.items(), key=lambda kv: -kv[1])[:5])})
    per_list = {}
    for path, found in sorted(lengths.items(), key=lambda kv: -max(n for n, _w in kv[1])):
        values = [n for n, _w in found]
        per_list[path] = {"rows": len(values), "median": _percentile(values, 0.5), "p90": _percentile(values, 0.9),
                          "p99": _percentile(values, 0.99), "longest": max(values),
                          "longest_rows": [f"{n}: {w}" for n, w in sorted(found, key=lambda t: -t[0])[:SHOWN]]}
    return {"rows": len(rows), "unreadable_answers": unreadable, "lists": per_list,
            "rows_naming_a_row_twice": len(repeated), "repeated": repeated[:50]}


def render(report: dict[str, Any]) -> str:
    out = [f"{report['rows']} training row(s); {report['unreadable_answers']} answer(s) not readable as JSON",
           "", f"{'list':<52} {'rows':>6} {'median':>7} {'p90':>5} {'p99':>5} {'longest':>8}"]
    for path, s in report["lists"].items():
        out.append(f"{path:<52} {s['rows']:>6} {s['median']:>7} {s['p90']:>5} {s['p99']:>5} {s['longest']:>8}")
    for path, s in list(report["lists"].items())[:3]:
        out.append(f"\nlongest {path}:")
        out += [f"  {line}" for line in s["longest_rows"][:5]]
    out.append(f"\nrows whose answer names one row twice or more in a list: {report['rows_naming_a_row_twice']}")
    out += [f"  {r['row']}  {r['list']}: {r['names_twice_or_more']}" for r in report["repeated"][:20]]
    return "\n".join(out)


def _rows_from_blob(version: str, tenant: str | None) -> list[dict[str, Any]]:
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from evaluation.validation_generation import read_rows
    from orchestration.label_noise import corpus_of

    client = BlobClient()
    corpus = corpus_of(client, version)
    return read_rows(client.read_text(paths.corpus_epoch_file(corpus, 1, tenant)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", help="the release whose training corpus to read (its first epoch file)")
    source.add_argument("--file", help="a corpus file (JSON lines) on disk")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.file:
        rows = [json.loads(line) for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = _rows_from_blob(args.version, args.tenant)
    report = measure(rows)
    print(json.dumps(report, indent=1, ensure_ascii=False) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
