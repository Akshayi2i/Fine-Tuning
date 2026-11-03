"""Every failing field of the gate's report, sorted by a cause the records themselves prove.

Rows of a forms list pair on their form number (``evaluation.metrics.field_accuracy``): a model row whose form
number no gold row holds faces an empty row, and every value in it counts as invented; a gold row nothing pairs
with counts as missed. This sorts the failures mechanically, with no judgement:

* forms rows, unpaired, whose form number differs from a gold one only by the edition written into it
  (``SSI SF 00 03`` against ``SSI SF 00 03 0123``): the same form written two ways, scored as two errors;
* forms rows the model wrote that no gold form number matches, even so;
* gold forms rows nothing paired with, even so;
* failures inside forms rows that paired;
* every other list section, and the fields outside lists, by the gate's own error class.

    python scripts/sort_gate_errors.py --version v3          # reads Blob (needs the AZURE_* settings)
    python scripts/sort_gate_errors.py --file report.json --golds golds.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.show_eval_errors import collect

FORMS = "forms_and_endorsements"
_ROW = re.compile(r"^(?P<section>[A-Za-z_.]+?)\[(?P<index>\d+)\]\.(?P<field>.+)$")
#: What may follow a form number as its edition once spaces and dashes are taken out: "0123", "2021", "1021".
_EDITION_TAIL = re.compile(r"^\d{2,6}$")

EDITION_TWIN = "forms: same form, edition written into the number on one side only"
MODEL_ONLY = "forms: row the model wrote, no gold form number matches"
GOLD_ONLY = "forms: gold row nothing paired with"
PAIRED = "forms: wrong or missing value inside a paired row"


def _squeeze(text: Any) -> str:
    return re.sub(r"[\s\-]", "", str(text or "")).upper()


def edition_twin(number: Any, others: list[Any]) -> Any | None:
    """The one of ``others`` that is ``number`` with an edition added or taken away, or None."""
    mine = _squeeze(number)
    if len(mine) < 4:
        return None
    for other in others:
        theirs = _squeeze(other)
        if theirs == mine or len(theirs) < 4:
            continue
        longer, shorter = (theirs, mine) if len(theirs) > len(mine) else (mine, theirs)
        if longer.startswith(shorter) and _EDITION_TAIL.match(longer[len(shorter):]):
            return other
    return None


def _gold_form_numbers(gold: dict[str, Any]) -> list[Any]:
    from common.canonical import values_view

    rows = values_view(gold).get(FORMS) or []
    return [r.get("form_number") for r in rows if isinstance(r, dict) and r.get("form_number")]


def sort_records(records: list[dict[str, Any]], golds: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Each record with a ``cause``."""
    gold_numbers = {source: _gold_form_numbers(gold) for source, gold in golds.items()}
    rows: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        match = _ROW.match(str(r.get("field_path", "")))
        if match and match["section"] == FORMS:
            rows[(r.get("source_id"), r.get("modality_mode"), int(match["index"]))].append(r)
    # The model's unpaired form numbers in each reading, to find a gold row's twin among them.
    model_numbers: dict[tuple, list[Any]] = defaultdict(list)
    for (source, mode, _index), row in rows.items():
        for r in row:
            if r["field_path"].endswith(".form_number") and r.get("error_class") == "invented":
                model_numbers[(source, mode)].append(r.get("got"))

    cause_of_row: dict[tuple, str] = {}
    for key, row in rows.items():
        source, mode, _index = key
        number = next((r for r in row if r["field_path"].endswith(".form_number")), None)
        if number is not None and number.get("error_class") == "invented":
            twin = edition_twin(number.get("got"), gold_numbers.get(source, []))
            cause_of_row[key] = EDITION_TWIN if twin is not None else MODEL_ONLY
        elif (number is not None and number.get("error_class") == "omission"
              and all(r.get("error_class") == "omission" for r in row)):
            twin = edition_twin(number.get("expected"), model_numbers.get((source, mode), []))
            cause_of_row[key] = EDITION_TWIN if twin is not None else GOLD_ONLY
        else:
            cause_of_row[key] = PAIRED

    out = []
    for r in records:
        match = _ROW.match(str(r.get("field_path", "")))
        if match and match["section"] == FORMS:
            cause = cause_of_row[(r.get("source_id"), r.get("modality_mode"), int(match["index"]))]
        elif match:
            cause = f"{match['section']}: {r.get('error_class') or 'unknown'}"
        else:
            cause = f"outside lists: {r.get('error_class') or 'unknown'}"
        out.append({**r, "cause": cause})
    return out


def summarise(sorted_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cause = Counter(r["cause"] for r in sorted_records)
    by_document: dict[str, Counter] = defaultdict(Counter)
    for r in sorted_records:
        by_document[str(r.get("source_id"))][r["cause"]] += 1
    return {"total": len(sorted_records), "by_cause": dict(by_cause.most_common()),
            "by_document": {k: dict(v.most_common()) for k, v in sorted(by_document.items())}}


def render(summary: dict[str, Any], top: int = 25) -> str:
    total = summary["total"] or 1
    out = [f"{summary['total']} failing field(s), sorted by cause"]
    out += [f"  {count:6} ({100 * count / total:4.1f}%)  {cause}"
            for cause, count in list(summary["by_cause"].items())[:top]]
    rest = list(summary["by_cause"].items())[top:]
    if rest:
        out.append(f"  {sum(c for _, c in rest):6}          ... {len(rest)} smaller cause(s)")
    out.append("\nForms causes by document")
    for document, causes in summary["by_document"].items():
        forms = {c: n for c, n in causes.items() if c.startswith("forms:")}
        out.append(f"  {document:14} " + ", ".join(f"{c.split(': ', 1)[1][:40]} {n}" for c, n in forms.items()))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--version", help="the release whose stored eval report to read, e.g. v3")
    source.add_argument("--file", help="a downloaded eval report (JSON)")
    parser.add_argument("--golds", help="with --file: JSON {source_id: gold} instead of reading Blob")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.file:
        report = json.loads(Path(args.file).read_text(encoding="utf-8"))
    else:
        from scripts.show_eval_errors import _from_blob

        report = _from_blob(args.version, args.tenant)
    if args.golds:
        golds = json.loads(Path(args.golds).read_text(encoding="utf-8"))
    else:
        from scripts.check_invented_values import _golds_and_pages

        golds = _golds_and_pages(args.tenant)[1]
    records, _metrics = collect(report)
    summary = summarise(sort_records(records, golds))
    print(json.dumps(summary, indent=2, ensure_ascii=False) if args.json else render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
