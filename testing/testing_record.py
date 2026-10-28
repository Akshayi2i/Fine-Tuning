"""The testing record: one row per document an extraction run read, in the
columns of the team's workbook ("Testing Record" sheet of
``Documentation/Testing Records/FIdeon SLM - Model Training & Test Record_v1.xlsx``).

The extraction command writes the rows as it goes (``testing/metrics/<model>/
test_record.csv``, :func:`record_row`); this command copies them into the
workbook, under the round they belong to::

    python -m testing.testing_record --model base --round Baseline \\
        --xlsx "Documentation/Testing Records/FIdeon SLM - Model Training & Test Record_v1.xlsx"

What each column measures:

* **F1 Score Accuracy** - the document's field match rate on its scalar fields
  (``field_normalized_match_rate``, the gate's matching rules). Each scalar
  field has one predicted and one expected value, so precision, recall and F1
  coincide (pilot/zero_shot_baseline.py); table rows are scored apart, as list
  recall, in the run's metrics. Empty when the gold was not scored.
* **Latency** - seconds the model spent generating the document's answer
  (``ExtractionResult.latency_ms``); OCR and page loading are not in it.
* **Input / Output Tokens** - summed over every window read, retries included
  (``window_stats``): what the document cost the GPU.
* **Carrier** - as the gold names it (``carrier.name``), else as the label
  metadata records it.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: The workbook's sheet and its headers, in order (matched without surrounding spaces).
SHEET = "Testing Record"
COLUMNS = ("Training Round", "Training/Testing Date", "Test/Val", "Carrier", "LOB", "Doc ID", "# Pages",
           "F1 Score Accuracy", "Latency", "Input Tokens", "Output Tokens")
#: Beside the sheet's columns in the CSV: what read the document, and how.
EXTRA = ("Model", "Mode")
RECORD_FILE = "test_record.csv"


class RecordError(RuntimeError):
    pass


def _carrier(golden: dict[str, Any] | None, metadata: dict[str, Any]) -> str | None:
    from common.canonical import values_view

    carrier = values_view(golden or {}).get("carrier")
    if isinstance(carrier, dict):
        # `name` on the common model, `company_name` on a self-contained schema.
        carrier = carrier.get("name") or carrier.get("company_name")
    if isinstance(carrier, str) and carrier.strip():
        return carrier.strip()
    return metadata.get("carrier") or None


def _tokens(result: Any, key: str) -> int | None:
    counts = [stat.get(key) for stat in getattr(result, "window_stats", None) or []]
    counts = [count for count in counts if isinstance(count, int)]
    return sum(counts) if counts else None


def record_row(result: Any, metrics: dict[str, Any], *, metadata: dict[str, Any], golden: dict[str, Any] | None,
               pages: int, round_label: str, date: str | None = None) -> dict[str, Any]:
    """One document's row: the sheet's columns, then :data:`EXTRA`."""
    match = metrics.get("field_normalized_match_rate")
    return {
        "Training Round": round_label,
        "Training/Testing Date": date or datetime.now(UTC).date().isoformat(),
        "Test/Val": metadata.get("split"),
        "Carrier": _carrier(golden, metadata),
        "LOB": metadata.get("lob"),
        "Doc ID": result.source_id,
        "# Pages": pages,
        "F1 Score Accuracy": round(match, 4) if isinstance(match, (int, float)) else None,
        "Latency": round(result.latency_ms / 1000, 1) if result.latency_ms is not None else None,
        "Input Tokens": _tokens(result, "prompt_tokens"),
        "Output Tokens": _tokens(result, "answer_tokens"),
        "Model": result.model_version,
        "Mode": result.mode,
    }


def append_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Add rows to the run's record, writing the header on a new file."""
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*COLUMNS, *EXTRA])
        if new:
            writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RecordError(f"no testing record at {path}: run the extraction command first")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> Any:
    """A CSV cell as the workbook should hold it: a number where it is one."""
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() and "." not in str(value) else number


def append_to_workbook(rows: Sequence[dict[str, Any]], xlsx: Path, *, sheet: str = SHEET) -> dict[str, int]:
    """Write ``rows`` into the workbook's sheet; returns how many were written and skipped.

    Columns are found by their header text, so a reordered sheet still fills
    right. A row lands in the first row whose data columns are empty and whose
    round is empty or already its own (the sheet's pre-typed "Baseline" rows),
    so nothing typed by hand is overwritten. A document already recorded for
    the same round and date is skipped: running the command twice adds nothing.
    """
    from copy import copy

    import openpyxl

    book = openpyxl.load_workbook(xlsx)
    # Matched without surrounding spaces, as the headers are: the team's sheet
    # is named "Testing Record " as typed.
    names = [name for name in book.sheetnames if name.strip() == sheet.strip()]
    if not names:
        raise RecordError(f"{xlsx} has no sheet {sheet!r} (it has {book.sheetnames})")
    ws = book[names[0]]
    headers = {str(cell.value).strip(): cell.column for cell in ws[1] if cell.value is not None}
    missing = [name for name in COLUMNS if name not in headers]
    if missing:
        raise RecordError(f"sheet {sheet!r} has no column {missing}; its headers are {sorted(headers)}")
    col = {name: headers[name] for name in COLUMNS}
    data = [name for name in COLUMNS if name != "Training Round"]

    def cell(row: int, name: str) -> Any:
        value = ws.cell(row=row, column=col[name]).value
        return value.strip() if isinstance(value, str) else value

    def key(round_label: Any, doc: Any, date: Any) -> tuple[str, str, str]:
        return (str(round_label or "").strip(), str(doc or "").strip(), str(date or "").strip()[:10])

    recorded = {key(cell(r, "Training Round"), cell(r, "Doc ID"), cell(r, "Training/Testing Date"))
                for r in range(2, ws.max_row + 1) if cell(r, "Doc ID")}
    style_row = 2
    written = skipped = 0
    for row in rows:
        if key(row["Training Round"], row["Doc ID"], row["Training/Testing Date"]) in recorded:
            skipped += 1
            continue
        target = next(r for r in range(2, ws.max_row + 2)
                      if all(cell(r, name) in (None, "") for name in data)
                      and cell(r, "Training Round") in (None, "", str(row["Training Round"]).strip()))
        for name in COLUMNS:
            out = ws.cell(row=target, column=col[name])
            if not out.has_style and ws.cell(row=style_row, column=col[name]).has_style:
                out._style = copy(ws.cell(row=style_row, column=col[name])._style)
            out.value = _number(row.get(name))
        recorded.add(key(row["Training Round"], row["Doc ID"], row["Training/Testing Date"]))
        written += 1
    book.save(xlsx)
    return {"written": written, "skipped": skipped}


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    from testing.run_extraction import TESTING_ROOT, metrics_dir

    parser = argparse.ArgumentParser(description="Copy an extraction run's testing record into the workbook")
    parser.add_argument("--model", required=True, help="the run's model tag, as under testing/metrics/")
    parser.add_argument("--xlsx", required=True, type=Path, help="the team's training & test record workbook")
    parser.add_argument("--round", dest="round_label", default=None,
                        help="the Training Round to file the rows under (default: as recorded by the run)")
    parser.add_argument("--sheet", default=SHEET)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows = read_csv(metrics_dir(args.model, TESTING_ROOT) / RECORD_FILE)
    if args.round_label:
        rows = [{**row, "Training Round": args.round_label} for row in rows]
    counts = append_to_workbook(rows, args.xlsx, sheet=args.sheet)
    print(f"{counts['written']} row(s) written to {args.xlsx} [{args.sheet}], "
          f"{counts['skipped']} already there")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
