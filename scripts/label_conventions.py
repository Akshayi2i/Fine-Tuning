"""What a folder of golds teaches for form numbers and coverages: counts, for the person writing the labelling rules.

Read-only. ``data_pipeline.audit`` already reports typed fields written in two formats and placeholders; this counts
what it does not: where a form's edition sits, what counts as a coverage row, how many names one coverage has, names
split by OCR, and how many additional fields a label holds.

    python -m scripts.label_conventions --input data/bundles_c1
    python -m scripts.label_conventions --input data/bundles_c1 --split train
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

#: A number ending in what may be its edition: `` 04 13``, `` 0123``, ``-2021``, ``(11-98)``.
_EDITION_IN_NUMBER = re.compile(r"(?:[ -](0[1-9]|1[0-2])[ -]?\d{2}|[ -]20\d\d|\(\s*\d\d\s*[-/ ]\s*\d{2,4}\s*\))$")
#: A coverage named as a limit, a retention or a heading rather than as cover.
_LIMIT_WORDS = re.compile(r"\b(limit|limits|aggregate|retention|deductible)\b", re.IGNORECASE)


def form_edition_place(row: dict[str, Any]) -> str | None:
    """Where a form row holds its edition: ``field``, ``number`` or ``none``; ``None`` for a row with no number."""
    number = str(row.get("form_number") or "").strip()
    if not number:
        return None
    if row.get("edition_date"):
        return "field"
    return "number" if _EDITION_IN_NUMBER.search(number) else "none"


def spacing_variants(names: list[str]) -> list[list[str]]:
    """Names with the same letters, spaced differently: ``each occurrence limit`` and ``each occurrence lim it``."""
    by_letters: dict[str, set[str]] = defaultdict(set)
    for name in names:
        by_letters[re.sub(r"[^a-z0-9]", "", name.casefold())].add(" ".join(name.casefold().split()))
    return [sorted(group) for group in by_letters.values() if len(group) > 1]


def tally(golds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The counts over golds given as values (``values_view``), by document name."""
    places: Counter = Counter()
    mixed = []
    coverage_rows, limit_named, with_no_coverage = 0, 0, []
    rows_per_document, additional = [], []
    names_by_code: dict[str, Counter] = defaultdict(Counter)
    all_names: list[str] = []
    for document, gold in sorted(golds.items()):
        here: Counter = Counter()
        for row in gold.get("forms_and_endorsements") or []:
            place = form_edition_place(row) if isinstance(row, dict) else None
            if place:
                here[place] += 1
        places.update(here)
        if here["field"] and here["number"]:
            mixed.append(document)
        rows = [r for r in gold.get("coverages") or [] if isinstance(r, dict)]
        rows_per_document.append(len(rows))
        if not rows:
            with_no_coverage.append(document)
        for row in rows:
            name = str(row.get("coverage_name") or "")
            coverage_rows += 1
            limit_named += bool(_LIMIT_WORDS.search(name))
            if name:
                all_names.append(name)
            if row.get("coverage_code"):
                names_by_code[str(row["coverage_code"])][name.casefold()] += 1
        additional.append(len(gold.get("additional_fields") or []))
    return {
        "documents": len(golds), "form_edition_place": dict(places), "documents_mixing_form_conventions": mixed,
        "coverage_rows": coverage_rows, "coverage_rows_named_as_limits": limit_named,
        "documents_with_no_coverage_rows": with_no_coverage,
        "coverage_rows_per_document": _spread(rows_per_document),
        "coverage_codes": len(names_by_code),
        "codes_with_most_names": [
            {"code": code, "rows": sum(names.values()), "names": len(names),
             "examples": [name for name, _count in names.most_common(5)]}
            for code, names in sorted(names_by_code.items(), key=lambda item: (-len(item[1]), item[0]))[:10]],
        "names_split_by_spacing": spacing_variants(all_names),
        "additional_fields_per_document": _spread(additional),
    }


def _spread(values: list[int]) -> dict[str, float]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)} if values else {}


def render(counts: dict[str, Any]) -> str:
    places = counts["form_edition_place"]
    total = sum(places.values()) or 1
    out = [f"{counts['documents']} gold(s)", "", "Form numbers: where the edition sits"]
    out += [f"  {place:8} {count:6} ({100 * count / total:.0f}%)" for place, count in sorted(places.items())]
    out.append(f"  {len(counts['documents_mixing_form_conventions'])} document(s) hold both 'field' and 'number'")
    spread = counts["coverage_rows_per_document"]
    limits = counts["coverage_rows_named_as_limits"]
    out += ["", f"Coverage rows: {counts['coverage_rows']}, per document {spread}",
            f"  named with limit, aggregate, retention or deductible: {limits}",
            f"  documents with no coverage row: {len(counts['documents_with_no_coverage_rows'])}",
            f"  distinct coverage codes: {counts['coverage_codes']}", "", "Coverage codes with the most names"]
    for row in counts["codes_with_most_names"]:
        out.append(f"  {row['code']:36} {row['rows']:4} row(s), {row['names']:3} name(s): "
                   + "; ".join(name[:32] for name in row["examples"]))
    variants = counts["names_split_by_spacing"]
    out += ["", f"Coverage names that differ only by spacing (OCR): {len(variants)} set(s)"]
    out += [f"  {' | '.join(group[:3])}" for group in variants[:15]]
    out += ["", f"Additional fields per document: {counts['additional_fields_per_document']}"]
    return "\n".join(out)


def read_golds(root: Path, split: str | None = None) -> dict[str, dict[str, Any]]:
    from common.canonical import values_view

    splits: dict[str, str] = {}
    split_file = root / "split.csv"
    if split_file.is_file():
        with split_file.open(encoding="utf-8", newline="") as handle:
            splits = {row["document"]: row.get("split", "") for row in csv.DictReader(handle)}
    golds = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and (p / "golden.json").is_file()):
        if split and splits.get(folder.name) != split:
            continue
        golds[folder.name] = values_view(json.loads((folder / "golden.json").read_text(encoding="utf-8")))
    return golds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="a folder of document folders, each with its golden.json")
    parser.add_argument("--split", default=None, help="only this split of its split.csv: train, val or test")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    counts = tally(read_golds(Path(args.input), args.split))
    print(json.dumps(counts, indent=1, ensure_ascii=False) if args.json else render(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
