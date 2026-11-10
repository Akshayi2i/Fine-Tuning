"""Why each failing field failed: the model, the gold, or the scoring - read off saved files, nothing re-run.

The scorer's classes (``training.vit_gate.classify_error``) say what the answer did - wrote nothing, wrote a value the
gold holds nowhere, wrote another value - not whose fault it is. On v3's own training documents (2026-11-09), 77 of
102 "invented" values were printed in the document. This sorts every failing field of a run into:

a) gold gap        - printed in the document, missing from the gold (the model read it; the gold did not hold it);
b) row alignment   - the same value or row is on both sides, written another way (case, spacing, an edition inside
                     the number) or under another name or code, so the scorer did not pair them;
c) field placement - the gold value is in the answer, in another field or in additional_fields;
d) labelling rules - a value the 2026-11-04 rules keep out of a gold: a form or coverage row cited only on
                     application pages (rule 5, ``data_pipeline.audit.application_pages``), a limit written as a
                     coverage row of its own (rule 2, ``data_pipeline.ingestion.label_rules.merge_coverage_limits``);
e) gold not printed - the gold value is printed on no page, while the answer's is (a candidate gold error);
f) model error     - with why: its window was cut off, it missed a printed value, it wrote a value printed nowhere,
                     it put a value in the wrong field, or it wrote another value;
g) cannot be checked - a value no page search can settle (a yes/no, a code or a choice from a list, a value too
                     short to look for, or no page text), in a row that is not a gold gap.

The page-text search is ``common.grounding``'s, the one serving and ``check_invented_values`` use. Rows are taken
from the documents as the scorer merges them (``validation_generation.merged_windows``) and matched by their own
identifiers (a coverage's code or name, a form's number - ``sort_gate_errors.edition_twin`` for an edition inside it).

    python -m scripts.sort_error_causes --errors run.errors.json --answers run.answers.jsonl --pages pages.json

``--errors`` is ``score_on_training_docs``'s whole-document report, ``--answers`` its rows, ``--pages``
``{source_id: {page: text}}``. Read-only: it changes no gold and no score.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.show_eval_errors import collect, field_pattern

GOLD_GAP = "a) gold gap"
ALIGNMENT = "b) row alignment"
PLACEMENT = "c) field placement"
RULES = "d) labelling rules"
GOLD_NOT_PRINTED = "e) gold not printed"
MODEL = "f) model error"
UNCHECKED = "g) cannot be checked"
CLASSES = (GOLD_GAP, ALIGNMENT, PLACEMENT, RULES, GOLD_NOT_PRINTED, MODEL, UNCHECKED)

FORMS = "forms_and_endorsements"
COVERAGES = "coverages"
#: Words that name a limit rather than a coverage (as ``sort_gate_errors`` counts them).
_LIMIT_WORDS = re.compile(r"\b(limit|limits|aggregate|retention|deductible)\b", re.IGNORECASE)
_SECTION = re.compile(r"^(?P<section>[A-Za-z_]+)\[\]")
#: Fields that are not values: identifiers and links the pipeline writes, never printed.
_NOT_VALUES = {"page_ref", "part_id", "unit_id", "location_ref", "applies_to", "doc_type", "modality", "page_count",
               "source_file"}


# -- values -------------------------------------------------------------------------------------------------------


def _value(cell: Any) -> Any:
    """A cell's value: an envelope's ``parsed``, else its ``raw``; a plain value as it is."""
    from common.canonical import is_field_value

    if is_field_value(cell):
        return cell.get("parsed") if cell.get("parsed") is not None else cell.get("raw")
    return cell


def _scalars(value: Any) -> list[Any]:
    """The plain values in a record's ``expected`` or ``got``: a list's items, an envelope's value."""
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in _scalars(item)]
    value = _value(value)
    if isinstance(value, dict):
        return [v for item in value.values() for v in _scalars(item)]
    return [] if value is None or value == "" else [value]


#: Fields whose value is a code or a choice from a list, never printed as such (the schema's enums and codes).
_CODE_FIELDS = re.compile(r"(?:^|\.)(?:coverage_code|basis_coverage_code|limit_type|deductible_type|form_type|"
                          r"included|valuation|role|lob|transaction_type|entity_type|applies_to_coverages)$")
#: A value in a code's shape: words joined by underscores, no spaces ("BOPGL_BPP", "per_occurrence").
_CODE_SHAPE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+$")


def _lookable(pattern: str, value: Any) -> bool:
    """Whether a page could print ``value`` as it is: not a yes/no, not a code or a choice from a list."""
    if isinstance(value, bool) or _CODE_FIELDS.search(pattern):
        return False
    return not (isinstance(value, str) and _CODE_SHAPE.match(value))


#: A value of digits and separators only: a date, an edition, a number written with slashes or dashes.
_DIGITS_ONLY = re.compile(r"^[\d\s/.\-]+$")


def _same(a: Any, b: Any) -> bool:
    """One value written two ways: the same number, the same digits once slashes and dashes are gone (an edition
    "09/02" and "0902"), or the same words once case and punctuation are gone."""
    from common.grounding import normalise

    try:
        return float(str(a).replace(",", "").replace("$", "")) == float(str(b).replace(",", "").replace("$", ""))
    except ValueError:
        pass
    if _DIGITS_ONLY.match(str(a)) and _DIGITS_ONLY.match(str(b)):
        digits_a, digits_b = re.sub(r"\D", "", str(a)), re.sub(r"\D", "", str(b))
        if len(digits_a) >= 3 and digits_a == digits_b:
            return True
    return bool(normalise(a)) and normalise(a) == normalise(b)


def _as_number(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _distinctive(pattern: str | None, value: Any, gold: list, answer: list) -> bool:
    """Whether finding ``value`` elsewhere says where it went: a name or phrase, or a figure that occurs once on each
    side (in ``pattern``'s field, or anywhere when ``pattern`` is None). A code, a choice from a list ("other") or a
    figure many rows share ("1,000,000", "1") is found in some other row whatever happened to this one."""
    from common.grounding import normalise

    if not _lookable(pattern or "", value):
        return False
    if _as_number(value) is None:
        return len(normalise(value)) >= 6

    def once(leaves: list) -> bool:
        return sum(1 for p, v, _r in leaves if (pattern is None or p == pattern) and _same(v, value)) == 1

    return once(gold) and once(answer)


def _leaves(node: Any, path: str = "") -> list[tuple[str, Any, list[int]]]:
    """Every value in a document or a row, as ``(field pattern, value, cited pages)``."""
    from common.canonical import is_field_value

    if is_field_value(node):
        value = _value(node)
        refs = [int(p) for p in node.get("page_ref") or [] if isinstance(p, int) or str(p).isdigit()]
        return [] if value is None or value == "" else [(field_pattern(path), value, refs)]
    if isinstance(node, dict):
        return [leaf for key, item in node.items() if key not in _NOT_VALUES
                for leaf in _leaves(item, f"{path}.{key}" if path else key)]
    if isinstance(node, list):
        return [leaf for index, item in enumerate(node) for leaf in _leaves(item, f"{path}[{index}]")]
    return [] if node is None or node == "" or isinstance(node, bool) else [(field_pattern(path), node, [])]


# -- one document as the scorer merged it ------------------------------------------------------------------------


class Document:
    """One document read in one mode: its merged gold and answer, its pages, the pages of its windows cut off."""

    def __init__(self, gold: dict[str, Any], answer: dict[str, Any], pages: dict[int, str], cut_pages: set[int]):
        from common.grounding import PageText
        from data_pipeline.audit import application_pages

        self.gold, self.answer = gold or {}, answer or {}
        self.pages = {number: PageText(text) for number, text in pages.items()}
        self.cut_pages = cut_pages
        ordered = [pages.get(number, "") for number in range(1, max(pages, default=0) + 1)]
        self.application = application_pages(ordered)
        self.gold_leaves = _leaves(self.gold)
        self.answer_leaves = _leaves(self.answer)

    def printed(self, value: Any) -> bool | None:
        """Whether a page prints ``value``; None when it is too short to look for or there is no text."""
        from common.grounding import checkable, on_page

        if not checkable(value) or not any(page.has_text for page in self.pages.values()):
            return None
        return any(on_page(value, page) for page in self.pages.values())

    def rows(self, side: dict[str, Any], section: str | None) -> list[dict[str, Any]]:
        rows = side.get(section) if section else None
        return [row for row in rows or [] if isinstance(row, dict)]


def _section(pattern: str) -> str | None:
    match = _SECTION.match(pattern)
    return match.group("section") if match else None


def _holds(row: dict[str, Any], value: Any) -> bool:
    return any(_same(leaf, value) for _p, leaf, _r in _leaves(row))


def _only_row(rows: list[dict[str, Any]], values: list[Any]) -> dict[str, Any] | None:
    """The one row holding any of ``values``, or None when none does or several do: a value many rows share
    ("other", "1,000,000") does not say which row it came from."""
    holding = [row for row in rows if any(_holds(row, v) for v in values)]
    return holding[0] if len(holding) == 1 else None


def _identity(row: dict[str, Any], section: str) -> tuple[Any, Any]:
    if section == FORMS:
        return _value(row.get("form_number")), None
    if section == COVERAGES:
        return _value(row.get("coverage_name")), _value(row.get("coverage_code"))
    for key in ("name", "label", "location_number", "building_number", "vin"):
        if row.get(key) is not None:
            return _value(row.get(key)), None
    return None, None


def _twin(row: dict[str, Any], others: list[dict[str, Any]], section: str) -> dict[str, Any] | None:
    """The row of ``others`` that is ``row`` written another way: the same coverage code, the same name or number
    once case and spacing are gone, or a form number with its edition added or taken away."""
    from scripts.sort_gate_errors import edition_twin

    name, code = _identity(row, section)
    for other in others:
        other_name, other_code = _identity(other, section)
        if code is not None and code == other_code:
            return other
        if name is not None and other_name is not None and _same(name, other_name):
            return other
        if section == FORMS and name is not None and other_name is not None and edition_twin(name, [other_name]):
            return other
    return None


def _cited(row: dict[str, Any]) -> set[int]:
    return {page for _p, _v, refs in _leaves(row) for page in refs}


# -- the sort -----------------------------------------------------------------------------------------------------


def sort_record(record: dict[str, Any], doc: Document | None) -> tuple[str, str]:
    """``(class, why)`` for one failing field."""
    pattern = field_pattern(str(record.get("field_path", "")))
    section = _section(pattern)
    kind = record.get("error_class")
    expected, got = _scalars(record.get("expected")), _scalars(record.get("got"))
    if doc is None:
        return MODEL, "no document context to check"
    if kind == "invented":
        return _invented(pattern, section, got, doc)
    if kind == "omission":
        return _omission(pattern, section, expected, doc)
    return _wrong_value(pattern, section, expected, got, doc)


def _why_not_looked_up(pattern: str, values: list[Any], doc: Document) -> str:
    """Why a page search cannot settle these values, in words: a building number "1" is no code (2026-11-10)."""
    from common.grounding import checkable

    if any(isinstance(value, bool) for value in values):
        return "a yes/no"
    if not all(_lookable(pattern, value) for value in values):
        return "a code or a choice from a list"
    if not any(page.has_text for page in doc.pages.values()):
        return "no page text to look in"
    if not all(checkable(value) for value in values):
        return "a value too short to look for"
    return "not found on a page or proved missing"


def _invented(pattern: str, section: str | None, got: list[Any], doc: Document) -> tuple[str, str]:
    for value in got:
        for gold_pattern, gold_value, _refs in doc.gold_leaves:
            if _same(value, gold_value) and _distinctive(gold_pattern if gold_pattern == pattern else None, value,
                                                          doc.gold_leaves, doc.answer_leaves):
                return ((ALIGNMENT, "the gold holds this value in this field, written another way")
                        if gold_pattern == pattern else
                        (PLACEMENT, f"the gold holds this value in {gold_pattern}"))
    row = _only_row(doc.rows(doc.answer, section), got)
    if row is not None:
        if _twin(row, doc.rows(doc.gold, section), section) is not None:
            return ALIGNMENT, "the gold holds this row under another name, code or edition"
        if section in (FORMS, COVERAGES) and _cited(row) and _cited(row) <= doc.application:
            return RULES, "rule 5: a form or coverage read only from application pages"
        name = _identity(row, section)[0] if section else None
        if section == COVERAGES and name and _LIMIT_WORDS.search(str(name)) and any(
                _holds(limit_holder, v) for limit_holder in doc.rows(doc.gold, COVERAGES)
                for v in _scalars(row.get("limits"))):
            return RULES, "rule 2: a limit the gold keeps inside its coverage, written as a coverage row"
    printed = [doc.printed(v) for v in got if _lookable(pattern, v)]
    if any(p is True for p in printed):
        return GOLD_GAP, "printed in the document, not in the gold"
    if printed and all(p is False for p in printed):
        return MODEL, "wrote a value printed nowhere"
    if row is not None and section:
        name = _identity(row, section)[0]
        if name is not None and doc.printed(name):
            return GOLD_GAP, "in a row the document prints and the gold does not hold"
    return UNCHECKED, f"{_why_not_looked_up(pattern, got, doc)}, in a row that is not a gold gap"


def _omission(pattern: str, section: str | None, expected: list[Any], doc: Document) -> tuple[str, str]:
    for value in expected:
        for answer_pattern, answer_value, _refs in doc.answer_leaves:
            if _same(value, answer_value) and _distinctive(answer_pattern if answer_pattern == pattern else None,
                                                            value, doc.gold_leaves, doc.answer_leaves):
                return ((ALIGNMENT, "the answer holds this value in this field, in a row the scorer did not pair")
                        if answer_pattern == pattern else
                        (PLACEMENT, f"the answer holds this value in {answer_pattern}"))
    row = _only_row(doc.rows(doc.gold, section), expected)
    if row is not None:
        twin = _twin(row, doc.rows(doc.answer, section), section)
        if twin is not None and any(_holds(twin, v) for v in expected):
            return ALIGNMENT, "the answer's row for it holds the value; the scorer paired another"
        if section in (FORMS, COVERAGES) and _cited(row) and _cited(row) <= doc.application:
            return RULES, "rule 5: the gold row is cited only on application pages"
    refs = {page for p, v, pages in doc.gold_leaves if p == pattern and any(_same(v, e) for e in expected)
            for page in pages}
    if refs and refs <= doc.cut_pages:
        return MODEL, "its window was cut off (a loop or the token limit)"
    printed = [doc.printed(v) for v in expected if _lookable(pattern, v)]
    if printed and all(p is False for p in printed):
        return GOLD_NOT_PRINTED, "the gold value is printed on no page"
    if any(p is True for p in printed):
        return MODEL, "missed a printed value"
    return MODEL, f"left out a value that cannot be looked up: {_why_not_looked_up(pattern, expected, doc)}"


def _wrong_value(pattern: str, section: str | None, expected: list[Any], got: list[Any],
                 doc: Document) -> tuple[str, str]:
    from scripts.sort_gate_errors import edition_twin

    if expected and got and all(any(_same(g, e) for e in expected) for g in got):
        return ALIGNMENT, "the same value written another way"
    if pattern.endswith("form_number") and any(edition_twin(g, expected) for g in got):
        return ALIGNMENT, "the same form, its edition inside the number on one side"
    if any(_same(g, value) and _distinctive(None, g, doc.gold_leaves, doc.answer_leaves)
           for g in got for p, value, _r in doc.gold_leaves if p != pattern):
        return MODEL, "put another field's value here"
    gold_printed = [doc.printed(e) for e in expected if _lookable(pattern, e)]
    answer_printed = [doc.printed(g) for g in got if _lookable(pattern, g)]
    if gold_printed and all(p is False for p in gold_printed) and any(p is True for p in answer_printed):
        return GOLD_NOT_PRINTED, "the gold value is printed nowhere; the answer's is printed"
    return MODEL, "wrote another value"


# -- reading the run ----------------------------------------------------------------------------------------------


def documents(answers: list[dict[str, Any]], pages: dict[str, dict[Any, str]]) -> dict[tuple[str, str], Document]:
    """Each (document, reading mode) of a run's rows, merged as the scorer merges them."""
    from evaluation.validation_generation import ValidationGeneration, merged_windows

    generations = [ValidationGeneration(row=a["row"], golden=a.get("gold") or {}, extraction=a.get("answer") or {})
                   for a in answers]
    cut: dict[tuple[str, str], set[int]] = defaultdict(set)
    for a in answers:
        if a.get("error") and (a.get("unparsed_text") or "token limit" in str(a.get("error"))):
            cut[(a["row"].get("source_id"), a["row"].get("modality_mode"))].update(
                int(p) for p in a["row"].get("window_pages") or [])
    out: dict[tuple[str, str], Document] = {}

    def text_of(source: str) -> dict[int, str]:
        return {int(k): v or "" for k, v in (pages.get(source) or {}).items()}

    for merged in merged_windows(generations):
        row = merged.members[0].row
        key = (row.get("source_id"), row.get("modality_mode"))
        out[key] = Document(merged.gold, merged.answer, text_of(key[0]), cut[key])
    for generation in generations:
        key = (generation.row.get("source_id"), generation.row.get("modality_mode"))
        if key not in out:
            out[key] = Document(generation.golden, generation.extraction or {}, text_of(key[0]), cut[key])
    return out


def sort_run(records: list[dict[str, Any]], docs: dict[tuple[str, str], Document]) -> list[dict[str, Any]]:
    out = []
    for record in records:
        doc = docs.get((record.get("source_id"), record.get("modality_mode")))
        cause, why = sort_record(record, doc)
        out.append({**record, "cause": cause, "why": why})
    return out


def summary(sorted_records: list[dict[str, Any]], examples: int = 3) -> dict[str, Any]:
    by_class = Counter(r["cause"] for r in sorted_records)
    by_document: dict[str, Counter] = defaultdict(Counter)
    by_field: dict[str, Counter] = defaultdict(Counter)
    by_why: dict[str, Counter] = defaultdict(Counter)
    shown: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted_records:
        by_document[str(r.get("source_id"))][r["cause"]] += 1
        by_field[field_pattern(str(r.get("field_path", "")))][r["cause"]] += 1
        by_why[r["cause"]][r["why"]] += 1
        if len(shown[r["cause"]]) < examples:
            shown[r["cause"]].append({k: r.get(k) for k in ("source_id", "field_path", "expected", "got", "why")})
    return {"failing_fields": len(sorted_records),
            "by_class": {c: by_class.get(c, 0) for c in CLASSES},
            "by_why": {c: dict(by_why[c].most_common()) for c in CLASSES if by_why[c]},
            "by_document": {d: dict(c) for d, c in sorted(by_document.items())},
            "by_field": {f: dict(c) for f, c in sorted(by_field.items(), key=lambda kv: -sum(kv[1].values()))},
            "examples": {c: shown[c] for c in CLASSES if shown[c]}}


def _cell(value: Any, width: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= width else text[:width - 3] + "..."


def render(report: dict[str, Any], fields: int = 20) -> str:
    total = report["failing_fields"] or 1
    out = [f"{report['failing_fields']} failing field(s), by cause"]
    for cause, count in report["by_class"].items():
        out.append(f"  {cause:<22} {count:>5}  ({100 * count / total:.0f}%)")
        for why, n in (report["by_why"].get(cause) or {}).items():
            out.append(f"      {n:>5}  {why}")
    out.append("\nBy document")
    for document, counts in report["by_document"].items():
        out.append(f"  {document:<14} " + ", ".join(f"{c[:1]} {n}" for c, n in sorted(counts.items())))
    out.append(f"\nBy field (top {fields})")
    for field, counts in list(report["by_field"].items())[:fields]:
        out.append(f"  {field:<48} " + ", ".join(f"{c[:1]} {n}" for c, n in sorted(counts.items())))
    out.append("\nExamples")
    for cause, rows in report["examples"].items():
        out.append(f"  {cause}")
        for r in rows:
            out.append(f"    {r['source_id']}  {r['field_path']}  gold {_cell(r['expected'], 40)}  "
                       f"answer {_cell(r['got'], 40)}  - {r['why']}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--errors", required=True, help="the whole-document report (score_on_training_docs .errors.json)")
    parser.add_argument("--answers", required=True, help="the run's rows (.answers.jsonl)")
    parser.add_argument("--pages", required=True, help="JSON {source_id: {page: text}}")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    records, _metrics = collect(json.loads(Path(args.errors).read_text(encoding="utf-8")))
    answers = [json.loads(line) for line in Path(args.answers).read_text(encoding="utf-8").splitlines() if line.strip()]
    pages = json.loads(Path(args.pages).read_text(encoding="utf-8"))
    report = summary(sort_run(records, documents(answers, pages)))
    print(json.dumps(report, indent=1, ensure_ascii=False, default=str) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
