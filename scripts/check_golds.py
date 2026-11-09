"""Check every gold before training: what it misses, what it holds that no page prints, and which labelling rules it
breaks. Read-only: it reports; it never edits a gold.

On v3's own training documents (2026-11-09) 77 of the 102 values scored as the model inventing them were printed in the
document: the golds missed them, so the model was marked wrong for reading right, and was trained on answers that left
them out. This puts each gold through the checks that already exist, on a copy, and lists every finding:

* **printed, not in the gold** - a form number printed at the head or foot of a page (where an attached form prints its
  own number) that the gold's forms list does not hold (``scripts.missed_form_check``). Numbers in a page's body are
  listed apart: a number mentioned in passing is not a form attached. Other values (coverages, limits) have no search
  of their own that needs no model: ``check_invented_values --review-csv`` lists those a model run found;
* **in the gold, printed nowhere** - a figure (amount, date, number, identifier) no reliable page prints
  (``label_rules.printed_nowhere``: only pages whose text holds nearly every figure the gold cites on them settle it);
* **breaks a labelling rule** (the 2026-11-04 rules, where ``label_rules_apply`` holds) - each change the rules would
  make to this gold (``label_rules.apply_label_rules`` on a copy: an edition still in its form number, an edition not
  parsed, a value in the wrong field...), limits of one coverage on rows of their own (rule 2,
  ``merge_coverage_limits``), a form number that may still hold its edition (``edition_problems``), a name OCR broke
  inside a word that another gold prints whole (rule 3, ``split_name_repairs``), and a form or coverage row cited only
  on application pages (rule 5, ``audit.rows_only_on_application_pages``).

    python -m scripts.check_golds --input data/bundles_c1 --csv <outside the repo>/gold_check.csv

``--input`` is a folder of document folders, each with ``golden.json``, ``document.pdf`` and ``metadata.json`` (its
``lob``), as ``prepare_bundles`` leaves them. Page text is the PDF's own text layer, read as the label rules read it
(``label_rules.page_texts``; the OCR laid over a scanned page shows where a value is printed, never that one is not).
The CSV holds label values: keep it out of git, as golds and audit files are.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PRINTED_NOT_IN_GOLD = "printed, not in the gold"
PRINTED_NOT_IN_GOLD_BODY = "printed in a page's body, not in the gold (may be a mention)"
NOT_PRINTED = "in the gold, printed nowhere"
BREAKS_RULE = "breaks a labelling rule"
CHECKS = (PRINTED_NOT_IN_GOLD, PRINTED_NOT_IN_GOLD_BODY, NOT_PRINTED, BREAKS_RULE)
COLUMNS = ("document", "check", "rule", "where", "detail")

#: The labelling rules' own names for what they change, in plain words (``label_rules.apply_label_rules``).
RULE_NAMES = {
    "form edition moved": "rule 1: the edition is still inside the form number",
    "edition parsed": "rule 1: the edition date is not written as MM/YY yet",
    "coverage limits merged": "rule 2: limits of one coverage are on rows of their own",
    "page list": "a value lists fewer of the pages that print it",
    "link names the building": "a link names a building another way",
    "number not printed": "a number the pages print another way",
}


def _rule_name(kind: str) -> str:
    if kind in RULE_NAMES:
        return RULE_NAMES[kind]
    return f"additional fields: {kind[len('extra field '):]}" if kind.startswith("extra field ") else kind


def _names(gold: dict[str, Any]) -> list[str]:
    """The coverage names a gold prints, for the broken-name rule across golds."""
    from common.canonical import values_view

    rows = values_view(gold).get("coverages") or []
    return [str(row["coverage_name"]) for row in rows if isinstance(row, dict) and row.get("coverage_name")]


def check_gold(gold: dict[str, Any], pages: dict[int, str], *, lob: Any, scanned: set[int] = frozenset(),
               name_repairs: dict[str, str] | None = None) -> list[dict[str, str]]:
    """Every finding for one gold, as ``{check, rule, where, detail}``. ``gold`` is not changed."""
    from data_pipeline.audit import application_pages, rows_only_on_application_pages
    from data_pipeline.ingestion.label_rules import (
        PRINTED_NOWHERE,
        apply_label_rules,
        edition_problems,
        label_rules_apply,
        merge_coverage_limits,
    )
    from scripts.missed_form_check import check_document

    findings: list[dict[str, str]] = []

    def add(check: str, rule: str, where: str, detail: str = "") -> None:
        findings.append({"check": check, "rule": rule, "where": where, "detail": detail})

    texts = {page: text for page, text in pages.items() if text and text.strip()}
    if texts:
        forms = check_document(gold, texts)
        for number in forms["new_edge"]:
            add(PRINTED_NOT_IN_GOLD, "", "forms_and_endorsements", number)
        for number in forms["new_body"]:
            add(PRINTED_NOT_IN_GOLD_BODY, "", "forms_and_endorsements", number)
    if not label_rules_apply("policy", lob):
        return findings
    from data_pipeline.audit import iter_envelopes

    printed_as = {path: envelope.get("raw") if envelope.get("raw") is not None else envelope.get("parsed")
                  for path, envelope in iter_envelopes(gold)}
    trial = copy.deepcopy(gold)
    for kind, detail in apply_label_rules(trial, texts, scanned=scanned):
        if kind == PRINTED_NOWHERE:
            add(NOT_PRINTED, "", str(detail), f"gold value {printed_as.get(str(detail))!r}")
        else:
            add(BREAKS_RULE, _rule_name(str(kind)), "", str(detail))
    for kind, detail in merge_coverage_limits(copy.deepcopy(gold)):
        add(BREAKS_RULE, _rule_name(str(kind)), "coverages", str(detail))
    # The editions the rules would move are reported above, one by one; only the numbers they cannot settle here.
    _errors, warnings = edition_problems(gold, "policy", lob)
    for message in warnings:
        add(BREAKS_RULE, "rule 1: a form number may hold its edition (check by hand)", "forms_and_endorsements",
            message)
    for name in _names(gold):
        if name in (name_repairs or {}):
            add(BREAKS_RULE, "rule 3: a coverage name is broken inside a word", "coverages",
                f"{name!r} -> {name_repairs[name]!r}")
    ordered = [pages.get(page, "") for page in range(1, max(pages, default=0) + 1)]
    for path, refs in rows_only_on_application_pages(gold, application_pages(ordered)):
        add(BREAKS_RULE, "rule 5: a form or coverage is read only from application pages", path, f"pages {refs}")
    return findings


def check_folder(root: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    """Every gold under ``root``: ``({document: findings}, documents with no text layer)``."""
    from data_pipeline.ingestion.label_rules import page_texts, scanned_pages, split_name_repairs

    folders = sorted(p for p in root.iterdir() if (p / "golden.json").is_file())
    golds = {folder.name: json.loads((folder / "golden.json").read_text(encoding="utf-8")) for folder in folders}
    repairs = split_name_repairs(name for gold in golds.values() for name in _names(gold))
    out: dict[str, list[dict[str, str]]] = {}
    without_text: list[str] = []
    for folder in folders:
        metadata_file = folder / "metadata.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.is_file() else {}
        pages = page_texts(folder / "document.pdf")
        if not any((text or "").strip() for text in pages.values()):
            without_text.append(folder.name)
        out[folder.name] = check_gold(golds[folder.name], pages, lob=metadata.get("lob"),
                                      scanned=scanned_pages(folder / "document.pdf"), name_repairs=repairs)
    return out, without_text


def summarise(results: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    by_check = Counter(f["check"] for findings in results.values() for f in findings)
    by_rule = Counter(f["rule"] for findings in results.values() for f in findings if f["check"] == BREAKS_RULE)
    by_document = {document: dict(Counter(f["check"] for f in findings)) for document, findings in results.items()
                   if findings}
    return {"golds": len(results), "golds_with_findings": len(by_document),
            "by_check": {check: by_check.get(check, 0) for check in CHECKS},
            "rules_broken": dict(by_rule.most_common()), "by_document": by_document}


def render(summary: dict[str, Any], results: dict[str, list[dict[str, str]]], without_text: list[str],
           shown: int = 20) -> str:
    out = [f"{summary['golds']} gold(s) checked; {summary['golds_with_findings']} with something to look at"]
    for check, count in summary["by_check"].items():
        out.append(f"  {check:<62} {count:>6}")
    if summary["rules_broken"]:
        out.append("\nRules broken")
        for rule, count in summary["rules_broken"].items():
            out.append(f"  {count:>6}  {rule}")
    out.append(f"\nBy document (most first, {shown} shown)")
    ranked = sorted(summary["by_document"].items(), key=lambda kv: -sum(kv[1].values()))
    for document, counts in ranked[:shown]:
        out.append(f"  {document:<70} " + ", ".join(f"{n} {check.split(',')[0]}" for check, n in counts.items()))
    examples: dict[str, list[str]] = defaultdict(list)
    for document, findings in results.items():
        for f in findings:
            key = f["rule"] or f["check"]
            if len(examples[key]) < 3:
                examples[key].append(f"{document}: {f['where']} {f['detail']}".strip())
    out.append("\nExamples")
    for key, lines in examples.items():
        out.append(f"  {key}")
        out.extend(f"    {line[:160]}" for line in lines)
    if without_text:
        out.append(f"\n{len(without_text)} document(s) with no text layer: only the rules that need no text were "
                   f"checked ({', '.join(without_text[:5])}{'...' if len(without_text) > 5 else ''})")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="a folder of document folders (golden.json, document.pdf)")
    parser.add_argument("--csv", default=None, help="also write every finding here (keep it out of git)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    # About 1.9 s a gold on a laptop (84 cycle-1 golds in 2 min 39 s, 2026-11-10): a 1,000-gold cycle is half an hour,
    # so on the pod it runs in tmux like every long job (orchestration.detach).
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("scripts.check_golds", argv):
        return 0
    results, without_text = check_folder(Path(args.input))
    summary = summarise(results)
    print(json.dumps(summary, indent=1, ensure_ascii=False) if args.json else render(summary, results, without_text))
    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for document, findings in sorted(results.items()):
                writer.writerows({"document": document, **f} for f in findings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
