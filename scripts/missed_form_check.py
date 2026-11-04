"""Could a check catch form numbers an answer left out? A measurement on the golds: nothing is served, nothing changes.

The printed-value check goes from the answer to the page. This asks the other way: which form numbers does a page print
that a list does not hold? Before such a warning is worth having, two things need counting, and the golds give both:

* **seen**: of the forms a gold lists, how many the search finds on the document's pages. A form it cannot see is a
  miss it could never warn about.
* **candidates not in the gold**: what it would warn about if the gold were the answer. Each is either a form the gold
  left out or a false alarm (a number mentioned in passing, a code that only looks like a form). The count alone cannot
  tell which, so they are split by where they sit, and listed for a person to read.

A form attached to a policy prints its number at the head or foot of its own pages; one mentioned in passing does not.
So each candidate is counted as ``edge`` (in the top or bottom lines of a page) or ``body``.

    python -m scripts.missed_form_check --input data/bundles_c1

Only pages with a text layer can be searched here: a document with none is counted and left out.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

#: Lines at the top and bottom of a page counted as its edge.
EDGE_LINES = 4
#: Letters then digits, in groups: ``CG 00 01 04 13``, ``IL00171198``, ``SSI SF 00 03 0123``, ``PGL 00200 0422``.
_CANDIDATE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,8}(?:[ -][A-Z]{1,10}){0,3}(?:[ -]?[A-Z]{0,4}\d[A-Z0-9]*){1,5}(?![A-Za-z0-9])")
#: A state and a ZIP code, which have a form number's shape.
_STATE_ZIP = re.compile(r"^[A-Z]{2} \d{5}(?:-?\d{4})?$")
#: Words a form number does not start with, though the shape allows it: running text in capitals.
_NOT_A_FORM = re.compile(r"^(?:PAGE|PAGES|POLICY|NO|NUMBER|DATE|FORM|ITEM|UNIT|SUITE|STE|BOX|PO|APT|ZIP|AND|THE|OF|TO|FOR|ID)\b")


def compact(value: Any) -> str:
    return re.sub(r"[^0-9a-z]", "", str(value if value is not None else "").casefold())


def candidates(text: str) -> list[str]:
    """What a page's text prints in the shape of a form number: at least four digits, not an address or a label."""
    found = []
    for match in _CANDIDATE.finditer(text or ""):
        value = " ".join(match.group(0).split())
        if sum(c.isdigit() for c in value) >= 4 and not _STATE_ZIP.match(value) and not _NOT_A_FORM.match(value):
            found.append(value)
    return found


def page_candidates(text: str) -> tuple[set[str], set[str]]:
    """``(at the page's edge, in its body)``: a value at the edge is not counted in the body as well."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    edge = {value for line in lines[:EDGE_LINES] + lines[-EDGE_LINES:] for value in candidates(line)}
    body = {value for value in candidates(text) if value not in edge}
    return edge, body


def _in_gold(candidate: str, numbers: list[tuple[str, str]]) -> bool:
    """Whether a candidate is one of the gold's forms: the number alone, or with its edition."""
    value = compact(candidate)
    return any(value in (number, number + edition) for number, edition in numbers)


def check_document(gold: dict[str, Any], pages: dict[Any, str]) -> dict[str, Any]:
    """One document: which of its gold forms the search sees, and which candidates its gold does not hold."""
    from common.canonical import values_view

    view = values_view(gold)
    rows = [r for r in view.get("forms_and_endorsements") or [] if isinstance(r, dict) and r.get("form_number")]
    numbers = [(compact(r["form_number"]), compact(r.get("edition_date"))) for r in rows]
    edge: set[str] = set()
    body: set[str] = set()
    for text in pages.values():
        at_edge, in_body = page_candidates(text)
        edge |= at_edge
        body |= in_body
    body -= edge
    all_found = {compact(value) for value in edge | body}
    seen = sum(any(found in (number, number + edition) for found in all_found) for number, edition in numbers)
    # A value the gold holds elsewhere (the policy number, a phone) is no form the list left out.
    other_values = {compact(value) for value in _scalars(view) if value}
    new_edge = sorted(v for v in edge if not _in_gold(v, numbers) and compact(v) not in other_values)
    new_body = sorted(v for v in body if not _in_gold(v, numbers) and compact(v) not in other_values)
    return {"gold_forms": len(numbers), "seen": seen, "new_edge": new_edge, "new_body": new_body}


def _scalars(node: Any) -> list[Any]:
    if isinstance(node, dict):
        return [v for value in node.values() for v in _scalars(value)]
    if isinstance(node, list):
        return [v for value in node for v in _scalars(value)]
    return [node]


def summarise(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    forms = sum(r["gold_forms"] for r in results.values())
    seen = sum(r["seen"] for r in results.values())
    return {
        "documents": len(results), "gold_forms": forms, "seen": seen,
        "seen_share": round(seen / forms, 4) if forms else None,
        "new_edge": sum(len(r["new_edge"]) for r in results.values()),
        "new_body": sum(len(r["new_body"]) for r in results.values()),
    }


def render(results: dict[str, dict[str, Any]], without_text: list[str], shown: int = 25) -> str:
    total = summarise(results)
    out = [f"{total['documents']} document(s) searched, {len(without_text)} left out (no text layer)", "",
           f"Gold forms the search can see: {total['seen']} of {total['gold_forms']} ({total['seen_share']})",
           f"Candidates not in the gold, at a page's edge: {total['new_edge']}",
           f"Candidates not in the gold, in a page's body: {total['new_body']}", "",
           "By document (gold forms, seen, new at the edge, new in the body)"]
    for name, r in sorted(results.items(), key=lambda item: -len(item[1]["new_edge"])):
        out.append(f"  {name[-44:]:44} {r['gold_forms']:4} {r['seen']:4} {len(r['new_edge']):4} {len(r['new_body']):5}")
    edge = Counter(v for r in results.values() for v in r["new_edge"])
    body = Counter(v for r in results.values() for v in r["new_body"])
    out += ["", f"Most common at the edge: {edge.most_common(shown)}", "", f"Most common in the body: {body.most_common(shown)}"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="a folder of document folders: document.pdf and golden.json")
    parser.add_argument("--json", default=None, help="also write every document's result to this file")
    args = parser.parse_args(argv)

    from data_pipeline.ingestion.label_rules import page_texts

    results: dict[str, dict[str, Any]] = {}
    without_text: list[str] = []
    for folder in sorted(p for p in Path(args.input).iterdir() if (p / "golden.json").is_file()):
        pages = {page: text for page, text in page_texts(folder / "document.pdf").items() if text and text.strip()}
        if not pages:
            without_text.append(folder.name)
            continue
        results[folder.name] = check_document(json.loads((folder / "golden.json").read_text(encoding="utf-8")), pages)
    print(render(results, without_text))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
