"""Where each answer that ran to its token limit began to loop, and what it was writing then. Read-only.

On v3's own training documents (2026-11-09) all five answers cut off at the 12,288-token limit were loops: one form
number 166 times, one coverage 111 times, three cycles of extra fields about ten times each - none of them in their
golds. Why the model loops is not known; this lists, for each, the evidence to find out:

* where the loop starts (characters into the answer) and how much was written before it;
* the list it is in (the last list opened before it starts) and how many entries the gold holds there;
* the passage that repeats, or the row name written again and again;
* the text just before the loop starts;
* its extra fields (``additional_fields``): how many label/value pairs, how many differ, and - with ``--pages`` -
  how many of the different values the window's pages print. An answer none of the tests sees as a loop is either
  a real long list (pairs different, values printed) or a loop that changes as it goes (pairs repeated, or values
  printed nowhere) (2026-11-12: two answers ran to the limit with no repeated passage).

Every reading is reported, the halves a cut-off window was read again in included.

    python -m scripts.loop_report --answers run.answers.jsonl --pages pages.json

``--answers`` is ``score_on_training_docs``'s rows file, which keeps the whole text of an answer that did not parse
(``unparsed_text``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

#: A list of entries opened in the answer: ``"coverages": [{`` - not a list of page numbers (``"p": [2]``).
_LIST_OPENED = re.compile(r'"([A-Za-z_]+)":\s*\[\s*\{')
#: One extra field as the model writes it: its label and the value as printed (``r``).
_LABEL_VALUE = re.compile(r'"label":\s*"((?:[^"\\]|\\.)*)",\s*"value":\s*\{\s*"r":\s*"((?:[^"\\]|\\.)*)"')
#: Occurrences of one row name closer than this belong to the same run.
RUN_GAP_CHARS = 2000


def exact_loop_start(text: str, period: int) -> int:
    """Where ``text`` starts repeating its last ``period`` characters, to its end - to within one passage: the end of
    the last real entry can match the end of the repeated ones, so the run reaches a little into it."""
    k = len(text) - period - 1
    while k >= 0 and text[k] == text[k + period]:
        k -= 1
    return k + 1


def row_loop_start(text: str, key: str, value: str) -> int:
    """Where the run of one row name written again and again begins: the first occurrence from which every next one
    follows within :data:`RUN_GAP_CHARS`."""
    pattern = re.compile('"' + re.escape(key) + r'":\s*(?:\{\s*"(?:r|v|raw|parsed)":\s*)?"' + re.escape(value) + '"')
    positions = [m.start() for m in pattern.finditer(text)]
    if not positions:
        return len(text)
    start = positions[-1]
    for earlier, later in zip(reversed(positions[:-1]), reversed(positions[1:]), strict=True):
        if later - earlier > RUN_GAP_CHARS:
            break
        start = earlier
    return start


def list_at(text: str, position: int) -> str | None:
    """The last list of entries opened before ``position``: the list a loop starting there is in."""
    found = [m.group(1) for m in _LIST_OPENED.finditer(text[:position])]
    return found[-1] if found else None


def gold_entries(gold: Any, name: str | None) -> int | None:
    """How many entries the gold holds in its list ``name``, at any depth (summed), or None when it has none."""
    if not name:
        return None
    total, seen = 0, False

    def walk(node: Any) -> None:
        nonlocal total, seen
        if isinstance(node, dict):
            for key, value in node.items():
                if key == name and isinstance(value, list):
                    total += len(value)
                    seen = True
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(gold)
    return total if seen else None


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def extra_fields(text: str, page_text: str | None) -> dict[str, Any]:
    """The answer's extra fields: label/value pairs written, how many differ, the most repeated, and - when the
    window's page text is known - how many of the different values it prints (case and spacing ignored)."""
    from collections import Counter

    pairs = Counter(_LABEL_VALUE.findall(text))
    out: dict[str, Any] = {"pairs": sum(pairs.values()), "different": len(pairs),
                           "most_repeated": [(label[:50], value[:30], n) for (label, value), n in pairs.most_common(3)]}
    if page_text is not None:
        plain = _plain(page_text)
        out["printed"] = sum(1 for _, value in pairs if _plain(value) and _plain(value) in plain)
    return out


def window_text(pages: dict[str, Any] | None, row: dict[str, Any]) -> str | None:
    """The window's pages' text from ``--pages`` ({source_id: {page: text}}), or None without it."""
    if pages is None:
        return None
    document = pages.get(str(row.get("source_id"))) or {}
    return " ".join(str(document.get(str(page)) or "") for page in row.get("window_pages") or [])


def report_one(answer: dict[str, Any], pages: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The loop of one cut-off answer, or None for an answer that parsed."""
    from inference_core.openai_backend import repeating_row, repeating_tail

    text = answer.get("unparsed_text") or ""
    if not text:
        return None
    row = answer.get("row") or {}
    out: dict[str, Any] = {"source_id": row.get("source_id"), "task": row.get("task"),
                           "window_pages": row.get("window_pages"), "chars": len(text),
                           "extra_fields": extra_fields(text, window_text(pages, row))}
    period = repeating_tail(text)
    if period is not None:
        start = exact_loop_start(text, period)
        out.update(kind="exact", period=period, repeats=round((len(text) - start) / period, 1),
                   passage=text[-period:][:200])
    else:
        found = repeating_row(text)
        if found is None:
            out.update(kind="none found", start=None)
            return out
        key, value = found
        start = row_loop_start(text, key, value)
        out.update(kind="row", key=key, value=value[:80])
    name = list_at(text, start)
    out.update(start=start, written_before=start, list=name, gold_entries_in_list=gold_entries(answer.get("gold"), name),
               before=text[max(0, start - 300):start])
    return out


def render(loops: list[dict[str, Any]]) -> str:
    lines = [f"{len(loops)} answer(s) cut off at the token limit"]
    for loop in loops:
        lines.append(f"\n{loop['source_id']} {loop['task']} pages {loop['window_pages']}: {loop['chars']:,} chars")
        extra = loop.get("extra_fields") or {}
        if extra.get("pairs"):
            printed = (f", {extra['printed']} of the different values printed on these pages" if "printed" in extra
                       else "")
            lines.append(f"  extra fields: {extra['pairs']} label/value pairs, {extra['different']} different{printed}")
            lines.append(f"  most repeated: {extra['most_repeated']}")
        if loop.get("kind") == "none found":
            lines.append("  no loop found: the answer may be long, not looping")
            continue
        what = (f"the same {loop['period']}-char passage x{loop['repeats']}" if loop["kind"] == "exact"
                else f"the same {loop['key']} {loop['value']!r} again and again")
        lines.append(f"  loop: {what}, from char {loop['start']:,} (written before it: {loop['written_before']:,})")
        lines.append(f"  in list: {loop['list']} (the gold holds {loop['gold_entries_in_list']} entr(ies) there)")
        if loop["kind"] == "exact":
            lines.append(f"  repeats: {loop['passage']!r}")
        lines.append(f"  just before: ...{loop['before'][-200:]!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True, help="score_on_training_docs' .answers.jsonl")
    parser.add_argument("--pages", default=None, help="JSON {source_id: {page: text}}: count the values the pages print")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from scripts.loop_guard_check import records

    pages = json.loads(Path(args.pages).read_text(encoding="utf-8")) if args.pages else None
    loops = [loop for answer in records(Path(args.answers)) if (loop := report_one(answer, pages)) is not None]
    print(json.dumps(loops, indent=1, ensure_ascii=False) if args.json else render(loops))
    return 0


if __name__ == "__main__":
    sys.exit(main())
