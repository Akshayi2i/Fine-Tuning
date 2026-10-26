"""A served common-model answer held to the conventions its labels follow
(accuracy plan, stage 4).

The training labels are corrected by ``data_pipeline.ingestion.label_rules``: a
value once, in its field, listing every page that prints it; extra fields only
for values with no field of their own; a unit's number only where it is
printed. A window answer cannot follow the first rule alone - it sees only its
own pages - and the model slips on the others. So the merged answer goes
through the SAME rules, against the document's page text, and what training
taught and what is served are one convention.

What differs from the labels is what happens to a value no page prints. A label
lists it and its windows are left out of training; a served answer flags it for
review (``<path>:not_printed``), and drops it only when it is an extra field of
a born-digital document whose page text settles it - nothing the document
prints is lost that way, and a value invented into the long tail is not served.
A scan's OCR text settles nothing (it misreads), so on a scan every value stays,
as do values on pages whose text lacks the answer's own figures (amounts drawn
in a font that records no characters).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

#: A value the answer holds that no page of the document prints.
NOT_PRINTED_FLAG = "not_printed"
#: A field left empty whose printed label opens a line of a page the model read.
LIKELY_MISSED_FLAG = "likely_missed"
#: Extra fields dropped because no page prints them.
EXTRA_DROPPED_FLAG = "additional_fields:not_printed_dropped"

_EXTRA = re.compile(r"additional_fields\[(\d+)\](.*)")
_MOVED = re.compile(r"(additional_fields\[\d+\]) .* -> (\S+)$")
_EDITION_MOVED = re.compile(r"^(forms_and_endorsements\[\d+\])\.form_number ")


@dataclass
class Conformed:
    """The answer after the rules, its spans re-keyed, and what changed."""

    extraction: dict[str, Any]
    spans: dict[str, Any]
    flags: list[str] = field(default_factory=list)
    #: Every rule's change, as the label rules log it (kind, detail).
    notes: list[tuple[str, str]] = field(default_factory=list)


def conform(
    extraction: dict[str, Any], spans: Mapping[str, Any], page_texts: Mapping[int, str] | None, *,
    native: bool,
) -> Conformed:
    """``extraction`` (merged model form, changed in place) held to the label
    conventions against ``page_texts`` (every page's text, by page number).
    ``native``: the document is born digital, its text a text layer rather than
    OCR of a scan."""
    from data_pipeline.ingestion.label_rules import PRINTED_NOWHERE, _envelopes, apply_label_rules

    before = list(extraction.get("additional_fields") or [])
    original = {id(entry): index for index, entry in enumerate(before)}
    pages = {int(number): text or "" for number, text in (page_texts or {}).items()}
    notes = apply_label_rules(extraction, pages, scanned=() if native else set(pages))

    unprinted = [detail for kind, detail in notes if kind == PRINTED_NOWHERE]
    drop = {int(m.group(1)) for path in unprinted if (m := _EXTRA.match(path))}
    flags = [f"{path}:{NOT_PRINTED_FLAG}" for path in unprinted if not _EXTRA.match(path)]
    if drop:
        extraction["additional_fields"] = [entry for index, entry in enumerate(extraction["additional_fields"])
                                           if index not in drop]
        flags.append(EXTRA_DROPPED_FLAG)
    if "additional_fields" in extraction and not extraction["additional_fields"] and before:
        del extraction["additional_fields"]

    # Spans follow their values: a kept extra field to its new place, a moved
    # one to the field it moved to; a value the rules removed takes its span.
    moved = {f"{m.group(1)}.value": m.group(2) for kind, detail in notes
             if kind == "extra field moved to its field" and (m := _MOVED.match(detail))}
    # An edition moved out of a form number was read in the number's tokens.
    editions = {f"{m.group(1)}.form_number": f"{m.group(1)}.edition_date" for kind, detail in notes
                if kind == "form edition moved" and (m := _EDITION_MOVED.match(detail))}
    now = {original[id(entry)]: index for index, entry in enumerate(extraction.get("additional_fields") or [])
           if id(entry) in original}
    rekeyed: dict[str, Any] = {}
    for key, span in spans.items():
        if key in moved:
            rekeyed[moved[key]] = span
            continue
        if key in editions:
            rekeyed[editions[key]] = span
        m = _EXTRA.match(key)
        if m:
            if int(m.group(1)) not in now:
                continue
            key = f"additional_fields[{now[int(m.group(1))]}]{m.group(2)}"
        rekeyed[key] = span
    present = {path for path, _ in _envelopes(extraction)}
    flags += _recited_off_page(extraction, notes)
    return Conformed(extraction, {k: v for k, v in rekeyed.items() if k in present}, flags, notes)


_RECITED = re.compile(r"^(?P<path>\S+) \[(?P<old>[\d, ]*)\] -> \[(?P<new>[\d, ]*)\]$")


def _recited_off_page(extraction: Mapping[str, Any], notes: Iterable[tuple[str, str]]) -> list[str]:
    """``<path>:printed_off_cited_page`` for each value the page-list rule
    re-cited to pages none of which it cited: printed, but only off the pages
    the model said (``serving.value_flags``). The rule mends the citation, as the
    labels follow it; the doubt it raised - the right figure, maybe read for
    another row or role - still goes to review, as the printed-values check
    sends one it finds itself (``serving.printed_values``). A cited page whose
    text cannot settle it keeps its place in the rule, so is never "off"."""
    from serving.printed_values import _at
    from serving.value_flags import OFF_CITED_PAGE_FLAG

    out: list[str] = []
    for kind, detail in notes:
        found = _RECITED.match(detail) if kind == "page list" else None
        if found is None:
            continue
        old = {int(page) for page in found["old"].split(",") if page.strip()}
        new = sorted(int(page) for page in found["new"].split(",") if page.strip())
        if not old or old & set(new):
            continue
        try:
            envelope = _at(extraction, found["path"])
        except (KeyError, IndexError, TypeError):
            continue                       # moved since (an extra field dropped): not named surely
        if isinstance(envelope, dict) and list(envelope.get("page_ref") or []) == new:
            out.append(f"{found['path']}:{OFF_CITED_PAGE_FLAG}")
    return out


#: The client's field list: each field's meaning and the labels documents print
#: for it ("Printed labels (aliases)"). Read, never edited.
FIELD_LIST = "canonical schema/common schema/common_model.fields.md"

#: A label shorter than this, or of one word ("Phone", "Issued"), opens lines
#: that are not that field's; it raises no flag.
MIN_LABEL_CHARS = 8

_DATE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b[A-Za-z]{3,9}\.? \d{1,2},? \d{4}\b")
_NUMBER = re.compile(r"(?<![\d/])\$?\d[\d,]*(?:\.\d+)?(?![\d/])")
_FIGURES = frozenset({"NumberValue", "MoneyValue", "PercentValue", "YearValue"})


@cache
def field_list() -> dict[str, tuple[str, tuple[str, ...]]]:
    """``block.field`` -> (its value type, the labels documents print for it),
    for every printed value of every block of the client's field list (types
    ending in ``Value``; not an address)."""
    from common.config import CONFIG_DIR

    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    block = None
    for line in (CONFIG_DIR / FIELD_LIST).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            block = line[3:].strip()
            continue
        if not (block and line.startswith("| `")):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        kind = cells[1].split(" ")[0] if len(cells) > 1 else ""
        if len(cells) < 4 or not kind.endswith("Value") or kind == "Address":
            continue
        labels = tuple(label.strip() for label in cells[-1].split(",") if label.strip())
        if labels:
            out[f"{block}.{cells[0].strip('`')}"] = (kind, labels)
    return out


@cache
def _labels_by_first_word() -> dict[str, list[tuple[str, tuple[tuple[str, str], ...], re.Pattern]]]:
    """First word -> ``(label words, ((path, kind), ...), pattern)``, longest
    label first: a line belongs to the longest label it opens with ("Policy Term
    Effective Date" is the effective date's, not the term's)."""
    from common import grounding

    owners: dict[str, list[tuple[str, str]]] = {}
    for path, (kind, labels) in field_list().items():
        for label in labels:
            words = grounding.normalise(label)
            if words and (path, kind) not in owners.setdefault(words, []):
                owners[words].append((path, kind))
    index: dict[str, list[tuple[str, tuple[tuple[str, str], ...], re.Pattern]]] = {}
    for words, fields in owners.items():
        pattern = re.compile(r"^[^0-9A-Za-z]*" + r"[^0-9A-Za-z]+".join(map(re.escape, words.split()))
                             + r"(?![0-9A-Za-z])(?P<rest>.*)$", re.IGNORECASE)
        index.setdefault(words.split()[0], []).append((words, tuple(fields), pattern))
    for entries in index.values():
        entries.sort(key=lambda entry: -len(entry[0]))
    return index


def _stated(kind: str, text: str) -> bool:
    """Whether ``text`` holds a value of ``kind``: a date for a date, a figure
    that is not part of a date for a number or an amount, anything for text."""
    if kind == "DateValue":
        return bool(_DATE.search(text))
    if kind in _FIGURES:
        return bool(_NUMBER.search(_DATE.sub(" ", text)))
    return bool(text.strip(" :.-\t"))


def likely_missed(extraction: Mapping[str, Any], page_texts: Mapping[int, str], pages: Iterable[int],
                  blocks: Iterable[str]) -> list[str]:
    """``<block>.<field>:likely_missed`` for each field of ``blocks`` (the
    declarations objects - one value per document, printed beside its label)
    that the answer left empty while a line of ``pages`` (those the declarations
    were read from) opens with its label and a value of its type follows.

    Only a label of two words or more that the field list gives that field
    alone, and only the longest label a line opens with. A miss is the error
    confidence cannot see - there is no value to score; each flag says where a
    person should look."""
    from common import grounding

    index = _labels_by_first_word()
    wanted = set(blocks)
    lines = [line for page in pages for line in (page_texts.get(page) or "").splitlines() if line.strip()]
    out: list[str] = []
    for number, line in enumerate(lines):
        first = grounding.normalise(line).split(" ", 1)[0]
        match = next(((words, fields, m.group("rest")) for words, fields, pattern in index.get(first, ())
                      if (m := pattern.match(line))), None)
        if match is None:
            continue
        words, fields, rest = match
        if len(fields) != 1 or len(words) < MIN_LABEL_CHARS or " " not in words:
            continue
        (path, kind), = fields
        block, _, name = path.partition(".")
        held = extraction.get(block)
        if isinstance(held, list):
            continue
        value = held.get(name) if isinstance(held, dict) else None
        if block not in wanted or path in out or (isinstance(value, dict) and value.get("raw") not in (None, "")):
            continue
        following = lines[number + 1] if number + 1 < len(lines) else ""
        if _stated(kind, rest) or (not rest.strip(" :.-\t") and _stated(kind, following)):
            out.append(path)
    return [f"{path}:{LIKELY_MISSED_FLAG}" for path in out]


def _first_value(kind: str, text: str) -> str | None:
    """The first value of ``kind`` printed in ``text``: a date for a date, a
    figure outside any date for a number or an amount; for text, all of it."""
    if kind == "DateValue":
        found = _DATE.search(text)
        return found.group() if found else None
    if kind in _FIGURES:
        found = _NUMBER.search(_DATE.sub(" ", text))
        return found.group() if found else None
    stripped = text.strip(" :.-\t|#")
    return stripped or None


def _same_value(kind: str, printed: str, served: Mapping[str, Any]) -> bool | None:
    """Whether ``printed`` (what a label is followed by) is the value ``served``
    holds; None when it cannot be told - a reading neither side gives, a text
    value with no digit (a name continues on the next line as often as not)."""
    from common import grounding
    from common.normalize import normalize_currency, normalize_date

    raw, parsed = served.get("raw"), served.get("parsed")
    if kind == "DateValue":
        on_page = normalize_date(printed)
        held = normalize_date(parsed) or normalize_date(raw)
        return None if on_page is None or held is None else on_page == held
    if kind in _FIGURES:
        on_page = normalize_currency(printed)
        held = (float(parsed) if isinstance(parsed, (int, float)) and not isinstance(parsed, bool)
                else normalize_currency(raw))
        return None if on_page is None or held is None else abs(abs(on_page) - abs(held)) < 0.005
    if not (isinstance(raw, str) and any(ch.isdigit() for ch in raw) and any(ch.isdigit() for ch in printed)):
        return None
    return grounding.ground_tolerant(raw, None, {1: printed}).printed


def role_mismatches(extraction: Mapping[str, Any], page_texts: Mapping[int, str], pages: Iterable[int],
                    blocks: Iterable[str]) -> list[str]:
    """``<block>.<field>:role_mismatch`` for each declarations field (``blocks``)
    the answer holds a value for while every line of ``pages`` that opens with
    the field's label - :func:`likely_missed`'s labels, the same rules - is
    followed by another value.

    The error printed values cannot catch: the value IS printed, as the
    expiration date or the prior policy's number, and served under the
    effective date or the policy number. Every labelled line is heard: one
    printing the served value clears it, so an endorsement's own "Effective
    Date" line beside the declarations' does not flag a right one. Dates and
    figures are compared as what they read (a date as a date, an amount as an
    amount); text only when both hold a digit - an identifier - looked up
    OCR-tolerantly on the label's line. A flag, never a change."""
    from common import grounding
    from serving.value_flags import ROLE_MISMATCH_FLAG

    index = _labels_by_first_word()
    wanted = set(blocks)
    lines = [line for page in pages for line in (page_texts.get(page) or "").splitlines() if line.strip()]
    verdicts: dict[str, list[bool]] = {}
    for number, line in enumerate(lines):
        first = grounding.normalise(line).split(" ", 1)[0]
        match = next(((words, fields, m.group("rest")) for words, fields, pattern in index.get(first, ())
                      if (m := pattern.match(line))), None)
        if match is None:
            continue
        words, fields, rest = match
        if len(fields) != 1 or len(words) < MIN_LABEL_CHARS or " " not in words:
            continue
        (path, kind), = fields
        block, _, name = path.partition(".")
        held = extraction.get(block)
        if block not in wanted or not isinstance(held, dict):
            continue
        value = held.get(name)
        if not isinstance(value, dict) or value.get("raw") in (None, ""):
            continue
        text = rest if rest.strip(" :.-\t|") else (lines[number + 1] if number + 1 < len(lines) else "")
        printed = _first_value(kind, text) if _stated(kind, text) else None
        verdict = _same_value(kind, printed, value) if printed else None
        if verdict is not None:
            verdicts.setdefault(path, []).append(verdict)
    return [f"{path}:{ROLE_MISMATCH_FLAG}" for path, seen in verdicts.items() if seen and not any(seen)]


def declaration_objects(lob: str | list[str] | None) -> list[str]:
    """The objects (not tables) of the group that reads the declarations: the
    blocks :func:`likely_missed` looks in."""
    from common.schema_sections import groups_for, reads_declarations, sections_for
    from common.schemas import resolve_local, resolved_schema

    schema = resolved_schema("policy", None, lob)
    defs, properties = schema.get("$defs") or {}, schema.get("properties") or {}
    return [name for group in groups_for(lob) if reads_declarations(group) for name in sections_for(group, lob)
            if name in properties and resolve_local(properties[name], defs).get("type") == "object"]
