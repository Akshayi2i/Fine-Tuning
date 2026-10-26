"""Printed values only: a served policy holds no value its pages do not print
(decision 2026-10-24).

The model reads each page's image as well as its text, and can write a value
no page prints - an amount from another policy of the same layout, a policy
number a digit off. Confidence cannot see that (the tokens can be certain), nor
can a person reviewing the JSON without the PDF beside it. So every value the
model wrote - each envelope's ``raw``, extra fields and table rows included -
is looked up on the document's pages, with the search that forgives what OCR
does to a scan (``common.grounding.ground_tolerant``). On scans too: the label
rules never judge a scan's text (``data_pipeline.ingestion.label_rules``), but a
value must be found to be served.

What is looked for is what the value prints: its ``raw``; or - a ``raw`` empty
or a placeholder ("", "-") while ``parsed`` holds a value - the forms ``parsed``
is printed in (``common.grounding.printed_forms``): no ``parsed`` is served
unchecked behind a blank ``raw``.

Where it is looked for, and what each outcome does:

1. the pages it cites that have text, then every other page with text - a
   blank page among those it cites stops nothing;
2. **printed** on some page - kept;
3. **printed nowhere** - on no page with text, and every page it cites has text
   (or it cites none) - in ``strict`` mode, the default, taken out of the JSON:
   its key stays, null; an extra field leaves its list; a table row goes once no
   value in it is still printed and the rows after it move up. Each is flagged
   ``not_printed_removed``. In ``flag`` mode it is kept and flagged
   ``not_printed``. The mode is ``printed_values.mode`` in
   ``configs/inference/vllm_serving.yaml``;
4. **on an unread page** - on no page with text, but a page it cites has none:
   usually an image MinerU could not read, which may well print it, so taking
   it out would lose true fields. Kept, and flagged
   ``<path>:unverified_unread_page`` - never silent - and counted unchecked;
5. **unchecked** - a reading (a yes/no, a ``bool``), a single character, or no
   page given has any text - kept, and counted.

Fields first holds its fields to the pages they cite before the rest are read
(``serving.fields_first``): there 3 means "not found yet", and the value is held
back as pending; 4 applies only where a page it cites read empty.

A row taken out never renumbers the rows left: a structural id names one row
from the merge to the served JSON, and the same row at both of fields first's
phases (``veh_2`` stays ``veh_2`` while ``veh_1`` is held back). A link to the
row taken out goes as the merge drops a link naming no row - a list keeps the
rest, a single link goes - flagged ``<path>:link_to_removed_row``.

A flag names a value by the path it is served at. A value taken out with its
row, or out of a list, has no such path: it is named by the list it left and
its place there in the answer as the model wrote it, marked ``#`` -
``vehicles[#0].vin:not_printed_removed`` is the vin of the answer's first
vehicle, whichever vehicle is served at ``vehicles[0]`` now. No flag names a
served path that holds another value. A value made null keeps its own path.

A value printed, but only on pages it does not cite (every page it cites read),
is kept, flagged ``<path>:printed_off_cited_page`` and sent to review; a value
whose ``parsed`` its ``raw`` clearly does not read as (``serving.raw_parsed``)
is taken out in strict mode and flagged ``<path>:parsed_mismatch``; a row whose
printed values share no page is flagged ``<row>:row_not_colocated``; and each of
a coverage's ``form_refs`` - printed strings, not envelopes - is looked up as a
value is, and one printed nowhere leaves its list in strict mode
(decision 2026-10-26, :func:`_check`).

What was found is the record (``ExtractionResult.printed_check``, read by the
evaluation)::

    {"mode": "strict"|"flag", "checked": int, "unchecked": int,
     "not_printed": [{"path": str, "raw": str, "pages": [int, ...]}],
     "removed": bool, "unread_page": int,
     "parsed_mismatch": int, "off_cited_page": int, "rows_not_colocated": int,
     "form_refs": {"checked": int, "not_printed": [{"path", "raw", "pages"}]}}

The form numbers are counted apart from ``checked`` and ``not_printed``: the
invented rate below measures the envelopes, as it always has.

``checked`` counts the values found printed or printed nowhere, ``not_printed``
lists the latter with the pages they cite (``raw`` is what was looked for, and
``path`` names it as its flag does), ``removed`` is True in strict mode.
``unchecked`` counts 4 and 5, ``unread_page`` the values of 4 among them:
``not_printed / checked`` is the invented rate, so a value the text could not
settle is in neither. With no page text at all - an image-only read - nothing
is checked: the answer is unchanged, the record is None and the flag
``printed_values:no_page_text`` says so. Fields first checks it when MinerU's
text arrives (:func:`hold_result_to_printed`).
"""

from __future__ import annotations

import copy
import re
from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: The modes ``printed_values.mode`` may name.
STRICT = "strict"
FLAG = "flag"
MODES = (STRICT, FLAG)

#: A value no page prints, taken out of the served JSON (strict mode).
REMOVED_FLAG = "not_printed_removed"
#: A value printed on no page with text while a page it cites has none: kept.
UNREAD_PAGE_FLAG = "unverified_unread_page"
#: A link to a row taken out, dropped with it (strict mode).
LINK_REMOVED_FLAG = "link_to_removed_row"
#: A policy whose pages carry no text to check it against.
NO_PAGE_TEXT_FLAG = "printed_values:no_page_text"
#: Serving's flag for an answer that fails the client's schema (``extract``).
_SCHEMA_INVALID_FLAG = "schema:invalid"

#: One step of a path: ``[3]`` or a key.
_STEP = re.compile(r"\[(\d+)\]|([^.\[\]]+)")


def configured_mode() -> str:
    """``printed_values.mode`` from the serving config; strict when it names none.

    A mode it does not know is refused, never read as either: a typo must not
    serve invented values, nor take printed ones out.
    """
    from common.config import SERVING_CONFIG, ConfigError, serving_config

    mode = (serving_config().get("printed_values") or {}).get("mode", STRICT)
    if mode not in MODES:
        raise ConfigError(f"printed_values.mode is {mode!r} in {SERVING_CONFIG.name}; expected one of {MODES}")
    return mode


# --------------------------------------------------------------------------
# What is taken out, by path
# --------------------------------------------------------------------------


@dataclass
class _Edits:
    """What strict mode takes out of one answer, by the paths it had."""

    #: Envelopes made null; their keys stay.
    nulled: set[str] = field(default_factory=set)
    #: List path -> the positions taken out of it, ascending: table rows, extra
    #: fields, values of a list of values.
    dropped: dict[str, list[int]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.nulled or self.dropped)

    def path(self, old: str) -> str | None:
        """Where what was at ``old`` is now; None when it was taken out."""
        if not self:
            return old
        new: list[str] = []
        prefix = ""
        for index, name in _STEP.findall(old):
            if name:
                prefix = f"{prefix}.{name}" if prefix else name
                new.append(f".{name}" if new else name)
            else:
                at = int(index)
                gone = self.dropped.get(prefix, ())
                before = bisect_left(gone, at)
                if before < len(gone) and gone[before] == at:
                    return None
                prefix = f"{prefix}[{at}]"
                new.append(f"[{at - before}]")
            if prefix in self.nulled:
                return None
        return "".join(new)

    def named(self, old: str) -> str:
        """How a flag names what was at ``old``: by the path it is served at -
        a value made null keeps its own - or, taken out with its row or out of
        its list, by the list it left and its place there in the answer as the
        model wrote it, marked ``#`` (``vehicles[#0].vin``). Never by a served
        path that now holds another row's value."""
        if not self.dropped:
            return old
        new: list[str] = []
        prefix = ""
        for step in _STEP.finditer(old):
            index, name = step.groups()
            if name:
                prefix = f"{prefix}.{name}" if prefix else name
                new.append(f".{name}" if new else name)
                continue
            at = int(index)
            gone = self.dropped.get(prefix, ())
            before = bisect_left(gone, at)
            if before < len(gone) and gone[before] == at:
                return f"{''.join(new)}[#{at}]{old[step.end():]}"
            prefix = f"{prefix}[{at}]"
            new.append(f"[{at - before}]")
        return "".join(new)

    def flag(self, flag: str) -> str | None:
        """``flag`` on the path its value now has; None when the value was taken
        out. A flag that names no path (``window_failed:...``), or a value
        already taken out (``[#0]``), is unchanged."""
        head, colon, rest = flag.partition(":")
        if "[#" in head or "[" not in head and head not in self.nulled:
            return flag
        moved = self.path(head)
        return None if moved is None else f"{moved}{colon}{rest}"

    def within(self, table: str) -> _Edits:
        """The edits to one top-level table."""
        return _Edits({path for path in self.nulled if path.startswith(f"{table}[")},
                      {path: at for path, at in self.dropped.items()
                       if path == table or path.startswith(f"{table}[")})


def _rows_of(path: str) -> list[str]:
    """The rows ``path`` lies in, outermost first: ``coverages[1]`` and
    ``coverages[1].limits[0]`` for ``coverages[1].limits[0].amount``. A value of
    a list of values (``vehicles[0].symbols[1]``) is not a row of its own."""
    return [path[:end + 1] for end, ch in enumerate(path) if ch == "]" and end + 1 < len(path)]


def _list_and_position(path: str) -> tuple[str, int]:
    head, _, index = path.rpartition("[")
    return head, int(index[:-1])


def _at(tree: Any, path: str) -> Any:
    node = tree
    for index, name in _STEP.findall(path):
        node = node[int(index)] if index else node[name]
    return node


def _apply(tree: dict[str, Any], edits: _Edits, null: Callable[[dict[str, Any], str], None]) -> dict[str, set[str]]:
    """Make ``edits`` to ``tree`` in place, a null written by ``null(parent,
    key)``. Returns the structural ids of the rows taken out, by table: the
    links to them go too (:func:`_drop_links`).

    The rows left keep their ids. Numbered again, the row after one taken out
    would take its id - and fields first would serve ``veh_1`` for one vehicle
    while another is held back, and for another once it is back."""
    from common.schema_sections import structural_ids

    for path in edits.nulled:
        head, _, key = path.rpartition(".")
        null(_at(tree, head) if head else tree, key)
    ids = structural_ids(None)       # a common-model line's; no line is the fallback, which is one
    taken: dict[str, set[str]] = {}
    # The deepest lists first: their paths still hold the positions of the rows
    # above them, which have not moved yet.
    for head in sorted(edits.dropped, key=lambda path: -len(_STEP.findall(path))):
        items = _at(tree, head)
        gone = set(edits.dropped[head])
        if spec := ids.get(head):
            taken.setdefault(head, set()).update(
                row[spec["field"]] for at, row in enumerate(items)
                if at in gone and isinstance(row, dict) and isinstance(row.get(spec["field"]), str))
        items[:] = [item for at, item in enumerate(items) if at not in gone]
    return taken


def _drop_links(tree: Any, taken: Mapping[str, set[str]]) -> list[str]:
    """Every link in ``tree`` to a row taken out (``taken``: table -> its ids),
    dropped as the merge drops a link naming no row: a list keeps the links
    left and goes when none is, a single link goes. Returns the paths, in
    ``tree`` as it is now, of the fields that lost one."""
    from common.schema_sections import references

    if not any(taken.values()):
        return []
    refs = references(None)
    lost: list[str] = []

    def gone(value: Any, tables: tuple[str, ...]) -> bool:
        return isinstance(value, str) and any(value in taken.get(table, ()) for table in tables)

    def walk(node: Any, path: str) -> None:
        if isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict):
            return
        for name in [name for name in node if name in refs]:
            value, where = node[name], f"{path}.{name}" if path else name
            if isinstance(value, list):
                kept = [item for item in value if not gone(item, refs[name])]
                if len(kept) == len(value):
                    continue
                if kept:
                    node[name] = kept
                else:
                    del node[name]
            elif gone(value, refs[name]):
                del node[name]
            else:
                continue
            lost.append(where)
        for key, value in node.items():
            if key not in refs:
                walk(value, f"{path}.{key}" if path else str(key))

    walk(tree, "")
    return lost


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


@dataclass
class _Checked:
    edits: _Edits
    flags: list[str]
    record: dict[str, Any]


def _prepared(page_texts: Mapping[int, str] | None) -> dict[int, Any] | None:
    """Every page prepared once for the document's lookups; None when no page
    has text."""
    from common.grounding import tolerant_pages

    if not page_texts:
        return None
    pages = tolerant_pages(page_texts)
    return pages if any(page.has_text for page in pages.values()) else None


def _grounded(forms: tuple[Any, ...], cited: Iterable[Any], pages: Mapping[int, Any]) -> Any:
    """``ground_tolerant`` over every page - searched first on the pages the
    value cites that have text, where nearly every printed value is. Only a
    value not found there is looked for on the whole document: searched there
    for every value, a 200-page policy's few hundred values took seconds. The
    verdict is the same, as printed is printed on any page; and a blank page it
    cites is judged on the whole document, never on the cited pages alone.

    ``forms``: what the value prints (``common.grounding.printed_forms``) -
    printed when any of them is."""
    from common.grounding import ground_tolerant

    wanted = {int(page) for page in cited if str(page).lstrip("-").isdigit()}
    on_cited = {page: pages[page] for page in wanted if page in pages and pages[page].has_text}
    if on_cited:
        for form in forms:
            found = ground_tolerant(form, cited, on_cited)
            if found.printed:
                return found
    verdicts = []
    for form in forms:
        found = ground_tolerant(form, cited, pages)
        if found.printed:
            return found
        verdicts.append(found)
    return verdicts[0]


def _cited_all_read(cited: Iterable[Any], pages: Mapping[int, Any]) -> bool:
    """Whether every page ``cited`` names was searched and has text: only then
    does "printed on another page" say the citation is wrong, not that a page
    OCR could not read prints it too."""
    wanted = [int(page) for page in cited if str(page).lstrip("-").isdigit()]
    return bool(wanted) and all(page in pages and pages[page].has_text for page in wanted)


def _printed_strings(extraction: Any) -> Iterable[tuple[str, str, list[int]]]:
    """Every plain string the schema says is printed (``common.model_view.
    PRINTED_STRING_FIELDS``: a coverage's ``form_refs``), as ``(path, string,
    the pages its row's values cite)``. Not envelopes, so the value check never
    saw them: a form number invented into a coverage was served."""
    from common.canonical import is_field_value
    from common.model_view import PRINTED_STRING_FIELDS

    names = {name for fields in PRINTED_STRING_FIELDS.values() for name in fields}

    def walk(node: Any, path: str) -> Iterable[tuple[str, str, list[int]]]:
        if isinstance(node, list):
            for index, item in enumerate(node):
                yield from walk(item, f"{path}[{index}]")
            return
        if not isinstance(node, dict) or is_field_value(node):
            return
        cited = sorted({int(page) for value in node.values() if is_field_value(value)
                        for page in value.get("page_ref") or [] if str(page).isdigit()})
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key in names and isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, str) and item.strip():
                        yield f"{here}[{index}]", item, cited
            elif not str(key).startswith("fideon:"):
                yield from walk(value, here)

    yield from walk(extraction, "")


def _check(extraction: Mapping[str, Any], pages: Mapping[int, Any], mode: str) -> _Checked:
    """Every printed value of ``extraction`` looked up on ``pages``, and what
    ``mode`` makes of those no page prints.

    Besides whether a value is printed, it says (decision 2026-10-26):

    * **printed off its cited pages** - printed, but only on pages it does not
      cite while every page it cites was read: the right figure, maybe from
      another row or another role. Kept, flagged
      ``<path>:printed_off_cited_page`` and sent to review
      (``serving.value_flags``);
    * **parsed against raw** - a ``parsed`` its ``raw`` clearly does not read as
      (``serving.raw_parsed``: "$1,000" holding 10000, a date's day and month
      swapped). Taken out in strict mode, kept in flag mode; flagged
      ``<path>:parsed_mismatch`` either way;
    * **a row of fragments** - a row whose values, each printed, share no page:
      flagged ``<row>:row_not_colocated``, nothing more;
    * **form numbers** - each of a coverage's ``form_refs`` looked up as a value
      is; one printed nowhere leaves its list in strict mode. Counted apart
      (``form_refs``), so the invented rate measures what it always measured.
    """
    from common import grounding
    from data_pipeline.ingestion.label_rules import _SYSTEM, _envelopes, _pages
    from serving.conventions import NOT_PRINTED_FLAG
    from serving.raw_parsed import parsed_mismatch
    from serving.value_flags import OFF_CITED_PAGE_FLAG, PARSED_MISMATCH_FLAG, ROW_NOT_COLOCATED_FLAG

    checked = unchecked = 0
    # Path -> what was looked for, for a value printed nowhere, with its pages.
    unprinted: dict[str, tuple[Any, list[int]]] = {}
    unread: list[str] = []
    off_cited: list[str] = []
    mismatched: list[str] = []
    # Row path -> [its values printed nowhere, its values still served with a printed form].
    rows: dict[str, list[int]] = {}
    # Row path -> (what was looked for, the pages found printing it) of each value found printed.
    found_in: dict[str, list[tuple[tuple[Any, ...], tuple[Any, ...], tuple[int, ...]]]] = {}
    # A table repeats its values ("Included", "$500"): each is looked up once per citation.
    looked_up: dict[tuple[Any, ...], grounding.Grounding] = {}
    for path, envelope in _envelopes(extraction):
        forms = grounding.printed_forms(envelope)
        if not forms or path in _SYSTEM:
            continue
        cited = envelope.get("page_ref") or []
        key = (tuple((type(form).__name__, form) for form in forms), tuple(map(str, cited)))
        if key not in looked_up:
            looked_up[key] = _grounded(forms, cited, pages)
        where = looked_up[key]
        if where.status == grounding.UNCHECKED:
            unchecked += 1
            if where.unread:
                unread.append(path)
        else:
            checked += 1
        missing = where.status == grounding.NOT_PRINTED
        if missing:
            unprinted[path] = (forms[0], _pages(envelope))
        elif parsed_mismatch(path, envelope):
            mismatched.append(path)
        if where.status == grounding.ON_OTHER_PAGE and _cited_all_read(cited, pages):
            off_cited.append(path)
        for row in _rows_of(path):
            rows.setdefault(row, [0, 0])[0 if missing else 1] += 1
            if where.printed:
                found_in.setdefault(row, []).append((key[0], forms, where.pages))

    # Coverage form numbers: plain printed strings, each looked up as a value is.
    form_refs: dict[str, tuple[str, list[int]]] = {}
    form_refs_checked = 0
    for path, item, cited in _printed_strings(extraction):
        key = ((("str", item),), tuple(map(str, cited)))
        if key not in looked_up:
            looked_up[key] = _grounded((item,), cited, pages)
        where = looked_up[key]
        if where.status == grounding.UNCHECKED:
            continue
        form_refs_checked += 1
        if where.status == grounding.NOT_PRINTED:
            form_refs[path] = (item, cited)

    edits = _Edits()
    taken: set[str] = set()
    if mode == STRICT:
        # A row goes once no value in it is still printed: one with a value
        # found, or one the text could not settle, stays with what it has.
        taken = {row for row, (out, kept) in rows.items() if out and not kept}
        for row in taken:
            if not any(up in taken for up in _rows_of(row)):
                head, at = _list_and_position(row)
                edits.dropped.setdefault(head, []).append(at)
        # A value parsed as another than it prints is taken out as one printed
        # nowhere is - its row stays: what it prints is on the page. A form
        # number printed nowhere leaves its list.
        for path in [*unprinted, *mismatched, *form_refs]:
            if any(up in taken for up in _rows_of(path)):
                continue                                    # its row goes
            if path.endswith("]"):
                head, at = _list_and_position(path)         # one of a list of values
                edits.dropped.setdefault(head, []).append(at)
            else:
                edits.nulled.add(path)
        for positions in edits.dropped.values():
            positions.sort()

    # Rows whose values are each printed, on no page they all share. Each was
    # searched first on the pages it cites, so the pages found are confirmed on
    # every page before a row is flagged.
    every_page: dict[tuple[Any, ...], set[int]] = {}

    def printing(forms: tuple[Any, ...], forms_key: tuple[Any, ...]) -> set[int]:
        if forms_key not in every_page:
            every_page[forms_key] = {page for form in forms
                                     for page in grounding.ground_tolerant(form, None, pages).pages}
        return every_page[forms_key]

    scattered: list[str] = []
    for row, values in found_in.items():
        if len(values) < 2 or row in taken:
            continue
        if set.intersection(*(set(found) for _key, _forms, found in values)):
            continue
        if not set.intersection(*(printing(forms, forms_key) for forms_key, forms, _found in values)):
            scattered.append(row)

    record = {"mode": mode, "checked": checked, "unchecked": unchecked,
              "not_printed": [{"path": edits.named(path), "raw": form if isinstance(form, str) else str(form),
                               "pages": cites} for path, (form, cites) in unprinted.items()],
              "removed": mode == STRICT, "unread_page": len(unread),
              "parsed_mismatch": len(mismatched),
              "off_cited_page": len(off_cited),
              "rows_not_colocated": len(scattered),
              "form_refs": {"checked": form_refs_checked,
                            "not_printed": [{"path": edits.named(path), "raw": item, "pages": cites}
                                            for path, (item, cites) in form_refs.items()]}}
    kind = REMOVED_FLAG if mode == STRICT else NOT_PRINTED_FLAG
    flags = [f"{edits.named(path)}:{kind}" for path in [*unprinted, *form_refs]]
    flags += [f"{edits.named(path)}:{UNREAD_PAGE_FLAG}" for path in unread]
    flags += [f"{edits.named(path)}:{PARSED_MISMATCH_FLAG}" for path in mismatched]
    flags += [f"{edits.named(path)}:{OFF_CITED_PAGE_FLAG}" for path in off_cited
              if edits.path(path) is not None and path not in edits.nulled]
    flags += [f"{edits.named(row)}:{ROW_NOT_COLOCATED_FLAG}" for row in scattered]
    return _Checked(edits, flags, record)


# --------------------------------------------------------------------------
# The merged model-form answer (extract)
# --------------------------------------------------------------------------


@dataclass
class Held:
    """A model-form answer after the check, its spans, the flags it raised and
    the record (``ExtractionResult.printed_check``); ``record`` is None when
    nothing could be checked (no page text)."""

    extraction: dict[str, Any]
    spans: dict[str, Any]
    flags: list[str] = field(default_factory=list)
    record: dict[str, Any] | None = None
    _edits: _Edits = field(default_factory=_Edits, init=False, repr=False, compare=False)

    def joined(self, earlier: Iterable[str]) -> list[str]:
        """The flags raised before the check (the merge's, the conventions'),
        each on the path its value now has, then the check's own.

        A flag on a value taken out goes with it: the value carries one flag,
        ``not_printed_removed``, which replaces the conventions' ``not_printed``.
        A flag the check raises again (``not_printed`` in flag mode) is listed once.
        """
        moved = [flag for flag in (self._edits.flag(flag) for flag in earlier) if flag is not None]
        return list(dict.fromkeys([*moved, *self.flags]))


#: ``lob`` not given: the line is read off the answer (:func:`_nullable`).
_UNTOLD: Any = object()


def hold_to_printed(extraction: dict[str, Any], spans: Mapping[str, Any],
                    page_texts: Mapping[int, str] | None, *, lob: Any = _UNTOLD) -> Held:
    """``extraction`` (merged model form, changed in place) held to the pages'
    text (page number -> text; None for an image-only read). ``lob``: the line
    it was read as (None is the fallback), which decides how a value is taken out.

    A value taken out is null where the client's schema allows it, and left out
    where it does not - on a self-contained line, whose values are never null:
    the every-key fill serves either as the same null. Its span goes, and the
    spans of the rows after a row taken out move up with them. The rows left
    keep their ids; a link to a row taken out goes (:func:`_drop_links`).
    """
    mode = configured_mode()
    pages = _prepared(page_texts)
    if pages is None:
        return Held(extraction, dict(spans), [NO_PAGE_TEXT_FLAG])
    checked = _check(extraction, pages, mode)
    edits = checked.edits
    flags = list(checked.flags)
    if edits:
        null = _set_none if _nullable(extraction, lob) else _leave_out
        lost = _drop_links(extraction, _apply(extraction, edits, null))
        flags += [f"{path}:{LINK_REMOVED_FLAG}" for path in lost]
        spans = {moved: span for key, span in spans.items() if (moved := edits.path(key)) is not None}
    held = Held(extraction, dict(spans), flags, checked.record)
    held._edits = edits
    return held


def _set_none(parent: dict[str, Any], key: str) -> None:
    parent[key] = None


def _leave_out(parent: dict[str, Any], key: str) -> None:
    del parent[key]


def _nullable(extraction: Mapping[str, Any], lob: Any = _UNTOLD) -> bool:
    """Whether the answer is on a common-model line, whose served JSON is judged
    with every key filled - so a null value is valid - and whose object sections
    must not be empty (a key left out could empty one). A self-contained line
    allows no null value.

    Told the line, its schema decides (``common.schemas.is_common_model``). Not
    told, it is read off the answer: a common-model answer's rows carry
    structural ids (the merge numbers every one); a self-contained one's never
    do, and its object sections hold keys the common model does not declare
    (``carrier.company_name``). Neither sign: the common model."""
    from common.schema_sections import structural_ids
    from common.schemas import is_common_model, load_schema, row_keys

    if lob is not _UNTOLD:
        return is_common_model("policy", None, lob)

    for table, spec in structural_ids(None).items():
        if any(isinstance(row, dict) and spec["field"] in row for row in extraction.get(table) or []):
            return True
    schema = load_schema("policy", None, None)          # no line: the fallback, on the common model
    defs, sections = schema.get("$defs") or {}, schema.get("properties") or {}
    for name, value in extraction.items():
        declared = row_keys(sections[name], defs) if name in sections and isinstance(value, dict) else None
        if declared and set(value) - set(declared):
            return False
    return True


# --------------------------------------------------------------------------
# A served result (fields first)
# --------------------------------------------------------------------------


def hold_result_to_printed(result: Any, page_texts: Mapping[int, str]) -> None:
    """A served result (fields first) held to the pages' text, in place.

    Its fields were read from the page images alone, and served: this is the
    same rule on the served form. A value taken out becomes the every-key fill's
    null; then the summaries follow the JSON - ``fields`` without it, the rows of
    ``list_fields`` and their completeness, ``overall_confidence`` as ``extract``
    averages it, the schema verdict - and every flag on a value is moved to its
    value's new path, or dropped with the value. A result already checked is
    left as it is.
    """
    if result.printed_check is not None:
        return
    mode = configured_mode()
    pages = _prepared(page_texts)
    if pages is None:
        result.review_flags = sorted({*result.review_flags, NO_PAGE_TEXT_FLAG})
        return
    checked = _check(result.extraction, pages, mode)
    edits = checked.edits
    flags = [flag for flag in result.review_flags if flag != NO_PAGE_TEXT_FLAG]
    if edits:
        flags = _hold_served(result, edits, flags)
    result.review_flags = sorted({*flags, *checked.flags})
    result.printed_check = checked.record
    # Served values the check doubts though it keeps them - printed only off the
    # pages they cite, a parse their print contradicts (flag mode) - go to
    # review, as extract sends them (serving.value_flags).
    from serving.value_flags import route_served

    route_served(result.extraction, result.review_flags)


def _hold_served(result: Any, edits: _Edits, flags: list[str]) -> list[str]:
    """``edits`` made to a served result and its summaries; its flags, moved."""
    from calibration.list_completeness import ROW_MISMATCH_SUFFIX, check_completeness
    from common.canonical import empty_field_value, values_view
    from common.schemas import SchemaError, is_common_model, iter_validation_errors
    from serving.pipeline import UNIT_LINK_FLAG, _every_key, unit_link_flags

    lob = _served_lob(result)
    try:
        common_model = is_common_model(result.doc_type, None, lob)
    except SchemaError:
        common_model = False
    rows_before = {name: len(result.extraction.get(name) or []) for name in result.list_fields}

    def served_null(parent: dict[str, Any], key: str) -> None:
        parent[key] = empty_field_value()

    taken = _apply(result.extraction, edits, served_null)
    lost = _drop_links(result.extraction, taken)
    if lost and common_model:
        # A link list left empty goes, as the merge drops one: filled again, the
        # JSON has every key of its schema once more.
        result.extraction = _every_key(result.extraction, result.doc_type, None, lob)

    flags = [flag for flag in (edits.flag(flag) for flag in flags) if flag is not None]
    flags += [f"{path}:{LINK_REMOVED_FLAG}" for path in lost]
    result.fields = {path: value for path, value in result.fields.items() if edits.path(path) is not None}
    for path in result.fields.keys() & edits.dropped.keys():        # a list of values, shorter
        result.fields[path] = {**result.fields[path], "value": values_view(_at(result.extraction, path))}

    # The rows of list_fields are the answer's values as extract reads them
    # (values_view of the model form): the same edits made to them, and the same
    # links dropped, they are what extract serves for the answer held.
    row_null = _set_none if common_model else _leave_out
    for name, entry in list(result.list_fields.items()):
        within = edits.within(name)
        rows = entry.get("rows")
        if not within and not any(taken.values()):
            continue
        try:
            if not isinstance(rows, list) or len(rows) != rows_before[name]:
                raise LookupError(name)
            rows = copy.deepcopy(rows)
            _apply({name: rows}, within, row_null)
            _drop_links({name: rows}, taken)
        except (LookupError, TypeError):
            # Not the served table row for row (an empty row the verdict dropped):
            # read again off what is served.
            rows = values_view(result.extraction.get(name) or [])
        if not within:
            result.list_fields[name] = {**entry, "rows": rows}      # its links only
            continue
        signal = check_completeness(name, len(rows), stated_count=entry.get("stated_count"),
                                    detected_rows=entry.get("detected_rows"))
        result.list_fields[name] = {"rows": rows, **signal.as_output()}
        flags = [flag for flag in flags if flag != f"{name}{ROW_MISMATCH_SUFFIX}"]
        flags += [f"{name}{ROW_MISMATCH_SUFFIX}"] if signal.flagged else []

    if common_model and edits.dropped:
        # As extract links them, on the answer it serves: a row gone can leave a
        # coverage with no unit, or one unit where there were two.
        flags = [flag for flag in flags if not flag.endswith(f".{UNIT_LINK_FLAG}")]
        flags += unit_link_flags(result.extraction)

    result.overall_confidence = _overall_confidence(result.extraction, flags)
    errors = list(iter_validation_errors(result.extraction, result.doc_type, None, lob))
    if errors:
        # Taking values out is meant to keep the answer valid; said if it did not.
        # A clean check leaves the verdict as it was: taking values out cannot
        # mend what extract refused (a required section no window wrote).
        result.schema_valid = False
        result.validation_errors = errors
        flags.append(_SCHEMA_INVALID_FLAG)
    return flags


def _served_lob(result: Any) -> Any:
    """The line ``extract`` validated the result against: its resolved line
    (``line_of_business.lines``), or none - the fallback - when the base model
    read it against the fallback or no line was resolved."""
    if (result.route_info or {}).get("lob_fallback_used"):
        return None
    lines = result.line_of_business.get("lines") if isinstance(result.line_of_business, dict) else None
    if not lines:
        return None
    return lines[0] if len(lines) == 1 else list(lines)


def _overall_confidence(extraction: Any, flags: Iterable[str]) -> float:
    """The mean confidence of the values the model read, as ``extract`` averages
    the calibrated fields: one score per field (a list of values is one), none
    for a field it had no confidence for (``<path>:no_confidence``), none for
    what the pipeline wrote (system fields, the every-key fill)."""
    from common.canonical import CONFIDENCE_SOURCE, is_field_value

    unscored = {flag.rsplit(":", 1)[0] for flag in flags if flag.endswith(":no_confidence")}
    scores: list[float] = []

    def walk(node: Any, path: str) -> None:
        if is_field_value(node):
            confidence = node.get("confidence")
            if (isinstance(confidence, dict) and confidence.get("source") == CONFIDENCE_SOURCE
                    and path not in unscored):
                scores.append(float(confidence.get("score") or 0.0))
        elif isinstance(node, dict):
            for key, value in node.items():
                if not str(key).startswith("fideon:"):
                    walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            if node and all(is_field_value(item) for item in node):
                walk(node[0], path)                     # one field, every value its score
            else:
                for index, item in enumerate(node):
                    walk(item, f"{path}[{index}]")

    walk(extraction, "")
    return round(sum(scores) / len(scores), 4) if scores else 0.0
