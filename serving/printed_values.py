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

* **printed** on some page - kept;
* **unchecked** - too short to look for, a reading such as a yes/no, or a cited
  page with no text - kept, and counted;
* **printed nowhere** - in ``strict`` mode, the default, taken out of the JSON:
  its key stays, null; an extra field leaves its list; a table row goes once no
  value in it is still printed, the rows after it move up and the structural
  ids are numbered again as the merge numbers them. Each is flagged
  ``<path>:not_printed_removed``. In ``flag`` mode it is kept and flagged
  ``<path>:not_printed``. The mode is ``printed_values.mode`` in
  ``configs/inference/vllm_serving.yaml``.

Paths are the answer's as the model wrote it: a flag and the record name where
a removed value was, before the rows after it moved up.

What was found is the record (``ExtractionResult.printed_check``, read by the
evaluation)::

    {"mode": "strict"|"flag", "checked": int, "unchecked": int,
     "not_printed": [{"path": str, "raw": str, "pages": [int, ...]}],
     "removed": bool}

``checked`` counts the values found printed or printed nowhere, ``not_printed``
lists the latter (before removal) with the pages they cite, ``removed`` is True
in strict mode. With no page text - an image-only read - nothing is checked:
the answer is unchanged, the record is None and the flag
``printed_values:no_page_text`` says so. Fields first checks it when MinerU's
text arrives (:func:`hold_result_to_printed`).
"""

from __future__ import annotations

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

    def flag(self, flag: str) -> str | None:
        """``flag`` on the path its value now has; None when the value was taken
        out. A flag that names no path (``window_failed:...``) is unchanged."""
        head, colon, rest = flag.partition(":")
        if "[" not in head and head not in self.nulled:
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


def _apply(tree: dict[str, Any], edits: _Edits, null: Callable[[dict[str, Any], str], None]) -> bool:
    """Make ``edits`` to ``tree`` in place, a null written by ``null(parent,
    key)``. Whether a row taken out carried a structural id, so the ids must be
    numbered again."""
    from common.schema_sections import structural_ids

    for path in edits.nulled:
        head, _, key = path.rpartition(".")
        null(_at(tree, head) if head else tree, key)
    ids = structural_ids(None)       # a common-model line's; no line is the fallback, which is one
    renumber = False
    # The deepest lists first: their paths still hold the positions of the rows
    # above them, which have not moved yet.
    for head in sorted(edits.dropped, key=lambda path: -len(_STEP.findall(path))):
        items = _at(tree, head)
        gone = set(edits.dropped[head])
        spec = ids.get(head)
        renumber = renumber or bool(spec) and any(
            isinstance(items[at], dict) and spec["field"] in items[at] for at in gone)
        items[:] = [item for at, item in enumerate(items) if at not in gone]
    return renumber


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


def _grounded(raw: Any, cited: Iterable[Any], pages: Mapping[int, Any]) -> Any:
    """``ground_tolerant`` over every page - searched first on the pages the
    value cites, where nearly every printed value is. Only a value not found
    there is looked for on the whole document: searched there for every value,
    a 200-page policy's few hundred values took seconds. The verdict is the
    same, as printed is printed on any page."""
    from common.grounding import ground_tolerant

    wanted = {int(page) for page in cited if str(page).lstrip("-").isdigit()}
    on_cited = {page: pages[page] for page in wanted if page in pages}
    if on_cited:
        found = ground_tolerant(raw, cited, on_cited)
        if found.printed:
            return found
    return ground_tolerant(raw, cited, pages)


def _check(extraction: Mapping[str, Any], pages: Mapping[int, Any], mode: str) -> _Checked:
    """Every printed value of ``extraction`` looked up on ``pages``, and what
    ``mode`` makes of those no page prints."""
    from common import grounding
    from data_pipeline.ingestion.label_rules import _SYSTEM, _envelopes, _pages, _printed
    from serving.conventions import NOT_PRINTED_FLAG

    checked = unchecked = 0
    not_printed: list[dict[str, Any]] = []
    unprinted: list[str] = []
    # Row path -> [its values printed nowhere, its values still served with a printed form].
    rows: dict[str, list[int]] = {}
    # A table repeats its values ("Included", "$500"): each is looked up once per citation.
    looked_up: dict[tuple[Any, ...], grounding.Grounding] = {}
    for path, envelope in _envelopes(extraction):
        raw = _printed(envelope)
        if raw is None or path in _SYSTEM:
            continue
        cited = envelope.get("page_ref") or []
        key = (type(raw).__name__, raw, tuple(map(str, cited)))
        if key not in looked_up:
            looked_up[key] = _grounded(raw, cited, pages)
        status = looked_up[key].status
        if status == grounding.UNCHECKED:
            unchecked += 1
        else:
            checked += 1
        missing = status == grounding.NOT_PRINTED
        if missing:
            unprinted.append(path)
            not_printed.append({"path": path, "raw": raw if isinstance(raw, str) else str(raw),
                                "pages": _pages(envelope)})
        for row in _rows_of(path):
            rows.setdefault(row, [0, 0])[0 if missing else 1] += 1

    record = {"mode": mode, "checked": checked, "unchecked": unchecked, "not_printed": not_printed,
              "removed": mode == STRICT}
    if mode == FLAG:
        return _Checked(_Edits(), [f"{path}:{NOT_PRINTED_FLAG}" for path in unprinted], record)

    edits = _Edits()
    # A row goes once no value in it is still printed: one with a value found,
    # or one too short to look for, stays with what it has.
    taken = {row for row, (out, kept) in rows.items() if out and not kept}
    for row in taken:
        if not any(up in taken for up in _rows_of(row)):
            head, at = _list_and_position(row)
            edits.dropped.setdefault(head, []).append(at)
    for path in unprinted:
        if any(up in taken for up in _rows_of(path)):
            continue                                    # its row goes
        if path.endswith("]"):
            head, at = _list_and_position(path)         # one of a list of values
            edits.dropped.setdefault(head, []).append(at)
        else:
            edits.nulled.add(path)
    for positions in edits.dropped.values():
        positions.sort()
    return _Checked(edits, [f"{path}:{REMOVED_FLAG}" for path in unprinted], record)


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
    spans of the rows after a row taken out move up with them.
    """
    mode = configured_mode()
    pages = _prepared(page_texts)
    if pages is None:
        return Held(extraction, dict(spans), [NO_PAGE_TEXT_FLAG])
    checked = _check(extraction, pages, mode)
    edits = checked.edits
    if edits:
        null = _set_none if _nullable(extraction, lob) else _leave_out
        if _apply(extraction, edits, null):
            from common.structural_ids import renumber_structural_ids

            # As the merge numbers them: in row order, every reference rewritten,
            # one to a row taken out dropped. In place, as there.
            renumber_structural_ids(extraction, None, assign=("coverage_id",), in_place=True)
        spans = {moved: span for key, span in spans.items() if (moved := edits.path(key)) is not None}
    held = Held(extraction, dict(spans), checked.flags, checked.record)
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

    if _apply(result.extraction, edits, served_null) and common_model:
        from common.structural_ids import renumber_structural_ids

        renumber_structural_ids(result.extraction, None, assign=("coverage_id",), in_place=True)
        # Renumbering drops each empty reference the fill wrote (`applies_to: []`):
        # filled again, the JSON has every key of its schema once more.
        result.extraction = _every_key(result.extraction, result.doc_type, None, lob)

    flags = [flag for flag in (edits.flag(flag) for flag in flags) if flag is not None]
    result.fields = {path: value for path, value in result.fields.items() if edits.path(path) is not None}
    for path in result.fields.keys() & edits.dropped.keys():        # a list of values, shorter
        result.fields[path] = {**result.fields[path], "value": values_view(_at(result.extraction, path))}

    row_null = _set_none if common_model else _leave_out
    for name, entry in list(result.list_fields.items()):
        within = edits.within(name)
        if not within:
            continue
        rows = entry.get("rows")
        try:
            if not isinstance(rows, list) or len(rows) != rows_before[name]:
                raise LookupError(name)
            _apply({name: rows}, within, row_null)
        except (LookupError, TypeError):
            # Not the served table row for row (an empty row the verdict dropped):
            # read again off what is served.
            rows = values_view(result.extraction.get(name) or [])
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
