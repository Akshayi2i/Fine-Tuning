"""One form written two ways is one form.

A form's number and edition are printed together: ``SSI SF 00 03 0123``. A label can hold them as one number, and an
answer as ``SSI SF 00 03`` with the edition ``01/23``, or the other way round. Form rows pair on the number, so the
two faced nothing: the answer's row counted as invented, in every field, and the label's as missed, in every
reading. The cycle-1 gate lost 3,464 of its 16,353 failing fields this way.

:func:`align_form_editions` writes such a label row the way the answer splits it, on a copy, before anything is
scored. Only when the two hold the same characters and all that moved is an edition: the longer number is the
shorter one followed by digits, and those digits are the other side's edition. Another form, or another edition of
the same form, stays what it was. Each row changed is counted by direction, so a report shows how much of its score
rests on this and which side wrote the edition where.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any

FORMS = "forms_and_endorsements"

#: The label held the edition inside its number and the answer in ``edition_date``, and the other way round.
EDITION_IN_LABEL_NUMBER = "edition_in_label_number"
EDITION_IN_ANSWER_NUMBER = "edition_in_answer_number"

#: A coverage form reference written with its edition on one side only (approved 2026-11-20, labelling rule R2).
FORM_REF_EDITION = "form_ref_edition"

#: An edition inside a number, once separators are gone: ``0123``, ``2021``, ``012023``.
_EDITION_DIGITS = re.compile(r"^\d{2,6}$")


def _compact(value: Any) -> str:
    """Letters and digits alone, without case: how a printed number or edition compares."""
    return re.sub(r"[^0-9a-z]", "", str(value if value is not None else "").casefold())


def _value(cell: Any) -> Any:
    """A cell's value: an envelope's ``parsed``, else its ``raw``; a plain value as it is."""
    from common.canonical import is_field_value, values_view

    return values_view(cell) if is_field_value(cell) else cell


def _same_form(answer_number: Any, answer_edition: Any, label_number: Any, label_edition: Any) -> str | None:
    """Which side wrote the edition into its number, when the two rows are one form; else ``None``."""
    answer, label = _compact(answer_number), _compact(label_number)
    if not answer or not label or answer == label:
        return None
    if answer + _compact(answer_edition) != label + _compact(label_edition):
        return None
    longer, shorter = (label, answer) if len(label) > len(answer) else (answer, label)
    if not longer.startswith(shorter) or not _EDITION_DIGITS.match(longer[len(shorter):]):
        return None
    return EDITION_IN_LABEL_NUMBER if len(label) > len(answer) else EDITION_IN_ANSWER_NUMBER


def _written_as(label_cell: Any, answer_cell: Any, fallback: Any) -> Any:
    """The answer's value, in the label's own shape: an envelope keeps its pages and confidence.

    An edition the label held nowhere has no envelope of its own: it takes the number's (``fallback``), since
    the two are printed together. An empty answer leaves the cell empty.
    """
    from common.canonical import is_field_value

    value = _value(answer_cell)
    if value in (None, ""):
        return None
    template = label_cell if is_field_value(label_cell) else fallback
    if not is_field_value(template):
        return value
    raw = answer_cell.get("raw") if is_field_value(answer_cell) else value
    return {**template, "raw": raw, "parsed": value}


def align_form_editions(expected: Any, got: Any) -> tuple[Any, Counter]:
    """``(expected, what moved)``: ``expected`` with each form row the answer splits another way written the way the
    answer splits it. The caller's document is never changed; with nothing to move it is returned as it is."""
    moved: Counter = Counter()
    if not isinstance(expected, dict) or not isinstance(got, dict):
        return expected, moved
    rows, answers = expected.get(FORMS), got.get(FORMS)
    if not isinstance(rows, list) or not isinstance(answers, list):
        return _align_form_refs(expected, got, moved), moved

    def row_of(row: Any) -> tuple[Any, Any]:
        return (_value(row.get("form_number")), _value(row.get("edition_date"))) if isinstance(row, dict) else (None, None)

    label_keys = [row_of(row) for row in rows]
    free = set(range(len(rows)))
    unpaired = []
    # A number both sides hold alike pairs as it always did, so it is never taken for another row's edition.
    for answer in answers:
        number, _edition = row_of(answer)
        exact = next((i for i in sorted(free) if _compact(number) and _compact(label_keys[i][0]) == _compact(number)),
                     None)
        if exact is None:
            unpaired.append(answer)
        else:
            free.discard(exact)

    out = None
    for answer in unpaired:
        number, edition = row_of(answer)
        for index in sorted(free):
            direction = _same_form(number, edition, *label_keys[index])
            if direction is None:
                continue
            out = out if out is not None else list(rows)
            label = copy.deepcopy(rows[index])
            number_cell = label.get("form_number")
            label["form_number"] = _written_as(number_cell, answer.get("form_number"), number_cell)
            label["edition_date"] = _written_as(label.get("edition_date"), answer.get("edition_date"), number_cell)
            out[index] = label
            free.discard(index)
            moved[direction] += 1
            break
    aligned = {**expected, FORMS: out} if out is not None else expected
    return _align_form_refs(aligned, got, moved), moved


def _ref_text(ref: Any) -> Any:
    return _value(ref) if isinstance(ref, dict) else ref


#: What a form reference's edition is, once separators are gone: a month and a year ("1018", "102018") or a year.
_REF_EDITION = re.compile(r"^(?:(?:0[1-9]|1[0-2])(?:\d{2}|(?:19|20)\d{2})|(?:19|20)\d{2})$")


def _same_ref(label: str, answer: str) -> bool:
    """One form, written with its edition by one side only: ``SP 30 18 10 18`` and ``SP 30 18`` - never ``SP 30 1``,
    whose leftover "81018" is no edition."""
    longer, shorter = (label, answer) if len(label) > len(answer) else (answer, label)
    return bool(shorter) and longer != shorter and longer.startswith(shorter)         and bool(_REF_EDITION.match(longer[len(shorter):]))


def _align_form_refs(expected: dict[str, Any], got: dict[str, Any], moved: Counter) -> dict[str, Any]:
    """Each label ``coverages[].form_refs`` entry the answer writes with or without its edition, written as the answer
    writes it (labelling rule R2: the form number as in the forms list, without the edition; labels made before it
    hold it in some). Only a reference the answer holds nowhere as it is, and only to one the answer holds: another
    form, or another edition, stays what it was."""
    coverages, answers = expected.get("coverages"), got.get("coverages")
    if not isinstance(coverages, list) or not isinstance(answers, list):
        return expected
    held = {_compact(_ref_text(ref)): _ref_text(ref) for row in answers if isinstance(row, dict)
            for ref in row.get("form_refs") or [] if _compact(_ref_text(ref))}
    out = None
    for index, row in enumerate(coverages):
        refs = row.get("form_refs") if isinstance(row, dict) else None
        if not isinstance(refs, list):
            continue
        new_refs = list(refs)
        for position, ref in enumerate(refs):
            key = _compact(_ref_text(ref))
            if not key or key in held:
                continue
            match = next((text for compact, text in held.items() if _same_ref(key, compact)), None)
            if match is None:
                continue
            new_refs[position] = {**ref, "raw": match, "parsed": match} if isinstance(ref, dict) else match
            moved[FORM_REF_EDITION] += 1
        if new_refs != refs:
            out = out if out is not None else list(coverages)
            out[index] = {**row, "form_refs": new_refs}
    return {**expected, "coverages": out} if out is not None else expected
