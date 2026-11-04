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
        return expected, moved

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
    return ({**expected, FORMS: out} if out is not None else expected), moved
