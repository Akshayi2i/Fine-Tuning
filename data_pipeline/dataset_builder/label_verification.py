"""A label value that a rule computed is trained on only if the page prints it.

Most of each synthetic label was not read off the page by anyone: it was added
from the source document's reviewed gold by ``fill_synthetic_labels`` - copied,
date-shifted, or mapped from a replaced value - and recorded under
``fideon:filled``. That is 205,922 values over 1,550 labels, about two thirds of
everything the labels state. ``ocr_check`` compares them with the page only as a
RATE (added fields found about as often as original ones), so a single computed
value the synthetic page does not print still reached the training target, and
the model was taught to write something it could not have read.

:func:`verified_label` checks each added value against the document's own OCR
text, with the audit's matching (``data_pipeline.audit``):

* printed on a page it cites - kept;
* printed on exactly one other page - kept, and it cites that page;
* printed nowhere, or only on several other pages - left out;
* shorter than the audit can check - kept, as the audit treats it.

Values a person or the generator placed (not under ``fideon:filled``) are never
touched: a value the OCR missed may still be on the image, and that is the
model's to read. With no OCR text at all - or none on a page the value cites -
nothing can be verified: the value is kept and counted as unverifiable.

Given the gold's ``doc_type`` and line, the gold is first read under the
form-edition rule where it holds (``label_rules.under_edition_rule``): the gate
reads its frozen golds through this function, and the corpus build loads its
golds under the same rule (``export_golden_labels.load_golden_label``), so a
model is taught and scored on one convention.

**The corpus build lists, never deletes** (``listed=True``, review 2026-10-26).
Deleting a value from a target teaches the window that shows it to skip it -
and the words-only matching above missed values the page prints as figures
("$250,000.00" on a page printing "$250,000"), and a scan's OCR misreads. So
the corpus build checks once per twin, on the CLEAN text (the digital render's,
which a scanned render shares through their one gold), with serving's matcher
(:func:`where_printed`: ``common.grounding.printed_forms`` and
``ground_tolerant``, figure-aware and tolerant of OCR), and applies the verdict
to every render of the twin:

* printed on a page it cites, or citing none and printed somewhere - kept
  (a value citing no page is placed by the build itself);
* printed only on other pages - it cites them: the one page, or every page
  printing a document-level value (the value-once convention);
* printed nowhere, or only on several other pages in a table row (another
  row's figure, as the label rules read one) - listed in
  :attr:`VerificationReport.unprinted`, left in the label: the build adds it to
  the document's ``unprinted_values`` and leaves out the windows holding it.

The default (``listed=False``) is the gate's reading of its frozen golds and is
unchanged: a value the page does not print is not expected of the model.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VerificationReport:
    """What checking the added values against the page changed."""

    checked: int = 0
    kept: int = 0
    #: Found only on one other page; now cites it.
    repaged: list[str] = field(default_factory=list)
    #: Not printed, or only on several other pages: left out of the label.
    dropped: list[str] = field(default_factory=list)
    #: Too short for the audit's matching; kept.
    too_short: int = 0
    #: Could not be checked: the document, or a page the value cites, has no
    #: OCR text. Kept.
    unverifiable: int = 0
    #: ``listed=True``: printed nowhere, or only on several other pages in a
    #: table row. Left in the label, for the build's ``unprinted_values``.
    unprinted: list[str] = field(default_factory=list)


def where_printed(envelope: Any, cited: Any, pages: Any) -> Any:
    """Where a value envelope is printed among ``pages`` (page number ->
    ``common.grounding.TolerantPage``), as serving decides it
    (``serving.printed_values``): each form the value prints
    (``grounding.printed_forms``) looked up with ``grounding.ground_tolerant``,
    the first one printed winning. A ``Grounding``; ``UNCHECKED`` for an
    envelope with nothing to look for."""
    from common import grounding

    forms = grounding.printed_forms(envelope) if isinstance(envelope, dict) else ()
    verdicts = []
    for form in forms:
        found = grounding.ground_tolerant(form, cited, pages)
        if found.printed:
            return found
        verdicts.append(found)
    return verdicts[0] if verdicts else grounding.Grounding(grounding.UNCHECKED)


def verified_label(
    label: dict[str, Any],
    page_texts: Sequence[str] | None,
    report: VerificationReport | None = None,
    *,
    doc_type: str | None = None,
    lob: Any = None,
    listed: bool = False,
) -> dict[str, Any]:
    """``label`` with each rule-added value kept only where the page prints it,
    read first under the form-edition rule when ``doc_type`` is given and the
    rule holds for it and ``lob`` (``label_rules.label_rules_apply``).

    ``listed``: the corpus build's check (module docstring) - serving's
    matcher, and a value not printed listed in ``report.unprinted`` rather
    than removed.

    A copy when anything changes; ``label`` itself otherwise.
    """
    from common.canonical import is_field_value, page_numbers
    from data_pipeline.audit import MIN_CHECKABLE_CHARS, _appears, _Page, normalise
    from data_pipeline.ingestion.label_rules import under_edition_rule

    label = under_edition_rule(label, doc_type, lob)
    report = report if report is not None else VerificationReport()
    added = ((label.get("fideon:filled") or {}).get("paths")) or []
    if not added:
        return label
    if listed:
        return _listed(label, page_texts, report, added)
    pages = [_Page(text or "") for text in (page_texts or [])]
    if not any(page.words for page in pages):
        report.unverifiable += len(added)
        return label

    out: dict[str, Any] | None = None
    removals: list[list[Any]] = []
    for path in added:
        steps = _steps(path)
        node = _at(label, steps)
        if not is_field_value(node):
            continue
        raw = node.get("raw") if node.get("raw") is not None else node.get("parsed")
        if raw is None:
            continue
        report.checked += 1
        value = normalise(str(raw))
        if len(value) < MIN_CHECKABLE_CHARS:
            report.too_short += 1
            continue
        # The raw value, not its normalised copy: normalised, a code's pieces
        # are words, found apart (common.grounding.appears_on).
        found = [number for number, page in enumerate(pages, start=1) if _appears(str(raw), page)]
        cited = set(page_numbers(node.get("page_ref")))
        if cited & set(found) or (not cited and found):
            # No page cited: placement is with_inferred_pages' decision.
            report.kept += 1
            continue
        if any(1 <= page <= len(pages) and not pages[page - 1].words for page in cited):
            # A page it cites has no OCR text (a scan MinerU could not read):
            # the value may well be printed there. Unverifiable, as the audit
            # calls it ("needs OCR") - not "not found".
            report.unverifiable += 1
            continue
        out = out if out is not None else copy.deepcopy(label)
        if len(found) == 1:
            _at(out, steps)["page_ref"] = found
            report.repaged.append(path)
        else:
            removals.append(steps)
            report.dropped.append(path)

    if out is None:
        return label
    # Highest list index first, so removing one row's value never renumbers
    # another path still to be removed.
    for steps in sorted(removals, key=_removal_order, reverse=True):
        parent = _at(out, steps[:-1])
        if isinstance(parent, dict):
            parent.pop(steps[-1], None)
        elif isinstance(parent, list) and isinstance(steps[-1], int) and steps[-1] < len(parent):
            parent.pop(steps[-1])
    return out


def _listed(
    label: dict[str, Any], page_texts: Sequence[str] | None, report: VerificationReport,
    added: Sequence[str],
) -> dict[str, Any]:
    """The corpus build's check (``listed=True``): serving's matcher on the
    clean text; nothing removed, a value no page prints listed instead."""
    from common import grounding
    from common.canonical import is_field_value, page_numbers

    pages = grounding.tolerant_pages(dict(enumerate(page_texts or [], start=1)))
    if not any(page.has_text for page in pages.values()):
        report.unverifiable += len(added)
        return label
    out: dict[str, Any] | None = None
    for path in added:
        steps = _steps(path)
        node = _at(label, steps)
        if not is_field_value(node) or not grounding.printed_forms(node):
            continue
        report.checked += 1
        where = where_printed(node, page_numbers(node.get("page_ref")), pages)
        if where.status == grounding.UNCHECKED:
            # A reading or a single character - or printed on no page with text
            # while a page it cites has none: the text cannot settle it.
            if where.unread:
                report.unverifiable += 1
            else:
                report.too_short += 1
            continue
        if where.status == grounding.ON_CITED_PAGE:
            # Citing no page, placement is the corpus build's (with_placed_pages).
            report.kept += 1
            continue
        if where.status == grounding.ON_OTHER_PAGE and (len(where.pages) == 1 or "[" not in path):
            out = out if out is not None else copy.deepcopy(label)
            _at(out, steps)["page_ref"] = list(where.pages)
            report.repaged.append(path)
            continue
        # Printed nowhere; or only on several other pages, in a table row -
        # where the same figure elsewhere may be another row's. Listed: the
        # windows holding it are left out, and no window is taught to skip it.
        report.unprinted.append(path)
    return out if out is not None else label


def paths_after_removal(paths: Sequence[str], removed: Sequence[str]) -> list[str]:
    """``paths``, recorded against a label, as they read on that label once the
    values at ``removed`` (recorded against the same label) are gone - as
    :func:`verified_label` removes them, its ``report.dropped``.

    Removing a list member renumbers every member after it, so a path recorded
    against the original gold (``unprinted_values``, from prepare_bundles)
    named another member on the verified label, or none: an unprinted
    ``additional_names[2]`` after ``[1]`` was dropped resolved to nothing, and
    that value no page prints was taught; an unprinted ``[1]`` resolved to
    what had been ``[2]``, and held a printed value's window out. Each index
    past a removed member of its own list moves down one per removal; a path at
    or under a removed value is left out, its value being gone.
    """
    gone = [_steps(path) for path in removed]
    out: list[str] = []
    for path in paths:
        steps = _steps(path)
        if any(steps[:len(r)] == r for r in gone):
            continue
        moved = list(steps)
        for r in gone:
            depth = len(r) - 1
            # Compared in the original numbering: every removal was recorded in it.
            if (r and isinstance(r[-1], int) and len(steps) > depth and steps[:depth] == r[:-1]
                    and isinstance(steps[depth], int) and steps[depth] > r[-1]):
                moved[depth] -= 1
        out.append(_path(moved))
    return out


def without_paths(label: dict[str, Any], paths: Sequence[str]) -> dict[str, Any]:
    """A copy of ``label`` without the values at ``paths`` - list members
    included, the highest index first so no removal renumbers another still to
    be made. A path that leads nowhere is skipped."""
    out = copy.deepcopy(label)
    for steps in sorted((_steps(path) for path in paths), key=_removal_order, reverse=True):
        if not steps:
            continue
        parent = _at(out, steps[:-1])
        if isinstance(parent, dict):
            parent.pop(steps[-1], None)
        elif isinstance(parent, list) and isinstance(steps[-1], int) and steps[-1] < len(parent):
            parent.pop(steps[-1])
    return out


def _path(steps: Sequence[Any]) -> str:
    """The path string :func:`_steps` reads: ``coverages[2].limit``."""
    text = ""
    for step in steps:
        text += f"[{step}]" if isinstance(step, int) else (f".{step}" if text else str(step))
    return text


def _steps(path: str) -> list[Any]:
    return [int(index) if index else key
            for key, index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", path)]


def _at(node: Any, steps: Sequence[Any]) -> Any:
    for step in steps:
        try:
            node = node[step]
        except (KeyError, IndexError, TypeError):
            return None
    return node


def _removal_order(steps: Sequence[Any]) -> tuple:
    return tuple((1, step) if isinstance(step, int) else (0, str(step)) for step in steps)
