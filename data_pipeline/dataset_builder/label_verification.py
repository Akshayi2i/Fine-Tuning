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


def verified_label(
    label: dict[str, Any],
    page_texts: Sequence[str] | None,
    report: VerificationReport | None = None,
    *,
    doc_type: str | None = None,
    lob: Any = None,
) -> dict[str, Any]:
    """``label`` with each rule-added value kept only where the page prints it,
    read first under the form-edition rule when ``doc_type`` is given and the
    rule holds for it and ``lob`` (``label_rules.label_rules_apply``).

    A copy when anything changes; ``label`` itself otherwise.
    """
    from common.canonical import is_field_value
    from data_pipeline.audit import MIN_CHECKABLE_CHARS, _appears, _Page, normalise
    from data_pipeline.ingestion.label_rules import under_edition_rule

    label = under_edition_rule(label, doc_type, lob)
    report = report if report is not None else VerificationReport()
    added = ((label.get("fideon:filled") or {}).get("paths")) or []
    if not added:
        return label
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
        found = [number for number, page in enumerate(pages, start=1) if _appears(value, page)]
        cited = {int(p) for p in node.get("page_ref") or []}
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
