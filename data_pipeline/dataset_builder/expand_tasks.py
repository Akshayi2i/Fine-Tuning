"""One document into many task examples (arch v2.1 §7b).

v1 had a single task: extract a whole document in one call. That does not survive
a forty-page policy or a dense Loss Run, and the v1 answer — route pages and ask
for a document-level JSON from each one — asked for a shape no training row ever
had.

v2.1 decomposes the work instead, and **trains one adapter on all of it**. The
task is declared in the prompt, so the model is never guessing which output shape
is wanted, and a window stays a subsequence of the full-document shape.

    ACORD      classify -> extract (all pages, one call)
    Policy     classify -> section-group windows over every page (policy_windows)
    Loss Run   classify -> lossrun_header -> lossrun_rows (windows) ->
                           lossrun_totals -> deterministic merge

Every example a document produces shares that document's split, because they all
share its group (§8.2). Getting that wrong would leak a document's header into
train and its rows into test.

**The window planner is output-bound, not input-bound.** A Loss Run page holding
forty claim rows produces far more output than a page holding eight, and the
constraint that binds first is the reserved output budget — not the page images.
So the window shrinks as row density rises, and the planner checks its own
estimate against the budget before committing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from common.tasks import Task
from contracts.window_sizing import (  # noqa: F401 - moved (review A1)  # noqa: F401 - moved (review A1)
    DENSE_WINDOW,
    OCR_TOKENS_PER_PAGE,
    TOKENS_PER_CLAIM_ROW,
    WINDOW_BANDS,
    ExpansionError,
    _call_task,
    _consecutive_runs,
    _fit_overlapping,
    _fit_window,
    _overlap_chains,
    _split_evenly,
    _split_overlapping,
    fit_windows_to_text,
    page_tokens,
    pages_per_extraction_call,
    plan_aligned_windows,
    plan_policy_windows,
    plan_windows,
    window_page_room,
)

log = logging.getLogger(__name__)

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class TaskExample:
    """One training example: a task, its pages, and what it should produce."""

    task: Task
    source_id: str
    doc_type: str
    pages: list[int]                      # 1-based page numbers, in order
    target: dict[str, Any] | list[Any]
    acord_form: str | None = None
    window_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def detect_row_density(page_markdown: str) -> int:
    """Claim rows on one page, from its MinerU markdown.

    Counts pipe-table body rows, discounting the header and its separator. In
    image-only mode there is no markdown, and the caller passes a density
    measured from the rendered page instead — the planner takes a number, not a
    source, precisely so both modes reach it the same way.
    """
    if not page_markdown:
        return 0
    rows = [line for line in page_markdown.splitlines() if _TABLE_ROW.match(line)]
    separators = [r for r in rows if set(r.replace("|", "").strip()) <= {"-", ":", " "}]
    return max(0, len(rows) - len(separators) - 1)


def lossrun_schema_has_totals() -> bool:
    """Whether the Loss Run schema declares printed totals to extract.

    It does not today (no ``totals``, no ``row_type``), so serving's
    reconciliation is always "unverifiable" and a totals example would teach
    the model to answer ``{}`` for pages that print totals.
    """
    from common.schemas import load_schema

    return "totals" in (load_schema("lossrun").get("properties") or {})


def expand_lossrun(
    *,
    source_id: str,
    golden_label: dict[str, Any],
    page_markdown: list[str],
    output_budget: int,
    densities: list[int] | None = None,
) -> list[TaskExample]:
    """Header, row windows and totals for one Loss Run (arch v2.1 §7b).

    Totals come from **two** sources: total and subtotal rows captured inside
    row windows, and the dedicated ``lossrun_totals`` task on the last two pages
    - produced only once the schema declares totals
    (:func:`lossrun_schema_has_totals`). v2.0 looked for printed totals on pages
    1-2, where they are not: they sit at the end of the report and per policy
    period (v2.1 correction).
    """
    if not page_markdown:
        raise ExpansionError(f"{source_id}: a Loss Run needs at least one page")

    measured = densities or [detect_row_density(p) for p in page_markdown]
    total_pages = len(page_markdown)
    claims = list(golden_label.get("claims") or [])
    examples: list[TaskExample] = []

    header_fields = {
        k: v for k, v in golden_label.items()
        if k not in {"claims", "totals"}
    }
    examples.append(TaskExample(
        task=Task.LOSSRUN_HEADER,
        source_id=source_id,
        doc_type="lossrun",
        pages=list(range(1, min(2, total_pages) + 1)),
        target=header_fields,
    ))

    windows = plan_windows(measured, output_budget=output_budget)
    # Rows are apportioned to windows by the page they were found on where
    # provenance records it, and evenly otherwise. Even apportionment is a
    # fallback, not a design: it is right on a uniform report and approximate on
    # a lumpy one, which is why provenance is preferred.
    for index, window in enumerate(windows):
        rows_in_window = [
            claim for claim in claims
            if _claim_page(claim) in window
        ] if any(_claim_page(c) for c in claims) else _apportion(claims, windows, index)

        examples.append(TaskExample(
            task=Task.LOSSRUN_ROWS,
            source_id=source_id,
            doc_type="lossrun",
            pages=window,
            target=rows_in_window,
            window_index=index,
            metadata={
                "window_pages": len(window),
                "row_density": [measured[p - 1] for p in window],
                # The first page's table header travels with every window, or a
                # window starting mid-table has unlabelled columns (§7b).
                "carries_table_header": True,
            },
        ))

    # Only when the schema has totals to extract: otherwise every target is {}
    # and the example teaches the model to ignore printed totals.
    if lossrun_schema_has_totals():
        examples.append(TaskExample(
            task=Task.LOSSRUN_TOTALS,
            source_id=source_id,
            doc_type="lossrun",
            pages=list(range(max(1, total_pages - 1), total_pages + 1)),
            target=golden_label.get("totals") or {},
        ))
    return examples


def _claim_page(claim: Any) -> int | None:
    if not isinstance(claim, dict):
        return None
    page = claim.get("_page") or claim.get("page")
    return int(page) if isinstance(page, (int, str)) and str(page).isdigit() else None


def _apportion(claims: list[Any], windows: list[list[int]], index: int) -> list[Any]:
    """Split rows evenly across windows when no page provenance exists."""
    if not windows:
        return []
    per_window = max(1, len(claims) // len(windows))
    start = index * per_window
    end = start + per_window if index < len(windows) - 1 else len(claims)
    return claims[start:end]


