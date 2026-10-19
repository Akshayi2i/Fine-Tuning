"""One document into many task examples (arch v2.1 §7b).

v1 had a single task: extract a whole document in one call. That does not survive
a forty-page policy or a dense Loss Run, and the v1 answer — route pages and ask
for a document-level JSON from each one — asked for a shape no training row ever
had.

v2.1 decomposes the work instead, and **trains one adapter on all of it**. The
task is declared in the prompt, so the model is never guessing which output shape
is wanted, and a routed request stays a subsequence of the full-document shape.

    ACORD      classify -> extract (all pages, one call)
    Policy     classify -> page_select -> extract (selected pages, one call)
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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from common.tasks import Task

log = logging.getLogger(__name__)

#: Row density -> (window pages, overlap pages). arch v2.1 §7b.
#: Overlap exists so a row split across a page break is seen whole by at least
#: one window; at density above the last band there is no overlap and the merge
#: rejoins the halves by matching the continued row.
WINDOW_BANDS: tuple[tuple[int, int, int], ...] = (
    (12, 3, 1),      # <= 12 rows/page -> 3-page window, 1-page overlap
    (25, 2, 1),      # 13-25          -> 2-page window, 1-page overlap
)
DENSE_WINDOW = (1, 0)   # > 25 rows/page -> single page, no overlap

#: Pages always added to a policy's routed set — the declarations area carries
#: the policy-level fields whatever the page selector says (§7b).
DECLARATIONS_PAGES = 3

#: OCR markdown charged per routed page when sizing a window. Deliberately thin:
#: this decides how many pages a call is given, and over-estimating the text
#: leaves capacity unused while under-estimating is caught by ``cap_check``,
#: which rejects the row rather than truncating it.
OCR_TOKENS_PER_PAGE = 700

#: Above this, thumbnails are sent in chunks and the selections unioned.
PAGE_SELECT_CHUNK = 60

#: Estimated output tokens per claim row. Used by the planner to shrink a window
#: before it is built. TODO Phase 0 (spike item 4): measure on real Loss Runs.
TOKENS_PER_CLAIM_ROW = 120

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


class ExpansionError(RuntimeError):
    """Raised when a document cannot be expanded into task examples."""


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


def plan_windows(
    densities: list[int],
    *,
    output_budget: int,
    tokens_per_row: int = TOKENS_PER_CLAIM_ROW,
) -> list[list[int]]:
    """Page windows for ``lossrun_rows``, sized from row density.

    ``densities`` is one count per page, in page order. Returns 1-based page
    windows, overlapping per the band table.

    The band gives a starting size and the **output estimate shrinks it**: three
    sparse pages that together hold ninety rows still overflow a budget sized for
    sixty-five, and the page count alone cannot see that.
    """
    if not densities:
        return []
    if output_budget <= 0:
        raise ExpansionError("output_budget must be positive to plan a window")

    windows: list[list[int]] = []
    page = 0
    total_pages = len(densities)

    while page < total_pages:
        density = densities[page]
        size, overlap = DENSE_WINDOW
        for threshold, band_size, band_overlap in WINDOW_BANDS:
            if density <= threshold:
                size, overlap = band_size, band_overlap
                break

        # Shrink until the estimated output fits. A window of one page is the
        # floor: if a single page's rows do not fit the budget, the budget or the
        # schema is wrong, and that surfaces in cap_check rather than here.
        while size > 1:
            estimated = sum(densities[page:page + size]) * tokens_per_row
            if estimated <= output_budget:
                break
            size -= 1

        window = list(range(page + 1, min(page + size, total_pages) + 1))
        if window:
            windows.append(window)
        if window and window[-1] == total_pages:
            # The last page is read: a further window would hold only pages
            # this one already has (the overlap), and read its rows again.
            break

        step = max(1, size - overlap) if size > 1 else 1
        page += step

    return windows


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


def select_policy_pages(
    provenance_pages: list[int],
    total_pages: int,
    *,
    distractors: list[int] | None = None,
) -> list[int]:
    """Every page a policy extraction must read, in page order.

    Pages 1-3 are **always** included: the declarations area carries the
    policy-level fields, and a selector that misses it produces an extraction with
    no policy number.

    Returns the whole set. It is :func:`plan_policy_windows` that decides how many
    calls the set takes — this function no longer truncates it.

    It used to. The cap was six pages, and beyond that it kept the declarations
    plus the numerically *lowest* remaining pages and logged the rest, under a
    docstring claiming the highest-relevance ones were kept. There was never a
    relevance score in it. On a 200-page policy with the vehicle schedule on
    pp.140-146, the locations on pp.150-152 and the endorsements on pp.180-190,
    that kept ``[1, 2, 3, 12, 140, 141]`` and discarded fourteen pages including
    every endorsement. The extraction then validated cleanly against the schema
    and was missing most of the policy, and the only trace was a log line nobody
    reads at corpus-build time.

    A page set larger than one call is a document that needs more than one call.
    It is not a document with fewer pages.
    """
    if total_pages <= 0:
        raise ExpansionError("a policy needs at least one page")

    declarations = list(range(1, min(DECLARATIONS_PAGES, total_pages) + 1))
    selected = set(declarations) | {p for p in provenance_pages if 1 <= p <= total_pages}
    selected |= {p for p in (distractors or []) if 1 <= p <= total_pages}
    return sorted(selected)


def plan_policy_windows(
    pages: list[int], *, pages_per_window: int, leading: Sequence[int] = (), overlap: int = 0,
) -> list[list[int]]:
    """Split a routed page set into calls, dropping nothing.

    ``pages_per_window`` is how many pages one extraction call can carry, which
    :func:`pages_per_extraction_call` derives from the line's own budget — a
    bigger canonical schema leaves room for fewer pages, so ocean marine gets
    four where dwelling fire gets six.

    Two rules, both about keeping a window readable rather than merely legal:

    * **The declarations lead — where the group reads them.** ``leading`` pages
      (the declarations group's: its leading pages, ``declarations_leading``,
      plus the page the declarations were found on) open the windows, together. Every group used to set pages 1-3
      apart, so a routed group spent a call on them alone even when they held
      none of its tables, and a declarations page found on page 5 was read in a
      window apart from the leading pages it continues.
    * **Runs stay whole where they fit.** A vehicle schedule printed across
      pp.140-146 is one table; splitting it at an arbitrary page boundary hands
      the model half a table with no header. Consecutive pages are grouped first
      and a run only breaks when it is longer than one window.

    ``overlap`` pages are shared by consecutive windows of a run that has to be
    split (:func:`_split_overlapping`): a row printed across their boundary - a
    vehicle's VIN at the foot of page 6, its coverages at the top of page 7 - is
    then whole in one window, where no window held all of it before.
    """
    if pages_per_window < 1:
        raise ExpansionError(f"a window must hold at least one page, got {pages_per_window}")
    if not pages:
        return []

    ordered = sorted(set(pages))
    lead = sorted(set(leading) & set(ordered))
    rest = [p for p in ordered if p not in lead]

    windows: list[list[int]] = []
    if lead:
        windows.extend(_split_evenly(lead, pages_per_window))

    # Whole runs are packed together while they fit, so scattered single pages —
    # the usual output of keyword routing — share a call rather than each paying
    # the whole system prompt for one page. A run is never split to fill space
    # and never mixed into another window once it has to be split: a run longer
    # than one window gets windows of its own.
    open_window: list[int] = []
    for run in _consecutive_runs(rest):
        if len(run) > pages_per_window:
            if open_window:
                windows.append(open_window)
                open_window = []
            windows.extend(_split_overlapping(run, pages_per_window, overlap) if overlap
                           else _split_evenly(run, pages_per_window))
        elif len(open_window) + len(run) <= pages_per_window:
            open_window.extend(run)
        else:
            windows.append(open_window)
            open_window = list(run)
    if open_window:
        windows.append(open_window)
    return windows


def _consecutive_runs(pages: list[int]) -> list[list[int]]:
    """``[12, 140, 141, 142, 150]`` -> ``[[12], [140, 141, 142], [150]]``."""
    runs: list[list[int]] = []
    for page in pages:
        if runs and page == runs[-1][-1] + 1:
            runs[-1].append(page)
        else:
            runs.append([page])
    return runs


def _split_evenly(pages: list[int], limit: int) -> list[list[int]]:
    """Split into the fewest windows of at most ``limit``, as evenly as possible.

    Seven pages at a limit of six is two windows, and they are 4+3 rather than
    6+1. Both are legal; the even one is cheaper and reads better. Cheaper
    because a window pays the whole system prompt whatever it holds — some eleven
    thousand tokens for a canonical line — so a one-page window spends a full
    prompt on a single page. Reads better because these runs are usually one
    table printed across several pages, and halving it leaves two comparable
    pieces rather than a body and an orphan.
    """
    windows_needed = -(-len(pages) // limit)
    if windows_needed <= 1:
        return [pages]
    size, remainder = divmod(len(pages), windows_needed)
    out: list[list[int]] = []
    start = 0
    for index in range(windows_needed):
        take = size + (1 if index < remainder else 0)
        out.append(pages[start:start + take])
        start += take
    return out


def _split_overlapping(pages: list[int], limit: int, overlap: int) -> list[list[int]]:
    """Split a run into the fewest windows of at most ``limit``, each after the
    first opening on the last ``overlap`` pages of the one before, as evenly as
    possible. Twelve pages at six with one shared: 5+5+4, sharing pages 5 and 9.
    A window that could hold no page beyond the shared ones splits without them."""
    step = limit - overlap
    if step < 1:
        return _split_evenly(pages, limit)
    windows_needed = 1 + max(0, -(-(len(pages) - limit) // step))
    if windows_needed <= 1:
        return [pages]
    size, remainder = divmod(len(pages) + overlap * (windows_needed - 1), windows_needed)
    out: list[list[int]] = []
    start = 0
    for index in range(windows_needed):
        take = size + (1 if index < remainder else 0)
        out.append(pages[start:start + take])
        start += take - overlap
    return out


def pages_per_extraction_call(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
    *,
    ocr_tokens_per_page: int = OCR_TOKENS_PER_PAGE,
) -> int:
    """How many pages one extraction call has room for, for this line.

    ``sections`` names a slice (``configs/schema_sections.yaml``). A sliced call
    is its group's own task — ``policy_declarations``, ``policy_schedule`` — so
    it is sized against that task's sequence and vision budget and against the
    sliced prompt, not the whole-schema ``extract`` call's. Training and serving
    both plan windows through :mod:`data_pipeline.dataset_builder.policy_windows`,
    which is what keeps the two sizes the same number.

    Derived rather than declared, because the answer moved when the schemas did.
    The prompt carries the whole canonical schema, so a line with a larger one
    leaves less room for pages: measured, dwelling fire fits six and ocean marine
    four. A single ``MAX_ROUTED_PAGES`` constant cannot express that, and the one
    that existed was set when a policy meant twelve flat fields.

    Never returns less than one: a budget that cannot hold a single page is a
    budget problem, and ``cap_check`` is where that surfaces with the arithmetic
    attached rather than here as an empty window list.
    """
    from common.config import sequence_for_task, vision_for_task
    from common.prompts import render_system_prompt
    from data_pipeline.dataset_builder.cap_check import (
        CHARS_PER_TOKEN,
        TEMPLATE_OVERHEAD_TOKENS,
    )

    task = str(Task.EXTRACT)
    if sections:
        from common.schema_sections import task_for

        task = task_for(sections)
    budget = sequence_for_task(task, doc_type)
    prompt = len(
        render_system_prompt(doc_type, "ocr_plus_image", acord_form, lob, sections)
    ) / CHARS_PER_TOKEN
    room = (
        budget["max_seq_len"]
        - budget["max_output_tokens"]
        - TEMPLATE_OVERHEAD_TOKENS
        - prompt
    )
    per_page = vision_for_task(task)["max_pixels"] // 1024 + ocr_tokens_per_page
    return max(1, int(room // per_page))


def page_select_chunks(total_pages: int, chunk: int = PAGE_SELECT_CHUNK) -> list[list[int]]:
    """Thumbnail batches for page selection, unioned by the caller.

    A 200-page policy cannot have every page thumbnailed in one call even at 256
    tokens a page, so selection runs in chunks and the results are unioned.
    """
    return [
        list(range(start + 1, min(start + chunk, total_pages) + 1))
        for start in range(0, max(total_pages, 0), chunk)
    ]


#: Schema fields that are tables of rows rather than policy-level values. Each
#: becomes `policy_schedule` windows; everything else belongs to declarations.
POLICY_SCHEDULE_FIELDS: tuple[str, ...] = ("coverage_schedule", "vehicles", "locations")

#: Where endorsements are recorded on a policy's golden label.
POLICY_ENDORSEMENT_FIELD = "endorsements"


def expand_policy(
    *,
    source_id: str,
    golden_label: dict[str, Any],
    page_markdown: list[str],
    output_budget: int,
    provenance_pages: list[int] | None = None,
    endorsement_pages: list[int] | None = None,
) -> list[TaskExample]:
    """Declarations, schedule windows and endorsements for one policy (§7b).

    The one-call ``extract`` path stays for short policies. This is what a long
    one is read with, because the routed read caps at six pages: a 60-page policy
    extracted that way silently drops everything past the sixth, and a missing
    field looks exactly like a field the document does not have.

    Schedules are windowed for the same reason Loss Run rows are — the binding
    constraint is the OUTPUT budget, not the input. Endorsements are found by
    page tagging rather than by position: they are scattered through a policy
    rather than gathered at one end.
    """
    if not page_markdown:
        raise ExpansionError(f"{source_id}: a policy needs at least one page")

    total_pages = len(page_markdown)
    examples: list[TaskExample] = []

    schedules = {
        name: list(golden_label.get(name) or [])
        for name in POLICY_SCHEDULE_FIELDS
        if golden_label.get(name)
    }
    endorsements = list(golden_label.get(POLICY_ENDORSEMENT_FIELD) or [])

    declared = {
        k: v for k, v in golden_label.items()
        if k not in schedules and k != POLICY_ENDORSEMENT_FIELD
    }
    examples.append(TaskExample(
        task=Task.POLICY_DECLARATIONS,
        source_id=source_id,
        doc_type="policy",
        pages=select_policy_pages(
            provenance_pages=provenance_pages or [], total_pages=total_pages
        ),
        target=declared,
    ))

    # One window sequence per schedule, because a vehicle schedule and a location
    # schedule are different tables on different pages: merging them would ask
    # one call to return two shapes.
    for name, rows in sorted(schedules.items()):
        densities = [detect_row_density(page) for page in page_markdown]
        windows = plan_windows(densities, output_budget=output_budget)
        for index, window in enumerate(windows):
            in_window = _apportion(rows, windows, index)
            if not in_window:
                continue
            examples.append(TaskExample(
                task=Task.POLICY_SCHEDULE,
                source_id=source_id,
                doc_type="policy",
                pages=window,
                target={name: in_window},
                window_index=index,
                metadata={"schedule": name, "window_pages": len(window),
                          "carries_table_header": True},
            ))

    if endorsements:
        pages = [p for p in (endorsement_pages or []) if 1 <= p <= total_pages]
        examples.append(TaskExample(
            task=Task.POLICY_ENDORSEMENTS,
            source_id=source_id,
            doc_type="policy",
            # No tagged pages means the whole document is the search space. That
            # is expensive and honest; guessing a range would drop the
            # endorsements that sit outside it, silently.
            pages=pages or list(range(1, total_pages + 1)),
            target={POLICY_ENDORSEMENT_FIELD: endorsements},
            metadata={"tagged_pages": bool(pages)},
        ))
    return examples
