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
from collections.abc import Callable, Sequence
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

#: OCR markdown charged per page when sizing a window. Deliberately thin:
#: this decides how many pages a call is given, and over-estimating the text
#: leaves capacity unused. A page of denser text than this is charged what its
#: own text costs (:func:`fit_windows_to_text`), so a window of policy-form
#: wording is split rather than run past the sequence cap.
OCR_TOKENS_PER_PAGE = 700

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


def plan_policy_windows(
    pages: list[int], *, pages_per_window: int, leading: Sequence[int] = (), overlap: int = 0,
) -> list[list[int]]:
    """Split a document's pages into calls, dropping nothing.

    ``pages_per_window`` is how many pages one extraction call can carry, which
    :func:`pages_per_extraction_call` derives from the line's own budget — a
    bigger canonical schema leaves room for fewer pages, so ocean marine gets
    four where dwelling fire gets six.

    Two rules, both about keeping a window readable rather than merely legal:

    * **The declarations lead — where the group reads them.** ``leading`` pages
      (the declarations group's: its leading pages, ``declarations_leading``,
      plus the page the declarations were found on) open the windows, together. Every group used to set pages 1-3
      apart, so another group spent a call on them alone even when they held
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

    # Whole runs are packed together while they fit, so short runs share a call
    # rather than each paying the whole system prompt for a page or two. A run is never split to fill space
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


def plan_aligned_windows(
    pages: list[int], *, pages_per_window: int, leading: Sequence[int] = (), overlap: int = 0,
    cost: Callable[[int], int] | None = None, room: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """One set of windows for every section group of a document (the fast
    format profile): ``(lead windows, the other windows)``.

    The declarations rule of :func:`plan_policy_windows` - the ``leading``
    pages open the document, together, split evenly where they must be - then
    every other page in runs, split with ``overlap`` shared pages. Each run that
    starts right after a leading page also opens on the last ``overlap`` leading
    pages before it: a group that does not read the declarations used to split
    the document as one run, so a table crossing the page after the leading
    ones was whole in one of its windows, and with every group on these windows
    it stays so.

    ``cost`` and ``room`` fit the windows to the pages' text as
    :func:`fit_windows_to_text` does, the lead windows and the others apart: a
    lead window and the window bridging it share pages, and refitted as one run
    the leading pages would no longer lead. A window holding only pages a lead
    window already reads is dropped - it would read nothing new.
    """
    if pages_per_window < 1:
        raise ExpansionError(f"a window must hold at least one page, got {pages_per_window}")
    ordered = sorted(set(pages))
    lead = sorted(set(leading) & set(ordered))
    rest = [page for page in ordered if page not in lead]
    lead_windows = _split_evenly(lead, pages_per_window) if lead else []
    bridged = set(rest)
    if overlap and pages_per_window > overlap:
        for run in _consecutive_runs(rest):
            for back in range(1, overlap + 1):
                if run[0] - back not in lead:
                    break
                bridged.add(run[0] - back)
    windows = plan_policy_windows(sorted(bridged), pages_per_window=pages_per_window, overlap=overlap)
    if cost is not None and room is not None:
        lead_windows = fit_windows_to_text(lead_windows, cost, room, overlap=overlap)
        windows = fit_windows_to_text(windows, cost, room, overlap=overlap)
    shown = set(lead)
    return lead_windows, [window for window in windows if not set(window) <= shown]


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
    from common.prompts import prompt_text
    from data_pipeline.dataset_builder.cap_check import (
        CHARS_PER_TOKEN,
        TEMPLATE_OVERHEAD_TOKENS,
    )

    task = _call_task(sections)
    budget = sequence_for_task(task, doc_type)
    # The system prompt - and under the fast layout the group block after the
    # pages too: both are in every call.
    prompt = len(
        prompt_text(doc_type, "ocr_plus_image", acord_form, lob, sections)
    ) / CHARS_PER_TOKEN
    room = (
        budget["max_seq_len"]
        - budget["max_output_tokens"]
        - TEMPLATE_OVERHEAD_TOKENS
        - prompt
    )
    per_page = vision_for_task(task)["max_pixels"] // 1024 + ocr_tokens_per_page
    return max(1, int(room // per_page))


def _call_task(sections: str | None) -> str:
    """The task one extraction call is sized against: its group's own, or the
    whole-schema ``extract`` call's."""
    if sections:
        from common.schema_sections import task_for

        return task_for(sections)
    return str(Task.EXTRACT)


def window_page_room(
    doc_type: str,
    acord_form: str | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> int:
    """The tokens one extraction call has for its pages, counted as the corpus
    build's cap check counts a row (``cap_check.estimate_row``): the task's
    sequence cap less its answer reservation, the template and the system prompt.

    The answer is reserved whole, as serving reserves it, so a window that fits
    here fits with any answer the reservation allows.
    """
    from common.config import sequence_for_task
    from common.prompts import prompt_text
    from data_pipeline.dataset_builder.cap_check import TEMPLATE_OVERHEAD_TOKENS, estimate_text_tokens

    budget = sequence_for_task(_call_task(sections), doc_type)
    prompt = estimate_text_tokens(
        prompt_text(doc_type, "ocr_plus_image", acord_form, lob, sections)
    )
    return (int(budget["max_seq_len"]) - int(budget["max_output_tokens"])
            - TEMPLATE_OVERHEAD_TOKENS - prompt)


def page_tokens(text: str | None, page: int, total: int, sections: str | None = None) -> int:
    """One page's cost in an extraction call, counted as the cap check counts it:
    its image at the task's pixel budget, and its text block - the page marker
    and the page's OCR text, as ``inference_core.input_builder`` writes it."""
    from data_pipeline.dataset_builder.cap_check import estimate_text_tokens, estimate_visual_tokens
    from inference_core.input_builder import EMPTY_PAGE_TEXT, PAGE_MARKER

    block = (PAGE_MARKER.format(page=page, total=total) + "\n\n"
             + (text if (text or "").strip() else EMPTY_PAGE_TEXT))
    return estimate_visual_tokens(1, _call_task(sections)) + estimate_text_tokens(block)


def fit_windows_to_text(
    windows: list[list[int]], cost: Callable[[int], int], room: int, *, overlap: int = 0,
) -> list[list[int]]:
    """``windows`` with each one whose pages cost more than ``room`` split into
    the fewest consecutive windows that fit, as evenly as they allow.

    A window's page count is sized for a typical page (:data:`OCR_TOKENS_PER_PAGE`).
    Policy-form wording runs to twice that, and a declarations window over five
    pages of it needed ~34,000 tokens against a 32,768 cap: the corpus build set
    the whole document aside, and serving would have sent a prompt the engine
    refuses. A window that fits is left exactly as planned, so a typical
    document reads as it did. A page that cannot fit even alone keeps a window
    of its own: the cap check names it, with the arithmetic.

    A run split with ``overlap`` shared pages (consecutive windows sharing
    pages) is refitted as one run, each window again opening on the last
    ``overlap`` pages of the one before (:func:`_fit_overlapping`). Fitted window
    by window, a dense personal-auto schedule became twelve calls where seven
    were planned, five page breaks covered by no window.

    With ``overlap``, a run is split with shared pages whatever split it: a
    window its page count kept whole and its text splits shares them too, where
    the cut falls between consecutive pages. Only a run one page longer than a
    window used to: three dense pages read as ``[1, 2], [3]``, four as
    ``[1, 2], [2, 3], [3, 4]``.
    """
    out: list[list[int]] = []
    for chain in _overlap_chains(windows):
        if all(sum(cost(page) for page in window) <= room for window in chain):
            out.extend(chain)
        elif overlap:
            run = sorted({page for window in chain for page in window})
            out.extend(_fit_overlapping(run, cost, room, overlap))
        else:
            for window in chain:
                out.extend(_fit_window(window, cost, room))
    return out


def _overlap_chains(windows: list[list[int]]) -> list[list[list[int]]]:
    """``windows`` grouped into chains of consecutive windows that share pages:
    one split run each, every other window a chain of its own."""
    chains: list[list[list[int]]] = []
    for window in windows:
        if chains and set(chains[-1][-1]) & set(window):
            chains[-1].append(window)
        else:
            chains.append([window])
    return chains


def _fit_overlapping(run: list[int], cost: Callable[[int], int], room: int, overlap: int) -> list[list[int]]:
    """Pages in page order as windows that fit ``room``, each after the first
    opening on the last ``overlap`` pages of the one before where it is cut
    between consecutive pages. A window that could hold no page beyond the
    shared ones starts after them, as :func:`_split_overlapping` does: opened
    on them, it would read only pages already read (``[1, 2], [2], [3]``)."""
    out: list[list[int]] = []
    start = 0
    while start < len(run):
        end, used = start, 0
        while end < len(run) and (end == start or used + cost(run[end]) <= room):
            used += cost(run[end])
            end += 1
        out.append(run[start:end])
        if end >= len(run):
            break
        shared = end - overlap
        opens_on_shared = (
            shared > start
            # Cut inside a run of consecutive pages, not between two runs a
            # window packed together.
            and run[end] - run[shared] == overlap
            and sum(cost(page) for page in run[shared:end]) + cost(run[end]) <= room
        )
        start = shared if opens_on_shared else end
    return out


def _fit_window(window: list[int], cost: Callable[[int], int], room: int) -> list[list[int]]:
    """One window as the fewest consecutive windows that fit, filled in page
    order, then as evenly as the same number allows when every one of the even
    split fits too."""
    if sum(cost(page) for page in window) <= room:
        return [window]
    greedy: list[list[int]] = []
    used = 0
    for page in window:
        if greedy and greedy[-1] and used + cost(page) <= room:
            greedy[-1].append(page)
            used += cost(page)
        else:
            greedy.append([page])
            used = cost(page)
    even = _split_evenly(window, -(-len(window) // len(greedy)))
    fits = len(even) == len(greedy) and all(sum(cost(p) for p in w) <= room for w in even)
    return even if fits else greedy


