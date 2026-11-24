"""How many pages a window holds, and how a policy's pages are cut into windows (review A1: moved from
``data_pipeline.dataset_builder.expand_tasks``, which re-exports them). Training rows and served requests are cut by
the same rule."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from common.tasks import Task

#: OCR markdown charged per page when sizing a window. Deliberately thin:
#: this decides how many pages a call is given, and over-estimating the text
#: leaves capacity unused. A page of denser text than this is charged what its
#: own text costs (:func:`fit_windows_to_text`), so a window of policy-form
#: wording is split rather than run past the sequence cap.
OCR_TOKENS_PER_PAGE = 700

class ExpansionError(RuntimeError):
    """Raised when a document cannot be expanded into task examples."""

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
    from contracts.token_estimates import (
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
    from contracts.token_estimates import TEMPLATE_OVERHEAD_TOKENS, estimate_text_tokens

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
    from contracts.page_text import EMPTY_PAGE_TEXT, PAGE_MARKER
    from contracts.token_estimates import estimate_text_tokens, estimate_visual_tokens

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


#: Row density -> (window pages, overlap pages). arch v2.1 §7b.
#: Overlap exists so a row split across a page break is seen whole by at least
#: one window; at density above the last band there is no overlap and the merge
#: rejoins the halves by matching the continued row.
WINDOW_BANDS: tuple[tuple[int, int, int], ...] = (
    (12, 3, 1),      # <= 12 rows/page -> 3-page window, 1-page overlap
    (25, 2, 1),      # 13-25          -> 2-page window, 1-page overlap
)

DENSE_WINDOW = (1, 0)   # > 25 rows/page -> single page, no overlap

#: Estimated output tokens per claim row. Used by the planner to shrink a window
#: before it is built. TODO Phase 0 (spike item 4): measure on real Loss Runs.
TOKENS_PER_CLAIM_ROW = 120

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
