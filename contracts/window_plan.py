"""A policy's window plan: every page, grouped by section, the windows each group reads (review A1: moved
from ``data_pipeline.dataset_builder.policy_windows``, which re-exports it). Training rows and served requests
are planned by this one function, and ``window_plan_id`` pins a corpus, a release and serving to the same plan."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class PolicyWindowPlan:
    """One extraction call: a section group over a set of pages."""

    group: str
    window_index: int
    pages: tuple[int, ...]
    #: The group reads its pages in one window.
    single: bool
    #: The window of a group its declarations pages lead (``decl``, and ``dtd``
    #: where the line carries it): the one holding the declarations page when
    #: those pages are split over several windows, else the first.
    leads: bool = False
    #: The group has a window that takes a value with no recorded page - its
    #: only one, or the one its declarations lead - so such a value is never
    #: reported unplaced from another of its windows.
    homed: bool = False

    @property
    def holds_unpaged(self) -> bool:
        """Whether a value with no recorded page is taught here: in a group's only
        window, or in the window its declarations pages lead.

        A value the label rules could not place on a page - a scan's, mostly - is
        a policy-level value: the declarations print it. Before every page was
        read, the declarations group was one window and taught it there; read
        over several windows, it was left out of all of them, and the window that
        shows the declarations was taught to leave out what it shows.
        """
        return self.single or self.leads

    @property
    def task(self) -> str:
        from common.schema_sections import task_for

        return task_for(self.group)

def pages_to_read(
    page_texts: Mapping[int, str] | Sequence[str] | None,
    page_count: int,
) -> tuple[list[int], int | None]:
    """``(every page, detected declarations page)`` — one rule for both sides.

    Every page, whatever the length (decision 2026-10-19). The declarations page
    is found in the OCR text; with none (``image_only``) there is none to find.
    """
    from contracts.page_plan import plan_pages

    pages = list(range(1, page_count + 1))
    if not page_texts:
        return pages, None
    texts = (
        dict(page_texts) if isinstance(page_texts, Mapping)
        else {index: text for index, text in enumerate(page_texts, start=1)}
    )
    return pages, plan_pages(texts).declarations_page

#: Bumped whenever the windows a policy is read with change for the same input,
#: and pinned in the corpus manifest: an adapter is served only the window plan
#: its corpus was built with (``assert_window_plan``). "2": every page read, each
#: window sized by its pages' own text (decisions 2026-10-19 and 2026-10-22).
#: "3": a split keeps its runs' shared pages, and the declarations-led window is
#: the one holding the declarations page. "4": a run split by its text shares
#: its pages whatever split it - a window its page count kept whole too - and a
#: window that could hold no page beyond the shared ones starts after them
#: (``expand_tasks.fit_windows_to_text``). Bump it whenever the rule changes;
#: changes to the planner's inputs are caught by :func:`window_plan_id`.
WINDOW_PLAN_VERSION = "4"

def window_plan_id() -> str:
    """The window plan this code reads policies with: :data:`WINDOW_PLAN_VERSION`
    and a fingerprint of every input the planner sizes windows from.

    The version alone missed them: raising ``policy_declarations``'
    ``max_output_tokens`` from 4,096 to 6,144 turned four declarations windows
    into seven with the version and the prompt hash unchanged. So the
    fingerprint covers the section map (groups, page rules, leading pages,
    overlap), each policy task's sequence budget and visual tokens per page,
    the estimates a page is charged by, and the prompt hash, whose prompts take
    their share of every window's room.
    """
    import hashlib
    import json

    from common.config import CONFIG_DIR, load_yaml, sequence_for_task
    from common.prompts import format_identity, prompt_hash
    from common.schema_sections import task_for
    from contracts.token_estimates import (
        CHARS_PER_TOKEN,
        TEMPLATE_OVERHEAD_TOKENS,
        estimate_visual_tokens,
    )
    from contracts.window_sizing import OCR_TOKENS_PER_PAGE

    section_map = load_yaml(CONFIG_DIR / "schema_sections.yaml")
    tasks = sorted({task_for(group) for group in section_map["groups"]})
    inputs = {
        "section_map": section_map,
        "budgets": {task: {"sequence": sequence_for_task(task, "policy"),
                           "visual_tokens_per_page": estimate_visual_tokens(1, task)}
                    for task in tasks},
        "page_estimates": [OCR_TOKENS_PER_PAGE, CHARS_PER_TOKEN, TEMPLATE_OVERHEAD_TOKENS],
        "prompt_hash": prompt_hash(),
    }
    # Under the fast profile its windows are another rule (one plan for every
    # group, ALIGNED_WINDOW_PLAN) and its answers another form: named in the id
    # itself, so the corpus manifest and the release bundle record them, and a
    # plan of one profile can never equal the other's.
    identity = format_identity()
    version = WINDOW_PLAN_VERSION
    if identity is not None:
        inputs["format"] = {**identity, "windows": ALIGNED_WINDOW_PLAN}
        version += f"+{identity['profile']}.{ALIGNED_WINDOW_PLAN}.{identity['answer_format']}"
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()
    return f"{version}:{digest[:12]}"

#: How the fast profile cuts a policy (:func:`_plan_aligned`): one window plan
#: for every section group of the document. Bumped whenever that rule changes.
ALIGNED_WINDOW_PLAN = "aligned-1"

def plan_windows(
    lob: str | list[str] | None,
    every_page: Sequence[int],
    declarations_page: int | None = None,
    page_texts: Mapping[int, str | None] | Sequence[str | None] | None = None,
) -> list[PolicyWindowPlan]:
    """Every window a policy is read with, in group order then page order.

    ``every_page`` is the pages to read - every page (:func:`pages_to_read`).
    ``page_texts`` is the OCR text the call will carry, by page number or in
    page order; a window whose pages' text would run past the call's budget is
    split (``expand_tasks.fit_windows_to_text``). None for ``image_only``, whose
    pages carry no text: a window is then sized by its page count alone.

    Under the fast format profile every group reads the same windows
    (:func:`_plan_aligned`).
    """
    from common.prompts import fast_layout
    from common.schema_sections import groups_for, leading_pages, pages_for, run_overlap
    from contracts.window_sizing import (
        fit_windows_to_text,
        page_tokens,
        pages_per_extraction_call,
        plan_policy_windows,
        window_page_room,
    )

    texts: dict[int, str | None] | None = None
    if page_texts is not None:
        texts = (dict(page_texts) if isinstance(page_texts, Mapping)
                 else dict(enumerate(page_texts, start=1)))
    total = max(every_page, default=0)
    groups = groups_for(lob)
    if groups and fast_layout("policy", None, lob, groups[0]):
        return _plan_aligned(lob, groups, every_page, declarations_page, texts)
    plans: list[PolicyWindowPlan] = []
    for group in groups_for(lob):
        pages = pages_for(group, list(every_page))
        capacity = pages_per_extraction_call("policy", None, lob, group)
        # A declarations group's first window is its leading pages and the page
        # the declarations were found on, read together; every other page
        # follows, packed by runs, as every other group's are.
        lead = leading_pages(group, pages, declarations_page=declarations_page)
        overlap = run_overlap(lob)
        windows = plan_policy_windows(pages, pages_per_window=capacity, leading=lead, overlap=overlap)
        if texts is not None:
            costs = {page: page_tokens(texts.get(page), page, total, group) for page in pages}
            windows = fit_windows_to_text(
                windows, costs.__getitem__, window_page_room("policy", None, lob, group),
                overlap=overlap)
        home = _home_window(windows, lead, declarations_page)
        single = len(windows) == 1
        plans.extend(
            PolicyWindowPlan(group, index, tuple(window), single=single, leads=index == home,
                             homed=single or home is not None)
            for index, window in enumerate(windows)
        )
    return plans

def _plan_aligned(
    lob: str | list[str] | None,
    groups: Sequence[str],
    every_page: Sequence[int],
    declarations_page: int | None,
    texts: dict[int, str | None] | None,
) -> list[PolicyWindowPlan]:
    """The fast profile's plan: ONE set of windows, read by every group.

    Each window's request is the shared system message, the window's pages,
    then its group's block (``common.prompts.render_group_block``); with the
    same windows for every group, every request of a window is the same tokens
    up to that block, and the engine's prefix cache reads the window's pages
    once for all its groups instead of once per group.

    The windows are planned once, as the most demanding group's would be:

    - **the declarations lead** - the declarations group's leading pages and
      the page the declarations were found on open the document, together,
      exactly as that group's windows open (:func:`leading_pages`). Every group
      reads them there; only a group the declarations lead (``decl``, ``dtd``)
      marks its window as led by them (:attr:`PolicyWindowPlan.leads`);
    - **the fewest pages any group's call has room for**: the smallest page
      count (``pages_per_extraction_call``) and the smallest text room
      (``window_page_room``) of the line's groups, so no group's request runs
      past its sequence cap or its reserved answer;
    - **runs overlap** as the line's map says (``run_overlap``), the page
      after the leading ones included (``expand_tasks.plan_aligned_windows``).

    A function of the line, the page count, the declarations page and the text
    - never of a gold label - as the standard plan is, and called by the
    dataset build and by serving alike.
    """
    from common.schema_sections import leading_pages, pages_for, reads_declarations, run_overlap
    from contracts.window_sizing import (
        page_tokens,
        pages_per_extraction_call,
        plan_aligned_windows,
        window_page_room,
    )

    pages = pages_for(groups[0], list(every_page))
    total = max(every_page, default=0)
    leaders = [group for group in groups if reads_declarations(group)]
    lead = leading_pages(leaders[0], pages, declarations_page=declarations_page) if leaders else []
    capacity = min(pages_per_extraction_call("policy", None, lob, group) for group in groups)
    overlap = run_overlap(lob)
    cost = room = None
    if texts is not None:
        costs = {page: max(page_tokens(texts.get(page), page, total, group) for group in groups)
                 for page in pages}
        cost = costs.__getitem__
        room = min(window_page_room("policy", None, lob, group) for group in groups)
    lead_windows, others = plan_aligned_windows(
        pages, pages_per_window=capacity, leading=lead, overlap=overlap, cost=cost, room=room)
    windows = lead_windows + others
    home = _home_window(windows, lead, declarations_page)
    single = len(windows) == 1
    return [
        PolicyWindowPlan(group, index, tuple(window), single=single,
                         leads=reads_declarations(group) and index == home,
                         homed=single or (reads_declarations(group) and home is not None))
        for group in groups
        for index, window in enumerate(windows)
    ]

def _home_window(windows: list[list[int]], lead: Sequence[int], declarations_page: int | None) -> int | None:
    """The window the declarations lead: of the windows the leading pages were
    split into, the one holding the declarations page, else the first. None
    for a group the declarations do not lead.

    Not simply the first: a ``dtd`` window holds three pages on some lines,
    and dense notices on pages 1-5 split a declarations window too, so pages
    [1, 2, 3] were taught the policy number the declarations print on page 4
    or 9 while the window showing it was taught to leave it out.
    """
    led = set(lead)
    pieces = [index for index, window in enumerate(windows) if led and set(window) <= led]
    if not pieces:
        return None
    return next((index for index in pieces if declarations_page in windows[index]), pieces[0])


class _Planned(NamedTuple):
    """One window to read: a section group over pages, and whether its group's
    declarations lead it (``PolicyWindowPlan.leads``)."""

    group: str
    pages: list[int]
    leads: bool = False
    #: Already sent once more after its request failed to reach the model (:func:`_settle`). A half is a new
    #: window and has its own one resend (code review 2026-11-10: halves had none).
    resent: bool = False

def _halves(window: _Planned, declarations_page: int | None) -> list[_Planned]:
    """A failed window as two smaller ones, in page order.

    The declarations lead the half holding their page - the first half when
    neither does - as they led the window: the rule training marks its windows
    by (``policy_windows._home_window``). Led by neither, a declarations window
    cut off on the declarations page lost its place to the half holding page 1,
    and a renewal notice's prior policy number was served.
    """
    half = len(window.pages) // 2
    pieces = [window.pages[:half], window.pages[half:]]
    home = next((index for index, pages in enumerate(pieces) if declarations_page in pages), 0)
    return [_Planned(window.group, pages, window.leads and index == home)
            for index, pages in enumerate(pieces)]
