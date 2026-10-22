"""How a policy is read: its windows, and each window's training target.

Every canonical policy — short or long — is read as the cross product of section
groups and page windows (``configs/schema_sections.yaml``): every group over
every page (decision 2026-10-19: every page, chunk by chunk, never selected
pages), ``decl`` and ``dtd`` with the declarations pages leading together in
their first window. Always, never above a length threshold: a threshold
computed from prompt length moves whenever the template, a schema or the vision
budget does, and would silently re-shape documents between two corpus builds.

**One planner, two callers.** :func:`plan_windows` is what the dataset build
expands a document with *and* what ``serving.pipeline`` extracts one with. It is
a function of the line, the page count and where the declarations were found —
never of a gold label — so the windows a model trains on are the windows it is
served. The declarations page comes from the same place on both sides too:
``serving.page_router.plan_pages`` over the OCR text the row actually carries
(none when there is no text).

**The target is the window's slice of the gold label** (:func:`window_target`):
the group's sections, narrowed to values printed on the window's own pages. A
value on a page the window was not shown is not in its target — asking the model
for it would teach it to invent. A row that spans a window boundary keeps the
fields each window can see, in both windows; the serving merge collapses the
halves. ``page_ref`` keeps the DOCUMENT's page numbers: the prompt's markers say
``<page 140 of 200>``, and a window-relative number would be wrong by an offset
nobody would notice - and only the window's own pages: a value printed on pages
1 and 30 is cited as ``[1]`` by the window showing page 1, which cannot see page
30 (Fideon SPEC_09 amendment item 1). The serving merge joins a value's pages
across windows again (``serving.policy_merge``).

Two sections are handled apart (Fideon SPEC_21): ``text_sections`` is never in
a target - the section builder attaches it at inference - and each
``additional_fields`` entry (label, value, section_hint, page_ref) is placed by
its own pages, as a field value is.

**A common-model line** (SPEC_21 overlays) links rows by id. Its ids, codes and
types are bare values no page prints, so they travel only with a row that keeps
a printed value on the window's pages, and only a printed value identifies a
row for the orphan rule; the target is narrowed to the window's own slice (a
reference to a table another group reads is not in it); and its ids are
renumbered from 1 in the window (common.structural_ids), so a window is taught
the ids it can see and never a reference to a row it is not shown. Its label
holds one part, of the line itself. Every finished target is checked against
the window's slice of the model view, the grammar its decoder is held to: a
target that grammar cannot write is refused, never taught.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from jsonschema import Draft202012Validator

from common.canonical import CanonicalLabelError, in_schema_order, is_field_value, to_model_target
from common.model_view import PRINTED_STRING_FIELDS

#: Never in an assistant target: attached at inference by the section builder
#: (Fideon SPEC_21, ``fideon:fsm_exclude``; SPEC_09 amendment item 1).
EXCLUDED_FROM_TARGETS = frozenset({"text_sections"})

#: Entries placed by their own page_ref (Fideon SPEC_21 §additional_fields).
ADDITIONAL_FIELDS = "additional_fields"

#: Keys of a common-model row whose plain strings are printed (a coverage's
#: form numbers): taught on a window only where its pages print them, not with
#: the ids and codes that ride with every fragment of the row.
PRINTED_STRINGS = frozenset(name for names in PRINTED_STRING_FIELDS.values() for name in names)


@dataclass(frozen=True)
class PolicyWindowPlan:
    """One extraction call: a section group over a set of pages."""

    group: str
    window_index: int
    pages: tuple[int, ...]
    #: The group reads its pages in one window. A value with no recorded page
    #: can only be placed when there is one window to place it in.
    single: bool

    @property
    def task(self) -> str:
        from common.schema_sections import task_for

        return task_for(self.group)


@dataclass
class TargetReport:
    """What slicing a gold label into windows left out, and why."""

    #: Values with no ``page_ref`` in a group read over several windows: there
    #: is no page to decide which window may be asked for them.
    unplaced: list[str] = field(default_factory=list)
    #: Values printed only on pages no window of their group reads. The same
    #: value is unreachable at serving; counting it is how a router or a page
    #: rule that misses real content shows up at corpus build.
    unread: list[str] = field(default_factory=list)
    #: Row fragments left out of a window because they hold none of their row's
    #: identifiers there (a premium with no coverage name): no window, and no
    #: merge, can tell which row such a fragment belongs to. On a common-model
    #: line only a fragment whose every value is also printed outside the
    #: window, where its row is taught, is left out.
    orphaned: list[str] = field(default_factory=list)
    #: Values with no recorded page that were given the one page whose OCR
    #: text prints them (:func:`with_inferred_pages`).
    inferred: list[str] = field(default_factory=list)
    #: References a window target left out because they name a row that window
    #: does not hold, or a table its slice leaves to another group (common-model
    #: lines), as ``group:path=id``. Expected across windows; counted so a line
    #: whose links mostly cross windows shows up at corpus build.
    dangling: list[str] = field(default_factory=list)
    #: Common-model rows left out of a window that shows none of their home
    #: pages - where their name or a figure of theirs is printed - and only a
    #: caption or a word the policy wording prints too (:data:`HOME_FIELDS`).
    away: list[str] = field(default_factory=list)


def pages_to_read(
    page_texts: Mapping[int, str] | Sequence[str] | None,
    page_count: int,
) -> tuple[list[int], int | None]:
    """``(every page, detected declarations page)`` — one rule for both sides.

    Every page, whatever the length (decision 2026-10-19). The declarations page
    is found in the OCR text; with none (``image_only``) there is none to find.
    """
    from serving.page_router import plan_pages

    pages = list(range(1, page_count + 1))
    if not page_texts:
        return pages, None
    texts = (
        dict(page_texts) if isinstance(page_texts, Mapping)
        else {index: text for index, text in enumerate(page_texts, start=1)}
    )
    return pages, plan_pages(texts).declarations_page


def plan_windows(
    lob: str | list[str] | None,
    every_page: Sequence[int],
    declarations_page: int | None = None,
) -> list[PolicyWindowPlan]:
    """Every window a policy is read with, in group order then page order.

    ``every_page`` is the pages to read - every page (:func:`pages_to_read`).
    """
    from common.schema_sections import groups_for, leading_pages, pages_for, run_overlap
    from data_pipeline.dataset_builder.expand_tasks import (
        pages_per_extraction_call,
        plan_policy_windows,
    )

    plans: list[PolicyWindowPlan] = []
    for group in groups_for(lob):
        pages = pages_for(group, list(every_page))
        capacity = pages_per_extraction_call("policy", None, lob, group)
        # A declarations group's first window is its leading pages and the page
        # the declarations were found on, read together; every other page
        # follows, packed by runs, as every other group's are.
        windows = plan_policy_windows(
            pages, pages_per_window=capacity,
            leading=leading_pages(group, pages, declarations_page=declarations_page),
            overlap=run_overlap(lob),
        )
        plans.extend(
            PolicyWindowPlan(group, index, tuple(window), single=len(windows) == 1)
            for index, window in enumerate(windows)
        )
    return plans


#: A printed value shorter than this is too common to place by searching:
#: "1", "Yes" and "100" are on most pages of a policy.
MIN_PLACEABLE_CHARS = 4


def multi_window_sections(lob: str | list[str] | None, plans: Sequence[PolicyWindowPlan]) -> set[str]:
    """The label sections read over several windows: the only ones where a
    value with no recorded page cannot be placed."""
    from common.schema_sections import sections_for

    return {name for plan in plans if not plan.single for name in sections_for(plan.group, lob)}


def with_inferred_pages(
    label: dict[str, Any], page_texts: Sequence[str] | None, report: TargetReport | None = None,
    *, sections: set[str] | None = None,
) -> dict[str, Any]:
    """``label`` with a page given to each value that records none, where the
    document's OCR text prints it on exactly one page.

    A value with no page in a group read over several windows can be asked of
    no window, so it was left out of training - 1,589 values on the delivered
    bundles. Placed only when the search is unambiguous: the value's printed
    text, at least :data:`MIN_PLACEABLE_CHARS` characters, matched on word
    boundaries, on one page and no other. On none or several it stays
    unplaced; a guessed page would teach the model to cite the wrong one.
    A copy when anything is placed; ``label`` itself otherwise.

    ``sections`` limits the search to those top-level sections
    (:func:`multi_window_sections`). A group read in ONE window keeps a value
    with no page as it is; given a page there, a declarations value whose only
    exact match was a notice on page 7 was moved to page 7 and then dropped
    from the declarations window it had always been taught in.
    """
    import copy
    import re

    from common.normalize import normalize_text

    if not page_texts:
        return label
    pages = [normalize_text(text) or "" for text in page_texts]
    placed: list[tuple[list[str], int]] = []

    def find(value: Any) -> int | None:
        needle = normalize_text(value)
        if not needle or len(needle) < MIN_PLACEABLE_CHARS:
            return None
        pattern = re.compile(rf"(?<![0-9a-z]){re.escape(needle)}(?![0-9a-z])")
        hits = [number for number, text in enumerate(pages, start=1) if pattern.search(text)]
        return hits[0] if len(hits) == 1 else None

    def walk(node: Any, path: list[Any]) -> None:
        if is_field_value(node):
            if node.get("page_ref") or (node.get("raw") is None and node.get("parsed") is None):
                return
            page = find(node.get("raw") if node.get("raw") is not None else node.get("parsed"))
            if page is not None:
                placed.append((path, page))
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, [*path, key])
        elif isinstance(node, list):
            for index, item in enumerate(node):
                walk(item, [*path, index])

    for name, section in label.items():
        if sections is None or name in sections:
            walk(section, [name])
    if not placed:
        return label
    out = copy.deepcopy(label)
    for path, page in placed:
        node = out
        for step in path:
            node = node[step]
        node["page_ref"] = [page]
        if report is not None:
            report.inferred.append("".join(f"[{s}]" if isinstance(s, int) else f".{s}" for s in path)[1:])
    return out


def window_target(
    label: dict[str, Any],
    lob: str | list[str] | None,
    plan: PolicyWindowPlan,
    report: TargetReport | None = None,
    *,
    printed_pages: Callable[[Any], set[int]] | None = None,
) -> dict[str, Any]:
    """The model-form target for one window: its sections, on its pages.

    ``printed_pages`` (:func:`printed_pages_in` over the document's text) places
    a coverage's form numbers, plain strings with no page of their own, on the
    windows whose pages print them; without it they ride with their row.

    On a common-model line, raises :class:`CanonicalLabelError` for a label
    with more than one part or a part of another line (:func:`with_own_part`),
    and for a target the window's decoder could not write (:func:`_check_writable`).
    """
    from common.canonical import schema_label, within_schema
    from common.schema_sections import sections_for
    from common.schemas import (
        cross_group_references,
        is_common_model,
        required_fields,
        resolved_schema,
    )
    from common.structural_ids import renumber_structural_ids

    common_model = is_common_model("policy", None, lob)
    # A label written for another line's schema, moved into this line's block
    # (configs/label_mappings.yaml). Then narrowed to what the schema can hold: a key the grammar refuses must not
    # be taught (common.canonical.within_schema).
    label = schema_label(label, "policy", None, lob)
    if common_model:
        label = _without_single_part(with_own_part(label, lob))
    pages = set(plan.pages)
    sliced = {
        name: (_additional_within if name == ADDITIONAL_FIELDS else _within)(
            label[name], name, pages, plan, report, lob=lob, common_model=common_model,
            printed_pages=printed_pages)
        for name in sections_for(plan.group, lob)
        if name in label and name not in EXCLUDED_FROM_TARGETS
    }
    sliced = {k: v for k, v in sliced.items() if v not in (None, {}, [])}
    view = resolved_schema("policy", None, lob, plan.group)
    if common_model:
        # A reference to a table another group reads is not in this window's
        # slice (common.schemas.cross_group_references), so it is left out of
        # the target before the target is written against the slice - which
        # has no place for it - and dangles in this window.
        left_out: list[str] = []
        sliced = _without_fields(sliced, cross_group_references(lob, plan.group), "", left_out)
        target = to_model_target(
            sliced, required=required_fields("policy", None, lob, plan.group), schema=view)
        # Narrowed to this window's slice: a key it does not declare is not
        # taught (common.canonical.within_schema).
        target, _ = within_schema(target, view)
        target, ids = renumber_structural_ids(target, lob)
        if report is not None:
            report.dangling.extend(f"{plan.group}:{ref}" for ref in [*left_out, *ids.dangling])
        target = in_schema_order(target, view)
        _check_writable(target, lob, plan)
        return target
    # Entries in SPEC_21's own shape (label, value, section_hint, page_ref), not
    # FieldValue envelopes: kept as they are rather than slimmed as envelopes.
    additional = sliced.pop(ADDITIONAL_FIELDS, None)
    target = to_model_target(sliced, required=required_fields("policy", None, lob, plan.group))
    if additional:
        target[ADDITIONAL_FIELDS] = [
            {key: entry[key] for key in ("label", "value", "section_hint", "page_ref") if key in entry}
            for entry in additional
        ]
    # In the order the decoding grammar writes keys (in_schema_order): the
    # window's own schema slice, the one it is constrained to.
    return in_schema_order(target, view)


def _without_single_part(label: dict[str, Any]) -> dict[str, Any]:
    """A single-part policy's label without ``part``: the schema says it is
    omitted then, and a window could not resolve it anyway (lob_parts is read
    with the declarations)."""
    parts = label.get("lob_parts")
    if isinstance(parts, list) and len(parts) > 1:
        return label

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "part"}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(label)


def with_own_part(label: dict[str, Any], lob: str | list[str] | None) -> dict[str, Any]:
    """A common-model label whose one part is written as the line itself, or refused.

    A common-model line's schema holds one part, of that line: the model view
    holds ``lob_parts[].lob`` to it. A package policy lists each of its lines
    in its lob and is read against the fallback schema
    (common.schemas.schema_key), so a label filed under one common-model line
    with two parts, or with a part of another line, is a mislabel - and its
    declarations target would fail the part's line anyway. A part whose line
    is read as this one (classic_auto, common.lob.merge_line) is written as
    this line, which is what the schema accepts. A copy when the part is
    rewritten; ``label`` itself otherwise.
    """
    from common.lob import merge_line
    from common.schemas import line_of_key, schema_key

    parts = label.get("lob_parts")
    if not isinstance(parts, list) or not parts:
        return label
    line = line_of_key(schema_key("policy", None, lob))
    if line is None:
        return label                 # the fallback: a package policy lists every one of its lines
    from common.schemas import LOB_SCHEMA_ALIASES

    named = [part.get("lob") if isinstance(part, dict) else None for part in parts]
    # A part named by its enum or L1 spelling (general_liability, cgl) is the line.
    spelled = {name: str(merge_line(str(name).strip().lower())) for name in named if name is not None}
    foreign = [name for name, spelling in spelled.items()
               if LOB_SCHEMA_ALIASES.get(spelling, spelling) != line]
    if len(parts) > 1 or foreign:
        raise CanonicalLabelError(
            f"lob_parts lists {len(parts)} part(s) ({', '.join(map(str, named))}), but a {line} "
            f"label holds one part, of {line} itself. A package policy lists every line in its "
            "lob and is read against the fallback schema, so a two-part or foreign-line label "
            "filed under one common-model line is a mislabel: its declarations target would "
            f"fail the part's line, which this line's schema holds to {line!r}."
        )
    if named[0] is None or named[0] == line:
        return label
    return {**label, "lob_parts": [{**parts[0], "lob": line}]}


def _without_fields(node: Any, names: frozenset[str], path: str, left_out: list[str]) -> Any:
    """``node`` without the fields ``names``, at any depth, as a copy: every
    window reads the same label. Each id a removed field held is added to
    ``left_out`` as ``path=id``, as renumbering records a dangling reference
    (common.structural_ids)."""
    if not names:
        return node
    if isinstance(node, dict):
        kept = {}
        for key, value in node.items():
            where = f"{path}.{key}" if path else key
            if key in names:
                left_out.extend(f"{where}={item}" for item in (value if isinstance(value, list) else [value])
                                if _is_structural(item))
            else:
                kept[key] = _without_fields(value, names, where, left_out)
        return kept
    if isinstance(node, list):
        return [_without_fields(item, names, f"{path}[{index}]", left_out) for index, item in enumerate(node)]
    return node


def _check_writable(target: dict[str, Any], lob: str | list[str] | None, plan: PolicyWindowPlan) -> None:
    """Refuse a common-model target the window's decoder could not write.

    Structured decoding holds a window to its slice of the model view. A target
    outside it teaches output the grammar never lets the model produce, and at
    serving the grammar forces something else in its place - a type changed, a
    value invented. So it is raised, never taught around: the label, or the
    view, is wrong. The first failing path is named.
    """
    from common.schemas import line_of_key, schema_key

    line = line_of_key(schema_key("policy", None, lob))
    errors = list(_view_validator(line, plan.group).iter_errors(target))
    if not errors:
        return
    first = min(errors, key=lambda e: [(isinstance(p, str), p) for p in e.absolute_path])
    where = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in first.absolute_path)
    raise CanonicalLabelError(
        f"{plan.group} window {list(plan.pages)}: {where.lstrip('.') or '<root>'} is not what the "
        f"window's decoder can write ({first.message[:200]}); a target outside the model view is "
        "never taught"
    )


@cache
def _view_validator(line: str | None, group: str) -> Draft202012Validator:
    """A common-model line's slice of the model view, as a validator: the schema
    a window of ``group`` is decoded against (common.schemas.resolved_schema).
    Cached: every window of every document of the line asks for it."""
    from common.schemas import resolved_schema

    return Draft202012Validator(resolved_schema("policy", None, line, group))


def _is_structural(value: Any) -> bool:
    """A value no page prints: an id, a reference, a code or a type."""
    if isinstance(value, (str, int, float, bool)):
        return True
    return isinstance(value, list) and bool(value) and all(
        isinstance(v, (str, int, float, bool)) for v in value)


def _within(
    node: Any, path: str, pages: set[int], plan: PolicyWindowPlan, report: TargetReport | None,
    *, lob: str | list[str] | None = None, common_model: bool = False,
    printed_pages: Callable[[Any], set[int]] | None = None,
) -> Any:
    if is_field_value(node):
        if node.get("raw") is None and node.get("parsed") is None:
            return None
        return _on_pages(node, path, pages, plan, report)
    if isinstance(node, dict):
        kept = {}
        for key, value in node.items():
            if common_model and (key in PRINTED_STRINGS or _is_structural(value)):
                continue
            child = _within(value, f"{path}.{key}", pages, plan, report,
                            lob=lob, common_model=common_model, printed_pages=printed_pages)
            if child not in (None, {}, []):
                kept[key] = child
        if common_model and kept:
            # The row's ids, codes and types ride with what this window shows
            # of it, in the label's own order; a row the window shows nothing
            # of is not in its target at all. Its printed strings (a coverage's
            # form numbers) ride only where these pages print them.
            out = {}
            for key, value in node.items():
                if key in PRINTED_STRINGS:
                    shown = _printed_here(value, pages, printed_pages)
                    if shown:
                        out[key] = shown
                elif _is_structural(value) or key in kept:
                    out[key] = value if _is_structural(value) else kept[key]
            return out
        return kept
    if isinstance(node, list):
        keys = _row_identifiers(node, path, lob=lob, common_model=common_model)
        home_fields = HOME_FIELDS.get(path) if common_model and not plan.single else None
        rows = []
        for i, item in enumerate(node):
            row = _within(item, f"{path}[{i}]", pages, plan, report,
                          lob=lob, common_model=common_model, printed_pages=printed_pages)
            if row in (None, {}, []):
                continue
            # A row whose name or figures are printed elsewhere, reaching this
            # window only by a caption ("Each Occurrence Limit") or a word
            # ("Included") the policy wording prints too: a label lists every
            # page that prints a value, so those pages carry the row's caption
            # and nothing of the row. Taught here, the model writes coverages
            # from wording. It is taught on the windows that show its home.
            home = _home_pages(item, home_fields) if home_fields else set()
            if home and not home & pages:
                if report is not None:
                    report.away.append(f"{plan.group}:{path}[{i}]")
                continue
            # A fragment that kept none of its row's identifiers in this window
            # is an orphan: labels record a short value such as a premium of
            # "1.0" on every page it happens to be printed on, so it reaches
            # windows that never show its coverage. Taught, the model writes
            # fragments the merge cannot place; the window that shows the
            # identifier carries the whole row.
            #
            # Unless it holds table rows of its own that ARE identified here: a
            # vehicle whose VIN is on page 6 and whose coverages table is on
            # page 7 is real content of the page-7 window. Dropping it taught
            # those coverages in no window at all. The serving merge joins such
            # a fragment to its row when only one row can own it.
            #
            # On a common-model line a fragment is an orphan only when nothing
            # is lost by it: every value it holds is also cited on a page
            # outside this window, where the row can be taught whole (a premium
            # printed again on a summary page). A coverage split at the window
            # boundary - its name at the foot of page 3, its limits and premium
            # at the top of page 4 - holds values no other window shows, so it
            # is taught here, or they are taught nowhere.
            spared = ((_holds_identified_rows(item, row, f"{path}[{i}]", lob)
                       or _cited_only_here(item, pages, plan)) if common_model
                      else _holds_rows(row))
            if keys and _identified(item, keys) and not _identified(row, keys) and not spared:
                if report is not None:
                    report.orphaned.append(f"{plan.group}:{path}[{i}]")
                continue
            rows.append(row)
        return rows
    return node


def _on_pages(
    node: dict[str, Any], path: str, pages: set[int], plan: PolicyWindowPlan, report: TargetReport | None
) -> dict[str, Any] | None:
    """``node`` as this window sees it, or None when it is printed on none of its pages.

    A copy, with ``page_ref`` narrowed to the window's pages: the label is
    cached per document and read by every window and mode, so trimming it in
    place would trim it for the windows after this one.
    """
    refs = {int(p) for p in node.get("page_ref") or []}
    if not refs:
        if plan.single:
            return node
        if report is not None and plan.window_index == 0:
            report.unplaced.append(f"{plan.group}:{path}")
        return None
    seen = refs & pages
    return {**node, "page_ref": sorted(seen)} if seen else None


def _printed_here(value: Any, pages: set[int], printed_pages: Callable[[Any], set[int]] | None) -> Any:
    """The items of a printed string list that this window may be taught: those
    printed on its pages. An item the document's text shows on none of its pages
    stays - OCR misreads form numbers on scans, and dropping one the image shows
    would teach skipping it - as does every item when there is no text at all."""
    if printed_pages is None or not isinstance(value, list):
        return value
    return [item for item in value if not printed_pages(item) or printed_pages(item) & pages]


def printed_pages_in(page_texts: Mapping[int, str | None] | Sequence[str | None]) -> Callable[[Any], set[int]]:
    """``value -> the pages whose text prints it``, looked up once per value: the
    lookup :func:`window_target` narrows a coverage's form numbers with. Takes a
    document's page texts by page number, or in page order from page 1."""
    from common import grounding

    items = page_texts.items() if isinstance(page_texts, Mapping) else enumerate(page_texts, start=1)
    texts = {int(number): grounding.PageText(text) for number, text in items}
    found: dict[str, set[int]] = {}

    def lookup(value: Any) -> set[int]:
        key = str(value)
        if key not in found:
            found[key] = ({number for number, text in texts.items() if grounding.on_page(value, text)}
                          if grounding.checkable(value) else set())
        return found[key]

    return lookup


def _additional_within(
    node: Any, path: str, pages: set[int], plan: PolicyWindowPlan, report: TargetReport | None,
    *, lob: str | list[str] | None = None, common_model: bool = False,
    printed_pages: Callable[[Any], set[int]] | None = None,
) -> list[Any]:
    """The ``additional_fields`` entries printed on this window's pages, page_ref trimmed."""
    if not isinstance(node, list):
        return []
    kept = []
    for index, entry in enumerate(node):
        value = entry.get("value") if isinstance(entry, dict) else None
        if value in (None, "", [], {}) or (
                is_field_value(value) and value.get("raw") is None and value.get("parsed") is None):
            # Nothing stated: an empty SPEC_21 envelope is no value either.
            continue
        where = f"{path}[{index}]"
        placed = (_entry_on_pages(entry, where, pages, plan, report) if common_model
                  else _on_pages(entry, where, pages, plan, report))
        if placed is not None:
            kept.append(placed)
    return kept


def _entry_on_pages(
    entry: dict[str, Any], path: str, pages: set[int], plan: PolicyWindowPlan,
    report: TargetReport | None,
) -> dict[str, Any] | None:
    """A SPEC_21 overflow entry as this window sees it. Placed by its own pages
    and its value's, both trimmed to the window's: a value printed on pages 1
    and 30 is cited as [1] by the window showing page 1."""
    value = entry.get("value")
    entry_refs = {int(p) for p in entry.get("page_ref") or []}
    value_refs = {int(p) for p in (value.get("page_ref") if isinstance(value, dict) else None) or []}
    refs = entry_refs | value_refs
    if not refs:
        if plan.single:
            return entry
        if report is not None and plan.window_index == 0:
            report.unplaced.append(f"{plan.group}:{path}")
        return None
    seen = refs & pages
    if not seen:
        return None
    placed = dict(entry)
    if "page_ref" in entry:
        placed["page_ref"] = sorted(entry_refs & pages) or sorted(seen)
    if isinstance(value, dict):
        placed["value"] = {**value, "page_ref": sorted(value_refs & pages) or sorted(seen)}
    return placed


def _row_identifiers(
    rows: list[Any], path: str, *, lob: str | list[str] | None = None, common_model: bool = False,
) -> list[str]:
    """The fields that identify this table's rows: the identifier scoring
    matches them on, plus a top-level section's declared key
    (``array_keys`` in configs/schema_sections.yaml), whose fields the merge
    joins on - a location's address on the page after its number is a half the
    merge can place.

    On a common-model line only a printed value identifies a row
    (:func:`_printed_identifiers`)."""
    from common.canonical import values_view
    from common.schema_sections import array_key
    from evaluation.metrics.field_accuracy import ROW_IDENTIFIERS, _infer_key_fields

    if not rows or not all(isinstance(r, dict) for r in rows):
        return []
    declared = list(array_key(path, lob)) if "." not in path and "[" not in path else []
    if common_model:
        return _printed_identifiers(rows, declared)
    keys = _infer_key_fields(values_view(rows))
    # Only a named identifier, never the all-fields fallback: every field
    # being "the key" would make every partial row an orphan.
    named = [k for k in keys if k in ROW_IDENTIFIERS or k == "building_number"]
    return [*named, *(k for k in declared if k not in named)]


def _printed_identifiers(rows: list[dict[str, Any]], declared: list[str]) -> list[str]:
    """A common-model table's identifiers, chosen among the values its pages print.

    A coverage code or an id rides with every fragment of its row, so counted as
    an identifier it would keep every fragment alive in every window; and picked
    first, as scoring picks it (a coverage's code comes before its name), it
    would hide the printed identifier behind it and leave the table with none.
    So the walk is over :data:`ROW_IDENTIFIERS`, then the declared key, keeping
    only fields that hold a value envelope: the first stated in at least half
    the rows wins, with a companion only if that is printed too (for coverages,
    ``coverage_name``). A declared key field printed in any row is added, as on
    a self-contained line.
    """
    from evaluation.metrics.field_accuracy import _COMPANIONS, ROW_IDENTIFIERS

    def stated(row: dict[str, Any], key: str) -> bool:
        value = row.get(key)
        return is_field_value(value) and (value.get("raw") is not None or value.get("parsed") is not None)

    def printed(key: str) -> bool:
        return sum(1 for row in rows if stated(row, key)) >= len(rows) / 2

    named: list[str] = []
    for key in dict.fromkeys((*ROW_IDENTIFIERS, *declared)):
        if printed(key):
            named = [key, *(c for c in _COMPANIONS.get(key, ()) if printed(c))]
            break
    return [*named, *(k for k in declared
                      if k not in named and any(is_field_value(r.get(k)) for r in rows))]


#: What places a common-model table's row on a page: its printed name, or a
#: figure of its own - a limit's or a deductible's amount or percentage, a
#: premium. Its other printed values are captions and words policy wording
#: prints too, so they do not say the row is on a page (:func:`_within`).
HOME_FIELDS: dict[str, frozenset[str]] = {
    "coverages": frozenset({"coverage_name", "amount", "percentage", "premium", "coinsurance_percent"}),
}


def _home_pages(node: Any, fields: frozenset[str]) -> set[int]:
    """The pages that print a row's :data:`HOME_FIELDS` values, its nested rows'
    included. Empty when it states none: a row of captions alone has no home,
    and stays wherever its values are cited."""
    pages: set[int] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if is_field_value(value):
                if key in fields and (value.get("raw") is not None or value.get("parsed") is not None):
                    pages.update(int(p) for p in value.get("page_ref") or [])
            else:
                pages |= _home_pages(value, fields)
    elif isinstance(node, list):
        for value in node:
            pages |= _home_pages(value, fields)
    return pages


def _holds_rows(row: Any) -> bool:
    """Whether a row still carries a table of its own after slicing. Nested
    rows went through the same rule, so any that remain are identified on
    these pages (or belong to a table nothing identifies)."""
    return isinstance(row, dict) and any(
        isinstance(value, list) and value and all(isinstance(r, dict) and not is_field_value(r) for r in value)
        for value in row.values()
    )


def _holds_identified_rows(item: dict[str, Any], row: dict[str, Any], path: str,
                           lob: str | list[str] | None) -> bool:
    """Whether a common-model fragment still carries a table of its own whose
    rows are identified on these pages - read against the label's whole table
    (``item``), as the fragment's own rows are. A nested table with no printed
    identifier (a coverage's limits: their types are bare) spares nothing."""
    for key, value in row.items():
        if not (isinstance(value, list) and value
                and all(isinstance(r, dict) and not is_field_value(r) for r in value)):
            continue
        keys = _row_identifiers(item.get(key) or [], f"{path}.{key}", lob=lob, common_model=True)
        if keys and any(_identified(r, keys) for r in value):
            return True
    return False


def _cited_only_here(node: Any, pages: set[int], plan: PolicyWindowPlan) -> bool:
    """Whether the label's row ``node`` holds a stated value, nested rows
    included, that this window teaches and no page outside it cites - a value
    that, left out here, no window would teach. Read on the label, not the
    fragment: the fragment's ``page_ref`` is already narrowed to these pages.
    A value with no page is in the window only when the window is the whole
    document (:func:`_on_pages`), and then it is cited nowhere else."""
    if is_field_value(node):
        if node.get("raw") is None and node.get("parsed") is None:
            return False
        refs = {int(p) for p in node.get("page_ref") or []}
        return refs <= pages if refs else plan.single
    if isinstance(node, dict):
        return any(_cited_only_here(value, pages, plan) for value in node.values())
    if isinstance(node, list):
        return any(_cited_only_here(value, pages, plan) for value in node)
    return False


def _identified(row: Any, keys: list[str]) -> bool:
    from common.canonical import values_view

    if not isinstance(row, dict):
        return False
    return any(values_view(row.get(k)) not in (None, "", []) for k in keys)


def unread_values(
    label: dict[str, Any], lob: str | list[str] | None, plans: Sequence[PolicyWindowPlan]
) -> list[str]:
    """Gold values printed on pages no window of their group reads."""
    from common.schema_sections import groups_for, sections_for

    out: list[str] = []
    for group in groups_for(lob):
        read = {p for plan in plans if plan.group == group for p in plan.pages}
        for name in sections_for(group, lob):
            if name in label:
                _collect_unread(label[name], name, read, out)
    return out


def _collect_unread(node: Any, path: str, read: set[int], out: list[str]) -> None:
    if is_field_value(node):
        refs = {int(p) for p in node.get("page_ref") or []}
        stated = node.get("raw") is not None or node.get("parsed") is not None
        if stated and refs and not refs & read:
            out.append(path)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _collect_unread(value, f"{path}.{key}", read, out)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _collect_unread(item, f"{path}[{index}]", read, out)
