"""Compile documents into chat-format JSONL training rows (IMPL-05, arch §6, §7).

Split first, expand second, inside whichever split the document was already
assigned to:

* **train** — one row per epoch, each in that epoch's sampled modality regime
  (arch v2.1 §6.1), written as ``train/epoch_1..4.jsonl``;
* **val / test** — one row per regime, so each is measured on the full eval
  population: all three, except validation under the fast format profile,
  which is read in the two its checkpoint selection reads (:func:`eval_modes`).
  Every val and test row says whether its carrier is among the train rows'
  (``carrier_seen_in_train``), and every row whether it is ``synthetic``.

A policy window's target is built once per document on its clean text
(:func:`_prepared`): a twin is checked on its digital render's text, the
verdict shared by its renders; a value no page prints is listed and the windows
holding it left out, never deleted; on a common-model line a value citing no
page is given every page that prints it, and a window that would still teach
one is left out rather than the document; a window whose answer runs past its
reservation is read as serving retries it, in halves (:func:`_window_rows`).
What all that did is counted (:data:`TARGET_COUNTS`) for the corpus manifest.

The rows are built through :func:`inference_core.input_builder.build_training_row`,
which is the same function the serving path uses to assemble a request. That is
what makes the training prompt and the inference prompt provably identical rather
than identical by convention (arch §7).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from common.canonical import CanonicalLabelError, has_envelopes, is_field_value, training_target
from common.constants import MODALITY_MODES, TRAINER_SPECIAL_TAGS
from common.schemas import is_canonical
from data_pipeline.dataset_builder.cap_check import (
    CapReport,
    answer_overflows,
    estimate_row,
    evaluate,
)
from data_pipeline.dataset_builder.label_verification import (
    VerificationReport,
    paths_after_removal,
    verified_label,
    without_paths,
)
from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
from data_pipeline.dataset_builder.policy_windows import (
    Placement,
    PolicyWindowPlan,
    TargetReport,
    multi_window_sections,
    pages_to_read,
    places_by_search,
    plan_windows,
    printed_pages_in,
    unread_values,
    window_halves,
    window_target,
    with_inferred_pages,
    with_placed_pages,
)
from data_pipeline.dataset_builder.sample_modes import EPOCH_FILES, ModeAssignment, sample_modes
from data_pipeline.dataset_builder.split_groups import (
    GroupSplitAssignment,
    assert_no_leakage,
    assert_single_tenant,
    assert_synthetic_is_train_only,
)
from inference_core.input_builder import build_training_row

log = logging.getLogger(__name__)


class CorpusBuildError(RuntimeError):
    """Raised when a corpus cannot be assembled correctly."""


@dataclass
class SourceDocument:
    """One labeled document, ready to expand into training rows."""

    source_id: str
    doc_type: str
    golden_label: dict[str, Any]
    #: One markdown string per page, in page order — never one joined blob.
    #: Joining is what destroyed the image/text pairing and split every table
    #: crossing a page boundary (a blank line ends a Markdown table).
    ocr_pages: list[str]
    image_paths: list[str]
    acord_form: str | None = None
    #: A policy's line of business, from its label metadata. Selects the
    #: canonical schema the prompt carries and the target is written in; absent,
    #: the policy uses the client's canonical fallback.
    lob: str | list[str] | None = None
    tenant_id: str | None = None
    field_provenance: dict[str, str] = field(default_factory=dict)
    is_scanned: bool = False

    #: The family this document belongs to (arch v2.1 §8.2). Splitting happens at
    #: this level, not at source_id: the same carrier's template and the same
    #: account's renewals must not span the split, or eval measures template
    #: memorisation. Defaults to the document's own id, which makes a document
    #: with no detected family its own group rather than silently ungrouped.
    group_id: str | None = None
    carrier: str | None = None

    #: Generated rather than collected (arch v2.1 §4d). Train-only: its labels
    #: are perfect because they were generated from them, so scoring against one
    #: measures the generator.
    synthetic: bool = False

    #: The split the document arrived with (metadata ``split``), when the
    #: delivery was split upstream; and the source document it was made from
    #: (metadata ``template_id``), the family that split was drawn by.
    delivered_split: str | None = None
    template_id: str | None = None
    #: A delivered test document of a carrier no training document has (metadata
    #: ``held_out_carrier``): reported apart, as a carrier the model never saw.
    delivered_held_out: bool = False
    #: Which twin of its seed a synthetic document is (None for a real one), and
    #: the render mode it was generated in (digital / scanned twins, SPEC_21).
    twin_index: int | None = None
    render_mode: str | None = None
    #: Label paths of values no page prints, in fields or extra fields
    #: (prepare_bundles' label rules): the windows holding one are left out of
    #: training.
    unprinted_values: list[str] = field(default_factory=list)
    #: The clean text a target is checked and placed against, when it is not
    #: this document's own: a scanned render's is its digital twin's, which
    #: prints the same values at the same places without the scan's OCR
    #: misreads (:func:`share_twin_text`). Never what a row carries - a row
    #: shows its own render's text, as serving would read it.
    clean_ocr_pages: list[str] | None = None

    @property
    def family(self) -> str:
        return self.group_id or self.source_id

    def __post_init__(self) -> None:
        # Classic auto is read as personal auto (common.lob.MERGED_LINES).
        from common.lob import merge_line

        self.lob = merge_line(self.lob)


@dataclass
class BuildResult:
    """The compiled corpus, plus what went into it."""

    rows_by_split: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    modality_counts: Counter = field(default_factory=Counter)
    corruption_details: dict[str, list[str]] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: Gold values no window's target could carry, per document: printed only on
    #: pages no window of their group reads, or with no page recorded in a group
    #: read over several windows. The same values are unreachable at serving, so
    #: this is where a page rule that misses real content becomes visible.
    window_notes: dict[str, list[str]] = field(default_factory=dict)
    #: Documents set aside, by reason: ``budget``, ``trainer_tag``, ``expansion``,
    #: ``empty_label`` (a canonical label whose every value is null).
    #: Every document that does not reach the corpus is counted here, so a stage
    #: summary can say how many were lost and why rather than a log line alone.
    set_aside: Counter = field(default_factory=Counter)
    #: Every row's estimated size against its task budget, and every document
    #: set aside because one of its rows did not fit (``cap_check``).
    cap_report: CapReport = field(default_factory=CapReport)
    #: What building the targets did, over the documents kept: rule-added values
    #: verified (``verify.*``), values with no page placed by the search or
    #: left without one and the windows that cost (``unpaged.*``), windows left
    #: out for a value no page prints (``unprinted.*``), and answers read as
    #: halves (``halves.*``). Recorded on the corpus manifest (:data:`TARGET_COUNTS`).
    target_counts: Counter = field(default_factory=Counter)

    @property
    def all_rows(self) -> list[dict[str, Any]]:
        return [row for rows in self.rows_by_split.values() for row in rows]

    def summary(self) -> str:
        counts = {split: len(rows) for split, rows in sorted(self.rows_by_split.items())}
        return f"{len(self.all_rows)} rows {counts}, modality {dict(self.modality_counts)}"


#: The keys of :attr:`BuildResult.target_counts`, each a count over the
#: documents kept (a twin's two renders are two documents):
#:
#: * ``verify.*`` - rule-added values checked on the clean text (review item 1):
#:   ``checked``, ``kept``, ``repaged`` (now citing the page(s) that print
#:   them), ``unprinted`` (listed, their windows left out), ``too_short``,
#:   ``unverifiable``; ``twin_renders_shared`` - renders checked on their
#:   digital twin's text, with its verdict;
#: * ``unpaged.*`` - values recording no page, on a line placed by search
#:   (item 2): ``values``, ``placed_one_page``, ``placed_several_pages``,
#:   ``not_printed`` and ``unchecked`` (left without a page); ``windows_dropped``
#:   (rows not built: the window would teach a value citing no page), and
#:   ``documents_saved`` - documents kept that held such a value, each of which
#:   was set aside whole before once one reached the declarations-led window;
#: * ``unprinted.windows_dropped`` - rows not built for a listed value no page prints;
#: * ``halves.*`` - answers over their reservation (item 38): ``windows_split``,
#:   ``rows`` built from the halves, ``documents_saved``.
TARGET_COUNTS: tuple[str, ...] = (
    "verify.checked", "verify.kept", "verify.repaged", "verify.unprinted", "verify.too_short",
    "verify.unverifiable", "verify.twin_renders_shared",
    "unpaged.values", "unpaged.placed_one_page", "unpaged.placed_several_pages", "unpaged.not_printed",
    "unpaged.unchecked", "unpaged.windows_dropped", "unpaged.documents_saved",
    "unprinted.windows_dropped",
    "halves.windows_split", "halves.rows", "halves.documents_saved",
)


def expand_document(
    document: SourceDocument,
    split: str,
    *,
    seed: int = 42,
    modes: tuple[str, ...] = MODALITY_MODES,
    counts: Counter | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand one document into rows, per mode in ``modes``.

    One row per mode for a flat document type. A canonical policy is read as
    windows - a section group over a set of pages (``policy_windows``) - so it
    becomes one row per window per mode, each prompting for its slice of the
    schema and targeting its slice of the label. Every row carries
    ``mode_index``, its position in ``modes``, which ``build_corpus`` stamps an
    epoch from.

    Every row lands in the **same** split, because the document was assigned
    before expansion. Returns ``(rows, details)``; details prefixed ``window``
    are the gold values no window could carry.

    Under arch v2.1 §6.1 the caller passes ONE mode per epoch, drawn by
    :mod:`data_pipeline.dataset_builder.sample_modes`. Passing all three — the v1
    behaviour — produces a 33/33/33 mix rather than the 50/20/30 the architecture
    specifies, and shows every document to the model three times per epoch, so a
    "3 epoch" run is nine passes. Val and test still take all three, so
    image-only accuracy is measured on the full eval population.

    ``counts`` receives what building the targets did (:data:`TARGET_COUNTS`).
    """
    counts = counts if counts is not None else Counter()
    windowed = document.doc_type == "policy" and is_canonical(
        document.doc_type, document.acord_form, document.lob
    )
    if windowed and not has_envelopes(document.golden_label):
        # Checked before slicing: a flat label has none of the canonical
        # sections, so every window's target would come out `{}` and the corpus
        # would teach "this policy states nothing" with no error anywhere.
        raise CanonicalLabelError(
            f"{document.source_id} is a policy with a flat, pre-canonical label. Convert it to "
            "the canonical schema before it enters the corpus."
        )
    if windowed and states_nothing(document.golden_label):
        # The same failure in the canonical shape: every value null, every
        # window's target empty sections. build_corpus sets these aside before
        # expansion (``empty_label``); refused here for any other caller.
        raise CanonicalLabelError(
            f"{document.source_id}: every value of its canonical label is null, so every window's "
            "target would be empty sections. Label it before it enters the corpus."
        )

    golden_json = ""
    if not windowed:
        # The target, not the label: every date is written MM/DD/YYYY.
        target = training_target(
            document.golden_label, document.doc_type, document.acord_form, document.lob
        )
        golden_json = json.dumps(target, ensure_ascii=False, sort_keys=False)
    rows: list[dict[str, Any]] = []
    details: list[str] = []

    for mode_index, mode in enumerate(modes):
        if mode == "image_only":
            ocr_pages = None
        elif mode == "noisy_ocr_image" and windowed:
            # Corrupted per WINDOW below, each under its own budget: a window is
            # what one row shows the model. One budget for a 200-page policy put
            # two corruptions in one or two of its windows, and every other
            # "noisy" row was byte-identical to its clean one.
            ocr_pages = list(document.ocr_pages)
        elif mode == "noisy_ocr_image":
            # One budget for the whole document, spent across its pages. Calling
            # the single-page corrupter per page would multiply the noise by the
            # page count and teach "OCR is always garbage" instead of arbitration.
            ocr_pages, page_details = corrupt_ocr_pages(
                document.ocr_pages, document.source_id, seed=seed
            )
            details.extend(page_details)
        else:
            ocr_pages = list(document.ocr_pages)

        if windowed:
            for row in _policy_window_rows(
                document, split, mode, ocr_pages, details, seed=seed, counts=counts
            ):
                row["mode_index"] = mode_index
                rows.append(row)
            continue

        row = build_training_row(
            document.doc_type,
            document.source_id,
            document.image_paths,
            ocr_pages,
            mode,
            golden_json,
            acord_form=document.acord_form,
            lob=document.lob,
            tenant_id=document.tenant_id,
            split=split,
            # Recorded, not enforced: no de-identification, Option A (IMPL-05 §1).
            deidentified=False,
        )
        row["mode_index"] = mode_index
        rows.append(row)

    if windowed:
        _count_prepared(_prepared(document), counts)
    return rows, details


def _policy_window_rows(
    document: SourceDocument,
    split: str,
    mode: str,
    ocr_pages: list[str] | None,
    details: list[str],
    *,
    seed: int = 42,
    counts: Counter | None = None,
) -> list[dict[str, Any]]:
    """One row per window, planned exactly as serving plans them.

    Every page is read; the declarations page is found, and each window sized,
    from the OCR text THIS row carries - clean or none - as serving does for the
    same input. A noisy row plans on the clean text: its corruption is drawn per
    window, after the plan, and keeps each page's length.

    The target is the same in every mode (:func:`_prepared`): the label checked
    and placed once per document, on its clean text. A window is left out - never
    the document - when it would teach a value no page prints: one listed as
    unprinted, or (on a line placed by search) one citing no page that no page
    was found to print. A window whose answer runs past its reservation is read
    as two halves, as serving retries it (:func:`_window_rows`).
    """
    counts = counts if counts is not None else Counter()
    page_count = len(document.image_paths)
    every_page, declarations_page = pages_to_read(ocr_pages, page_count)
    plans = plan_windows(document.lob, every_page, declarations_page, ocr_pages)

    report = TargetReport()
    prepared = _prepared(document)
    label, checked = prepared.label, prepared.checked
    multi = multi_window_sections(document.lob, plans)
    if prepared.placement is None:
        # Only in sections read over several windows: one window keeps a value
        # that records no page as it is (policy_windows.with_inferred_pages).
        label = with_inferred_pages(label, prepared.clean, report, sections=multi)
    unprinted = _unprinted_values(label, prepared.unprinted_paths)
    # A value no page prints and none is recorded for, in a section read over
    # several windows: taught nowhere, as before every page was read. The
    # declarations-led window takes values with no page, and holding this one
    # it was left out whole - the window that shows the declarations lost to
    # one derived value. Taught there, it would be a value to invent.
    unpaged = [path for path, section, pages in unprinted if not pages and section in multi]
    if unpaged:
        label = _without_paths(label, unpaged)
        report.unplaced.extend(f"{path.split('.', 1)[0].split('[', 1)[0]}:{path}" for path in unpaged)
        unprinted = [entry for entry in unprinted if entry[0] not in unpaged]
    build = _WindowBuild(document, split, mode, ocr_pages, label, report, prepared.printed, details,
                         counts, seed=seed, declarations_page=declarations_page, page_count=page_count,
                         search_placed=prepared.placement is not None)
    rows: list[dict[str, Any]] = []
    for plan in plans:
        held = _held_unprinted(plan, document.lob, unprinted)
        if held:
            # A value no page prints, in this window's sections and pages: left
            # out whole. Taught, it is a value to invent; deleted from the
            # target, a printed value the search missed would be taught as one
            # to skip.
            details.append(f"window {mode}: {plan.group}:{plan.window_index} left out, it holds "
                           f"{len(held)} value(s) no page prints ({held[0]})")
            counts["unprinted.windows_dropped"] += 1
            continue
        rows += _window_rows(build, plan)

    notes = [f"window {mode}: unread {path}" for path in unread_values(
        label, document.lob, plans
    )]
    notes += [f"window {mode}: unplaced {path}" for path in dict.fromkeys(report.unplaced)]
    # Row fragments left out because their window shows none of the row's
    # identifiers and no table of their own: not taught, so said.
    notes += [f"window {mode}: orphaned {path}" for path in sorted(set(report.orphaned))]
    # Rows left out of a window that shows only their caption, away from the
    # pages that print their name or figures: taught on those, so said.
    notes += [f"window {mode}: away from home {path}" for path in sorted(set(report.away))]
    if report.dangling:
        # A common-model row's link to a row another window reads: left out of
        # the window that cannot see the row it names, by design. Counted, not
        # listed per value, so a line whose links mostly cross windows shows.
        details.append(
            f"window {mode}: {len(set(report.dangling))} reference(s) left out of a window "
            "that does not hold the row they name"
        )
    if checked.dropped or checked.repaged or checked.unverifiable or checked.unprinted:
        details.append(
            f"window {mode}: of {checked.checked} rule-added label value(s), "
            f"{len(checked.dropped) + len(checked.unprinted)} not printed on the page and listed as "
            f"unprinted (the windows holding them left out), "
            f"{len(checked.repaged)} moved to the page(s) that print them, "
            f"{checked.unverifiable} unverifiable (no OCR text)"
        )
    if notes:
        log.warning(
            "%s (%s): %d gold value(s) no window can carry - printed on pages no window of "
            "their group reads, or with no page recorded. Serving cannot reach them either.",
            document.source_id, mode, len(notes),
        )
        details.extend(notes)
    return rows


@dataclass
class _Prepared:
    """A document's label as every window of every mode reads it (:func:`_prepared`)."""

    #: Checked (``verified_label(listed=True)``) and, on a line placed by
    #: search, every value that records no page given the pages that print it.
    label: dict[str, Any]
    checked: VerificationReport
    #: The values no page prints: those verification listed, then the document's own.
    unprinted_paths: list[str]
    #: None on a line not placed by search (``policy_windows.places_by_search``).
    placement: Placement | None
    printed: Any
    #: The clean text all of it was read against.
    clean: list[str]
    #: Whether a digital twin's text and verdict stood in for this render's.
    twin_shared: bool = False


def _prepared(document: SourceDocument) -> _Prepared:
    """The label every window of ``document`` slices, built once per document:
    it depends on neither the reading mode nor the window.

    All of it on the CLEAN text (:attr:`SourceDocument.clean_ocr_pages`, else
    the document's own): a rule-added value is taught only if that text prints
    it (``label_verification``, listed rather than deleted), and on a line
    placed by search each value recording no page is given every page that text
    prints it on (``policy_windows.with_placed_pages``).
    """
    cached = getattr(document, "_prepared_targets", None)
    if cached is not None:
        return cached
    clean = list(document.clean_ocr_pages if document.clean_ocr_pages is not None else document.ocr_pages)
    label, checked = _verified(document)
    # Verification's own first: it is what this build found, the document's
    # list what the label rules found before it. A path recorded against the
    # gold is read on the verified label (paths_after_removal), which a listed
    # check never shortens.
    own = paths_after_removal(document.unprinted_values, checked.dropped)
    unprinted_paths = list(dict.fromkeys([*checked.unprinted, *own]))
    placement = None
    if places_by_search(document.lob):
        label, placement = with_placed_pages(label, clean, skip=unprinted_paths)
    # Where the clean text prints a coverage's form numbers, which carry no page
    # of their own: they are taught on the windows that show them.
    printed = printed_pages_in(clean) if clean else None
    prepared = _Prepared(label, checked, unprinted_paths, placement, printed, clean,
                         twin_shared=bool(getattr(document, "_twin_shared", False)))
    document._prepared_targets = prepared
    return prepared


def _count_prepared(prepared: _Prepared, counts: Counter) -> None:
    """Add one document's verification and placement to ``counts`` (:data:`TARGET_COUNTS`)."""
    checked = prepared.checked
    counts["verify.checked"] += checked.checked
    counts["verify.kept"] += checked.kept
    counts["verify.repaged"] += len(checked.repaged)
    counts["verify.unprinted"] += len(checked.unprinted) + len(checked.dropped)
    counts["verify.too_short"] += checked.too_short
    counts["verify.unverifiable"] += checked.unverifiable
    counts["verify.twin_renders_shared"] += int(prepared.twin_shared)
    placement = prepared.placement
    if placement is not None:
        counts["unpaged.values"] += (len(placement.placed) + len(placement.not_printed)
                                     + len(placement.unchecked))
        counts["unpaged.placed_one_page"] += sum(1 for pages in placement.placed.values() if len(pages) == 1)
        counts["unpaged.placed_several_pages"] += sum(1 for pages in placement.placed.values() if len(pages) > 1)
        counts["unpaged.not_printed"] += len(placement.not_printed)
        counts["unpaged.unchecked"] += len(placement.unchecked)


@dataclass
class _WindowBuild:
    """What every window row of one document in one mode is built from."""

    document: SourceDocument
    split: str
    mode: str
    ocr_pages: list[str] | None
    label: dict[str, Any]
    report: TargetReport
    printed: Any
    details: list[str]
    counts: Counter
    seed: int
    declarations_page: int | None
    page_count: int
    #: The line places values by search: a window that would teach a value
    #: citing no page is left out (its decoder cannot write one).
    search_placed: bool


def _window_rows(build: _WindowBuild, plan: PolicyWindowPlan, part: str = "") -> list[dict[str, Any]]:
    """The rows of one planned window: one, none, or its halves'.

    * None when its target would teach a value citing no page - a value no page
      was found to print (review item 2): the window is left out, never the
      document, as a window holding an unprinted value is.
    * Its halves' when its answer runs past the reservation (review item 38):
      serving retries such a window as two halves (``serving.pipeline._halves``,
      ``policy_windows.window_halves``), and the halves of a half again, down to
      one page; so are its rows built. A row keeps ``window_index`` and records
      which half it is (``window_half``: "0", "1", "1.0" ...). A single page
      whose answer still overruns keeps its row, and the cap check sets the
      document aside, as before.
    """
    document, mode = build.document, build.mode
    report = TargetReport()
    unpaged: list[str] | None = [] if build.search_placed else None
    target = window_target(build.label, document.lob, plan, report, printed_pages=build.printed,
                           unpaged=unpaged)
    name = f"{plan.group}:{plan.window_index}" + (f"/{part}" if part else "")
    if unpaged:
        build.details.append(f"window {mode}: {name} left out, it holds {len(unpaged)} value(s) "
                             f"citing no page that no page was found to print ({unpaged[0]})")
        build.counts["unpaged.windows_dropped"] += 1
        return []
    indices = [page - 1 for page in plan.pages]
    ocr_pages = build.ocr_pages
    window_ocr = None if ocr_pages is None else [ocr_pages[i] for i in indices]
    corruption: list[str] = []
    if window_ocr is not None and mode == "noisy_ocr_image":
        # The document-level budget, applied to the pages this row carries.
        # Seeded per window, so a rebuild is byte-identical.
        key = f"{document.source_id}#{plan.group}:{plan.window_index}" + (f".{part}" if part else "")
        window_ocr, window_details = corrupt_ocr_pages(window_ocr, key, seed=build.seed)
        corruption = [f"window {name} {d}" for d in window_details]
    row = build_training_row(
        document.doc_type,
        document.source_id,
        [document.image_paths[i] for i in indices],
        window_ocr,
        mode,
        json.dumps(target, ensure_ascii=False, sort_keys=False),
        acord_form=document.acord_form,
        lob=document.lob,
        sections=plan.group,
        tenant_id=document.tenant_id,
        split=build.split,
        deidentified=False,
        # The document's own page numbers, never window-relative: the
        # target's page_ref is the document's, and so are the markers.
        page_numbers=list(plan.pages),
        total_pages=build.page_count,
    )
    if len(plan.pages) > 1 and answer_overflows(plan.task, row["messages"][-1]["content"], document.doc_type):
        build.details.append(f"window {mode}: {name} answer over its reservation; read as two halves, "
                             "as serving retries it")
        build.counts["halves.windows_split"] += 1
        rows: list[dict[str, Any]] = []
        for index, half in enumerate(window_halves(plan, build.declarations_page)):
            rows += _window_rows(build, half, f"{part}.{index}" if part else str(index))
        return rows
    _merge_report(build.report, report)
    build.details.extend(corruption)
    row["task"] = plan.task
    row["window_index"] = plan.window_index
    row["window_pages"] = list(plan.pages)
    # Whether the group's declarations lead this window: the validation
    # merge (validation_generation.merged_windows) merges it first within
    # its group, as serving does (PolicyWindow.leads), so a split
    # declarations window's lead piece wins there too.
    row["window_leads"] = plan.leads
    if part:
        row["window_half"] = part
        build.counts["halves.rows"] += 1
    return [row]


def _merge_report(into: TargetReport, report: TargetReport) -> None:
    """A kept window's notes added to its document's: a window read as halves
    reports what its halves left out, not what it did."""
    for name in ("unplaced", "unread", "orphaned", "inferred", "dangling", "away"):
        getattr(into, name).extend(getattr(report, name))


def states_nothing(label: Any) -> bool:
    """Whether a canonical label states no value at all: it has envelopes, and
    each one's ``raw`` and ``parsed`` are null or blank.

    The flat-label guard (:func:`has_envelopes`) passes such a label - it IS in
    the canonical shape - and every window's target came out as empty sections
    (``{"carrier": {}, ...}``), teaching the model that the policy prints
    nothing. Bare values beside the envelopes (row ids, bookkeeping) are not
    values the model is asked to read and do not count. A flat label is not
    judged here: it has no envelopes, and the flat-label guard refuses it for a
    policy.
    """
    return has_envelopes(label) and not _any_envelope_value(label)


def _any_envelope_value(node: Any) -> bool:
    if is_field_value(node):
        return any(not _blank(node.get(key)) for key in ("raw", "parsed"))
    if isinstance(node, dict):
        return any(_any_envelope_value(value) for value in node.values())
    if isinstance(node, list):
        return any(_any_envelope_value(value) for value in node)
    return False


def _blank(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return value is None


def _unprinted_values(label: dict[str, Any], paths: list[str]) -> list[tuple[str, str, set[int]]]:
    """``(path, section, pages it cites)`` for each label value no page prints."""
    import re

    out = []
    for path in paths or ():
        node: Any = label
        for key, index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", path):
            step: Any = int(index) if index else key
            try:
                node = node[step]
            except (KeyError, IndexError, TypeError):
                node = None
                break
        if isinstance(node, dict):
            pages = {int(p) for p in node.get("page_ref") or [] if str(p).isdigit()}
            out.append((path, re.split(r"[.\[]", path, maxsplit=1)[0], pages))
    return out


def _without_paths(label: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    """A copy of ``label`` without the values at ``paths`` (``policy.term``,
    ``coverages[2].limit``, ``named_insured.additional_names[1]``); a path that
    leads nowhere is skipped.

    A list member too: only a dict key was removed before, so an unprinted
    member of a list (an additional name) stayed in the label - and, taken off
    the unprinted list as "removed", was taught with nothing holding its window.
    """
    return without_paths(label, paths)


def _held_unprinted(plan: Any, lob: Any, unprinted: list[tuple[str, str, set[int]]]) -> list[str]:
    """The unprinted values this window's target would hold: its group's
    sections, on its pages - or, citing no page, its group's only window (one
    read over several windows has them removed first)."""
    from common.schema_sections import sections_for

    if not unprinted:
        return []
    sections = set(sections_for(plan.group, lob))
    window = set(plan.pages)
    return [path for path, section, pages in unprinted
            if section in sections and (pages & window if pages else plan.single)]


def classify_rows(
    document: SourceDocument, split: str, modes: tuple[str, ...], *, seed: int = 42,
) -> list[dict[str, Any]]:
    """The adapter's line question for one policy: one row per mode.

    A later stage of line detection (serving.doc_type_classifier): the adapter
    itself chooses the policy's line among every known line - where the
    confusable pairs are (homeowners and dwelling fire; personal auto,
    motorcycle and RV) - rather than the base model alone. The messages are the
    classifier's own (``classifier_messages``) over the input serving picks
    (``classifier_input``: the first two pages' images, their text joined, none
    under ``image_only``) - the text in the row's own mode, corrupted under
    ``noisy_ocr_image`` as a window's is, so the input trained is the input
    served. Serving does not ask this question yet (``build_corpus`` refuses
    the rows until it does). Nothing for a document that is no policy, names
    several lines, or has a line with no schema.
    """
    from common.lob import known_lines
    from serving.doc_type_classifier import (
        CLASSIFIER_PAGES,
        classifier_input,
        classifier_messages,
        known_line,
    )

    lines = document.lob if isinstance(document.lob, list) else [document.lob]
    line = known_line(lines[0]) if document.doc_type == "policy" and len(lines) == 1 else None
    if line is None:
        return []
    choices = sorted(known_lines())
    # Page N is the N-th image and the N-th text, as in every other row.
    pages = range(1, len(document.image_paths) + 1)
    rows = []
    for index, mode in enumerate(modes):
        texts: list[str] | None = None
        if mode != "image_only":
            texts = list(document.ocr_pages)
            if mode == "noisy_ocr_image":
                # The document-level budget on the pages this row carries, seeded
                # per row as a window's is: spread over a long policy, it would
                # rarely touch the opening pages and leave the row clean.
                opening, _details = corrupt_ocr_pages(
                    texts[:CLASSIFIER_PAGES], f"{document.source_id}#classify", seed=seed)
                texts[:CLASSIFIER_PAGES] = opening
        images, text = classifier_input(
            pages, lambda page: document.image_paths[page - 1],
            None if texts is None else (lambda page, t=texts: t[page - 1] if page <= len(t) else None),
        )
        messages = classifier_messages(images, text, lines=choices, line_only=True)
        messages.append({"role": "assistant", "content": json.dumps({"lob": line})})
        rows.append({
            "doc_type": "policy", "acord_form": None, "lob": [line], "modality_mode": mode,
            "source_id": document.source_id, "tenant_id": document.tenant_id, "split": split,
            "deidentified": False, "messages": messages, "task": "classify", "mode_index": index,
        })
    return rows


def build_corpus(
    documents: list[SourceDocument],
    assignment: GroupSplitAssignment,
    *,
    seed: int = 42,
    mode_assignment: ModeAssignment | None = None,
    max_expansion_failures: int = 0,
    with_classify_rows: bool = False,
) -> BuildResult:
    """Compile every document into its split's rows, then assert no leakage.

    ``mode_assignment`` carries the per-document, per-epoch modality draw
    (arch v2.1 §6.1). Without one, it is drawn here over the **train** documents
    with the same seed. Train always gets one row per epoch; there is no path
    that expands a train document into all three regimes.

    ``with_classify_rows`` would add, per policy with a known line, the
    adapter's line question (:func:`classify_rows`). It is refused: serving
    does not ask the adapter that question yet - its classifier asks every
    document the type-and-line question on the base model - so the rows would
    teach an answer nothing reads. That stage at serving and these rows are
    turned on together, once the base
    model's zero-shot line detection has been measured and found short of its
    target; that is also when ``common.tasks.CORPUS_TASKS`` gains ``classify``.

    A document that fails to expand is set aside - except on a common-model
    line, where more than ``max_expansion_failures`` such documents fail the
    build. Those labels are new (SPEC_21) and their target builder is new: a
    systematic error there would otherwise shrink the corpus quietly, one
    "expansion" set-aside at a time, and train on whatever survived.
    """
    from common.schemas import is_common_model

    if with_classify_rows:
        raise CorpusBuildError(
            "with_classify_rows trains the adapter's line question, which serving does not ask "
            "it yet: its classifier asks every document the type-and-line question on the base "
            "model. Add that stage to serving and turn both on together, or the rows teach an "
            "answer nothing reads."
        )
    result = BuildResult(rows_by_split={"train": [], "val": [], "test": []})
    common_model_failures: list[tuple[str, str]] = []
    if mode_assignment is None:
        mode_assignment = sample_modes(train_source_ids(documents, assignment), seed=seed)
    # A twin's renders are checked once, on the digital render's clean text
    # (review item 1): one target for both, whatever the scan's OCR misread.
    share_twin_text(documents)

    for document in documents:
        try:
            split = assignment.split_of(document.family)
        except Exception as exc:
            result.skipped.append((document.source_id, str(exc)))
            continue

        if states_nothing(document.golden_label):
            # Before expansion, and its own reason: an empty gold is a labelling
            # gap to fill, not a target builder to debug, so it is not counted
            # among a common-model line's expansion failures.
            reason = (
                f"{document.source_id}: every value of its canonical label is null, so every "
                "target built from it would be empty sections and teach 'this document states "
                "nothing'. Label it, or leave it out of the delivery."
            )
            result.skipped.append((document.source_id, reason))
            result.set_aside["empty_label"] += 1
            log.warning("setting aside %s", reason)
            continue

        # Train draws ONE mode per epoch (§6.1); val and test keep every mode
        # they are read in (eval_modes) so each is measured on the full eval
        # population rather than a sample of it.
        if split == "train":
            epoch_modes = tuple(
                mode_assignment.mode_for(document.source_id, e)
                for e in range(1, mode_assignment.epochs + 1)
            )
        else:
            epoch_modes = eval_modes(split, document)

        doc_counts: Counter = Counter()
        try:
            rows, details = expand_document(document, split, seed=seed, modes=epoch_modes, counts=doc_counts)
        except Exception as exc:  # noqa: BLE001 - one bad document must not stop a corpus
            result.skipped.append((document.source_id, f"expansion failed: {exc}"))
            result.set_aside["expansion"] += 1
            log.warning("skipping %s: %s", document.source_id, exc)
            if document.doc_type == "policy" and is_common_model("policy", None, document.lob):
                common_model_failures.append((document.source_id, str(exc)))
                if len(common_model_failures) > max_expansion_failures:
                    shown = "; ".join(f"{sid}: {err[:160]}" for sid, err in common_model_failures[:3])
                    raise CorpusBuildError(
                        f"{len(common_model_failures)} common-model document(s) failed to expand "
                        f"(allowed {max_expansion_failures}): {shown}"
                    ) from exc
            continue

        if with_classify_rows:  # refused above until serving asks the adapter's line question
            rows += classify_rows(document, split, epoch_modes, seed=seed)

        # Reject, never truncate. A row over its task budget would be clipped on
        # the pod, and a clipped target trains the model to stop early. The
        # whole document is set aside, not the one row: a document missing from
        # some epochs and not others is a run whose epoch count describes
        # something else, which train_rows_by_epoch refuses.
        refusal = _refusal(document.source_id, rows, result.cap_report)
        if refusal:
            category, reason = refusal
            result.skipped.append((document.source_id, reason))
            result.set_aside[category] += 1
            log.warning("setting aside %s: %s", document.source_id, reason)
            continue
        _count_kept(document, doc_counts, result)

        # Train rows are stamped with the epoch they belong to, so the corpus can
        # be written as epoch_1..4.jsonl and a run reproduced from the files
        # alone rather than from a sampler behaving identically at training time.
        for row in rows:
            mode_index = row.pop("mode_index")
            if split == "train":
                row["epoch"] = mode_index + 1

        # Stamped here rather than inside expand_document: the family is a
        # property of the corpus build, not of one document's expansion, and the
        # leakage assertion reads it off every row.
        for row in rows:
            row["group_id"] = document.family
            # A bool on every row, so selection and the reports can score the
            # real documents apart (review item 6).
            row["synthetic"] = bool(document.synthetic)
            # Fideon SPEC_09 amendment item 8: carrier and twin index on every
            # row (the seed is group_id). twin_index is None for real documents.
            row["carrier"] = document.carrier
            row["twin_index"] = document.twin_index
            # Which render of its twin it is (digital / scanned): the training view
            # trains one render of a twin per epoch (training.corpus_view).
            row["render_mode"] = document.render_mode
            # What the data mix's scanned_share counts (training.data_mix), and the
            # scanned subset of a validation report.
            row["is_scanned"] = document.is_scanned
            if assignment.half_of(document.family):
                row["val_half"] = assignment.half_of(document.family)

        result.rows_by_split[split].extend(rows)
        for row in rows:
            result.modality_counts[row["modality_mode"]] += 1
        corruption = [d for d in details if not d.startswith("window ")]
        windows = [d for d in details if d.startswith("window ")]
        if corruption:
            result.corruption_details[document.source_id] = corruption
        if windows:
            result.window_notes[document.source_id] = windows

    stamp_carrier_seen(result.rows_by_split)
    # The checks that make the ordering rule real rather than documented.
    assert_no_leakage(assignment, result.all_rows)
    if not assignment.delivered:
        # A delivered split placed synthetic twins of held-out sources in val and
        # test on purpose (assign_delivered_splits); the gate reports real ones apart.
        assert_synthetic_is_train_only(result.all_rows)
    assert_single_tenant(result.all_rows)

    log.info("corpus built: %s", result.summary())
    return result


#: The modes a validation document is read in under the fast format profile
#: (review item 6): the two its checkpoint selection reads (``training/unified.yaml``
#: ``format_profiles.fast.evaluation.selection_document_modes``). noisy_ocr_image
#: keeps a token share of training and is measured by the golden eval, which
#: reads every mode of the frozen set itself.
FAST_VAL_MODES: tuple[str, ...] = ("ocr_plus_image", "image_only")


def eval_modes(split: str, document: SourceDocument | None = None) -> tuple[str, ...]:
    """The reading modes a ``val`` or ``test`` document is built in: every mode -
    except a validation policy read in windows under the fast profile, which
    takes :data:`FAST_VAL_MODES`. Only a canonical policy: the fast profile
    writes nothing else differently (``common.prompts.fast_layout``), and an
    ACORD form or a Loss Run keeps its rows as they were. The test split keeps
    every mode in either profile: it is the set the golden eval set is frozen
    from (``freeze_eval_set``, which reads its documents, not its rows)."""
    from common.config import fast_profile

    if (split == "val" and fast_profile() and document is not None and document.doc_type == "policy"
            and is_canonical(document.doc_type, document.acord_form, document.lob)):
        return FAST_VAL_MODES
    return MODALITY_MODES


def share_twin_text(documents: list[SourceDocument]) -> int:
    """Point each render of a twin at its digital render's clean text, and give
    it the digital render's verification, so the twin is checked once and every
    render teaches one target (review item 1). Returns the renders that now
    share a digital twin's.

    A twin is a synthetic document's family and twin number; its digital render
    is the one whose ``render_mode`` is ``digital``. A render of another page
    count, or a gold that differs from the digital render's, is checked on the
    digital text but on its own label. A render already given clean text is
    left as it is.
    """
    renders: dict[tuple[str, int], list[SourceDocument]] = {}
    for document in documents:
        if document.synthetic and document.twin_index is not None and document.render_mode:
            renders.setdefault((document.family, document.twin_index), []).append(document)
    shared = 0
    for twin in renders.values():
        digital = next((d for d in twin if d.render_mode == "digital"), None)
        if digital is None or len(twin) < 2:
            continue
        for document in twin:
            if (document is digital or document.clean_ocr_pages is not None
                    or len(document.ocr_pages) != len(digital.ocr_pages)):
                continue
            document.clean_ocr_pages = list(digital.clean_ocr_pages or digital.ocr_pages)
            if document.golden_label == digital.golden_label:
                document._verified = _verified(digital)
            document._twin_shared = True
            shared += 1
    return shared


def _verified(document: SourceDocument) -> tuple[dict[str, Any], VerificationReport]:
    """``(label, report)``: the document's label checked on its clean text, once."""
    cached = getattr(document, "_verified", None)
    if cached is None:
        checked = VerificationReport()
        clean = document.clean_ocr_pages if document.clean_ocr_pages is not None else document.ocr_pages
        cached = (verified_label(document.golden_label, clean, checked, listed=True), checked)
        document._verified = cached
    return cached


def _count_kept(document: SourceDocument, counts: Counter, result: BuildResult) -> None:
    """Add a kept document's target counts to the build's (:data:`TARGET_COUNTS`)."""
    prepared = getattr(document, "_prepared_targets", None)
    if counts["halves.windows_split"]:
        # Its overflowing window set it aside whole before.
        counts["halves.documents_saved"] += 1
    placement = prepared.placement if prepared is not None else None
    # Placed by search only on a common-model line (places_by_search), whose
    # decoder refused a value citing no page - and with it the whole document.
    if placement is not None and (placement.placed or counts["unpaged.windows_dropped"]):
        counts["unpaged.documents_saved"] += 1
    result.target_counts.update(counts)
    result.cap_report.windows_halved += counts["halves.windows_split"]
    result.cap_report.rows_from_halves += counts["halves.rows"]


def stamp_carrier_seen(rows_by_split: dict[str, list[dict[str, Any]]]) -> None:
    """``carrier_seen_in_train`` on every val and test row: whether its carrier
    is among the carriers of the train rows (review item 42), so a report can
    tell a layout the model was trained on from one it never saw. A row with no
    carrier has not been seen."""
    from common.normalize import normalize_carrier

    def key(carrier: Any) -> str | None:
        return normalize_carrier(carrier) if isinstance(carrier, str) and carrier.strip() else None

    seen = {key(row.get("carrier")) for row in rows_by_split.get("train", [])} - {None}
    for split in ("val", "test"):
        for row in rows_by_split.get(split, []):
            carrier = key(row.get("carrier"))
            row["carrier_seen_in_train"] = carrier is not None and carrier in seen


def _refusal(
    source_id: str, rows: list[dict[str, Any]], report: CapReport
) -> tuple[str, str] | None:
    """``(category, why)`` this document cannot enter the corpus, or ``None``.

    Records the budget verdict on ``report``. Each row's estimate is computed
    once and reused for the verdict and the record.

    Two reasons, both judged over EVERY row before anything is recorded, because
    the document is the unit that is kept or lost:

    * **Text the trainer would parse as a tag.** ms-swift reads ``<image>``,
      ``<video>`` and the rest as special tags wherever they appear, so OCR or a
      target containing one would be rewritten and train on a prompt serving
      never sends. Checked here rather than at staging, so one bad page is set
      aside and reported at build instead of aborting a paid training run.
    * **A row over its task budget** — reject, never truncate.
    """
    for row in rows:
        for text in _texts(row):
            tag = next((t for t in TRAINER_SPECIAL_TAGS if t in text), None)
            if tag:
                return "trainer_tag", (
                    f"its text contains {tag!r}, which the trainer parses as a special tag — "
                    "the prompt it would train on is not the one serving sends"
                )

    estimates = [_estimate(row) for row in rows]
    verdicts = [evaluate(estimate) for estimate in estimates]
    failed = [(e, v) for e, v in zip(estimates, verdicts, strict=True) if not v[0]]
    if failed:
        estimate, (_fits, cap, reason) = failed[0]
        report.reject(source_id, estimate.task, estimate, cap, reason=reason)
        report.documents_rejected += 1
        return "budget", f"{len(failed)} of {len(rows)} row(s) exceed their task budget: {reason}"
    for estimate in estimates:
        report.record(estimate.task, estimate)
    report.documents_accepted += 1
    return None


def _texts(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for message in row["messages"]:
        content = message["content"]
        if isinstance(content, str):
            out.append(content)
        else:
            out += [b.get("text", "") for b in content if b.get("type") == "text"]
    return out


def _estimate(row: dict[str, Any]):
    """One built row's estimated cost against its task budget."""
    messages = row["messages"]
    user = messages[1]["content"] if len(messages) > 1 else []
    blocks = user if isinstance(user, list) else []
    return estimate_row(
        # A flat row is the whole-document `extract` task; a window names its own.
        task=row.get("task") or "extract",
        system_prompt=messages[0]["content"],
        ocr_pages=[b.get("text", "") for b in blocks if b.get("type") == "text"],
        page_count=sum(1 for b in blocks if b.get("type") == "image"),
        target_json=messages[-1]["content"] if messages[-1]["role"] == "assistant" else "",
        doc_type=row.get("doc_type"),
    )


def train_source_ids(
    documents: list[SourceDocument], assignment: GroupSplitAssignment
) -> list[str]:
    """The documents that will be sampled into epochs.

    Drawing modes for val and test documents too would not change their rows —
    they take all three regimes — but it would put their draws into the realised
    mix that the mix check reads, measuring something other than what trains.
    """
    found = []
    for document in documents:
        try:
            if assignment.split_of(document.family) == "train":
                found.append(document.source_id)
        except Exception:  # noqa: BLE001 - unassigned documents are reported by build_corpus
            continue
    return found


def train_rows_by_epoch(
    result: BuildResult, epochs: int = EPOCH_FILES
) -> dict[int, list[dict[str, Any]]]:
    """Split the train rows into one list per epoch file.

    Every epoch file must hold every train document, and no row twice. A
    document missing from an epoch, or a row present twice, means the run trains
    on something other than what the manifest's epoch count says, so that is
    refused here rather than discovered in a loss curve.

    A canonical policy is several windows, so it is several rows in every epoch.
    Its windows can differ between epochs - each epoch draws the document in its
    own input mode, and an image-only row routes every page where an OCR row
    routes by keyword - so what must hold across epochs is the set of DOCUMENTS,
    not the set of rows.
    """
    by_epoch: dict[int, list[dict[str, Any]]] = {e: [] for e in range(1, epochs + 1)}
    for row in result.rows_by_split.get("train", []):
        epoch = row.get("epoch")
        if epoch not in by_epoch:
            raise CorpusBuildError(
                f"train row for {row.get('source_id')!r} has epoch {epoch!r}; expected 1-{epochs}. "
                "Every train row belongs to exactly one epoch file (arch v2.1 §6.1)."
            )
        by_epoch[epoch].append(row)

    expected = sorted({row["source_id"] for rows in by_epoch.values() for row in rows})
    for epoch, rows in by_epoch.items():
        keys = [_row_identity(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise CorpusBuildError(f"epoch {epoch} holds a row twice.")
        ids = sorted({row["source_id"] for row in rows})
        if ids != expected:
            raise CorpusBuildError(
                f"epoch {epoch} does not hold every train document "
                f"({len(ids)} documents for {len(expected)} expected)."
            )
    return by_epoch


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, int, str]:
    """One row: its document, input mode, schema slice, window, and which half
    of it (``window_half``, "" for a window read whole)."""
    return (
        row["source_id"], row["modality_mode"],
        row.get("sections") or "", int(row.get("window_index") or 0),
        str(row.get("window_half") or ""),
    )


def write_jsonl(rows: list[dict[str, Any]]) -> str:
    """Serialise rows to JSONL — one complete JSON object per line.

    Sorted by ``(source_id, modality_mode, sections, window_index)`` so a rebuild
    with the same seed is byte-identical. A corpus that changes between builds
    cannot be compared across model versions.
    """
    ordered = sorted(rows, key=_row_identity)
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
