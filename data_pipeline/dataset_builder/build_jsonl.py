"""Compile documents into chat-format JSONL training rows (IMPL-05, arch §6, §7).

Split first, expand second, inside whichever split the document was already
assigned to:

* **train** — one row per epoch, each in that epoch's sampled modality regime
  (arch v2.1 §6.1), written as ``train/epoch_1..4.jsonl``;
* **val / test** — three rows, one per regime, so image-only and noisy-OCR
  accuracy are measured on the full eval population.

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

from common.canonical import CanonicalLabelError, has_envelopes, training_target
from common.constants import MODALITY_MODES, TRAINER_SPECIAL_TAGS
from common.schemas import is_canonical
from data_pipeline.dataset_builder.cap_check import CapReport, estimate_row, evaluate
from data_pipeline.dataset_builder.label_verification import VerificationReport, verified_label
from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
from data_pipeline.dataset_builder.policy_windows import (
    TargetReport,
    multi_window_sections,
    pages_to_read,
    plan_windows,
    printed_pages_in,
    unread_values,
    window_target,
    with_inferred_pages,
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
    #: Documents set aside, by reason: ``budget``, ``trainer_tag``, ``expansion``.
    #: Every document that does not reach the corpus is counted here, so a stage
    #: summary can say how many were lost and why rather than a log line alone.
    set_aside: Counter = field(default_factory=Counter)
    #: Every row's estimated size against its task budget, and every document
    #: set aside because one of its rows did not fit (``cap_check``).
    cap_report: CapReport = field(default_factory=CapReport)

    @property
    def all_rows(self) -> list[dict[str, Any]]:
        return [row for rows in self.rows_by_split.values() for row in rows]

    def summary(self) -> str:
        counts = {split: len(rows) for split, rows in sorted(self.rows_by_split.items())}
        return f"{len(self.all_rows)} rows {counts}, modality {dict(self.modality_counts)}"


def expand_document(
    document: SourceDocument,
    split: str,
    *,
    seed: int = 42,
    modes: tuple[str, ...] = MODALITY_MODES,
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
    """
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
                document, split, mode, ocr_pages, details, seed=seed
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

    return rows, details


def _policy_window_rows(
    document: SourceDocument,
    split: str,
    mode: str,
    ocr_pages: list[str] | None,
    details: list[str],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """One row per window, planned exactly as serving plans them.

    Every page is read; the declarations page is found, and each window sized,
    from the OCR text THIS row carries - clean or none - as serving does for the
    same input. A noisy row plans on the clean text: its corruption is drawn per
    window, after the plan, and keeps each page's length.
    """
    page_count = len(document.image_paths)
    every_page, declarations_page = pages_to_read(ocr_pages, page_count)
    plans = plan_windows(document.lob, every_page, declarations_page, ocr_pages)

    report = TargetReport()
    # A value a rule added to a synthetic label is taught only if the document's
    # own text prints it (label_verification). From the CLEAN text, whatever
    # this row's mode: the target is the same answer in every mode, only the
    # input differs.
    # Once per document, not once per reading mode: it depends on neither.
    cached = getattr(document, "_verified", None)
    if cached is None:
        checked = VerificationReport()
        cached = (verified_label(document.golden_label, document.ocr_pages, checked), checked)
        document._verified = cached
    label, checked = cached
    # Only in sections read over several windows: one window keeps a value
    # that records no page as it is (policy_windows.with_inferred_pages).
    multi = multi_window_sections(document.lob, plans)
    label = with_inferred_pages(label, document.ocr_pages, report, sections=multi)
    # Where the clean text prints a coverage's form numbers, which carry no page
    # of their own: they are taught on the windows that show them.
    printed = getattr(document, "_printed_pages", None)
    if printed is None and document.ocr_pages:
        printed = document._printed_pages = printed_pages_in(document.ocr_pages)
    unprinted = _unprinted_values(label, document.unprinted_values)
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
            continue
        target = window_target(label, document.lob, plan, report, printed_pages=printed)
        indices = [page - 1 for page in plan.pages]
        window_ocr = None if ocr_pages is None else [ocr_pages[i] for i in indices]
        if window_ocr is not None and mode == "noisy_ocr_image":
            # The document-level budget, applied to the pages this row carries.
            # Seeded per window, so a rebuild is byte-identical.
            window_ocr, window_details = corrupt_ocr_pages(
                window_ocr, f"{document.source_id}#{plan.group}:{plan.window_index}", seed=seed
            )
            details.extend(
                f"window {plan.group}:{plan.window_index} {d}" for d in window_details
            )
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
            split=split,
            deidentified=False,
            # The document's own page numbers, never window-relative: the
            # target's page_ref is the document's, and so are the markers.
            page_numbers=list(plan.pages),
            total_pages=page_count,
        )
        row["task"] = plan.task
        row["window_index"] = plan.window_index
        row["window_pages"] = list(plan.pages)
        rows.append(row)

    notes = [f"window {mode}: unread {path}" for path in unread_values(
        label, document.lob, plans
    )]
    notes += [f"window {mode}: unplaced {path}" for path in report.unplaced]
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
    if checked.dropped or checked.repaged or checked.unverifiable:
        details.append(
            f"window {mode}: of {checked.checked} rule-added label value(s), "
            f"{len(checked.dropped)} not printed on the page and left out, "
            f"{len(checked.repaged)} moved to the page that prints them, "
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
    ``coverages[2].limit``); a path that leads nowhere is skipped."""
    import copy
    import re

    out = copy.deepcopy(label)
    for path in paths:
        steps = [int(index) if index else key for key, index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", path)]
        node: Any = out
        try:
            for step in steps[:-1]:
                node = node[step]
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(node, dict):
            node.pop(steps[-1], None)
    return out


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
    """The family adapter's line question for one policy: one row per mode.

    Stage 2 of line detection (serving.doc_type_classifier): once the base model
    has placed a policy in a layout family, the family's own adapter chooses
    among the family's lines - where the confusable pairs are (homeowners and
    dwelling fire; personal auto, motorcycle and RV). The messages are the
    classifier's own (``classifier_messages``) over the input serving picks
    (``classifier_input``: the first two pages' images, their text joined, none
    under ``image_only``) - the text in the row's own mode, corrupted under
    ``noisy_ocr_image`` as a window's is, so the input trained is the input
    served. Serving does not ask this question yet (``build_corpus`` refuses
    the rows until it does). Nothing for a document that is no policy, names
    several lines, or has a line in no family.
    """
    from common.config import lob_to_layout_family, lobs_in_family
    from serving.doc_type_classifier import (
        CLASSIFIER_PAGES,
        classifier_input,
        classifier_messages,
        known_line,
    )

    lines = document.lob if isinstance(document.lob, list) else [document.lob]
    line = known_line(lines[0]) if document.doc_type == "policy" and len(lines) == 1 else None
    family = lob_to_layout_family().get(line) if line else None
    if family is None:
        return []
    choices = list(lobs_in_family(family))
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

    ``with_classify_rows`` would add, per policy whose line has a layout
    family, the family adapter's line question (:func:`classify_rows`). It is
    refused: those rows train the family stage's question, which serving does
    not ask yet - its classifier asks every document the type-and-line
    question - so the rows would teach an answer nothing reads. The family
    stage at serving and these rows are turned on together, once the base
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
            "with_classify_rows trains the family stage's line question, which serving does not "
            "ask yet: its classifier asks every document the type-and-line question. Add the "
            "family stage to serving and turn both on together, or the rows teach an answer "
            "nothing reads."
        )
    result = BuildResult(rows_by_split={"train": [], "val": [], "test": []})
    common_model_failures: list[tuple[str, str]] = []
    if mode_assignment is None:
        mode_assignment = sample_modes(train_source_ids(documents, assignment), seed=seed)

    for document in documents:
        try:
            split = assignment.split_of(document.family)
        except Exception as exc:
            result.skipped.append((document.source_id, str(exc)))
            continue

        # Train draws ONE mode per epoch (§6.1); val and test keep all three so
        # image-only and noisy-OCR accuracy are measured on the full eval
        # population rather than a sample of it.
        if split == "train":
            epoch_modes = tuple(
                mode_assignment.mode_for(document.source_id, e)
                for e in range(1, mode_assignment.epochs + 1)
            )
        else:
            epoch_modes = MODALITY_MODES

        try:
            rows, details = expand_document(document, split, seed=seed, modes=epoch_modes)
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

        if with_classify_rows:  # refused above until serving asks the family question
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
            row["synthetic"] = document.synthetic
            # Fideon SPEC_09 amendment item 8: carrier and twin index on every
            # row (the seed is group_id). twin_index is None for real documents.
            row["carrier"] = document.carrier
            row["twin_index"] = document.twin_index
            # Which render of its twin it is (digital / scanned): a scope's view
            # trains one render of a twin per epoch (training.corpus_view).
            row["render_mode"] = document.render_mode
            # What a scope's scanned_share counts (training.data_mix), and the
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

    # The checks that make the ordering rule real rather than documented.
    assert_no_leakage(assignment, result.all_rows)
    if not assignment.delivered:
        # A delivered split placed synthetic twins of held-out sources in val and
        # test on purpose (assign_delivered_splits); the gate reports real ones apart.
        assert_synthetic_is_train_only(result.all_rows)
    assert_single_tenant(result.all_rows)

    log.info("corpus built: %s", result.summary())
    return result


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


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """One row: its document, input mode, schema slice and window."""
    return (
        row["source_id"], row["modality_mode"],
        row.get("sections") or "", int(row.get("window_index") or 0),
    )


def write_jsonl(rows: list[dict[str, Any]]) -> str:
    """Serialise rows to JSONL — one complete JSON object per line.

    Sorted by ``(source_id, modality_mode, sections, window_index)`` so a rebuild
    with the same seed is byte-identical. A corpus that changes between builds
    cannot be compared across model versions.
    """
    ordered = sorted(rows, key=_row_identity)
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
