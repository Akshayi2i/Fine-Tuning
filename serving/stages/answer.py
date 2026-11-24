"""Stage 3: the model's answer (``serving.pipeline.extract``, review A2).

Every page of every document is read (decision 2026-10-19): a policy in windows, a long
Loss Run in windows, anything else in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from serving import pipeline as P
from serving.stages.release import Chosen


@dataclass
class Answer:
    extraction: Any
    spans: Any
    latency: float | None
    pages_used: list[int]
    #: Review flags raised while reading, merging and checking the answer.
    merge_flags: list[str] = field(default_factory=list)
    reconciliation: dict[str, Any] | None = None
    window_latencies: list[float] = field(default_factory=list)
    window_stats: list[dict[str, Any]] = field(default_factory=list)
    printed_check: dict[str, Any] | None = None
    #: Calibration, completeness and the per-field output all read VALUES. For a
    #: canonical document that is each envelope's `parsed`; a flat document is
    #: its own values view. The spans were re-keyed to the same paths.
    values: Any = None


def read_answer(request: Any, model: Any, chosen: Chosen) -> Answer:
    from serving.stages.printed import hold_policy_answer

    route_, lob = chosen.route, chosen.lob
    latency: float | None = None
    merge_flags: list[str] = []
    reconciliation: dict[str, Any] | None = None
    window_latencies: list[float] = []
    window_stats: list[dict[str, Any]] = []
    printed_check: dict[str, Any] | None = None
    windows = P.lossrun_windows(request) if route_.schema_doc_type == "lossrun" else []

    if chosen.canonical and route_.schema_doc_type == "policy":
        # Every canonical policy is read as windows — section group x page
        # window — planned by the SAME function the corpus build expanded its
        # training rows with, so a served window is a shape a training row had.
        # Always, whatever the length: a threshold computed from prompt length
        # would move between corpus builds and re-shape documents silently.
        extraction, all_spans, latency, pages_used, merge_flags = P._extract_policy_windows(
            request, model, route_, lob, latencies=window_latencies, stats=window_stats,
        )
        extraction, all_spans, merge_flags, printed_check = hold_policy_answer(
            request, chosen, extraction, all_spans, merge_flags)
    elif len(windows) > 1:
        # A Loss Run longer than one window: read by window, merged and
        # reconciled. One that fits a window is read in one call below.
        extraction, all_spans, latency, pages_used, merge_flags, reconciliation = (
            P._extract_lossrun_windows(request, model, route_, windows, latencies=window_latencies))
    else:
        # Derived from the images actually sent, not from page_texts: an
        # image_only document has no page_texts, and defaulting to [1] made
        # check_document sum table_row_counts for page 1 alone — flagging every
        # multi-page image_only extraction as row-incomplete.
        pages_used = (
            sorted(request.page_texts) if request.page_texts
            else list(range(1, len(request.image_paths) + 1))
        )
        page_ocr = None if request.modality_mode == "image_only" else P._all_page_texts(request)
        # Images by page, in the order the texts are in. Sent as the caller
        # listed them, `page_10.png` listed before `page_2.png` put page 10's
        # image beside page 2's text.
        extraction, all_spans, latency = P._generate_once(
            model, route_, [P._image_for(request, page) for page in pages_used],
            page_ocr, request.modality_mode,
            page_numbers=pages_used if request.page_texts else None,
            lob=lob,
        )

    if not window_latencies and latency is not None:
        window_latencies.append(latency)                  # read in one call
    if chosen.common_model:
        # A coverage whose unit was read in another window comes back unlinked.
        merge_flags.extend(P.unit_link_flags(extraction))
    if route_.schema_doc_type == "lossrun":
        if reconciliation is None:
            from serving.lossrun_merge import reconcile_extraction

            reconciliation = reconcile_extraction(extraction).as_dict()
        # A mismatch is a claims list that does not add up to what the document
        # prints. Unverifiable (nothing printed to check against) is reported,
        # not flagged: the Loss Run schema carries no printed totals yet.
        if reconciliation.get("status") == "mismatch":
            merge_flags.append(P.LOSSRUN_TOTALS_MISMATCH_FLAG)
    return Answer(extraction=extraction, spans=all_spans, latency=latency, pages_used=pages_used,
                  merge_flags=merge_flags, reconciliation=reconciliation, window_latencies=window_latencies,
                  window_stats=window_stats, printed_check=printed_check, values=P.values_view(extraction))
