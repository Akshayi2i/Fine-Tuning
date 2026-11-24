"""Stage 3b: a policy's answer held to the conventions and to what its pages print
(``serving.pipeline.extract``, review A2)."""

from __future__ import annotations

from typing import Any

from serving import pipeline as P
from serving.stages.release import Chosen


def hold_policy_answer(request: Any, chosen: Chosen, extraction: Any, all_spans: Any,
                       merge_flags: list[str]) -> tuple[Any, Any, list[str], dict[str, Any] | None]:
    """``(extraction, spans, merge flags, printed-values record)`` after both checks."""
    lob = chosen.lob
    if chosen.common_model:
        # The conventions the training labels follow, applied to the merged
        # answer against every page's text (serving.conventions): each value
        # once, citing every page that prints it; no extra field repeating a
        # field; a value no page prints flagged.
        from contracts.modality import NATIVE
        from serving.conventions import conform

        page_texts = None if request.modality_mode == "image_only" else (
            dict(request.page_texts) or dict(enumerate(P._all_page_texts(request), start=1)))
        conformed = conform(extraction, all_spans, page_texts,
                            native=(request.ocr_meta or {}).get("modality") == NATIVE)
        extraction, all_spans = conformed.extraction, conformed.spans
        merge_flags.extend(conformed.flags)
        if page_texts:
            merge_flags.extend(P._likely_missed(extraction, page_texts, lob))
    # Printed values only (decision 2026-10-24): a value the answer holds
    # that no page prints is taken out of the JSON and listed for review
    # (serving.printed_values). With no page text (image only) nothing can be
    # checked yet: fields first checks it when MinerU's text arrives. The
    # text is the request's printed_texts when it carries one (the golden
    # eval's noisy mode), else the text the model read.
    from serving.printed_values import hold_to_printed

    checked_texts = P._printed_texts(request)
    if chosen.common_model and checked_texts:
        # Right value, wrong role: a declarations label printing another value
        # than the one served under it. Before the check, so its flags follow
        # their values (held.joined).
        merge_flags.extend(P._role_mismatches(extraction, checked_texts, lob))
    held = hold_to_printed(extraction, all_spans, checked_texts, lob=lob)
    extraction, all_spans, printed_check = held.extraction, held.spans, held.record
    # Flags raised before it follow their values; a value taken out keeps one.
    merge_flags = held.joined(merge_flags)
    return extraction, all_spans, merge_flags, printed_check
