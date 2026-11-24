"""Stage 5: list completeness - the signal logprobs cannot see - and the review flags
(``serving.pipeline.extract``, review A2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from serving import pipeline as P
from serving.stages.answer import Answer
from serving.stages.release import Chosen


@dataclass
class Reviewed:
    #: The answer's values view (taken where extract took it, before confidence).
    values: Any
    completeness: Any
    flags: list[str]


def review_document(request: Any, chosen: Chosen, answer: Answer, calibrated: Any) -> Reviewed:
    values = answer.values
    completeness = P.check_document(
        values, chosen.route.schema_doc_type,
        ocr_meta=request.ocr_meta, pages_used=answer.pages_used,
    )
    # A value whose tokens all but chose another (serving.value_flags): on the
    # spans as served, after the merge and the printed-values check.
    answer.merge_flags.extend(P.near_tie_flags(answer.spans))
    flags = P.merge_review_flags(
        completeness, calibrated.review_flags + chosen.route.review_flags + answer.merge_flags
        + ([P.LOB_FALLBACK_FLAG] if chosen.lob_fallback_used else [])
    )
    return Reviewed(values=values, completeness=completeness, flags=flags)
