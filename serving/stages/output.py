"""Stage 6: the client's canonical envelope, the full-text tier, and no key the client's
schema does not name (``serving.pipeline.extract``, review A2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from serving import pipeline as P
from serving.stages.answer import Answer
from serving.stages.release import Chosen
from serving.stages.review import Reviewed


@dataclass
class Served:
    output: Any
    full_text_pages: Any
    unknown_keys: list[str] = field(default_factory=list)


def served_output(request: Any, chosen: Chosen, answer: Answer, calibrated: Any, reviewed: Reviewed) -> Served:
    """The JSON served before validation; review flags it raises join ``reviewed.flags``."""
    route_, lob, extraction, flags = chosen.route, chosen.lob, answer.extraction, reviewed.flags
    full_text_pages = P._full_text_pages(request)

    # confidence and flagged are filled here, from the calibrated fields, never
    # by the model. A list whose row count disagrees with the document is
    # flagged on every value in it: the rows present may each be confident, and
    # the list as a whole still needs a person.
    if chosen.canonical:
        # Values that cannot all be true together (serving.consistency): checked on
        # the merged answer, so a date from the declarations and one from a later
        # window are compared, and routed to review below with the other value flags.
        flags.extend(P.consistency_flags(extraction))
        incomplete = tuple(
            flag.split(":", 1)[0] for flag in flags if P.is_row_completeness_flag(flag)
        )
        # Values a flag sends to review though they are served: printed only off
        # the pages they cite, a near tie, a label printing another value
        # (serving.value_flags).
        routed = P.routed_paths(flags)
        scores = {
            path: (
                f.confidence,
                f.needs_review or path in routed or any(
                    path == name or path.startswith((f"{name}[", f"{name}."))
                    for name in incomplete
                ),
            )
            for path, f in calibrated.fields.items()
        }
        output = P.with_system_fields(
            P.envelope(extraction, scores),
            page_count=len(request.image_paths) or len(request.page_texts) or None,
            source_file_name=request.source_file_name,
            lob=lob, modality=(request.ocr_meta or {}).get("modality"),
        )
        if chosen.common_model:
            # The full-text tier: every printed word by form and section, built
            # from the page text by Fideon's section builder, never by the model
            # (SPEC_21 §4.6, common.full_text). Before validation, so the client's
            # schema judges it too.
            text_sections = P.build_text_sections(full_text_pages)
            if text_sections:
                output["text_sections"] = text_sections
            else:
                flags.append(P.NO_FULL_TEXT_FLAG)
    else:
        output = extraction

    # Decoding is held to a closed schema, so the model cannot write one - but
    # that rests on the server's grammar alone, and a backend without it (or a
    # bug) would serve an invented key flagged schema:invalid at most
    # (fields first serves with strict_schema off). Taken out here, whatever
    # the backend, and said: nothing is served that the schema has no place for.
    unknown_keys: list[str] = []
    if chosen.canonical and isinstance(output, dict):
        output, unknown_keys = P._without_unknown_keys(output, route_, lob)
        flags.extend(f"{path}:{P.UNKNOWN_KEY_FLAG}" for path in unknown_keys)
    return Served(output=output, full_text_pages=full_text_pages, unknown_keys=unknown_keys)
