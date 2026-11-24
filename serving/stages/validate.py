"""Stage 7: schema validation, mirroring the audit gate, then every key of the line's schema
(``serving.pipeline.extract``, review A2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from serving import pipeline as P
from serving.stages.answer import Answer
from serving.stages.output import Served
from serving.stages.release import Chosen


@dataclass
class Checked:
    output: Any
    schema_valid: bool
    validation_errors: list[str] = field(default_factory=list)
    #: SPEC_21's every-word rule, measured on what is served.
    coverage: Any = None


def validate_output(request: Any, chosen: Chosen, answer: Answer, served: Served, settings: Any,
                    flags: list[str] | None = None) -> Checked:
    """``served.output`` judged by the client's schema; ``flags`` (the review flags) gets
    ``schema:invalid`` when it fails and ``strict_schema`` is off."""
    route_, lob, common_model = chosen.route, chosen.lob, chosen.common_model
    # Against the client's FULL schema for a canonical document, envelope and
    # all — the model form it was generated in is ours, the contract is theirs.
    verdict = P.schema_verdict(
        answer.extraction, served.output, doc_type=route_.schema_doc_type,
        acord_form=route_.schema_acord_form, lob=lob, common_model=common_model,
    )
    output, filled = verdict.output, verdict.filled
    schema_valid = verdict.valid
    validation_errors = list(verdict.errors)
    if not schema_valid:
        if settings.strict_schema:
            raise P.PipelineError(
                f"the extraction for {request.source_id} failed schema validation and will not be "
                f"returned (Fideon SPEC_07 Stage 3 mirrors this on every production call): "
                + "; ".join(validation_errors[:4])
            )
        if flags is not None:
            flags.append("schema:invalid")

    if chosen.canonical:
        # Every key of the line's schema, whatever the model found: a value it
        # did not extract is null, not absent, so every served JSON - from the
        # base model or a trained one - has the same keys (common.canonical.
        # with_all_keys). After validation, which judges the model's own answer
        # (an answer missing a required section must still fail; on a
        # common-model line, the filled answer with its sections checked as
        # written, above), and after confidence, so the nulls carry none of it.
        output = filled if filled is not None else P._with_every_key(output, route_, lob)

    coverage = P.text_coverage(served.full_text_pages, output) if chosen.canonical and common_model else None
    return Checked(output=output, schema_valid=schema_valid, validation_errors=validation_errors,
                   coverage=coverage)
