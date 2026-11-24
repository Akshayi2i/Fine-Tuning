"""Stage 8: the result served (``serving.pipeline.extract``, review A2)."""

from __future__ import annotations

from typing import Any

from serving import pipeline as P
from serving.stages.answer import Answer
from serving.stages.output import Served
from serving.stages.release import Chosen
from serving.stages.review import Reviewed
from serving.stages.route import Routed
from serving.stages.validate import Checked


def assemble(routed: Routed, model: Any, chosen: Chosen, answer: Answer, calibrated: Any, reviewed: Reviewed,
             served: Served, checked: Checked) -> Any:
    """The ``ExtractionResult``, logged."""
    request, route_, resolution = routed.request, chosen.route, routed.resolution
    unknown_keys, completeness, values = served.unknown_keys, reviewed.completeness, reviewed.values
    calibrators, thresholds = chosen.calibrators, chosen.thresholds
    result = P.ExtractionResult(
        source_id=request.source_id,
        doc_type=route_.schema_doc_type,
        model_version=model.tag,
        mode=request.modality_mode,
        schema_valid=checked.schema_valid,
        overall_confidence=calibrated.overall_confidence,
        extraction=checked.output,
        fields={
            path: f.as_output()
            for path, f in sorted(calibrated.fields.items())
            if path != "line_of_business" and "[" not in path
            and not any(path == gone or path.startswith(f"{gone}.") for gone in unknown_keys)
        },
        list_fields={
            name: {"rows": values.get(name, []), **signal.as_output()}
            for name, signal in sorted(completeness.items())
        },
        line_of_business=(resolution.as_output() if resolution.source is not None
                          else P._lob_output(calibrated.fields)),
        pages_used=answer.pages_used,
        review_flags=sorted(set(reviewed.flags)),
        route_info={
            "adapter": route_.adapter or None,
            "foundation_only": route_.foundation_only,
            "classifier_confidence": (
                route_.classification.confidence if route_.classification else None
            ),
            "classifier_method": (
                route_.classification.method if route_.classification else None
            ),
            "lob_fallback_used": chosen.lob_fallback_used,

            "lob_source": resolution.source,
        },
        latency_ms=answer.latency,
        validation_errors=checked.validation_errors,
        reconciliation=answer.reconciliation,
        window_latencies_ms=answer.window_latencies,
        text_coverage=checked.coverage,
        window_stats=answer.window_stats,
        printed_check=answer.printed_check,
        confidence_features=dict(calibrated.features),
        calibrated_with=P.CalibratedWith(calibrators, thresholds) if calibrators is not None else None,
    )

    P.log.info(
        "extracted %s as %s (confidence %.3f, %d review flag(s), adapter=%s)",
        request.source_id, result.doc_type, result.overall_confidence,
        len(result.review_flags), route_.adapter or "foundation-only",
    )
    return result
