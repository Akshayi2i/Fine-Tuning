"""Stage 4: each field's calibrated confidence (``serving.pipeline.extract``, review A2).

One span map for the answer returned, on both paths: one interleaved call, or windows whose
spans the merge re-keyed - never one call per page, so the spans describe exactly the
document that was returned. The previous per-page loop left `all_spans` holding only the last
page's map while `extraction` was the merged document, so fields merged from earlier pages
carried no confidence and a field the declarations page won still reported the rejected
page's value.
"""

from __future__ import annotations

from typing import Any

from serving import pipeline as P
from serving.stages.answer import Answer
from serving.stages.release import Chosen


def score_confidence(request: Any, chosen: Chosen, answer: Answer, settings: Any) -> Any:
    """A ``CalibratedResult`` for ``answer``."""
    extraction, all_spans, pages_used = answer.extraction, answer.spans, answer.pages_used
    calibrators, thresholds, common_model = chosen.calibrators, chosen.thresholds, chosen.common_model
    # The release bundle's calibrator set when the endpoint holds one, the v1
    # transform otherwise. Not a silent fallback: a release whose calibrate stage
    # never ran has no per-field-type curves and no measured thresholds, and
    # serving it through the v1 path is a knowing downgrade rather than an
    # equivalent.
    if calibrators is not None:
        calibrated = P._feature_calibrated(
            # The model form, envelopes and all: flattened it gives the same
            # values, and it still carries each value's printed form. A
            # common-model line's ids and codes are no reading, and not scored.
            extraction=P.without_bare_values(extraction) if common_model else extraction,
            spans=all_spans,
            calibrators=calibrators, thresholds=thresholds,
            # The same definition validation fits the calibrators with: the OCR
            # text of the pages the model was shown, none for image_only.
            page_text=None if request.modality_mode == "image_only" else P.shown_ocr_text(
                [request.page_texts.get(page) for page in pages_used]
                if request.page_texts else [request.ocr_text]
            ),
            common_model=common_model,
            # Each page's own text, so a value is grounded on the pages it cites.
            page_texts=None if request.modality_mode == "image_only" else (
                {int(p): t for p, t in request.page_texts.items()} or None),
            modality_mode=request.modality_mode,
        )
    else:
        P.log.warning(
            "no calibrator set supplied for %s, so confidence falls back to the v1 raw "
            "aggregate, and EVERY field is flagged for review: that number is length-biased "
            "and tied to no measured error rate (arch v2.1 §5.1) — load the release bundle's "
            "calibrators to get the guarantee.", request.source_id,
        )
        raw = P.field_confidences(all_spans)
        calibrated = P.apply_calibration(raw, chosen.calibration, review_threshold=settings.review_threshold)
        # No measured threshold, no promise. Accepting a field above a fixed 0.70
        # claimed an error rate nobody measured; the confidence is still reported,
        # as a diagnostic, but nothing is accepted on it.
        for path, f in calibrated.fields.items():
            if not f.needs_review:
                calibrated.fields[path] = P.replace(
                    f, needs_review=True, reason="no fitted calibrator; nothing is auto-accepted"
                )
                calibrated.review_flags.append(f"{path}:uncalibrated")
    return calibrated
