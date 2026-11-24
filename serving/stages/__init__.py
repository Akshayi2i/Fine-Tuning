"""One document read through the pipeline, stage by stage (review A2, 2026-11-24).

:func:`serving.pipeline.extract` was one 455-line function. Its stages are functions now, each
with a typed input and output, in the order it ran them:

1. :func:`.route.route_document` - the document type, its route and its line;
2. :func:`.release.choose_release` - the release, its adapter and calibrators, the schema;
3. :func:`.answer.read_answer` - the model's answer: a policy in windows, a long Loss Run in
   windows, anything else in one call; for a policy, :func:`.printed.hold_policy_answer` -
   the conventions and the printed-values check;
4. :func:`.confidence.score_confidence` - calibrated confidence;
5. :func:`.review.review_document` - list completeness and the review flags;
6. :func:`.output.served_output` - the client's envelope, the full-text tier, no unknown key;
7. :func:`.validate.validate_output` - schema validation and every key of the schema;
8. :func:`.result.assemble` - the result served.

Moved, not rewritten: each stage's body is the block of ``extract`` it was, and it reaches
the pipeline's helpers through the module (``serving.pipeline.<name>``) as ``extract`` did,
so a test patching one still reaches the reading.
"""

from __future__ import annotations

from typing import Any


def read(request: Any, model: Any, classifier: Any, calibration: Any, settings: Any) -> Any:
    """``request`` read through every stage; an ``ExtractionResult``."""
    from serving.stages.answer import read_answer
    from serving.stages.confidence import score_confidence
    from serving.stages.output import served_output
    from serving.stages.release import choose_release
    from serving.stages.result import assemble
    from serving.stages.review import review_document
    from serving.stages.route import route_document
    from serving.stages.validate import validate_output

    routed = route_document(request, classifier, settings)
    chosen = choose_release(routed, model, calibration, settings)
    answer = read_answer(routed.request, model, chosen)
    calibrated = score_confidence(routed.request, chosen, answer, settings)
    reviewed = review_document(routed.request, chosen, answer, calibrated)
    served = served_output(routed.request, chosen, answer, calibrated, reviewed)
    checked = validate_output(routed.request, chosen, answer, served, settings, flags=reviewed.flags)
    return assemble(routed, model, chosen, answer, calibrated, reviewed, served, checked)
