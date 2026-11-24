"""Stage 1: the document's type, its route and its line (``serving.pipeline.extract``, review A2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from serving import pipeline as P


@dataclass
class Routed:
    #: The request with its pages numbered, as every later stage reads it.
    request: Any
    route: Any
    #: Which line the policy is read as, and where that came from.
    resolution: Any


def route_document(request: Any, classifier: Any, settings: Any) -> Routed:
    request = P._numbered_pages(request)
    route_ = P._resolve_route(
        request, classifier,
        confidence_threshold=settings.classifier_threshold,
        adapter_map=settings.adapter_map or {},
        fallback_doc_type=settings.fallback_doc_type,
    )

    # The policy's line: the caller's, else the classifier's (or none).
    resolution = P._resolve_lob(
        request, classifier, route_, detect=settings.detect_lob,
        line_threshold=settings.lob_confidence_threshold, uncertain_threshold=settings.lob_uncertain_threshold,
    )
    if resolution.flags:
        route_ = P.replace(route_, review_flags=[*route_.review_flags, *resolution.flags])
    return Routed(request, route_, resolution)
