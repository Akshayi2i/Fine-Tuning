"""Confidence once the page text arrives (fields first).

Production reads a policy's fields from the page images while MinerU reads its
text (``serving.fields_first``): the fields are calibrated with no page text,
so text agreement - is the value printed on the pages it cites - is absent
from every field's features. When the text arrives the fields are held to it
(``serving.printed_values``); this is the other half: each field's
text-dependent features are computed again on that text, its calibrator
applied again, and its threshold, so the confidence and the review flag a user
receives are the ones the release's calibrators were fitted to give
(``calibration.population``: under the fast profile they are fitted on
image-only readings with their document's text, the population production
serves).

What it needs from the result: each field's feature vector as serving built it
(``result.confidence_features``, or ``features=``: values-view path ->
``calibration.features.FieldFeatures``). Without them nothing can be computed
again - the logprob features are not in the served JSON - and the result is
left as it is, with a warning.
"""

from __future__ import annotations

import contextlib
import copy
import logging
from collections.abc import Mapping
from typing import Any

log = logging.getLogger(__name__)

#: The review flags confidence raises on a field (serving.pipeline): replaced
#: on every field confidence is computed again for.
_CONFIDENCE_FLAGS = (":low_confidence", ":no_confidence", ":uncalibrated")


def recalibrate_after_text(
    result: Any,
    page_texts: Mapping[int, str] | None,
    *,
    calibrators: Any = None,
    thresholds: Any = None,
    features: Mapping[str, Any] | None = None,
) -> None:
    """``result`` (a served ``ExtractionResult``) with its fields' confidence
    and review flags computed again on ``page_texts`` (page number -> text), in
    place.

    For every field whose feature vector the result carries and whose value is
    still served as it was read (a value the printed-values hold took out is
    not scored again - its null carries no confidence): text agreement on the
    pages it cites (``features.refresh_text_features``), then ``calibrators``
    (``feature_calibrator.CalibratorSet``) and ``thresholds``
    (``thresholds.ThresholdSet``; none: every field reviewed, as serving
    reviews it). Updated: ``result.fields`` confidences, each envelope's
    ``confidence`` and ``flagged`` in ``result.extraction`` (a value under a
    table whose row count disagrees stays flagged), the ``<path>:low_confidence``
    and ``<path>:no_confidence`` review flags, ``overall_confidence``, and the
    features kept on the result.

    Left as it is: with no calibrators (serving had none either, and flagged
    everything), no feature vectors, or no page text.
    """
    from calibration.features import refresh_text_features

    kept = features if features is not None else getattr(result, "confidence_features", None)
    if calibrators is None:
        log.info("%s: no calibrators, so confidence is not computed again on the page text",
                 getattr(result, "source_id", "?"))
        return
    if not kept:
        log.warning(
            "%s: the result carries no feature vectors (confidence_features), so its confidence "
            "cannot be computed again on the page text: the flags stay as read from the images",
            getattr(result, "source_id", "?"))
        return
    texts = {int(page): str(text or "") for page, text in (page_texts or {}).items()}
    if not any(text.strip() for text in texts.values()):
        return

    served = _served_values(result.extraction)
    rescored: dict[str, tuple[float | None, bool]] = {}
    refreshed: dict[str, Any] = dict(kept)
    for path, original in kept.items():
        if path not in served or not _same(served[path], original.value):
            continue
        updated = refresh_text_features(copy.copy(original), texts)
        refreshed[path] = updated
        confidence = calibrators.predict(updated) if updated.is_usable else None
        needs_review = (thresholds.needs_review(updated.field_type, confidence)
                        if thresholds is not None else True)
        rescored[path] = (confidence, needs_review or confidence is None)

    if not rescored:
        return
    flags = [flag for flag in result.review_flags
             if not any(flag == f"{path}{suffix}" for path in rescored for suffix in _CONFIDENCE_FLAGS)]
    for path, (confidence, needs_review) in rescored.items():
        if confidence is None:
            flags.append(f"{path}:no_confidence")
        elif needs_review:
            flags.append(f"{path}:low_confidence")
    result.review_flags = sorted(set(flags))

    for path, (confidence, _review) in rescored.items():
        if isinstance(result.fields.get(path), dict):
            result.fields[path] = {**result.fields[path], "confidence": round(confidence or 0.0, 4)}
    incomplete = _incomplete_tables(result.review_flags)
    _set_envelopes(result.extraction, "", rescored, incomplete)
    result.overall_confidence = _overall(result, rescored)
    with contextlib.suppress(AttributeError):       # a result type that takes no new attribute
        result.confidence_features = refreshed


def _served_values(extraction: Any) -> dict[str, Any]:
    """Every scalar field of what is served, by values-view path."""
    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import flatten_scalars

    return flatten_scalars(values_view(extraction or {}))


def _same(served: Any, read: Any) -> bool:
    return served == read or (str(served) == str(read) and served is not None and read is not None)


def _incomplete_tables(flags: list[str]) -> tuple[str, ...]:
    from calibration.list_completeness import is_row_completeness_flag

    return tuple(flag.split(":", 1)[0] for flag in flags if is_row_completeness_flag(flag))


def _set_envelopes(node: Any, path: str, rescored: Mapping[str, tuple[float | None, bool]],
                   incomplete: tuple[str, ...]) -> None:
    """Each envelope at a rescored values-view path given its new score and
    flag, as ``common.canonical.envelope`` writes them; a list of envelopes is
    one field, every element its score."""
    from common.canonical import CONFIDENCE_SOURCE, is_field_value

    def set_one(envelope: dict[str, Any], where: str) -> None:
        confidence, needs_review = rescored[where]
        under = any(where == name or where.startswith((f"{name}[", f"{name}.")) for name in incomplete)
        envelope["confidence"] = {"score": round(float(confidence or 0.0), 4), "source": CONFIDENCE_SOURCE}
        envelope["flagged"] = bool(needs_review or under)

    if is_field_value(node):
        if path in rescored:
            set_one(node, path)
    elif isinstance(node, dict):
        for key, value in node.items():
            _set_envelopes(value, f"{path}.{key}" if path else str(key), rescored, incomplete)
    elif isinstance(node, list):
        if node and all(is_field_value(item) for item in node):
            if path in rescored:
                for item in node:
                    set_one(item, path)
            return
        for index, item in enumerate(node):
            _set_envelopes(item, f"{path}[{index}]", rescored, incomplete)


def _overall(result: Any, rescored: Mapping[str, tuple[float | None, bool]]) -> float:
    """The mean confidence of the values the model read, as serving averages
    it: off the envelopes for a canonical answer (``serving.printed_values``),
    off ``fields`` otherwise."""
    try:
        from serving.printed_values import _overall_confidence

        if _has_envelope(result.extraction):
            return _overall_confidence(result.extraction, result.review_flags)
    except ImportError:             # pragma: no cover - serving is part of this repository
        pass
    unscored = {flag.rsplit(":", 1)[0] for flag in result.review_flags if flag.endswith(":no_confidence")}
    scores = [float(body.get("confidence") or 0.0) for path, body in result.fields.items()
              if isinstance(body, dict) and path not in unscored]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _has_envelope(node: Any) -> bool:
    from common.canonical import is_field_value

    if is_field_value(node):
        return True
    if isinstance(node, dict):
        return any(_has_envelope(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_envelope(v) for v in node)
    return False
