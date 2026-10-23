"""Measure line-of-business detection before it routes production policies.

A policy that reaches L3 with no line can have its line read by the classifier
(``serving.pipeline._resolve_lob``), and routing then trusts that reading: at or
above the line threshold it goes to the line's adapter and schema with no flag.
Nothing measured how often the reading is right - the golden eval always sends
the known line, and the gate's ``lob_detection_accuracy`` scores another field -
so detection ships off (``routing.detect_lob``) until this has been run.

Each frozen policy with one known line is read as serving would read it with no
line and no L1/L2 hint, and resolved by serving's own :func:`_resolve_lob` at
serving's own thresholds, so the numbers are the routing's, not a copy of it.
The classifier is asked once per document; the threshold sweep replays its
answer. Reported: the per-line report against
:data:`~evaluation.metrics.lob_detection.DETECTION_FLOOR`, the confidently-wrong
rate (a wrong line at or above the line threshold: routed with no flag, the
error nobody reviews), how many documents took each routing source, the same
over scanned and native documents and held-out and seen carriers, and a sweep
of both thresholds to fit them from.
"""

from __future__ import annotations

import copy
import inspect
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from evaluation.metrics.lob_detection import DETECTION_FLOOR, PER_LINE_MIN_SUPPORT, score_lob_detection

#: The thresholds the sweep tries. An uncertain threshold above the line
#: threshold is skipped: no reading could then be uncertain.
SWEEP_LINE_THRESHOLDS: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 0.99)
SWEEP_UNCERTAIN_THRESHOLDS: tuple[float, ...] = (0.40, 0.50, 0.60, 0.70)


@dataclass
class DetectionCase:
    """One policy whose line is known, as serving would receive it without one."""

    source_id: str
    #: The line the policy is, as a registered line name.
    lob: str
    #: The request serving would get - its ``known_lob`` and ``lob_hypothesis``
    #: are withheld whatever this carries.
    request: Any
    #: None when the metadata does not say, so the slice is not reported.
    is_scanned: bool | None = None
    held_out_carrier: bool | None = None


class _ReadOnce:
    """The classifier, asked once per document; its answer - or its failure - replayed.

    The sweep resolves the same reading at many thresholds, and a second
    generation would be another sample, not the same reading.
    """

    def __init__(self, classifier: Any) -> None:
        self._classifier = classifier
        # _resolve_lob asks whether a classifier reads lines at all.
        self.reads_lob = getattr(classifier, "reads_lob", False)
        self.answer: Any = None
        self.failed = False

    def classify(self, image_paths: list[str], ocr_text: str | None) -> Any:
        if self.answer is None:
            try:
                self.answer = self._classifier.classify(image_paths, ocr_text)
            except Exception as exc:  # noqa: BLE001 - replayed, and _resolve_lob handles it
                self.answer, self.failed = exc, True
        if self.failed:
            raise self.answer
        # A copy: the hint reconciliation writes into the classification.
        return copy.deepcopy(self.answer)


def serving_lob_thresholds() -> tuple[float, float]:
    """``(line threshold, uncertain threshold)`` as the endpoint would serve them:

    configs/inference/vllm_serving.yaml, else :func:`serving.pipeline.extract`'s
    own defaults."""
    from serving.pipeline import extract
    from serving.vllm_entrypoint import serving_thresholds

    defaults = inspect.signature(extract).parameters
    tuning = serving_thresholds()
    return (
        tuning.get("lob_confidence_threshold", defaults["lob_confidence_threshold"].default),
        tuning.get("lob_uncertain_threshold", defaults["lob_uncertain_threshold"].default),
    )


def detection_cases(
    documents: Iterable[Any], local_images: Mapping[str, str], *,
    mode: str = "ocr_plus_image", seed: int = 42,
) -> list[DetectionCase]:
    """The frozen policies with one known line, as requests in ``mode``.

    Built as :func:`evaluation.golden_eval.evaluate` builds its requests - the
    same pages, no text for ``image_only`` - with the type sent and the line
    not. ``noisy_ocr_image`` corrupts only the pages the classifier reads, as
    the classify rows it is trained on do (build_jsonl.classify_rows, the same
    seed): the document's budget spread over a long policy would rarely touch
    its opening pages, and the noisy measurement would read clean text. A
    package naming several lines, and a line no longer registered, have no one
    right answer and are left out.
    """
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
    from serving.doc_type_classifier import CLASSIFIER_PAGES, known_line
    from serving.pipeline import ExtractionRequest

    cases = []
    for doc in documents:
        line = known_line(doc.lob) if doc.doc_type == "policy" and isinstance(doc.lob, str) else None
        if line is None:
            continue
        texts = dict(doc.page_texts)
        if mode == "noisy_ocr_image":
            opening = sorted(texts)[:CLASSIFIER_PAGES]
            corrupted, _details = corrupt_ocr_pages(
                [texts[p] for p in opening], f"{doc.source_id}#classify", seed=seed)
            texts.update(zip(opening, corrupted, strict=True))
        request = ExtractionRequest(
            source_id=doc.source_id,
            image_paths=[local_images[key] for key in doc.image_keys],
            page_texts={} if mode == "image_only" else texts,
            modality_mode=mode,
            known_doc_type="policy",
        )
        cases.append(DetectionCase(doc.source_id, line, request, is_scanned=doc.is_scanned,
                                   held_out_carrier=doc.held_out_carrier))
    return cases


def _resolve(case: DetectionCase, reader: _ReadOnce, line_threshold: float, uncertain_threshold: float) -> Any:
    """Serving's own resolution of ``case`` with no line and no hint."""
    from serving.adapter_router import Route
    from serving.pipeline import _resolve_lob

    # No classification on the route: _resolve_lob reads the line itself, under
    # its own failure handling, as for a caller that named the type alone.
    route_ = Route(doc_type="policy", acord_form=None, adapter=None,
                   schema_doc_type="policy", schema_acord_form=None)
    request = replace(case.request, known_lob=None, lob_hypothesis=None)
    return _resolve_lob(request, reader, route_, detect=True,
                        line_threshold=line_threshold, uncertain_threshold=uncertain_threshold)


def _summary(resolved: Sequence[tuple[DetectionCase, Any]], *, floor: float, min_support: int) -> dict[str, Any]:
    """The report over ``resolved``: per line, against the floor, and how it routed."""
    report = score_lob_detection(
        (case.lob, resolution.lob if isinstance(resolution.lob, str) else None)
        for case, resolution in resolved
    )
    # Routed with no flag to another line's adapter and schema: nobody reviews it.
    confidently_wrong = sum(1 for case, resolution in resolved
                            if resolution.source == "detected" and resolution.lob != case.lob)
    scored = report.scored
    return {
        "documents": scored,
        "overall": None if report.overall is None else round(report.overall, 4),
        "by_line": report.by_line(),
        "below_floor": report.below_floor(floor, min_support=min_support),
        "confidently_wrong": confidently_wrong,
        "confidently_wrong_rate": round(confidently_wrong / scored, 4) if scored else None,
        "sources": dict(sorted(Counter(str(r.source) for _c, r in resolved).items())),
        "confusion": [[expected, got, n] for (expected, got), n
                      in sorted(report.confusion.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))],
    }


def _slices(resolved: Sequence[tuple[DetectionCase, Any]], **kwargs: Any) -> dict[str, Any]:
    """The report over scanned and native documents, and held-out and seen
    carriers, for whichever of the two the metadata records."""
    out: dict[str, Any] = {}
    for attribute, yes, no in (("is_scanned", "scanned", "native"),
                               ("held_out_carrier", "held_out_carrier", "seen_carrier")):
        for name, wanted in ((yes, True), (no, False)):
            subset = [(c, r) for c, r in resolved if getattr(c, attribute) is wanted]
            if subset:
                out[name] = _summary(subset, **kwargs)
    return out


def measure_lob_detection(
    cases: Iterable[DetectionCase],
    classifier: Any,
    *,
    line_threshold: float,
    uncertain_threshold: float,
    floor: float = DETECTION_FLOOR,
    min_support: int = PER_LINE_MIN_SUPPORT,
    line_thresholds: Sequence[float] = SWEEP_LINE_THRESHOLDS,
    uncertain_thresholds: Sequence[float] = SWEEP_UNCERTAIN_THRESHOLDS,
) -> dict[str, Any]:
    """How the line ``classifier`` reads would have routed each of ``cases``.

    ``clears_floor`` is the answer to "can detection be switched on at these
    thresholds": overall accuracy at the floor or above, and no line with
    enough documents below it. A failed read counts as a wrong one, as it
    routes to the fallback at serving.
    """
    from serving.doc_type_classifier import StaticClassifier

    if classifier is None or isinstance(classifier, StaticClassifier):
        raise ValueError("line detection needs a classifier that reads lines; this one never does")

    readers = [(case, _ReadOnce(classifier)) for case in cases]
    kwargs = {"floor": floor, "min_support": min_support}
    resolved = [(case, _resolve(case, reader, line_threshold, uncertain_threshold)) for case, reader in readers]
    body = _summary(resolved, **kwargs)
    body.update({
        "line_threshold": line_threshold,
        "uncertain_threshold": uncertain_threshold,
        "floor": floor,
        "min_support": min_support,
        "clears_floor": body["overall"] is not None and body["overall"] >= floor and not body["below_floor"],
        "read_failures": sum(1 for _case, reader in readers if reader.failed),
        "slices": _slices(resolved, **kwargs),
    })

    sweep = []
    for line_at in line_thresholds:
        for uncertain_at in uncertain_thresholds:
            if uncertain_at > line_at:
                continue
            routed = [(case, _resolve(case, reader, line_at, uncertain_at)) for case, reader in readers]
            summary = _summary(routed, **kwargs)
            total = summary["documents"] or 1
            sources = summary["sources"]
            sweep.append({
                "line_threshold": line_at, "uncertain_threshold": uncertain_at,
                "accuracy": summary["overall"],
                "confidently_wrong_rate": summary["confidently_wrong_rate"],
                "detected_share": round(sources.get("detected", 0) / total, 4),
                "uncertain_share": round(sources.get("detected_uncertain", 0) / total, 4),
                "undetected_share": round(sources.get("undetected", 0) / total, 4),
            })
    body["sweep"] = sweep
    return body
