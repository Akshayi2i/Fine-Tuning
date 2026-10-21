"""The request pipeline — the canonical extraction path (IMPL-11).

Every extraction goes through here: production requests and the testing harness
alike. **The harness calls this function; it does not reimplement it.** That is
the whole basis of "test == prod" — two code paths would be two systems, and the
tests would describe the wrong one.

    OCR (if provided)
      -> classify document type (+ ACORD form)
      -> route adapter / prompt / schema
      -> page-route if the document is long
      -> generate with logprobs
      -> per-field confidence, calibrated
      -> list-completeness cross-check
      -> validate against the schema
      -> JSON + confidence + review flags

The schema validation at the end mirrors the Fideon SPEC_07 Stage 3 audit gate: a
schema-invalid response is an **error**, not a returned result.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from calibration.apply_calibration import CalibratedField, CalibratedResult, apply_calibration
from calibration.features import shown_ocr_text
from calibration.fit_calibration import CalibrationParams
from calibration.list_completeness import (
    check_document,
    is_row_completeness_flag,
    merge_review_flags,
)
from calibration.logprob_confidence import field_confidences
from common.canonical import (
    collapse_spans,
    envelope,
    values_view,
    with_output_dates,
    with_system_fields,
    without_bare_values,
)
from common.constants import DEFAULT_REVIEW_CONFIDENCE_THRESHOLD
from common.schemas import (
    is_canonical,
    is_common_model,
    is_valid,
    iter_validation_errors,
    iter_validation_errors_by_keyword,
    required_fields,
    resolved_schema,
    with_page_bounds,
)
from inference_core.input_builder import EMPTY_PAGE_TEXT, build_messages
from inference_core.model_runner import Generation, LoadedModel, generate, generate_batch
from inference_core.span_map import SpanMapError, map_field_spans
from serving.adapter_router import Route, RoutingError, route
from serving.doc_type_classifier import Classifier

log = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Raised when a request cannot produce a valid extraction."""


@dataclass
class ExtractionRequest:
    """One document to extract."""

    source_id: str
    image_paths: list[str]
    ocr_text: str | None = None
    page_texts: dict[int, str] = field(default_factory=dict)
    ocr_meta: dict[str, Any] = field(default_factory=dict)
    modality_mode: str = "ocr_plus_image"
    #: Skip classification when the caller already knows the type.
    known_doc_type: str | None = None
    known_acord_form: str | None = None
    #: A policy's line of business, when the caller (L1/L2) knows it. Selects the
    #: line's canonical schema; absent — or naming several lines — the policy is
    #: extracted into the client's canonical fallback. Either way the output is
    #: canonical JSON.
    known_lob: str | list[str] | None = None
    #: The uploaded file's name, when the caller knows it. Written into the
    #: canonical output by the system, never asked of the model
    #: (common.canonical.SYSTEM_SUPPLIED_FIELDS), as is the page count.
    source_file_name: str | None = None
    #: Read a policy whose line is missing or in no layout family with the base
    #: model against _fallback.json, rather than refusing it. Off unless the
    #: caller asks: such a policy is never routed silently.
    allow_lob_fallback: bool = False
    #: A line of business L1/L2 proposed without being sure enough to send it as
    #: ``known_lob``: reconciled with the line the classifier reads.
    lob_hypothesis: str | None = None


@dataclass
class ExtractionResult:
    """The output contract (master §9)."""

    source_id: str
    doc_type: str
    model_version: str
    mode: str
    schema_valid: bool
    overall_confidence: float
    extraction: dict[str, Any] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    list_fields: dict[str, Any] = field(default_factory=dict)
    line_of_business: dict[str, Any] | None = None
    pages_used: list[int] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)
    route_info: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    validation_errors: list[str] = field(default_factory=list)
    #: A Loss Run's claims against its printed totals (calibration.reconciliation);
    #: None for every other type.
    reconciliation: dict[str, Any] | None = None
    #: Milliseconds of every generation round this document took - one per round
    #: of windows (a round's windows run together), or one for a document read
    #: in one call (release measurements).
    window_latencies_ms: list[float] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.review_flags)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "doc_type": self.doc_type,
            "model_version": self.model_version,
            "mode": self.mode,
            "schema_valid": self.schema_valid,
            "overall_confidence": self.overall_confidence,
            "line_of_business": self.line_of_business,
            # The canonical JSON itself: every key of the line's schema, values in
            # their envelopes (raw, parsed, confidence, page_ref, flagged). The
            # entries below are summaries of it.
            "extraction": self.extraction,
            "fields": self.fields,
            "list_fields": self.list_fields,
            "pages_used": self.pages_used,
            "review_flags": sorted(self.review_flags),
            "route": self.route_info,
            # Loss Runs only, so every other type's output is unchanged.
            **({"reconciliation": self.reconciliation} if self.reconciliation is not None else {}),
        }


def _resolve_route(request: ExtractionRequest, classifier: Classifier, *,
                   confidence_threshold: float, adapter_map: dict[str, Any],
                   fallback_doc_type: str | None = None) -> Route:
    if request.known_doc_type:
        from serving.doc_type_classifier import StaticClassifier

        classifier = StaticClassifier(request.known_doc_type, request.known_acord_form)
    classification = classifier.classify(*_classifier_input(request))
    try:
        return route(
            classification,
            confidence_threshold=confidence_threshold,
            adapter_map=adapter_map,
            fallback_doc_type=fallback_doc_type,
        )
    except RoutingError as exc:
        # Before generation, deliberately: a document nobody could route costs
        # nothing to refuse and a full extraction to answer wrongly.
        raise PipelineError(str(exc)) from exc


def _classifier_input(request: ExtractionRequest) -> tuple[list[str], str | None]:
    """The first pages' images and text, in page order, for the classifier.

    A multi-page request carries its text per page (``page_texts``) and no
    ``ocr_text``, so the classifier was handed no text at all and the images in
    the order the caller listed them - ``page_10.png`` before ``page_2.png``.
    It reads the first two pages: their images by page number, their text
    joined. None under ``image_only``, where the prompt says no text exists.
    Picked by ``classifier_input``, the rule a training row is built by too.
    """
    from serving.doc_type_classifier import classifier_input

    if not request.page_texts:
        return request.image_paths, request.ocr_text
    image_only = request.modality_mode == "image_only"
    images, text = classifier_input(
        request.page_texts, lambda page: _image_for(request, page),
        None if image_only else request.page_texts.get,
    )
    return images, None if image_only else text or request.ocr_text


def _image_for(request: ExtractionRequest, page: int) -> str:
    """The image for one page, by the `page_N.` convention the OCR stage writes.

    Otherwise by position: page ``N`` is the ``N``-th image (the endpoint checks
    that ``page_texts`` numbers exactly 1..n against the images). The fallback
    used to index into ``page_texts``, which an image_only request does not
    have — so every page of it resolved to image 1 — and clamped an
    out-of-range page to the last image instead of saying it did not exist.
    """
    for path in request.image_paths:
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        if name.startswith(f"page_{page}."):
            return path
    if not 1 <= page <= len(request.image_paths):
        raise PipelineError(
            f"{request.source_id}: page {page} requested but only "
            f"{len(request.image_paths)} page image(s) were sent"
        )
    return request.image_paths[page - 1]


#: The blank-page placeholder, defined where training's messages are built too.
_EMPTY_PAGE = EMPTY_PAGE_TEXT


def _all_page_texts(request: ExtractionRequest) -> list[str]:
    """One markdown string per image, in page order.

    ``page_texts`` is the real source. ``ocr_text`` is accepted for a
    single-page request only: a joined blob cannot be split back into pages, and
    guessing where the seams were is exactly the failure this change removes.
    """
    if request.page_texts:
        return [request.page_texts[page] or _EMPTY_PAGE for page in sorted(request.page_texts)]
    if len(request.image_paths) == 1:
        return [request.ocr_text or ""]
    raise PipelineError(
        f"{request.source_id} supplies {len(request.image_paths)} page images but only a single "
        "joined ocr_text. Each image needs its own page text so the model can tell which text "
        "belongs to which page; populate page_texts from the per-page markdown the OCR stage "
        "wrote rather than joining it."
    )


def _build_request(
    model: LoadedModel,
    route_: Route,
    image_paths: list[str],
    ocr_pages: list[str] | None,
    modality_mode: str,
    *,
    page_numbers: list[int] | None = None,
    total_pages: int | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """The messages and the decoding schema for one generation.

    ``sections`` names the schema slice a policy window asks for; the prompt and
    the structured-decoding schema are both that slice, from the same selectors.
    """
    built = build_messages(
        route_.schema_doc_type,
        image_paths,
        ocr_pages,
        modality_mode,
        acord_form=route_.schema_acord_form,
        lob=lob,
        sections=sections,
        page_numbers=page_numbers,
        total_pages=total_pages,
    )
    # Constrained to the routed type's schema when serving is configured for it
    # (arch v2.1 §13). Without this the schema-validity floor in §13b measured an
    # unconstrained model, so one malformed bf16 output made every quantized
    # format unvalidatable.
    # Page references bounded to the pages this request shows (with_page_bounds):
    # an unbounded list let a model count past the last page to max_new_tokens,
    # and a window may cite only its own pages, as training taught it.
    schema = (
        with_page_bounds(
            resolved_schema(route_.schema_doc_type, route_.schema_acord_form, lob, sections),
            total_pages or len(image_paths),
            pages=page_numbers,
        )
        if getattr(model.config, "structured_outputs", False) else None
    )
    return built.messages, schema


def _window_answer_cap(group: str, doc_type: str | None) -> int:
    """A window's answer limit: its section task's reserved budget, as in training."""
    from common.config import answer_cap
    from common.schema_sections import task_for

    return answer_cap(task_for(group), doc_type)


def _parse_generation(
    result: Generation, route_: Route, *, refuse_truncated: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(extraction, spans)`` from one generation, or :class:`PipelineError`.

    The extraction comes back with every date in ``MM/DD/YYYY`` and, for a
    canonical document, still in the model's sparse ``raw``/``parsed``/
    ``page_ref`` form; the spans are keyed by values-view path.

    ``refuse_truncated`` treats output cut off at the token limit as a failure
    even when what was cut still parses. A window can afford that: it is retried
    over fewer pages. Parsing a truncated answer as complete would drop every
    row after the cut with nothing to say so.
    """
    if refuse_truncated and result.truncated():
        raise PipelineError(
            f"the {route_.schema_doc_type} generation hit the token limit, so its output is "
            "incomplete"
        )
    # Parse and span-map together: both fail for the same underlying reason — the
    # model did not return JSON — and reporting that as two different errors from
    # two layers would obscure a single cause.
    try:
        extraction = json.loads(result.text)
        if not isinstance(extraction, dict):
            # A bare array or scalar parses cleanly and maps cleanly, then dies
            # with an AttributeError several layers down. The contract is one
            # JSON object, so this is the same generation failure as unparseable
            # output and is reported as one.
            raise TypeError(f"expected a JSON object, got {type(extraction).__name__}")
        spans = map_field_spans(result.text, result.tokens, result.token_logprobs)
    except (ValueError, TypeError, SpanMapError) as exc:
        raise PipelineError(
            f"the model did not return parseable JSON for {route_.schema_doc_type}: "
            f"{result.text[:160]!r}. The prompt asks for JSON only with no fences, so this is a "
            f"generation failure rather than a formatting quirk to work around ({exc})."
        ) from exc
    # Dates are reformatted AFTER the span map: the spans describe the tokens the
    # model generated, and a reformatted date is the same value in other
    # characters, not a different value with different evidence.
    return with_output_dates(extraction), collapse_spans(spans)


def _generate_once(
    model: LoadedModel,
    route_: Route,
    image_paths: list[str],
    ocr_pages: list[str] | None,
    modality_mode: str,
    *,
    page_numbers: list[int] | None = None,
    total_pages: int | None = None,
    lob: str | list[str] | None = None,
    sections: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float | None]:
    """One scoped generation. Returns ``(extraction, spans, latency_ms)``."""
    messages, schema = _build_request(
        model, route_, image_paths, ocr_pages, modality_mode,
        page_numbers=page_numbers, total_pages=total_pages, lob=lob, sections=sections,
    )
    result = generate(model, messages, adapter=route_.adapter, json_schema=schema)
    extraction, spans = _parse_generation(result, route_)
    return extraction, spans, result.latency_ms


#: The review flag a window that could not be read leaves on the document. The
#: values its pages carried are missing, and nothing else would say so.
WINDOW_FAILED_FLAG = "window_failed"


def _extract_policy_windows(
    request: ExtractionRequest,
    model: LoadedModel,
    route_: Route,
    lob: str | list[str] | None,
    *,
    latencies: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float | None, list[int], list[str]]:
    """Read a canonical policy window by window, then merge.

    Returns ``(extraction, spans, latency_ms, pages_used, flags)`` — the merged
    model-form document and its spans, in the shape the single-call path
    returns, so confidence, completeness and the envelope run on it unchanged.

    **Concurrent.** Every window of a round goes to the model in one batch; vLLM
    schedules them together, so a 17-window policy costs roughly the time of its
    slowest window rather than the sum of all of them. ``latency_ms`` is the
    wall time actually spent.

    **One bad window does not lose the document.** A window whose output is not
    parseable JSON, or was cut off at the token limit — the usual cause on a
    dense schedule page — is split in half and retried in the next round, since
    fewer pages means less to write. Retrying it unchanged would be pointless:
    decoding is greedy, so it would write the same thing again. A window that
    still fails at a single page is dropped and flagged ``window_failed``, so the
    document comes back with everything else and a person is told which pages
    it could not read. Only when no window at all succeeds is the request an
    error.
    """
    from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
    from serving.policy_merge import PolicyWindow, merge_policy_windows

    image_only = request.modality_mode == "image_only"
    page_count = len(request.page_texts) or len(request.image_paths)
    texts: dict[int, str] = {}
    if not image_only:
        texts = dict(request.page_texts) or dict(enumerate(_all_page_texts(request), start=1))

    # Every page, with the declarations found in the same text by the same
    # rule as the training rows were.
    routed, declarations_page = pages_to_read(texts or None, page_count)
    pending: list[tuple[str, list[int]]] = [
        (plan.group, list(plan.pages)) for plan in plan_windows(lob, routed, declarations_page)
    ]

    windows: list[PolicyWindow] = []
    failed: list[str] = []
    first_cause: str | None = None
    wall_ms = 0.0
    rounds = 0
    while pending:
        rounds += 1
        requests = [
            (*_build_request(
                model, route_,
                [_image_for(request, page) for page in pages],
                None if image_only else [texts.get(page) or _EMPTY_PAGE for page in pages],
                request.modality_mode,
                page_numbers=pages, total_pages=page_count,
                lob=lob, sections=group,
            ), _window_answer_cap(group, route_.schema_doc_type))
            for group, pages in pending
        ]
        started = time.perf_counter()
        results = generate_batch(model, requests, adapter=route_.adapter)
        round_ms = (time.perf_counter() - started) * 1000
        wall_ms += round_ms
        if latencies is not None:
            # One entry per round: its windows run together, and vLLM gives each
            # the round's wall time, so one per window would count it n times.
            latencies.append(round(round_ms, 1))

        retry: list[tuple[str, list[int]]] = []
        for (group, pages), result in zip(pending, results, strict=True):
            try:
                if isinstance(result, Exception):
                    raise PipelineError(str(result))
                extraction, spans = _parse_generation(result, route_, refuse_truncated=True)
            except PipelineError as exc:
                first_cause = first_cause or str(exc)
                if len(pages) > 1:
                    half = len(pages) // 2
                    retry += [(group, pages[:half]), (group, pages[half:])]
                    log.warning(
                        "%s: %s window over pages %s failed (%s); retrying as two smaller "
                        "windows", request.source_id, group, pages, exc,
                    )
                else:
                    failed.append(f"{group}:p{pages[0]}")
                    log.warning(
                        "%s: %s window over page %s failed even alone (%s); dropped and "
                        "flagged for review", request.source_id, group, pages[0], exc,
                    )
                continue
            windows.append(PolicyWindow(group, pages, extraction, spans, result.latency_ms))
        pending = retry

    if not windows:
        raise PipelineError(
            f"no window of {request.source_id} could be read ({len(failed)} failed: "
            f"{', '.join(failed[:6])}). The first failure: {first_cause}"
        )

    merged = merge_policy_windows(windows, lob=lob)
    flags = merged.review_flags + [f"{WINDOW_FAILED_FLAG}:{entry}" for entry in failed]
    log.info(
        "%s read in %d window(s) over %d routed page(s) in %d round(s); %d duplicate row(s) "
        "collapsed, %d conflict(s), %d window(s) unreadable", request.source_id, len(windows),
        len(routed), rounds, merged.duplicates_collapsed, len(merged.conflicts), len(failed),
    )
    return merged.extraction, merged.spans, round(wall_ms, 1), sorted(routed), flags


def _likely_missed(extraction: dict[str, Any], page_texts: dict[int, str], lob: Any) -> list[str]:
    """Declarations fields left empty whose label and a value of their type open
    a line of a page the declarations windows read - every page
    (serving.conventions)."""
    from serving.conventions import declaration_objects, likely_missed

    return likely_missed(extraction, page_texts, sorted(page_texts), declaration_objects(lob))


#: A coverage served with no unit while the answer holds several units of a kind.
UNIT_LINK_FLAG = "applies_to:unit_link_unknown"

#: The unit tables a coverage is written per unit of. Not drivers (coverages are
#: not per driver), locations (a building is the insured unit) or scheduled
#: items (they carry their own coverage; Coverage A is not theirs).
COVERAGE_UNIT_TABLES = ("vehicles", "watercraft", "buildings")


def unit_link_flags(extraction: dict[str, Any]) -> list[str]:
    """Review flags for common-model coverages that may have lost their unit.

    A window links a coverage to the vehicle (boat, building) it applies to only
    when both are on its pages; read apart, the link is dropped - a window
    cannot cite a unit it cannot see - and the merged row has no
    ``applies_to``, which the client reads as "the whole policy". That cannot
    be told from a coverage that does apply to the whole policy, so the row is
    flagged for a person, not changed: when its unit table holds two or more
    units (with one vehicle or one dwelling, the policy and the unit are the
    same thing), or when another row of its code does carry a link - the plain
    sign of one that was lost.
    """
    coverages = [row for row in extraction.get("coverages") or [] if isinstance(row, dict)]
    several = any(len(extraction.get(table) or []) >= 2 for table in COVERAGE_UNIT_TABLES)
    linked = {row.get("coverage_code") for row in coverages if row.get("applies_to")}
    return [f"coverages[{index}].{UNIT_LINK_FLAG}"
            for index, row in enumerate(extraction.get("coverages") or [])
            if isinstance(row, dict) and not row.get("applies_to")
            and (several or row.get("coverage_code") in linked)]


def _without_empty(node: Any, keep: frozenset[str] = frozenset()) -> Any:
    """``node`` without the objects written with nothing in them: an empty
    optional object, or an empty row. The decoder lets a window write ``{}``
    (it drops the client's minProperties); served, it is a row of nulls nobody
    read, and judged, it refuses the whole document. ``keep`` names the
    required top-level sections, which stay so that one written empty still
    fails as a section nobody read."""
    if isinstance(node, dict):
        pruned = {}
        for key, value in node.items():
            value = _without_empty(value)
            if value == {} and key not in keep:
                continue
            pruned[key] = value
        return pruned
    if isinstance(node, list):
        return [item for item in (_without_empty(v) for v in node) if item != {}]
    return node


#: A policy read by the base model against _fallback.json (Fideon SPEC_06 §9a).
LOB_FALLBACK_FLAG = "route:lob_fallback_used"


@dataclass
class LobResolution:
    """Which line of business a policy is read as, and where that came from.

    ``source``: ``caller`` (sent with the request, as L1/L2 do), ``detected``
    (the classifier, confidently), ``detected_uncertain`` (the classifier, below
    the line threshold: its family's adapter and its line's schema, flagged), or
    ``undetected`` (no line found, or the read failed: the base model against
    the fallback schema, flagged). None for a document that is no policy, when
    detection is off, or when nothing wired reads lines - no detection at all.
    """

    lob: Any = None
    source: str | None = None
    confidence: float | None = None
    candidates: list[tuple[str, float]] = field(default_factory=list)
    hypothesis_agreed: bool | None = None
    flags: list[str] = field(default_factory=list)

    def as_output(self) -> dict[str, Any]:
        lines = [self.lob] if isinstance(self.lob, str) else list(self.lob or [])
        return {"lines": lines, "source": self.source, "confidence": self.confidence,
                "candidates": [list(c) for c in self.candidates],
                "hypothesis_agreed": self.hypothesis_agreed}


def _resolve_lob(
    request: ExtractionRequest, classifier: Classifier | None, route_: Route, *,
    detect: bool, line_threshold: float, family_threshold: float,
) -> LobResolution:
    """The line a policy is read with: the caller's, or the classifier's.

    A policy reaching L3 with no line - a scan L1/L2 could not read - used to be
    refused, or read by the base model against the generic schema. With
    detection on, the classifier reads the line too, and its confidence (the
    line's own token probability) decides how far it is trusted.

    Only a classifier that reads lines can report one missing. With none wired,
    a :class:`~serving.doc_type_classifier.StaticClassifier`, or an answer from a
    classifier that never asks for the line, there is no detection: the policy
    is refused, or read against the fallback when the caller asked
    (``allow_lob_fallback``), as before detection existed - a reading nobody
    made must not switch the fallback on. A read that fails (an answer that is
    no JSON, a backend error) is ``undetected``: an optional lookup does not
    fail the extraction.
    """
    from serving.doc_type_classifier import StaticClassifier, combine_lob_with_hypothesis, known_line

    classification = route_.classification
    if route_.doc_type != "policy":
        return LobResolution(request.known_lob)
    if request.known_lob:
        # The caller's line wins. A confident classifier reading another one is
        # worth a person's look, and evidence about L1/L2 when it is systematic.
        caller = known_line(request.known_lob) if isinstance(request.known_lob, str) else None
        disagrees = (
            classification is not None and caller is not None and classification.lob not in (None, caller)
            and (classification.lob_confidence or 0.0) >= line_threshold
        )
        return LobResolution(request.known_lob, "caller", flags=["lob:caller_disagrees"] if disagrees else [])
    if not detect:
        return LobResolution(None)
    if classification is None or classification.method == "static":
        # The caller named the type, so the classifier has not read this document.
        if classifier is None or isinstance(classifier, StaticClassifier):
            return LobResolution(None)
        try:
            classification = classifier.classify(*_classifier_input(request))
        except Exception as exc:  # noqa: BLE001 - a failed lookup is no line, not a failed request
            # The type only: a parse failure's message quotes the model's
            # answer, and that can quote the document.
            log.warning("%s: the line of business could not be read (%s); reading the policy "
                        "against the fallback, flagged", request.source_id, type(exc).__name__)
            # No hint either: a hint is too weak to route on alone.
            return LobResolution(None, "undetected", flags=["lob:undetected"])
    reads_lines = (classification.method in ("zero_shot", "zero_shot_base")
                   or getattr(classifier, "reads_lob", False))
    if classification.lob is None and not reads_lines:
        # Never asked for the line, so its absence says nothing about the policy.
        return LobResolution(None)
    classification = combine_lob_with_hypothesis(
        classification, request.lob_hypothesis, confidence_threshold=line_threshold)
    found = LobResolution(
        classification.lob, confidence=classification.lob_confidence,
        candidates=list(classification.lob_candidates),
        hypothesis_agreed=classification.lob_hypothesis_agreed,
    )
    confidence = classification.lob_confidence
    if classification.lob is None or (confidence is not None and confidence < family_threshold):
        found.lob, found.source, found.flags = None, "undetected", ["lob:undetected"]
    elif confidence is not None and confidence >= line_threshold:
        found.source = "detected"
    else:
        # Below the line threshold, or no logprobs to measure it by: the family
        # is likely right even when the line is not, and a person should check.
        found.source, found.flags = "detected_uncertain", ["lob:uncertain_within_family"]
    return found

#: Review flags a Loss Run's merge and reconciliation leave on the document.
LOSSRUN_TOTALS_MISMATCH_FLAG = "claims:totals_mismatch"
LOSSRUN_MERGE_CONFLICT_FLAG = "claims:merge_conflict"


def lossrun_window_budget() -> int:
    """Output tokens one Loss Run window plans for AND may write: one number for
    both, so a window sized for N rows is allowed to write N rows."""
    from common.config import answer_cap

    return answer_cap("lossrun_rows", "lossrun")


def lossrun_windows(request: ExtractionRequest) -> list[list[int]]:
    """A Loss Run's page windows, sized from its row density as the corpus plans them.

    Windowed only when its rows cannot fit one answer. Each window is the Loss
    Run ``extract`` prompt over a page subset, a shape no training row has yet
    (Loss Runs train as whole documents until the lossrun_rows task is built),
    so it is used only where one call would be cut off: the estimated rows of
    the whole document overflow the window budget. Everything else is one
    window - read in one call, as before. So is a request without per-page
    text: nothing to estimate from (image only), or a joined ``ocr_text``,
    which the one-call path refuses for a multi-page document.
    """
    from data_pipeline.dataset_builder.expand_tasks import TOKENS_PER_CLAIM_ROW, plan_windows
    from data_pipeline.ocr.run_mineru import count_table_rows

    pages = sorted(request.page_texts) or list(range(1, len(request.image_paths) + 1))
    if not request.page_texts:
        return [pages]
    # MinerU writes tables as HTML, which count_table_rows counts as well as
    # pipe tables.
    densities = [count_table_rows(request.page_texts.get(page) or "") for page in pages]
    budget = lossrun_window_budget()
    if sum(densities) * TOKENS_PER_CLAIM_ROW <= budget:
        return [pages]
    windows = plan_windows(densities, output_budget=budget)
    return [[pages[index - 1] for index in window] for window in windows]


def _extract_lossrun_windows(
    request: ExtractionRequest,
    model: LoadedModel,
    route_: Route,
    windows: list[list[int]],
    latencies: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float | None, list[int], list[str], dict[str, Any]]:
    """Read a Loss Run too long for one answer window by window, then merge.

    Each window is asked for the whole Loss Run schema over its pages (the
    trained ``extract`` prompt, on a page subset); the windows overlap so a row
    cut by a page break is seen whole by one of them. Their claims are merged by
    :func:`serving.lossrun_merge.merge_extracted_windows` and reconciled against
    the printed totals (Fideon SPEC_09 handoff item 8, arch v2.1 §7b). A window
    that fails is split and retried over fewer pages, and one that fails alone
    is flagged, as for a policy.

    Returns ``(extraction, spans, latency_ms, pages_used, flags, reconciliation)``.
    """
    from serving.lossrun_merge import merge_extracted_windows

    image_only = request.modality_mode == "image_only"
    total = len(request.page_texts) or len(request.image_paths)
    # The budget the windows were planned for (lossrun_windows), never a
    # smaller one: a window allowed fewer rows than it was sized for is cut
    # off, split, and a dense page is lost.
    cap = lossrun_window_budget()
    pending = [list(pages) for pages in windows]
    read: list[tuple[list[int], dict[str, Any], dict[str, Any]]] = []
    failed: list[str] = []
    first_cause: str | None = None
    wall_ms = 0.0
    while pending:
        requests = [
            (*_build_request(
                model, route_, [_image_for(request, page) for page in pages],
                None if image_only else [request.page_texts.get(page) or _EMPTY_PAGE for page in pages],
                request.modality_mode, page_numbers=pages, total_pages=total,
            ), cap)
            for pages in pending
        ]
        started = time.perf_counter()
        results = generate_batch(model, requests, adapter=route_.adapter)
        round_ms = (time.perf_counter() - started) * 1000
        wall_ms += round_ms
        if latencies is not None:
            # One entry per round: its windows run together, and vLLM gives each
            # the round's wall time, so one per window would count it n times.
            latencies.append(round(round_ms, 1))
        retry: list[list[int]] = []
        for pages, result in zip(pending, results, strict=True):
            try:
                if isinstance(result, Exception):
                    raise PipelineError(str(result))
                extraction, spans = _parse_generation(result, route_, refuse_truncated=True)
            except PipelineError as exc:
                first_cause = first_cause or str(exc)
                if len(pages) > 1:
                    retry += [pages[:len(pages) // 2], pages[len(pages) // 2:]]
                else:
                    failed.append(f"claims:p{pages[0]}")
                continue
            read.append((pages, extraction, spans))
        pending = retry

    if not read:
        raise PipelineError(
            f"no window of Loss Run {request.source_id} could be read ({len(failed)} failed). "
            f"The first failure: {first_cause}"
        )
    read.sort(key=lambda item: item[0][0])
    extraction, spans, merged, reconciliation = merge_extracted_windows(
        [(extraction, spans) for _pages, extraction, spans in read])
    flags = [f"{WINDOW_FAILED_FLAG}:{entry}" for entry in failed]
    if merged.flagged:
        flags.append(LOSSRUN_MERGE_CONFLICT_FLAG)
    log.info(
        "%s: Loss Run read in %d window(s); %d duplicate row(s) collapsed, %d conflict(s), "
        "reconciliation %s", request.source_id, len(read), merged.duplicates_collapsed,
        len(merged.conflicts), reconciliation.status,
    )
    pages_used = sorted({page for pages, _e, _s in read for page in pages})
    return extraction, spans, round(wall_ms, 1), pages_used, flags, reconciliation.as_dict()


def _calibration_for(
    calibration: CalibrationParams | Mapping[str, CalibrationParams],
    doc_type: str | None,
) -> CalibrationParams:
    """Pick this document's calibration once its type is known."""
    if not isinstance(calibration, Mapping):
        return calibration
    if doc_type and doc_type in calibration:
        return calibration[doc_type]
    raise PipelineError(
        f"no calibration parameters for doc_type {doc_type!r}; the endpoint holds calibration for "
        f"{sorted(calibration)}. Confidence would otherwise be served raw, which is the one thing "
        "apply_calibration exists to prevent (IMPL-09)."
    )


def _feature_calibrated(
    *,
    extraction: dict[str, Any],
    spans: dict[str, Any],
    calibrators: Any,
    thresholds: Any,
    page_text: str | None,
    common_model: bool = False,
    page_texts: dict[int, str] | None = None,
) -> CalibratedResult:
    """Per-field-type feature calibration (arch v2.1 §5.1-5.4).

    Replaces the v1 path, which collapsed each field's span to its MINIMUM token
    probability and compared it to one hardcoded threshold. The minimum of n
    draws falls as n grows, so a long correct value scored below a short wrong
    one — ``ABC-1234567-01`` is nine tokens and ``2026`` is one, and the policy
    number looked less trustworthy on every document.

    Two absences are routing instructions here, never defaults:

    * **No calibrator for this field type** — the type had too little data to fit
      one, so every field of it goes to review rather than carrying a number
      nobody measured.
    * **No threshold for this field type** — no threshold on the grid achieved
      the §0d error target with enough accepted fields, so no promise can be
      made and review is the honest outcome.
    """
    from calibration.features import build_document_features

    result = CalibratedResult()
    usable: list[float] = []

    # The span mapper returns FieldSpan objects; the feature builder takes raw
    # logprob lists. Unmapped spans are dropped here rather than passed as empty
    # ones — an empty list is indistinguishable from a null, and a field the
    # mapper could not locate is a different problem from a field the model
    # deliberately left empty.
    logprobs_by_path = {
        path: span.token_logprobs
        for path, span in spans.items()
        if getattr(span, "mapped", False)
    }
    # A common-model line's fields are typed as its schema declares them, as
    # calibration fitting types them (validation_generation.calibration_samples).
    for features in build_document_features(
        extraction=extraction, spans=logprobs_by_path, page_text=page_text,
        common_model=common_model, page_texts=page_texts,
    ):
        confidence = calibrators.predict(features) if calibrators else None
        needs_review = (
            thresholds.needs_review(features.field_type, confidence)
            if thresholds else True
        )

        if confidence is None:
            reason = (
                "no span located in the generation" if not features.is_usable
                else f"no enforced calibrator for {features.field_type} fields"
            )
            result.fields[features.field_path] = CalibratedField(
                field_path=features.field_path, value=features.value,
                raw_confidence=0.0, confidence=0.0, needs_review=True, reason=reason,
            )
            result.review_flags.append(f"{features.field_path}:no_confidence")
            continue

        result.fields[features.field_path] = CalibratedField(
            field_path=features.field_path, value=features.value,
            # The raw minimum is kept as a diagnostic, not as the confidence:
            # comparing it against the calibrated number is how a miscalibration
            # is spotted at all.
            raw_confidence=round(math.exp(features.min_logprob), 4) if features.token_count else 0.0,
            confidence=confidence, needs_review=needs_review,
            reason=(
                f"below the {features.field_type} review threshold" if needs_review else None
            ),
        )
        if needs_review:
            result.review_flags.append(f"{features.field_path}:low_confidence")
        usable.append(confidence)

    result.overall_confidence = round(sum(usable) / len(usable), 4) if usable else 0.0
    return result


def _with_every_key(output: dict[str, Any], route_: Route, lob: Any) -> dict[str, Any]:
    """``output`` with every key of its schema (common.canonical.with_all_keys)."""
    return _every_key(output, route_.schema_doc_type, route_.schema_acord_form, lob)


def _every_key(output: dict[str, Any], doc_type: str, acord_form: str | None, lob: Any) -> dict[str, Any]:
    """``output`` with every key of its schema (common.canonical.with_all_keys).

    Without the client's annotation properties (``fideon:provenance``): they are
    not the answer's keys, and filled they would be keys of nulls.
    """
    from common.canonical import with_all_keys
    from common.schemas import _strip_prefixed, load_schema

    return with_all_keys(output, _strip_prefixed(
        load_schema(doc_type, acord_form, lob), ("fideon:",)))


@dataclass(frozen=True)
class SchemaVerdict:
    """Serving's schema judgement of one answer: the audit gate's verdict."""

    #: The answer as judged before the every-key fill. On a common-model line,
    #: with its empty optional objects and empty rows dropped - served that way.
    output: Any
    #: ``output`` with every key of its schema, on a common-model line; else None.
    filled: dict[str, Any] | None
    valid: bool
    errors: tuple[str, ...] = ()


def schema_verdict(
    extraction: Mapping[str, Any],
    output: Any,
    *,
    doc_type: str,
    acord_form: str | None,
    lob: Any,
    common_model: bool,
) -> SchemaVerdict:
    """Judge ``output`` against the client's full schema, as ``extract`` does.

    ``extraction`` is the model's merged answer and ``output`` the same answer
    as served so far (enveloped, system fields filled, on a canonical line).
    Shared with evaluation, so checkpoint selection judges an answer the way
    production would return it, not a stricter copy of it.
    """
    filled = None
    unwritten: list[str] = []
    empty: list[str] = []
    if common_model:
        # A common-model line's model view is looser than the client's schema on
        # purpose (common.model_view): a window may hold a flat deductible whose
        # amount it cannot see, where the client's rule requires the key. The
        # every-key fill supplies what such an answer lacks, as a null, so the
        # served JSON is what is judged - the answer the audit gate sees. A
        # section the model view requires that no window wrote (its declarations
        # windows all failed) still fails: the system fields and the fill would
        # pass it off as an empty section nobody read.
        # Empty optional objects and empty rows are dropped from the answer
        # itself - judged and served without them - not only from a copy.
        output = _without_empty(output, keep=frozenset(required_fields(doc_type, acord_form, lob)))
        filled = _every_key(output, doc_type, acord_form, lob)
        unwritten = [
            name for name in required_fields(doc_type, acord_form, lob) if name not in extraction
        ]
        # The fill would also pass off an object written with nothing in it - a
        # carrier: {} the decoder allows, since it drops the client's
        # minProperties, or a [{}] row - as a row of nulls the client's rule
        # accepts. Emptiness is judged on the answer before the fill, after the
        # system fields (document and policy carry those, so are never empty).
        empty = list(iter_validation_errors_by_keyword(
            output, doc_type, acord_form, lob, keywords=("minProperties",)))
    judged = output if filled is None else filled
    valid = not unwritten and not empty and is_valid(judged, doc_type, acord_form, lob)
    errors: tuple[str, ...] = ()
    if not valid:
        errors = tuple(
            [f"{name}: a required section no window wrote" for name in unwritten] + empty
            + list(iter_validation_errors(judged, doc_type, acord_form, lob))
        )
    return SchemaVerdict(output=output, filled=filled, valid=valid, errors=errors)


def served_schema_verdict(
    answer: Mapping[str, Any],
    *,
    doc_type: str = "policy",
    acord_form: str | None = None,
    lob: Any = None,
    page_count: int | None = None,
    modality: str | None = None,
    source_file_name: str | None = None,
) -> SchemaVerdict:
    """A merged common-model answer judged the way ``extract`` would serve it.

    Enveloped and given the system fields first, as serving does, then
    :func:`schema_verdict`. For a scorer holding the model's merged answer but
    no serving verdict (checkpoint selection): checked as written, an answer
    lacks the system fields and the every-key fill serving adds, and fails for
    what the pipeline - not the model - supplies.
    """
    output = with_system_fields(
        envelope(dict(answer), {}), page_count=page_count, source_file_name=source_file_name,
        lob=lob, modality=modality,
    )
    return schema_verdict(
        answer, output, doc_type=doc_type, acord_form=acord_form, lob=lob, common_model=True)


def _lob_output(fields: dict[str, Any]) -> dict[str, Any] | None:
    """Collapse the per-value LoB spans into one ``{value, confidence}``.

    ``line_of_business`` is a list under arch v2.1 §0b, so the span mapper emits
    ``line_of_business[0]``, ``[1]`` … — correctly, because each emitted value has
    its own tokens and therefore its own logprob. The API returns one field, so
    they are collapsed here.

    Confidence is the **minimum** across the values, matching the aggregation used
    for a multi-token span: the weakest line is what makes the set worth
    reviewing, and averaging would let one confident line hide an invented one.

    An empty list is a real answer — the document determines no line — and is
    returned with full confidence rather than as a missing field, so it is not
    confused with a field the model failed to emit.
    """
    exact = fields.get("line_of_business")
    if exact is not None and isinstance(exact.value, list):
        return exact.as_output()

    elements = [
        (path, f) for path, f in fields.items()
        if path.startswith("line_of_business[")
    ]
    if not elements:
        # Distinguish "emitted as empty" from "never emitted". Only the former
        # is an answer.
        return {"value": [], "confidence": 1.0} if exact is not None else None

    elements.sort(key=lambda kv: kv[0])
    return {
        "value": [f.value for _, f in elements],
        "confidence": round(min(f.confidence for _, f in elements), 4),
    }


def extract(
    request: ExtractionRequest,
    model: LoadedModel,
    classifier: Classifier,
    calibration: CalibrationParams | Mapping[str, CalibrationParams],
    *,
    adapter_map: dict[str, Any] | None = None,
    #: The release bundle's fitted calibrators and thresholds for the serving
    #: format in use (arch v2.1 §5.3). Per FORMAT, always: quantization moves the
    #: logprob distribution, so a bf16 calibrator reports confidence for a
    #: distribution FP8 does not produce.
    calibrators: Any = None,
    thresholds: Any = None,
    classifier_threshold: float = 0.70,
    review_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
    strict_schema: bool = True,
    plan: Any = None,
    fallback_doc_type: str | None = None,
    release_runtimes: Mapping[str, Any] | None = None,
    detect_lob: bool = False,
    base_model_loaded: bool = True,
    lob_confidence_threshold: float = 0.90,
    lob_family_threshold: float = 0.60,
) -> ExtractionResult:
    """Run one document through the full pipeline.

    Args:
        strict_schema: raise on a schema-invalid response rather than returning
            it. Mirrors the audit gate — a response that fails validation is an
            error, not a result.
        plan: the endpoint's :class:`~serving.release_router.ServingPlan`. Given
            one, a document type no promoted release covers is refused HERE,
            before any generation — extracting it would run a model that never
            trained on the type against a schema it has never seen.
        detect_lob: read the line of a policy that arrives with none
            (:func:`_resolve_lob`). Off by default, and in the endpoint's config,
            until ``scripts/measure_lob_detection.py`` has measured the reading on
            held-out policies: a confident wrong line routes to the wrong adapter
            and schema with no flag. Off, such a policy is refused, or read
            against the fallback when the caller asked.
    """
    route_ = _resolve_route(
        request, classifier,
        confidence_threshold=classifier_threshold,
        adapter_map=adapter_map or {},
        fallback_doc_type=fallback_doc_type,
    )

    # The policy's line: the caller's, else the classifier's (or none).
    resolution = _resolve_lob(
        request, classifier, route_, detect=detect_lob,
        line_threshold=lob_confidence_threshold, family_threshold=lob_family_threshold,
    )
    if resolution.flags:
        route_ = replace(route_, review_flags=[*route_.review_flags, *resolution.flags])

    release = None
    lob_fallback_used = False
    layout_family = None
    if plan is not None:
        from serving.release_router import UnservedDocType

        try:
            # A policy no line could be found for is read by the base model
            # against the fallback, flagged - the caller need not opt in, since
            # the pipeline itself looked. A caller's unknown line is still refused
            # unless the caller asked for the fallback.
            routed = plan.route(route_.doc_type, resolution.lob,
                                allow_fallback=request.allow_lob_fallback
                                or resolution.source == "undetected")
        except UnservedDocType as exc:
            raise PipelineError(str(exc)) from exc
        release, layout_family = routed.release, routed.layout_family
        if routed.lob_fallback_used:
            if not base_model_loaded:
                # The engine is one release's merged model: reading with "no
                # adapter" would be that release, against a schema it never
                # trained on, reported as the base model.
                raise PipelineError(
                    f"{request.source_id} would be read by the base model against the fallback "
                    "schema, but this endpoint's engine is a release's merged model, not the base. "
                    "Serve the releases as LoRAs on the base (load_release_runtimes) to read "
                    "unrouted lines.")
            # The base model, no LoRA, against the canonical _fallback.json:
            # no adapter trained on this line's layout is promoted.
            lob_fallback_used = True
            from inference_core.model_runner import NO_ADAPTER

            route_ = replace(route_, adapter=NO_ADAPTER)
            # A release's calibrators were fitted to its adapter, not the base.
            calibrators = thresholds = None

    # Serve THROUGH the release the plan chose: its adapter and its calibrators.
    # The choice used to be a yes/no check and was then thrown away, so a
    # personal-lines release "answered" a homeowners policy that the unified
    # model then read, with the unified model's confidence.
    runtime = (release_runtimes or {}).get(release.release_id) if release is not None else None
    if runtime is not None:
        if runtime.adapter is not None:
            route_ = replace(route_, adapter=runtime.adapter)
        if runtime.calibrators is not None:
            calibrators, thresholds = runtime.calibrators, runtime.thresholds

    # Calibration is per document type, and which type this is only becomes
    # known once the classifier has run — so it is selected here, not by the
    # caller. The endpoint used to resolve it from the caller-supplied
    # `doc_type`, which is absent on every classification-driven request; the
    # lookup fell through to `.get("")` and the request was refused before
    # extraction ever ran.
    # The v1 transform is needed only when no fitted calibrator set applies.
    if lob_fallback_used:
        # The base model has no fitted calibrators, and the endpoint may hold no
        # v1 parameters for the type: confidence is then raw. Every field is
        # flagged for review either way (no calibrator set below).
        try:
            calibration = _calibration_for(calibration, route_.doc_type)
        except PipelineError:
            calibration = None
        if calibration is None:
            from calibration.fit_calibration import CalibrationParams

            # The identity: raw confidence, reported as a diagnostic only.
            calibration = CalibrationParams(method="temperature", doc_type=route_.doc_type,
                                            model_version=model.tag, temperature=1.0)
    else:
        calibration = _calibration_for(calibration, route_.doc_type) if calibrators is None else None

    # The line selects the policy's canonical schema. It is the caller's to
    # supply; with none, `schema_key` selects the client's canonical fallback,
    # so a policy's output is canonical JSON either way.
    lob = None if lob_fallback_used else resolution.lob
    canonical = is_canonical(route_.schema_doc_type, route_.schema_acord_form, lob)
    common_model = canonical and route_.schema_doc_type == "policy" and is_common_model(
        route_.schema_doc_type, route_.schema_acord_form, lob)

    # Every page of every document is read (decision 2026-10-19): a policy in
    # windows, a long Loss Run in windows, anything else in one call.
    latency: float | None = None
    merge_flags: list[str] = []
    reconciliation: dict[str, Any] | None = None
    window_latencies: list[float] = []
    windows = lossrun_windows(request) if route_.schema_doc_type == "lossrun" else []

    if canonical and route_.schema_doc_type == "policy":
        # Every canonical policy is read as windows — section group x page
        # window — planned by the SAME function the corpus build expanded its
        # training rows with, so a served window is a shape a training row had.
        # Always, whatever the length: a threshold computed from prompt length
        # would move between corpus builds and re-shape documents silently.
        extraction, all_spans, latency, pages_used, merge_flags = _extract_policy_windows(
            request, model, route_, lob, latencies=window_latencies,
        )
        if common_model:
            # The conventions the training labels follow, applied to the merged
            # answer against every page's text (serving.conventions): each value
            # once, citing every page that prints it; no extra field repeating a
            # field; a value no page prints flagged.
            from data_pipeline.ocr.modality import NATIVE
            from serving.conventions import conform

            page_texts = None if request.modality_mode == "image_only" else (
                dict(request.page_texts) or dict(enumerate(_all_page_texts(request), start=1)))
            conformed = conform(extraction, all_spans, page_texts,
                                native=(request.ocr_meta or {}).get("modality") == NATIVE)
            extraction, all_spans = conformed.extraction, conformed.spans
            merge_flags.extend(conformed.flags)
            if page_texts:
                merge_flags.extend(_likely_missed(extraction, page_texts, lob))
    elif len(windows) > 1:
        # A Loss Run longer than one window: read by window, merged and
        # reconciled. One that fits a window is read in one call below.
        extraction, all_spans, latency, pages_used, merge_flags, reconciliation = (
            _extract_lossrun_windows(request, model, route_, windows, latencies=window_latencies))
    else:
        # Derived from the images actually sent, not from page_texts: an
        # image_only document has no page_texts, and defaulting to [1] made
        # check_document sum table_row_counts for page 1 alone — flagging every
        # multi-page image_only extraction as row-incomplete.
        pages_used = (
            sorted(request.page_texts) if request.page_texts
            else list(range(1, len(request.image_paths) + 1))
        )
        page_ocr = None if request.modality_mode == "image_only" else _all_page_texts(request)
        # Images by page, in the order the texts are in. Sent as the caller
        # listed them, `page_10.png` listed before `page_2.png` put page 10's
        # image beside page 2's text.
        extraction, all_spans, latency = _generate_once(
            model, route_, [_image_for(request, page) for page in pages_used],
            page_ocr, request.modality_mode,
            page_numbers=pages_used if request.page_texts else None,
            lob=lob,
        )

    if not window_latencies and latency is not None:
        window_latencies.append(latency)                  # read in one call
    if common_model:
        # A coverage whose unit was read in another window comes back unlinked.
        merge_flags.extend(unit_link_flags(extraction))
    if route_.schema_doc_type == "lossrun":
        if reconciliation is None:
            from serving.lossrun_merge import reconcile_extraction

            reconciliation = reconcile_extraction(extraction).as_dict()
        # A mismatch is a claims list that does not add up to what the document
        # prints. Unverifiable (nothing printed to check against) is reported,
        # not flagged: the Loss Run schema carries no printed totals yet.
        if reconciliation.get("status") == "mismatch":
            merge_flags.append(LOSSRUN_TOTALS_MISMATCH_FLAG)

    # Calibration, completeness and the per-field output all read VALUES. For a
    # canonical document that is each envelope's `parsed`; a flat document is
    # its own values view. The spans were re-keyed to the same paths.
    values = values_view(extraction)

    # --- confidence --------------------------------------------------------
    # One span map for one generation, on both paths: a routed request is now a
    # single interleaved call rather than one call per page, so the spans
    # describe exactly the document that was returned. The previous per-page
    # loop left `all_spans` holding only the last page's map while `extraction`
    # was the merged document, so fields merged from earlier pages carried no
    # confidence and a field the declarations page won still reported the
    # rejected page's value.
    # The release bundle's calibrator set when the endpoint holds one, the v1
    # transform otherwise. Not a silent fallback: a release whose calibrate stage
    # never ran has no per-field-type curves and no measured thresholds, and
    # serving it through the v1 path is a knowing downgrade rather than an
    # equivalent.
    if calibrators is not None:
        calibrated: CalibratedResult = _feature_calibrated(
            # The model form, envelopes and all: flattened it gives the same
            # values, and it still carries each value's printed form. A
            # common-model line's ids and codes are no reading, and not scored.
            extraction=without_bare_values(extraction) if common_model else extraction,
            spans=all_spans,
            calibrators=calibrators, thresholds=thresholds,
            # The same definition validation fits the calibrators with: the OCR
            # text of the pages the model was shown, none for image_only.
            page_text=None if request.modality_mode == "image_only" else shown_ocr_text(
                [request.page_texts.get(page) for page in pages_used]
                if request.page_texts else [request.ocr_text]
            ),
            common_model=common_model,
            # Each page's own text, so a value is grounded on the pages it cites.
            page_texts=None if request.modality_mode == "image_only" else (
                {int(p): t for p, t in request.page_texts.items()} or None),
        )
    else:
        log.warning(
            "no calibrator set supplied for %s, so confidence falls back to the v1 raw "
            "aggregate, and EVERY field is flagged for review: that number is length-biased "
            "and tied to no measured error rate (arch v2.1 §5.1) — load the release bundle's "
            "calibrators to get the guarantee.", request.source_id,
        )
        raw = field_confidences(all_spans)
        calibrated = apply_calibration(raw, calibration, review_threshold=review_threshold)
        # No measured threshold, no promise. Accepting a field above a fixed 0.70
        # claimed an error rate nobody measured; the confidence is still reported,
        # as a diagnostic, but nothing is accepted on it.
        for path, f in calibrated.fields.items():
            if not f.needs_review:
                calibrated.fields[path] = replace(
                    f, needs_review=True, reason="no fitted calibrator; nothing is auto-accepted"
                )
                calibrated.review_flags.append(f"{path}:uncalibrated")

    # --- list completeness: the signal logprobs cannot see ------------------
    completeness = check_document(
        values, route_.schema_doc_type,
        ocr_meta=request.ocr_meta, pages_used=pages_used,
    )
    flags = merge_review_flags(
        completeness, calibrated.review_flags + route_.review_flags + merge_flags
        + ([LOB_FALLBACK_FLAG] if lob_fallback_used else [])
    )

    # --- the client's canonical envelope -----------------------------------
    # confidence and flagged are filled here, from the calibrated fields, never
    # by the model. A list whose row count disagrees with the document is
    # flagged on every value in it: the rows present may each be confident, and
    # the list as a whole still needs a person.
    if canonical:
        incomplete = tuple(
            flag.split(":", 1)[0] for flag in flags if is_row_completeness_flag(flag)
        )
        scores = {
            path: (
                f.confidence,
                f.needs_review or any(
                    path == name or path.startswith((f"{name}[", f"{name}."))
                    for name in incomplete
                ),
            )
            for path, f in calibrated.fields.items()
        }
        output = with_system_fields(
            envelope(extraction, scores),
            page_count=len(request.image_paths) or len(request.page_texts) or None,
            source_file_name=request.source_file_name,
            lob=lob, modality=(request.ocr_meta or {}).get("modality"),
        )
    else:
        output = extraction

    # --- schema validation, mirroring the audit gate ------------------------
    # Against the client's FULL schema for a canonical document, envelope and
    # all — the model form it was generated in is ours, the contract is theirs.
    verdict = schema_verdict(
        extraction, output, doc_type=route_.schema_doc_type,
        acord_form=route_.schema_acord_form, lob=lob, common_model=common_model,
    )
    output, filled = verdict.output, verdict.filled
    schema_valid = verdict.valid
    validation_errors = list(verdict.errors)
    if not schema_valid:
        if strict_schema:
            raise PipelineError(
                f"the extraction for {request.source_id} failed schema validation and will not be "
                f"returned (Fideon SPEC_07 Stage 3 mirrors this on every production call): "
                + "; ".join(validation_errors[:4])
            )
        flags.append("schema:invalid")

    if canonical:
        # Every key of the line's schema, whatever the model found: a value it
        # did not extract is null, not absent, so every served JSON - from the
        # base model or a trained one - has the same keys (common.canonical.
        # with_all_keys). After validation, which judges the model's own answer
        # (an answer missing a required section must still fail; on a
        # common-model line, the filled answer with its sections checked as
        # written, above), and after confidence, so the nulls carry none of it.
        output = filled if filled is not None else _with_every_key(output, route_, lob)

    result = ExtractionResult(
        source_id=request.source_id,
        doc_type=route_.schema_doc_type,
        model_version=model.tag,
        mode=request.modality_mode,
        schema_valid=schema_valid,
        overall_confidence=calibrated.overall_confidence,
        extraction=output,
        fields={
            path: f.as_output()
            for path, f in sorted(calibrated.fields.items())
            if path != "line_of_business" and "[" not in path
        },
        list_fields={
            name: {"rows": values.get(name, []), **signal.as_output()}
            for name, signal in sorted(completeness.items())
        },
        line_of_business=(resolution.as_output() if resolution.source is not None
                          else _lob_output(calibrated.fields)),
        pages_used=pages_used,
        review_flags=sorted(set(flags)),
        route_info={
            "adapter": route_.adapter or None,
            "foundation_only": route_.foundation_only,
            "classifier_confidence": (
                route_.classification.confidence if route_.classification else None
            ),
            "classifier_method": (
                route_.classification.method if route_.classification else None
            ),
            "layout_family": layout_family,
            "lob_fallback_used": lob_fallback_used,
            "lob_source": resolution.source,
        },
        latency_ms=latency,
        validation_errors=validation_errors,
        reconciliation=reconciliation,
        window_latencies_ms=window_latencies,
    )

    log.info(
        "extracted %s as %s (confidence %.3f, %d review flag(s), adapter=%s)",
        request.source_id, result.doc_type, result.overall_confidence,
        len(result.review_flags), route_.adapter or "foundation-only",
    )
    return result
