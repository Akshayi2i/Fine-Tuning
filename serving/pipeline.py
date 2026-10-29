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
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

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
    compact_schema,
    envelope,
    expand_answer,
    is_field_value,
    values_view,
    with_output_dates,
    with_system_fields,
    without_bare_values,
)
from common.constants import DEFAULT_REVIEW_CONFIDENCE_THRESHOLD
from common.full_text import build_text_sections, text_coverage
from common.prompts import fast_layout
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
from inference_core.span_map import SpanMapError, map_compact_field_spans, map_field_spans
from serving.adapter_router import Route, RoutingError, route
from serving.doc_type_classifier import Classifier
from serving.value_flags import near_tie_flags, route_served, routed_paths

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
    #: Read a policy whose line is missing or has no schema with the base
    #: model against _fallback.json, rather than refusing it. Off unless the
    #: caller asks: such a policy is never routed silently.
    allow_lob_fallback: bool = False
    #: A line of business L1/L2 proposed without being sure enough to send it as
    #: ``known_lob``: reconciled with the line the classifier reads.
    lob_hypothesis: str | None = None
    #: The order a policy's windows go to the model: ``planned`` (the longest
    #: expected answers first, :func:`_send_order`) or ``reversed``. The answers
    #: are merged in planned order either way. The golden eval's repeat pass sends
    #: them reversed, so the engine batches them differently, and measures whether
    #: the same document reads the same (``run_to_run_agreement``).
    send_order: str = "planned"
    #: The text the printed-values check holds a policy's values to (page number
    #: -> text), when it is not the text the model read: the golden eval's noisy
    #: mode shows the model degraded OCR and checks its values against the
    #: document's own text. None, the check reads ``page_texts`` (or the joined
    #: ``ocr_text``), as it always has. Also what the role check
    #: (``role_mismatch``) reads the labels' values from.
    printed_texts: Mapping[int, str] | None = None


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
    #: Milliseconds of every generation batch this document took - one per batch
    #: of windows (a batch's windows run together: a policy's in batches of up to
    #: MAX_WINDOWS_PER_BATCH, a Loss Run's one batch per round), or one for a
    #: document read in one call (release measurements).
    window_latencies_ms: list[float] = field(default_factory=list)
    #: The share of the page text's words found in the served JSON (SPEC_21's
    #: every-word rule, common.full_text.text_coverage). Common-model policies
    #: only; None for every other type and for a request with no page text.
    text_coverage: float | None = None
    #: What the printed-values check found (serving.printed_values, decision
    #: 2026-10-24): the values it could check, and each one the model wrote that
    #: no page prints - removed from the JSON in strict mode. None when no check
    #: ran: not a policy, or no page text yet (fields first: attach_full_text).
    printed_check: dict[str, Any] | None = None
    #: A policy's windows, one entry each time one was read
    #: (``_window_stat``): its group, pages, attempt, prompt and answer
    #: tokens, seconds and outcome - where the time went. Not in the served
    #: JSON; the test runners record it.
    window_stats: list[dict[str, Any]] = field(default_factory=list)
    #: Each field's feature vector as the release's calibrators scored it
    #: (``calibration.features.FieldFeatures`` by values-view path), so its
    #: confidence can be computed again on the page text that arrives after
    #: fields first served it (``calibration.post_text``). Not served: not in
    #: :meth:`as_dict`.
    confidence_features: dict[str, Any] = field(default_factory=dict, repr=False)
    #: The release calibrators and thresholds this result was scored with
    #: (:class:`CalibratedWith`), for that second scoring; None when it had none.
    #: Not served, never copied with the result.
    calibrated_with: Any = field(default=None, repr=False, compare=False)

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
            "text_coverage": self.text_coverage,
            # Loss Runs only, so every other type's output is unchanged.
            **({"reconciliation": self.reconciliation} if self.reconciliation is not None else {}),
        }


class CalibratedWith:
    """The release calibrators and thresholds a result was scored with, kept
    on it (``ExtractionResult.calibrated_with``) for fields first's second
    scoring once the page text arrives (``serving.fields_first.add_full_text``).
    Shared, never copied: a copy of the result (the fields served at the gate)
    is scored with the same release."""

    def __init__(self, calibrators: Any, thresholds: Any) -> None:
        self.calibrators = calibrators
        self.thresholds = thresholds

    def __deepcopy__(self, memo: dict[int, Any]) -> CalibratedWith:
        return self

    def __copy__(self) -> CalibratedWith:
        return self

    def __repr__(self) -> str:
        return "CalibratedWith(<the release's calibrators and thresholds>)"


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


def _numbered_pages(request: ExtractionRequest) -> ExtractionRequest:
    """``request`` with one page text per image, numbered 1..n, or :class:`PipelineError`.

    The endpoint checks this (``vllm_entrypoint``); a caller of :func:`extract`
    itself - fields first, the golden eval, the test harness - was not checked.
    Three texts for four images planned three pages: page 4 was read by no
    window and the document served as whole. No image at all was refused as "a
    single joined ocr_text". A key written as a digit string is read as the
    page number it is (a copy of the request; the caller's is not changed).
    """
    images = len(request.image_paths)
    if not images:
        raise PipelineError(
            f"{request.source_id} supplies no page images, so there is no page to read: both "
            "modes read each page's image (image_only has nothing else, ocr_plus_image reads its "
            "text beside it). Send one image per page."
        )
    if not request.page_texts:
        return request
    numbered: dict[int, str] = {}
    for key, text in request.page_texts.items():
        page = key if isinstance(key, int) and not isinstance(key, bool) else (
            int(key) if isinstance(key, str) and key.strip().isdigit() else None)
        if page is None:
            raise PipelineError(f"{request.source_id}: page_texts key {key!r} is not a page number")
        numbered[page] = text
    if sorted(numbered) != list(range(1, images + 1)):
        raise PipelineError(
            f"{request.source_id}: page_texts covers pages {sorted(numbered)} but {images} page "
            f"image(s) were sent; it must hold exactly pages 1..{images}, one text per image (an "
            "empty string for a page with none). A page with no text entry would be read by no "
            "window, or against another page's image."
        )
    return request if numbered == request.page_texts else replace(request, page_texts=numbered)


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
    if schema is not None and built.answer_format is not None:
        # The fast profile's compact answer (configs/shared/format.yaml): the
        # same schema, its envelopes written {"r", "v", "p"}, as trained.
        schema = compact_schema(schema)
    return built.messages, schema


def _window_answer_cap(group: str, doc_type: str | None) -> int:
    """A window's answer limit: its section task's reserved budget, as in training."""
    from common.config import answer_cap
    from common.schema_sections import task_for

    return answer_cap(task_for(group), doc_type)


def _parse_failure(exc: BaseException) -> str:
    """Why an answer did not parse, without a word of it: a JSON error's own
    message and position, a span-map error's kind and offset (its message quotes
    the character or the trailing text it met)."""
    if isinstance(exc, json.JSONDecodeError):
        return f"{exc.msg} at line {exc.lineno} column {exc.colno}"
    if isinstance(exc, SpanMapError):
        offset = re.search(r"offset (\d+)", str(exc))
        return f"the answer's structure is broken{f' at offset {offset.group(1)}' if offset else ''}"
    return type(exc).__name__


def _parse_generation(
    result: Generation, route_: Route, *, refuse_truncated: bool = False, compact: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(extraction, spans)`` from one generation, or :class:`PipelineError`.

    The extraction comes back with every date in ``MM/DD/YYYY`` and, for a
    canonical document, still in the model's sparse ``raw``/``parsed``/
    ``page_ref`` form; the spans are keyed by values-view path.

    ``refuse_truncated`` treats output cut off at the token limit as a failure
    even when what was cut still parses. A window can afford that: it is retried
    over fewer pages. Parsing a truncated answer as complete would drop every
    row after the cut with nothing to say so.

    ``compact``: the answer is in the fast profile's compact form
    (``common.canonical.to_compact_answer``). It is expanded to the full
    envelope here, and its spans keyed as the full form's
    (``span_map.map_compact_field_spans``), so nothing after this - the shape
    check, the page check, the merge, the printed-values check, calibration -
    ever sees the compact form.
    """
    if refuse_truncated and result.truncated():
        raise PipelineError(
            f"the {route_.schema_doc_type} generation was cut off (finish {result.finish_reason}: the "
            "token limit, an answer stopped for repeating itself, or a stream that broke off), so its "
            "output is incomplete"
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
        # Each token's best alternative too, where the backend gave one: each
        # value's near-tie margin (serving.value_flags.near_tie_flags).
        top2 = getattr(result, "token_top2_logprobs", None) or None
        if compact:
            spans = map_compact_field_spans(result.text, result.tokens, result.token_logprobs,
                                            token_top2_logprobs=top2)
            extraction = expand_answer(extraction)
        else:
            spans = map_field_spans(result.text, result.tokens, result.token_logprobs,
                                    token_top2_logprobs=top2)
    except (ValueError, TypeError, SpanMapError) as exc:
        # The answer is never quoted: it holds the document's values, and this
        # message is logged, kept in the window's stats and returned in a job's error.
        raise PipelineError(
            f"the model did not return parseable JSON for {route_.schema_doc_type} "
            f"({len(result.text)} characters; {_parse_failure(exc)}). The prompt asks for JSON only "
            f"with no fences, so this is a generation failure rather than a formatting quirk to work around."
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


#: The most windows sent to the model in one call. Within a call vLLM runs as
#: many windows as its memory holds and starts the next the moment one ends; a
#: call ends with its longest answer, and the next call cannot start before it,
#: so every call boundary leaves the GPU waiting on one long window. At 32 a
#: call, about one call in four of the CGL checkpoint selection waited ~140 s on
#: one long schedule with the GPU nearly idle, ~25% of its time; 96 (the
#: validation's `validation_batch_rows`) reads a 69-page policy, 81 windows, in
#: one call. Bounded, not the whole document: every window's page images are
#: decoded before the engine starts on any, and a 200-page policy is some 230
#: windows and 800 page images. Not yet measured in serving: record a pod
#: measurement here when one is taken.
MAX_WINDOWS_PER_BATCH = 96


def _extract_policy_windows(
    request: ExtractionRequest,
    model: LoadedModel,
    route_: Route,
    lob: str | list[str] | None,
    *,
    latencies: list[float] | None = None,
    stats: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float | None, list[int], list[str]]:
    """Read a canonical policy window by window, then merge.

    Returns ``(extraction, spans, latency_ms, pages_used, flags)`` — the merged
    model-form document and its spans, in the shape the single-call path
    returns, so confidence, completeness and the envelope run on it unchanged.
    ``stats`` receives one entry per window read (:func:`_window_stat`): its
    pages, tokens, seconds and what came of it.

    **Concurrent.** Every window goes to the model at once, the longest expected
    answers first (:func:`_send_order`), and vLLM schedules them together, so a
    17-window policy costs roughly the time of its slowest window rather than
    the sum of all of them. A backend that answers each window as it finishes
    (a served model, :func:`inference_core.model_runner.can_submit`) is read
    that way (:func:`_read_as_answered`): a failed window's halves go out the
    moment it fails, not after the slowest window of its round. The in-process
    engine runs a call to its end, so with it the windows go in rounds of calls
    of up to :data:`MAX_WINDOWS_PER_BATCH` (:func:`_read_in_rounds`). Either way
    only the sending is reordered: the answers are merged in planned order,
    round by round, so the merge sees the same windows in the same order
    whichever finished first. ``latency_ms`` is the wall time actually spent.

    **One refused prompt does not fail its call.** vLLM fails every prompt of a
    call when it refuses one - a page too large for the context window, say -
    so a call whose windows all failed with one error is read again in halves
    (:func:`_batch_in_halves`), and only the window that caused it fails.

    **One bad window does not lose the document.** A window whose output is not
    parseable JSON, answers a section in a shape its schema refuses (a bare
    string for an object), or was cut off - at the token limit, the usual cause
    on a dense schedule page, or stopped repeating itself - is split in half and
    retried, since fewer pages means less to write. Retrying it unchanged would
    be pointless: decoding is greedy, so it would write the same thing again. A
    window that still fails at a single page is dropped and flagged
    ``window_failed``, so the document comes back with everything else and a
    person is told which pages it could not read. Only when no window at all
    succeeds is the request an error. The declarations lead the half holding
    their page, as they led the window (:func:`_halves`).

    **A window cites only the pages it was shown.** A page reference to any
    other page - past the end of the document, page 0, a page of another
    window - is dropped from the value and the value flagged
    (:data:`PAGE_REF_OUTSIDE_WINDOW_FLAG`), as training's targets never cite one.
    """
    from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
    from inference_core.model_runner import can_submit
    from serving.policy_merge import merge_policy_windows

    image_only = request.modality_mode == "image_only"
    page_count = len(request.page_texts) or len(request.image_paths)
    texts: dict[int, str] = {}
    if not image_only:
        texts = dict(request.page_texts) or dict(enumerate(_all_page_texts(request), start=1))

    # Every page, with the declarations found, and each window sized, from the
    # same text by the same rule as the training rows were - and the window its
    # declarations lead marked as training marks it (PolicyWindowPlan.leads):
    # the merge keeps that window's values first within its group.
    every_page, declarations_page = pages_to_read(texts or None, page_count)
    plan: list[_Planned] = [
        _Planned(window.group, list(window.pages), window.leads)
        for window in plan_windows(lob, every_page, declarations_page, None if image_only else texts)
    ]
    shapes = {group: _section_shapes(route_, lob, group) for group in {window.group for window in plan}}
    # Which groups answer compact: all of them under the fast profile
    # (common.prompts.fast_layout), none otherwise.
    compact = {group: fast_layout(route_.schema_doc_type, route_.schema_acord_form, lob, group)
               for group in shapes}

    # Each window's messages until its answer is settled: the prompt's length
    # the backend reports is held to what its pages imply (_check_prompt_tokens).
    prompts: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
    warned: list[str] = []

    def settle(window: _Planned, result: Any, attempt: int) -> _Settled:
        settled = _settle(request.source_id, route_, window, result, attempt,
                          declarations_page=declarations_page, shapes=shapes[window.group],
                          compact=compact[window.group])
        _check_prompt_tokens(request.source_id, settled.stat, result,
                             prompts.pop((window.group, tuple(window.pages)), None), warned)
        return settled

    def window_request(group: str, pages: list[int]) -> tuple[Any, ...]:
        messages, schema = _build_request(
            model, route_,
            [_image_for(request, page) for page in pages],
            None if image_only else [texts.get(page) or _EMPTY_PAGE for page in pages],
            request.modality_mode,
            page_numbers=pages, total_pages=page_count,
            lob=lob, sections=group,
        )
        prompts[(group, tuple(pages))] = messages
        return messages, schema, _window_answer_cap(group, route_.schema_doc_type)

    read_with = _read_as_answered if can_submit(model) else _read_in_rounds
    read = read_with(request, model, route_, plan, window_request, settle)
    if latencies is not None:
        latencies.extend(read.calls_ms)
    if stats is not None:
        stats.extend(read.stats)

    if read.lost:
        # Never a document served as read when part of it was never sent to the
        # model: dropped and flagged, those windows read as a finished job with
        # holes in it (a pod that went down mid-document answered 502 to every
        # window after it). Refused, the caller tries the document again.
        raise PipelineError(
            f"{len(read.lost)} window(s) of {request.source_id} never reached the model "
            f"({', '.join(read.lost[:6])}): the server could not be reached or failed, so the "
            f"document was not fully read and is not served. The first failure: {read.first_cause}"
        )
    if not read.windows:
        raise PipelineError(
            f"no window of {request.source_id} could be read ({len(read.failed)} failed: "
            f"{', '.join(read.failed[:6])}). The first failure: {read.first_cause}"
        )

    merged = merge_policy_windows(read.windows, lob=lob)
    flags = (merged.review_flags + [f"{WINDOW_FAILED_FLAG}:{entry}" for entry in read.failed]
             + sorted(set(read.flags)))
    log.info(
        "%s read in %d window(s) over %d page(s) in %d round(s); %d duplicate row(s) "
        "collapsed, %d conflict(s), %d window(s) unreadable", request.source_id, len(read.windows),
        len(every_page), read.rounds, merged.duplicates_collapsed, len(merged.conflicts), len(read.failed),
    )
    return merged.extraction, merged.spans, round(read.wall_ms, 1), sorted(every_page), flags


@dataclass
class _WindowsRead:
    """What reading a policy's windows came to, in merge order."""

    windows: list[Any] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    #: The windows in ``failed`` that never reached the model (transport).
    lost: list[str] = field(default_factory=list)
    first_cause: str | None = None
    rounds: int = 0
    wall_ms: float = 0.0
    #: One entry per call to the model; one for the whole reading when the
    #: windows were answered one by one (``window_latencies_ms``).
    calls_ms: list[float] = field(default_factory=list)
    stats: list[dict[str, Any]] = field(default_factory=list)
    #: Review flags the windows' answers raised (:data:`PAGE_REF_OUTSIDE_WINDOW_FLAG`).
    flags: list[str] = field(default_factory=list)


class _Planned(NamedTuple):
    """One window to read: a section group over pages, and whether its group's
    declarations lead it (``PolicyWindowPlan.leads``)."""

    group: str
    pages: list[int]
    leads: bool = False


@dataclass
class _Settled:
    """One window's answer: read, split into two smaller windows, or failed."""

    stat: dict[str, Any]
    window: Any = None
    retry: list[_Planned] = field(default_factory=list)
    failed: str | None = None
    cause: str | None = None
    flags: list[str] = field(default_factory=list)
    #: ``failed`` because the call itself failed (``ModelTransportError``): the
    #: window never reached the model, whatever its pages hold.
    lost: bool = False


#: The review flag on a value whose page reference named a page its window was
#: not shown - past the document's end, page 0, another window's page - or no
#: page number at all. That reference is dropped; the value is kept, flagged.
PAGE_REF_OUTSIDE_WINDOW_FLAG = "page_ref_outside_window"


def _settle(source_id: str, route_: Route, window: _Planned, result: Any, attempt: int, *,
            declarations_page: int | None = None,
            shapes: Mapping[str, str] | None = None, compact: bool = False) -> _Settled:
    """What one window's answer comes to (see :func:`_extract_policy_windows`).

    ``shapes`` is the JSON type each section of the window's schema must be
    (:func:`_section_shapes`); an answer giving another is refused as one that
    does not parse is. ``compact``: the answer is in the fast profile's compact
    form, expanded to full envelopes before anything reads it
    (:func:`_parse_generation`).
    """
    from serving.policy_merge import PolicyWindow

    group, pages = window.group, window.pages
    from inference_core.model_runner import ModelTransportError

    lost = isinstance(result, ModelTransportError)
    try:
        if isinstance(result, Exception):
            raise PipelineError(str(result))
        extraction, spans = _parse_generation(result, route_, refuse_truncated=True, compact=compact)
        _check_section_shapes(extraction, shapes or {}, route_)
    except PipelineError as exc:
        if len(pages) > 1:
            log.warning("%s: %s window over pages %s failed (%s); retrying as two smaller windows",
                        source_id, group, pages, exc)
            return _Settled(_window_stat(group, pages, attempt, result, "split", str(exc)),
                            retry=_halves(window, declarations_page), cause=str(exc))
        log.warning("%s: %s window over page %s failed even alone (%s); dropped and flagged for review",
                    source_id, group, pages[0], exc)
        return _Settled(_window_stat(group, pages, attempt, result, "failed", str(exc)),
                        failed=f"{group}:p{pages[0]}", cause=str(exc), lost=lost)
    outside = _cite_shown_pages(extraction, pages)
    if outside:
        log.warning("%s: %s window over pages %s cited pages it was not shown at %s; those "
                    "references dropped and flagged", source_id, group, pages, ", ".join(outside[:6]))
    return _Settled(_window_stat(group, pages, attempt, result, "read"),
                    window=PolicyWindow(group, pages, extraction, spans, result.latency_ms,
                                        leads=window.leads),
                    flags=[f"{path}:{PAGE_REF_OUTSIDE_WINDOW_FLAG}" for path in outside])


def _halves(window: _Planned, declarations_page: int | None) -> list[_Planned]:
    """A failed window as two smaller ones, in page order.

    The declarations lead the half holding their page - the first half when
    neither does - as they led the window: the rule training marks its windows
    by (``policy_windows._home_window``). Led by neither, a declarations window
    cut off on the declarations page lost its place to the half holding page 1,
    and a renewal notice's prior policy number was served.
    """
    half = len(window.pages) // 2
    pieces = [window.pages[:half], window.pages[half:]]
    home = next((index for index, pages in enumerate(pieces) if declarations_page in pages), 0)
    return [_Planned(window.group, pages, window.leads and index == home)
            for index, pages in enumerate(pieces)]


def _section_shapes(route_: Route, lob: str | list[str] | None, group: str) -> dict[str, str]:
    """Each top-level section of a window's schema slice and the JSON type its
    answer must be: ``object`` or ``array``. A section given by reference
    (``#/$defs/policy``, on a common-model line) is looked up."""
    schema = resolved_schema(route_.schema_doc_type, route_.schema_acord_form, lob, group)
    defs = schema.get("$defs") or {}
    shapes: dict[str, str] = {}
    for name, node in (schema.get("properties") or {}).items():
        ref = node.get("$ref") if isinstance(node, dict) else None
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            node = defs.get(ref.rsplit("/", 1)[-1])
        kind = node.get("type") if isinstance(node, dict) else None
        kinds = {kind} if isinstance(kind, str) else set(kind or ())
        shape = next((k for k in ("object", "array") if k in kinds), None)
        if shape:
            shapes[name] = shape
    return shapes


def _check_section_shapes(extraction: dict[str, Any], shapes: Mapping[str, str], route_: Route) -> None:
    """Refuse an answer that gives a section in another JSON type than its schema's.

    Structured decoding cannot write one; without it a model can, and a policy
    section answered as a bare string (``"policy": "WC-1"``) passed the merge
    and failed the request with a TypeError where the system fields are written
    into it. The same generation failure as an answer that does not parse.
    """
    wrong = [
        f"{name} ({type(value).__name__}, not an {shape})"
        for name, shape in shapes.items()
        if (value := extraction.get(name)) is not None
        and not isinstance(value, dict if shape == "object" else list)
    ]
    if wrong:
        raise PipelineError(
            f"the model answered {route_.schema_doc_type} section(s) in a shape their schema does "
            f"not allow: {', '.join(wrong)}"
        )


def _cite_shown_pages(extraction: dict[str, Any], pages: list[int]) -> list[str]:
    """Every ``page_ref`` in a window's answer held to the pages it was shown,
    in place; the paths of the values whose references named another.

    A reference written as a string or a bare number is read as the page it
    names (``"12"`` is page 12, never pages 1 and 2, as iterating the string
    made it); any reference that names no page the window shows is dropped.
    Structured decoding bounds references to the window's pages
    (``with_page_bounds``); without it nothing did, and ``[99]`` on an
    eight-page policy was served as valid.
    """
    shown = set(pages)
    outside: list[str] = []

    def number(page: Any) -> int | None:
        if isinstance(page, bool):
            return None
        if isinstance(page, int):
            return page
        if isinstance(page, str) and page.strip().isdigit():
            return int(page.strip())
        return None

    def walk(node: Any, path: str) -> None:
        if is_field_value(node):
            cited = node.get("page_ref")
            if cited is None:
                return
            listed = cited if isinstance(cited, list) else [cited]
            numbers = [number(page) for page in listed]
            dropped = any(page not in shown for page in numbers)
            if dropped:
                outside.append(path)
            if dropped or listed is not cited or numbers != listed:
                node["page_ref"] = list(dict.fromkeys(page for page in numbers if page in shown))
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for item in node:
                walk(item, f"{path}[]")

    walk(extraction, "")
    return sorted(set(outside))


def _window_stat(group: str, pages: list[int], attempt: int, result: Any, outcome: str,
                 error: str | None = None) -> dict[str, Any]:
    """One window read, for the record: where the time went and what came of it.

    ``seconds`` is the window's own time where the backend answers it alone, and
    its call's where a call's windows run to one end (the in-process engine).
    ``first_token_s`` is the time to its first answer token - queued, then
    reading the prompt - and the rest is writing.
    """
    generation = result if isinstance(result, Generation) else None

    def seconds(ms: float | None) -> float | None:
        return round(ms / 1000, 2) if ms is not None else None

    return {
        "group": group,
        "pages": list(pages),
        "attempt": attempt,
        "outcome": outcome,
        "seconds": seconds(generation.latency_ms) if generation else None,
        "first_token_s": seconds(generation.first_token_ms) if generation else None,
        "prompt_tokens": generation.prompt_tokens if generation else None,
        "answer_tokens": len(generation.tokens) if generation else None,
        "finish": generation.finish_reason if generation else None,
        **({"error": error[:200]} if error else {}),
    }


def _check_prompt_tokens(source_id: str, stat: dict[str, Any], result: Any,
                         messages: list[dict[str, Any]] | None, warned: list[str]) -> None:
    """The prompt's length the backend reports (``prompt_tokens``), held to
    what the window's pages and text imply (``inference_core.prompt_tokens``),
    recorded in the window's stat as ``prompt_check``. One implausible length
    is warned about once per document: a server that resizes pages to another
    budget than the request's shows the model pages it was not trained on, and
    nothing else would say so. Never fails a window."""
    if messages is None or not isinstance(result, Generation) or result.prompt_tokens is None:
        return
    from inference_core.prompt_tokens import check_prompt, expected_prompt

    try:
        checked = check_prompt(result.prompt_tokens, expected_prompt(messages))
    except Exception as exc:  # noqa: BLE001 - a diagnostic: it never costs the window
        log.debug("%s: the prompt-length check could not run: %s", source_id, exc)
        return
    if checked is None:
        return
    stat["prompt_check"] = checked
    if not checked["plausible"] and not warned:
        warned.append(source_id)
        log.warning(
            "%s: the model server counted %d prompt tokens for a window over pages %s, where its "
            "pages are %d image tokens at the pixel budget sent (ratio %s, expected %s): the server "
            "may resize pages to another budget than the request's (mm_processor_kwargs) - the model "
            "then reads pages it was not trained on", source_id, checked["prompt_tokens"],
            stat.get("pages"), checked["image_tokens"], checked["ratio"], checked["expected"])


def _read_in_rounds(request: ExtractionRequest, model: LoadedModel, route_: Route,
                    plan: list[_Planned], window_request: Any, settle: Any) -> _WindowsRead:
    """Every window in calls of up to :data:`MAX_WINDOWS_PER_BATCH`; the failed
    ones' halves in the next round, once the whole round has finished.
    ``settle(window, result, attempt)`` is what one answer comes to
    (:func:`_settle`)."""
    read = _WindowsRead()
    pending = plan
    while pending:
        read.rounds += 1
        # Indexed by planned position: sent in another order, read back in this one.
        results: list[Any] = [None] * len(pending)
        order = _sending(request, pending, route_.schema_doc_type)
        for start in range(0, len(order), MAX_WINDOWS_PER_BATCH):
            positions = order[start:start + MAX_WINDOWS_PER_BATCH]
            requests = [window_request(pending[position].group, pending[position].pages)
                        for position in positions]
            batch, batches = _batch_in_halves(model, requests, route_.adapter, request.source_id)
            for position, result in zip(positions, batch, strict=True):
                results[position] = result
            read.wall_ms += sum(batches)
            # One entry per call: its windows run together, and vLLM gives
            # each the call's wall time, so one per window would count it n times.
            read.calls_ms.extend(round(ms, 1) for ms in batches)

        retry: list[_Planned] = []
        for window, result in zip(pending, results, strict=True):
            settled = settle(window, result, read.rounds)
            _record(read, settled)
            retry += settled.retry
        pending = retry
    return read


def _record(read: _WindowsRead, settled: _Settled) -> None:
    """One settled window, added to what the reading came to."""
    read.stats.append(settled.stat)
    read.first_cause = read.first_cause or settled.cause
    if settled.window is not None:
        read.windows.append(settled.window)
    if settled.failed:
        read.failed.append(settled.failed)
        if settled.lost:
            read.lost.append(settled.failed)
    read.flags.extend(settled.flags)


def _read_as_answered(request: ExtractionRequest, model: LoadedModel, route_: Route,
                      plan: list[_Planned], window_request: Any, settle: Any) -> _WindowsRead:
    """Every window sent at once and taken as it is answered; a failed window's
    halves sent the moment it fails.

    The answers are put back in the order :func:`_read_in_rounds` reads them -
    by round, then by planned position, a half after the other - so the merge
    is the same whichever window finished first. A window's place is its path:
    its planned position, then 0 or 1 for each halving.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    from inference_core.model_runner import submit_generation

    started = time.perf_counter()
    running: dict[Any, tuple[tuple[int, ...], _Planned]] = {}

    def send(path: tuple[int, ...], window: _Planned) -> None:
        future = submit_generation(model, window_request(window.group, window.pages), adapter=route_.adapter)
        running[future] = (path, window)

    for position in _sending(request, plan, route_.schema_doc_type):
        send((position,), plan[position])

    settled_at: list[tuple[tuple[int, tuple[int, ...]], _Settled]] = []
    while running:
        finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
        for future in finished:
            path, window = running.pop(future)
            settled = settle(window, future.result(), len(path))
            settled_at.append(((len(path), path), settled))
            for half, retry in enumerate(settled.retry):
                send((*path, half), retry)

    settled_at.sort(key=lambda item: item[0])
    wall_ms = (time.perf_counter() - started) * 1000
    read = _WindowsRead(rounds=max(place[0] for place, _ in settled_at) if settled_at else 0,
                        wall_ms=wall_ms, calls_ms=[round(wall_ms, 1)])
    for _place, settled in settled_at:
        _record(read, settled)
    return read


def _send_order(pending: list[Any], doc_type: str | None) -> list[int]:
    """The positions of ``pending``, in the order they go to the model.

    The windows expected to write the longest answers first: the larger answer
    reservation (a schedule's), then more pages; otherwise as planned. A call
    ends when its longest answer does, so started first that answer runs beside
    the short ones instead of after them. The validation orders its calls the
    same way (evaluation.validation_generation), by the gold answer's length,
    which serving does not have. A wrong guess costs speed, never an answer.

    Under the fast format profile every group reads the same windows, and a
    window's requests share every token up to their group block: they go
    together, so the engine admits them one after another and each after the
    first finds the pages' prefill in the prefix cache, not evicted by the time
    the window's last group comes round. Windows in the same order as above -
    the one with the largest reservation among its groups, then more pages -
    and within a window its groups by their reservation.
    """
    from common.config import fast_profile

    caps = {window[0]: _window_answer_cap(window[0], doc_type) for window in pending}
    if doc_type == "policy" and fast_profile():
        widest: dict[tuple[int, ...], int] = {}
        for group, pages, *_rest in pending:
            widest[tuple(pages)] = max(widest.get(tuple(pages), 0), caps[group])

        def by_window(position: int) -> tuple[Any, ...]:
            group, pages = pending[position][0], tuple(pending[position][1])
            return (-widest[pages], -len(pages), pages, -caps[group])

        return sorted(range(len(pending)), key=by_window)
    return sorted(range(len(pending)),
                  key=lambda position: (-caps[pending[position][0]], -len(pending[position][1])))


def _sending(request: ExtractionRequest, pending: list[Any],
             doc_type: str | None) -> list[int]:
    """:func:`_send_order`, reversed when the request asks (``send_order``)."""
    order = _send_order(pending, doc_type)
    if request.send_order == "reversed":
        return order[::-1]
    if request.send_order != "planned":
        raise PipelineError(f"send_order is {request.send_order!r}; expected 'planned' or 'reversed'")
    return order


def _batch_in_halves(
    model: LoadedModel, requests: list[Any], adapter: Any, source_id: str,
) -> tuple[list[Any], list[float]]:
    """``requests`` in one call: the results in order, and each call's wall time
    in milliseconds.

    A call the engine refused whole (:func:`_whole_batch_failed`) is read again
    in halves until the window that caused it is alone, so only that window
    fails, in about 2 log2(n) calls. Window by window, a call of
    :data:`MAX_WINDOWS_PER_BATCH` was as many calls again, one after another.
    """
    results, ms = _timed_batch(model, requests, adapter)
    if len(requests) == 1 or not _whole_batch_failed(results):
        return results, [ms]
    log.warning(
        "%s: every window of a call of %d failed with one error (%s); reading them again "
        "in halves", source_id, len(requests), results[0],
    )
    middle = len(requests) // 2
    first, first_ms = _batch_in_halves(model, requests[:middle], adapter, source_id)
    second, second_ms = _batch_in_halves(model, requests[middle:], adapter, source_id)
    return first + second, [ms, *first_ms, *second_ms]


def _timed_batch(model: LoadedModel, requests: list[Any], adapter: Any) -> tuple[list[Any], float]:
    """``generate_batch`` and its wall time in milliseconds."""
    started = time.perf_counter()
    results = generate_batch(model, requests, adapter=adapter)
    return results, (time.perf_counter() - started) * 1000


def _whole_batch_failed(results: list[Any]) -> bool:
    """Whether every result is the same error: the engine refused the batch, not
    each window on its own (``VLLMBackend.generate_batch`` gives every slot the
    exception)."""
    return (bool(results) and all(isinstance(r, Exception) for r in results)
            and len({str(r) for r in results}) == 1)


def _likely_missed(extraction: dict[str, Any], page_texts: dict[int, str], lob: Any) -> list[str]:
    """Declarations fields left empty whose label and a value of their type open
    a line of a page the declarations windows read - every page
    (serving.conventions)."""
    from serving.conventions import declaration_objects, likely_missed

    return likely_missed(extraction, page_texts, sorted(page_texts), declaration_objects(lob))


def _role_mismatches(extraction: dict[str, Any], page_texts: Mapping[int, str], lob: Any) -> list[str]:
    """Declarations values served under a field whose printed label is followed
    by another value on every line it opens (serving.conventions)."""
    from serving.conventions import declaration_objects, role_mismatches

    try:
        blocks = declaration_objects(lob)
    except Exception:  # noqa: BLE001 - a line with no declarations group has nothing to hold
        return []
    return role_mismatches(extraction, page_texts, sorted(page_texts), blocks)


def _printed_texts(request: ExtractionRequest) -> dict[int, str] | None:
    """The page text a policy's values are held to (``serving.printed_values``):
    the request's ``printed_texts`` when it carries one (the golden eval's noisy
    mode: the model read degraded text, its values are held to the document's
    own), else the text it read - None for an image-only read, which has none
    until fields first's MinerU text arrives."""
    if request.printed_texts is not None:
        held_to: dict[int, str] = {}
        for page, text in request.printed_texts.items():
            if isinstance(page, bool) or not str(page).strip().isdigit():
                raise PipelineError(f"{request.source_id}: printed_texts key {page!r} is not a page number")
            held_to[int(str(page).strip())] = text or ""
        return held_to
    if request.modality_mode == "image_only":
        return None
    return dict(request.page_texts) or dict(enumerate(_all_page_texts(request), start=1))


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

#: A common-model policy served with no page text - an image-only request - so
#: with no full-text tier: its wording is in no part of the JSON.
NO_FULL_TEXT_FLAG = "text_sections:no_page_text"


def _full_text_pages(request: ExtractionRequest) -> list[str]:
    """The page text the full-text tier is built from: one string per page, in
    page order, as the request carried it (no placeholder for a blank page).
    Empty for a request with none - an image-only one, or a multi-page request
    with only a joined ``ocr_text``, which cannot be split back into pages."""
    if request.page_texts:
        return [request.page_texts.get(page) or "" for page in sorted(request.page_texts)]
    if len(request.image_paths) == 1 and request.ocr_text:
        return [request.ocr_text]
    return []


#: A full-text tier built after the fields that did not pass the client's
#: schema: the fields stand, served without it.
FULL_TEXT_INVALID_FLAG = "text_sections:invalid"


def attach_full_text(result: ExtractionResult, page_texts: dict[int, str], *, lob: str) -> ExtractionResult:
    """The full-text tier, added to a result whose fields were read without page
    text (fields first: the model read the page images while the one GPU's
    MinerU waited, ``serving.fields_first``).

    As :func:`extract` builds it for a request that carries the text: Fideon's
    section builder over the page text (``common.full_text``), the result
    checked against the client's schema with it, ``text_coverage`` measured on
    what is served. Only a result flagged ``text_sections:no_page_text`` gets
    one - :func:`extract` flags exactly the documents that have the tier (a
    common-model policy) and were read without text; any other is left as it
    is. A tier that fails the schema is left out and flagged, the fields
    untouched. An image-only read is also looked over for declarations fields
    left empty that the text prints (``likely_missed``), as :func:`extract`
    looks over a read that had its text. Returns ``result``.
    """
    if page_texts and result.doc_type == "policy" and result.printed_check is None:
        # Printed values only (decision 2026-10-24): the fields were read from
        # the images alone, so this is the first text to hold them to - for
        # every policy, not only the lines with a full-text tier.
        from serving.printed_values import hold_result_to_printed

        hold_result_to_printed(result, page_texts)
    if page_texts and result.doc_type == "policy" and result.mode == "image_only":
        # Read from the images alone, the fields were never looked for on the
        # page text: a declarations field left empty while a page prints its
        # label and a value is flagged now, as extract flags it with text - and
        # so is one served under a label that prints another value.
        missed = _likely_missed(result.extraction, page_texts, lob)
        missed += _role_mismatches(result.extraction, page_texts, lob)
        if missed:
            result.review_flags = sorted({*result.review_flags, *missed})
            route_served(result.extraction, result.review_flags)
    if NO_FULL_TEXT_FLAG not in result.review_flags or not page_texts:
        return result
    pages = [page_texts.get(page) or "" for page in sorted(page_texts)]
    sections = build_text_sections(pages)
    if not sections:
        return result                                    # still flagged text_sections:no_page_text
    output = {**result.extraction, "text_sections": sections}
    # The tier is judged by the errors it brings: fields that already break the
    # schema (flagged schema:invalid when they were read) must not cost the
    # document its full text too.
    already = set(iter_validation_errors(result.extraction, result.doc_type, None, lob))
    errors = [error for error in iter_validation_errors(output, result.doc_type, None, lob) if error not in already]
    if errors:
        # The path only: the schema's message quotes the value it refused.
        log.warning("%s: the full-text tier fails the schema (at %s, %d error(s)); served without it",
                    result.source_id, errors[0].partition(": ")[0], len(errors))
        result.review_flags = sorted({*result.review_flags, FULL_TEXT_INVALID_FLAG})
        return result
    result.extraction = output
    result.review_flags = [flag for flag in result.review_flags if flag != NO_FULL_TEXT_FLAG]
    result.text_coverage = text_coverage(pages, output)
    return result


@dataclass
class LobResolution:
    """Which line of business a policy is read as, and where that came from.

    ``source``: ``caller`` (sent with the request, as L1/L2 do), ``detected``
    (the classifier, confidently), ``detected_uncertain`` (the classifier, below
    the line threshold: its line's schema, flagged), or
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
    detect: bool, line_threshold: float, uncertain_threshold: float,
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
    if classification.lob is None or (confidence is not None and confidence < uncertain_threshold):
        found.lob, found.source, found.flags = None, "undetected", ["lob:undetected"]
    elif confidence is not None and confidence >= line_threshold:
        found.source = "detected"
    else:
        # Below the line threshold, or no logprobs to measure it by: the line is
        # used, and a person should check it.
        found.source, found.flags = "detected_uncertain", ["lob:uncertain"]
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
    modality_mode: str | None = None,
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

    # The mapped FieldSpan objects themselves: beside their token logprobs the
    # feature builder reads whether the printed text was written apart from the
    # value (the span is the envelope's raw: r_written) and how near another
    # token came (top2_margin). Unmapped spans are dropped here rather than
    # passed as empty ones — an empty list is indistinguishable from a null, and
    # a field the mapper could not locate is a different problem from a field
    # the model deliberately left empty.
    mapped_spans = {path: span for path, span in spans.items() if getattr(span, "mapped", False)}
    # A common-model line's fields are typed as its schema declares them, as
    # calibration fitting types them (validation_generation.calibration_samples).
    for features in build_document_features(
        extraction=extraction, spans=mapped_spans, page_text=page_text,
        common_model=common_model, page_texts=page_texts, modality_mode=modality_mode,
    ):
        # Kept, so the text-dependent features can be computed again when the
        # page text arrives after the fields (calibration.post_text).
        result.features[features.field_path] = features
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
    lob_uncertain_threshold: float = 0.60,
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
    request = _numbered_pages(request)
    route_ = _resolve_route(
        request, classifier,
        confidence_threshold=classifier_threshold,
        adapter_map=adapter_map or {},
        fallback_doc_type=fallback_doc_type,
    )

    # The policy's line: the caller's, else the classifier's (or none).
    resolution = _resolve_lob(
        request, classifier, route_, detect=detect_lob,
        line_threshold=lob_confidence_threshold, uncertain_threshold=lob_uncertain_threshold,
    )
    if resolution.flags:
        route_ = replace(route_, review_flags=[*route_.review_flags, *resolution.flags])

    release = None
    lob_fallback_used = fallback = False
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
        release = routed.release
        fallback = routed.lob_fallback_used
    elif route_.doc_type == "policy" and resolution.lob:
        # No serving plan - fields first, the golden eval, compare_models: a
        # line no schema has is refused as the plan refuses it, or read against
        # the fallback, flagged, when the caller asked. It used to be read
        # against the fallback with whatever adapter was routed, unflagged.
        from serving.release_router import _unroutable

        problem = _unroutable(resolution.lob)
        if problem and not request.allow_lob_fallback:
            raise PipelineError(f"{request.source_id}: {problem}")
        fallback = bool(problem)
    if fallback:
        if not base_model_loaded:
            # The engine is one release's merged model: reading with "no
            # adapter" would be that release, against a schema it never
            # trained on, reported as the base model.
            raise PipelineError(
                f"{request.source_id} would be read by the base model against the fallback "
                "schema, but this endpoint's engine is the release's merged model, not the base: "
                "the release covers every known line, so the engine holds no base model. A "
                "policy with no line, a package or an undetected line cannot be read here; send "
                "its single line of business.")
        # The base model, no LoRA, against the canonical _fallback.json: no
        # adapter trained on this line's layout is promoted.
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
    window_stats: list[dict[str, Any]] = []
    printed_check: dict[str, Any] | None = None
    windows = lossrun_windows(request) if route_.schema_doc_type == "lossrun" else []

    if canonical and route_.schema_doc_type == "policy":
        # Every canonical policy is read as windows — section group x page
        # window — planned by the SAME function the corpus build expanded its
        # training rows with, so a served window is a shape a training row had.
        # Always, whatever the length: a threshold computed from prompt length
        # would move between corpus builds and re-shape documents silently.
        extraction, all_spans, latency, pages_used, merge_flags = _extract_policy_windows(
            request, model, route_, lob, latencies=window_latencies, stats=window_stats,
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
        # Printed values only (decision 2026-10-24): a value the answer holds
        # that no page prints is taken out of the JSON and listed for review
        # (serving.printed_values). With no page text (image only) nothing can be
        # checked yet: fields first checks it when MinerU's text arrives. The
        # text is the request's printed_texts when it carries one (the golden
        # eval's noisy mode), else the text the model read.
        from serving.printed_values import hold_to_printed

        checked_texts = _printed_texts(request)
        if common_model and checked_texts:
            # Right value, wrong role: a declarations label printing another value
            # than the one served under it. Before the check, so its flags follow
            # their values (held.joined).
            merge_flags.extend(_role_mismatches(extraction, checked_texts, lob))
        held = hold_to_printed(extraction, all_spans, checked_texts, lob=lob)
        extraction, all_spans, printed_check = held.extraction, held.spans, held.record
        # Flags raised before it follow their values; a value taken out keeps one.
        merge_flags = held.joined(merge_flags)
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
    # One span map for the answer returned, on both paths: one interleaved
    # call, or windows whose spans the merge re-keyed - never one call per
    # page, so the spans describe exactly the document that was returned. The previous per-page
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
            modality_mode=request.modality_mode,
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
    # A value whose tokens all but chose another (serving.value_flags): on the
    # spans as served, after the merge and the printed-values check.
    merge_flags.extend(near_tie_flags(all_spans))
    flags = merge_review_flags(
        completeness, calibrated.review_flags + route_.review_flags + merge_flags
        + ([LOB_FALLBACK_FLAG] if lob_fallback_used else [])
    )

    full_text_pages = _full_text_pages(request)

    # --- the client's canonical envelope -----------------------------------
    # confidence and flagged are filled here, from the calibrated fields, never
    # by the model. A list whose row count disagrees with the document is
    # flagged on every value in it: the rows present may each be confident, and
    # the list as a whole still needs a person.
    if canonical:
        incomplete = tuple(
            flag.split(":", 1)[0] for flag in flags if is_row_completeness_flag(flag)
        )
        # Values a flag sends to review though they are served: printed only off
        # the pages they cite, a near tie, a label printing another value
        # (serving.value_flags).
        routed = routed_paths(flags)
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
        output = with_system_fields(
            envelope(extraction, scores),
            page_count=len(request.image_paths) or len(request.page_texts) or None,
            source_file_name=request.source_file_name,
            lob=lob, modality=(request.ocr_meta or {}).get("modality"),
        )
        if common_model:
            # The full-text tier: every printed word by form and section, built
            # from the page text by Fideon's section builder, never by the model
            # (SPEC_21 §4.6, common.full_text). Before validation, so the client's
            # schema judges it too.
            text_sections = build_text_sections(full_text_pages)
            if text_sections:
                output["text_sections"] = text_sections
            else:
                flags.append(NO_FULL_TEXT_FLAG)
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

    # SPEC_21's every-word rule, measured on what is served.
    coverage = text_coverage(full_text_pages, output) if canonical and common_model else None

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
            "lob_fallback_used": lob_fallback_used,

            "lob_source": resolution.source,
        },
        latency_ms=latency,
        validation_errors=validation_errors,
        reconciliation=reconciliation,
        window_latencies_ms=window_latencies,
        text_coverage=coverage,
        window_stats=window_stats,
        printed_check=printed_check,
        confidence_features=dict(calibrated.features),
        calibrated_with=CalibratedWith(calibrators, thresholds) if calibrators is not None else None,
    )

    log.info(
        "extracted %s as %s (confidence %.3f, %d review flag(s), adapter=%s)",
        request.source_id, result.doc_type, result.overall_confidence,
        len(result.review_flags), route_.adapter or "foundation-only",
    )
    return result
