"""Document-type classification (arch §4a, §4b, §4c).

**Load-bearing, not preprocessing.** A wrong answer loads the wrong adapter *and*
the wrong prompt *and* the wrong schema, and the extraction fails regardless of
how good the model is. A classifier at 92% caps the whole system at 92%, which is
why it carries its own accuracy target and its own gating metric.

Two-level by necessity (arch §4b): type, then ACORD form number. One shared ACORD
adapter covers forms 25/125/140, but each form has a distinct schema, so the form
still has to be identified.

Default implementation is **zero-shot via the model already loaded** (arch §4a
Option C) — no extra model to train, serve, version, or keep in sync. The
interface is swappable for a dedicated vision classifier, which is the right
escalation if classification becomes a *measured* bottleneck, because it works in
both OCR and image-only modes.

On day zero this runs against the **base** model (arch §4c). That path is
explicitly temporary and retires when Foundation v1.0 is promoted.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from common.constants import ACORD_FORMS, ACTIVE_DOC_TYPES
from common.prompts import render_classifier_prompt

log = logging.getLogger(__name__)


class ClassificationError(RuntimeError):
    """Raised when the classifier returns something unusable."""


@dataclass
class Classification:
    """A classification result, with the confidence that drives routing."""

    doc_type: str | None
    acord_form: str | None
    confidence: float
    raw_response: str = ""

    #: ``trained`` under arch v2.1 §4a — classify is a task in the unified
    #: corpus, not a zero-shot prompt against whatever model is loaded. The
    #: zero-shot path survives for day-zero bootstrap (§4c) and retires when the
    #: first fine-tuned release is promoted.
    method: str = "trained"

    #: Ranked alternatives, best first, including ``doc_type`` itself. What makes
    #: the §4a low-confidence path possible: extracting for the top TWO candidate
    #: types is only an option if the classifier says what the second one was.
    candidates: list[tuple[str, float]] = field(default_factory=list)

    #: The L1/L2 hypothesis this was reconciled against, when there was one.
    hypothesis: str | None = None
    hypothesis_agreed: bool | None = None

    #: ``acord_edition`` is recorded in label metadata and used as an evaluation
    #: slice (§4b) — never to select a schema, because editions share one.
    acord_edition: str | None = None

    #: A policy's line of business, when the classifier was asked for one (a
    #: policy that reached L3 with none from L1/L2). One of the registered lines
    #: (a package line such as ``commercial_package`` among them), or None when
    #: no single line describes the policy, for an unknown line, or for a
    #: document that is no policy.
    lob: str | None = None
    #: The probability of the line's own tokens (the product over them), from
    #: the generation's logprobs - never a number the model writes about itself.
    #: None when the backend returned no logprobs.
    lob_confidence: float | None = None
    lob_candidates: list[tuple[str, float]] = field(default_factory=list)
    #: The L1/L2 line hint this was reconciled against, when there was one.
    lob_hypothesis: str | None = None
    lob_hypothesis_agreed: bool | None = None
    #: "base": the base model read the line, among every known line.
    lob_stage: str | None = None

    @property
    def is_usable(self) -> bool:
        return self.doc_type in ACTIVE_DOC_TYPES

    def meets(self, threshold: float) -> bool:
        return self.is_usable and self.confidence >= threshold

    def top_two(self) -> list[str]:
        """The two most likely types, for the §4a low-confidence path.

        Falls back to the single answer when the classifier offered no ranking —
        one candidate is worse than two, and better than refusing to extract.
        """
        ranked = [t for t, _ in sorted(self.candidates, key=lambda kv: -kv[1])
                  if t in ACTIVE_DOC_TYPES]
        if self.doc_type and self.doc_type not in ranked:
            ranked.insert(0, self.doc_type)
        return ranked[:2]


#: How a classifier answer and an L1/L2 hypothesis combine (arch v2.1 §4a).
#: The model wins a confident disagreement: L1 is a carrier registry lookup and
#: L2 is structural inference, and neither has seen the page. The disagreement is
#: logged either way, because a systematic one is evidence about L1/L2.
def combine_with_hypothesis(
    classification: Classification,
    hypothesis: str | None,
    *,
    confidence_threshold: float = 0.70,
) -> Classification:
    """Reconcile the model's answer with the upstream L1/L2 hypothesis."""
    if not hypothesis:
        classification.hypothesis = None
        classification.hypothesis_agreed = None
        return classification

    classification.hypothesis = hypothesis
    agreed = classification.doc_type == hypothesis
    classification.hypothesis_agreed = agreed

    if agreed:
        # Two independent signals agreeing is worth more than either alone, and
        # this is the case that lets a modest model confidence clear the bar.
        classification.confidence = max(classification.confidence, confidence_threshold)
        return classification

    log.warning(
        "L1/L2 proposed %r, the model classified %r at %.2f confidence. %s "
        "A systematic disagreement here is evidence about L1/L2, not noise — the rate is "
        "a reported metric (§15.2).",
        hypothesis, classification.doc_type, classification.confidence,
        "Accepting the model." if classification.confidence >= confidence_threshold
        else "Neither is confident: routing to the low-confidence path.",
    )
    # The hypothesis becomes a candidate rather than an override: it is evidence,
    # and the top-2 path is what evidence this weak supports.
    known = {t for t, _ in classification.candidates}
    if classification.confidence < confidence_threshold and hypothesis not in known:
        classification.candidates.append((hypothesis, classification.confidence))
    return classification


class Classifier(Protocol):
    """The swappable classifier interface."""

    def classify(self, image_paths: list[str], ocr_text: str | None) -> Classification: ...


def _json_object_span(text: str) -> tuple[int, int] | None:
    """Where the answer's JSON object sits in ``text``, as ``(start, end)``.

    Inside the fence when the model fenced it, else from the first brace to the
    last - past a ``<think>`` block or a sentence before or after it. None when
    there is no brace at all. The one place the object is found, so the line's
    confidence is read from the object the answer was parsed from.
    """
    if fenced := re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S):
        return fenced.span(1)
    if braced := re.search(r"\{.*\}", text, re.S):
        return braced.span(0)
    return None


def parse_classification(response: str) -> Classification:
    """Parse the model's JSON answer, tolerating the usual wrapping.

    Models occasionally wrap JSON in fences despite instruction. Recovering from
    that is worth it here: a parse failure routes an otherwise-classifiable
    document to human review for no reason.
    """
    text = response.strip()
    if (where := _json_object_span(text)) is not None:
        text = text[where[0]:where[1]]

    try:
        payload = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ClassificationError(
            f"the classifier response was not JSON: {response[:120]!r}"
        ) from exc

    doc_type = (payload.get("doc_type") or "").strip().lower() or None
    acord_form = payload.get("acord_form")
    acord_form = str(acord_form).strip() if acord_form not in (None, "", "null") else None

    if doc_type and doc_type not in ACTIVE_DOC_TYPES:
        log.warning("classifier returned unknown doc_type %r", doc_type)
        doc_type = None
    if acord_form and acord_form not in ACORD_FORMS:
        log.warning("classifier returned unknown ACORD form %r", acord_form)
        acord_form = None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    # A classification with no form for ACORD cannot select a schema, so it is
    # not usable however confident the model claims to be.
    if doc_type == "acord" and acord_form is None:
        log.warning("ACORD classified without a form number — no schema can be selected")
        confidence = min(confidence, 0.0)

    return Classification(
        doc_type=doc_type,
        acord_form=acord_form,
        confidence=max(0.0, min(1.0, confidence)),
        raw_response=response,
        lob=known_line(payload.get("lob")) if doc_type in (None, "policy") else None,
    )


def known_line(value: Any) -> str | None:
    """``value`` as a registered line of business, or None.

    Read as the rest of the pipeline reads a line (classic auto is personal
    auto; workers' comp is ``wc``), and refused when it names no registered
    line - a guessed schema is worse than the fallback.
    """
    from common.lob import known_lines, merge_line
    from common.schemas import LOB_SCHEMA_ALIASES


    if not isinstance(value, str) or not value.strip():
        return None
    line = str(merge_line(value.strip().lower()))
    line = LOB_SCHEMA_ALIASES.get(line, line)
    if line not in known_lines():
        log.warning("classifier returned unknown line of business %r", value)
        return None
    return line


def line_confidence(text: str, tokens: list[str], logprobs: list[float]) -> float | None:
    """The probability of the ``lob`` value's tokens: exp of their summed logprobs.

    Read from the generation, so a line the model wrote hesitantly scores low
    even when the model claims to be sure. The value is looked for in the JSON
    object :func:`parse_classification` reads - inside a fence, or past a
    ``<think>`` block or a sentence around it - and its characters are mapped to
    the tokens of the raw text. None, never an exception, when it cannot be
    located: the tokens do not rebuild the text, the object has no ``lob``, or
    it does not scan. The line is then one read with no confidence (uncertain,
    flagged); an exception here failed the whole request instead, even one whose
    caller had sent the line.
    """
    import math

    from inference_core.span_map import scan_json_spans, token_char_offsets

    if not tokens or len(tokens) != len(logprobs) or "".join(tokens) != text:
        return None
    where = _json_object_span(text)
    if where is None:
        return None
    start = where[0]
    try:
        # The last `lob` when the object repeats the key, as json.loads reads it.
        span = {s.field_path: s for s in scan_json_spans(text[start:where[1]])}.get("lob")
        if span is None:
            return None
        first, last = start + span.char_start, start + span.char_end
        chosen = [logprob for (t_start, t_end), logprob
                  in zip(token_char_offsets(tokens), logprobs, strict=True)
                  if t_start < last and t_end > first]
        return round(math.exp(sum(chosen)), 4) if chosen else None
    except (ValueError, TypeError, OverflowError):
        # SpanMapError is a ValueError, and so is a malformed \u escape; a
        # logprob that is no number is a TypeError.
        return None


#: How many of a document's pages the classifier reads, from the first: the
#: type and the line show in the opening pages - a form's header, a policy's
#: declarations - and a 50-page policy sent whole to answer a one-word
#: question is waste.
CLASSIFIER_PAGES = 2


def classifier_input(
    pages: Iterable[int],
    image_of: Callable[[int], str],
    text_of: Callable[[int], str | None] | None,
) -> tuple[list[str], str | None]:
    """What the classifier reads of a document: its first two pages by page
    number - their images, and their text joined with a blank line between.

    ``text_of`` is None under ``image_only``, where the prompt says no text
    exists; the text is None then, and when those pages have none. One rule for
    both sides - a served request (``serving.pipeline._classifier_input``) and a
    training row (``build_jsonl.classify_rows``) - so the classifier is trained
    on the input it is served.
    """
    first = sorted(pages)[:CLASSIFIER_PAGES]
    images = [image_of(page) for page in first]
    if text_of is None:
        return images, None
    return images, "\n\n".join(t for page in first if (t := text_of(page))) or None


def classifier_messages(
    image_paths: list[str], ocr_text: str | None, *,
    lines: list[str] | None = None, line_only: bool = False,
) -> list[dict[str, Any]]:
    """The classifier's messages, the same whether a model answers them at
    serving or is trained on them: the first two page images and the text
    :func:`classifier_input` picked for them. Shared so the two cannot drift."""
    content: list[dict[str, Any]] = [{"type": "image", "image": p} for p in image_paths[:CLASSIFIER_PAGES]]
    if ocr_text:
        # Cut at 4,000 characters: the type and the line show in the headers
        # and the layout, not in every word of the opening pages.
        content.append({"type": "text", "text": ocr_text[:4000]})
    return [
        {"role": "system", "content": render_classifier_prompt(lines=lines, line_only=line_only)},
        {"role": "user", "content": content},
    ]


def combine_lob_with_hypothesis(
    classification: Classification, hypothesis: str | None, *, confidence_threshold: float,
) -> Classification:
    """Reconcile the detected line with an L1/L2 line hint, as
    :func:`combine_with_hypothesis` does for the document type.

    Agreement lifts the line's confidence to the threshold: two independent
    signals naming one line. A disagreement keeps a confident detection (L1 is a
    registry lookup, L2 structural inference, and neither read the page) and
    otherwise makes the hint a candidate - evidence too weak to route on alone.
    """
    hint = known_line(hypothesis) if hypothesis else None
    classification.lob_hypothesis = hint
    if hint is None:
        classification.lob_hypothesis_agreed = None
        return classification
    agreed = classification.lob == hint
    classification.lob_hypothesis_agreed = agreed
    if agreed:
        classification.lob_confidence = max(classification.lob_confidence or 0.0, confidence_threshold)
        return classification
    log.warning(
        "L1/L2 proposed line %r, the classifier read %r at %s. %s", hint, classification.lob,
        classification.lob_confidence,
        "Keeping the classifier's line." if (classification.lob_confidence or 0) >= confidence_threshold
        else "Neither is confident.",
    )
    if hint not in {line for line, _ in classification.lob_candidates}:
        classification.lob_candidates.append((hint, classification.lob_confidence or 0.0))
    return classification


class ZeroShotClassifier:
    """Classify by prompting the already-loaded model (arch §4a Option C)."""

    #: It asks a policy for its line, so a line it leaves null was looked for
    #: and not found (serving.pipeline._resolve_lob): no line is a reading.
    reads_lob = True

    def __init__(self, model: Any, generate_fn: Any, *, adapter: str | None = None) -> None:
        self.model = model
        self.generate = generate_fn
        #: The LoRA it reads with; None leaves ``generate_fn`` its own default.
        #: ``model_runner.NO_ADAPTER`` reads with the engine's base weights alone,
        #: as the line must be: it decides which release's adapter reads the
        #: policy, so no adapter can have been chosen yet.
        self.adapter = adapter

    def classify(self, image_paths: list[str], ocr_text: str | None) -> Classification:
        from inference_core.model_runner import NO_ADAPTER

        messages = classifier_messages(image_paths, ocr_text)
        named = {} if self.adapter is None else {"adapter": self.adapter}
        # Logprobs for the line's confidence: the one number here routing acts on.
        result = self.generate(self.model, messages, want_logprobs=True, **named)
        classification = parse_classification(result.text)
        base = self.adapter == NO_ADAPTER or getattr(self.model, "is_base", False)
        classification.method = "zero_shot_base" if base else "zero_shot"
        if classification.lob is not None:
            classification.lob_stage = "base"
            classification.lob_confidence = line_confidence(
                result.text, list(getattr(result, "tokens", None) or []),
                list(getattr(result, "token_logprobs", None) or []))
            classification.lob_candidates = [(classification.lob, classification.lob_confidence or 0.0)]
        return classification


class StaticClassifier:
    """A fixed answer — for tests, and for callers that already know the type."""

    def __init__(self, doc_type: str, acord_form: str | None = None, confidence: float = 1.0) -> None:
        self.result = Classification(doc_type, acord_form, confidence, method="static")

    def classify(self, image_paths: list[str], ocr_text: str | None) -> Classification:
        return self.result
