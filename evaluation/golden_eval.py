"""Evaluate a model on the frozen golden eval set, through the serving pipeline.

The promotion gate's numbers come from here. Every document goes through
:func:`serving.pipeline.extract` — never a bespoke inference path — so the gate
measures what production serves: the same routing, the same policy windows and
merge, the same date post-process, the same calibrated confidence. Validation
scoring reads the corpus rows' own prompts, which is right for checkpoint
selection and wrong for a gate: the serving-only steps would be invisible to it.

Layout of the frozen set in Blob (``paths.golden_eval_set_dir(tenant_id)``), one
set per tenant and one directory per document::

    golden-eval-set/{tenant}/manifest.json                  what was frozen, from which corpus
    golden-eval-set/{tenant}/{source_id}/golden.json        the golden label
    golden-eval-set/{tenant}/{source_id}/metadata.json      doc_type, acord_form, lob, is_scanned
    golden-eval-set/{tenant}/{source_id}/page_N.png         page images, 1-based
    golden-eval-set/{tenant}/{source_id}/page_N.md          OCR text per page (optional)

Per tenant because each tenant's releases are gated on its own documents, and
because source ids are numbered per tenant: one shared set would gate every
tenant on whichever froze first, and read another tenant's ``policy_0001`` as
this one's.

Each document is run in all three input modes. ``noisy_ocr_image`` uses the
corpus build's own seeded corruption, and renders the same prompt serving sends
for clean OCR: the noise lives in the data, never in the instruction. Its
answer is held to the document's clean frozen text (``printed_texts``), not to
the corruption it was sent: a value the page prints is not invented because
the noise garbled it. ``golden_noisy_share`` reads the mode on a deterministic
share of the set (default: all of it).

A reading that fails to extract is tried once more; one that fails again is an
infrastructure failure - not scored, counted, and the gate refuses to decide
until the eval is run again.

Then each document is read once more in the repeat mode (``run_to_run_mode`` in
the adapter's training config: image_only, the mode production reads), its
windows sent in reverse order, so the engine batches them differently: how
often the two answers agree is ``run_to_run_agreement`` (decisions 2026-10-24,
2026-10-26); ``run_to_run_share`` repeats a deterministic share of the set
(default: all of it). And each reading carries serving's own printed-values record,
which ``hallucination_rate`` reads. An image_only reading is sent no text, so
serving records none: its answer is checked here on the document's frozen page
text - the text its ocr_plus_image reading is sent - as production checks the
fields it read from the images on MinerU's text (:func:`borrowed_printed_check`).
A document with no page text to borrow is unscored, and counted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: The modes every golden document is evaluated in.
GOLDEN_MODES: tuple[str, ...] = ("ocr_plus_image", "noisy_ocr_image", "image_only")

#: The mode read with the document's clean page text: the one an image_only
#: reading borrows its text from to be checked for invention.
MAIN_MODE = "ocr_plus_image"

#: The mode read twice for run_to_run_agreement when the config names none:
#: image_only, the one production reads (fields first). ``run_to_run_mode`` under
#: ``evaluation`` in the adapter's training config (:func:`run_to_run_mode`).
RUN_TO_RUN_MODE = "image_only"

_PAGE = re.compile(r"page_(\d+)\.(png|md)$")


class GoldenEvalError(RuntimeError):
    """Raised when the frozen eval set cannot be read or evaluated."""


@dataclass
class GoldenDocument:
    """One frozen eval document: its label, its pages, and what selects its schema."""

    source_id: str
    doc_type: str
    golden: dict[str, Any]
    image_keys: list[str]
    page_texts: dict[int, str] = field(default_factory=dict)
    acord_form: str | None = None
    lob: str | list[str] | None = None
    is_scanned: bool = False
    synthetic: bool = False
    #: The document's carrier was held out of train and val (Fideon SPEC_09
    #: amendment item 4); evaluation reports these documents apart.
    held_out_carrier: bool = False
    #: The carrier as the freeze recorded it (``freeze_eval_set``: the corpus
    #: build's key), for the report's per-carrier error table; None for a set
    #: frozen before it was recorded.
    carrier: str | None = None
    #: Whether the carrier's documents were in the training split, where the
    #: frozen metadata says (reported apart: seen and unseen carriers).
    carrier_seen_in_train: bool | None = None


def load_golden_set(
    client: BlobClient, doc_types: Sequence[str] | None = None, *, tenant_id: str | None = None
) -> list[GoldenDocument]:
    """Every document in this tenant's frozen set, optionally narrowed to ``doc_types``.

    Refused, never narrowed, while any of them has no review record in its
    frozen metadata (``export_golden_labels.review_problem``, decision
    2026-10-22): a gold nobody reviewed does not score, and skipping it would
    quietly score a smaller yardstick than every earlier version was scored on.
    """
    from data_pipeline.labeling.export_golden_labels import review_problem
    from evaluation.run_eval import eval_set_keys

    prefix = paths.golden_eval_set_dir(tenant_id)
    by_source: dict[str, list[str]] = {}
    for key in eval_set_keys(client, tenant_id):
        rest = key[len(prefix):].strip("/")
        if "/" in rest:
            by_source.setdefault(rest.split("/", 1)[0], []).append(key)

    documents = []
    unreviewed: list[tuple[str, str]] = []
    for source_id, keys in sorted(by_source.items()):
        base = f"{prefix}/{source_id}"
        if f"{base}/golden.json" not in keys:
            continue
        metadata = (
            client.read_json(f"{base}/metadata.json") if f"{base}/metadata.json" in keys else {}
        )
        doc_type = metadata.get("doc_type")
        if not doc_type:
            raise GoldenEvalError(
                f"{source_id} has no doc_type in metadata.json, so no schema can be selected"
            )
        if doc_types and doc_type not in doc_types:
            continue
        if problem := review_problem(None, metadata):
            unreviewed.append((source_id, problem))
            continue
        images: dict[int, str] = {}
        texts: dict[int, str] = {}
        for key in keys:
            match = _PAGE.search(key)
            if not match:
                continue
            page = int(match.group(1))
            if match.group(2) == "png":
                images[page] = key
            else:
                texts[page] = client.read_text(key)
        if not images:
            raise GoldenEvalError(f"{source_id} has no page images")
        if sorted(images) != list(range(1, len(images) + 1)):
            raise GoldenEvalError(f"{source_id}: page images are not numbered 1..n")
        documents.append(GoldenDocument(
            source_id=source_id,
            doc_type=doc_type,
            # A rule-added value the page does not print is not expected of
            # the model either (label_verification), as in its training targets.
            golden=verified_label(
                client.read_json(f"{base}/golden.json"), [texts.get(p, "") for p in sorted(images)],
                doc_type=doc_type, lob=metadata.get("lob"),
            ),
            image_keys=[images[p] for p in sorted(images)],
            page_texts={p: texts.get(p, "") for p in sorted(images)},
            acord_form=metadata.get("acord_form"),
            lob=merge_line(metadata.get("lob")),
            is_scanned=bool(metadata.get("is_scanned", False)),
            synthetic=bool(metadata.get("synthetic", False)),
            held_out_carrier=bool(metadata.get("held_out_carrier", False)),
            carrier=str(metadata["carrier"]) if metadata.get("carrier") else None,
            carrier_seen_in_train=(metadata["carrier_seen_in_train"]
                                   if isinstance(metadata.get("carrier_seen_in_train"), bool) else None),
        ))
    if unreviewed:
        raise GoldenEvalError(
            f"{len(unreviewed)} frozen document(s) have no review record, so they cannot score "
            f"(decision 2026-10-22): {unreviewed[0][0]}: {unreviewed[0][1]}. Others: "
            f"{[s for s, _ in unreviewed[1:6]]}. Record each one's review in its label metadata and "
            "copy it into the frozen set (evaluation.freeze_eval_set.mark_reviewed), or delete the "
            "set and freeze again from a corpus of reviewed golds."
        )
    return documents


def verified_label(
    label: dict[str, Any], page_texts: list[str], *, doc_type: str = "policy", lob: Any = None,
) -> dict[str, Any]:
    """The frozen gold as training would teach it, through the function the
    corpus build reads its golds with (label_verification): a form's edition
    moved out of its number where the label rules hold for ``doc_type`` and
    ``lob`` (label_rules.label_rules_apply), then each rule-added value checked
    as the corpus build checks it (``listed=True``: serving's figure-aware,
    OCR-tolerant matcher, so "$250,000.00" is printed on a page printing
    "$250,000") and re-paged where it is printed elsewhere. A value no page
    prints (``report.unprinted``) is not expected of the model - serving takes
    such a value out - so it is left out of the gold scored here; the build
    keeps it in the label and leaves out the windows holding it. A set frozen
    before the edition rule is not rewritten in Blob; it is read under it."""
    from data_pipeline.dataset_builder.label_verification import VerificationReport, without_paths
    from data_pipeline.dataset_builder.label_verification import verified_label as verify

    report = VerificationReport()
    verified = verify(label, page_texts, report, doc_type=doc_type, lob=lob, listed=True)
    if report.unprinted:
        return without_paths(verified, report.unprinted)
    return copy.deepcopy(verified)


def merge_line(lob: Any) -> Any:
    """Classic auto is read as personal auto (common.lob.MERGED_LINES)."""
    from common.lob import merge_line as merged

    return merged(lob)


class GoldenTriples(list):
    """The ``(expected, got, metadata)`` triples :func:`evaluate` returns, with
    what could not be scored at all: ``infrastructure_failures`` - each reading
    that failed to extract twice, ``{"source_id", "mode", "pass", "error"}`` -
    and ``retried``, the readings that failed once and were read on the retry."""

    def __init__(self, triples: Sequence[Any] = ()) -> None:
        super().__init__(triples)
        self.infrastructure_failures: list[dict[str, Any]] = []
        self.retried: list[dict[str, Any]] = []


@dataclass
class _Reading:
    """One document read in one mode, as :func:`evaluate` leaves it."""

    triple: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    failure: dict[str, Any] | None = None
    retried: dict[str, Any] | None = None
    again: tuple[Any, Any, Any, dict[str, Any]] | None = None
    unchecked: str | None = None
    unscored: str | None = None


#: How many times a reading is tried before it is an infrastructure failure.
READING_ATTEMPTS = 2


def evaluate(
    documents: Sequence[GoldenDocument],
    model: Any,
    client: BlobClient,
    images_root: str | Path,
    *,
    calibrators: Any = None,
    thresholds: Any = None,
    modes: Sequence[str] = GOLDEN_MODES,
    seed: int = 42,
    run_to_run: bool | None = None,
    repeat_mode: str | None = None,
    noisy_share: float | None = None,
    repeat_share: float | None = None,
    in_flight: int | None = None,
) -> GoldenTriples:
    """``(expected, got, metadata)`` triples for :func:`run_eval.build_report`,
    in document order, then mode order - however many documents were in flight.

    A reading that fails to extract is tried once more. One that fails again is
    an **infrastructure failure**: not scored - an engine that fell over says
    nothing about the model - and recorded on the result
    (:class:`GoldenTriples`), so the report counts it and the gate refuses to
    decide on a partial measurement (``gating.INFRASTRUCTURE_FAILURES``).

    ``noisy_share`` (default :func:`golden_noisy_share`, 1.0: every document)
    reads noisy_ocr_image on that share of the documents alone, a deterministic
    subset (:func:`in_share`). A noisy reading is SENT its corrupted text, and
    held to the document's clean frozen text (``ExtractionRequest.printed_texts``):
    the corruption is noise in what the model reads, never in what the page
    prints. Until serving takes ``printed_texts`` the reading is held to the
    corrupted text, as before, and the metadata says so.

    ``run_to_run`` (default: :func:`run_to_run_check`): after the normal
    passes, the ``repeat_mode`` reading (default: :func:`run_to_run_mode`,
    image_only) of the documents in ``repeat_share`` (default
    :func:`run_to_run_share`, 1.0: every one) is extracted again with its
    windows sent in reverse order (``send_order="reversed"``), and the second
    answer rides on that reading's metadata (``run_eval.REPEAT_ANSWER``) for
    ``run_to_run_agreement``. None when ``modes`` leaves the mode out. A repeat
    that fails twice is an infrastructure failure too, and its reading carries
    no second answer.

    An image_only policy reading comes back from serving with no printed-values
    record (no text was sent): it is checked on the document's own page text
    (:func:`borrowed_printed_check`), marked ``run_eval.BORROWED_TEXT``, and its
    confidence computed again on that text (``calibration.post_text``), as
    production does once MinerU's text arrives. With none to borrow its record
    stays None, and the report counts it unscored.

    ``in_flight`` (default :func:`documents_in_flight`): how many readings go to
    the model at once - more than one only on a backend that answers requests
    as they finish (a served engine); results are put back in order, so the
    triples are the sequential run's.
    """
    from evaluation.run_eval import REPEAT_ANSWER
    from serving.pipeline import ExtractionRequest

    repeat = run_to_run_check() if run_to_run is None else run_to_run
    repeat_mode = _known_mode(run_to_run_mode() if repeat_mode is None else repeat_mode)
    noisy_share = golden_noisy_share() if noisy_share is None else float(noisy_share)
    repeat_share = run_to_run_share() if repeat_share is None else float(repeat_share)
    if repeat and repeat_mode not in modes:
        log.warning("golden eval: the run-to-run repeat reads %s, which this pass does not read (%s): "
                    "nothing is read twice", repeat_mode, list(modes))
    from training.stage_data import localize_keys

    local = localize_keys(
        client, sorted({key for doc in documents for key in doc.image_keys}), images_root
    )
    clean_text = _takes_printed_texts(ExtractionRequest)
    if not clean_text and "noisy_ocr_image" in modes:
        log.warning("golden eval: serving takes no printed_texts yet, so noisy_ocr_image readings are held "
                    "to the corrupted text they are sent (not the clean frozen text)")
    workers = documents_in_flight(model) if in_flight is None else max(1, int(in_flight))
    context = _ReadContext(model=model, local=local, calibrators=calibrators, thresholds=thresholds,
                           seed=seed, clean_text=clean_text)

    plan = [(doc, mode) for doc in documents for mode in modes
            if mode != "noisy_ocr_image" or in_share(doc.source_id, noisy_share, "noisy")]
    readings = _in_order([
        (lambda doc=doc, mode=mode: _read(context, doc, mode,
                                          repeat=repeat and mode == repeat_mode
                                          and in_share(doc.source_id, repeat_share, "repeat")))
        for doc, mode in plan
    ], workers)

    triples = GoldenTriples()
    again: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    unchecked = [r.unchecked for r in readings if r.unchecked]
    unscored = [r.unscored for r in readings if r.unscored]
    for reading in readings:
        if reading.retried:
            triples.retried.append(reading.retried)
        if reading.failure:
            triples.infrastructure_failures.append(reading.failure)
            continue
        triples.append(reading.triple)
        if reading.again:
            again.append(reading.again)
    if unchecked:
        log.warning(
            "golden eval: %d policy reading(s) were sent page text and came back with no "
            "printed-values record, so hallucination_rate does not count their values: %s",
            len(unchecked), unchecked[:5],
        )
    if unscored:
        log.warning(
            "golden eval: %d image_only policy reading(s) have no page text to borrow, so their values "
            "are not checked for invention (unscored, never clean): %s", len(unscored), unscored[:5],
        )

    # The repeat pass, after every document's own: the same request, its
    # windows sent in reverse order (the merge keeps planned order), so the
    # engine batches them differently.
    repeats = _in_order([(lambda entry=entry: _read_again(context, *entry)) for entry in again], workers)
    for (_doc, _request, _calibration, metadata), (answer, failure, retried) in zip(again, repeats, strict=True):
        if retried:
            triples.retried.append(retried)
        if failure:
            metadata["repeat_error"] = failure["error"]
            triples.infrastructure_failures.append(failure)
            continue
        metadata[REPEAT_ANSWER] = answer
    if triples.infrastructure_failures:
        log.warning("golden eval: %d reading(s) failed to extract twice and are not scored (%s): the gate "
                    "will not decide on this measurement", len(triples.infrastructure_failures),
                    [f"{f['source_id']} ({f['mode']}, {f['pass']})" for f in triples.infrastructure_failures[:5]])
    return triples


@dataclass
class _ReadContext:
    model: Any
    local: dict[str, str]
    calibrators: Any
    thresholds: Any
    seed: int
    clean_text: bool


def _attempts(call: Any, what: str) -> tuple[Any, list[str]]:
    """``call()``'s result and the errors of the attempts that failed before it;
    ``(None, errors)`` when every one of :data:`READING_ATTEMPTS` failed."""
    errors: list[str] = []
    for attempt in range(1, READING_ATTEMPTS + 1):
        try:
            return call(), errors
        except Exception as exc:  # noqa: BLE001 - retried, then recorded, never scored as an answer
            errors.append(f"{type(exc).__name__}: {exc}")
            log.warning("golden eval: %s failed (attempt %d of %d): %s", what, attempt, READING_ATTEMPTS, exc)
    return None, errors


def _read(context: _ReadContext, doc: GoldenDocument, mode: str, *, repeat: bool) -> _Reading:
    """One document read in one mode through serving, as :func:`evaluate` scores it."""
    from calibration.fit_calibration import CalibrationParams
    from data_pipeline.dataset_builder.noisy_ocr_augment import corrupt_ocr_pages
    from evaluation.run_eval import BORROWED_TEXT, IMAGE_ONLY_MODE, PRINTED_CHECK, PRINTED_TEXT, UNREAD_PAGES
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import ExtractionRequest, extract

    calibration = CalibrationParams(
        method="temperature", doc_type=doc.doc_type,
        model_version=getattr(context.model, "tag", "eval"), temperature=1.0,
    )
    texts = dict(doc.page_texts)
    held_to: dict[str, Any] = {}
    if mode == "noisy_ocr_image":
        corrupted, _details = corrupt_ocr_pages(
            [texts[p] for p in sorted(texts)], doc.source_id, seed=context.seed
        )
        texts = dict(zip(sorted(texts), corrupted, strict=True))
        if context.clean_text:
            # Sent the corrupted text to read; held to what the page prints.
            held_to = {"printed_texts": dict(doc.page_texts)}
    request = ExtractionRequest(
        source_id=doc.source_id,
        image_paths=[context.local[key] for key in doc.image_keys],
        page_texts={} if mode == "image_only" else texts,
        modality_mode=mode,
        known_doc_type=doc.doc_type,
        known_acord_form=doc.acord_form,
        known_lob=doc.lob,
        **held_to,
    )
    metadata: dict[str, Any] = {
        "source_id": doc.source_id,
        "doc_type": doc.doc_type,
        "acord_form": doc.acord_form,
        "lob": doc.lob,
        "modality_mode": mode,
        "is_scanned": doc.is_scanned,
        "synthetic": doc.synthetic,
        "held_out_carrier": doc.held_out_carrier,
        "page_count": len(doc.image_keys),
    }
    if doc.carrier:
        metadata["carrier"] = doc.carrier
    if doc.carrier_seen_in_train is not None:
        metadata["carrier_seen_in_train"] = doc.carrier_seen_in_train
    if mode == "noisy_ocr_image":
        # What the noisy reading's values were held to (decision: the page's
        # clean text, never the corruption it was sent).
        metadata[PRINTED_TEXT] = "clean" if held_to else "corrupted"
    reading = _Reading()
    result, errors = _attempts(
        lambda: extract(request, context.model, StaticClassifier(doc.doc_type, doc.acord_form), calibration,
                        calibrators=context.calibrators, thresholds=context.thresholds, strict_schema=False),
        f"{doc.source_id} ({mode})",
    )
    if errors and result is not None:
        reading.retried = {"source_id": doc.source_id, "mode": mode, "pass": "first", "error": errors[0]}
    if result is None:
        reading.failure = {"source_id": doc.source_id, "mode": mode, "pass": "first", "error": errors[-1]}
        return reading
    if repeat:
        reading.again = (doc, request, calibration, metadata)
    # Serving's own printed-values check (serving.printed_values): the
    # values it could check and those no page prints, recorded before
    # strict mode removed any. None with no page text: nothing checked.
    metadata[PRINTED_CHECK] = getattr(result, "printed_check", None)
    if metadata[PRINTED_CHECK] is None and doc.doc_type == "policy" and mode == IMAGE_ONLY_MODE:
        # Read from the images alone, as production reads: checked on
        # the document's page text, as production checks it on
        # MinerU's (decision 2026-10-26).
        held, unread = borrowed_printed_check(result, doc.page_texts)
        if held is None:
            reading.unscored = doc.source_id
        else:
            # Scored as served: the values no page prints taken out, and the
            # confidence computed again on the text, as production does once
            # the text arrives (calibration.post_text).
            _recalibrate(held, doc.page_texts, context)
            result = held
            metadata.update({PRINTED_CHECK: held.printed_check, PRINTED_TEXT: BORROWED_TEXT,
                             UNREAD_PAGES: unread})
    elif (metadata[PRINTED_CHECK] is None and doc.doc_type == "policy" and mode != IMAGE_ONLY_MODE
            and any(str(text).strip() for text in texts.values())):
        reading.unchecked = f"{doc.source_id} ({mode})"
    metadata["field_confidence"] = {
        path: body["confidence"] for path, body in result.fields.items()
    }
    if result.reconciliation is not None:
        # The serving path's own report: the gate reads what production did.
        metadata["reconciliation"] = result.reconciliation
    # For the release measurements: which adapter served, and each call's time.
    metadata["adapter"] = result.route_info.get("adapter")
    metadata["window_latencies_ms"] = list(result.window_latencies_ms)
    # Serving's own verdict, the audit gate's: a common-model policy is
    # served with every key filled, so a section no window wrote is there
    # as nulls and only serving knows it was never written.
    # schema_validity_rate reads this verdict (run_eval.score_subset).
    metadata["schema_valid"] = result.schema_valid
    metadata["validation_errors"] = list(result.validation_errors[:4])
    reading.triple = (doc.golden, result.extraction, metadata)
    return reading


def _read_again(context: _ReadContext, doc: GoldenDocument, request: Any, calibration: Any,
                metadata: dict[str, Any]) -> tuple[Any, dict[str, Any] | None, dict[str, Any] | None]:
    """The repeat reading: ``(second answer, failure, retried)``."""
    from dataclasses import replace

    from evaluation.run_eval import IMAGE_ONLY_MODE
    from serving.doc_type_classifier import StaticClassifier
    from serving.pipeline import extract

    mode = metadata["modality_mode"]
    result, errors = _attempts(
        lambda: extract(replace(request, send_order="reversed"), context.model,
                        StaticClassifier(doc.doc_type, doc.acord_form), calibration,
                        calibrators=context.calibrators, thresholds=context.thresholds, strict_schema=False),
        f"{doc.source_id} ({mode}, repeat)",
    )
    retried = ({"source_id": doc.source_id, "mode": mode, "pass": "repeat", "error": errors[0]}
               if errors and result is not None else None)
    if result is None:
        return None, {"source_id": doc.source_id, "mode": mode, "pass": "repeat", "error": errors[-1]}, None
    if (getattr(result, "printed_check", None) is None and doc.doc_type == "policy"
            and mode == IMAGE_ONLY_MODE):
        # Held as the first reading was, so the two compare as served.
        held, _unread = borrowed_printed_check(result, doc.page_texts)
        result = held if held is not None else result
    return result.extraction, None, retried


def _recalibrate(result: Any, page_texts: Any, context: _ReadContext) -> None:
    if context.calibrators is None:
        return
    from calibration.post_text import recalibrate_after_text

    try:
        recalibrate_after_text(result, page_texts, calibrators=context.calibrators,
                               thresholds=context.thresholds)
    except Exception as exc:  # noqa: BLE001 - scored with the confidence it was served with
        log.warning("golden eval: %s: confidence could not be computed again on the page text: %s",
                    getattr(result, "source_id", "?"), exc)


def _in_order(tasks: Sequence[Any], workers: int) -> list[Any]:
    """Each task's result, in the tasks' order; ``workers`` at a time."""
    if workers <= 1 or len(tasks) <= 1:
        return [task() for task in tasks]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="golden-eval") as pool:
        return list(pool.map(lambda task: task(), tasks))


def _takes_printed_texts(request_type: Any) -> bool:
    """Whether serving's request can be given the text values are held to
    apart from the text the model is sent (``printed_texts``)."""
    return "printed_texts" in (getattr(request_type, "__dataclass_fields__", None) or {})


def in_share(source_id: str, share: float, salt: str) -> bool:
    """Whether ``source_id`` is in the deterministic ``share`` of the frozen set
    picked for ``salt`` (``noisy``, ``repeat``): a hash of the two, so the same
    documents are picked on every run and machine, and a larger share holds
    every document of a smaller one. 1.0 picks every document, 0 none."""
    if share >= 1.0:
        return True
    if share <= 0.0:
        return False
    digest = hashlib.sha256(f"{salt}:{source_id}".encode()).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12) < share


def _share(key: str) -> float:
    from common.adapter import ADAPTER
    from common.config import training_config

    evaluation = training_config(ADAPTER.training_config).get("evaluation") or {}
    value = evaluation.get(key, 1.0)
    try:
        share = float(1.0 if value is None else value)
    except (TypeError, ValueError) as exc:
        raise GoldenEvalError(f"{key} is {value!r}; expected a share between 0 and 1") from exc
    if not 0.0 < share <= 1.0:
        raise GoldenEvalError(f"{key} is {share}; expected a share above 0 and at most 1 (1.0: every document)")
    return share


def golden_noisy_share() -> float:
    """The share of the frozen set the golden eval reads in noisy_ocr_image:
    ``golden_noisy_share`` under ``evaluation`` in the adapter's training
    config, 1.0 (every document) when unset. A deterministic subset
    (:func:`in_share`); part of the yardstick (``run_eval.yardstick_settings``)."""
    return _share("golden_noisy_share")


def run_to_run_share() -> float:
    """The share of the frozen set read a second time for run_to_run_agreement:
    ``run_to_run_share`` under ``evaluation``, 1.0 (every document) when unset.
    A deterministic subset (:func:`in_share`); part of the yardstick."""
    return _share("run_to_run_share")


#: Readings in flight at once on a backend that can take them
#: (``golden_documents_in_flight`` under ``evaluation``).
DOCUMENTS_IN_FLIGHT = 4


def documents_in_flight(model: Any) -> int:
    """How many readings the golden eval sends at once: ``golden_documents_in_flight``
    under ``evaluation`` (default :data:`DOCUMENTS_IN_FLIGHT`) on a backend that
    answers requests as they finish (``model_runner.can_submit``: a served
    engine batches them), one on the in-process engine, which runs one batch
    to its end - a policy's windows already go to it together."""
    from common.adapter import ADAPTER
    from common.config import training_config

    backend = getattr(model, "backend", None)
    supports = getattr(backend, "supports_submit", None)
    try:
        concurrent = bool(supports()) if callable(supports) else False
    except Exception:  # noqa: BLE001 - a backend that cannot say runs one at a time
        concurrent = False
    if not concurrent:
        return 1
    evaluation = training_config(ADAPTER.training_config).get("evaluation") or {}
    return max(1, int(evaluation.get("golden_documents_in_flight") or DOCUMENTS_IN_FLIGHT))


def run_to_run_check() -> bool:
    """Whether the golden eval reads each document twice in :func:`run_to_run_mode`,
    for ``run_to_run_agreement``: ``run_to_run_check`` under ``evaluation`` in the
    adapter's training config (configs/training/unified.yaml), on unless it says
    false. On, it costs one more pass in that mode over the frozen set."""
    from common.adapter import ADAPTER
    from common.config import training_config

    evaluation = training_config(ADAPTER.training_config).get("evaluation") or {}
    return bool(evaluation.get("run_to_run_check", True))


def run_to_run_mode() -> str:
    """The mode the golden eval reads a second time for ``run_to_run_agreement``:
    ``run_to_run_mode`` under ``evaluation`` in the adapter's training config,
    :data:`RUN_TO_RUN_MODE` (image_only, the mode production reads) when it
    names none (decision 2026-10-26). Part of the gate's yardstick
    (``run_eval.yardstick_settings``): agreement over one mode's readings is
    another number than over another's. A mode the gate does not read is
    refused, never read as either."""
    from common.adapter import ADAPTER
    from common.config import training_config

    evaluation = training_config(ADAPTER.training_config).get("evaluation") or {}
    return _known_mode(str(evaluation.get("run_to_run_mode") or RUN_TO_RUN_MODE))


def _known_mode(mode: str) -> str:
    if mode not in GOLDEN_MODES:
        raise GoldenEvalError(f"run_to_run_mode is {mode!r}; the golden eval reads {list(GOLDEN_MODES)}")
    return mode


def borrowed_printed_check(result: Any, page_texts: Any) -> tuple[Any | None, list[int]]:
    """A result read from the page images alone, as production serves it once
    the document's text is read: held to the document's page text (page number
    -> text) by production's own fields-first check
    (``serving.printed_values.hold_result_to_printed``) - a value no page prints
    taken out under strict mode, its confidence, rows and flags with it - and
    its ``printed_check`` the record of what was checked, made before anything
    was taken out. Returns ``(that result, pages it cites that have no text)``;
    the result given is left as it was.

    The gate scores this result, not the model's raw answer (decision
    2026-10-26): what a user receives. Invention is counted from the record.

    A page it cites with no text is checked as blank: a value found on no page
    with text while citing one is unchecked (on an unread page), kept, never
    invented. ``(None, [])`` when no page has text, or the check cannot be made:
    unscored, never counted as nothing invented."""
    from evaluation.run_eval import cited_pages, page_has_text
    from serving.printed_values import hold_result_to_printed

    texts = {int(page): str(text or "") for page, text in (page_texts or {}).items()}
    if not any(page_has_text(text) for text in texts.values()):
        return None, []
    unread = sorted(page for page in cited_pages(result.extraction) if not page_has_text(texts.get(page)))
    texts.update({page: "" for page in unread if page not in texts})
    held = copy.deepcopy(result)
    try:
        hold_result_to_printed(held, texts)
    except Exception as exc:  # noqa: BLE001 - an answer the check cannot read is unscored, not clean
        log.warning("golden eval: the printed-values check on borrowed text failed: %s", exc)
        return None, []
    if held.printed_check is None:
        return None, []
    return held, unread


def assert_scoreable(
    client: BlobClient, doc_types: Sequence[str] | None = None, *, tenant_id: str | None = None,
) -> None:
    """Refuse, before any model is loaded, a frozen set the gate cannot score: a
    freeze that was interrupted, or a frozen gold of ``doc_types`` with no review
    record (decision 2026-10-22, checked as :func:`load_golden_set` checks it).

    Reads each frozen document's metadata and nothing else. Without it the gate
    loaded the merged bf16 model, then learned from load_golden_set that a gold
    had no record - and, retried, loaded the model again to learn it twice.
    """
    from data_pipeline.labeling.export_golden_labels import review_problem
    from evaluation.run_eval import eval_set_keys

    _refuse_interrupted_freeze(client, tenant_id)
    prefix = paths.golden_eval_set_dir(tenant_id)
    keys = set(eval_set_keys(client, tenant_id))
    unreviewed: list[tuple[str, str]] = []
    for key in sorted(keys):
        rest = key[len(prefix):].strip("/")
        source_id, _sep, name = rest.partition("/")
        if name != "golden.json":
            continue
        base = f"{prefix}/{source_id}"
        metadata = (
            client.read_json(f"{base}/metadata.json") if f"{base}/metadata.json" in keys else {}
        )
        if doc_types and metadata.get("doc_type") not in doc_types:
            continue
        if problem := review_problem(None, metadata):
            unreviewed.append((source_id, problem))
    if unreviewed:
        raise GoldenEvalError(
            f"{len(unreviewed)} frozen document(s) have no review record, so they cannot score "
            f"(decision 2026-10-22): {unreviewed[0][0]}: {unreviewed[0][1]}. Others: "
            f"{[s for s, _ in unreviewed[1:6]]}. Record each one's review in its label metadata and "
            "copy it into the frozen set (evaluation.freeze_eval_set.mark_reviewed), or delete the "
            "set and freeze again from a corpus of reviewed golds. Refused before any model was loaded."
        )


def _refuse_interrupted_freeze(client: BlobClient, tenant_id: str | None) -> None:
    from evaluation.freeze_eval_set import is_frozen
    from evaluation.run_eval import eval_set_keys

    # is_frozen first, always: it is also what refuses a set still frozen at the
    # root, which this tenant's empty prefix would otherwise let pass unnoticed.
    if not is_frozen(client, tenant_id=tenant_id) and eval_set_keys(client, tenant_id):
        raise GoldenEvalError(
            "the golden eval set has documents but no manifest: a freeze was interrupted. Re-run "
            "freeze-eval-set from the same corpus to finish it before gating on it."
        )


def evaluate_version(
    client: BlobClient,
    model: Any,
    *,
    version: str,
    corpus_version: str,
    tenant_id: str | None = None,
    calibrators: Any = None,
    thresholds: Any = None,
    lines: Collection[str] | None = None,
    adapter_checkpoint: str | None = None,
    promoted_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate on this tenant's frozen set, write the eval report where the gate
    reads it, return its dict.

    ``promoted_report``: the promoted release's eval report, for the per-field
    delta (``field_delta_vs_promoted``); default: found from the release index.

    A reading that failed to extract twice is not scored: the report counts it
    (``infrastructure_failures``) and the gate metrics returned carry the count,
    so the gate blocks and asks for a re-run instead of scoring the failure as
    wrong answers. The report written to Blob then holds its metrics apart
    (``gate_metrics_partial``), so that re-run scores again.

    ``adapter_checkpoint`` names the checkpoint ``model`` applies (the pipeline
    passes it), recorded so the report is never reused for another one.

    Each document is read twice in :func:`run_to_run_mode` (image_only) when
    :func:`run_to_run_check` is on (the default): one more pass in that mode,
    for ``run_to_run_agreement``. The mode is recorded (``run_to_run_mode``,
    and in ``eval_set``, whose fingerprint it is part of), so a report made on
    another repeat mode is not reused or compared as if it were this one.

    ``lines`` are the lines of business the release serves: only frozen policies
    of one of them are scored, because serving sends every other policy (another
    line, a package, no line) to the base model, not to this release. ``None``
    scores every frozen document of the adapter's types (``run_eval``'s command
    line). The report records them (``lines_scored``), so the gate reuses it, or
    compares a baseline with it, only for the same lines.

    Every reading's served answer is stored as scored
    (``evaluation.rescore.store_answers``, ``paths.eval_answers_key``) and the
    report records where (``answers_stored``): a later gate whose scoring code
    or settings moved re-scores them (``rescore.rescore_report``) instead of
    judging its candidate on floors alone. The model is never run again.
    """
    from common.adapter import ADAPTER
    from common.lob import single_line
    from evaluation.run_eval import assert_eval_set_disjoint, build_report

    _refuse_interrupted_freeze(client, tenant_id)
    # Before anything is evaluated: a leaked eval set makes every number meaningless.
    assert_eval_set_disjoint(client, corpus_version, tenant_id)
    documents = load_golden_set(client, ADAPTER.doc_types, tenant_id=tenant_id)
    if lines is not None:
        served = set(lines)
        documents = [d for d in documents if single_line(d.lob) in served]
    if not documents:
        raise GoldenEvalError(
            f"the frozen eval set at {paths.golden_eval_set_dir(tenant_id)}/ holds no "
            f"{list(ADAPTER.doc_types)} documents"
            + (f" of the lines this release serves ({sorted(lines)})" if lines is not None else "")
            + ", so there is nothing to gate on"
        )
    from evaluation.release_measurements import GpuMemorySampler

    # The repeat pass for run_to_run_agreement: one more pass in the repeat
    # mode - image_only, the mode production reads.
    repeat, repeat_mode = run_to_run_check(), run_to_run_mode()
    noisy_share, repeat_share = golden_noisy_share(), run_to_run_share()
    with GpuMemorySampler() as memory:
        triples = evaluate(
            documents, model, client,
            paths.staging_train_images_dir(f"golden-{corpus_version}", tenant_id),
            calibrators=calibrators, thresholds=thresholds, run_to_run=repeat, repeat_mode=repeat_mode,
            noisy_share=noisy_share, repeat_share=repeat_share,
        )
    failures = list(getattr(triples, "infrastructure_failures", []))
    first_failed = [f for f in failures if f.get("pass") == "first"]
    report = build_report(version, triples, corpus_version=corpus_version)
    body = report.as_dict()
    # The readings that failed to extract twice: not scored (an engine that fell
    # over says nothing about the model). Counted here; the gate refuses to
    # decide on a measurement missing them (gating.INFRASTRUCTURE_FAILURES).
    body["documents_failed"] = len(first_failed)
    body["infrastructure_failures"] = {
        "count": len(failures),
        "documents": sorted({str(f["source_id"]) for f in failures}),
        "readings": failures[:50],
        # Failed once, read on the retry: scored, and shown.
        "retried_ok": len(getattr(triples, "retried", [])),
    }
    # Whether each document was read twice (run_to_run_agreement), in which
    # mode, over which share of the set, and how many second readings failed
    # twice (not compared). The mode and the shares are part of the yardstick
    # too (eval_set, run_eval.yardstick_settings).
    body["run_to_run_check"] = repeat
    body["run_to_run_mode"] = repeat_mode
    body["run_to_run_share"] = repeat_share
    body["repeats_failed"] = sum(1 for f in failures if f.get("pass") == "repeat")
    # The share of the set read in noisy_ocr_image, and the text those readings
    # were held to: the clean frozen text when serving takes it.
    from serving.pipeline import ExtractionRequest

    body["golden_noisy_share"] = noisy_share
    body["noisy_held_to"] = "clean" if _takes_printed_texts(ExtractionRequest) else "corrupted"
    # What hallucination_rate could not check directly: image_only readings
    # checked on the document's page text (borrowed: the text the
    # ocr_plus_image reading is sent), the pages they cite with none, and the
    # readings with no text at all - unscored, never counted clean.
    from evaluation.run_eval import TEXT_DONOR_MODE, field_deltas

    body["invention_coverage"] = {"borrowed_from": TEXT_DONOR_MODE, **report.invention_coverage()}
    # Which lines were scored (None: every frozen document), so the gate reuses
    # this report, or compares a baseline with it, only for the same lines.
    body["lines_scored"] = sorted(lines) if lines is not None else None
    # Whether the documents were served through fitted calibrators. Without
    # them every field is flagged, and the gate must not reuse this report
    # once the release has calibrators (pipeline_dag.eval_report_metrics).
    body["calibrated"] = calibrators is not None
    # Which frozen set of which tenant this was scored on, under which scoring
    # code, and how many of its documents: the gate compares two reports only
    # on one yardstick, and decides from the real documents' counts whether the
    # release was scored on enough to be production (synthetic twins do not count).
    from evaluation.run_eval import eval_set_record

    body["eval_set"] = eval_set_record(client, tenant_id, documents)
    body.update(real_only_section(version, triples, corpus_version=corpus_version))
    body["windowing_ceiling"] = windowing_ceiling(documents)
    # Each field's accuracy against the promoted release's report (reported,
    # never gated): where this release gained and lost.
    promoted = promoted_report if promoted_report is not None else _promoted_report(client, tenant_id, version)
    body["field_delta_vs_promoted"] = field_deltas(body, promoted)
    # P95 latency per adapter and peak GPU memory, for the release bundle
    # (recorded, not gated; evaluation.release_measurements).
    from evaluation.release_measurements import measurements

    # A call with no adapter here ran on the model this eval loaded: the base
    # with the release's adapter applied, named by its run.
    body["release_measurements"] = measurements(triples, served_as=ADAPTER.run_id(version),
                                                peak_mb=memory.peak_mb)
    # Which adapter checkpoint was scored: the gate reuses this report only for
    # that checkpoint, never for another one selected since (decision 2026-10-23).
    body["adapter_checkpoint"] = adapter_checkpoint
    # Every reading's served answer, as scored, under the tenant: a later gate
    # on other scoring code re-scores them rather than judging on floors alone
    # (evaluation.rescore, decision 2026-10-26). The model is never run again.
    from evaluation.rescore import store_answers

    body["answers_stored"] = store_answers(client, body, triples, documents, version=version,
                                           tenant_id=tenant_id)
    stored = body
    if failures:
        # The gate this body goes to blocks on them ("re-run the gate"). The
        # report kept in Blob holds its metrics apart, so the re-run scores
        # again rather than reading this partial measurement back
        # (pipeline_dag.eval_report_metrics scores again a report with no gate
        # metrics).
        body["gate_metrics"] = {**body.get("gate_metrics", {}), INFRASTRUCTURE_FAILURES: len(failures)}
        stored = {**body, "gate_metrics": {}, "gate_metrics_partial": body["gate_metrics"]}
    client.write_json(paths.eval_report(version, scope=ADAPTER.name), stored)

    log.info("golden eval of %s: %d document-mode run(s), %d infrastructure failure(s)",
             version, len(triples), len(failures))
    return body


#: Where the eval report's gate metrics carry the readings that failed to
#: extract twice (``gating.INFRASTRUCTURE_FAILURES`` reads the same name).
INFRASTRUCTURE_FAILURES = "infrastructure_failures"


def _promoted_report(client: BlobClient, tenant_id: str | None, version: str) -> dict[str, Any] | None:
    """This tenant's promoted release's eval report, other than ``version``'s,
    for the per-field delta; None when there is none, or it cannot be read
    (a diagnostic never fails the eval). Found as the gate finds it
    (``pipeline_dag.promoted_release``): the release index, a release of this
    adapter."""
    from common.adapter import ADAPTER

    try:
        from common.run_ids import parse_run_id
        from serving.release_router import promoted_rows

        for row in promoted_rows(client, tenant_id):
            release_id = str(row.get("release_id") or "")
            if release_id == version:
                continue
            promoted = release_id if paths.version_number(release_id) is not None else None
            if promoted is None:
                parsed = parse_run_id(str(row.get("adapter") or ""))
                if parsed is None or parsed.lineage != ADAPTER.name:
                    return None
                promoted = parsed.version
            if promoted == version:
                continue
            key = paths.eval_report(promoted, scope=ADAPTER.name)
            return client.read_json(key) if client.exists(key) else None
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail the eval
        log.info("golden eval: the promoted release's report could not be read for the field delta: %s", exc)
    return None


def windowing_ceiling(documents: Sequence[GoldenDocument]) -> dict[str, Any] | None:
    """What reading these policies in windows loses before any model runs
    (evaluation.windowing_ceiling): the oracle merge over the same documents
    and pages the gate just scored. Reported beside the gate, never gated -
    it measures the pipeline, not the candidate. Read against
    field_normalized_match it says how much of the gap is the model's.
    """
    from evaluation.windowing_ceiling import ceiling, oracle

    results = []
    for doc in documents:
        if doc.doc_type != "policy":
            continue
        texts = [doc.page_texts.get(p, "") for p in sorted(doc.page_texts)]
        try:
            results.append(oracle(doc.source_id, doc.golden, doc.lob, len(doc.image_keys),
                                  texts if any(texts) else None, doc.synthetic))
        except Exception as exc:  # noqa: BLE001 - a diagnostic must never fail the gate
            log.warning("windowing ceiling: %s could not be measured: %s", doc.source_id, exc)
    return ceiling(results) if results else None


def real_only_section(
    version: str, triples: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], **report_kwargs: Any
) -> dict[str, Any]:
    """The same report over the real documents alone, when the set mixes in synthetic ones.

    A delivered split can put synthetic twins of held-out sources into the eval
    set. Their layouts are unseen, which makes them a fair test, but their labels
    came from a generator — so the real documents are always scored apart, and
    the gap between the two is how much the generator flatters the model. Reported
    beside the gate, never instead of it.
    """
    from evaluation.run_eval import build_report

    real = [t for t in triples if not t[2].get("synthetic")]
    synthetic = len(triples) - len(real)
    if not synthetic:
        return {}
    section: dict[str, Any] = {
        "composition": {
            "real_documents": len({t[2]["source_id"] for t in real}),
            "synthetic_documents": len({t[2]["source_id"] for t in triples if t[2].get("synthetic")}),
        },
    }
    if real:
        real_report = build_report(version, real, **report_kwargs).as_dict()
        section["real_only"] = {"gate_metrics": real_report.get("gate_metrics")}
    else:
        section["real_only"] = None
        log.warning("the eval set holds no real documents: every score is on generated labels")
    return section


def dumps(body: dict[str, Any]) -> str:  # pragma: no cover - CLI helper
    return json.dumps(body, indent=2, sort_keys=True)
