"""The shadow check: a gated release read on the real server path before it is promoted.

    python -m orchestration.run shadow --version v3 \\
        --model-url https://<pod>-8000.proxy.runpod.net --mineru-url https://<pod>-8001.proxy.runpod.net \\
        [--documents 50] [--tenant T]

Decision 2026-10-26 (#44): ``package`` no longer promotes. A release that
passed its gate is written ``gated``, and ``promote`` asks for this check first
(``registry_utils.releases.promotion_clearance``).

**Why.** The gate scores the candidate in the training pod: vLLM in-process,
the frozen set's page images and text. Production reads differently: the Azure
worker (``azure_service``) renders each PDF, asks the pod's stock vLLM for the
release's LoRA by name (``policy-vN``) and reads the fields from the page
images first, holds them to MinerU 4's text of the pages they cite, then to the
whole document's (``serving.fields_first.read_fields``, ``add_full_text``).
This check reads documents THAT way - the same functions and the same runtime
``azure_service.runtime`` builds (the cold-start checks, the release's
calibrators, the serving thresholds, the pinned MinerU reader) - twice: once
with the candidate ``policy-vN`` and once with the promoted release
``policy-vK``. With nothing promoted, the candidate alone is read and held to
the golds.

**The pod.** Its vLLM serves both adapters at once: both ``--lora-modules``
and ``--max-loras 2`` (:data:`MAX_LORAS`; ``scripts.served_adapter`` prints
``--max-loras 1``, enough for production's one release). Each name is checked
as production checks it: vLLM serves it, on our base
(``inference_core.openai_backend.OpenAIChatBackend``); a name it does not serve,
or serves on another base, fails the check. The tokens are the Azure worker's:
``MODEL_TOKEN`` / ``MINERU_TOKEN``, else ``FIDEON_REMOTE_TOKEN``.

**Documents.** About :data:`SHADOW_DOCUMENTS` of the tenant's frozen eval set,
chosen deterministically (:func:`select_documents`): policies of the lines the
candidate serves, a line at a time in turn (stratified by line), every real
document before any synthetic twin, each line's in a fixed hashed order. Each
is sent as its original PDF (``raw-documents/``, read with the OCR stage's
login: the MinerU API reads the PDF, as production's does); a document with
none is passed over and counted.

**Scoring.** The gate's own (``evaluation.run_eval.build_report``), against
the frozen golds as the gate reads them (``evaluation.golden_eval.load_golden_set``):
served field match (``field_normalized_match``), invented-value rate
(``hallucination_rate``, from serving's printed-values check on MinerU's text)
and schema validity (``schema_validity_rate``, serving's own verdict) - on the
documents both releases read (paired). A reading that fails is tried once more
(:data:`READING_ATTEMPTS`); one that fails again is an infrastructure failure,
a timeout among them when a timeout caused it.

**Pass rule** (:func:`verdict`). Against the promoted release, on the same
documents: field match no more than :data:`MAX_FIELD_MATCH_DROP` below it
(a point estimate: at 50 documents a paired interval is wider than the rule),
an invented-value rate no more than :data:`MAX_INVENTION_RISE` above it,
schema validity not lower, no infrastructure failure. With nothing promoted:
no infrastructure failure and schema validity of at least
:data:`MIN_SCHEMA_VALIDITY_ALONE`. A metric not measured on a side fails its
rule: absence is not success. A promoted release this code cannot read fails
the check too - nothing could be compared - with the candidate's own numbers
reported, for the person deciding whether to skip it in writing.

**Time to the fields** (:func:`latency`, :func:`latency_check`). The
production target (2026-10-24): a 200-page scan's fields in 2 minutes on one
H100 (:data:`FIELDS_TARGET_SECONDS`, :data:`FIELDS_TARGET_PAGES`). Each
candidate reading is timed from the PDF to its fields being ready - phase 1
served, or, when nothing could be checked and the fields wait for the full
text, the whole document read. Two rules, both on the candidate alone:

* each document against its budget (:func:`fields_budget`: 2 minutes up to
  200 pages, in proportion beyond): the slowest 5% may not exceed it;
* a 200-page scan's time projected from the documents read
  (:func:`projected_seconds`): a line through their times (an overhead plus
  seconds a page), at 200 pages, raised by how far the slowest 5% ran above
  the line. Most frozen policies are far shorter than 200 pages, so the first
  rule alone would pass a release too slow for the long ones. The projection
  needs :data:`MIN_PROJECTION_DOCUMENTS` documents of two lengths at least;
  with fewer, the first rule decides and the report says so.

The documents are read one at a time, each with the GPU to itself - as the
Azure worker reads a job - and the time includes sending the pages from where
the check runs: run it near the pod, as the worker runs, or the time measured
is the upload's. A release slower than the target is promoted only knowingly:
``promote --accept-latency "<why>"``, recorded with the person's name
(``registry_utils.releases.promotion_clearance``). The promoted release's times
are reported beside the candidate's, not ruled on.

**Recorded, not ruled on**: each reading's ``prompt_check`` (``window_stats``):
a window whose prompt length the server reported is implausible for the pages
sent means vLLM resized the pages to another budget than the request's - a
parity signal between the server path and training. Also failed windows and
seconds per document.

Writes ``shadow_report.json`` under the release
(``registry_utils.releases.shadow_report_key``) and prints a short table.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient

log = logging.getLogger(__name__)

#: Documents of the frozen eval set read by default (``--documents``).
SHADOW_DOCUMENTS = 50
#: The candidate's served field match may be at most this far below the
#: promoted release's, on the same documents.
MAX_FIELD_MATCH_DROP = 0.02
#: The candidate's invented-value rate may be at most this far above the promoted release's.
MAX_INVENTION_RISE = 0.005
#: With nothing promoted: the schema validity the candidate needs on its own.
MIN_SCHEMA_VALIDITY_ALONE = 0.98
#: Tries of one reading before it counts as an infrastructure failure (as the golden eval's).
READING_ATTEMPTS = 2
#: The LoRAs the pod's vLLM must hold at once: the candidate and the promoted release.
MAX_LORAS = 2
#: The production target (2026-10-24): a scan of this many pages has its fields in this many seconds.
FIELDS_TARGET_SECONDS = 120.0
FIELDS_TARGET_PAGES = 200
#: The share of documents held to their time: the slowest 5% may not exceed it.
LATENCY_QUANTILE = 0.95
#: Fewest timed documents (of two lengths at least) a 200-page scan's time is projected from.
MIN_PROJECTION_DOCUMENTS = 10
#: Salts the fixed order each line's documents are taken in.
SELECTION_SALT = "shadow-check"
#: The metrics compared, as ``EvalReport.gate_metrics`` names them.
FIELD_MATCH, INVENTION, SCHEMA_VALIDITY = "field_normalized_match", "hallucination_rate", "schema_validity_rate"
SHADOW_METRICS = (FIELD_MATCH, INVENTION, SCHEMA_VALIDITY)
#: The time check's name, as the report and ``promote --accept-latency`` know it.
LATENCY = "latency"
CANDIDATE, PROMOTED = "candidate", "promoted"


class ShadowError(RuntimeError):
    """Raised when the shadow check cannot start: nothing is read, no report written."""


def thresholds() -> dict[str, Any]:
    """The pass rule's numbers, as the report records them."""
    return {"max_field_match_drop": MAX_FIELD_MATCH_DROP, "max_invention_rise": MAX_INVENTION_RISE,
            "min_schema_validity_alone": MIN_SCHEMA_VALIDITY_ALONE, "reading_attempts": READING_ATTEMPTS,
            "fields_target_seconds": FIELDS_TARGET_SECONDS, "fields_target_pages": FIELDS_TARGET_PAGES,
            "latency_quantile": LATENCY_QUANTILE, "min_projection_documents": MIN_PROJECTION_DOCUMENTS}


# --------------------------------------------------------------------------
# Which documents
# --------------------------------------------------------------------------


def document_line(doc: Any) -> str | None:
    """A frozen document's one line, as serving names a request's
    (``serving.vllm_entrypoint.request_lob``); ``None`` for none, a package of
    several, or a line serving refuses: the Azure API takes one line a job."""
    from serving.vllm_entrypoint import ServingError, request_lob

    lob = doc.lob
    if isinstance(lob, (list, tuple)):
        lob = lob[0] if len(lob) == 1 else None
    if not isinstance(lob, str) or not lob.strip():
        return None
    try:
        line = request_lob({"lob": lob})
    except ServingError:
        return None
    return line if isinstance(line, str) else None


def _order(source_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{source_id}".encode()).hexdigest()


def select_documents(documents: Sequence[Any], count: int, *, serves: Callable[[str], bool],
                     has_pdf: Callable[[str], bool]) -> tuple[list[tuple[Any, str]], dict[str, Any]]:
    """``count`` frozen policies, deterministically, and what was passed over.

    Policies of a line the candidate serves (``serves``), stratified by line:
    one line's next document, then the next line's, in turn. Every real
    document first, then synthetic twins; each line's in a fixed hashed order
    (:data:`SELECTION_SALT`), so the same set gives the same documents. A
    document without its original PDF (``has_pdf``) is passed over. Returns
    ``([(document, line)], passed_over)``.
    """
    passed: dict[str, list[str]] = {"not_a_policy": [], "no_single_line": [], "line_not_served": [], "no_pdf": []}
    strata: dict[bool, dict[str, list[Any]]] = {False: {}, True: {}}
    for doc in documents:
        if doc.doc_type != "policy":
            passed["not_a_policy"].append(doc.source_id)
            continue
        line = document_line(doc)
        if line is None:
            passed["no_single_line"].append(doc.source_id)
            continue
        if not serves(line):
            passed["line_not_served"].append(doc.source_id)
            continue
        strata[bool(doc.synthetic)].setdefault(line, []).append(doc)
    chosen: list[tuple[Any, str]] = []
    for synthetic in (False, True):
        queues = {line: sorted(docs, key=lambda d: _order(d.source_id))
                  for line, docs in sorted(strata[synthetic].items())}
        while len(chosen) < count and any(queues.values()):
            for line in list(queues):
                if len(chosen) >= count:
                    break
                while queues[line]:
                    doc = queues[line].pop(0)
                    if has_pdf(doc.source_id):
                        chosen.append((doc, line))
                        break
                    passed["no_pdf"].append(doc.source_id)
    return chosen, {key: sorted(ids) for key, ids in passed.items() if ids}


# --------------------------------------------------------------------------
# Who reads: each release as the Azure worker loads it
# --------------------------------------------------------------------------


@dataclass
class Side:
    """One release as the Azure worker would read with it."""

    name: str
    release_id: str
    model_name: str
    plan: Any = None
    runtime: Any = None
    #: Why this side cannot read at all: the adapter check, or the cold-start checks.
    error: str | None = None


def side_for(name: str, bundle: dict[str, Any], tenant: str) -> Side:
    """The release's serving plan, held to the cold-start checks the Azure
    worker makes (``azure_service.runtime.check_release``); ``error`` says why
    it fails them."""
    from azure_service.runtime import check_release
    from scripts.served_adapter import lora_name
    from serving.release_router import ServingPlan, _as_release
    from serving.vllm_entrypoint import ColdStartError

    release_id = str(bundle.get("release_id"))
    side = Side(name, release_id, lora_name(release_id), plan=ServingPlan(release=_as_release(bundle, "bf16")))
    try:
        check_release(side.plan)
    except ColdStartError as exc:
        side.error = f"this code would not serve {release_id}: {exc}"
    return side


def load_side(side: Side, client: BlobClient, tenant: str, models: Callable[[str], Any]) -> None:
    """The served model by its LoRA name (vLLM checked to serve it, on our
    base) and the release's calibrators: ``azure_service.runtime.load_tenant``,
    for a release that need not be the promoted one."""
    from azure_service.runtime import TenantRuntime, load_calibrators

    if side.error:
        return
    try:
        model = models(side.model_name)
    except Exception as exc:  # noqa: BLE001 - the adapter check: reported, never a crash
        side.error = f"{type(exc).__name__}: {exc}"
        return
    side.runtime = TenantRuntime(tenant=tenant, plan=side.plan, release_model=model,
                                 release_runtime=load_calibrators(side.plan.release, client))


# --------------------------------------------------------------------------
# One reading: the Azure worker's two phases
# --------------------------------------------------------------------------


@dataclass
class Reading:
    """One document read by one release."""

    source_id: str
    result: Any = None
    pages: int = 0
    seconds: float = 0.0
    #: From the PDF to the fields being ready, on the attempt that read it (:func:`read_once`).
    fields_seconds: float | None = None
    attempts: int = 0
    route: str | None = None
    fields_served: bool | None = None
    error: str | None = None
    timeout: bool = False
    #: The release does not serve the document's line (the promoted one may not).
    not_served: str | None = None
    retried: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.result is not None


def _is_timeout(exc: BaseException | None) -> bool:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def read_once(side: Side, doc: Any, line: str, pdf: bytes, *, reader: Any, tuning: dict[str, Any],
              work_dir: Path) -> tuple[Any, int, str, bool, float]:
    """``doc`` read as the Azure worker reads a job: fields first, checked on
    the pages they cite, then the whole document's text
    (``azure_service.worker.Worker._read_fields`` and ``_full_text``). Returns
    ``(result, pages, route, fields served, seconds to the fields)``: the
    fields are ready when phase 1 serves them or, when it serves none (nothing
    could be checked), once the whole document is read."""
    from azure_service.runtime import identity_calibration
    from serving.doc_type_classifier import StaticClassifier
    from serving.fields_first import add_full_text, read_fields

    reading = side.runtime.reading(line)
    folder = work_dir / side.name / doc.source_id
    name = f"{doc.source_id}.pdf"
    began = time.perf_counter()
    try:
        fields = read_fields(
            pdf, name=name, lob=line, source_id=doc.source_id, model=reading.model, reader=reader,
            pages_dir=folder / "pages", classifier=StaticClassifier("policy"),
            calibration=identity_calibration(reading.model), verify=True,
            extract_options=reading.extract_options(tuning))
        fields_ready = time.perf_counter() - began
        result, count = fields["result"], len(fields["images"])
        add_full_text(pdf, result, name=name, lob=line, reader=reader, pages_dir=folder / "pages",
                      page_count=count, known_texts=fields["cited_texts"],
                      calibrators=reading.calibrators, thresholds=reading.thresholds)
        served = fields["served"] is not None
        if not served:
            fields_ready = time.perf_counter() - began
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return result, count, reading.route, served, fields_ready


def read_document(side: Side, doc: Any, line: str, pdf: bytes, *, reader: Any, tuning: dict[str, Any],
                  work_dir: Path) -> Reading:
    """:func:`read_once`, tried :data:`READING_ATTEMPTS` times; a line the
    release does not serve is recorded, never tried."""
    from azure_service.runtime import JobRefused

    reading = Reading(doc.source_id)
    began = time.perf_counter()
    for attempt in range(1, READING_ATTEMPTS + 1):
        reading.attempts = attempt
        try:
            (reading.result, reading.pages, reading.route, reading.fields_served,
             reading.fields_seconds) = read_once(side, doc, line, pdf, reader=reader, tuning=tuning,
                                                 work_dir=work_dir)
            reading.error, reading.timeout = None, False
            break
        except JobRefused as exc:
            reading.not_served = str(exc)
            break
        except Exception as exc:  # noqa: BLE001 - retried, then an infrastructure failure, never a crash
            reading.error = f"{type(exc).__name__}: {exc}"[:500]
            reading.timeout = _is_timeout(exc)
            reading.retried.append(reading.error)
            log.warning("shadow: %s read by %s failed (attempt %d of %d): %s", doc.source_id,
                        side.model_name, attempt, READING_ATTEMPTS, reading.error)
    reading.seconds = round(time.perf_counter() - began, 2)
    if reading.ok and reading.retried:
        reading.retried = reading.retried[:1]
    return reading


# --------------------------------------------------------------------------
# Scoring: the gate's own
# --------------------------------------------------------------------------


def triple(doc: Any, reading: Reading) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """``(gold, served answer, metadata)`` as the golden eval hands
    ``build_report`` a reading through serving (``golden_eval._read``):
    image_only - fields first reads the images - carrying serving's own
    printed-values record and its schema verdict."""
    from evaluation.run_eval import PRINTED_CHECK

    result = reading.result
    metadata: dict[str, Any] = {
        "source_id": doc.source_id, "doc_type": doc.doc_type, "acord_form": doc.acord_form, "lob": doc.lob,
        "modality_mode": "image_only", "is_scanned": doc.is_scanned, "synthetic": doc.synthetic,
        "held_out_carrier": doc.held_out_carrier, "page_count": reading.pages or len(doc.image_keys),
        PRINTED_CHECK: getattr(result, "printed_check", None),
        "field_confidence": {path: body.get("confidence") for path, body in (result.fields or {}).items()
                             if isinstance(body, dict)},
        "schema_valid": result.schema_valid,
        "validation_errors": list(result.validation_errors[:4]),
    }
    if doc.carrier:
        metadata["carrier"] = doc.carrier
    if doc.carrier_seen_in_train is not None:
        metadata["carrier_seen_in_train"] = doc.carrier_seen_in_train
    return doc.golden, result.extraction, metadata


def _rounded(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def score(label: str, documents: Sequence[Any], readings: dict[str, Reading]) -> tuple[dict[str, Any], dict[str, float]]:
    """The gate's metrics over ``documents`` as ``readings`` read them
    (``evaluation.run_eval.build_report``), and each document's field match."""
    from evaluation.run_eval import build_report

    triples = [triple(doc, readings[doc.source_id]) for doc in documents]
    if not triples:
        return {"documents": 0, **dict.fromkeys(SHADOW_METRICS)}, {}
    report = build_report(label, triples)
    metrics = report.gate_metrics()
    per_document = report.per_document_scores().get(FIELD_MATCH, {})
    return {"documents": len(triples), **{name: _rounded(metrics.get(name)) for name in SHADOW_METRICS}}, per_document


def prompt_parity(readings: dict[str, Reading]) -> dict[str, Any]:
    """The ``prompt_check`` records of every window read (``window_stats``):
    how many were checked, how many the server's prompt length makes
    implausible, and on which documents. Reported, not ruled on."""
    checked = implausible = 0
    documents: list[str] = []
    for source_id, reading in readings.items():
        if not reading.ok:
            continue
        flagged = False
        for stat in getattr(reading.result, "window_stats", None) or []:
            check = stat.get("prompt_check") if isinstance(stat, dict) else None
            if not isinstance(check, dict):
                continue
            checked += 1
            if check.get("plausible") is False:
                implausible += 1
                flagged = True
        if flagged:
            documents.append(source_id)
    return {"windows_checked": checked, "implausible": implausible, "documents": sorted(documents)}


def _failed_windows(reading: Reading) -> int:
    return sum(1 for stat in getattr(reading.result, "window_stats", None) or []
               if isinstance(stat, dict) and stat.get("outcome") == "failed")


def failures(side: Side | None, readings: dict[str, Reading]) -> dict[str, Any]:
    """A side's infrastructure failures: the adapter check, and each reading
    that failed every attempt (timeouts among them)."""
    if side is None:
        return {"count": 0, "timeouts": 0, "adapter_check": None, "documents": {}}
    failed = {source_id: r.error for source_id, r in readings.items() if not r.ok and r.not_served is None}
    timeouts = sum(1 for source_id in failed if readings[source_id].timeout)
    return {"count": len(failed) + (1 if side.error else 0), "timeouts": timeouts,
            "adapter_check": side.error, "documents": failed}


# --------------------------------------------------------------------------
# Time to the fields
# --------------------------------------------------------------------------


def fields_budget(pages: int) -> float:
    """A document's time to its fields under the target: :data:`FIELDS_TARGET_SECONDS`
    up to :data:`FIELDS_TARGET_PAGES` pages, in proportion beyond."""
    return FIELDS_TARGET_SECONDS * max(1.0, pages / FIELDS_TARGET_PAGES)


def _quantile(values: Sequence[float], q: float) -> float:
    """The nearest-rank quantile: of 50 values, the 95th percentile is the 48th
    smallest, so the slowest 5% are the two above it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def projected_seconds(timed: Sequence[tuple[int, float]]) -> dict[str, Any] | None:
    """A 200-page scan's time to its fields, from ``(pages, seconds)`` of the
    documents read; ``None`` with fewer than :data:`MIN_PROJECTION_DOCUMENTS`
    or a single length.

    A least-squares line, ``overhead + per_page x pages``, at
    :data:`FIELDS_TARGET_PAGES`, times the :data:`LATENCY_QUANTILE` quantile of
    each document's time over the line's (``spread``): the slowest 5% ran that
    far above it. A line that does not rise with the pages, or reaches zero
    within them, is noise on too narrow a range: the time is then taken in
    proportion to the pages (``proportional``), which overstates a short
    document's share and so errs slow.
    """
    timed = [(pages, seconds) for pages, seconds in timed if pages > 0]
    if len(timed) < MIN_PROJECTION_DOCUMENTS or len({pages for pages, _ in timed}) < 2:
        return None
    count = len(timed)
    mean_pages = sum(pages for pages, _ in timed) / count
    mean_seconds = sum(seconds for _, seconds in timed) / count
    spread_pages = sum((pages - mean_pages) ** 2 for pages, _ in timed)
    per_page = sum((pages - mean_pages) * (seconds - mean_seconds) for pages, seconds in timed) / spread_pages
    overhead, fit = mean_seconds - per_page * mean_pages, "least_squares"
    if per_page <= 0 or any(overhead + per_page * pages <= 0 for pages, _ in timed):
        per_page = sum(seconds for _, seconds in timed) / sum(pages for pages, _ in timed)
        overhead, fit = 0.0, "proportional"
    spread = (_quantile([seconds / (overhead + per_page * pages) for pages, seconds in timed], LATENCY_QUANTILE)
              if per_page > 0 else 1.0)
    return {"seconds": round((overhead + per_page * FIELDS_TARGET_PAGES) * spread, 1), "fit": fit,
            "overhead_s": round(overhead, 2), "per_page_s": round(per_page, 3), "spread": round(spread, 3),
            "documents": count, "longest_pages": max(pages for pages, _ in timed)}


def latency(readings: dict[str, Reading]) -> dict[str, Any]:
    """Time to the fields of the documents ``readings`` read: each against its
    budget (:func:`fields_budget`) and a 200-page scan's projected
    (:func:`projected_seconds`)."""
    timed = sorted((r.pages, r.fields_seconds, source_id) for source_id, r in readings.items()
                   if r.ok and r.fields_seconds is not None and r.pages > 0)
    if not timed:
        return {"documents": 0, "projection": None}
    seconds = [s for _p, s, _id in timed]
    shares = [s / fields_budget(p) for p, s, _id in timed]
    return {
        "documents": len(timed),
        "pages": {"shortest": timed[0][0], "longest": timed[-1][0]},
        "fields_seconds": {"median": round(_quantile(seconds, 0.5), 2),
                           "p95": round(_quantile(seconds, LATENCY_QUANTILE), 2), "longest": round(max(seconds), 2)},
        # The slowest 5%'s time as a share of their budget: above 1, more than 5% were over it.
        "budget_share": round(_quantile(shares, LATENCY_QUANTILE), 3),
        "over_budget": sorted(source_id for p, s, source_id in timed if s > fields_budget(p)),
        "projection": projected_seconds([(p, s) for p, s, _id in timed]),
    }


def latency_check(measured: dict[str, Any]) -> dict[str, Any]:
    """The time check on the candidate's :func:`latency`: the slowest 5% within
    their budget and, where it could be projected, a 200-page scan's fields
    within :data:`FIELDS_TARGET_SECONDS`. Nothing timed fails it."""
    rule = f"{FIELDS_TARGET_PAGES} pages <= {FIELDS_TARGET_SECONDS:g} s"
    if not measured.get("documents"):
        return _check(LATENCY, False, "no document's fields were timed", rule=rule)
    share, projection = measured["budget_share"], measured["projection"]
    within = share <= 1.0 + 1e-9
    over = measured["over_budget"]
    slowest = f"the slowest {100 - round(LATENCY_QUANTILE * 100)}% at {share:.2f} of their budget"
    parts = [slowest + (f" ({len(over)} of {measured['documents']} over it)" if over else "")]
    projected = None
    if projection is None:
        parts.append(f"a {FIELDS_TARGET_PAGES}-page scan not projected ({measured['documents']} document(s) "
                     f"timed; {MIN_PROJECTION_DOCUMENTS} of two lengths at least are needed)")
    else:
        projected = projection["seconds"]
        within = within and projected <= FIELDS_TARGET_SECONDS + 1e-9
        parts.append(f"a {FIELDS_TARGET_PAGES}-page scan projected at {projected:g} s from documents of "
                     f"{measured['pages']['shortest']}-{measured['pages']['longest']} pages")
    return _check(LATENCY, within, f"time to the fields: {'; '.join(parts)} (rule {rule})",
                  budget_share=share, projected_seconds=projected, rule=rule)


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def _check(name: str, passed: bool, detail: str, **numbers: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail, **numbers}


def verdict(*, candidate: dict[str, Any], promoted: dict[str, Any] | None, infrastructure: int,
            baseline_problem: str | None = None) -> list[dict[str, Any]]:
    """The checks the shadow report passes on. ``candidate``/``promoted``: the
    metrics on the paired documents (``promoted`` None with nothing promoted);
    ``infrastructure``: failures on either side; ``baseline_problem``: why a
    promoted release could not be compared."""
    checks = [_check("infrastructure", infrastructure == 0,
                     "no reading failed, timed out or found its adapter missing" if infrastructure == 0
                     else f"{infrastructure} infrastructure failure(s)", failures=infrastructure)]
    if baseline_problem:
        checks.append(_check("comparison", False, baseline_problem))
    if promoted is None:
        validity = candidate.get(SCHEMA_VALIDITY)
        checks.append(_check(
            "schema_validity", isinstance(validity, (int, float)) and validity >= MIN_SCHEMA_VALIDITY_ALONE,
            f"schema validity {validity} on its own, at least {MIN_SCHEMA_VALIDITY_ALONE}",
            candidate=validity, rule=f">= {MIN_SCHEMA_VALIDITY_ALONE}"))
        return checks
    if not candidate.get("documents"):
        checks.append(_check("paired_documents", False, "no document was read by both releases"))
        return checks

    def compare(metric: str, rule: str, ok: Callable[[float], bool], what: str) -> None:
        mine, theirs = candidate.get(metric), promoted.get(metric)
        if not isinstance(mine, (int, float)) or not isinstance(theirs, (int, float)):
            checks.append(_check(metric, False, f"{what} not measured on both releases "
                                 f"(candidate {mine}, promoted {theirs})", candidate=mine, promoted=theirs, rule=rule))
            return
        delta = round(mine - theirs, 4)
        checks.append(_check(metric, ok(delta), f"{what} {mine} against {theirs} ({delta:+.4f}, rule {rule})",
                             candidate=mine, promoted=theirs, delta=delta, rule=rule))

    compare(FIELD_MATCH, f">= -{MAX_FIELD_MATCH_DROP}", lambda d: d >= -MAX_FIELD_MATCH_DROP - 1e-9,
            "served field match")
    compare(INVENTION, f"<= +{MAX_INVENTION_RISE}", lambda d: d <= MAX_INVENTION_RISE + 1e-9,
            "invented-value rate")
    compare(SCHEMA_VALIDITY, ">= 0", lambda d: d >= -1e-9, "schema validity")
    return checks


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------


def _token(*names: str) -> str | None:
    return next((os.environ[name].strip() for name in names if (os.environ.get(name) or "").strip()), None)


def served_settings(model_url: str, mineru_url: str, tenant: str) -> Any:
    """The Azure worker's settings for these servers (``azure_service.settings``),
    its tokens read as it reads them."""
    from azure_service.settings import Settings

    return Settings(model_url=model_url, model_token=_token("MODEL_TOKEN", "FIDEON_REMOTE_TOKEN"),
                    mineru_url=mineru_url, mineru_token=_token("MINERU_TOKEN", "FIDEON_REMOTE_TOKEN"),
                    tenants=(tenant,))


def run_shadow(
    client: BlobClient,
    version: str,
    *,
    model_url: str | None = None,
    mineru_url: str | None = None,
    documents: int = SHADOW_DOCUMENTS,
    tenant_id: str | None = None,
    raw_client: BlobClient | None = None,
    models: Callable[[str], Any] | None = None,
    reader: Any = None,
    golden: Sequence[Any] | None = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Read ``version`` (and the promoted release) on the server path, score,
    judge, and write the shadow report. Returns the report.

    ``models``/``reader``/``golden``/``raw_client`` stand in for the pod's
    vLLM, the MinerU API, the frozen set and the raw-documents login (tests);
    by default they are the Azure worker's (``azure_service.runtime.openai_models``,
    ``served_reader``), the tenant's frozen set and the OCR stage's login.
    """
    from common.config import served_mineru, serving_config
    from registry_utils.releases import (
        ReleaseError,
        gate_tier,
        promotable_bundle,
        promoted_release_id,
        shadow_report_key,
    )
    from serving.vllm_entrypoint import assert_no_retired_settings, serving_thresholds

    tenant = paths._tenant(tenant_id)
    if documents < 1:
        raise ShadowError("--documents is 1 or more")
    index_key = paths.release_index(tenant_id)
    status = next((row.get("status") for row in (client.read_json(index_key) if client.exists(index_key) else [])
                   if row.get("release_id") == version), None)
    if status == "promoted":
        raise ShadowError(f"{version} is already the promoted release for tenant {tenant}: nothing to shadow")
    if status == "archived":
        raise ShadowError(f"{version} was promoted before (archived): a rollback to it needs no shadow check "
                          f"(promote --version {version} --by \"<person>\")")
    try:
        bundle = promotable_bundle(client, version, tenant_id)
    except ReleaseError as exc:
        raise ShadowError(f"{version} would not be promoted whatever its shadow check says: {exc}") from exc
    candidate = side_for(CANDIDATE, bundle, tenant)
    if candidate.error:
        raise ShadowError(candidate.error)

    current = promoted_release_id(client, tenant_id, excluding=version)
    promoted: Side | None = None
    baseline_problem = None
    if current is not None:
        key = paths.release_bundle(current, tenant_id)
        if client.exists(key):
            promoted = side_for(PROMOTED, client.read_json(key), tenant)
        else:
            promoted = Side(PROMOTED, current, f"policy-{current}", error=f"{current} has no bundle at {key}")
        if promoted.error:
            baseline_problem = (f"the promoted release {current} cannot be read by this code, so nothing was "
                                f"compared ({promoted.error}); the candidate's own numbers are reported")
            promoted.runtime = None

    if golden is None:
        from evaluation.freeze_eval_set import is_frozen
        from evaluation.golden_eval import load_golden_set

        if not is_frozen(client, tenant_id=tenant_id):
            raise ShadowError(f"tenant {tenant} has no frozen eval set: there are no golds to score against")
        golden = load_golden_set(client, ["policy"], tenant_id=tenant_id)
    raw = raw_client
    if raw is None:
        from artifact_registry.blob_client import for_ocr

        raw = for_ocr()

    def serves(line: str) -> bool:
        try:
            return candidate.plan.route("policy", line, allow_fallback=True).release is not None
        except Exception:  # noqa: BLE001 - a line the plan cannot route is not served
            return False

    def has_pdf(source_id: str) -> bool:
        return raw.exists(paths.raw_pdf("policy", source_id, tenant_id))

    chosen, passed_over = select_documents(golden, documents, serves=serves, has_pdf=has_pdf)
    if not chosen:
        raise ShadowError(f"no frozen policy of a line {version} serves has its original PDF in raw-documents/ "
                          f"(passed over: { {k: len(v) for k, v in passed_over.items()} })")

    assert_no_retired_settings(serving_config())
    tuning = serving_thresholds()
    if models is None or reader is None:
        if not model_url or not mineru_url:
            raise ShadowError("the shadow check needs --model-url and --mineru-url")
        from azure_service.runtime import openai_models, served_reader

        settings = served_settings(model_url, mineru_url, tenant)
        models = models or openai_models(settings)
        reader = reader or served_reader(settings)
    load_side(candidate, client, tenant, models)
    if promoted is not None and not baseline_problem:
        load_side(promoted, client, tenant, models)
    # Nothing is read when the candidate's own adapter is missing: the check has failed.
    sides = ([side for side in (candidate, promoted) if side is not None and side.runtime is not None]
             if candidate.runtime is not None else [])
    if candidate.runtime is None:
        log.error("shadow %s: %s is not served as production would read it: %s", version,
                  candidate.model_name, candidate.error)

    owned = work_dir is None
    work = Path(work_dir or tempfile.mkdtemp(prefix=f"shadow-{version}-"))
    readings: dict[str, dict[str, Reading]] = {CANDIDATE: {}, PROMOTED: {}}
    try:
        for number, (doc, line) in enumerate(chosen, 1):
            try:
                pdf = raw.read_bytes(paths.raw_pdf("policy", doc.source_id, tenant_id))
            except Exception as exc:  # noqa: BLE001 - the document's own failure, on the candidate's side
                readings[CANDIDATE][doc.source_id] = Reading(doc.source_id, error=f"{type(exc).__name__}: {exc}")
                continue
            for side in sides:
                readings[side.name][doc.source_id] = read_document(side, doc, line, pdf, reader=reader,
                                                                   tuning=tuning, work_dir=work)
            log.info("shadow %s: %d of %d read (%s)", version, number, len(chosen), doc.source_id)
    finally:
        if owned:
            shutil.rmtree(work, ignore_errors=True)

    report = _report(
        client, version, tenant_id, bundle=bundle, candidate=candidate, promoted=promoted,
        baseline_problem=baseline_problem, chosen=chosen, passed_over=passed_over, readings=readings,
        requested=documents, model_url=model_url, mineru_url=mineru_url, pin=served_mineru(),
        tier=gate_tier(client, version, tenant_id, bundle),
    )
    key = shadow_report_key(version, tenant_id)
    client.write_json(key, report)
    report["report_key"] = key
    log.info("shadow %s: %s; report at %s", version, "PASSED" if report["passed"] else "FAILED", key)
    return report


def _report(client: BlobClient, version: str, tenant_id: str | None, *, bundle: dict[str, Any], candidate: Side,
            promoted: Side | None, baseline_problem: str | None, chosen: list[tuple[Any, str]],
            passed_over: dict[str, Any], readings: dict[str, dict[str, Reading]], requested: int,
            model_url: str | None, mineru_url: str | None, pin: dict[str, Any],
            tier: tuple[str | None, Any]) -> dict[str, Any]:
    docs = [doc for doc, _line in chosen]
    mine, theirs = readings[CANDIDATE], readings[PROMOTED]
    compared = promoted is not None and promoted.runtime is not None
    if promoted is not None and not compared and not baseline_problem:
        # Its adapter check failed (counted as an infrastructure failure too):
        # never judged by the rule for a tenant with nothing promoted.
        baseline_problem = f"the promoted release {promoted.release_id} could not be read: {promoted.error}"
        failed_baseline = failures(promoted, theirs)
    else:
        failed_baseline = None
    read_by_candidate = [doc for doc in docs if doc.source_id in mine and mine[doc.source_id].ok]
    paired = ([doc for doc in read_by_candidate if doc.source_id in theirs and theirs[doc.source_id].ok]
              if compared else read_by_candidate)
    own, own_scores = score(f"shadow:{version}", read_by_candidate, mine)
    paired_mine, paired_scores = score(f"shadow:{version}", paired, mine)
    paired_theirs, their_scores = (score(f"shadow:{promoted.release_id}", paired, theirs) if compared
                                   else (None, {}))
    failed = {CANDIDATE: failures(candidate, mine),
              PROMOTED: failed_baseline or (failures(promoted, theirs) if compared else failures(None, {}))}
    infrastructure = sum(f["count"] for f in failed.values())
    checks = verdict(candidate=paired_mine, promoted=paired_theirs if compared else None,
                     infrastructure=infrastructure, baseline_problem=baseline_problem)
    timing = {CANDIDATE: latency(mine), PROMOTED: latency(theirs) if compared else None}
    checks.append(latency_check(timing[CANDIDATE]))
    passed = all(check["passed"] for check in checks)
    lines: dict[str, int] = {}
    for _doc, line in chosen:
        lines[line] = lines.get(line, 0) + 1

    def row(doc: Any, line: str) -> dict[str, Any]:
        entry: dict[str, Any] = {"source_id": doc.source_id, "line": line, "synthetic": bool(doc.synthetic)}
        for name, by_source, scores in ((CANDIDATE, mine, own_scores), (PROMOTED, theirs, their_scores)):
            reading = by_source.get(doc.source_id)
            if reading is None:
                continue
            check = getattr(reading.result, "printed_check", None) if reading.ok else None
            entry[name] = {
                "ok": reading.ok, "error": reading.error if not reading.ok else None,
                "timeout": reading.timeout, "not_served": reading.not_served, "attempts": reading.attempts,
                "seconds": reading.seconds, "route": reading.route, "fields_served": reading.fields_served,
                "fields_seconds": (round(reading.fields_seconds, 2) if reading.fields_seconds is not None
                                   else None),
                "pages": reading.pages,
                "field_match": _rounded(scores.get(doc.source_id)),
                "schema_valid": reading.result.schema_valid if reading.ok else None,
                "invented": len(check.get("not_printed") or []) if isinstance(check, dict) else None,
                "checked": check.get("checked") if isinstance(check, dict) else None,
                "failed_windows": _failed_windows(reading) if reading.ok else None,
            }
        return entry

    return {
        "release_id": version,
        "tenant_id": paths._tenant(tenant_id),
        "created_at": datetime.now(UTC).isoformat(),
        # What the report read: promote holds it to the bundle as now packaged
        # and to the release promoted now (registry_utils.releases.shadow_problem).
        "bundle_created_at": bundle.get("created_at"),
        "promoted_release_id": promoted.release_id if promoted is not None else None,
        "candidate": {"release_id": version, "model": candidate.model_name, "adapter": bundle.get("adapter"),
                      "tier": tier[0], "tier_confidence": tier[1]},
        "baseline": ({"release_id": promoted.release_id, "model": promoted.model_name, "compared": compared,
                      "problem": baseline_problem} if promoted is not None else None),
        "servers": {"model_url": model_url, "mineru_url": mineru_url, "mineru_pin": dict(pin),
                    "max_loras": MAX_LORAS if promoted is not None else 1},
        "documents": {"requested": requested, "selected": len(chosen), "by_line": dict(sorted(lines.items())),
                      "real": sum(1 for doc in docs if not doc.synthetic),
                      "synthetic": sum(1 for doc in docs if doc.synthetic),
                      "paired": len(paired), "passed_over": passed_over,
                      "not_served_by_promoted": sorted(s for s, r in theirs.items() if r.not_served)},
        "metrics": {"paired": {CANDIDATE: paired_mine, PROMOTED: paired_theirs}, "candidate_all": own},
        "failures": failed,
        "prompt_check": {CANDIDATE: prompt_parity(mine), PROMOTED: prompt_parity(theirs) if compared else None},
        "latency": timing,
        "thresholds": thresholds(),
        "checks": checks,
        "passed": passed,
        "reasons": [check["detail"] for check in checks if not check["passed"]],
        "per_document": [row(doc, line) for doc, line in chosen],
    }


# --------------------------------------------------------------------------
# The console table
# --------------------------------------------------------------------------


def _number(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "-"


def render(report: dict[str, Any]) -> str:
    """The short console table of a shadow report."""
    docs = report["documents"]
    baseline = report.get("baseline")
    against = (f"vs {baseline['release_id']} (promoted)" if baseline and baseline.get("compared")
               else f"vs the golds ({baseline['release_id']} not compared)" if baseline
               else "vs the golds (nothing promoted)")
    by_line = ", ".join(f"{line} {n}" for line, n in docs["by_line"].items())
    out = [f"shadow {report['release_id']} {against}: {docs['selected']} document(s), {docs['paired']} paired "
           f"({docs['real']} real, {docs['synthetic']} synthetic; {by_line})"]
    mine = report["metrics"]["paired"][CANDIDATE] or {}
    theirs = report["metrics"]["paired"][PROMOTED] or {}
    rules = {check["check"]: check for check in report["checks"]}
    out.append(f"  {'metric':<24}{'candidate':>11}{'promoted':>11}{'delta':>9}  rule")
    for metric, label in ((FIELD_MATCH, "served field match"), (INVENTION, "invented-value rate"),
                          (SCHEMA_VALIDITY, "schema validity")):
        check = rules.get(metric, {})
        delta = check.get("delta")
        verdict_word = "" if not check else ("ok" if check["passed"] else "FAIL")
        out.append(f"  {label:<24}{_number(mine.get(metric)):>11}{_number(theirs.get(metric)):>11}"
                   f"{(f'{delta:+.3f}' if isinstance(delta, (int, float)) else '-'):>9}  "
                   f"{check.get('rule', '')} {verdict_word}".rstrip())
    failed = report["failures"]
    out.append(f"  {'failures / timeouts':<24}"
               f"{str(failed[CANDIDATE]['count']) + ' / ' + str(failed[CANDIDATE]['timeouts']):>11}"
               f"{str(failed[PROMOTED]['count']) + ' / ' + str(failed[PROMOTED]['timeouts']):>11}"
               f"{'':>9}  0 {'ok' if rules['infrastructure']['passed'] else 'FAIL'}")
    timing = report.get("latency") or {}
    if timing:
        def p95(side: str) -> str:
            measured = timing.get(side) or {}
            seconds = (measured.get("fields_seconds") or {}).get("p95")
            return f"{seconds:.1f} s" if isinstance(seconds, (int, float)) else "-"

        check = rules.get(LATENCY, {})
        projected = check.get("projected_seconds")
        out.append(f"  {'fields time, p95':<24}{p95(CANDIDATE):>11}{p95(PROMOTED):>11}{'':>9}  "
                   f"{check.get('rule', '')}; projected "
                   f"{f'{projected:g} s' if isinstance(projected, (int, float)) else '-'} "
                   f"{'ok' if check.get('passed') else 'FAIL'}")
    parity = report["prompt_check"]
    shown = [f"{side} {p['implausible']} of {p['windows_checked']}" for side, p in parity.items() if p]
    out.append(f"  {'prompt_check implausible':<24} {'; '.join(shown)}   (parity signal, not ruled on)")
    for check in report["checks"]:
        if not check["passed"] and check["check"] not in (FIELD_MATCH, INVENTION, SCHEMA_VALIDITY):
            out.append(f"  FAIL {check['check']}: {check['detail']}")
    where = report.get("report_key", "")
    out.append(f"  {'PASSED' if report['passed'] else 'FAILED'}: {where}")
    return "\n".join(out)
