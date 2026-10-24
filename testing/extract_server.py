"""PDF in, canonical JSON and timings out - an extraction test server for a RunPod pod.

    python -m testing.extract_server                       # on the pod, in /workspace/venv
    python -m testing.extract_server --port 8000 --gpu-memory-utilization 0.75
    python -m testing.extract_server --port 8002 --model-url http://localhost:8000 \\
        --mineru-url http://localhost:8001                 # the pod's own servers, already up

Reached from the laptop through the pod's RunPod proxy URL
(``https://<pod id>-<port>.proxy.runpod.net``; expose the port in the pod's
settings), with ``python -m testing.extract_client``. It exists to check the
extraction routine end to end on real PDFs and to time it: the production
endpoint (``serving/vllm_entrypoint.py``) is a Serverless handler that takes
page images and page text, never a PDF, and serves only a promoted release.

**One pipeline, as everywhere.** A document is OCR'd by MinerU and read by
:func:`serving.pipeline.extract` - the function production, the gate and the
testing harness call - with the base model (``--model base``'s weights) or a
LoRA adapter on it (``--adapter``). Nothing here reads a document its own way.

**Two environments.** MinerU runs in ``/workspace/venv-ocr``, vLLM here: MinerU
is a child process (:mod:`data_pipeline.ocr.ocr_worker`) kept loaded between
documents. vLLM starts first, holding ``--gpu-memory-utilization`` of the GPU
(0.75, not serving's 0.90), and MinerU takes what is left.

**Or the pod's own servers.** With ``--model-url`` and ``--mineru-url`` (a
``vllm serve`` and a MinerU 4 API on the same pod, on localhost) this loads
nothing itself: the pages are rendered while that MinerU reads them
(:mod:`data_pipeline.ocr.served_ocr`), and the windows go to that model, each
taken as it is answered (:mod:`inference_core.openai_backend`). Only the PDF
crosses the proxy - from a laptop, sending the page images with every window
was most of the time. ``--images file`` sends each image's path, not its
bytes; start vLLM with ``--allowed-local-media-path`` covering ``--work-dir``.
Give this server a port of its own (``--port 8002``) and expose it. Their
token is ``--remote-token`` or ``$FIDEON_REMOTE_TOKEN``, read at start - on the
pod an inline ``FIDEON_REMOTE_TOKEN=... python -m ...`` reaches the detached
server (``scripts/pod_run.sh`` keeps it over ``.env``). MinerU's tier is the
pinned one (``configs/shared/ocr.yaml``): another ``--tier`` is refused unless
``--any-mineru`` (timing only), before the command detaches.

**Jobs, not one long request.** The RunPod proxy closes a request that has not
been answered within about 100 seconds, and an 85-page policy takes minutes. A
PDF is posted and answered at once with a job id; the job is polled. Jobs run one
at a time, in the order posted, so each one's timings are its own.

**A token on every call but /health.** A proxy URL is public. The token is
``$EXTRACT_SERVER_TOKEN``, or one generated at start, written to
``<work dir>/token.txt`` (owner-only) and printed once. The documents stay on the
pod's volume under ``<work dir>/<job>/`` - the result JSON always, the PDF and
pages only with ``--keep-files`` - and their content is never logged.

    GET  /health                                       ready, model, jobs, the largest PDF taken
    POST /jobs?lob=personal_auto&mode=ocr_plus_image&name=a.pdf   body: the PDF -> 202 {"job": ...}
    GET  /jobs                                         every job, summarised
    GET  /jobs/<id>                                    one job: status, then its summary and timings
    GET  /jobs/<id>/result                             the summary and the served canonical JSON
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import queue
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

#: The token a caller must send as ``Authorization: Bearer <token>``.
TOKEN_ENV = "EXTRACT_SERVER_TOKEN"

#: The bearer token of the pod's own vLLM and MinerU, when they want one.
REMOTE_TOKEN_ENV = "FIDEON_REMOTE_TOKEN"

#: A PDF larger than this is refused before it is read: the proxy itself stops
#: at about 100 MB, and the 114-page scan in the data is 84 MB.
MAX_PDF_BYTES = 150 * 1024 * 1024

MODES = ("ocr_plus_image", "image_only", "fields_first")
#: The model reads the page images first, MinerU's text joins after (serving.fields_first): the
#: one-GPU production path. Needs the served MinerU (--mineru-url).
FIELDS_FIRST = "fields_first"

#: What the server runs MinerU with, in the OCR environment.
OCR_WORKER_COMMAND = "import sys; from data_pipeline.ocr.ocr_worker import serve; sys.exit(serve())"

OcrCall = Callable[..., dict[str, Any]]


class RequestError(ValueError):
    """A request this server refuses, with the reason the caller is told."""


def positive_int(value: str) -> int:
    """An argparse type: a whole number of at least 1 (``--concurrency 0`` would
    start no request and wait for ever)."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


#: The files a run of the client or the timing runner writes beside its documents' own.
RUN_FILES = frozenset({"summary"})


def unique_names(names: list[str], taken: frozenset[str] = RUN_FILES) -> list[str]:
    """``names`` made unique for files in one folder: a name already used, or one
    of the run's own files (``summary.pdf`` would be written over by
    ``summary.json``), takes ``-2``, ``-3`` ... - two folders can each hold a
    ``policy.pdf``. Compared ignoring case, as Windows compares file names."""
    used = {name.lower() for name in taken}
    out = []
    for name in names:
        unique, number = name, 2
        while unique.lower() in used:
            unique, number = f"{name}-{number}", number + 1
        used.add(unique.lower())
        out.append(unique)
    return out


def tier_problem(tier: str | None, any_mineru: bool) -> str | None:
    """Why ``--tier`` cannot be used without ``--any-mineru``; None when it can.
    Checked before a command detaches on the pod: :class:`ServedOcr` refuses
    the same tier, but only once the run has started, unseen, in tmux."""
    from common.config import served_mineru

    pinned = served_mineru()["tier"]
    if tier is None or tier == pinned or any_mineru:
        return None
    return (f"--tier {tier} is not the pinned tier ({pinned}): the model is trained on the pinned one's "
            "text. Add --any-mineru to time it anyway.")


# --------------------------------------------------------------------------
# OCR, in the other environment
# --------------------------------------------------------------------------


class OcrWorker:
    """The OCR environment's MinerU as a child process, one PDF at a time."""

    def __init__(self, python: str, cwd: Path = ROOT) -> None:
        if not Path(python).exists():
            raise FileNotFoundError(
                f"no OCR environment at {python}: build it with `bash scripts/setup_pod.sh ocr`, "
                "or pass --ocr-python")
        # stderr is left to the server's own (the tmux log), where MinerU's progress goes.
        self.process = subprocess.Popen(
            [python, "-c", OCR_WORKER_COMMAND], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, pdf: Path, out: Path, *, force_ocr: bool = False) -> dict[str, Any]:
        from data_pipeline.ocr.ocr_worker import RESULT_PREFIX

        request = json.dumps({"pdf": str(pdf), "out": str(out), "force_ocr": force_ocr})
        with self.lock:
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(request + "\n")
            self.process.stdin.flush()
            for line in self.process.stdout:
                if line.startswith(RESULT_PREFIX):
                    return json.loads(line[len(RESULT_PREFIX):])
        raise RuntimeError(f"the OCR worker stopped (exit {self.process.poll()}); see the log above")

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.process.wait(timeout=30)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    name: str
    lob: str
    mode: str
    submitted: float = field(default_factory=time.time)
    status: str = "queued"          # queued -> running -> (fields_ready ->) done | failed
    #: Seconds from submission to the fields (fields_first: what the gate holds the job to).
    fields_s: float | None = None
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None

    def view(self) -> dict[str, Any]:
        """What a poll answers: never the extraction itself (``/result`` holds it)."""
        out: dict[str, Any] = {"job": self.id, "name": self.name, "lob": self.lob, "mode": self.mode,
                               "status": self.status}
        if self.status == "queued":
            out["waiting_s"] = round(time.time() - self.submitted, 1)
        if self.status == "running" and self.started is not None:
            out["running_s"] = round(time.time() - self.started, 1)
        if self.fields_s is not None:
            out["fields_s"] = self.fields_s
        if self.error:
            out["error"] = self.error
        if self.summary is not None:
            out["summary"] = self.summary
        return out


def write_whole(path: Path, text: str) -> None:
    """``text`` into ``path`` at once: a reader sees the old file or the new one,
    never half of one (a fields-first result is rewritten while it is served)."""
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    for _ in range(40):
        try:
            os.replace(part, path)
            return
        except PermissionError:                  # Windows: a reader holds the old file for a moment
            time.sleep(0.05)
    os.replace(part, path)


class ExtractService:
    """Runs posted PDFs through OCR and :func:`serving.pipeline.extract`, one at a time."""

    def __init__(self, model: Any, ocr: OcrCall, work_dir: Path, *, keep_files: bool = False,
                 start: bool = True, default_mode: str = "ocr_plus_image",
                 text_max_wait_s: float | None = None, unverified_fields: bool = False) -> None:
        from serving.fields_first import GpuTurns

        self.model = model
        self.ocr = ocr
        self.work_dir = work_dir
        self.keep_files = keep_files
        self.default_mode = default_mode
        #: Fields first serves fields unchecked (timing tests only).
        self.unverified_fields = unverified_fields
        self.jobs: dict[str, Job] = {}
        #: Held while ``jobs`` is changed or listed: request threads insert into it
        #: while others list it (a dict changed while iterated raises).
        self.jobs_lock = threading.Lock()
        self.queue: queue.Queue[Job] = queue.Queue()
        #: Fields-first jobs whose fields are out, waiting for MinerU's text.
        self.text_queue: queue.Queue[Job] = queue.Queue()
        self.turns = GpuTurns() if text_max_wait_s is None else GpuTurns(text_max_wait_s)
        self._fields: dict[str, Any] = {}
        if start:
            threading.Thread(target=self._worker, name="extract-jobs", daemon=True).start()
            threading.Thread(target=self._text_worker, name="full-text", daemon=True).start()

    def submit(self, pdf: bytes, *, name: str, lob: str, mode: str | None = None) -> Job:
        from serving.vllm_entrypoint import ServingError, request_lob

        mode = mode or self.default_mode
        if not pdf.startswith(b"%PDF"):
            raise RequestError("the body is not a PDF (it does not start with %PDF)")
        if mode not in MODES:
            raise RequestError(f"mode must be one of {list(MODES)}, got {mode!r}")
        if mode == FIELDS_FIRST and not hasattr(self.ocr, "ocr_text"):
            raise RequestError("fields_first needs the served MinerU: start the server with --mineru-url")
        if not lob:
            raise RequestError("lob is required (e.g. lob=personal_auto): a policy is read against "
                               "its line's schema")
        try:
            line = request_lob({"lob": lob})
        except ServingError as exc:
            raise RequestError(str(exc)) from exc
        job = Job(id=uuid.uuid4().hex[:12], name=Path(name or "document.pdf").name, lob=str(line), mode=mode)
        folder = self.work_dir / job.id
        folder.mkdir(parents=True)
        (folder / "document.pdf").write_bytes(pdf)
        with self.jobs_lock:
            self.jobs[job.id] = job
        self.queue.put(job)
        log.info("job %s queued: %d bytes, lob %s, mode %s", job.id, len(pdf), job.lob, mode)
        return job

    def result_path(self, job: Job) -> Path:
        return self.work_dir / job.id / "result.json"

    def job_list(self) -> list[Job]:
        """Every job, in the order posted: a copy, safe to read while jobs are posted."""
        with self.jobs_lock:
            return list(self.jobs.values())

    def job(self, job_id: str) -> Job | None:
        with self.jobs_lock:
            return self.jobs.get(job_id)

    def _worker(self) -> None:
        while True:
            self.run(self.queue.get())

    def _text_worker(self) -> None:
        while True:
            self.add_text(self.text_queue.get())

    def run(self, job: Job) -> None:
        job.status, job.started = "running", time.time()
        handed_on = False
        try:
            if job.mode == FIELDS_FIRST:
                job.summary = self._read_fields(job)
                if job.summary.get("fields_served"):
                    job.status, job.fields_s = "fields_ready", round(time.time() - job.submitted, 2)
                # Otherwise the fields could not be checked against their pages:
                # nothing is served until the full text has checked them.
                self.text_queue.put(job)
                handed_on = True
            else:
                job.summary = self._extract(job)
                job.status = "done"
        except Exception as exc:  # noqa: BLE001 - a failed job is reported, the server stays up
            job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
            log.exception("job %s failed", job.id)
        finally:
            if not handed_on:
                self._finish(job)

    def add_text(self, job: Job) -> None:
        """A fields-first job's second phase: MinerU's text, in the GPU's idle turns."""
        from serving.fields_first import add_full_text

        result, count, known = self._fields.pop(job.id)
        folder = self.work_dir / job.id
        status = "done"
        try:
            full = add_full_text((folder / "document.pdf").read_bytes(), result, name=job.name, lob=job.lob,
                                 reader=self.ocr, pages_dir=folder / "pages", page_count=count, turns=self.turns,
                                 known_texts=known)
            timings = {**(job.summary or {}).get("timings_s", {}), **full["timings_s"],
                       "total": round(time.time() - job.submitted, 2)}
            job.summary = {**(job.summary or {}), **self._outcome(result), "timings_s": timings, "ocr": full["ocr"]}
            self._write(job, result)
        except Exception as exc:  # noqa: BLE001 - checked fields stand; the full text is reported missing
            job.error = f"full text: {type(exc).__name__}: {exc}"
            log.exception("job %s: the full text failed", job.id)
            if not (job.summary or {}).get("fields_served"):
                status = "failed"                       # nothing checked was ever served
        finally:
            job.status = status
            self._finish(job)

    def _finish(self, job: Job) -> None:
        job.finished = time.time()
        if not self.keep_files:
            folder = self.work_dir / job.id
            (folder / "document.pdf").unlink(missing_ok=True)
            shutil.rmtree(folder / "pages", ignore_errors=True)

    def _read_fields(self, job: Job) -> dict[str, Any]:
        """A fields-first job's first phase: the model reads the page images."""
        from calibration.fit_calibration import CalibrationParams
        from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
        from serving.doc_type_classifier import StaticClassifier
        from serving.fields_first import read_fields

        folder = self.work_dir / job.id
        fields = read_fields(
            (folder / "document.pdf").read_bytes(), name=job.name, lob=job.lob, source_id=job.id, model=self.model,
            reader=self.ocr, pages_dir=folder / "pages", classifier=StaticClassifier("policy"),
            calibration=CalibrationParams(method="temperature", doc_type="policy", model_version=self.model.tag,
                                          temperature=1.0),
            turns=self.turns, verify=not self.unverified_fields)
        result, served, count = fields["result"], fields["served"], len(fields["images"])
        every_page, declarations_page = pages_to_read(None, count)
        summary = {
            "job": job.id, "document": job.name, "lob": job.lob, "mode": job.mode, "model": self.model.tag,
            "pages": count, "windows_planned": len(plan_windows(job.lob, every_page, declarations_page, None)),
            "model_calls": len(result.window_latencies_ms),
            "timings_s": {**fields["timings_s"], "queued": round((job.started or job.submitted) - job.submitted, 2)},
            "windows": window_summary(result.window_stats),
            # Served now only once held to the text of the pages they cite.
            "fields_served": served is not None,
            "fields_verified": fields["verified"],
            "cited_pages": len(fields["cited_pages"]),
            **self._outcome(served if served is not None else result),
        }
        job.summary = summary
        if served is not None:
            self._write(job, served)
        # The model's own answer, for the full text to hold to the whole document.
        self._fields[job.id] = (result, count, fields["cited_texts"])
        log.info("job %s: fields of %d pages in %.1f s (%s); the full text follows", job.id, count,
                 fields["timings_s"]["fields"],
                 f"checked on {len(fields['cited_pages'])} cited page(s)" if fields["verified"]
                 else "not served: their pages could not be read" if served is None else "UNVERIFIED")
        return summary

    @staticmethod
    def _outcome(result: Any) -> dict[str, Any]:
        return {
            "schema_valid": result.schema_valid,
            "validation_errors": list(result.validation_errors)[:20],
            "review_flags": len(result.review_flags),
            "review_flag_kinds": _flag_kinds(result.review_flags),
            "text_coverage": result.text_coverage,
            "overall_confidence": result.overall_confidence,
            "extracted": _extracted_counts(result.extraction),
        }

    def _write(self, job: Job, result: Any) -> None:
        write_whole(self.result_path(job), json.dumps(
            {**(job.summary or {}), "review_flags_all": result.review_flags,
             "window_stats": result.window_stats, "extraction": result.extraction},
            ensure_ascii=False, indent=2))

    def _extract(self, job: Job) -> dict[str, Any]:
        from calibration.fit_calibration import CalibrationParams
        from data_pipeline.dataset_builder.policy_windows import pages_to_read, plan_windows
        from serving.doc_type_classifier import StaticClassifier
        from serving.pipeline import ExtractionRequest, extract

        folder = self.work_dir / job.id
        pages_dir = folder / "pages"
        began = time.perf_counter()
        ocr = self.ocr(folder / "document.pdf", pages_dir)
        if not ocr.get("ok"):
            raise RuntimeError(f"OCR failed: {ocr.get('error')}")
        ocr_s = time.perf_counter() - began
        count = int(ocr["pages"])
        images = [str(pages_dir / f"page_{page:03d}.png") for page in range(1, count + 1)]
        texts: dict[int, str] = {}
        if job.mode != "image_only":
            texts = {page: (pages_dir / f"page_{page:03d}.md").read_text(encoding="utf-8")
                     for page in range(1, count + 1)}
        request = ExtractionRequest(
            source_id=job.id, image_paths=images, page_texts=texts, modality_mode=job.mode,
            known_doc_type="policy", known_lob=job.lob, source_file_name=job.name,
            ocr_meta={"modality": ocr.get("modality")},
        )
        # The windows the pipeline plans, by the function it plans them with.
        every_page, declarations_page = pages_to_read(texts or None, count)
        windows = len(plan_windows(job.lob, every_page, declarations_page, texts or None))

        read_from = time.perf_counter()
        result = extract(
            request, self.model, StaticClassifier("policy"),
            CalibrationParams(method="temperature", doc_type="policy", model_version=self.model.tag,
                              temperature=1.0),
            strict_schema=False,
        )
        done = time.perf_counter()
        summary = {
            "job": job.id, "document": job.name, "lob": job.lob, "mode": job.mode, "model": self.model.tag,
            "pages": count, "modality": ocr.get("modality"), "scanned_pages": ocr.get("scanned_pages"),
            "ocr_failed_pages": ocr.get("ocr_failed_pages") or [],
            "windows_planned": windows,
            "model_calls": len(result.window_latencies_ms),
            "timings_s": {
                "ocr": round(ocr_s, 2),
                "model": round(done - read_from, 2),
                "generation": round((result.latency_ms or 0.0) / 1000, 2),
                "calls": [round(ms / 1000, 2) for ms in result.window_latencies_ms],
                "total": round(done - began, 2),
                "queued": round((job.started or job.submitted) - job.submitted, 2),
                # A served MinerU's reading: the rendering and the OCR it ran beside.
                **({"ocr_detail": ocr["timings_s"]} if ocr.get("timings_s") else {}),
            },
            "windows": window_summary(result.window_stats),
            "schema_valid": result.schema_valid,
            "validation_errors": list(result.validation_errors)[:20],
            "review_flags": len(result.review_flags),
            "review_flag_kinds": _flag_kinds(result.review_flags),
            "text_coverage": result.text_coverage,
            "overall_confidence": result.overall_confidence,
            "extracted": _extracted_counts(result.extraction),
        }
        write_whole(self.result_path(job), json.dumps(
            {**summary, "review_flags_all": result.review_flags, "window_stats": result.window_stats,
             "extraction": result.extraction}, ensure_ascii=False, indent=2))
        log.info("job %s done: %d pages, %d windows, OCR %.1f s, model %.1f s",
                 job.id, count, windows, ocr_s, done - read_from)
        return summary


def window_summary(stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Where a document's model time went, from its windows' stats
    (``ExtractionResult.window_stats``)."""
    timed = [s for s in stats if s.get("seconds") is not None]
    answered = [s for s in stats if s.get("answer_tokens") is not None]
    slowest = max(timed, key=lambda s: s["seconds"]) if timed else None
    return {
        "read": sum(s["outcome"] == "read" for s in stats),
        "split": sum(s["outcome"] == "split" for s in stats),
        "failed": sum(s["outcome"] == "failed" for s in stats),
        "rounds": max((s["attempt"] for s in stats), default=0),
        "cut_off": sum(s.get("finish") in ("length", "max_tokens") for s in stats),
        "stopped_repeating": sum(s.get("finish") == "repetition" for s in stats),
        "prompt_tokens": sum(s.get("prompt_tokens") or 0 for s in stats),
        "answer_tokens": sum(s["answer_tokens"] for s in answered),
        "longest_answer_tokens": max((s["answer_tokens"] for s in answered), default=0),
        "slowest_window": ({k: slowest.get(k) for k in ("group", "pages", "attempt", "seconds",
                                                          "first_token_s", "answer_tokens", "finish")}
                           if slowest else None),
        # Long waits for a first token: windows queued, or put back and read again for want of memory.
        "first_token_s_max": max((s["first_token_s"] for s in stats if s.get("first_token_s") is not None),
                                 default=None),
        # What each job writes - what its answer limit should be set from.
        "by_group": _answers_by_group(answered),
    }


def _answers_by_group(answered: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Each job's answer lengths: windows, median, 95th percentile and longest."""
    out: dict[str, dict[str, Any]] = {}
    for group in sorted({s["group"] for s in answered}):
        tokens = sorted(s["answer_tokens"] for s in answered if s["group"] == group)
        out[group] = {"windows": len(tokens), "median": tokens[len(tokens) // 2],
                      "p95": tokens[min(len(tokens) - 1, round(0.95 * (len(tokens) - 1)))], "max": tokens[-1]}
    return out


#: Flags about the document rather than one field (``window_failed:arrays:p7``,
#: ``lob:uncertain``, ``text_sections:no_page_text``): counted under their own name.
DOCUMENT_FLAGS = ("window_failed", "lob", "route", "text_sections", "schema")


def _flag_kinds(flags: list[str]) -> dict[str, int]:
    """Review flags by kind: a field's flag (``policy.policy_number:low_confidence``)
    under its kind, a document's under its name, and every failed window as one
    kind. Enough to see what a reviewer would face without reading the flags."""
    kinds: Counter[str] = Counter()
    for flag in flags:
        head = flag.split(":", 1)[0]
        if head == "window_failed":
            kinds[head] += 1
        elif head in DOCUMENT_FLAGS:
            kinds[flag] += 1
        else:
            kinds[flag.rsplit(":", 1)[-1]] += 1
    return dict(sorted(kinds.items()))


def _extracted_counts(extraction: dict[str, Any]) -> dict[str, int]:
    """How many rows each table of the served JSON holds, and how many sections of
    full text: a quick look at what came back without reading it."""
    counts = {name: sum(1 for row in value if row) for name, value in extraction.items()
              if isinstance(value, list)}
    if isinstance(extraction.get("text_sections"), dict):
        counts["text_sections"] = len(extraction["text_sections"])
    return {name: count for name, count in sorted(counts.items()) if count}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def handler_for(service: ExtractService, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "extract-server"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the stdlib's name
            # The path only: a query names the document.
            log.info("%s %s", self.command, urlparse(self.path).path)

        def _send(self, status: int, body: Any) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self) -> bool:
            sent = self.headers.get("Authorization", "")
            # As bytes: compare_digest raises on a str holding anything but ASCII
            # (a header is read as Latin-1), and the connection was dropped unanswered.
            if sent.startswith("Bearer ") and hmac.compare_digest(
                    sent[7:].strip().encode("utf-8", "surrogatepass"), token.encode("utf-8")):
                return True
            self._send(401, {"error": "send Authorization: Bearer <token> (see token.txt on the pod)"})
            return False

        def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            if parts == ["health"]:
                jobs = service.job_list()
                self._send(200, {"status": "ready", "model": service.model.tag,
                                 # The largest PDF taken: the client checks before it uploads.
                                 "max_pdf_bytes": MAX_PDF_BYTES,
                                 "jobs": {s: sum(1 for j in jobs if j.status == s)
                                          for s in ("queued", "running", "fields_ready", "done", "failed")}})
                return
            if not self._authorised():
                return
            job = service.job(parts[1]) if len(parts) in (2, 3) and parts[0] == "jobs" else None
            if parts == ["jobs"]:
                self._send(200, [each.view() for each in service.job_list()])
            elif job is not None:
                if len(parts) == 2:
                    self._send(200, job.view())
                elif parts[2] == "result" and job.status in ("fields_ready", "done"):
                    # fields_ready: the fields, with the full-text tier still to come.
                    self._send(200, json.loads(service.result_path(job).read_text(encoding="utf-8")))
                elif parts[2] == "result":
                    self._send(409, {"error": f"job {job.id} is {job.status}", **job.view()})
                else:
                    self._send(404, {"error": "no such path"})
            else:
                self._send(404, {"error": "no such job or path"})

        def do_POST(self) -> None:  # noqa: N802 - the stdlib's name
            if urlparse(self.path).path.rstrip("/") != "/jobs":
                self._send(404, {"error": "POST /jobs with the PDF as the body"})
                return
            if not self._authorised():
                return
            sent = (self.headers.get("Content-Length") or "0").strip()
            if not (sent.isascii() and sent.isdigit()):     # "abc", "-5": not a length at all
                self._send(400, {"error": f"Content-Length must be the body's size in bytes, got {sent!r}"})
                return
            length = int(sent)
            if not 0 < length <= MAX_PDF_BYTES:
                self._send(413 if length else 411, {"error": f"send the PDF as the body, up to "
                                                             f"{MAX_PDF_BYTES // 2**20} MB"})
                return
            query = {k: v[-1] for k, v in parse_qs(urlparse(self.path).query).items()}
            body = self.rfile.read(length)
            try:
                job = service.submit(body, name=query.get("name", "document.pdf"), lob=query.get("lob", ""),
                                     mode=query.get("mode", service.default_mode))
            except RequestError as exc:
                self._send(400, {"error": str(exc)})
                return
            self._send(202, {**job.view(), "poll": f"/jobs/{job.id}", "result": f"/jobs/{job.id}/result"})

    return Handler


# --------------------------------------------------------------------------
# Start-up
# --------------------------------------------------------------------------


def load_model(gpu_memory_utilization: float, adapter: str | None = None) -> Any:
    """The base model (or base + adapter) in vLLM, warmed: its engine starts here,
    not inside the first document's timings."""
    import dataclasses

    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import generate
    from inference_core.model_runner import load_model as load
    from inference_core.runner_config import load_runner_config

    config = dataclasses.replace(load_runner_config(), gpu_memory_utilization=gpu_memory_utilization)
    # The base resolves to the local weights; nothing is read from Blob.
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = load("base", client, config=config, adapter=adapter)
    started = time.perf_counter()
    generate(model, [{"role": "user", "content": [{"type": "text", "text": "Reply with OK."}]}],
             max_new_tokens=4, want_logprobs=False)
    log.info("vLLM ready (%s) in %.0f s", model.tag, time.perf_counter() - started)
    return model


def served_model(url: str, token: str | None, *, served: str | None = None, images: str = "inline",
                 concurrency: int = 128) -> Any:
    """The model a ``vllm serve`` on ``url`` serves, as the pipeline loads one: each
    window its own request, taken as it is answered. A LoRA adapter is served by
    vLLM itself (``--lora-modules name=path``) and named with ``served``."""
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from inference_core.model_runner import load_model as load
    from inference_core.openai_backend import OpenAIChatBackend

    backend = OpenAIChatBackend(url, token, model=served, max_concurrency=concurrency, images=images)
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = load("base", client, backend_impl=backend, label=backend.model)
    log.info("served model %s at %s (images %s, up to %d windows at once)", backend.model, url, images,
             concurrency)
    return model


def warm_ocr(ocr: OcrCall, work_dir: Path) -> None:
    """One scanned-looking page through MinerU, so its models load before the first job."""
    import pymupdf

    folder = work_dir / "_warmup"
    folder.mkdir(parents=True, exist_ok=True)
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Warm-up page. Policy Number WU-1.", fontsize=14)
    pdf.save(folder / "warmup.pdf")
    started = time.perf_counter()
    result = ocr(folder / "warmup.pdf", folder / "pages", force_ocr=True)
    if not result.get("ok"):
        raise RuntimeError(f"MinerU could not read a warm-up page: {result.get('error')}")
    shutil.rmtree(folder, ignore_errors=True)
    log.info("MinerU ready in %.0f s", time.perf_counter() - started)


def server_token(work_dir: Path) -> str:
    """``$EXTRACT_SERVER_TOKEN``, else a new one; written to ``token.txt`` either way."""
    token = os.environ.get(TOKEN_ENV) or secrets.token_urlsafe(24)
    path = work_dir / "token.txt"
    path.write_text(token + "\n", encoding="utf-8")
    path.chmod(0o600)
    return token


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000, help="the port the pod exposes over HTTP")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--work-dir", type=Path, default=Path("/workspace/extract_server"))
    parser.add_argument("--ocr-python", default="/workspace/venv-ocr/bin/python",
                        help="the OCR environment's python (MinerU)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75,
                        help="vLLM's share of the GPU; MinerU runs in what is left")
    parser.add_argument("--adapter", default=None, help="a LoRA adapter applied to the base model")
    parser.add_argument("--keep-files", action="store_true",
                        help="keep each job's PDF and page images (the result JSON is always kept)")
    served = parser.add_argument_group("the pod's own servers, instead of loading vLLM and MinerU here")
    served.add_argument("--model-url", default=None, help="a vllm serve on this pod, e.g. http://localhost:8000")
    served.add_argument("--served-model", default=None,
                        help="the name to ask it for, when it serves more than one (a LoRA adapter)")
    served.add_argument("--mineru-url", default=None, help="a MinerU 4 API on this pod, e.g. http://localhost:8001")
    served.add_argument("--remote-token", default=os.environ.get(REMOTE_TOKEN_ENV),
                        help=f"their bearer token (default ${REMOTE_TOKEN_ENV})")
    served.add_argument("--images", default="inline", choices=["inline", "file"],
                        help="page images in each request, or by path (vLLM needs --allowed-local-media-path)")
    served.add_argument("--concurrency", type=positive_int, default=128, help="windows in flight at most")
    served.add_argument("--tier", default=None, help="the MinerU tier (default the pinned one, "
                        "configs/shared/ocr.yaml)")
    served.add_argument("--any-mineru", action="store_true",
                        help="accept a MinerU version or tier other than the pinned one (timing only)")
    served.add_argument("--ocr-parts", type=int, default=1,
                        help="a PDF to MinerU as this many runs of pages (fields first: the turns MinerU waits for)")
    served.add_argument("--render-workers", type=int, default=None, help="processes rendering pages at once")
    served.add_argument("--run-pages", type=int, default=20,
                        help="fields first: pages per MinerU run - a document arriving while another's text is "
                             "read waits for at most this many pages' reading")
    served.add_argument("--text-max-wait", type=float, default=600.0,
                        help="fields first: seconds a MinerU run waits for its turn before it goes anyway")
    served.add_argument("--unverified-fields", action="store_true",
                        help="fields first: serve the fields before any page text has checked them "
                             "(flagged fields:unverified) - timing tests only")
    served.add_argument("--fields-first", action="store_true",
                        help="jobs read fields first by default: the model reads the page images, then MinerU's "
                             "text adds the full-text tier in the GPU's idle turns (one GPU for both)")
    args = parser.parse_args(argv)
    if args.fields_first and not args.mineru_url:
        parser.error("--fields-first needs the served MinerU (--mineru-url): its text comes in a second phase")
    if args.model_url and args.adapter:
        parser.error("with --model-url the adapter is vLLM's to serve: start it with --lora-modules "
                     "<name>=<path> and pass --served-model <name>")
    # Before the detach below: refused after it, in tmux, nobody would see why.
    if problem := tier_problem(args.tier, args.any_mineru):
        parser.error(problem)
    from common.transport import InsecureUrlError, assert_encrypted

    try:
        if args.model_url:
            assert_encrypted(args.model_url, "the model server")
        if args.mineru_url:
            assert_encrypted(args.mineru_url, "the MinerU server")
    except InsecureUrlError as exc:
        parser.error(str(exc))

    from orchestration.detach import POD_ENV, detach_module_if_needed

    if detach_module_if_needed("testing.extract_server", argv, "extract-server"):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    token = server_token(args.work_dir)
    if args.model_url:
        model = served_model(args.model_url, args.remote_token, served=args.served_model, images=args.images,
                             concurrency=args.concurrency)
    else:
        model = load_model(args.gpu_memory_utilization, args.adapter)    # vLLM first: MinerU takes the rest
    if args.mineru_url:
        from common.config import served_mineru
        from data_pipeline.ocr.served_ocr import ServedOcr

        pin = served_mineru()
        ocr: Any = ServedOcr(args.mineru_url, args.remote_token, tier=args.tier or pin["tier"],
                             ocr_parts=args.ocr_parts, render_workers=args.render_workers,
                             run_pages=args.run_pages,
                             expected_version=None if args.any_mineru else pin["version"])
    else:
        ocr = OcrWorker(args.ocr_python)
    warm_ocr(ocr, args.work_dir)
    service = ExtractService(model, ocr, args.work_dir, keep_files=args.keep_files,
                             default_mode="fields_first" if args.fields_first else "ocr_plus_image",
                             text_max_wait_s=args.text_max_wait, unverified_fields=args.unverified_fields)

    server = ThreadingHTTPServer((args.host, args.port), handler_for(service, token))
    pod = os.environ.get(POD_ENV)
    url = f"https://{pod}-{args.port}.proxy.runpod.net" if pod else f"http://localhost:{args.port}"
    print(f"\nextract server ready: {url}\ntoken: {token}  (also in {args.work_dir / 'token.txt'})\n",
          flush=True)
    try:
        server.serve_forever()
    finally:
        if hasattr(ocr, "close"):
            ocr.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
