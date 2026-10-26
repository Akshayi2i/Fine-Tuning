"""The worker: jobs from the queue, read fields first on the one GPU, state and results to Blob.

    python -m azure_service.worker            # in the worker container (azure_service/Dockerfile)

Two threads share one GPU's turns (:class:`serving.fields_first.GpuTurns`), as
the testing extraction server's do (``testing.extract_server.ExtractService``):

* **fields** (:meth:`Worker.poll_once`): takes the next job, renders its pages,
  has the model read their images and checks the fields against the pages they
  cite (:func:`serving.fields_first.read_fields`). Checked fields are written as
  the result and the job is ``fields_ready``; the model's answer is kept as a
  checkpoint and the job handed to the second thread;
* **full text** (:meth:`Worker.text_once`): MinerU reads the remaining pages in
  the GPU's idle turns, runs of ``RUN_PAGES`` pages, while no document's fields
  are read (:func:`serving.fields_first.add_full_text`). The final result
  replaces the fields and the job is ``done``.

**Job lifecycle.** ``queued`` (the API) -> ``running`` -> ``fields_ready`` ->
``done``, or ``failed``. Fields that could not be checked (no cited page had
text) are never served: such a job goes from ``running`` to ``done``.

**The queue keeps a job alive** (:mod:`azure_service.job_queue`). A received job
stays hidden while the worker renews its lease, and its message is deleted only
once the job is ``done`` or ``failed``:

* a failure worth retrying (the model or MinerU unreachable, a Blob error)
  puts the job back, visible again after ``RETRY_DELAY_S``;
* after ``MAX_ATTEMPTS`` deliveries the job is given up: a poison job - one
  that kills the worker every time - never loops;
* a job the worker can never read (a tenant or line it does not serve) fails at once.

**A restart mid-job loses nothing.** The worker that dies stops renewing; the
message comes back after ``VISIBILITY_TIMEOUT_S`` with one more delivery
counted. A job whose phase-1 checkpoint exists (``fields_ready``, or
``running`` with fields that could not be served) **resumes phase 2**: MinerU
reads its pages and the final result is written, with no model time spent
again and the served fields left as they were. A job with no checkpoint (the
worker died during phase 1) is **read again from the start**. A job whose
fields were served but whose full text fails for good ends ``done`` with
``full_text: false``: its checked fields stand (as ExtractService does).

On SIGTERM (a scale-in, a new revision) the worker takes no new job and makes
its jobs visible again at once, for the next replica.

**Retention for review is off by default** (:mod:`azure_service.retention`,
decision 2026-10-26). Only for a tenant named in ``REVIEW_RETENTION_TENANTS``, a
job that ends ``done`` flagged, below the confidence threshold or in the
sample has its PDF, result and a manifest copied to ``REVIEW_CONTAINER`` before
its PDF is deleted. A failed copy is logged and never fails the job; nothing
of it shows in the job's status or result.

**Nothing of a document is logged**: job ids, sizes, counts and timings only.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import queue
import shutil
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

from azure_service.job_queue import JobQueue, LeaseLostError, QueueError, QueueMessage
from azure_service.runtime import (
    JobRefused,
    ServingRuntime,
    identity_calibration,
)
from azure_service.store import FINAL, JobStore

log = logging.getLogger(__name__)


class Lease:
    """A received message while its job runs: renewed by the lease keeper."""

    def __init__(self, message: QueueMessage) -> None:
        self.message = message
        self.lost = False
        self.lock = threading.Lock()


@dataclass
class _Phase2:
    record: dict[str, Any]
    lease: Lease


def restore_result(answer: dict[str, Any]) -> Any:
    """The model's answer as phase 1 left it, from its checkpoint."""
    from serving.pipeline import ExtractionResult

    names = {f.name for f in dataclasses.fields(ExtractionResult)}
    return ExtractionResult(**{key: value for key, value in answer.items() if key in names})


def _scrub(message: str, record: dict[str, Any]) -> str:
    """An error message without the document's file name (a caller's name may name a person)."""
    name = str(record.get("document") or "")
    text = message.replace(name, "<document>") if len(name) > 3 else message
    return text[:500]


class Worker:
    """Reads queued jobs fields first on one GPU; see the module's docstring."""

    def __init__(self, store: JobStore, queue_: JobQueue, runtime: ServingRuntime, settings: Any, *,
                 webhooks: Any, links: Any = None, clock: Any = time.time, retention: Any = None) -> None:
        from serving.fields_first import GpuTurns

        self.store = store
        self.queue = queue_
        self.runtime = runtime
        self.settings = settings
        self.webhooks = webhooks
        #: Page images published for the model to fetch (PAGE_IMAGES=link).
        self.links = links
        #: :class:`azure_service.retention.ReviewRetention`, or None (the default: off).
        self.retention = retention
        self.clock = clock
        self.turns = GpuTurns(settings.text_max_wait_s)
        self._text: queue.Queue[_Phase2] = queue.Queue()
        self._leases: dict[str, Lease] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Leases
    # ------------------------------------------------------------------

    def _hold(self, job_id: str, lease: Lease) -> None:
        with self._lock:
            previous = self._leases.get(job_id)
            if previous is not None:
                previous.lost = True                 # its receipt is stale now: this delivery owns the job
            self._leases[job_id] = lease

    def _drop(self, job_id: str, lease: Lease) -> None:
        with self._lock:
            if self._leases.get(job_id) is lease:
                del self._leases[job_id]

    def held(self) -> list[str]:
        with self._lock:
            return sorted(self._leases)

    def renew_leases(self) -> None:
        """Keep every running job hidden for another visibility timeout."""
        with self._lock:
            leases = list(self._leases.items())
        for job_id, lease in leases:
            with lease.lock:
                if lease.lost:
                    continue
                try:
                    lease.message = self.queue.extend(lease.message, self.settings.visibility_timeout_s)
                except LeaseLostError:
                    lease.lost = True
                    log.warning("job %s: its lease was lost; another delivery owns it now", job_id)
                except QueueError as exc:
                    log.warning("job %s: its lease could not be renewed (%s); trying again", job_id, exc)

    def _release(self, job_id: str, lease: Lease, delay_s: int) -> None:
        """The job back on the queue, visible again after ``delay_s``."""
        with lease.lock:
            if not lease.lost:
                try:
                    lease.message = self.queue.extend(lease.message, delay_s)
                except QueueError as exc:
                    log.warning("job %s: could not put it back (%s); it returns when its lease runs out",
                                job_id, exc)
            lease.lost = True                        # nothing more is written for this delivery
        self._drop(job_id, lease)

    def _delete(self, job_id: str, lease: Lease) -> None:
        with lease.lock:
            try:
                self.queue.delete(lease.message)
            except QueueError as exc:
                # It comes back once its lease runs out, finds the job final and is deleted then.
                log.warning("job %s: its message could not be deleted (%s)", job_id, exc)
            lease.lost = True
        self._drop(job_id, lease)

    @staticmethod
    def _owned(lease: Lease) -> None:
        if lease.lost:
            raise LeaseLostError("the job's lease was lost")

    def _assert_current(self, record: dict[str, Any], lease: Lease) -> None:
        """Raise :class:`LeaseLostError` when a later delivery took the job, or it
        ended meanwhile: a worker whose lease ran out unnoticed (a stalled renewal)
        must not put an older state or result over a newer one."""
        self._owned(lease)
        current = self.store.read_status(record["tenant"], record["job_id"])
        if current is not None and (current.get("status") in FINAL
                                    or int(current.get("attempts") or 0) > int(record.get("attempts") or 0)):
            lease.lost = True
            raise LeaseLostError("the job was taken by a later delivery, or ended")

    def _write_status(self, record: dict[str, Any], lease: Lease) -> None:
        self._assert_current(record, lease)
        self.store.write_status(record)

    # ------------------------------------------------------------------
    # Phase 1: the fields
    # ------------------------------------------------------------------

    def poll_once(self) -> bool:
        """Take one job from the queue and read its fields. False when none was taken."""
        if self._stop.is_set() or self._text.qsize() >= self.settings.max_text_backlog:
            return False
        message = self.queue.receive(self.settings.visibility_timeout_s)
        if message is None:
            return False
        self._take(message)
        return True

    def _take(self, message: QueueMessage) -> None:
        try:
            ref = message.job()
        except ValueError:
            log.error("message %s names no job; deleted", message.id)
            self._delete(message.id, Lease(message))
            return
        try:
            record = self.store.read_status(ref["tenant"], ref["job_id"])
        except Exception as exc:  # noqa: BLE001 - Blob unreachable: the message returns after its timeout
            log.warning("job %s: its status could not be read (%s); it comes back later", ref["job_id"],
                        type(exc).__name__)
            return
        if record is None or record.get("status") in FINAL:
            # Deleted since, or a second delivery of a job already finished.
            self._delete(ref["job_id"], Lease(message))
            if record is not None and record.get("status") == "done":
                self._retain(record)         # a worker that died between `done` and its retention
            return
        job_id = record["job_id"]
        lease = Lease(message)
        self._hold(job_id, lease)
        record["attempts"] = message.dequeue_count
        if message.dequeue_count > self.settings.max_attempts:
            self._give_up(record, lease, f"delivered {message.dequeue_count} times without finishing: "
                                         f"given up after {self.settings.max_attempts} attempts")
            return
        try:
            checkpoint = self.store.read_checkpoint(record["tenant"], job_id)
            if checkpoint is not None:
                self._write_status(record, lease)        # this delivery's claim on the job
        except LeaseLostError:
            self._drop(job_id, lease)
            return
        except Exception as exc:  # noqa: BLE001 - Blob unreachable: tried again later
            self._failed(record, lease, exc, phase=1)
            return
        if checkpoint is not None:
            log.info("job %s: phase 1 was done before a restart; its full text resumes (delivery %d)",
                     job_id, message.dequeue_count)
            self._text.put(_Phase2(record, lease))
            return
        self._read_fields(record, lease)

    def _read_fields(self, record: dict[str, Any], lease: Lease) -> None:
        from serving.doc_type_classifier import StaticClassifier
        from serving.fields_first import read_fields

        job_id, tenant = record["job_id"], record["tenant"]
        folder = self.settings.work_dir / job_id
        published: list[str] = []
        try:
            reading = self.runtime.tenant(tenant).reading(record["lob"])
            already_served = record["status"] == "fields_ready"
            now = self.clock()
            record.update(status="fields_ready" if already_served else "running", error=None,
                          started=record.get("started") or now, model=reading.model.tag,
                          release=reading.release_id, route=reading.route,
                          calibration="release" if reading.calibrators is not None else "identity")
            record.setdefault("timings_s", {})["queued"] = round(record["started"] - record["submitted"], 2)
            self._write_status(record, lease)
            pdf = self.store.read_pdf(tenant, job_id)

            def publish(paths: list[str]) -> dict[str, Any]:
                published.extend(paths)
                return self.links.publish(paths, job_id)

            fields = read_fields(
                pdf, name=record["document"], lob=record["lob"], source_id=job_id, model=reading.model,
                reader=self.runtime.reader, pages_dir=folder / "pages", classifier=StaticClassifier("policy"),
                calibration=identity_calibration(reading.model),
                publish_pages=publish if self.links is not None else None, turns=self.turns, verify=True,
                extract_options=reading.extract_options(self.runtime.tuning))
            result, served = fields["result"], fields["served"]
            count = len(fields["images"])
            now = self.clock()
            timings = {key: value for key, value in fields["timings_s"].items() if value is not None}
            record["timings_s"].update(timings)
            record.update(pages=count, fields_served=served is not None or already_served,
                          fields_verified=bool(fields["verified"]), cited_pages=len(fields["cited_pages"]),
                          model_calls=len(result.window_latencies_ms))
            self._assert_current(record, lease)
            # The checkpoint first: a restart from here on resumes phase 2.
            self.store.write_checkpoint(tenant, job_id, {
                # Not the calibrators: phase 2 takes them from the release (below), never a repr.
                "answer": json.loads(json.dumps(dataclasses.asdict(dataclasses.replace(result, calibrated_with=None)),
                                                default=str)),
                "page_count": count,
                "cited_texts": {str(page): text for page, text in fields["cited_texts"].items()},
            })
            if served is not None and not already_served:
                record["timings_s"]["fields_ready"] = round(now - record["submitted"], 2)
                self.store.write_result(tenant, job_id, self._result({**record, "status": "fields_ready"}, served,
                                                                     final=False))
                record.update(status="fields_ready", fields_ready_at=now)
                self._write_status(record, lease)
                self.webhooks.notify(record, "fields_ready")
            else:
                self._write_status(record, lease)
            log.info("job %s: fields of %d page(s) in %.1f s (%s); the full text follows", job_id, count,
                     timings.get("fields", 0.0),
                     "served" if served is not None else "not served: no cited page had text")
        except LeaseLostError:
            log.warning("job %s: lease lost during its fields; left to the delivery that owns it", job_id)
            self._drop(job_id, lease)
            self._clean_local(job_id)
            return
        except JobRefused as exc:
            self._failed(record, lease, exc, phase=1, retry=False)
            return
        except Exception as exc:  # noqa: BLE001 - a failed job is reported; the worker stays up
            self._failed(record, lease, exc, phase=1)
            return
        finally:
            self._unpublish(published)
            shutil.rmtree(folder / "pages", ignore_errors=True)
        self._text.put(_Phase2(record, lease))

    # ------------------------------------------------------------------
    # Phase 2: the full text
    # ------------------------------------------------------------------

    def text_once(self, timeout_s: float | None = None) -> bool:
        """Add the full text to one job whose fields are out. False when none waits."""
        try:
            item = self._text.get(timeout=timeout_s) if timeout_s else self._text.get_nowait()
        except queue.Empty:
            return False
        self._full_text(item.record, item.lease)
        return True

    def _full_text(self, record: dict[str, Any], lease: Lease) -> None:
        from serving.fields_first import add_full_text

        job_id, tenant = record["job_id"], record["tenant"]
        folder = self.settings.work_dir / job_id
        try:
            self._owned(lease)
            checkpoint = self.store.read_checkpoint(tenant, job_id)
            if checkpoint is None:
                raise RuntimeError("the phase-1 checkpoint is missing")
            result = restore_result(checkpoint["answer"])
            known = {int(page): text for page, text in (checkpoint.get("cited_texts") or {}).items()}
            pdf = self.store.read_pdf(tenant, job_id)
            # The release's calibrators and thresholds, as phase 1 scored with: the
            # result restored from its checkpoint does not carry them, and without
            # them its confidence is not scored again on the full text.
            reading = self.runtime.tenant(tenant).reading(record["lob"])
            full = add_full_text(pdf, result, name=record["document"], lob=record["lob"],
                                 reader=self.runtime.reader, pages_dir=folder / "pages",
                                 page_count=int(checkpoint["page_count"]), turns=self.turns, known_texts=known,
                                 calibrators=reading.calibrators, thresholds=reading.thresholds)
            now = self.clock()
            record.setdefault("timings_s", {}).update(full["timings_s"])
            record["timings_s"]["total"] = round(now - record["submitted"], 2)
            ocr = full.get("ocr") or {}
            record["ocr"] = {key: ocr.get(key) for key in ("version", "tier", "ocr_mode") if ocr.get(key)}
            record["full_text"] = True
            self._assert_current(record, lease)
            self.store.write_result(tenant, job_id, self._result({**record, "status": "done"}, result, final=True))
            record.update(status="done", finished=now, error=None)
            self._write_status(record, lease)
            log.info("job %s done in %.1f s", job_id, record["timings_s"]["total"])
        except LeaseLostError:
            log.warning("job %s: lease lost during its full text; left to the delivery that owns it", job_id)
            self._drop(job_id, lease)
            self._clean_local(job_id)
            return
        except Exception as exc:  # noqa: BLE001 - reported on the job
            self._failed(record, lease, exc, phase=2)
            return
        self._finish(record, lease, "done")

    # ------------------------------------------------------------------
    # Endings
    # ------------------------------------------------------------------

    def _result(self, record: dict[str, Any], result: Any, *, final: bool) -> dict[str, Any]:
        """The served result: the output contract (``ExtractionResult.as_dict``) and how it was read."""
        return {
            "job_id": record["job_id"], "status": record["status"], "final": final,
            "full_text": bool(record.get("full_text")), "model": record.get("model"),
            "release": record.get("release"), "route": record.get("route"),
            "calibration": record.get("calibration"), "lob": record.get("lob"), "pages": record.get("pages"),
            "fields_verified": record.get("fields_verified"), "timings_s": dict(record.get("timings_s") or {}),
            "output": result.as_dict(),
        }

    def _failed(self, record: dict[str, Any], lease: Lease, exc: BaseException, *, phase: int,
                retry: bool = True) -> None:
        """A phase that raised: the job back on the queue while attempts remain, else ended."""
        job_id = record["job_id"]
        reason = _scrub(f"{'full text: ' if phase == 2 else ''}{type(exc).__name__}: {exc}", record)
        attempts = lease.message.dequeue_count
        self._clean_local(job_id)
        if lease.lost:
            self._drop(job_id, lease)
            return
        if retry and attempts < self.settings.max_attempts:
            log.warning("job %s: attempt %d of %d failed (%s); retried in %d s", job_id, attempts,
                        self.settings.max_attempts, reason, self.settings.retry_delay_s)
            if phase == 1 and record.get("status") == "running":
                record["status"] = "queued"
            record.update(error=reason, retry_at=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.clock() + self.settings.retry_delay_s)))
            try:
                self._write_status(record, lease)
            except Exception as write_error:  # noqa: BLE001 - the queue still brings it back
                log.warning("job %s: its status could not be written (%s)", job_id, type(write_error).__name__)
            self._release(job_id, lease, self.settings.retry_delay_s)
            return
        log.error("job %s: %s (attempt %d; not retried)", job_id, reason, attempts)
        self._give_up(record, lease, reason)

    def _give_up(self, record: dict[str, Any], lease: Lease, reason: str) -> None:
        """End the job for good: ``done`` without the full text when its checked
        fields were served, ``failed`` otherwise."""
        job_id, tenant = record["job_id"], record["tenant"]
        now = self.clock()
        record.update(error=reason, finished=now)
        record.pop("retry_at", None)
        status = "failed"
        try:
            if record.get("fields_served") and self.store.client.exists(self.store.result_key(tenant, job_id)):
                served = self.store.read_result(tenant, job_id)
                served.update(status="done", final=True, full_text=False, full_text_error=reason)
                record.update(status="done", full_text=False)
                record.setdefault("timings_s", {})["total"] = round(now - record["submitted"], 2)
                self._assert_current(record, lease)
                self.store.write_result(tenant, job_id, served)
                status = "done"
            else:
                record["status"] = "failed"
            self._write_status(record, lease)
        except LeaseLostError:
            self._drop(job_id, lease)
            return
        except Exception as exc:  # noqa: BLE001 - the message stays; the next delivery ends it again
            log.error("job %s: could not record its end (%s)", job_id, type(exc).__name__)
            self._release(job_id, lease, self.settings.retry_delay_s)
            return
        self._finish(record, lease, status)

    def _finish(self, record: dict[str, Any], lease: Lease, event: str) -> None:
        job_id = record["job_id"]
        self._delete(job_id, lease)
        if event == "done":
            self._retain(record)                 # before the PDF is deleted
        if not self.settings.keep_files:
            self.store.delete_transient(record["tenant"], job_id)
        self._clean_local(job_id)
        self.webhooks.notify(record, event)

    def _retain(self, record: dict[str, Any]) -> None:
        """Keep a finished job for review when its tenant is opted in and its result
        calls for it (:mod:`azure_service.retention`; off by default). It never
        fails or delays the job's end beyond the copy, and logs no content."""
        if self.retention is None:
            return
        try:
            self.retention.keep(self.store, record)
        except Exception as exc:  # noqa: BLE001 - the job is done whatever happens here
            log.warning("job %s: not retained for review (%s)", record.get("job_id"), type(exc).__name__)

    def _clean_local(self, job_id: str) -> None:
        if not self.settings.keep_files:
            shutil.rmtree(self.settings.work_dir / job_id, ignore_errors=True)

    def _unpublish(self, paths: list[str]) -> None:
        if self.links is not None and paths and not self.settings.keep_files:
            self.links.discard(paths)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run_forever(self, *, idle_s: float = 2.0, max_idle_s: float = 10.0) -> None:
        """The fields loop here; the full-text loop and the lease keeper on their own threads."""
        threading.Thread(target=self._text_loop, name="full-text", daemon=True).start()
        threading.Thread(target=self._lease_loop, name="lease-keeper", daemon=True).start()
        wait = idle_s
        while not self._stop.is_set():
            try:
                took = self.poll_once()
            except QueueError as exc:
                log.warning("the queue could not be read (%s)", exc)
                took = False
            except Exception:  # noqa: BLE001 - the loop outlives any one job
                log.exception("taking a job failed")
                took = False
            # An empty queue is polled less often: each receive is a storage transaction.
            wait = idle_s if took else min(max_idle_s, wait * 1.5)
            if not took:
                self._stop.wait(wait)

    def _text_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.text_once(timeout_s=1.0)
            except Exception:  # noqa: BLE001 - the loop outlives any one job
                log.exception("a full text failed outside its job")

    def _lease_loop(self) -> None:
        every = max(5.0, self.settings.visibility_timeout_s / 3)
        while not self._stop.wait(every):
            self.renew_leases()

    def shutdown(self) -> None:
        """Take no new job, and let every held job be delivered again at once."""
        self._stop.set()
        with self._lock:
            leases = list(self._leases.items())
        for job_id, lease in leases:
            self._release(job_id, lease, 0)
        log.info("stopping: %d job(s) handed back to the queue", len(leases))


def main(argv: list[str] | None = None) -> int:
    from azure_service.retention import review_retention
    from azure_service.runtime import load_runtime
    from azure_service.settings import Settings, SettingsError
    from azure_service.store import blob_backend, job_queue, jobs_client, registry_client
    from azure_service.webhooks import WebhookSender
    from serving.vllm_entrypoint import ColdStartError

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        settings.check_worker()
    except SettingsError as exc:
        log.error("not configured: %s", exc)
        return 2
    log.info("worker settings: %s", settings.describe())
    backend = blob_backend(settings)
    jobs = jobs_client(settings, backend)
    # Retention for review: None unless REVIEW_RETENTION_TENANTS names a tenant (off by default).
    retention = review_retention(settings, backend)
    if retention is not None:
        try:
            retention.check()
        except Exception as exc:  # noqa: BLE001 - extraction runs on; nothing is retained
            log.error("retention for review is OFF: the review container %s cannot be written (%s)",
                      settings.review_container, type(exc).__name__)
            retention = None
    links = None
    if settings.images == "link":
        import socket

        from inference_core.page_links import LINK_PREFIX, PageLinks

        links = PageLinks(jobs, f"{LINK_PREFIX}/{socket.gethostname()}", read_token=settings.page_link_sas)
        links.check()
    try:
        runtime = load_runtime(settings, registry_client(settings), links=links)
    except ColdStartError as exc:
        log.error("cold start refused: %s", exc)
        return 3
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    worker = Worker(JobStore(jobs), job_queue(settings), runtime, settings, webhooks=WebhookSender(settings),
                    links=links, retention=retention)

    def stop(signum: int, _frame: Any) -> None:
        log.info("signal %d received", signum)
        worker.shutdown()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    worker.run_forever()
    worker.webhooks.drain(timeout_s=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
