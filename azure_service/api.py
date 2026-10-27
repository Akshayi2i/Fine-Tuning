"""The HTTPS API: a PDF in, a job id back; the job's state and its result polled.

    python -m azure_service.api                # in the api container (azure_service/Dockerfile)

    POST /v1/extractions?lob=personal_auto[&webhook_url=https://...][&name=a.pdf]
         body: the PDF (application/pdf), or multipart/form-data with `file`
         (and `lob`, `webhook_url`, `name`, `password` as form fields)
         [X-PDF-Password: <the PDF's password>]                  -> 202 {"job_id", "status_url", "result_url"}
    GET  /v1/extractions/{job_id}          the state: queued, running, fields_ready, done or failed, with timings
    GET  /v1/extractions/{job_id}/result   the JSON: the checked fields at fields_ready, the final one at done;
                                           409 before either
    GET  /healthz                          liveness, no key

Every call but ``/healthz`` carries an API key, ``Authorization: Bearer <key>``
or ``X-API-Key: <key>``; the key names the tenant (``EXTRACTION_API_KEYS``), so a
tenant sees its own jobs only. Keys are compared as SHA-256 digests in constant
time, against every configured key.

Refused before anything is stored: a body over ``MAX_PDF_MB`` (413), one that
is not a PDF or cannot be opened, needs a password and came without its own, or
has too many pages (400), a line of business with no canonical policy schema
(400, by ``serving.vllm_entrypoint.request_lob``), a webhook that is not HTTPS
(400), a password in the URL (400).

**Encrypted PDFs** (``common.pdf_access``). One that only restricts printing
or copying is taken as it is. One that needs a password to open is read with
the password sent beside it - the multipart field ``password`` or the
``X-PDF-Password`` header, never the query string, which proxies log. Either
is stored unencrypted, as every reader downstream opens it; the password is
used once, here, and never stored, logged, queued or echoed. The job's record
says how the PDF came (``encrypted``: ``password``, ``restrictions`` or null).

The PDF goes to the jobs container (``transient/jobs/...``, never the raw
documents' container) for the worker to read, the job's record beside it, and
the job onto the queue - nothing else. The access log is off: a query names the
document; this logs the job id, sizes and the line only.
"""

from __future__ import annotations

import hmac
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from azure_service.job_queue import JobQueue, QueueError, job_message
from azure_service.settings import Settings, key_digest
from azure_service.store import READABLE, JobStore, new_record, public_view
from azure_service.webhooks import WebhookUrlError, check_webhook_url

log = logging.getLogger(__name__)

#: Room for a multipart body's boundaries and fields beyond the PDF itself.
FORM_OVERHEAD = 64 * 1024

#: Content types taken as the PDF itself.
PDF_TYPES = ("application/pdf", "application/octet-stream", "")


class Refused(HTTPException):
    """A request refused with a reason the caller is told."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(status_code=status, detail=reason)


def unlock(pdf: bytes, password: str | None) -> tuple[bytes, str | None]:
    """``pdf`` unencrypted (``common.pdf_access.unlocked_pdf``) and how it came
    encrypted - ``"password"``, ``"restrictions"`` or None; refused when it
    needs a password and came without its own. Never names the password."""
    from common.pdf_access import PdfLocked, encryption_kind, unlocked_pdf

    if not pdf.startswith(b"%PDF"):
        raise Refused(400, "the body is not a PDF (it does not start with %PDF)")
    try:
        kind = encryption_kind(pdf)
    except ImportError:                                  # pragma: no cover - the [data] extra has PyMuPDF
        return pdf, None
    except Exception:  # noqa: BLE001 - any reason is the same answer
        raise Refused(400, "the PDF cannot be opened") from None
    if kind is None:
        return pdf, None
    try:
        return unlocked_pdf(pdf, password), kind
    except PdfLocked as exc:
        how = "" if password else ": send its password as the multipart field `password` or the X-PDF-Password header"
        raise Refused(400, f"{exc}{how}") from None


def pdf_pages(pdf: bytes, max_pages: int) -> int | None:
    """The page count of ``pdf``, once it is seen to be a PDF the worker can
    render; None when PyMuPDF is not installed (the worker then finds out)."""
    if not pdf.startswith(b"%PDF"):
        raise Refused(400, "the body is not a PDF (it does not start with %PDF)")
    try:
        import pymupdf
    except ImportError:                                  # pragma: no cover - the [data] extra has it
        return None
    try:
        document = pymupdf.open(stream=pdf, filetype="pdf")
    except Exception:  # noqa: BLE001 - any reason is the same answer
        raise Refused(400, "the PDF cannot be opened") from None
    with document:
        if document.needs_pass:                           # unlock() opened it first
            raise Refused(400, "the PDF needs a password to open: send it with its password")
        count = document.page_count
    if count < 1:
        raise Refused(400, "the PDF has no pages")
    if count > max_pages:
        raise Refused(413, f"the PDF has {count} pages; at most {max_pages} are read")
    return count


def checked_lob(lob: str | None) -> str:
    """One line of business with a canonical policy schema, as serving takes it."""
    from serving.vllm_entrypoint import ServingError, request_lob

    if not lob or not lob.strip():
        raise Refused(400, "lob is required (e.g. lob=personal_auto): a policy is read against its line's schema")
    try:
        line = request_lob({"lob": lob.strip()})
    except ServingError as exc:
        raise Refused(400, str(exc)) from exc
    return str(line)


def _file_name(name: str | None) -> str:
    base = Path(str(name or "")).name.strip()
    return (base or "document.pdf")[:200]


def create_app(settings: Settings, store: JobStore, queue: JobQueue) -> FastAPI:
    """The API over ``store`` and ``queue``."""
    app = FastAPI(title="Policy extraction", version="1", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next: Any) -> Any:
        if settings.require_https and request.url.path != "/healthz" \
                and request.headers.get("x-forwarded-proto", "").lower() != "https":
            return JSONResponse({"error": "HTTPS only"}, status_code=403)
        response = await call_next(request)
        # Results are documents: never kept by a cache on the way.
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(HTTPException)
    async def refused(_request: Request, exc: HTTPException) -> JSONResponse:
        headers = getattr(exc, "headers", None)
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=headers)

    def tenant_of(request: Request) -> str:
        sent = request.headers.get("x-api-key") or ""
        authorization = request.headers.get("authorization") or ""
        if not sent and authorization[:7].lower() == "bearer ":
            sent = authorization[7:]
        sent = sent.strip()
        if not sent:
            raise HTTPException(401, "send Authorization: Bearer <API key>", headers={"WWW-Authenticate": "Bearer"})
        digest = key_digest(sent)
        tenant = None
        for known, name in settings.api_keys.items():     # every key, matched or not: constant time
            if hmac.compare_digest(known, digest):
                tenant = name
        if tenant is None:
            raise HTTPException(401, "unknown API key", headers={"WWW-Authenticate": "Bearer"})
        return tenant

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/extractions", status_code=202)
    async def submit(request: Request, tenant: str = Depends(tenant_of)) -> JSONResponse:
        if any(key.lower() == "password" for key in request.query_params):
            raise Refused(400, "never put the PDF's password in the URL: send it as the multipart field "
                               "`password` or the X-PDF-Password header")
        declared = (request.headers.get("content-length") or "").strip()
        cap = settings.max_pdf_bytes + FORM_OVERHEAD
        limit = settings.max_pdf_bytes
        too_large = f"the PDF must be at most {f'{limit // 2**20} MB' if limit >= 2**20 else f'{limit} bytes'}"
        if declared.isdigit() and int(declared) > cap:
            raise Refused(413, too_large)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > cap:
                raise Refused(413, too_large)
        fields: dict[str, Any] = dict(request.query_params)
        kind = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        password = request.headers.get("x-pdf-password") or None
        if kind == "multipart/form-data":
            pdf, name, form_password = await _from_form(request, bytes(body), fields)
            password = form_password or password
        elif kind in PDF_TYPES:
            pdf, name = bytes(body), fields.get("name")
        else:
            raise Refused(415, "send the PDF as the body (application/pdf) or as multipart/form-data `file`")
        if not pdf:
            raise Refused(400, "no PDF was sent")
        if len(pdf) > settings.max_pdf_bytes:
            raise Refused(413, too_large)
        record = await run_in_threadpool(accept, tenant, pdf, _file_name(name), fields.get("lob"),
                                         fields.get("webhook_url"), password)
        status_url = f"/v1/extractions/{record['job_id']}"
        return JSONResponse(
            {"job_id": record["job_id"], "status": record["status"], "status_url": status_url,
             "result_url": f"{status_url}/result"},
            status_code=202, headers={"Location": status_url})

    async def _from_form(request: Request, body: bytes,
                         fields: dict[str, Any]) -> tuple[bytes, str | None, str | None]:
        delivered = False

        async def receive() -> dict[str, Any]:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        try:
            form = await Request(request.scope, receive).form(max_files=1, max_fields=10)
        except Exception:  # noqa: BLE001 - a malformed form
            raise Refused(400, "the multipart body cannot be read") from None
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise Refused(400, "the multipart body has no `file` part")
        for key in ("lob", "webhook_url", "name"):
            value = form.get(key)
            if isinstance(value, str) and value.strip():
                fields[key] = value
        # Kept apart from `fields`: the password goes to unlock() and nowhere else.
        password = form.get("password")
        return (await upload.read(), fields.get("name") or upload.filename,
                password if isinstance(password, str) and password else None)

    def accept(tenant: str, pdf: bytes, name: str, lob: str | None, webhook_url: str | None,
               password: str | None = None) -> dict[str, Any]:
        line = checked_lob(lob)
        pdf, encrypted = unlock(pdf, password)
        pages = pdf_pages(pdf, settings.max_pages)
        if webhook_url:
            try:
                webhook_url = check_webhook_url(webhook_url, allow_private=settings.allow_private_webhooks)
            except WebhookUrlError as exc:
                raise Refused(400, str(exc)) from exc
            if not settings.webhook_secret_for(tenant):
                raise Refused(400, "webhooks are not configured for this tenant (no webhook secret)")
        job_id = uuid.uuid4().hex
        record = new_record(job_id, tenant, lob=line, name=name, size=len(pdf), pages=pages,
                            webhook_url=webhook_url or None, encrypted=encrypted)
        store.put_pdf(tenant, job_id, pdf)
        store.write_status(record)
        try:
            queue.send(job_message(job_id, tenant))
        except QueueError as exc:
            record.update(status="failed", error="the job could not be queued")
            store.write_status(record)
            store.delete_transient(tenant, job_id)
            log.error("job %s could not be queued: %s", job_id, exc)
            raise Refused(503, "the job could not be queued; send it again") from exc
        log.info("job %s queued: tenant %s, %d bytes, %s pages, lob %s%s%s", job_id, tenant, len(pdf), pages, line,
                 ", webhook" if webhook_url else "", f", unencrypted (it came with {encrypted})" if encrypted else "")
        return record

    @app.get("/v1/extractions/{job_id}")
    async def status(job_id: str, tenant: str = Depends(tenant_of)) -> dict[str, Any]:
        record = await run_in_threadpool(store.read_status, tenant, job_id)
        if record is None:
            raise Refused(404, "no such job")
        return public_view(record)

    @app.get("/v1/extractions/{job_id}/result")
    async def result(job_id: str, tenant: str = Depends(tenant_of)) -> JSONResponse:
        record = await run_in_threadpool(store.read_status, tenant, job_id)
        if record is None:
            raise Refused(404, "no such job")
        if record["status"] not in READABLE:
            view = public_view(record)
            return JSONResponse({"error": f"job {job_id} is {record['status']}: no result yet"
                                 if record["status"] != "failed" else f"job {job_id} failed", **view},
                                status_code=409)
        return JSONResponse(await run_in_threadpool(store.read_result, tenant, job_id))

    return app


def main() -> int:
    import os

    import uvicorn

    from azure_service.settings import SettingsError
    from azure_service.store import job_queue, jobs_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = Settings.from_env()
        settings.check_api()
    except SettingsError as exc:
        log.error("not configured: %s", exc)
        return 2
    log.info("api settings: %s", settings.describe())
    app = create_app(settings, JobStore(jobs_client(settings)), job_queue(settings))
    # TLS ends at the Container Apps ingress; the access log is off (a query names the document).
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", "8080")), access_log=False,
                proxy_headers=True, forwarded_allow_ips="*", log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
