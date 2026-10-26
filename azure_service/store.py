"""Job state and results as JSON blobs, through ``artifact_registry.blob_client``.

Layout in the jobs container (``JOBS_CONTAINER``, default ``extraction-jobs``;
never the raw-documents container, which serving may not read):

    jobs/<tenant>/<job id>/status.json            the job's state and timings - no document content
    jobs/<tenant>/<job id>/result.json            the served JSON: the fields at fields_ready, the final at done
    transient/jobs/<tenant>/<job id>/document.pdf the PDF, until the job ends (KEEP_FILES keeps it)
    transient/jobs/<tenant>/<job id>/phase1.json  the model's answer, for phase 2 after a restart

Every write is one whole object (``BlobClient.write_json``: a single upload,
which Azure commits atomically): a reader sees the old blob or the new one,
never half of one - the result is rewritten while it is served. The result is
written before the status that announces it, so a poll that reads
``fields_ready`` or ``done`` always finds its result.

``transient/`` holds document content only while a job runs; a lifecycle rule
on the prefix (DEPLOY.md) removes anything a crash left behind.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from artifact_registry.blob_client import BlobClient, BlobError

log = logging.getLogger(__name__)

#: A job id: 32 lowercase hex characters (uuid4). Anything else is no job.
JOB_ID = re.compile(r"^[0-9a-f]{32}$")

STATUSES = ("queued", "running", "fields_ready", "done", "failed")
#: States whose result can be read.
READABLE = frozenset({"fields_ready", "done"})
#: States a job never leaves.
FINAL = frozenset({"done", "failed"})

#: What a status poll never shows: where the webhook goes (its query may carry a token).
PRIVATE_KEYS = frozenset({"webhook_url"})


def new_record(job_id: str, tenant: str, *, lob: str, name: str, size: int, pages: int | None,
               webhook_url: str | None, now: float | None = None) -> dict[str, Any]:
    return {
        "job_id": job_id, "tenant": tenant, "status": "queued", "lob": lob, "document": name,
        "bytes": size, "pages": pages, "submitted": now if now is not None else time.time(),
        "started": None, "fields_ready_at": None, "finished": None, "attempts": 0,
        "webhook_url": webhook_url, "error": None, "timings_s": {},
    }


def public_view(record: dict[str, Any], *, now: float | None = None) -> dict[str, Any]:
    """What ``GET /v1/extractions/{id}`` answers: the state and timings, never the
    extraction (``/result`` holds it) and never the webhook's address."""
    now = now if now is not None else time.time()
    view = {key: value for key, value in record.items() if key not in PRIVATE_KEYS}
    view["webhook"] = bool(record.get("webhook_url"))
    submitted = record.get("submitted") or now
    timings = dict(record.get("timings_s") or {})
    if record.get("status") == "queued":
        timings["waiting"] = round(now - submitted, 2)
    elif record.get("status") in ("running", "fields_ready") and record.get("started"):
        timings["running"] = round(now - record["started"], 2)
    view["timings_s"] = timings
    for key in ("submitted", "started", "fields_ready_at", "finished"):
        if view.get(key):
            view[key] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(view[key]))
    return view


class JobStore:
    """The blobs of the service's jobs, in one container."""

    def __init__(self, client: BlobClient) -> None:
        self.client = client

    # -- keys -----------------------------------------------------------------

    @staticmethod
    def _folder(tenant: str, job_id: str) -> str:
        if not JOB_ID.match(job_id):
            raise KeyError(f"not a job id: {job_id!r}")
        return f"{tenant}/{job_id}"

    def status_key(self, tenant: str, job_id: str) -> str:
        return f"jobs/{self._folder(tenant, job_id)}/status.json"

    def result_key(self, tenant: str, job_id: str) -> str:
        return f"jobs/{self._folder(tenant, job_id)}/result.json"

    def pdf_key(self, tenant: str, job_id: str) -> str:
        return f"transient/jobs/{self._folder(tenant, job_id)}/document.pdf"

    def checkpoint_key(self, tenant: str, job_id: str) -> str:
        return f"transient/jobs/{self._folder(tenant, job_id)}/phase1.json"

    # -- status and result ------------------------------------------------------

    def read_status(self, tenant: str, job_id: str) -> dict[str, Any] | None:
        """The job's record, or None when the tenant has no such job."""
        try:
            key = self.status_key(tenant, job_id)
        except KeyError:
            return None
        if not self.client.exists(key):
            return None
        return self.client.read_json(key)

    def write_status(self, record: dict[str, Any]) -> None:
        self.client.write_json(self.status_key(record["tenant"], record["job_id"]), record)

    def write_result(self, tenant: str, job_id: str, result: dict[str, Any]) -> None:
        self.client.write_json(self.result_key(tenant, job_id), result)

    def read_result(self, tenant: str, job_id: str) -> dict[str, Any]:
        return self.client.read_json(self.result_key(tenant, job_id))

    # -- the document, while the job runs -----------------------------------------

    def put_pdf(self, tenant: str, job_id: str, pdf: bytes) -> None:
        self.client.write_bytes(self.pdf_key(tenant, job_id), pdf)

    def read_pdf(self, tenant: str, job_id: str) -> bytes:
        return self.client.read_bytes(self.pdf_key(tenant, job_id))

    def write_checkpoint(self, tenant: str, job_id: str, checkpoint: dict[str, Any]) -> None:
        self.client.write_json(self.checkpoint_key(tenant, job_id), checkpoint)

    def read_checkpoint(self, tenant: str, job_id: str) -> dict[str, Any] | None:
        key = self.checkpoint_key(tenant, job_id)
        return self.client.read_json(key) if self.client.exists(key) else None

    def delete_transient(self, tenant: str, job_id: str) -> None:
        """The PDF and the checkpoint: the document's content beyond its result."""
        for key in (self.pdf_key(tenant, job_id), self.checkpoint_key(tenant, job_id)):
            try:
                if self.client.exists(key):
                    self.client.delete(key)
            except BlobError as exc:          # the lifecycle rule removes it later
                log.warning("job %s: could not delete %s: %s", job_id, key.rsplit("/", 1)[-1], exc)


# --------------------------------------------------------------------------
# Logins, from the settings
# --------------------------------------------------------------------------


def _credential(settings: Any) -> Any:
    """The container's managed identity (azure-identity)."""
    from azure.identity import ManagedIdentityCredential

    if settings.managed_identity_client_id:
        return ManagedIdentityCredential(client_id=settings.managed_identity_client_id)
    return ManagedIdentityCredential()


def blob_backend(settings: Any) -> Any:
    """The Azure Blob backend: the managed identity, or a SAS / connection string
    as ``artifact_registry.blob_client.azure_from_environment`` reads them."""
    from artifact_registry.blob_client import AzureBackend, azure_from_environment

    if settings.storage_auth == "managed_identity":
        # BlobServiceClient takes a token credential where it takes a SAS string.
        return AzureBackend(account_url=settings.blob_account_url, credential=_credential(settings))
    return azure_from_environment()


def jobs_client(settings: Any, backend: Any = None) -> BlobClient:
    """The jobs container, in the ``serving`` context: raw documents are refused."""
    return BlobClient(backend=backend or blob_backend(settings), container=settings.jobs_container,
                      context="serving")


def registry_client(settings: Any, backend: Any = None) -> BlobClient:
    """The registry's container (release index, bundles, calibrators), read-only in use."""
    return BlobClient(backend=backend or blob_backend(settings), container=settings.registry_container,
                      context="serving")


def job_queue(settings: Any) -> Any:
    """The Azure Storage Queue the settings name."""
    import os

    from azure_service.job_queue import AzureStorageQueue

    if settings.storage_auth == "managed_identity":
        return AzureStorageQueue(settings.queue_name, account_url=settings.queue_account_url,
                                 credential=_credential(settings))
    connection = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection:
        return AzureStorageQueue(settings.queue_name, connection_string=connection)
    sas = (os.environ.get("AZURE_QUEUE_SAS_TOKEN") or "").strip().lstrip("?")
    if settings.queue_account_url and sas:
        return AzureStorageQueue(settings.queue_name, account_url=settings.queue_account_url, credential=sas)
    from azure_service.job_queue import QueueError

    raise QueueError("no queue login: STORAGE_AUTH=managed_identity, AZURE_STORAGE_CONNECTION_STRING, or "
                     "STORAGE_ACCOUNT with AZURE_QUEUE_SAS_TOKEN")
