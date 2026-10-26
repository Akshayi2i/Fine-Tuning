"""The Azure extraction service (azure_service/): API, worker, queue, webhooks, cold start.

Synthetic only: the in-memory queue and Blob backend, the echo model and a fake
served MinerU - no network, no Azure, no GPU.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="the [service] extra is not installed")
pytest.importorskip("httpx", reason="the [service] extra is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from artifact_registry.blob_client import BlobClient, InMemoryBackend  # noqa: E402
from azure_service.api import create_app
from azure_service.job_queue import InMemoryQueue, LeaseLostError, job_message
from azure_service.runtime import BASE_FALLBACK_ROUTE, ServingRuntime, TenantRuntime, served_model
from azure_service.settings import Settings, parse_api_keys
from azure_service.store import JobStore
from azure_service.webhooks import SIGNATURE_HEADER, WebhookSender, verify
from azure_service.worker import Worker
from inference_core.model_runner import EchoBackend
from serving.release_router import ServedRelease, ServingPlan
from serving.vllm_entrypoint import ReleaseRuntime

KEY = "k" * 40
OTHER_KEY = "o" * 40
SECRET = "webhook-secret-for-tests"
PAGES = ["Personal Auto Policy Declarations\nPolicy Number PA-1\n| Vehicle | VIN |\n|---|---|\n| 1 | 1HGCM |",
         "PART A - LIABILITY COVERAGE\nWe will pay damages for bodily injury.",
         "PART B - MEDICAL PAYMENTS\nWe will pay reasonable expenses."]
ANSWER = json.dumps({"policy": {"policy_number": {"raw": "PA-1", "parsed": "PA-1", "page_ref": [1]}}})
HOOK = "https://hooks.example.com/extractions?token=abc"


def make_pdf(pages: int = 3) -> bytes:
    import pymupdf

    document = pymupdf.open()
    for number in range(pages):
        document.new_page().insert_text((72, 72), f"Synthetic page {number + 1}", fontsize=12)
    return document.tobytes()


PDF = make_pdf()


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeReader:
    """The served MinerU as fields first uses it: pages rendered without text,
    then text alone - page 1's for the check, the rest in the GPU's idle turns."""

    def __init__(self, *, fail_reads: int = 0, fail_full_text: bool = False) -> None:
        self.asked: list = []
        self.fail_reads = fail_reads
        self.fail_full_text = fail_full_text

    def read(self, pdf, pages_dir, *, name, with_text, publish_pages=None):
        assert not with_text and pdf.startswith(b"%PDF")
        if self.fail_reads:
            self.fail_reads -= 1
            raise ConnectionError("MinerU unreachable")
        pages_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for number in range(1, len(PAGES) + 1):
            (pages_dir / f"page_{number:03d}.png").write_bytes(b"png")
            paths.append(str(pages_dir / f"page_{number:03d}.png"))
        return {"images": paths, "texts": {}, "ocr": {}, "links": None, "timings_s": {"render": 0.1, "links": None}}

    def ocr_text(self, pdf, *, name, page_count=None, gate=None, pages=None):
        self.asked.append(pages)
        if gate is not None:
            gate()
            if self.fail_full_text:
                raise ConnectionError("MinerU unreachable")
        numbers = pages or list(range(1, len(PAGES) + 1))
        texts = {number: PAGES[number - 1] for number in numbers}
        return {"pages": list(texts.values()), "texts": texts,
                "info": {"version": "4.0.12", "tier": "basic", "ocr_mode": "ocr"}, "seconds": 0.2}


class Hooks:
    """A webhook receiver: records each POST, answers 200."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, dict]] = []

    def __call__(self, url, body, headers):
        self.calls.append((url, body, headers))
        return 200

    def events(self) -> list[str]:
        return [json.loads(body)["event"] for _url, body, _headers in self.calls]


def release(lines=("personal_auto",)) -> ServedRelease:
    return ServedRelease(release_id="v3", base_model="qwen", merged_model="", lines=tuple(lines),
                         doc_types=("policy",), served_as="lora_on_base", adapter_dir="adapters/scope/policy/v3")


class Service:
    """The API, the queue, the store and a worker over them - as wired in production."""

    def __init__(self, tmp_path, *, reader=None, clock=None, lines=("personal_auto",), base_model=False,
                 **overrides) -> None:
        self.clock = clock or Clock()
        settings = dict(api_keys=parse_api_keys(json.dumps({"default": KEY, "other": [OTHER_KEY]})),
                        webhook_secret=SECRET, public_base_url="https://api.example.test",
                        work_dir=tmp_path / "work", tenants=("default", "other"), retry_delay_s=30)
        settings.update(overrides)
        self.settings = Settings(**settings)
        self.blob = InMemoryBackend()
        self.store = JobStore(BlobClient(backend=self.blob, container="extraction-jobs", context="serving"))
        self.queue = InMemoryQueue(self.clock)
        self.backend = EchoBackend(ANSWER)
        self.reader = reader or FakeReader()
        model = served_model(self.backend, "policy-v3")
        base = served_model(EchoBackend(ANSWER), "base") if base_model else None
        tenants = {name: TenantRuntime(name, plan=ServingPlan(release=release(lines)), release_model=model,
                                       base_model=base, release_runtime=ReleaseRuntime("v3"))
                   for name in self.settings.tenants}
        self.runtime = ServingRuntime(tenants=tenants, reader=self.reader)
        self.hooks = Hooks()
        self.client = TestClient(create_app(self.settings, self.store, self.queue))

    def worker(self) -> Worker:
        webhooks = WebhookSender(self.settings, post=self.hooks, background=False, backoff_s=(0.0,))
        return Worker(self.store, self.queue, self.runtime, self.settings, webhooks=webhooks, clock=self.clock)

    def submit(self, *, key=KEY, lob="personal_auto", webhook=None, body=PDF, **params) -> dict:
        query = {"lob": lob, **params}
        if webhook:
            query["webhook_url"] = webhook
        answer = self.client.post("/v1/extractions", params=query, content=body,
                                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/pdf"})
        assert answer.status_code == 202, answer.text
        return answer.json()

    def get(self, path: str, key=KEY):
        return self.client.get(path, headers={"X-API-Key": key})

    def status(self, job_id: str) -> dict:
        return self.get(f"/v1/extractions/{job_id}").json()

    def stored(self) -> list[str]:
        return sorted(key for (_container, key) in self.blob._store)


@pytest.fixture
def service(tmp_path) -> Service:
    return Service(tmp_path)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_a_pdf_is_served_its_checked_fields_then_its_final_result(service):
    job = service.submit(webhook=HOOK)
    job_id = job["job_id"]
    assert job["status"] == "queued" and job["result_url"] == f"/v1/extractions/{job_id}/result"
    assert service.status(job_id)["status"] == "queued"
    early = service.get(f"/v1/extractions/{job_id}/result")
    assert early.status_code == 409 and early.json()["status"] == "queued"

    worker = service.worker()
    assert worker.poll_once()                                      # phase 1: the fields
    view = service.status(job_id)
    assert view["status"] == "fields_ready" and view["fields_verified"] and view["pages"] == 3
    assert view["model"] == "policy-v3" and view["release"] == "v3" and view["route"] == "release"
    assert {"queued", "render", "model", "verify", "fields", "fields_ready"} <= set(view["timings_s"])
    fields = service.get(f"/v1/extractions/{job_id}/result").json()
    assert fields["status"] == "fields_ready" and fields["final"] is False and fields["full_text"] is False
    assert fields["output"]["extraction"]["policy"]["policy_number"]["raw"] == "PA-1"
    assert fields["output"]["extraction"].get("text_sections") is None
    assert service.reader.asked == [[1]]                           # checked on the page they cite
    assert service.backend.calls                                   # the model read the page images

    assert worker.text_once()                                      # phase 2: the full text
    view = service.status(job_id)
    assert view["status"] == "done" and view["error"] is None and view["full_text"] is True
    assert {"ocr", "full_text", "total"} <= set(view["timings_s"]) and view["ocr"]["version"] == "4.0.12"
    final = service.get(f"/v1/extractions/{job_id}/result").json()
    assert final["final"] is True and final["output"]["extraction"]["text_sections"]
    assert final["output"]["text_coverage"] == 1.0
    assert service.reader.asked[-1] == [2, 3]                      # the cited page is not read again

    # Ended: the message, the PDF and the checkpoint are gone; status and result stay.
    assert len(service.queue) == 0 and worker.held() == []
    assert service.stored() == [f"jobs/default/{job_id}/result.json", f"jobs/default/{job_id}/status.json"]
    assert not (service.settings.work_dir / job_id).exists()
    assert service.hooks.events() == ["extraction.fields_ready", "extraction.done"]


def test_a_multipart_upload_is_taken_as_the_pdf_with_its_form_fields(service):
    answer = service.client.post("/v1/extractions", files={"file": ("plymouth scan.pdf", PDF, "application/pdf")},
                                 data={"lob": "classic_auto"}, headers={"Authorization": f"Bearer {KEY}"})
    assert answer.status_code == 202, answer.text
    view = service.status(answer.json()["job_id"])
    # Classic auto is read as personal auto, as serving reads it.
    assert view["document"] == "plymouth scan.pdf" and view["lob"] == "personal_auto" and view["pages"] == 3
    assert view["bytes"] == len(PDF) and view["webhook"] is False


def test_the_worker_threads_take_a_job_to_done_on_their_own(tmp_path):
    service = Service(tmp_path, clock=time.time)
    worker = service.worker()
    thread = threading.Thread(target=worker.run_forever, kwargs={"idle_s": 0.01, "max_idle_s": 0.05}, daemon=True)
    thread.start()
    try:
        job_id = service.submit()["job_id"]
        deadline = time.monotonic() + 60
        while service.status(job_id)["status"] != "done" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert service.status(job_id)["status"] == "done"
    finally:
        worker.shutdown()
        thread.join(timeout=10)
    assert not thread.is_alive()


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_every_call_but_health_needs_a_key_and_sees_its_own_tenants_jobs_only(service):
    assert service.client.get("/healthz").json() == {"status": "ok"}
    for headers in ({}, {"Authorization": "Bearer " + "x" * 40}, {"X-API-Key": KEY[:-1]}, {"Authorization": KEY}):
        answer = service.client.post("/v1/extractions", params={"lob": "personal_auto"}, content=PDF,
                                     headers={"Content-Type": "application/pdf", **headers})
        assert answer.status_code == 401 and answer.headers["WWW-Authenticate"] == "Bearer"
    assert len(service.queue) == 0 and service.stored() == []

    job_id = service.submit()["job_id"]
    assert service.get(f"/v1/extractions/{job_id}").status_code == 200
    assert service.get(f"/v1/extractions/{job_id}", key="nope" * 10).status_code == 401
    # Another tenant's key does not find it, nor its result.
    assert service.get(f"/v1/extractions/{job_id}", key=OTHER_KEY).status_code == 404
    assert service.get(f"/v1/extractions/{job_id}/result", key=OTHER_KEY).status_code == 404
    assert service.get("/v1/extractions/../../jobs").status_code == 404
    assert service.get("/v1/extractions/" + "a" * 31).status_code == 404


@pytest.mark.parametrize("body, params, status, why", [
    (b"not a pdf", {"lob": "personal_auto"}, 400, "not a PDF"),
    (b"%PDF-1.7 truncated", {"lob": "personal_auto"}, 400, "cannot be opened"),
    (PDF, {}, 400, "lob is required"),
    (PDF, {"lob": "underwater_basket_weaving"}, 400, "no canonical policy schema"),
    (PDF, {"lob": "personal_auto", "webhook_url": "http://hooks.example.com/x"}, 400, "https"),
    (PDF, {"lob": "personal_auto", "webhook_url": "https://169.254.169.254/latest"}, 400, "private network"),
    (b"%PDF" + b"0" * 5000, {"lob": "personal_auto"}, 413, "at most"),
])
def test_a_request_the_pipeline_cannot_read_is_refused_before_anything_is_stored(tmp_path, body, params, status,
                                                                                  why):
    service = Service(tmp_path, max_pdf_bytes=4096 if status == 413 else 150 * 2**20)
    answer = service.client.post("/v1/extractions", params=params, content=body,
                                 headers={"X-API-Key": KEY, "Content-Type": "application/pdf"})
    assert answer.status_code == status and why in answer.json()["error"], answer.text
    assert service.stored() == [] and len(service.queue) == 0


def test_a_page_count_over_the_limit_and_an_unknown_content_type_are_refused(tmp_path):
    service = Service(tmp_path, max_pages=2)
    answer = service.client.post("/v1/extractions", params={"lob": "personal_auto"}, content=PDF,
                                 headers={"X-API-Key": KEY, "Content-Type": "application/pdf"})
    assert answer.status_code == 413 and "3 pages" in answer.json()["error"]
    answer = service.client.post("/v1/extractions", params={"lob": "personal_auto"}, content=PDF,
                                 headers={"X-API-Key": KEY, "Content-Type": "text/plain"})
    assert answer.status_code == 415


def test_a_webhook_needs_a_secret_for_the_tenant(tmp_path):
    service = Service(tmp_path, webhook_secret=None, webhook_secrets={"other": "s" * 20})
    answer = service.client.post("/v1/extractions", params={"lob": "personal_auto", "webhook_url": HOOK},
                                 content=PDF, headers={"X-API-Key": KEY, "Content-Type": "application/pdf"})
    assert answer.status_code == 400 and "not configured" in answer.json()["error"]
    service.submit(key=OTHER_KEY, webhook=HOOK)                     # the other tenant has one


def test_https_only_refuses_a_request_the_ingress_took_over_http(tmp_path):
    service = Service(tmp_path, require_https=True)
    assert service.get("/v1/extractions/" + "a" * 32).status_code == 403
    answer = service.client.get("/v1/extractions/" + "a" * 32,
                                headers={"X-API-Key": KEY, "X-Forwarded-Proto": "https"})
    assert answer.status_code == 404
    assert service.client.get("/healthz").status_code == 200       # the platform's probe


# --------------------------------------------------------------------------
# Webhooks
# --------------------------------------------------------------------------


def test_a_webhook_is_signed_and_carries_no_document_content(service):
    job_id = service.submit(webhook=HOOK)["job_id"]
    worker = service.worker()
    worker.poll_once()
    worker.text_once()
    assert len(service.hooks.calls) == 2
    for url, body, headers in service.hooks.calls:
        assert url == HOOK
        assert verify(SECRET, body, headers[SIGNATURE_HEADER])
        assert not verify(SECRET, body + b" ", headers[SIGNATURE_HEADER])
        assert not verify("another secret", body, headers[SIGNATURE_HEADER])
        sent = json.loads(body)
        assert set(sent) == {"event", "job_id", "status", "status_url", "result_url", "sent_at", "delivery_id"}
        assert sent["job_id"] == job_id and headers["X-Extraction-Delivery"] == sent["delivery_id"]
        assert sent["result_url"] == f"https://api.example.test/v1/extractions/{job_id}/result"
        assert b"PA-1" not in body and b"Policy" not in body
    # The address is never shown back: its query may hold the receiver's token.
    assert "webhook_url" not in service.status(job_id) and service.status(job_id)["webhook"] is True


def test_webhook_addresses_inside_a_network_or_with_credentials_are_refused():
    from azure_service.webhooks import WebhookUrlError, check_webhook_url

    for url in ("http://hooks.example.com/x", "https://10.0.0.4/x", "https://127.0.0.1/x", "https://localhost/x",
                "https://user:pw@hooks.example.com/x", "ftp://hooks.example.com", "", "https://[::1]/x"):
        with pytest.raises(WebhookUrlError):
            check_webhook_url(url)
    assert check_webhook_url(HOOK) == HOOK
    assert check_webhook_url("http://localhost:9000/x", allow_private=True)


def test_a_failed_delivery_is_tried_again_and_never_fails_the_job(tmp_path):
    service = Service(tmp_path)
    tries = []

    def flaky(url, body, headers):
        tries.append(body)
        if len(tries) == 1:
            raise ConnectionError("receiver down")
        return 503 if len(tries) == 2 else 204

    sender = WebhookSender(service.settings, post=flaky, background=False, backoff_s=(0, 0, 0))
    record = {"job_id": "a" * 32, "tenant": "default", "status": "done", "webhook_url": HOOK}
    sender.notify(record, "done")
    assert len(tries) == 3 and len(set(tries)) == 1 and sender.outcomes[("a" * 32, "done")] == "delivered"


# --------------------------------------------------------------------------
# The queue: retries, poison, restarts
# --------------------------------------------------------------------------


def test_the_in_memory_queue_keeps_azure_queue_semantics():
    clock = Clock()
    queue = InMemoryQueue(clock)
    queue.send(job_message("a" * 32, "default"))
    first = queue.receive(60)
    assert first.dequeue_count == 1 and first.job() == {"job_id": "a" * 32, "tenant": "default"}
    assert queue.receive(60) is None                                # hidden while leased
    renewed = queue.extend(first, 60)
    with pytest.raises(LeaseLostError):
        queue.delete(first)                                         # the old receipt is stale
    clock.advance(61)
    second = queue.receive(60)                                      # the lease ran out: delivered again
    assert second.dequeue_count == 2
    with pytest.raises(LeaseLostError):
        queue.extend(renewed, 60)
    queue.delete(second)
    assert len(queue) == 0


def test_the_azure_queue_maps_receipts_and_lost_leases(monkeypatch):
    """AzureStorageQueue over a stand-in for azure-storage-queue's QueueClient."""
    import sys
    from types import ModuleType, SimpleNamespace

    from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

    from azure_service.job_queue import AzureStorageQueue, QueueError

    def error(status):
        exc = HttpResponseError(message=f"HTTP {status}")
        exc.status_code, exc.reason = status, "PopReceiptMismatch" if status == 400 else "Server Error"
        return exc

    class Client:
        sent: list = []

        def __init__(self, url=None, name=None, credential=None, **policies):
            assert set(policies) == {"message_encode_policy", "message_decode_policy"}
            self.receipt, self.fail_with = "r1", None

        @classmethod
        def from_connection_string(cls, connection, name, **policies):
            return cls(**policies)

        def send_message(self, content, time_to_live=None):
            Client.sent.append((content, time_to_live))

        def receive_message(self, visibility_timeout=None):
            return SimpleNamespace(id="m1", pop_receipt=self.receipt, dequeue_count=2, content=Client.sent[-1][0])

        def update_message(self, message_id, pop_receipt=None, visibility_timeout=None):
            if self.fail_with:
                raise self.fail_with
            if pop_receipt != self.receipt:
                raise error(400)
            self.receipt = "r2"
            return SimpleNamespace(pop_receipt="r2")

        def delete_message(self, message_id, pop_receipt=None):
            raise ResourceNotFoundError("gone")

    module = ModuleType("azure.storage.queue")
    module.QueueClient = Client
    module.TextBase64EncodePolicy = module.TextBase64DecodePolicy = object
    monkeypatch.setitem(sys.modules, "azure.storage.queue", module)

    queue = AzureStorageQueue("jobs", connection_string="UseDevelopmentStorage=true")
    queue.send(job_message("b" * 32, "default"))
    assert Client.sent[-1][1] == -1                                  # never expires
    message = queue.receive(300)
    assert message.receipt == "r1" and message.dequeue_count == 2 and message.job()["job_id"] == "b" * 32
    renewed = queue.extend(message, 300)
    assert renewed.receipt == "r2"
    with pytest.raises(LeaseLostError):
        queue.extend(message, 300)                                  # the stale receipt
    with pytest.raises(LeaseLostError):
        queue.delete(renewed)
    queue._client.fail_with = error(500)
    with pytest.raises(QueueError) as raised:
        queue.extend(renewed, 300)
    assert not isinstance(raised.value, LeaseLostError)             # an outage is not a lost lease


def test_a_lease_is_renewed_while_the_job_runs(service):
    service.submit()
    worker = service.worker()
    worker.poll_once()                                              # fields out, full text waiting
    for _ in range(4):
        service.clock.advance(200)
        worker.renew_leases()
        assert service.queue.receive(300) is None                   # no other worker gets it
    worker.text_once()
    assert len(service.queue) == 0


def test_a_failure_worth_retrying_puts_the_job_back_then_it_succeeds(tmp_path):
    service = Service(tmp_path, reader=FakeReader(fail_reads=1))
    job_id = service.submit()["job_id"]
    worker = service.worker()
    worker.poll_once()
    view = service.status(job_id)
    assert view["status"] == "queued" and "ConnectionError" in view["error"] and view["attempts"] == 1
    assert not worker.poll_once()                                   # not before the retry delay
    service.clock.advance(31)
    worker.poll_once()
    worker.text_once()
    view = service.status(job_id)
    assert view["status"] == "done" and view["attempts"] == 2 and view["error"] is None


def test_a_poison_job_is_failed_after_its_attempts_and_never_loops(service):
    job_id = service.submit(webhook=HOOK)["job_id"]
    for _ in range(service.settings.max_attempts):
        assert service.queue.receive(300) is not None              # a worker took it, then died
        service.clock.advance(301)
    worker = service.worker()
    assert worker.poll_once()
    view = service.status(job_id)
    assert view["status"] == "failed" and "given up after 3 attempts" in view["error"] and view["attempts"] == 4
    assert service.get(f"/v1/extractions/{job_id}/result").status_code == 409
    assert len(service.queue) == 0 and not service.backend.calls     # never read again
    assert service.stored() == [f"jobs/default/{job_id}/status.json"]
    assert service.hooks.events() == ["extraction.failed"]
    assert json.loads(service.hooks.calls[0][1])["result_url"] is None


def test_a_job_the_worker_can_never_read_fails_at_once(tmp_path):
    # The release trained on personal auto; homeowners would need the base model, not configured.
    service = Service(tmp_path)
    job_id = service.submit(lob="homeowners")["job_id"]
    service.worker().poll_once()
    view = service.status(job_id)
    assert view["status"] == "failed" and view["attempts"] == 1 and "ALLOW_BASE_FALLBACK" in view["error"]
    assert len(service.queue) == 0


def test_a_line_the_release_did_not_train_on_is_read_by_the_base_model_when_configured(tmp_path):
    service = Service(tmp_path, base_model=True)
    job_id = service.submit(lob="homeowners")["job_id"]
    worker = service.worker()
    worker.poll_once()
    worker.text_once()
    view = service.status(job_id)
    assert view["status"] == "done" and view["route"] == BASE_FALLBACK_ROUTE and view["model"] == "base"
    assert view["release"] is None and not service.backend.calls     # the release's model never read it


def test_a_worker_that_dies_after_the_fields_leaves_the_full_text_to_the_next(service):
    job_id = service.submit()["job_id"]
    first = service.worker()
    first.poll_once()                                               # fields served; then the worker dies
    assert service.status(job_id)["status"] == "fields_ready"
    model_calls = len(service.backend.calls)
    service.clock.advance(301)                                      # its lease runs out

    second = service.worker()
    assert second.poll_once() and service.status(job_id)["attempts"] == 2
    assert second.text_once()
    view = service.status(job_id)
    assert view["status"] == "done" and view["full_text"] is True
    assert len(service.backend.calls) == model_calls                # phase 2 resumed: no model time again
    assert service.reader.asked[-1] == [2, 3]
    # The dead worker's leftover cannot write over the newer state.
    first.text_once()
    assert service.status(job_id)["status"] == "done" and len(service.queue) == 0


def test_a_worker_that_dies_during_the_fields_has_them_read_again(service):
    job_id = service.submit()["job_id"]
    assert service.queue.receive(300) is not None                   # taken, then the worker died
    service.clock.advance(301)
    worker = service.worker()
    worker.poll_once()
    assert service.status(job_id)["status"] == "fields_ready" and service.status(job_id)["attempts"] == 2
    worker.text_once()
    assert service.status(job_id)["status"] == "done"


def test_checked_fields_stand_when_the_full_text_fails_for_good(tmp_path):
    service = Service(tmp_path, reader=FakeReader(fail_full_text=True), max_attempts=1)
    job_id = service.submit(webhook=HOOK)["job_id"]
    worker = service.worker()
    worker.poll_once()
    worker.text_once()
    view = service.status(job_id)
    assert view["status"] == "done" and view["full_text"] is False and "full text: ConnectionError" in view["error"]
    result = service.get(f"/v1/extractions/{job_id}/result").json()
    assert result["final"] is True and result["full_text"] is False
    assert result["output"]["extraction"]["policy"]["policy_number"]["raw"] == "PA-1"
    assert service.hooks.events() == ["extraction.fields_ready", "extraction.done"]


def test_a_stopping_worker_hands_its_jobs_back_at_once(service):
    job_id = service.submit()["job_id"]
    worker = service.worker()
    worker.poll_once()
    worker.shutdown()
    assert not worker.poll_once()                                   # takes nothing more
    again = service.queue.receive(300)                              # visible to the next replica now
    assert again is not None and again.job()["job_id"] == job_id and again.dequeue_count == 2


def test_a_second_delivery_of_a_finished_job_is_only_deleted(service):
    job_id = service.submit()["job_id"]
    worker = service.worker()
    worker.poll_once()
    worker.text_once()
    service.queue.send(job_message(job_id, "default"))             # a duplicate
    service.queue.send("not json")                                  # and garbage
    calls = len(service.backend.calls)
    assert worker.poll_once() and worker.poll_once()
    assert len(service.queue) == 0 and len(service.backend.calls) == calls


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def test_settings_come_from_the_environment_and_never_show_a_secret():
    from azure_service.settings import SettingsError

    env = {"EXTRACTION_API_KEYS": json.dumps({"Default": [KEY, OTHER_KEY]}), "WEBHOOK_SECRET": SECRET,
           "MODEL_URL": "https://pod-8000.proxy.runpod.net", "MINERU_URL": "https://pod-8001.proxy.runpod.net",
           "MODEL_TOKEN": "model-token", "STORAGE_ACCOUNT": "fideonstore", "MAX_PDF_MB": "50",
           "SERVED_TENANTS": "default, acme"}
    settings = Settings.from_env(env)
    settings.check_api()
    settings.check_worker()
    assert set(settings.api_keys.values()) == {"default"} and len(settings.api_keys) == 2
    assert settings.max_pdf_bytes == 50 * 2**20 and settings.tenants == ("default", "acme")
    assert settings.queue_account_url == "https://fideonstore.queue.core.windows.net"
    shown = repr(settings) + json.dumps(settings.describe())
    for secret in (KEY, OTHER_KEY, SECRET, "model-token"):
        assert secret not in shown
    with pytest.raises(SettingsError, match="shorter than"):
        parse_api_keys(json.dumps({"default": "short"}))
    with pytest.raises(SettingsError, match="tenants"):
        parse_api_keys(json.dumps({"a": KEY, "b": KEY}))
    with pytest.raises(SettingsError, match="BASE_MODEL_NAME"):
        Settings.from_env({**env, "ALLOW_BASE_FALLBACK": "true"})
    with pytest.raises(SettingsError, match="not HTTPS"):
        Settings.from_env({**env, "MODEL_URL": "http://pod.example.com"}).check_worker()
    with pytest.raises(SettingsError, match="no API keys"):
        Settings.from_env({}).check_api()


def test_the_root_dockerignore_leaves_out_what_the_service_one_does():
    """A builder reading only the root .dockerignore (az acr build) must send no
    more of the working tree - client documents, labels - than BuildKit would."""
    root = Path(__file__).resolve().parent.parent

    def entries(path: Path) -> list[str]:
        return [line for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]

    assert entries(root / ".dockerignore") == entries(root / "azure_service" / "Dockerfile.dockerignore")
    assert entries(root / ".dockerignore")[0] == "*"
