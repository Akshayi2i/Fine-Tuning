"""Webhooks: a signed POST when a job's fields are ready, when it is done, and when it failed.

The body names the job and where to fetch its result - **no document
content**: the result itself is fetched with the API key, so a webhook sent to
the wrong place, or logged by a proxy, gives nothing away::

    {"event": "extraction.fields_ready", "job_id": "...", "status": "fields_ready",
     "status_url": "https://.../v1/extractions/<id>", "result_url": "https://.../v1/extractions/<id>/result",
     "sent_at": "2026-10-24T10:00:00Z", "delivery_id": "..."}

``X-Extraction-Signature: sha256=<hex>`` is the HMAC-SHA256 of the exact body
bytes with the tenant's webhook secret (:func:`sign`; receivers check it with
:func:`verify`, in constant time, and reject a ``sent_at`` older than a few
minutes against replays). ``X-Extraction-Event`` and ``X-Extraction-Delivery``
repeat the event and the delivery id (the same on every retry of one event, so
a receiver can drop a duplicate).

Only HTTPS URLs are taken (:func:`common.transport.assert_encrypted`), with no
credentials in them, and - unless ``ALLOW_PRIVATE_WEBHOOKS`` - none whose host
is or resolves to a loopback, private, link-local or reserved address: the
worker inside Azure must not be made to POST into its own network. Redirects
are refused (:func:`common.transport.open_url`). Delivery is best-effort, a few
tries with a back-off on a background thread; polling stays the source of truth.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-Extraction-Signature"
EVENT_HEADER = "X-Extraction-Event"
DELIVERY_HEADER = "X-Extraction-Delivery"

#: The events a webhook is sent for.
EVENTS = ("fields_ready", "done", "failed")

#: Seconds before each try of one delivery.
BACKOFF_S = (0.0, 2.0, 10.0)

Post = Callable[[str, bytes, dict[str, str]], int]


class WebhookUrlError(ValueError):
    """A webhook address that is refused."""


def _blocked(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
            or ip.is_unspecified)


def check_webhook_url(url: str, *, allow_private: bool = False) -> str:
    """``url`` when it may be called back; :class:`WebhookUrlError` otherwise."""
    from common.transport import InsecureUrlError, assert_encrypted

    if not isinstance(url, str) or not url.strip() or len(url) > 2048:
        raise WebhookUrlError("webhook_url must be an https:// URL of at most 2048 characters")
    url = url.strip()
    parts = urlsplit(url)
    if parts.username or parts.password:
        raise WebhookUrlError("webhook_url must not carry credentials; the body is signed instead")
    try:
        assert_encrypted(url, "the webhook")
    except InsecureUrlError as exc:
        raise WebhookUrlError("webhook_url must be https://: a job's status would travel unencrypted") from exc
    host = (parts.hostname or "").lower()
    if not host:
        raise WebhookUrlError("webhook_url has no host")
    if not allow_private:
        if parts.scheme != "https":
            raise WebhookUrlError("webhook_url must be https://")
        try:
            literal = _blocked(host)
        except ValueError:
            literal = host == "localhost" or host.endswith(".localhost")
        if literal:
            raise WebhookUrlError("webhook_url points into a private network")
    return url


def sign(secret: str, body: bytes) -> str:
    """The signature header's value for ``body``."""
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify(secret: str, body: bytes, header: str | None) -> bool:
    """Whether ``header`` is ``body``'s signature with ``secret`` (constant time)."""
    if not header:
        return False
    return hmac.compare_digest(sign(secret, body).encode("ascii"), header.strip().encode("utf-8", "replace"))


def payload(record: dict[str, Any], event: str, base_url: str | None, *, now: float | None = None,
            delivery_id: str | None = None) -> bytes:
    """The body of one event: the job, its state and where to read it - nothing of the document."""
    job_id = record["job_id"]
    base = (base_url or "").rstrip("/")
    status_url = f"{base}/v1/extractions/{job_id}"
    body = {
        "event": f"extraction.{event}",
        "job_id": job_id,
        "status": record["status"],
        "status_url": status_url,
        "result_url": f"{status_url}/result" if record["status"] in ("fields_ready", "done") else None,
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time())),
        "delivery_id": delivery_id or uuid.uuid4().hex,
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def post_https(url: str, body: bytes, headers: dict[str, str], *, allow_private: bool = False,
               timeout: float = 10.0) -> int:
    """POST ``body`` to ``url``; returns the HTTP status. Checked again before it
    goes, the host's addresses included (a public name may resolve inward)."""
    from common.transport import open_url

    check_webhook_url(url, allow_private=allow_private)
    if not allow_private:
        parts = urlsplit(url)
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
        if any(_blocked(str(info[4][0])) for info in infos):
            raise WebhookUrlError("webhook_url resolves into a private network")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json",
                                                              "User-Agent": "fideon-extraction-webhook/1",
                                                              **headers}, method="POST")
    try:
        with open_url(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


class WebhookSender:
    """Signs and delivers each job's events, a few tries each, off the worker's threads."""

    def __init__(self, settings: Any, *, post: Post | None = None, background: bool = True,
                 backoff_s: Sequence[float] = BACKOFF_S, sleep: Callable[[float], None] = time.sleep) -> None:
        self.settings = settings
        self.post = post or (lambda url, body, headers: post_https(
            url, body, headers, allow_private=settings.allow_private_webhooks))
        self.backoff_s = tuple(backoff_s) or (0.0,)
        self.sleep = sleep
        #: Each event's outcome, by (job id, event): what the logs also say.
        self.outcomes: dict[tuple[str, str], str] = {}
        self._pending: queue.Queue[tuple[str, str, str, bytes, dict[str, str]] | None] | None = None
        if background:
            self._pending = queue.Queue()
            threading.Thread(target=self._run, name="webhooks", daemon=True).start()

    def notify(self, record: dict[str, Any], event: str) -> None:
        """Send ``event`` for the job ``record`` describes, if it asked for webhooks."""
        url = record.get("webhook_url")
        if not url:
            return
        job_id, tenant = record["job_id"], record["tenant"]
        secret = self.settings.webhook_secret_for(tenant)
        if not secret:
            log.error("job %s: no webhook secret for tenant %s; the %s webhook is not sent", job_id, tenant, event)
            self.outcomes[(job_id, event)] = "no_secret"
            return
        delivery = uuid.uuid4().hex
        body = payload(record, event, self.settings.public_base_url, delivery_id=delivery)
        headers = {SIGNATURE_HEADER: sign(secret, body), EVENT_HEADER: f"extraction.{event}",
                   DELIVERY_HEADER: delivery}
        if self._pending is not None:
            self._pending.put((job_id, event, url, body, headers))
        else:
            self._deliver(job_id, event, url, body, headers)

    def drain(self, timeout_s: float = 30.0) -> None:
        """Wait until every queued delivery has been tried (shutdown, tests)."""
        if self._pending is None:
            return
        deadline = time.monotonic() + timeout_s
        while self._pending.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)

    def _run(self) -> None:
        assert self._pending is not None
        while True:
            item = self._pending.get()
            try:
                if item is not None:
                    self._deliver(*item)
            finally:
                self._pending.task_done()

    def _deliver(self, job_id: str, event: str, url: str, body: bytes, headers: dict[str, str]) -> bool:
        last = "not tried"
        for attempt, wait in enumerate(self.backoff_s, start=1):
            if wait:
                self.sleep(wait)
            try:
                status = self.post(url, body, headers)
            except WebhookUrlError as exc:
                last = f"refused: {exc}"
                break                                    # the address will not change between tries
            except Exception as exc:  # noqa: BLE001 - a receiver's failure is logged, never the job's
                last = type(exc).__name__
            else:
                if 200 <= status < 300:
                    self.outcomes[(job_id, event)] = "delivered"
                    log.info("job %s: %s webhook delivered (%d, try %d)", job_id, event, status, attempt)
                    return True
                last = f"HTTP {status}"
        self.outcomes[(job_id, event)] = f"failed: {last}"
        # The address is not logged: its query may hold the receiver's token.
        log.warning("job %s: the %s webhook was not delivered (%s)", job_id, event, last)
        return False
