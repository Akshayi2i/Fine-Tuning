"""The job queue: Azure Storage Queue in production, an in-memory one in tests.

A message holds a job's id and tenant, never the document: ``{"job_id": ...,
"tenant": ...}``. The queue's at-least-once semantics are what keep a job
alive across a crash:

* :meth:`JobQueue.receive` hides the message for ``visibility_timeout_s`` and
  counts the delivery (``dequeue_count``);
* while the worker reads the document it renews that lease
  (:meth:`JobQueue.extend`, which also hands back a new receipt);
* only a job that ended - done, or failed for good - is deleted
  (:meth:`JobQueue.delete`). A worker that dies leaves the message to come back
  once its lease runs out, to the next worker, with ``dequeue_count`` one higher.

A receipt that no longer matches - another worker received the message after
this one's lease ran out - raises :class:`LeaseLostError`: the job is that
worker's now.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any


class QueueError(RuntimeError):
    """The queue could not be reached or answered with an error."""


class LeaseLostError(QueueError):
    """The message's receipt is stale: another delivery owns the job now."""


@dataclass(frozen=True)
class QueueMessage:
    id: str
    receipt: str
    body: str
    #: How many times it has been delivered, this delivery included.
    dequeue_count: int

    def job(self) -> dict[str, str]:
        """The job this message names; a malformed one raises ``ValueError``."""
        data = json.loads(self.body)
        if not isinstance(data, dict) or not isinstance(data.get("job_id"), str) \
                or not isinstance(data.get("tenant"), str):
            raise ValueError("a queue message must be {\"job_id\": ..., \"tenant\": ...}")
        return {"job_id": data["job_id"], "tenant": data["tenant"]}


def job_message(job_id: str, tenant: str) -> str:
    return json.dumps({"job_id": job_id, "tenant": tenant})


class JobQueue(ABC):
    """The four operations the service needs of a queue."""

    @abstractmethod
    def send(self, body: str) -> None: ...

    @abstractmethod
    def receive(self, visibility_timeout_s: int) -> QueueMessage | None:
        """The next visible message, hidden for ``visibility_timeout_s``; None when there is none."""

    @abstractmethod
    def extend(self, message: QueueMessage, visibility_timeout_s: int) -> QueueMessage:
        """Hide ``message`` for ``visibility_timeout_s`` more seconds from now (0: visible
        at once). Returns it with its new receipt; :class:`LeaseLostError` when stale."""

    @abstractmethod
    def delete(self, message: QueueMessage) -> None:
        """Remove ``message`` for good; :class:`LeaseLostError` when its receipt is stale."""


class InMemoryQueue(JobQueue):
    """Azure Storage Queue's semantics in a dict: visibility, receipts, delivery counts."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        #: id -> [body, visible_at, dequeue_count, receipt, inserted]
        self._messages: dict[str, list[Any]] = {}
        self._order = 0

    def send(self, body: str) -> None:
        with self._lock:
            self._order += 1
            self._messages[uuid.uuid4().hex] = [body, self.clock(), 0, None, self._order]

    def receive(self, visibility_timeout_s: int) -> QueueMessage | None:
        with self._lock:
            now = self.clock()
            visible = sorted((m[4], key) for key, m in self._messages.items() if m[1] <= now)
            if not visible:
                return None
            key = visible[0][1]
            entry = self._messages[key]
            entry[1] = now + visibility_timeout_s
            entry[2] += 1
            entry[3] = uuid.uuid4().hex
            return QueueMessage(id=key, receipt=entry[3], body=entry[0], dequeue_count=entry[2])

    def _owned(self, message: QueueMessage) -> list[Any]:
        entry = self._messages.get(message.id)
        if entry is None or entry[3] != message.receipt:
            raise LeaseLostError(f"message {message.id} was received again or deleted")
        return entry

    def extend(self, message: QueueMessage, visibility_timeout_s: int) -> QueueMessage:
        with self._lock:
            entry = self._owned(message)
            entry[1] = self.clock() + visibility_timeout_s
            entry[3] = uuid.uuid4().hex
            return replace(message, receipt=entry[3])

    def delete(self, message: QueueMessage) -> None:
        with self._lock:
            self._owned(message)
            del self._messages[message.id]

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)


class AzureStorageQueue(JobQueue):
    """An Azure Storage Queue (``azure-storage-queue``, the ``service`` extra).

    Messages are base64-encoded text, as the portal and ``az storage message``
    show them. Logged in with a connection string, an account URL and a SAS, or
    the container's managed identity (``credential``: an azure-identity
    credential object).
    """

    def __init__(self, queue_name: str, *, connection_string: str | None = None,
                 account_url: str | None = None, credential: Any = None) -> None:
        try:
            from azure.storage.queue import QueueClient, TextBase64DecodePolicy, TextBase64EncodePolicy
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise QueueError('azure-storage-queue is not installed: pip install -e ".[service]"') from exc
        policies = {"message_encode_policy": TextBase64EncodePolicy(),
                    "message_decode_policy": TextBase64DecodePolicy()}
        if connection_string:
            self._client = QueueClient.from_connection_string(connection_string, queue_name, **policies)
        elif account_url:
            self._client = QueueClient(account_url, queue_name, credential=credential, **policies)
        else:
            raise QueueError("the queue needs a connection string or an account URL")

    @staticmethod
    def _message(item: Any) -> QueueMessage:
        return QueueMessage(id=item.id, receipt=item.pop_receipt, body=item.content,
                            dequeue_count=int(item.dequeue_count or 1))

    def send(self, body: str) -> None:
        try:
            # Never expires (the default is 7 days): a job waits for a worker, however long it is down;
            # the delivery count, not the age, ends a job that cannot be read.
            self._client.send_message(body, time_to_live=-1)
        except Exception as exc:  # noqa: BLE001 - named, never with the message's content
            raise QueueError(f"could not enqueue a job: {type(exc).__name__}") from exc

    def receive(self, visibility_timeout_s: int) -> QueueMessage | None:
        try:
            item = self._client.receive_message(visibility_timeout=visibility_timeout_s)
        except Exception as exc:  # noqa: BLE001
            raise QueueError(f"could not receive from the queue: {type(exc).__name__}: {exc}") from exc
        return None if item is None else self._message(item)

    def extend(self, message: QueueMessage, visibility_timeout_s: int) -> QueueMessage:
        from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

        try:
            updated = self._client.update_message(message.id, pop_receipt=message.receipt,
                                                  visibility_timeout=visibility_timeout_s)
        except ResourceNotFoundError as exc:
            raise LeaseLostError(f"message {message.id} was received again or deleted") from exc
        except HttpResponseError as exc:
            if exc.status_code in (400, 404):         # a pop receipt that no longer matches
                raise LeaseLostError(f"message {message.id}: {exc.reason}") from exc
            raise QueueError(f"could not renew message {message.id}: {exc.status_code} {exc.reason}") from exc
        return replace(message, receipt=updated.pop_receipt)

    def delete(self, message: QueueMessage) -> None:
        from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

        try:
            self._client.delete_message(message.id, message.receipt)
        except ResourceNotFoundError as exc:
            raise LeaseLostError(f"message {message.id} was received again or deleted") from exc
        except HttpResponseError as exc:
            if exc.status_code in (400, 404):
                raise LeaseLostError(f"message {message.id}: {exc.reason}") from exc
            raise QueueError(f"could not delete message {message.id}: {exc.status_code} {exc.reason}") from exc
