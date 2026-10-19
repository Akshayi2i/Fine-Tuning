"""Azure Blob access — the only module that imports the Azure SDK.

Everything else reaches storage through here (master §7). Beyond wrapping the
SDK, this module enforces two access rules that the architecture treats as
security properties rather than conventions:

**The raw container is separate.** ``raw-documents/`` is the only layer holding
unredacted PII, so it lives in its own container with tighter RBAC. Reading it
from a training or serving context is *refused*, not merely discouraged — by the
time data reaches ``corpus/`` it has already been through OCR and labeling, so
nothing downstream has a legitimate reason to touch the originals (arch §18a).

**Raw documents are write-once.** Overwriting ``original.pdf`` raises. A
corrected document is ingested as a new ``source_id`` so historical training runs
stay reproducible against the exact bytes they trained on.

The :class:`InMemoryBackend` exists so the entire test suite runs with no Azure
account and no network. It is not a toy: it enforces the same write-once and
container-isolation rules, so a test that passes against it is testing the real
contract.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging as _logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from artifact_registry.paths import requires_raw_container

# The Azure SDK logs every HTTP request and response (headers, request ids) at
# INFO. With the pipeline's INFO logging that buried a run's own progress under
# thousands of lines per upload; its warnings and errors still come through.
_logging.getLogger("azure").setLevel(_logging.WARNING)

log = _logging.getLogger(__name__)


def file_md5(path: str | Path, chunk: int = 8 << 20) -> bytes:
    """A local file's MD5, read in chunks: model shards run to gigabytes."""
    digest = hashlib.md5()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.digest()

#: Contexts permitted to touch ``raw-documents/``. Anything else is refused.
RAW_ALLOWED_CONTEXTS = frozenset({"ingestion", "ocr"})


class BlobError(RuntimeError):
    """Raised on auth failure, a missing container, or an access-rule violation."""


class ImmutableBlobError(BlobError):
    """Raised on an attempt to overwrite a write-once artifact."""


class AccessDeniedError(BlobError):
    """Raised when a context reaches for a layer it has no business reading."""


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class BlobBackend(ABC):
    """Minimal storage surface. Two implementations: Azure, and in-memory."""

    @abstractmethod
    def read(self, container: str, key: str) -> bytes: ...

    @abstractmethod
    def write(self, container: str, key: str, data: bytes) -> None: ...

    @abstractmethod
    def exists(self, container: str, key: str) -> bool: ...

    @abstractmethod
    def list(self, container: str, prefix: str) -> Iterator[str]: ...

    @abstractmethod
    def delete(self, container: str, key: str) -> None: ...

    def md5(self, container: str, key: str) -> bytes | None:
        """The blob's recorded MD5, or ``None`` when it is missing or records
        none. A backend that cannot tell says ``None``: then nothing is skipped."""
        return None


class InMemoryBackend(BlobBackend):
    """Dict-backed backend for tests. Same rules as the real one."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], bytes] = {}

    def read(self, container: str, key: str) -> bytes:
        try:
            return self._store[(container, key)]
        except KeyError:
            raise BlobError(f"blob not found: {container}/{key}") from None

    def write(self, container: str, key: str, data: bytes) -> None:
        self._store[(container, key)] = data

    def exists(self, container: str, key: str) -> bool:
        return (container, key) in self._store

    def list(self, container: str, prefix: str) -> Iterator[str]:
        prefix = prefix.strip("/")
        for (c, k) in sorted(self._store):
            if c == container and k.startswith(prefix):
                yield k

    def delete(self, container: str, key: str) -> None:
        self._store.pop((container, key), None)

    def md5(self, container: str, key: str) -> bytes | None:
        data = self._store.get((container, key))
        return None if data is None else hashlib.md5(data).digest()


class AzureBackend(BlobBackend):
    """Real Azure Blob Storage. Imported lazily so CI needs no Azure SDK."""

    def __init__(
        self, connection_string: str | None = None, *,
        account_url: str | None = None, credential: str | None = None,
    ) -> None:
        """A connection string (it carries the account key), or an account URL
        with a SAS token as ``credential`` - short-lived and scoped by its
        permissions, so a leaked pod ``.env`` exposes less."""
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise BlobError(
                "azure-storage-blob is not installed. Install the [data] extra: "
                'pip install -e ".[data]"'
            ) from exc
        if account_url:
            self._service = BlobServiceClient(account_url=account_url, credential=credential)
        elif connection_string:
            self._service = BlobServiceClient.from_connection_string(connection_string)
        else:
            raise BlobError("AzureBackend needs a connection string or an account URL")

    def _client(self, container: str, key: str):
        return self._service.get_blob_client(container=container, blob=key)

    def read(self, container: str, key: str) -> bytes:
        try:
            return self._client(container, key).download_blob().readall()
        except Exception as exc:
            raise BlobError(f"failed reading {container}/{key}: {exc}") from exc

    def write(self, container: str, key: str, data: bytes) -> None:
        from azure.storage.blob import ContentSettings

        try:
            # The MD5 recorded with the blob: Azure stores one itself only for a
            # single-shot upload, and a multi-gigabyte model shard goes up in
            # blocks - without it a later push could not tell it is already there.
            self._client(container, key).upload_blob(
                io.BytesIO(data), overwrite=True,
                content_settings=ContentSettings(content_md5=bytearray(hashlib.md5(data).digest())),
            )
        except Exception as exc:
            raise BlobError(f"failed writing {container}/{key}: {exc}") from exc

    def md5(self, container: str, key: str) -> bytes | None:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            recorded = self._client(container, key).get_blob_properties().content_settings.content_md5
        except ResourceNotFoundError:
            return None
        except Exception as exc:
            raise BlobError(f"failed reading the properties of {container}/{key}: {exc}") from exc
        return bytes(recorded) if recorded else None

    def exists(self, container: str, key: str) -> bool:
        try:
            return bool(self._client(container, key).exists())
        except Exception as exc:
            raise BlobError(f"failed checking {container}/{key}: {exc}") from exc

    def list(self, container: str, prefix: str) -> Iterator[str]:
        try:
            container_client = self._service.get_container_client(container)
            for blob in container_client.list_blobs(name_starts_with=prefix.strip("/")):
                yield blob.name
        except Exception as exc:
            raise BlobError(f"failed listing {container}/{prefix}: {exc}") from exc

    def delete(self, container: str, key: str) -> None:
        try:
            self._client(container, key).delete_blob()
        except Exception as exc:
            raise BlobError(f"failed deleting {container}/{key}: {exc}") from exc


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------

class BlobClient:
    """Typed access to the two containers, with the access rules enforced.

    Args:
        backend: storage backend; defaults to Azure from the environment.
        container: the main container.
        raw_container: the separately-permissioned container for
            ``raw-documents/``. Defaults to ``container`` only when unset, which
            is a valid single-container deployment but loses the RBAC split.
        context: what is doing the reading — ``ingestion`` and ``ocr`` may reach
            ``raw-documents/``; everything else is refused.
        max_retries: transient-failure retries, with exponential backoff.
    """

    def __init__(
        self,
        backend: BlobBackend | None = None,
        container: str | None = None,
        raw_container: str | None = None,
        context: str = "general",
        max_retries: int = 3,
    ) -> None:
        self.container = container or os.environ.get("AZURE_BLOB_CONTAINER") or "insurance-extraction"
        self.raw_container = raw_container or os.environ.get("AZURE_RAW_CONTAINER") or self.container
        self.context = context
        self.max_retries = max_retries

        if backend is not None:
            self._backend = backend
        else:
            self._backend = azure_from_environment()

    # -- routing and guards -------------------------------------------------

    def _container_for(self, key: str) -> str:
        """Route to the raw container, refusing contexts that may not read it."""
        if requires_raw_container(key):
            if self.context not in RAW_ALLOWED_CONTEXTS:
                raise AccessDeniedError(
                    f"context {self.context!r} may not access {key!r}. raw-documents/ holds "
                    f"unredacted PII and is reachable only from {sorted(RAW_ALLOWED_CONTEXTS)}. "
                    "By the time data reaches corpus/ it has already been through OCR and "
                    "labeling, so training and serving have no reason to read originals (arch §18a)."
                )
            return self.raw_container
        return self.container

    @staticmethod
    def _is_write_once(key: str) -> bool:
        return requires_raw_container(key) and key.endswith("original.pdf")

    def _retry(self, op, *args):
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return op(*args)
            except (ImmutableBlobError, AccessDeniedError):
                raise  # never retry a rule violation
            except BlobError as exc:
                last = exc
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt * 0.1)
        assert last is not None  # the loop cannot exit without one
        raise last

    # -- operations ---------------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._backend.exists(self._container_for(key), key)

    def read_bytes(self, key: str) -> bytes:
        return self._retry(self._backend.read, self._container_for(key), key)

    def write_bytes(self, key: str, data: bytes, *, allow_overwrite: bool = True) -> None:
        container = self._container_for(key)
        if self._is_write_once(key) and self._backend.exists(container, key):
            raise ImmutableBlobError(
                f"{key} already exists and raw documents are write-once (arch §18a). "
                "Ingest a corrected document as a NEW source_id — overwriting would make every "
                "historical training run irreproducible against the bytes it actually trained on."
            )
        if not allow_overwrite and self._backend.exists(container, key):
            raise BlobError(f"{key} already exists and allow_overwrite=False")
        self._retry(self._backend.write, container, key, data)

    def read_text(self, key: str) -> str:
        return self.read_bytes(key).decode("utf-8")

    def write_text(self, key: str, text: str, **kwargs: Any) -> None:
        self.write_bytes(key, text.encode("utf-8"), **kwargs)

    def read_json(self, key: str) -> Any:
        return json.loads(self.read_text(key))

    def write_json(self, key: str, obj: Any, **kwargs: Any) -> None:
        self.write_text(key, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False), **kwargs)

    def list(self, prefix: str) -> list[str]:
        return list(self._backend.list(self._container_for(prefix), prefix))

    def delete(self, key: str) -> None:
        if self._is_write_once(key):
            raise ImmutableBlobError(
                f"{key} is write-once. Deleting a raw document is a compliance action (a client "
                "purge request), not a routine operation — it must also cascade to processed/, "
                "golden-labels/ and any corpus entries derived from it (arch §18a)."
            )
        self._retry(self._backend.delete, self._container_for(key), key)

    # -- file and directory transfer ---------------------------------------

    def upload_file(self, local: str | Path, key: str, **kwargs: Any) -> None:
        self.write_bytes(key, Path(local).read_bytes(), **kwargs)

    def download_file(self, key: str, local: str | Path) -> None:
        path = Path(local)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.read_bytes(key))

    def holds(self, local: str | Path, key: str) -> bool:
        """Whether ``key`` already holds exactly the bytes of ``local``: by the
        MD5 recorded with the blob. A blob that records none does not count."""
        # Retried as reads and writes are: one throttled properties call must not
        # fail an 18 GB push.
        recorded = self._retry(self._backend.md5, self._container_for(key), key)
        return recorded is not None and recorded == file_md5(local)

    def upload_dir(
        self, local_dir: str | Path, prefix: str, *, names: Iterable[str] | None = None,
        skip_identical: bool = False, **kwargs: Any,
    ) -> int:
        """Upload a folder under ``prefix``; returns the files written.

        ``names``: only these paths, relative to the folder (each must exist).
        ``skip_identical``: leave a blob that already holds the file's bytes
        (:meth:`holds`) - a re-push of the same model sends nothing.
        """
        root = Path(local_dir)
        if not root.is_dir():
            raise BlobError(f"not a directory: {root}")
        if names is None:
            files = sorted(p for p in root.rglob("*") if p.is_file())
        else:
            files = [root / name for name in names]
            missing = [str(p) for p in files if not p.is_file()]
            if missing:
                raise BlobError(f"cannot upload {missing}: no such file")
        count = skipped = 0
        for path in files:
            key = f"{prefix.rstrip('/')}/{path.relative_to(root).as_posix()}"
            if skip_identical and self.holds(path, key):
                skipped += 1
                continue
            self.write_bytes(key, path.read_bytes(), **kwargs)
            count += 1
        if skipped:
            log.info("%s: %d file(s) already held their bytes, %d written", prefix, skipped, count)
        return count

    def download_dir(self, prefix: str, local_dir: str | Path) -> int:
        root = Path(local_dir)
        prefix = prefix.rstrip("/")
        count = 0
        # The boundary matters: `corpus/default/v1` is a prefix of
        # `corpus/default/v10`, so without it a v1 download also pulled v10 in
        # and wrote it to a mangled path built from the leftover "0".
        boundary = prefix.rstrip("/") + "/"
        for key in self.list(prefix):
            if not key.startswith(boundary):
                continue
            rel = key[len(boundary):].lstrip("/")
            self.download_file(key, root / rel)
            count += 1
        return count


def azure_from_environment() -> AzureBackend:
    """The Azure backend the environment logs in to.

    A SAS token (``AZURE_STORAGE_ACCOUNT_URL`` + ``AZURE_STORAGE_SAS_TOKEN``) when
    set - preferred, because the connection string carries the account key, which
    reaches everything for ever - and the connection string otherwise. The token
    is never logged or put in an error message.
    """
    url = (os.environ.get("AZURE_STORAGE_ACCOUNT_URL") or "").strip()
    sas = (os.environ.get("AZURE_STORAGE_SAS_TOKEN") or "").strip().lstrip("?")
    if url or sas:
        if not (url and sas):
            missing = "AZURE_STORAGE_SAS_TOKEN" if url else "AZURE_STORAGE_ACCOUNT_URL"
            raise BlobError(
                f"{missing} is not set: a SAS login needs the account URL and the token both. "
                "Half of one is refused rather than falling back to the connection string."
            )
        return AzureBackend(account_url=url, credential=sas)
    conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise BlobError(
            "no storage login: set AZURE_STORAGE_ACCOUNT_URL and AZURE_STORAGE_SAS_TOKEN "
            "(preferred), or AZURE_STORAGE_CONNECTION_STRING. For tests, pass InMemoryBackend() "
            "explicitly rather than relying on a live account."
        )
    return AzureBackend(conn)


def for_ingestion(**kwargs: Any) -> BlobClient:
    """A client permitted to write ``raw-documents/``."""
    return BlobClient(context="ingestion", **kwargs)


def for_ocr(**kwargs: Any) -> BlobClient:
    """A client permitted to read ``raw-documents/`` and write ``processed/``."""
    return BlobClient(context="ocr", **kwargs)
