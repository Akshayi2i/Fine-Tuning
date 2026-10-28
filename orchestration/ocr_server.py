"""The OCR server, run by the pipeline for the OCR stage and no longer.

Documents are read by the served MinerU the corpus is pinned to
(``configs/shared/ocr.yaml``, :mod:`data_pipeline.ocr.served_ocr`). On the training
pod that server is this machine's own (``scripts/mineru_server.sh``, installed by
``scripts/pod_bootstrap.sh``), and it shares the GPU with training, so the pipeline

* starts it when documents still need reading and none is answering,
* waits until it answers its health check,
* reads the pages through it, and
* stops it afterwards, before anything else wants the GPU.

A corpus already read starts nothing. A server somebody else runs (a ``MINERU_URL``
that is not this machine) is used as it is and never stopped, and so is a local one
that was already up. Nothing is started where there is nothing to start: off the pod,
or with no OCR environment installed, the stage reads as it always did and refuses
clearly when it cannot.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "mineru_server.sh"
#: How long a server may take to load its models and answer, once started.
START_TIMEOUT_S = 900.0
POLL_S = 5.0
LOOPBACK = ("localhost", "127.0.0.1", "::1")
#: Parse jobs the server runs at once (scripts/mineru_server.sh, MINERU_CONCURRENCY): the
#: documents the OCR stage reads in parallel against it.
SERVER_CONCURRENCY = 4


class OcrServerError(RuntimeError):
    """The OCR server could not be started, or never answered."""


def _env_file_values(path: Path) -> dict[str, str]:
    """``MINERU_URL`` and ``MINERU_TOKEN`` as ``.env`` holds them (what ``start`` writes)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key in ("MINERU_URL", "MINERU_TOKEN"):
            values[key] = value.strip().strip('"').strip("'")
    return values


def is_local(url: str) -> bool:
    """Whether ``url`` is this machine's."""
    return (urlsplit(url).hostname or "").lower() in LOOPBACK


def healthy(url: str, token: str | None, *, timeout: float = 10.0) -> bool:
    """Whether the server at ``url`` answers its health check."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        request = urllib.request.Request(f"{url.rstrip('/')}/v1/health", headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001 - not answering is the answer
        return False


def installed() -> bool:
    """Whether this machine has the OCR server installed (``scripts/pod_bootstrap.sh``)."""
    venv = Path(os.environ.get("MINERU_VENV")
                or Path(os.environ.get("FIDEON_WORKSPACE") or "/workspace") / "venv-mineru4")
    return SCRIPT.is_file() and (venv / "bin" / "mineru-api").is_file()


def _run_script(action: str, runner: Callable[..., Any]) -> None:
    result = runner(["bash", str(SCRIPT), action], cwd=REPO, capture_output=True, text=True)
    if getattr(result, "returncode", 0) != 0:
        raise OcrServerError(
            f"scripts/mineru_server.sh {action} failed: "
            f"{(getattr(result, 'stderr', '') or getattr(result, 'stdout', '') or '').strip()[-400:]}")


@contextmanager
def managed_ocr_server(
    pending: bool, *, runner: Callable[..., Any] | None = None,
    probe: Callable[..., bool] | None = None, sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic, can_start: Callable[[], bool] | None = None,
    env_file: Path | None = None, timeout_s: float = START_TIMEOUT_S,
) -> Iterator[Any]:
    """Run the OCR server for the length of the ``with`` block, when it is needed.

    Yields the engine (:class:`~data_pipeline.ocr.served_ocr.ServedMinerUEngine`) that reads
    through the server this block started, or ``None`` when the caller's own engine stands:
    nothing to read, somebody else's server, or one that was already up.
    ``MINERU_URL`` and ``MINERU_TOKEN`` are set for the block and put back after it.
    """
    from data_pipeline.ocr.served_ocr import ServedMinerUEngine

    # Looked up when called, not when defined, so a replacement of any of them takes effect.
    runner = runner or subprocess.run
    probe = probe or healthy
    can_start = can_start or installed
    url = (os.environ.get("MINERU_URL") or "").strip()
    if not pending or (url and not is_local(url)):
        yield None                       # nothing to read, or a server that is not ours to run
        return
    token = os.environ.get("MINERU_TOKEN") or os.environ.get("FIDEON_REMOTE_TOKEN") or None
    if url and probe(url, token):
        yield None                       # ours, already up: used as it is, left running
        return
    if not can_start():
        yield None                       # nothing to start here: the stage refuses as it always did
        return

    log.info("documents need OCR: starting the OCR server (scripts/mineru_server.sh start)")
    saved = {key: os.environ.get(key) for key in ("MINERU_URL", "MINERU_TOKEN")}
    started = False
    try:
        started = True
        _run_script("start", runner)
        # `start` writes the address and a token it made to .env; this process read .env
        # before they were there.
        written = _env_file_values(env_file or REPO / ".env")
        url, token = written.get("MINERU_URL", url), written.get("MINERU_TOKEN", token or "")
        if not url:
            raise OcrServerError("the OCR server started, but wrote no MINERU_URL to .env")
        os.environ["MINERU_URL"], os.environ["MINERU_TOKEN"] = url, token or ""
        deadline = clock() + timeout_s
        while not probe(url, token or None):
            if clock() > deadline:
                raise OcrServerError(
                    f"the OCR server did not answer {url}/v1/health within {int(timeout_s)} s; "
                    "read it with: bash scripts/pod_run.sh tail mineru-server")
            sleep(POLL_S)
        log.info("the OCR server answers at %s", url)
        yield ServedMinerUEngine(url, token or None)
    finally:
        if started:
            log.info("stopping the OCR server: training needs the GPU")
            try:
                _run_script("stop", runner)
            except OcrServerError as exc:
                log.warning("%s", exc)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
