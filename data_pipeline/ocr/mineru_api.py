"""MinerU behind its own HTTP API (MinerU 4's ``/v1/tasks``), as one markdown string per page.

This reads a MinerU someone already serves - on a pod, behind its RunPod proxy
URL or on localhost. Since 2026-10-23 it is the OCR of record for training and
serving alike (``configs/shared/ocr.yaml``: version and tier pinned;
``data_pipeline.ocr.served_ocr``): the corpus is read through it
(``ServedMinerUEngine``) and so is every served document (``ServedOcr``). A
corpus read in-process by MinerU 3.4.5 (``run_mineru.MinerUEngine``) is another
distribution - close text, not identical - and is read again before it trains.

**One page string per page, formatted as the OCR stage formats it**
(:func:`run_mineru.pages_from_content_list`): a header at the head of its page,
footers and page numbers at its foot, a title as ``#`` per level, a table as its
captions, body and footnotes, a picture by its captions only. MinerU 4 names the
pieces differently - ``structured_content`` pages of blocks with ``content`` and
``level`` - and gives tables as markdown where 3.x gave HTML.

**Every task is held to the pages it was sent.** A PDF read as several tasks
(``parts``, ``run_pages``, a subset of its pages) has as many pages as its runs
were sent, whatever MinerU returned, so a page one task left out would pass as
a blank page. Each task is checked on its own (:func:`missing_pages`) and
refused when it lacks one; each task's version and tier are reported
(``version_parts``, ``tier_parts``) for the pin check (``served_ocr``).

**A poll survives a hiccup; an upload is sent once.** Polling a task and
fetching its result are GETs, which ask again what they asked: a proxy's 5xx,
a dropped connection or a timeout is tried again a few times
(:data:`RETRY_WAITS_S`) before the PDF fails. The upload is a POST that may
have reached the server: sent again, it would start a second task. A server's
answer this client cannot read is a :class:`MinerUApiError` with the answer,
and a redirect is refused (``common.transport.open_url``).
"""

from __future__ import annotations

import http.client
import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

from common.transport import RedirectRefusedError, open_url
from data_pipeline.ocr.run_mineru import _PAGE_FOOT, _PAGE_HEAD

log = logging.getLogger(__name__)

#: Block types MinerU 4 gives a heading level.
TITLE_TYPES = ("doc_title", "paragraph_title", "title")
#: Block types whose picture is in the page image: only their captions are text.
PICTURE_TYPES = ("image", "chart", "figure")
#: The statuses a task ends in.
FINISHED = ("completed", "partial", "failed", "canceled")
#: Statuses MinerU 4 is not known to use but that can only mean a task ended
#: unread (another spelling, another server's word): such a task fails at once
#: instead of being polled to the timeout - an hour. Any other status this
#: client does not know is polled on: it may be a step of the reading not seen
#: named yet, and failing a document on it would fail every document.
FAILED_LIKE = ("cancelled", "error", "errored", "aborted", "rejected", "expired", "timeout", "timed_out",
               "killed")
#: Seconds waited before each new try of a GET that met a transient failure.
RETRY_WAITS_S = (1.0, 3.0, 8.0)
#: HTTP statuses tried again: the server, or a proxy before it, busy or briefly down.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 520, 521, 522, 523, 524})


class MinerUApiError(RuntimeError):
    """The served MinerU refused a document, failed it, or could not be reached."""


class _TransientError(MinerUApiError):
    """A failure another try of the same request may not meet (:func:`_get`)."""


def pages_from_structured_content(content: dict[str, Any], page_count: int | None = None) -> list[str]:
    """MinerU 4's ``structured_content`` as one markdown string per page."""
    pages = content.get("pages") or []
    count = page_count if page_count is not None else len(pages)
    head: list[list[str]] = [[] for _ in range(count)]
    body: list[list[str]] = [[] for _ in range(count)]
    foot: list[list[str]] = [[] for _ in range(count)]

    def texts(block: dict[str, Any], key: str) -> list[str]:
        # A caption or footnote comes as ``{"bbox": ..., "content": ...}``; a bare string is taken as is.
        value = block.get(key) or []
        items = value if isinstance(value, list) else [value]
        items = [item.get("content") if isinstance(item, dict) else item for item in items]
        return [str(item).strip() for item in items if item is not None and str(item).strip()]

    for page in pages:
        index = page.get("page_idx")
        if not isinstance(index, int) or not 0 <= index < count:
            continue
        for block in page.get("blocks") or []:
            kind = block.get("type")
            text = block.get("content")
            text = "\n".join(map(str, text)) if isinstance(text, list) else str(text or "").strip()
            parts: list[str] = []
            if kind in TITLE_TYPES:
                level = block.get("level") if isinstance(block.get("level"), int) else 1
                parts.append("#" * min(max(level, 1), 6) + " " + text if text else "")
            elif kind == "table":
                parts += texts(block, "captions") + [text] + texts(block, "footnotes")
            elif kind in PICTURE_TYPES:
                parts += texts(block, "captions") + texts(block, "footnotes")
            else:
                parts.append(text)
            target = head if kind in _PAGE_HEAD else foot if kind in _PAGE_FOOT else body
            target[index] += [part for part in parts if part]
    return ["\n\n".join(head[i] + body[i] + foot[i]) for i in range(count)]


def missing_pages(content: dict[str, Any], page_count: int | None = None) -> list[int]:
    """The pages (1-based) of the ``page_count`` a task was sent - of as many
    as ``content`` lists, when not given - that ``content`` has no entry for.

    :func:`pages_from_structured_content` reads such a page as blank. One task
    is caught by its page count (``ServedOcr`` compares it with the renderer's);
    a PDF read as several tasks is not - its count is the runs' counts added up
    - so each task is checked here, against the pages it was sent.
    """
    pages = content.get("pages") or []
    count = page_count if page_count is not None else len(pages)
    present = {page.get("page_idx") for page in pages if isinstance(page, dict)}
    return [index + 1 for index in range(count) if index not in present]


def _shown(value: Any) -> str:
    """A server's answer, short, for an error message."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")[:300]
    try:
        return json.dumps(value)[:300]
    except (TypeError, ValueError):
        return repr(value)[:300]


def _call(url: str, token: str | None, *, data: bytes | None = None, timeout: float = 120,
          content_type: str = "application/json") -> Any:
    """The JSON answer to ``url`` - a POST of ``data``, else a GET.

    Raises :class:`MinerUApiError` with the server's reason; a failure that
    another try may not meet (a 5xx, no connection, a timeout, an answer that
    is not JSON - a proxy's error page) as :class:`_TransientError`.
    """
    headers = {"User-Agent": "fideon-mineru-client/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    path = urllib.parse.urlparse(url).path
    try:
        with open_url(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        try:
            reason = exc.read()
        except (OSError, http.client.HTTPException):
            reason = b""
        error = _TransientError if exc.code in RETRY_STATUSES else MinerUApiError
        raise error(f"{exc.code} from {path}: {_shown(reason)}") from exc
    except RedirectRefusedError as exc:
        raise MinerUApiError(str(exc.reason)) from exc
    except urllib.error.URLError as exc:
        raise _TransientError(f"could not reach {url}: {exc.reason}") from exc
    except (OSError, http.client.HTTPException) as exc:
        # The answer broke off while it was read: a timeout, a dropped connection.
        raise _TransientError(f"the answer from {path} broke off: {type(exc).__name__}: {exc}") from exc
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as exc:
        raise _TransientError(f"{path} answered with something other than JSON: {_shown(body)}") from exc


def _get(url: str, token: str | None, *, what: str, timeout: float = 120,
         stop: threading.Event | None = None) -> Any:
    """:func:`_call`'s GET of ``url``, tried again after a transient failure
    (:data:`RETRY_WAITS_S`) - a status poll or a result fetch, which asks again
    what it asked - and abandoned when ``stop`` is set while it waits."""
    attempt = 0
    while True:
        try:
            return _call(url, token, timeout=timeout)
        except _TransientError as exc:
            if attempt >= len(RETRY_WAITS_S):
                raise MinerUApiError(f"{exc} (tried {attempt + 1} times)") from exc
            wait = RETRY_WAITS_S[attempt]
            attempt += 1
            log.warning("MinerU, %s: %s; trying again in %.0f s", what, exc, wait)
            _pause(wait, stop, what)


def _check_stop(stop: threading.Event | None, what: str) -> None:
    if stop is not None and stop.is_set():
        raise MinerUApiError(f"{what} abandoned: the PDF's reading was stopped (another run of its pages "
                             "failed, or its rendering did)")


def _pause(seconds: float, stop: threading.Event | None, what: str) -> None:
    """``seconds`` waited - cut short, raising, when ``stop`` is set meanwhile."""
    if stop is None:
        time.sleep(seconds)
    elif stop.wait(seconds):
        _check_stop(stop, what)


def select_pages(pdf: bytes, pages: list[int]) -> bytes:
    """A PDF of ``pdf``'s pages ``pages`` (1-based), in that order, and nothing else.

    The pages are copied into a new PDF, as :func:`split_pdf` copies its runs,
    so a filled form's fields come with them: ``Document.select`` dropped the
    AcroForm the fields' values are drawn from, and a field's value - the
    named insured typed into a form - was gone from the page MinerU read.
    """
    import pymupdf

    if not pages:
        raise MinerUApiError("no pages to read: select_pages was given none")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        missing = [page for page in pages if not 1 <= page <= doc.page_count]
        if missing:
            raise MinerUApiError(f"pages {missing} are not in a {doc.page_count}-page PDF")
        runs: list[list[int]] = []
        for page in pages:
            if runs and page == runs[-1][-1] + 1:
                runs[-1].append(page)
            else:
                runs.append([page])
        with pymupdf.open() as picked:
            for run in runs:
                picked.insert_pdf(doc, from_page=run[0] - 1, to_page=run[-1] - 1)
            return picked.tobytes()


def split_pdf(pdf: bytes, parts: int, *, run_pages: int | None = None) -> list[tuple[int, bytes]]:
    """``pdf`` as up to ``parts`` PDFs of consecutive pages, in order: ``(pages, bytes)`` each;
    with ``run_pages``, as many as it takes for none to hold more pages than that."""
    import pymupdf

    out: list[tuple[int, bytes]] = []
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        count = doc.page_count
        if run_pages:
            parts = max(parts, -(-count // run_pages))
        parts = max(1, min(parts, count))
        bounds = [count * part // parts for part in range(parts + 1)]
        for start, stop in zip(bounds, bounds[1:], strict=False):
            with pymupdf.open() as piece:
                piece.insert_pdf(doc, from_page=start, to_page=stop - 1)
                out.append((stop - start, piece.tobytes()))
    return out


def ocr_via_api(
    pdf: bytes, base_url: str, token: str | None = None, *, filename: str = "document.pdf",
    tier: str = "basic", ocr_mode: str = "auto", poll_s: float = 2.0, timeout_s: float = 3600,
    parts: int = 1, uploaded: threading.Event | None = None, gate: Callable[[], None] | None = None,
    run_pages: int | None = None, stop: threading.Event | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """``(one markdown string per page, what the server reports)`` for ``pdf``.

    ``tier`` picks the server's quality/speed trade-off (``basic``: its local
    lightweight models, the nearest to the pipeline backend the corpus used);
    ``ocr_mode`` ``auto`` reads a scan by OCR and a digital page from its text.

    ``parts`` above 1 sends the PDF as that many runs of consecutive pages, each
    a task of its own, and joins their pages in order. The runs are uploaded
    one after another, in page order, and each is polled from the moment it is
    in: the server reads the first pages while the rest are still on their way.
    Through a slow upload (a laptop behind the proxy: an 88 MB scan is 1.5
    minutes of it) the reading then starts that much sooner; on a fast one the
    gain is only where the server reads several tasks at once. Pages are read
    a run at a time, so a table running across a run's last page is not joined
    to the next run's: measure against one task before relying on it. The
    first run to fail fails the PDF at once, whichever run it is, and the
    others stop.

    ``uploaded``, when given, is set once the whole PDF is at the server (the
    caller can start sending something else then without slowing this upload).

    ``gate``, when given, is called before each run is sent, and the runs are
    then read one after another, each to its end: MinerU's GPU work comes in
    turns of one run that the caller can hold back. On one GPU shared with the
    model, the model's turn - a document's fields - comes first
    (``serving.fields_first.GpuTurns``). A turn is as long as its run: with
    ``run_pages`` no run holds more pages than that, however long the PDF, so
    a document arriving meanwhile waits for at most that many pages' reading.

    ``stop``, when set, abandons the reading: no run is sent and every run
    polling stops, raising (the caller's rendering failed, so nobody waits for
    the text). A failed run sets it for the others.

    Every task must return each page it was sent (:func:`missing_pages`); the
    report holds every run's version and tier (``version_parts``, ``tier_parts``).

    An encrypted PDF is sent unencrypted (``common.pdf_access.unlocked_pdf``):
    one that restricts printing or copying is re-saved without it - its runs
    already were (:func:`split_pdf`), a PDF sent whole was not - and one that
    needs a password to open is refused here, before any upload.
    """
    from common.pdf_access import UNLOCK_HINT, PdfLocked, unlocked_pdf

    try:
        pdf = unlocked_pdf(pdf)
    except PdfLocked as exc:
        raise PdfLocked(f"{filename}: {exc}; {UNLOCK_HINT}") from None
    except Exception:  # noqa: BLE001 - a PDF PyMuPDF cannot open goes to MinerU as before, to be judged there
        pass
    pieces = split_pdf(pdf, parts, run_pages=run_pages) if parts > 1 or run_pages else []
    if len(pieces) <= 1:
        _check_stop(stop, filename)
        if gate is not None:
            gate()
        return _ocr_task(pdf, base_url, token, filename=filename, tier=tier, ocr_mode=ocr_mode,
                         poll_s=poll_s, timeout_s=timeout_s, uploaded=uploaded, stop=stop,
                         page_count=pieces[0][0] if pieces else None)
    stem = filename.rsplit(".", 1)[0]
    if gate is not None:
        done = []
        for index, (count, piece) in enumerate(pieces):
            gate()
            name = f"{stem}.part{index + 1}.pdf"
            _check_stop(stop, name)
            task_id = _submit(piece, base_url, token, filename=name, tier=tier, ocr_mode=ocr_mode)
            done.append(_finish(task_id, base_url, token, poll_s=poll_s, timeout_s=timeout_s, page_count=count,
                                stop=stop))
        if uploaded is not None:
            uploaded.set()
        return _joined(done, overlapped=False)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stop = stop if stop is not None else threading.Event()
    with ThreadPoolExecutor(max_workers=len(pieces), thread_name_prefix="mineru-run") as pool:
        futures: list[Future] = []
        try:
            for index, (count, piece) in enumerate(pieces):
                _raise_failed(futures)                    # a run has failed: the rest need not go up
                name = f"{stem}.part{index + 1}.pdf"
                _check_stop(stop, name)
                # Here, one at a time: each run's upload has the whole connection.
                task_id = _submit(piece, base_url, token, filename=name, tier=tier, ocr_mode=ocr_mode)
                futures.append(pool.submit(_finish, task_id, base_url, token, poll_s=poll_s,
                                           timeout_s=timeout_s, page_count=count, stop=stop))
            if uploaded is not None:
                uploaded.set()
            # In the order they end, not in page order: a last run failing at
            # once fails the PDF now, not once the first has been read.
            for future in as_completed(futures):
                future.result()
            done = [future.result() for future in futures]
        except BaseException:
            stop.set()                                    # the other runs stop polling: the PDF has failed
            raise
    return _joined(done, overlapped=True)


def _raise_failed(futures: list[Future]) -> None:
    """Raise the failure of the first of ``futures`` already failed, if any."""
    for future in futures:
        if future.done():
            error = future.exception()
            if error is not None:
                raise error


def _one(values: list[Any]) -> Any:
    """The value every run reported, or the distinct ones joined (``"4.0.12 / 4.1.0"``)."""
    distinct = list(dict.fromkeys(values))
    return distinct[0] if len(distinct) == 1 else " / ".join(map(str, distinct))


def _joined(done: list[tuple[list[str], dict[str, Any]]], *, overlapped: bool) -> tuple[list[str], dict[str, Any]]:
    """The runs' pages in order, and what the server reported for them together."""
    pages = [page for part_pages, _info in done for page in part_pages]
    infos = [info for _pages, info in done]
    seconds = [info["server_seconds"] for info in infos]
    versions = [info["version"] for info in infos]
    tiers = [info["tier"] for info in infos]
    return pages, {
        "task_id": [info["task_id"] for info in infos], "pages": len(pages), "parts": len(done),
        # Overlapping runs: the slowest is the nearest to how long the server
        # took; runs one after another: their sum.
        "server_seconds": max(seconds) if overlapped else round(sum(seconds), 2),
        "server_seconds_parts": seconds,
        # Every run's own, so a run read by another MinerU is seen (served_ocr's
        # pin check), not hidden behind the first run's.
        "version": _one(versions), "tier": _one(tiers),
        "version_parts": versions, "tier_parts": tiers,
    }


def _ocr_task(
    pdf: bytes, base_url: str, token: str | None, *, filename: str, tier: str, ocr_mode: str,
    poll_s: float, timeout_s: float, page_count: int | None = None, uploaded: threading.Event | None = None,
    stop: threading.Event | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """One MinerU task: ``pdf`` posted, polled to its end, and its pages read."""
    task_id = _submit(pdf, base_url, token, filename=filename, tier=tier, ocr_mode=ocr_mode)
    if uploaded is not None:
        uploaded.set()
    return _finish(task_id, base_url, token, poll_s=poll_s, timeout_s=timeout_s, page_count=page_count,
                   stop=stop)


def _submit(pdf: bytes, base_url: str, token: str | None, *, filename: str, tier: str, ocr_mode: str) -> str:
    """``pdf`` posted as a task; its id once the server has the whole file. Sent once, never again."""
    query = urllib.parse.urlencode({"tier": tier, "ocr_mode": ocr_mode, "filename": filename,
                                    "output_formats": "structured_content"})
    task = _call(f"{base_url.rstrip('/')}/v1/tasks?{query}", token, data=pdf, timeout=900,
                 content_type="application/octet-stream")
    task_id = task.get("task_id") if isinstance(task, dict) else None
    if task_id is None or task_id == "":
        raise MinerUApiError(f"the server took {filename} but named no task_id: {_shown(task)}")
    return str(task_id)


def _reason(task: Any) -> str:
    """What a failed task says went wrong, wherever it says it."""
    if not isinstance(task, dict):
        return _shown(task)
    files = task.get("files")
    first = files[0] if isinstance(files, list) and files and isinstance(files[0], dict) else {}
    reason = first.get("error") or task.get("error") or task.get("message") or task.get("detail")
    return str(reason) if reason else f"no reason given ({_shown(task)})"


def _finish(
    task_id: str, base_url: str, token: str | None, *, poll_s: float, timeout_s: float,
    page_count: int | None = None, stop: threading.Event | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Task ``task_id`` polled to its end, and its pages read: every one of the
    ``page_count`` it was sent, or it is refused."""
    base = base_url.rstrip("/")
    what = f"task {task_id}"
    status = ""
    task: Any = {}
    deadline = time.monotonic() + timeout_s
    while status not in FINISHED and status not in FAILED_LIKE:
        _check_stop(stop, what)
        if time.monotonic() > deadline:
            raise MinerUApiError(f"{what} not finished after {timeout_s:.0f} s (last status {status or 'none'})")
        _pause(poll_s, stop, what)
        task = _get(f"{base}/v1/tasks/{task_id}", token, what=what, stop=stop)
        if not isinstance(task, dict) or not isinstance(task.get("status"), str):
            raise MinerUApiError(f"{what}: the server's answer names no status: {_shown(task)}")
        status = task["status"].strip().lower()
    if status != "completed":
        raise MinerUApiError(f"{what} {status}: {_reason(task)}")
    result = _get(f"{base}/v1/tasks/{task_id}/result?response_format=json", token, what=what, timeout=300,
                  stop=stop)
    files = result.get("files") if isinstance(result, dict) else None
    file = files[0] if isinstance(files, list) and files and isinstance(files[0], dict) else None
    if file is None:
        raise MinerUApiError(f"{what} completed, but its result holds no file: {_shown(result)}")
    content = file.get("content")
    content = content.get("structured_content") if isinstance(content, dict) else None
    if not isinstance(content, dict):
        raise MinerUApiError(f"{what} returned no structured_content")
    try:
        missing = missing_pages(content, page_count)
        pages = pages_from_structured_content(content, page_count)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MinerUApiError(f"{what} returned structured_content this client cannot read "
                             f"({type(exc).__name__}: {exc})") from exc
    if missing:
        raise MinerUApiError(
            f"{what} returned no page {', '.join(map(str, missing))} of the {len(pages)} it was sent: a page "
            "left out would be read as blank. Read the PDF again; if it recurs, MinerU drops that page."
        )
    parse = file.get("parse") if isinstance(file.get("parse"), dict) else {}
    duration = parse.get("duration_ms")
    return pages, {
        "task_id": task_id, "pages": len(pages),
        "server_seconds": round(duration / 1000, 2) if isinstance(duration, int | float) else 0.0,
        "version": parse.get("parser_version"), "tier": result.get("tier"),
    }
