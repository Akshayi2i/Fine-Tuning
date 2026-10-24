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
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from data_pipeline.ocr.run_mineru import _PAGE_FOOT, _PAGE_HEAD

#: Block types MinerU 4 gives a heading level.
TITLE_TYPES = ("doc_title", "paragraph_title", "title")
#: Block types whose picture is in the page image: only their captions are text.
PICTURE_TYPES = ("image", "chart", "figure")
#: The statuses a task ends in.
FINISHED = ("completed", "partial", "failed", "canceled")


class MinerUApiError(RuntimeError):
    """The served MinerU refused a document, failed it, or could not be reached."""


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


def _call(url: str, token: str | None, *, data: bytes | None = None, timeout: float = 120,
          content_type: str = "application/json") -> Any:
    headers = {"User-Agent": "fideon-mineru-client/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise MinerUApiError(f"{exc.code} from {urllib.parse.urlparse(url).path}: "
                             f"{exc.read().decode('utf-8', 'replace')[:300]}") from exc
    except urllib.error.URLError as exc:
        raise MinerUApiError(f"could not reach {url}: {exc.reason}") from exc


def select_pages(pdf: bytes, pages: list[int]) -> bytes:
    """A PDF of ``pdf``'s pages ``pages`` (1-based), in that order, and nothing else."""
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        missing = [page for page in pages if not 1 <= page <= doc.page_count]
        if missing:
            raise MinerUApiError(f"pages {missing} are not in a {doc.page_count}-page PDF")
        doc.select([page - 1 for page in pages])
        return doc.tobytes()


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
    run_pages: int | None = None,
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
    to the next run's: measure against one task before relying on it.

    ``uploaded``, when given, is set once the whole PDF is at the server (the
    caller can start sending something else then without slowing this upload).

    ``gate``, when given, is called before each run is sent, and the runs are
    then read one after another, each to its end: MinerU's GPU work comes in
    turns of one run that the caller can hold back. On one GPU shared with the
    model, the model's turn - a document's fields - comes first
    (``serving.fields_first.GpuTurns``). A turn is as long as its run: with
    ``run_pages`` no run holds more pages than that, however long the PDF, so
    a document arriving meanwhile waits for at most that many pages' reading.
    """
    pieces = split_pdf(pdf, parts, run_pages=run_pages) if parts > 1 or run_pages else []
    if len(pieces) <= 1:
        if gate is not None:
            gate()
        return _ocr_task(pdf, base_url, token, filename=filename, tier=tier, ocr_mode=ocr_mode,
                         poll_s=poll_s, timeout_s=timeout_s, uploaded=uploaded)
    stem = filename.rsplit(".", 1)[0]
    if gate is not None:
        done = []
        for index, (count, piece) in enumerate(pieces):
            gate()
            task_id = _submit(piece, base_url, token, filename=f"{stem}.part{index + 1}.pdf", tier=tier,
                              ocr_mode=ocr_mode)
            done.append(_finish(task_id, base_url, token, poll_s=poll_s, timeout_s=timeout_s, page_count=count))
        if uploaded is not None:
            uploaded.set()
        return _joined(done, overlapped=False)
    from concurrent.futures import ThreadPoolExecutor

    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=len(pieces), thread_name_prefix="mineru-run") as pool:
        try:
            futures = []
            for index, (count, piece) in enumerate(pieces):
                # Here, one at a time: each run's upload has the whole connection.
                task_id = _submit(piece, base_url, token, filename=f"{stem}.part{index + 1}.pdf", tier=tier,
                                  ocr_mode=ocr_mode)
                futures.append(pool.submit(_finish, task_id, base_url, token, poll_s=poll_s,
                                           timeout_s=timeout_s, page_count=count, stop=stop))
            if uploaded is not None:
                uploaded.set()
            done = [future.result() for future in futures]
        except BaseException:
            stop.set()                                    # the other runs stop polling: the PDF has failed
            raise
    return _joined(done, overlapped=True)


def _joined(done: list[tuple[list[str], dict[str, Any]]], *, overlapped: bool) -> tuple[list[str], dict[str, Any]]:
    """The runs' pages in order, and what the server reported for them together."""
    pages = [page for part_pages, _info in done for page in part_pages]
    infos = [info for _pages, info in done]
    seconds = [info["server_seconds"] for info in infos]
    return pages, {
        "task_id": [info["task_id"] for info in infos], "pages": len(pages), "parts": len(done),
        # Overlapping runs: the slowest is the nearest to how long the server
        # took; runs one after another: their sum.
        "server_seconds": max(seconds) if overlapped else round(sum(seconds), 2),
        "server_seconds_parts": seconds,
        "version": infos[0]["version"], "tier": infos[0]["tier"],
    }


def _ocr_task(
    pdf: bytes, base_url: str, token: str | None, *, filename: str, tier: str, ocr_mode: str,
    poll_s: float, timeout_s: float, page_count: int | None = None, uploaded: threading.Event | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """One MinerU task: ``pdf`` posted, polled to its end, and its pages read."""
    task_id = _submit(pdf, base_url, token, filename=filename, tier=tier, ocr_mode=ocr_mode)
    if uploaded is not None:
        uploaded.set()
    return _finish(task_id, base_url, token, poll_s=poll_s, timeout_s=timeout_s, page_count=page_count)


def _submit(pdf: bytes, base_url: str, token: str | None, *, filename: str, tier: str, ocr_mode: str) -> str:
    """``pdf`` posted as a task; its id once the server has the whole file."""
    query = urllib.parse.urlencode({"tier": tier, "ocr_mode": ocr_mode, "filename": filename,
                                    "output_formats": "structured_content"})
    task = _call(f"{base_url.rstrip('/')}/v1/tasks?{query}", token, data=pdf, timeout=900,
                 content_type="application/octet-stream")
    return str(task["task_id"])


def _finish(
    task_id: str, base_url: str, token: str | None, *, poll_s: float, timeout_s: float,
    page_count: int | None = None, stop: threading.Event | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Task ``task_id`` polled to its end, and its pages read."""
    base = base_url.rstrip("/")
    task: dict[str, Any] = {"task_id": task_id}
    deadline = time.monotonic() + timeout_s
    while task.get("status") not in FINISHED:
        if stop is not None and stop.is_set():
            raise MinerUApiError(f"task {task_id} abandoned: another run of the PDF's pages failed")
        if time.monotonic() > deadline:
            raise MinerUApiError(f"task {task_id} not finished after {timeout_s:.0f} s")
        time.sleep(poll_s)
        task = _call(f"{base}/v1/tasks/{task_id}", token)
    if task["status"] != "completed":
        raise MinerUApiError(f"task {task_id} {task['status']}: {task.get('files', [{}])[0].get('error')}")
    result = _call(f"{base}/v1/tasks/{task_id}/result?response_format=json", token, timeout=300)
    file = result["files"][0]
    content = (file.get("content") or {}).get("structured_content")
    if not isinstance(content, dict):
        raise MinerUApiError(f"task {task_id} returned no structured_content")
    parse = file.get("parse") or {}
    pages = pages_from_structured_content(content, page_count)
    return pages, {
        "task_id": task_id, "pages": len(pages), "server_seconds": round((parse.get("duration_ms") or 0) / 1000, 2),
        "version": parse.get("parser_version"), "tier": result.get("tier"),
    }
