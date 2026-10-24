"""Fields first, full text second: one H100 for the model and MinerU (decision 2026-10-24).

The production gate is a 200-page scan extracted in 2 minutes, on ONE GPU that
serves both the model and MinerU. MinerU alone reads such a scan in 4-5
minutes (1.0-1.5 s a page), so it cannot stand before the model. And no value
may be served that the PDF does not print (decision 2026-10-24,
``serving.printed_values``), which needs page text. A document is therefore
read in two phases:

1. **Fields** (:func:`read_fields`, the gate): the pages are rendered and the
   model reads their images - ``image_only``, a mode it trains on - with the
   GPU to itself. Then MinerU reads **only the pages the fields cite** (their
   ``page_ref``: on a 200-page policy a few dozen, ~15-40 s) and the fields are
   held to that text before anything is served (:func:`verify_fields`): a value
   found there is served; one not found there is **held back**, flagged
   ``<path>:pending_full_text`` - not yet "printed nowhere", since the model
   may have cited the wrong page - and every declarations field left empty
   while those pages print its label and a value is flagged ``likely_missed``.
   A value whose cited page MinerU read empty is served flagged
   ``unverified_unread_page``, as the standard path serves it
   (``serving.printed_values``). So the fields served at the gate hold no value
   the PDF was not seen to print, nor one not said to be unverified. When no
   page the fields cite has any text, nothing could be checked: nothing is
   served before the full text. A row held back keeps the ids of the rows left
   as they are - a served row's id and links are the same at the gate and at
   the end - and a link to it is held back with it (``pending_full_text``).
2. **Full text** (:func:`add_full_text`): MinerU reads the remaining pages, and
   the fields as the model wrote them are held to the whole document's text -
   a value printed on a page it did not cite comes back; one printed nowhere is
   taken out - and the full-text tier joins them
   (``serving.pipeline.attach_full_text``). The final result is what the
   standard path serves for the same answer.

:class:`GpuTurns` keeps the order on the shared GPU: MinerU reads a run of
pages only while no document's fields are being read, so a document arriving
while another's text is read waits at most one run (``ServedOcr.run_pages``,
20 pages), not a whole document's OCR. So that full texts are not held back
for ever while fields keep arriving, a run waits at most ``text_max_wait_s``
and then goes regardless. A document's own verification is part of its
fields' turn.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: The longest a run of MinerU waits for its turn before it goes anyway.
TEXT_MAX_WAIT_S = 600.0

#: A value the pages its fields cite do not print, held back from the served
#: fields until the whole document's text has been searched.
PENDING_FLAG = "pending_full_text"

#: Fields served without being held to any page text (``verify=False``: timing
#: tests only - production serves verified fields).
UNVERIFIED_FLAG = "fields:unverified"

#: Pages read to verify fields that cite none: the declarations' usual place.
FALLBACK_PAGES = 5


class GpuTurns:
    """Whose turn the one GPU is: any document's fields before anyone's full text."""

    def __init__(self, text_max_wait_s: float | None = TEXT_MAX_WAIT_S) -> None:
        self._reading_fields = 0
        self._changed = threading.Condition()
        self.text_max_wait_s = text_max_wait_s
        #: Runs that went before their turn, having waited the longest allowed.
        self.overdue_runs = 0

    @contextmanager
    def fields(self) -> Iterator[None]:
        """Held while a document's fields are read; MinerU starts no new run meanwhile."""
        with self._changed:
            self._reading_fields += 1
        try:
            yield
        finally:
            with self._changed:
                self._reading_fields -= 1
                self._changed.notify_all()

    def wait_for_text_turn(self) -> None:
        """Return when no document's fields are being read (``ocr_via_api``'s
        gate) - or, past ``text_max_wait_s``, regardless: fields arriving one
        after another would otherwise hold every full text back for ever."""
        with self._changed:
            if self._changed.wait_for(lambda: self._reading_fields == 0, timeout=self.text_max_wait_s):
                return
            self.overdue_runs += 1
        log.warning("a run of MinerU waited %.0f s for the GPU while fields kept arriving; it goes now, "
                    "beside a document's fields", self.text_max_wait_s)


def cited_pages(extraction: Any, page_count: int) -> list[int]:
    """Every page a value of ``extraction`` cites (its ``page_ref``), within the document."""
    found: set[int] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            refs = node.get("page_ref")
            if isinstance(refs, list):
                found.update(page for page in refs if isinstance(page, int) and 1 <= page <= page_count)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(extraction)
    return sorted(found)


def _pending(flag: str) -> str:
    """A not-printed flag from a check on some pages only - on a value, or on a
    link to a row of such values - as what it means then."""
    from serving.conventions import NOT_PRINTED_FLAG
    from serving.printed_values import LINK_REMOVED_FLAG, REMOVED_FLAG

    path, _, kind = flag.rpartition(":")
    return (f"{path}:{PENDING_FLAG}" if path and kind in (REMOVED_FLAG, NOT_PRINTED_FLAG, LINK_REMOVED_FLAG)
            else flag)


def verify_fields(pdf: bytes, result: Any, *, name: str, lob: str, reader: Any, page_count: int) -> dict[str, Any]:
    """The fields as they may be served now: held to the text of the pages they cite.

    ``result`` is left as the model's answer, for the full text to hold to the
    whole document later; the served copy is returned. Returns ``{"served",
    "verified", "texts", "pages", "ocr", "seconds"}`` - ``texts`` by page
    number, kept so the full text does not read those pages again. ``verified``
    is False when no page read has any text: nothing was checked, and the copy
    must not be served.
    """
    from serving.pipeline import _likely_missed
    from serving.printed_values import hold_result_to_printed

    pages = cited_pages(result.extraction, page_count) or list(range(1, min(page_count, FALLBACK_PAGES) + 1))
    started = time.perf_counter()
    text = reader.ocr_text(pdf, name=name, pages=pages)
    served = copy.deepcopy(result)
    hold_result_to_printed(served, text["texts"])
    flags = {_pending(flag) for flag in served.review_flags}
    if result.doc_type == "policy" and result.mode == "image_only":
        flags.update(_likely_missed(served.extraction, text["texts"], lob))
    served.review_flags = sorted(flags)
    if served.printed_check is not None:
        served.printed_check = {**served.printed_check, "pages_read": pages, "whole_document": False}
    return {"served": served, "verified": served.printed_check is not None, "texts": text["texts"],
            "pages": pages, "ocr": text["info"], "seconds": round(time.perf_counter() - started, 2)}


def read_fields(pdf: bytes, *, name: str, lob: str, source_id: str, model: Any, reader: Any, pages_dir: Path,
                classifier: Any, calibration: Any, publish_pages: Any = None,
                turns: GpuTurns | None = None, verify: bool = True) -> dict[str, Any]:
    """Phase 1: the pages rendered, the model reading their images, and the
    fields held to the pages they cite (:func:`verify_fields`).

    Returns ``{"result", "served", "verified", "cited_texts", "images",
    "timings_s"}``: ``result`` is the model's answer, kept for the full text;
    ``served`` the fields to serve now. Should the verification fail - or the
    pages the fields cite have no text, so that nothing could be checked -
    ``served`` is None and ``verified`` False: nothing is served before the full
    text. ``verify=False`` serves the answer unchecked, flagged
    ``fields:unverified`` - for timing tests only.
    """
    from serving.pipeline import ExtractionRequest, extract

    began = time.perf_counter()
    # The turn is held from the start, rendering included, though rendering is
    # the CPU's: a run of MinerU started while the pages render would still be
    # reading when the model starts, on the GPU the gate needs for the model.
    # The verification's few pages are this document's own turn too.
    with turns.fields() if turns is not None else nullcontext():
        read = reader.read(pdf, pages_dir, name=name, with_text=False, publish_pages=publish_pages)
        rendered = time.perf_counter()
        request = ExtractionRequest(
            source_id=source_id, image_paths=read["images"], page_texts={}, modality_mode="image_only",
            known_doc_type="policy", known_lob=lob, source_file_name=name,
        )
        result = extract(request, model, classifier, calibration, strict_schema=False)
        answered = time.perf_counter()
        checked: dict[str, Any] | None = None
        served: Any = None
        if verify:
            try:
                checked = verify_fields(pdf, result, name=name, lob=lob, reader=reader,
                                        page_count=len(read["images"]))
            except Exception:  # noqa: BLE001 - nothing unchecked is served: the fields wait for the full text
                log.exception("%s: the fields could not be checked against their pages; they are served with "
                              "the full text", source_id)
            else:
                if checked["verified"]:
                    served = checked["served"]
                else:
                    log.warning("%s: no page the fields cite has text (pages %s); nothing was checked, so they "
                                "are served with the full text", source_id, checked["pages"])
        else:
            served = copy.deepcopy(result)
            served.review_flags = sorted({*served.review_flags, UNVERIFIED_FLAG})
    done = time.perf_counter()
    return {
        "result": result,
        "served": served,
        "verified": checked is not None and checked["verified"],
        "cited_texts": checked["texts"] if checked else {},
        "cited_pages": checked["pages"] if checked else [],
        "images": read["images"],
        "links": read.get("links"),
        "timings_s": {
            "render": round(read["timings_s"]["render"], 2),
            "links": read["timings_s"].get("links"),
            "model": round(answered - rendered, 2),
            "verify": checked["seconds"] if checked else None,
            "fields": round(done - began, 2),
        },
    }


def add_full_text(pdf: bytes, result: Any, *, name: str, lob: str, reader: Any, pages_dir: Path,
                  page_count: int, turns: GpuTurns | None = None,
                  known_texts: dict[int, str] | None = None) -> dict[str, Any]:
    """Phase 2: MinerU reads the pages not yet read - a run of pages at a time,
    in the GPU's idle turns when ``turns`` is given - and ``result``, the
    model's answer, is held to the whole document's text and joined by the
    full-text tier.

    ``known_texts``: pages already read (the cited pages, by
    :func:`verify_fields`). Each page's text lands in ``<pages dir>/page_NNN.md``.
    Returns what MinerU reported and how long it took.
    """
    from serving.pipeline import attach_full_text

    began = time.perf_counter()
    texts = dict(known_texts or {})
    remaining = [page for page in range(1, page_count + 1) if page not in texts]
    info: dict[str, Any] = {}
    seconds = 0.0
    if remaining:
        text = reader.ocr_text(pdf, name=name, pages=remaining if texts else None,
                               page_count=None if texts else page_count,
                               gate=turns.wait_for_text_turn if turns is not None else None)
        texts.update(text["texts"])
        info, seconds = text["info"], text["seconds"]
    texts = dict(sorted(texts.items()))
    pages_dir.mkdir(parents=True, exist_ok=True)
    for number, page in texts.items():
        (pages_dir / f"page_{number:03d}.md").write_text(page, encoding="utf-8")
    attach_full_text(result, texts, lob=lob)
    return {"ocr": info, "timings_s": {"ocr": round(seconds, 2), "pages_read": len(remaining),
                                       "full_text": round(time.perf_counter() - began, 2)}}
