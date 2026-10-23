"""A PDF's page images and page text, from a served MinerU, rendered while it reads.

For the extraction test runners (``testing/latency_check.py``,
``testing/extract_server.py``), next to a model and a MinerU already served on
a pod. The OCR stage proper (``run_mineru``) renders and reads in-process, for
corpus building; this reads the way a served deployment does:

* the pages are rendered by the shared renderer at the resolution cap - the
  pixels training and serving use - in ``render_workers`` processes at once;
* at the same time the PDF goes to MinerU's task API
  (:func:`data_pipeline.ocr.mineru_api.ocr_via_api`), in ``ocr_parts`` runs of
  pages when asked. Rendering is CPU work and MinerU's is the server's, so
  neither waits for the other: the slower one is the time both took;
* when the pages go to the model as links (``publish_pages``,
  :class:`inference_core.page_links.PageLinks`), they are uploaded once the PDF
  is at MinerU - not before, where they would share the connection with it and
  hold MinerU back - while MinerU reads it.

Each page lands in ``<pages dir>/page_NNN.png`` and, read, ``page_NNN.md`` - the
names the OCR stage writes and :func:`serving.pipeline._image_for` reads.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class ServedOcrError(RuntimeError):
    """The served MinerU and the renderer disagree on a PDF, or one of them failed."""


class ServedOcr:
    """Reads PDFs with a served MinerU; also an ``OcrCall`` for the job server."""

    def __init__(self, mineru_url: str, token: str | None = None, *, tier: str = "basic",
                 ocr_mode: str = "auto", ocr_parts: int = 1, render_workers: int | None = None) -> None:
        self.mineru_url = mineru_url
        self.token = token
        self.tier = tier
        self.ocr_mode = ocr_mode
        self.ocr_parts = ocr_parts
        self.render_workers = render_workers or min(8, os.cpu_count() or 1)

    def read(self, pdf: bytes, pages_dir: Path, *, name: str = "document.pdf", with_text: bool = True,
             force_ocr: bool = False,
             publish_pages: Callable[[list[str]], dict[str, Any]] | None = None) -> dict[str, Any]:
        """Render ``pdf`` into ``pages_dir`` while MinerU reads it.

        Returns the page image paths, the page texts by page number (none
        without ``with_text``), what MinerU reported and how long each part took.
        ``publish_pages`` is given the rendered pages once the PDF is at MinerU
        (``PageLinks.publish``); what it returns is reported as ``links``.
        """
        from common.config import resolution_cap_px
        from data_pipeline.ocr.mineru_api import ocr_via_api
        from data_pipeline.ocr.render_only import render_pdf_pages

        pages_dir.mkdir(parents=True, exist_ok=True)
        read: dict[str, Any] = {}
        uploaded = threading.Event()

        def ocr() -> None:
            started = time.perf_counter()
            try:
                read["pages"], read["info"] = ocr_via_api(
                    pdf, self.mineru_url, self.token, filename=name, tier=self.tier,
                    ocr_mode="ocr" if force_ocr else self.ocr_mode, parts=self.ocr_parts, uploaded=uploaded)
            except Exception as exc:  # noqa: BLE001 - raised below, once the rendering is done
                read["error"] = exc
            read["seconds"] = time.perf_counter() - started

        began = time.perf_counter()
        reader = threading.Thread(target=ocr, name="served-ocr", daemon=True) if with_text else None
        if reader is not None:
            reader.start()
        images = render_pdf_pages(pdf, resolution_cap_px(), workers=self.render_workers)
        paths = []
        for number, image in enumerate(images, start=1):
            path = pages_dir / f"page_{number:03d}.png"
            path.write_bytes(image)
            paths.append(str(path))
        rendered = time.perf_counter() - began

        links: dict[str, Any] | None = None
        if publish_pages is not None:
            # Not while the PDF is still going up: the two would share the
            # connection, and MinerU, which everything waits for, would start later.
            while reader is not None and reader.is_alive() and not uploaded.wait(0.2):
                pass
            links = publish_pages(paths)

        texts: dict[int, str] = {}
        if reader is not None:
            reader.join()
            if "error" in read:
                raise read["error"]
            if len(read["pages"]) != len(images):
                raise ServedOcrError(f"{name}: MinerU read {len(read['pages'])} pages, the renderer {len(images)}")
            for number, text in enumerate(read["pages"], start=1):
                (pages_dir / f"page_{number:03d}.md").write_text(text, encoding="utf-8")
                texts[number] = text
        return {
            "images": paths,
            "texts": texts,
            "ocr": read.get("info") or {},
            "links": links,
            "timings_s": {
                "render": round(rendered, 2),
                "ocr": round(read["seconds"], 2) if "seconds" in read else None,
                "ocr_server": (read.get("info") or {}).get("server_seconds"),
                "links": links["seconds"] if links else None,
                # Rendering, OCR and the links at once: the time the pages and text took together.
                "pages_and_text": round(time.perf_counter() - began, 2),
            },
        }

    def __call__(self, pdf: Path, out: Path, *, force_ocr: bool = False) -> dict[str, Any]:
        """The job server's ``OcrCall``: ``{"ok", "pages", "modality", ...}``."""
        from data_pipeline.ocr.modality import detect_modality

        data = pdf.read_bytes()
        try:
            read = self.read(data, out, name=pdf.name, force_ocr=force_ocr)
        except Exception as exc:  # noqa: BLE001 - reported to the job, as the OCR worker reports it
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"ok": True, "pages": len(read["images"]), "modality": detect_modality(data),
                "ocr_failed_pages": [], "timings_s": read["timings_s"], "ocr": read["ocr"]}
