"""A PDF's page images and page text, from a served MinerU, rendered while it reads.

The OCR of record since 2026-10-23 (``configs/shared/ocr.yaml``): the MinerU 4
API served on the pod reads the training corpus (:class:`ServedMinerUEngine`,
the OCR stage's engine) and every served document (:class:`ServedOcr`, for the
extraction routine and its test runners) - one MinerU, one version, one tier,
so the model is served the text it was trained on. Each task reports the
version that read it, and a task read by another version is refused.

* the pages are rendered by the shared renderer at the resolution cap - the
  pixels training and serving use - in ``render_workers`` processes at once;
* at the same time the PDF goes to MinerU's task API
  (:func:`data_pipeline.ocr.mineru_api.ocr_via_api`), in ``ocr_parts`` runs of
  pages when asked. Rendering is CPU work and MinerU's is the server's, so
  neither waits for the other: the slower one is the time both took;
* OCR or text mode is decided as the OCR stage decides it
  (:func:`data_pipeline.ocr.modality.reads_by_ocr`: a scan, a page without a
  text layer, or drawn figures send the whole document through OCR), unless an
  ``ocr_mode`` is given;
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

#: Where the served MinerU is, for the OCR stage: its base URL and bearer token.
MINERU_URL_ENV = "MINERU_URL"
MINERU_TOKEN_ENV = "MINERU_TOKEN"


class ServedOcrError(RuntimeError):
    """The served MinerU and the renderer disagree on a PDF, one of them failed,
    or the server is not the MinerU the model was trained on."""


def ocr_mode_for(pdf: bytes, *, modality: str | None = None, force_ocr: bool | None = None) -> str:
    """MinerU's mode for ``pdf`` by the OCR stage's rule: ``ocr`` or ``txt``."""
    from data_pipeline.ocr.modality import reads_by_ocr

    return "ocr" if reads_by_ocr(pdf, modality=modality, force_ocr=force_ocr) else "txt"


class ServedOcr:
    """Reads PDFs with a served MinerU; also an ``OcrCall`` for the job server.

    ``expected_version``: the MinerU version every task must report (the pin,
    ``common.config.served_mineru``); None checks nothing (a timing test against
    another MinerU). ``ocr_mode``: forced for every document; None decides per
    document by the OCR stage's rule (:func:`ocr_mode_for`).
    """

    def __init__(self, mineru_url: str, token: str | None = None, *, tier: str = "basic",
                 ocr_mode: str | None = None, ocr_parts: int = 1, render_workers: int | None = None,
                 expected_version: str | None = None) -> None:
        from common.transport import assert_encrypted

        assert_encrypted(mineru_url, "the MinerU server")
        self.mineru_url = mineru_url
        self.token = token
        self.tier = tier
        self.ocr_mode = ocr_mode
        self.ocr_parts = ocr_parts
        self.render_workers = render_workers or min(8, os.cpu_count() or 1)
        self.expected_version = expected_version

    def read(self, pdf: bytes, pages_dir: Path, *, name: str = "document.pdf", with_text: bool = True,
             force_ocr: bool = False, ocr_mode: str | None = None,
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
        mode = None
        if with_text:
            mode = ("ocr" if force_ocr else None) or ocr_mode or self.ocr_mode or ocr_mode_for(pdf)

        def ocr() -> None:
            started = time.perf_counter()
            try:
                read["pages"], read["info"] = ocr_via_api(
                    pdf, self.mineru_url, self.token, filename=name, tier=self.tier,
                    ocr_mode=mode, parts=self.ocr_parts, uploaded=uploaded)
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
            self._assert_version(read["info"], name)
            if len(read["pages"]) != len(images):
                raise ServedOcrError(f"{name}: MinerU read {len(read['pages'])} pages, the renderer {len(images)}")
            for number, text in enumerate(read["pages"], start=1):
                (pages_dir / f"page_{number:03d}.md").write_text(text, encoding="utf-8")
                texts[number] = text
        return {
            "images": paths,
            "texts": texts,
            "ocr": {**(read.get("info") or {}), "ocr_mode": mode} if reader is not None else {},
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

    def _assert_version(self, info: dict[str, Any], name: str) -> None:
        """Refuse text from a MinerU other than the pinned one, tier included."""
        if self.expected_version is None:
            return
        version, tier = info.get("version"), info.get("tier")
        if version != self.expected_version or (tier and tier != self.tier):
            raise ServedOcrError(
                f"{name} was read by MinerU {version} (tier {tier}) at {self.mineru_url}, but the model is "
                f"trained on text from MinerU {self.expected_version} (tier {self.tier}): its text is "
                "formatted differently, input the model never saw (arch §8a). Serve the pinned MinerU, or "
                "re-OCR the corpus with this one and retrain (configs/shared/ocr.yaml)."
            )

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


class ServedMinerUEngine:
    """The served MinerU as the OCR stage's engine (``run_mineru.OcrEngine``).

    The training corpus is read by the same server, version and tier that
    serving reads with (``configs/shared/ocr.yaml``), so what the model learns
    MinerU's text to look like is what it is served. Runs anywhere that reaches
    the server - no MinerU, no GPU here: the GPU is the server's, which is why
    the environment it records is ``cuda``.
    """

    def __init__(self, mineru_url: str, token: str | None = None, *, version: str | None = None,
                 tier: str | None = None, parts: int = 1, render_workers: int | None = None) -> None:
        from common.config import served_mineru

        pin = served_mineru()
        self.version = version or pin["version"]
        self.tier = tier or pin["tier"]
        self.reader = ServedOcr(mineru_url, token, tier=self.tier, ocr_parts=parts,
                                render_workers=render_workers, expected_version=self.version)

    @classmethod
    def from_environment(cls) -> ServedMinerUEngine | None:
        """The engine for ``$MINERU_URL`` (token ``$MINERU_TOKEN``, else
        ``$FIDEON_REMOTE_TOKEN``), or None when no served MinerU is configured."""
        url = (os.environ.get(MINERU_URL_ENV) or "").strip()
        if not url:
            return None
        token = os.environ.get(MINERU_TOKEN_ENV) or os.environ.get("FIDEON_REMOTE_TOKEN") or None
        return cls(url, token)

    def environment(self) -> Any:
        """What the OCR stage records and checks a stored reading against."""
        from data_pipeline.ocr.mineru_version import OcrEnvironment

        return OcrEnvironment(mineru_version=self.version, device="cuda", gpu_name=None)

    def meta(self) -> dict[str, Any]:
        """Added to each document's ``ocr_meta.json``."""
        return {"mineru_backend": "api", "mineru_tier": self.tier}

    def process(self, pdf_bytes: bytes, *, device: str, max_long_side_px: int,
                modality: str | None = None, force_ocr: bool | None = None) -> list[Any]:
        import tempfile

        from common.config import resolution_cap_px
        from data_pipeline.ocr.run_mineru import OcrError, PageOutput, count_table_rows, text_layer_pages

        if max_long_side_px != resolution_cap_px():
            raise OcrError(f"the served reader renders at the resolution cap ({resolution_cap_px()} px), "
                           f"not {max_long_side_px}")
        mode = ocr_mode_for(pdf_bytes, modality=modality, force_ocr=force_ocr)
        text_layer = text_layer_pages(pdf_bytes)
        with tempfile.TemporaryDirectory() as tmp:
            read = self.reader.read(pdf_bytes, Path(tmp), ocr_mode=mode)
            images = [Path(path).read_bytes() for path in read["images"]]
        scanned = mode == "ocr"
        return [
            PageOutput(
                page_number=number,
                markdown=text,
                image_bytes=image,
                table_row_count=count_table_rows(text),
                # As the in-process engine: nothing read on a page with no text
                # layer, or on any page of a document read by OCR, is a failure.
                ocr_failed=(scanned or not text_layer[number - 1]) and not text.strip(),
                scanned=not text_layer[number - 1],
                read_by_ocr=scanned,
            )
            for number, (text, image) in enumerate(zip(read["texts"].values(), images, strict=True), start=1)
        ]
