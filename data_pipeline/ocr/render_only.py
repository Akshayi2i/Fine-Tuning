"""Page rendering without OCR — the ``image_only`` pathway (arch §6).

``image_only`` is 30% of the Foundation corpus and a real production mode: the
document arrives, no OCR is available or wanted, and the model reads the pixels
directly. That path needs page images but no markdown, so rendering is separated
from OCR rather than being a flag on it.

The resolution cap is the same one used everywhere else. Rendering ``image_only``
pages at a different resolution than ``ocr_plus_image`` pages would teach the
model that the two modes look different, which is not a distinction that exists
at inference.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from artifact_registry import paths
from artifact_registry.blob_client import BlobClient, for_ocr
from common.config import resolution_cap_px
from common.constants import ACTIVE_DOC_TYPES
from data_pipeline.ocr.mineru_version import Device, current_environment

log = logging.getLogger(__name__)


class RenderError(RuntimeError):
    """Raised when a PDF cannot be rendered to page images."""


def render_pdf_pages(pdf_bytes: bytes, max_long_side_px: int, workers: int = 1) -> list[bytes]:
    """Render each page to PNG with its long side at ``max_long_side_px``.

    Aspect ratio is preserved. The scale factor converts PDF points (72 per inch)
    to pixels, so a US Letter page at a 1792px cap renders at roughly 163 DPI.

    ``workers`` above 1 renders runs of pages in that many processes at once -
    a 114-page scan took 93 s in one - each page exactly as one process renders
    it, so the images are the same bytes either way.

    Note what this does **not** do: a page whose content is an embedded 100-DPI
    scan is still rendered to the cap, which upsamples that raster and spends
    vision tokens on no extra information. Avoiding that means inspecting the
    embedded image resolution per page, which is a separate concern from the cap
    and is not attempted here.
    """
    count = _page_count(pdf_bytes)
    workers = max(1, min(workers, count))
    if workers == 1:
        images = _render_range(pdf_bytes, max_long_side_px, 0, count)
    else:
        from concurrent.futures import ProcessPoolExecutor

        bounds = [count * part // workers for part in range(workers + 1)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = pool.map(_render_range, [pdf_bytes] * workers, [max_long_side_px] * workers,
                            bounds[:-1], bounds[1:])
            images = [image for run in runs for image in run]
    if not images:
        raise RenderError("PDF produced no pages")
    return images


def _fitz() -> Any:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RenderError(
            'PyMuPDF is not installed. Install the [data] extra: pip install -e ".[data]"'
        ) from exc
    return fitz


def _page_count(pdf_bytes: bytes) -> int:
    with _fitz().open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count


def _render_range(pdf_bytes: bytes, max_long_side_px: int, start: int, stop: int) -> list[bytes]:
    """Pages ``start`` to ``stop - 1`` (0-based) as PNG: one process's share."""
    fitz = _fitz()
    images: list[bytes] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index in range(start, stop):
            page = doc[index]
            rect = page.rect
            longest = max(rect.width, rect.height)
            if longest <= 0:
                raise RenderError("page has zero dimensions")
            # `rect` is in points; scaling by cap/longest puts the long side
            # exactly on the cap in pixels.
            scale = max_long_side_px / longest
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            images.append(pixmap.tobytes("png"))
    return images


def render_document(
    doc_type: str,
    source_id: str,
    client: BlobClient,
    *,
    tenant_id: str | None = None,
    device: Device | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Render one document's pages, writing an OCR-free ``ocr_meta.json``.

    The metadata skeleton is written even with no OCR content, so an
    ``image_only`` document is as traceable as any other — same version pinning,
    same resolution record, same provenance.
    """
    cap = resolution_cap_px()
    env = current_environment(device, strict=device == "cuda")
    meta_key = paths.ocr_meta(doc_type, source_id, tenant_id)

    was_ocred = False
    existing: dict[str, Any] = {}
    if client.exists(meta_key):
        existing = client.read_json(meta_key)
        # An OCR'd document lacks `render_only`, so its images are re-rendered
        # rather than assumed present — but its metadata is preserved, not
        # replaced. Overwriting it discarded table_row_counts and flipped the
        # document to image_only, which silently stopped the row-completeness
        # cross-check from flagging it while its page_*.md sat there unread.
        was_ocred = not existing.get("render_only")
        if not force and not was_ocred and existing.get("resolution_cap_px") == cap:
            log.info("skipping %s: already rendered at %dpx", source_id, cap)
            return existing

    raw_meta = client.read_json(paths.raw_metadata(doc_type, source_id, tenant_id))
    pdf_bytes = client.read_bytes(paths.raw_pdf(doc_type, source_id, tenant_id))
    images = render_pdf_pages(pdf_bytes, cap)

    for page_number, image in enumerate(images, start=1):
        client.write_bytes(
            paths.processed_page(doc_type, source_id, page_number, "png", tenant_id), image
        )

    meta = {
        "source_id": source_id,
        "doc_type": doc_type,
        "tenant_id": paths._tenant(tenant_id),
        "page_count": len(images),
        # False for a document that was OCR'd: its markdown still exists, so
        # calling it image_only would be a lie the whole downstream pipeline
        # acts on — pre_annotate would stop passing page_texts and
        # list_completeness would lose its detected-row cross-check.
        "render_only": not was_ocred,
        "mineru_version": env.mineru_version,
        "ocr_device": env.device,
        "preprocessing_date": datetime.now(UTC).isoformat(),
        "resolution_cap_px": cap,
        "source_checksum": raw_meta.get("checksum_sha256"),
        # Preserved. They cannot be recomputed without re-running OCR.
        "table_row_counts": existing.get("table_row_counts", {}) if was_ocred else {},
        "failed_pages": existing.get("failed_pages", []) if was_ocred else [],
    }
    # Preserved too. Dropped, a scan was later read as a digital document: out
    # of the scanned eval subset, and scored against the wrong comparison.
    if was_ocred and "is_scanned" in existing:
        meta["is_scanned"] = existing["is_scanned"]
    if was_ocred:
        log.warning(
            "%s was OCR'd; re-rendered its page images at %dpx and kept its OCR metadata. "
            "It stays an ocr_plus_image document — rendering does not remove text that exists.",
            source_id, cap,
        )
    client.write_json(meta_key, meta)
    log.info("rendered %s: %d pages at %dpx (no OCR)", source_id, len(images), cap)
    return meta


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Render page images without OCR (image_only mode)")
    parser.add_argument("--doc-type", required=True, choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--source-ids", nargs="+", required=True)
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('data_pipeline.ocr.render_only', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = for_ocr()

    failed = []
    for source_id in args.source_ids:
        try:
            render_document(args.doc_type, source_id, client,
                            tenant_id=args.tenant, force=args.force)
        except RenderError as exc:
            failed.append((source_id, str(exc)))
            log.warning("render failed for %s: %s", source_id, exc)

    print(f"{len(args.source_ids) - len(failed)} rendered, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
