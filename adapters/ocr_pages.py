"""A page of MinerU's output, its errors, and whether a PDF page carries a text layer (review A1: moved
from ``data_pipeline.ocr.run_mineru``, which re-exports them). The served MinerU client and the OCR
stage share them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class OcrError(RuntimeError):
    """Raised when a document cannot be processed."""

@dataclass
class PageOutput:
    """One page's OCR text and rendered image."""

    page_number: int          # 1-based; ordering is positionally meaningful
    markdown: str
    image_bytes: bytes
    table_row_count: int = 0  # feeds the row-completeness signal (IMPL-09)
    ocr_failed: bool = False
    #: The page has no text layer (a scan). A document with any such page is
    #: OCR'd whole and recorded as ``is_scanned``.
    scanned: bool = False
    #: MinerU read the document in OCR mode (a scan, a mixed document, drawn
    #: figures, or the caller's force_ocr), not from its text layer.
    read_by_ocr: bool = False

#: Characters of embedded text below which a page counts as having no text layer.
TEXT_LAYER_MIN_CHARS = 30

#: Text render modes that draw nothing on the page: 3 is invisible text - the
#: layer a scanner's OCR lays under a page image to make it searchable - and 7
#: adds the text to a clipping path only.
INVISIBLE_TEXT_MODES = (3, 7)

def visible_text_chars(page: Any) -> int:
    """Characters of text drawn visibly on a page, not counting an invisible
    OCR layer or fully transparent text."""
    return sum(
        len(span.get("chars", ()))
        for span in page.get_texttrace()
        if span.get("type") not in INVISIBLE_TEXT_MODES and span.get("opacity", 1) > 0
    )

def text_layer_pages(pdf_bytes: bytes) -> list[bool]:
    """Per page, whether it is a digital page rather than a scan.

    A page with visible text is digital. Only visible text counts: a scan made
    searchable is a page image with the scanner's OCR text laid invisibly under
    it, and counting that layer called it digital - its own OCR was read
    instead of MinerU's, and it was recorded as not scanned. An EMPTY page - no
    text, no image, no drawing - is digital too: there is nothing on it to OCR.
    Counting it as a scan sent a whole digital policy with one blank page
    through OCR mode and into the scanned eval subset. A page with no visible
    text but an image or drawings on it is a scan (or might be), and is read by
    OCR.
    """
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [
            visible_text_chars(page) >= TEXT_LAYER_MIN_CHARS
            or not (page.get_images() or page.get_drawings())
            for page in doc
        ]

#: Content-list blocks at a page's edges. MinerU 3.x lists them after the page's
#: body; a page's header is put back at its head and the rest at its foot, so the
#: text reads top to bottom. MinerU 1.x dropped them: a form's number, printed in
#: its footer, was in no page's text.
_PAGE_HEAD = ("header",)

_PAGE_FOOT = ("footer", "page_number", "page_footnote", "aside_text")
