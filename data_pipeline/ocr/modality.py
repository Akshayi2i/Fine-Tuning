"""Which reader a PDF needs: its modality, and whether its figures are drawn
(Fideon SPEC_05 §4 and §4.1).

Two modalities, three cases. A PDF is ``native_pdf`` when every page carries a
text layer and ``scanned_pdf`` otherwise - one page without one sends the whole
document through OCR, because text mode leaves a mixed document's scanned pages
unread. The third case is a native PDF whose figures are DRAWN as filled vector
outlines instead of typed: its text layer carries the headings but not the
numbers, so it is native and still needs OCR. That is ``force_ocr``, not a
second modality: the document is native, and everything downstream that keys
on modality should keep seeing it so (SPEC_05 §4.1).

In serving the caller knows both - the routing stage upstream decided them -
and passes them in. Only the offline tools, which have no upstream stage,
detect them here. Upstream, both come from SPEC_01's step 0, which is not in
this repository; these are this repository's own and the one place to swap
that function in.
"""

from __future__ import annotations

from typing import Any

NATIVE = "native_pdf"
SCANNED = "scanned_pdf"
MODALITIES = (NATIVE, SCANNED)

#: A filled vector path this small is the size of a glyph: a digit drawn as an
#: outline rather than typed. Points; a 6-12 pt figure is 4-12 pt tall.
GLYPH_MAX_PT = 24.0
GLYPH_MIN_PT = 0.5

#: A page draws its data when it holds at least this many glyph-sized filled
#: paths and at least this many per readable text span. SPEC_05 §4.1's case
#: (NYCM's homeowner summary) carries 380 such paths against 37 spans in one
#: coverage band alone; a typed page with a vector logo has many spans per path.
DRAWN_MIN_GLYPH_PATHS = 100
DRAWN_PATHS_PER_SPAN = 2.0


class ModalityError(ValueError):
    """Raised on a modality that is neither native nor scanned."""


def detect_modality(pdf_bytes: bytes) -> str:
    """``native_pdf`` when every page has a text layer (or nothing to read), else ``scanned_pdf``."""
    from data_pipeline.ocr.run_mineru import text_layer_pages

    return NATIVE if all(text_layer_pages(pdf_bytes)) else SCANNED


#: A glyph is at least this thick both ways. A table's rules and cell borders,
#: which many PDF writers draw as thin filled rectangles, are thinner.
GLYPH_MIN_THICKNESS_PT = 1.5

#: Path segments (pymupdf drawing items) an outline without curves needs: a
#: digit drawn in straight lines has five or more, a box four.
GLYPH_MIN_SEGMENTS = 5


def _is_glyph(drawing: dict[str, Any]) -> bool:
    """A filled, glyph-sized, glyph-thick OUTLINE: curves or several segments.

    Not a box: a rectangle (one ``re`` item, or four lines) is a cell border, a
    checkbox or a shading block, however small, and an ordinary digital table
    is drawn from hundreds of them.
    """
    if drawing.get("fill") is None:
        return False
    rect = drawing.get("rect")
    if rect is None or not (GLYPH_MIN_PT <= rect.width <= GLYPH_MAX_PT
                            and GLYPH_MIN_PT <= rect.height <= GLYPH_MAX_PT):
        return False
    if min(rect.width, rect.height) < GLYPH_MIN_THICKNESS_PT:
        return False
    ops = [item[0] for item in drawing.get("items") or () if item]
    return "c" in ops or len(ops) >= GLYPH_MIN_SEGMENTS


def _glyph_paths(page: Any) -> int:
    return sum(1 for drawing in page.get_drawings() if _is_glyph(drawing))


def _text_spans(page: Any) -> int:
    return sum(
        1
        for block in page.get_text("dict").get("blocks", [])
        for line in block.get("lines", [])
        for span in line.get("spans", [])
        if str(span.get("text") or "").strip()
    )


def page_is_drawn(page: Any) -> bool:
    paths = _glyph_paths(page)
    return paths >= DRAWN_MIN_GLYPH_PATHS and paths >= DRAWN_PATHS_PER_SPAN * _text_spans(page)


def data_is_drawn(pdf_bytes: bytes) -> bool:
    """Whether any page draws its figures as vector outlines (SPEC_05 §4.1)."""
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return any(page_is_drawn(page) for page in doc)


def reads_by_ocr(pdf_bytes: bytes, *, modality: str | None = None, force_ocr: bool | None = None) -> bool:
    """Whether the document takes MinerU's OCR path.

    Every document does when ``read_every_page_by_ocr`` is on
    (``common.config.ocr_every_page``, configs/shared/ocr.yaml): training and
    serving then read a hybrid PDF's pages as they read a scan's. Otherwise
    ``modality`` and ``force_ocr`` are the caller's when given and trusted as
    given; unset (the offline tools), the modality is detected and a native
    document is checked for drawn figures.
    """
    from common.config import ocr_every_page

    if ocr_every_page():
        return True
    detected = modality is None
    if detected:
        modality = detect_modality(pdf_bytes)
    elif modality not in MODALITIES:
        raise ModalityError(f"modality {modality!r} is neither of {list(MODALITIES)}")
    if force_ocr is None:
        force_ocr = detected and modality == NATIVE and data_is_drawn(pdf_bytes)
    return modality == SCANNED or bool(force_ocr)
