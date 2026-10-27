"""MinerU takes modality and force_ocr from its caller, forces OCR on drawn
figures, and keeps to the native and scanned time budgets (Fideon SPEC_05 §4,
§4.1, §9; handoff item 7)."""

from __future__ import annotations

import asyncio
import time

import pytest

from data_pipeline.ocr import mineru_runner
from data_pipeline.ocr.mineru_runner import MinerUError, run_mineru
from data_pipeline.ocr.modality import NATIVE, SCANNED, data_is_drawn, detect_modality, reads_by_ocr
from data_pipeline.ocr.run_mineru import MinerUEngine, OcrError, PageOutput
from tests.test_mineru_engine import CONTENT, _fake_mineru

pymupdf = pytest.importorskip("pymupdf")


@pytest.fixture(autouse=True)
def _per_document_rule(monkeypatch):
    """The per-document rule these tests pin is the one read_every_page_by_ocr
    off leaves (configs/shared/ocr.yaml); on, every document is read by OCR
    (tests/test_ocr_every_page.py)."""
    monkeypatch.setattr("common.config.ocr_every_page", lambda: False)


def _native(lines=3, glyph_paths=0):
    """A typed page; ``glyph_paths`` small filled outlines (curved, as a drawn
    digit is) beside the text."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for i in range(lines):
        page.insert_text((72, 60 + 12 * i), f"LOCATION COVERAGES line {i} with a real text layer")
    for i in range(glyph_paths):
        x, y = 72 + (i % 40) * 12, 500 + (i // 40) * 14
        page.draw_oval(pymupdf.Rect(x, y, x + 5, y + 8), color=None, fill=(0, 0, 0))
    return doc.tobytes()


def _typed_table(rows=30, cols=10, filled_every=10):
    """A digital schedule as Word or Excel export it: typed values, and each cell's
    borders drawn as thin filled rectangles, some cells shaded."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for r in range(rows):
        for c in range(cols):
            x, y = 40 + c * 52, 40 + r * 22
            page.draw_rect(pymupdf.Rect(x, y, x + 0.75, y + 16), color=None, fill=(0, 0, 0))   # left rule
            page.draw_rect(pymupdf.Rect(x, y + 16, x + 52, y + 16.75), color=None, fill=(0, 0, 0))
            if (r * cols + c) % filled_every == 0:
                page.draw_rect(pymupdf.Rect(x + 2, y + 2, x + 10, y + 10), color=None, fill=(0.9, 0.9, 0.9))
            if c % 5 == 0:
                page.insert_text((x + 3, y + 12), f"{r}.{c}")
    return doc.tobytes()


def _drawn():
    """SPEC_05 §4.1: headings typed, Coverage A through F drawn as outlines."""
    return _native(lines=3, glyph_paths=380)


def _scan():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(pymupdf.Rect(50, 50, 560, 740), color=None, fill=(0.8, 0.8, 0.8))
    return doc.tobytes()


# --------------------------------------------------------------------------
# Modality and the drawn-figure check
# --------------------------------------------------------------------------

def test_offline_detection_reads_native_and_scanned():
    assert detect_modality(_native()) == NATIVE
    assert detect_modality(_drawn()) == NATIVE            # it IS native: the flag says OCR, not the modality
    assert detect_modality(_scan()) == SCANNED


def _searchable_scan(pages=2):
    """A scan made searchable: each page an image, with the scanner's OCR text laid
    invisibly over it (render mode 3)."""
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page(width=612, height=792)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 80), 0)
        pix.clear_with(220)
        page.insert_image(page.rect, pixmap=pix)
        page.insert_text((72, 72), f"Named Insured Rivera Fabrication LLC policy page {i + 1} of the declarations",
                         render_mode=3)
    return doc.tobytes()


def test_a_searchable_scan_is_a_scan_its_invisible_ocr_layer_is_no_text_layer():
    from data_pipeline.ocr.run_mineru import text_layer_pages

    scan = _searchable_scan()
    assert text_layer_pages(scan) == [False, False]
    assert detect_modality(scan) == SCANNED and reads_by_ocr(scan)
    assert text_layer_pages(_native()) == [True]                     # visible text: digital


def test_a_blank_page_does_not_make_a_digital_pdf_a_scan():
    from data_pipeline.ocr.run_mineru import text_layer_pages

    doc = pymupdf.open(stream=_native(), filetype="pdf")
    doc.new_page(width=612, height=792)                               # nothing on it
    pdf = doc.tobytes()
    assert text_layer_pages(pdf) == [True, True] and detect_modality(pdf) == NATIVE


def test_the_engine_reads_a_searchable_scan_by_ocr(monkeypatch):
    calls = _fake_mineru(monkeypatch, CONTENT, scanned=False)          # MinerU itself would say text
    pages = MinerUEngine().process(_searchable_scan(3), device="cuda", max_long_side_px=800)
    assert calls["mode"] == "ocr" and all(p.scanned for p in pages)


def test_drawn_figures_are_found_and_a_vector_logo_beside_typed_text_is_not():
    assert data_is_drawn(_drawn())
    assert not data_is_drawn(_native())
    assert not data_is_drawn(_native(lines=60, glyph_paths=100))   # many spans per path: typed


def test_a_typed_table_drawn_from_thin_border_rectangles_is_not_drawn_data():
    """Hundreds of filled border segments and shaded boxes: boxes, not glyph outlines."""
    assert not data_is_drawn(_typed_table())


def test_the_caller_decides_when_it_says_and_the_document_otherwise():
    drawn = _drawn()
    assert reads_by_ocr(drawn)                                         # offline: detected and forced
    assert not reads_by_ocr(drawn, modality=NATIVE)                    # caller said native, no force
    assert reads_by_ocr(_native(), modality=NATIVE, force_ocr=True)
    assert reads_by_ocr(_native(), modality=SCANNED)


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------

def test_a_native_document_with_drawn_figures_goes_through_ocr(monkeypatch):
    calls = _fake_mineru(monkeypatch, CONTENT, scanned=False)          # MinerU itself says text
    MinerUEngine().process(_drawn(), device="cuda", max_long_side_px=800)
    assert calls["mode"] == "ocr" and calls["ocr"] is True
    calls = _fake_mineru(monkeypatch, CONTENT, scanned=False)
    MinerUEngine().process(_native(), device="cuda", max_long_side_px=800)
    assert calls["mode"] == "txt"


def test_the_engine_takes_the_callers_modality_and_force_ocr(monkeypatch):
    calls = _fake_mineru(monkeypatch, CONTENT, scanned=True)           # MinerU would say OCR
    MinerUEngine().process(_native(), device="cuda", max_long_side_px=800, modality=NATIVE)
    assert calls["mode"] == "txt"                                      # the caller is not second-guessed
    MinerUEngine().process(_native(), device="cuda", max_long_side_px=800, modality=NATIVE, force_ocr=True)
    assert calls["mode"] == "ocr"
    with pytest.raises(OcrError, match="neither"):
        MinerUEngine().process(_native(), device="cuda", max_long_side_px=800, modality="native")


# --------------------------------------------------------------------------
# The timed entry point
# --------------------------------------------------------------------------

class _Engine:
    def __init__(self, seconds=0.0, text="Named Insured: Rivera Fabrication LLC"):
        self.seconds, self.text, self.calls = seconds, text, []

    def process(self, pdf_bytes, *, device, max_long_side_px, modality=None, force_ocr=None):
        self.calls.append((modality, force_ocr))
        time.sleep(self.seconds)
        return [PageOutput(page_number=1, markdown=self.text, image_bytes=b"")]


def test_a_timeout_raises_the_mineru_error(monkeypatch):
    monkeypatch.setattr(mineru_runner, "NATIVE_TIMEOUT_S", 0.05)
    with pytest.raises(MinerUError, match="timed out after 0.05s on the native path"):
        asyncio.run(run_mineru(b"%PDF", NATIVE, engine=_Engine(seconds=0.3), max_long_side_px=800))


def test_force_ocr_takes_the_scanned_path_and_its_budget(monkeypatch):
    monkeypatch.setattr(mineru_runner, "NATIVE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(mineru_runner, "SCANNED_TIMEOUT_S", 5)
    engine = _Engine(seconds=0.2)
    out = asyncio.run(run_mineru(b"%PDF", NATIVE, force_ocr=True, engine=engine, max_long_side_px=800))
    assert out.path_used == "scanned" and engine.calls == [(NATIVE, True)]
    assert out.total_pages == 1 and out.duration_ms >= 150


def test_nothing_read_or_an_unknown_modality_is_a_mineru_error():
    with pytest.raises(MinerUError, match="fewer than 10 characters"):
        asyncio.run(run_mineru(b"%PDF", SCANNED, engine=_Engine(text="  "), max_long_side_px=800))
    with pytest.raises(MinerUError, match="neither"):
        asyncio.run(run_mineru(b"%PDF", "pdf", engine=_Engine(), max_long_side_px=800))


def test_the_spec_budgets():
    assert (mineru_runner.NATIVE_TIMEOUT_S, mineru_runner.SCANNED_TIMEOUT_S) == (30, 90)


def test_an_unreadable_pdf_is_a_mineru_error_not_pymupdfs(monkeypatch):
    _fake_mineru(monkeypatch, CONTENT)
    for bad in (b"not a pdf", b""):
        with pytest.raises(MinerUError, match="MinerU failed"):
            asyncio.run(run_mineru(bad, NATIVE, engine=MinerUEngine(), max_long_side_px=800))


def test_a_timeout_raised_inside_the_engine_is_a_failure_not_the_budget():
    class _Raises(_Engine):
        def process(self, *args, **kwargs):
            raise TimeoutError("socket read timed out")

    with pytest.raises(MinerUError, match="MinerU failed: TimeoutError"):
        asyncio.run(run_mineru(b"%PDF", NATIVE, engine=_Raises(), max_long_side_px=800))
