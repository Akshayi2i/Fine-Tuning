"""The MinerU engine: one markdown string per page, rows counted in HTML tables,
GPU only. MinerU itself is a stand-in here; the Phase 0 spike runs the real one."""

from __future__ import annotations

import sys
import types

import pytest

from data_pipeline.ocr.run_mineru import (
    OCR_LANG,
    MinerUEngine,
    OcrError,
    count_table_rows,
    pages_from_content_list,
)

# --------------------------------------------------------------------------
# Row counting — MinerU writes HTML tables
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,rows", [
    ("<table><tr><th>Claim</th><th>Paid</th></tr><tr><td>1</td><td>5</td></tr>"
     "<tr><td>2</td><td>7</td></tr></table>", 2),
    ("<table><tr><td>Claim</td></tr><tr><td>1</td></tr></table>", 1),       # first row is the header
    ("<table><tr><th>a</th></tr></table><p/><table><tr><th>b</th></tr><tr><td>1</td></tr></table>", 1),
    ("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |", 2),                          # pipe tables still count
    ("no tables here", 0),
])
def test_table_rows_are_counted_in_html_and_pipe_tables(text, rows):
    assert count_table_rows(text) == rows


# --------------------------------------------------------------------------
# One markdown string per page
# --------------------------------------------------------------------------


#: MinerU 3.x's content list: a page's body, then its edges (header, footer,
#: page number), as union_make lists them.
CONTENT = [
    {"type": "text", "text": "COMMON POLICY DECLARATIONS", "text_level": 1, "page_idx": 0},
    {"type": "text", "text": "Named Insured: Rivera Fabrication LLC", "page_idx": 0},
    {"type": "image", "img_path": "images/x.jpg", "image_caption": ["Carrier logo"], "page_idx": 0},
    {"type": "header", "text": "POLICY NUMBER: CGL 1234567", "page_idx": 0},
    {"type": "footer", "text": "IL DS 00 09 08", "page_idx": 0},
    {"type": "page_number", "text": "Page 1 of 3", "page_idx": 0},
    {"type": "table", "table_body": "<table><tr><th>Vehicle</th></tr><tr><td>VIN 1</td></tr></table>",
     "table_caption": ["Schedule of Vehicles"], "table_footnote": ["* garaged in NY"], "page_idx": 2},
    {"type": "list", "sub_type": "ref_text", "list_items": ["CG 00 01 04 13", "IL 00 17 11 98"], "page_idx": 2},
    {"type": "equation", "text": "$x=1$", "page_idx": 2},
    {"type": "text", "text": "ignored: out of range", "page_idx": 9},
]


def test_blocks_become_one_markdown_string_per_page():
    pages = pages_from_content_list(CONTENT, 3)
    # The header back at the head of its page, the footer and page number at its foot.
    assert pages[0] == ("POLICY NUMBER: CGL 1234567\n\n# COMMON POLICY DECLARATIONS\n\n"
                        "Named Insured: Rivera Fabrication LLC\n\nCarrier logo\n\nIL DS 00 09 08\n\nPage 1 of 3")
    assert pages[1] == ""                          # MinerU found nothing on page 2
    assert pages[2].startswith("Schedule of Vehicles\n\n<table>")
    assert "* garaged in NY" in pages[2] and "$x=1$" in pages[2]
    assert "CG 00 01 04 13\nIL 00 17 11 98" in pages[2]      # a list, one item per line
    assert "images/x.jpg" not in pages[0]         # the picture is in the page image


def test_a_content_list_of_mineru_1_still_reads_its_image_captions():
    blocks = [{"type": "image", "img_caption": ["Carrier logo"], "img_footnote": ["est. 1901"], "page_idx": 0}]
    assert pages_from_content_list(blocks, 1) == ["Carrier logo\n\nest. 1901"]


# --------------------------------------------------------------------------
# The engine, with MinerU stood in
# --------------------------------------------------------------------------


def _pdf(pages: int) -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page(width=612, height=792).insert_text(
            (72, 72), f"page {i + 1} of a policy with a real text layer on it")
    return doc.tobytes()


def _fake_mineru(monkeypatch, content, *, scanned=False):
    """MinerU 3.x's pipeline backend, stood in: ``calls`` records how it was asked
    to read (``mode`` txt/ocr, ``ocr``, ``lang``, ``formula_enable``)."""
    calls = {}

    def doc_analyze_streaming(pdf_bytes_list, image_writer_list, lang_list, on_doc_ready,
                              parse_method="auto", formula_enable=True, table_enable=True, **_):
        calls.update(mode=parse_method, ocr=parse_method == "ocr", lang=lang_list[0],
                     formula_enable=formula_enable, table_enable=table_enable, bytes=len(pdf_bytes_list[0]))
        on_doc_ready(0, [], {"pdf_info": ["page infos"]}, parse_method == "ocr")

    def union_make(pdf_info, make_mode, image_dir=""):
        calls["make_mode"] = make_mode
        return content

    modules = {
        "mineru": types.ModuleType("mineru"),
        "mineru.backend": types.ModuleType("mineru.backend"),
        "mineru.backend.pipeline": types.ModuleType("mineru.backend.pipeline"),
        "mineru.backend.pipeline.pipeline_analyze": types.SimpleNamespace(
            doc_analyze_streaming=doc_analyze_streaming),
        "mineru.backend.pipeline.pipeline_middle_json_mkcontent": types.SimpleNamespace(union_make=union_make),
        "mineru.data": types.ModuleType("mineru.data"),
        "mineru.data.data_reader_writer": types.SimpleNamespace(FileBasedDataWriter=lambda d: d),
        "mineru.utils": types.ModuleType("mineru.utils"),
        "mineru.utils.enum_class": types.SimpleNamespace(
            MakeMode=types.SimpleNamespace(CONTENT_LIST="content_list")),
        "mineru.utils.pdf_classify": types.SimpleNamespace(
            classify=lambda pdf_bytes: "ocr" if scanned else "txt"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("common.gpu.require_cuda", lambda what: "NVIDIA H100")
    monkeypatch.setattr("data_pipeline.ocr.mineru_config.assert_on_cuda", lambda path=None: None)
    monkeypatch.setattr("data_pipeline.ocr.mineru_config.apply_run_environment",
                        lambda: calls.__setitem__("environment_applied", True))
    return calls


@pytest.mark.parametrize("scanned,mode", [(True, "ocr"), (False, "txt")])
def test_the_engine_returns_a_page_per_page_with_its_image(monkeypatch, scanned, mode):
    # The per-document rule, as read_every_page_by_ocr off leaves it.
    monkeypatch.setattr("common.config.ocr_every_page", lambda: False)
    calls = _fake_mineru(monkeypatch, CONTENT, scanned=scanned)
    pages = MinerUEngine().process(_pdf(3), device="cuda", max_long_side_px=800)
    assert [p.page_number for p in pages] == [1, 2, 3]
    assert calls["mode"] == mode and calls["ocr"] is scanned
    assert pages[0].markdown.startswith("POLICY NUMBER: CGL 1234567\n\n# COMMON POLICY DECLARATIONS")
    # An empty page is an OCR failure only on a scan; on a text layer it is blank.
    assert pages[1].ocr_failed is scanned and not pages[0].ocr_failed
    assert not any(p.scanned for p in pages)          # every page has a text layer
    assert pages[2].table_row_count == 1
    assert all(p.image_bytes.startswith(b"\x89PNG") for p in pages)


def test_the_engine_reads_in_one_pinned_way(monkeypatch):
    """On the GPU and the pinned local weights (set before MinerU is imported),
    in the pinned OCR language, with formula recognition off - policies carry no
    equations - and tables on; the content list is what the pages are built from."""
    calls = _fake_mineru(monkeypatch, CONTENT)
    MinerUEngine().process(_pdf(3), device="cuda", max_long_side_px=800)
    assert calls["environment_applied"] is True
    assert calls["lang"] == OCR_LANG == "ch"
    assert calls["formula_enable"] is False and calls["table_enable"] is True
    assert calls["make_mode"] == "content_list"


def test_a_content_list_returned_as_json_text_is_read(monkeypatch):
    import json

    _fake_mineru(monkeypatch, json.dumps(CONTENT))
    pages = MinerUEngine().process(_pdf(3), device="cuda", max_long_side_px=800)
    assert pages[0].markdown.startswith("POLICY NUMBER")


def test_the_engine_refuses_the_cpu():
    with pytest.raises(OcrError, match="GPU only"):
        MinerUEngine().process(b"%PDF", device="cpu", max_long_side_px=800)  # type: ignore[arg-type]


def test_the_engine_refuses_without_cuda(monkeypatch):
    from common.gpu import GPUError

    def no_gpu(what):
        raise GPUError(f"{what} runs on the GPU only")

    monkeypatch.setattr("common.gpu.require_cuda", no_gpu)
    with pytest.raises(OcrError, match="GPU only"):
        MinerUEngine().process(b"%PDF", device="cuda", max_long_side_px=800)
