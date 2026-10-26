"""One resize of every page, in the shared renderer (review 2026-10-26, #7).

Under the fast profile a page rendered at the 1792 px cap (~2.48 MP) is over
the 1.25 MP budget, and was downscaled twice over - by qwen-vl-utils (PIL) at
training and by the HF processor (torch) in vLLM. The renderer now resizes it
once, to the size both processors compute, so neither resamples it again. The
pixel comparison itself needs the pod (tests/test_parity.py::test_pixel_parity).
"""

from __future__ import annotations

import io

import pytest

from data_pipeline.ocr.render_only import smart_resize


@pytest.mark.parametrize(("height", "width"), [(1792, 1385), (1385, 1792), (1792, 1100), (2318, 1792),
                                               (300, 200), (1188, 1188)])
def test_a_resized_page_is_a_size_the_processors_leave_as_it_is(height, width):
    """smart_resize's answer is a fixed point: the processors, given a page
    that size, compute the same size back - an identity resize, no resampling."""
    from training.length_check import image_tokens

    for budget in ((200704, 1254400), (200704, 2483200)):
        h, w = smart_resize(height, width, min_pixels=budget[0], max_pixels=budget[1])
        assert h % 32 == 0 and w % 32 == 0 and budget[0] <= h * w <= budget[1]
        assert smart_resize(h, w, min_pixels=budget[0], max_pixels=budget[1]) == (h, w)
        # The token count the sequence caps are built on is unchanged by the resize.
        assert image_tokens(w, h, min_pixels=budget[0], max_pixels=budget[1]) == \
            image_tokens(width, height, min_pixels=budget[0], max_pixels=budget[1]) == h * w // 1024


def _pdf(sizes):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for width, height in sizes:
        page = document.new_page(width=width, height=height)
        page.insert_text((72, 72), "POLICY NUMBER PA-2026-0042  Total Premium $1,665.00", fontsize=10)
    return document.tobytes()


def _sizes(pngs):
    from PIL import Image

    return [Image.open(io.BytesIO(png)).size for png in pngs]


def test_the_fast_profile_renders_each_page_at_its_processed_size():
    from common.config import pixel_budget
    from data_pipeline.ocr.render_only import page_pixel_budget, render_pdf_pages

    assert page_pixel_budget() == pixel_budget() == (200704, 1254400)       # the shipped profile: fast
    pngs = render_pdf_pages(_pdf([(612, 792), (792, 612)]), 1792)
    assert _sizes(pngs) == [(960, 1248), (1248, 960)]
    for width, height in _sizes(pngs):
        assert smart_resize(height, width, min_pixels=200704, max_pixels=1254400) == (height, width)


def test_the_same_page_renders_to_the_same_bytes_in_one_process_or_several():
    from data_pipeline.ocr.render_only import render_pdf_pages

    pdf = _pdf([(612, 792), (792, 612), (612, 1008)])
    assert render_pdf_pages(pdf, 1792) == render_pdf_pages(pdf, 1792, workers=2)


@pytest.mark.format_profile("standard")
def test_the_standard_profile_renders_as_it_always_has():
    from data_pipeline.ocr.render_only import page_pixel_budget, render_pdf_pages

    assert page_pixel_budget() is None
    pngs = render_pdf_pages(_pdf([(612, 792)]), 1792)
    assert _sizes(pngs) == [(1385, 1792)]                                    # the long side on the cap


def test_a_render_records_the_budget_its_pages_were_resized_to(monkeypatch):
    """A page rendered before the resize, or under another profile, is other
    pixels: re-rendered, as a cap change re-renders."""
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from data_pipeline.ocr import mineru_version, render_only

    monkeypatch.setattr(mineru_version, "cuda_available", lambda: (True, "NVIDIA L4"))   # a GPU pod
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw", context="ocr")
    client.write_bytes(paths.raw_pdf("policy", "p1"), b"%PDF-1.7")
    client.write_json(paths.raw_metadata("policy", "p1"), {"source_id": "p1", "checksum_sha256": "x"})
    calls = []
    monkeypatch.setattr(render_only, "render_pdf_pages", lambda pdf, cap: calls.append(cap) or [b"\x89PNG"])

    meta = render_only.render_document("policy", "p1", client)
    assert meta["page_pixel_budget"] == [200704, 1254400]
    render_only.render_document("policy", "p1", client)
    assert len(calls) == 1                                                   # same cap, same budget: kept

    stale = {**client.read_json(paths.ocr_meta("policy", "p1")), "page_pixel_budget": None}
    client.write_json(paths.ocr_meta("policy", "p1"), stale)
    render_only.render_document("policy", "p1", client)
    assert len(calls) == 2
