"""A PDF's pages rendered as page images at the pixel budget (review A1: moved from
``data_pipeline.ocr.render_only``, which re-exports them and keeps its command-line stage). The served
MinerU client, the OCR stage and training render pages with these same functions."""

from __future__ import annotations

from typing import Any


class RenderError(RuntimeError):
    """Raised when a PDF cannot be rendered to page images."""

#: Pixels per side of one visual token (16 px patches merged 2x2): the factor
#: Qwen's ``smart_resize`` rounds each side to (``training.length_check.RESIZE_FACTOR``).
RESIZE_FACTOR = 32

#: ``budget`` not given: the active format profile's (:func:`page_pixel_budget`).
_ACTIVE: Any = object()

def page_pixel_budget() -> tuple[int, int] | None:
    """``(min_pixels, max_pixels)`` pages are resized to here, once, for every
    reader: the active format profile's budget (``common.config.pixel_budget``)
    under a profile other than standard; None under standard, whose pages are
    rendered as they always were."""
    from common.config import fast_profile, pixel_budget

    return pixel_budget() if fast_profile() else None

def smart_resize(height: int, width: int, *, min_pixels: int, max_pixels: int,
                 factor: int = RESIZE_FACTOR) -> tuple[int, int]:
    """``(height, width)`` a Qwen-VL image processor resizes a page to: each
    side a multiple of ``factor``, the area within ``[min_pixels, max_pixels]``,
    the aspect kept - the rule of ``qwen_vl_utils.smart_resize`` (ms-swift, at
    training) and of transformers' Qwen2-VL processor (vLLM, at serving), and of
    ``training.length_check.image_tokens``. A page already that size is resized
    by neither: both compute the same size back."""
    import math

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def render_pdf_pages(pdf_bytes: bytes, max_long_side_px: int, workers: int = 1,
                     budget: Any = _ACTIVE) -> list[bytes]:
    """Render each page to PNG with its long side at ``max_long_side_px``.

    Aspect ratio is preserved. The scale factor converts PDF points (72 per inch)
    to pixels, so a US Letter page at a 1792px cap renders at roughly 163 DPI.

    Under a format profile with its own pixel budget (the fast profile's 1.25
    MP; ``budget``, default :func:`page_pixel_budget`), each page is then
    resized ONCE, here, to the size the model's image processors would resize
    it to (:func:`smart_resize`), with one filter (bicubic, the processors'
    own). A 1792 px render is about 2.48 MP; left to the readers, ms-swift's
    qwen-vl-utils (PIL) downscaled it at training and transformers' processor
    in vLLM (torch) at serving - two resamplers, two sets of pixels for one
    page. Pre-resized, both compute the same size back and leave the pixels as
    they are, so training and serving see the same pixels
    (tests/test_parity.py measures it on the pod). Under the standard profile
    (None) the pages are rendered as before.

    ``workers`` above 1 renders runs of pages in that many processes at once -
    a 114-page scan took 93 s in one - each page exactly as one process renders
    it, so the images are the same bytes either way.

    Note what this does **not** do: a page whose content is an embedded 100-DPI
    scan is still rendered to the cap, which upsamples that raster and spends
    vision tokens on no extra information. Avoiding that means inspecting the
    embedded image resolution per page, which is a separate concern from the cap
    and is not attempted here.
    """
    if budget is _ACTIVE:
        budget = page_pixel_budget()
    budget = tuple(budget) if budget is not None else None
    count = _page_count(pdf_bytes)
    workers = max(1, min(workers, count))
    if workers == 1:
        images = _render_range(pdf_bytes, max_long_side_px, 0, count, budget)
    else:
        from concurrent.futures import ProcessPoolExecutor

        bounds = [count * part // workers for part in range(workers + 1)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = pool.map(_render_range, [pdf_bytes] * workers, [max_long_side_px] * workers,
                            bounds[:-1], bounds[1:], [budget] * workers)
            images = [image for run in runs for image in run]
    if not images:
        raise RenderError("PDF produced no pages")
    return images

def _resized_png(pixmap: Any, budget: tuple[int, int]) -> bytes:
    """A rendered page as PNG at the size the image processors resize it to
    (:func:`smart_resize`): bicubic, once; as rendered when it is that size."""
    import io

    from PIL import Image

    min_pixels, max_pixels = budget
    height, width = smart_resize(pixmap.height, pixmap.width, min_pixels=min_pixels, max_pixels=max_pixels)
    if (height, width) == (pixmap.height, pixmap.width):
        return pixmap.tobytes("png")
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(pixmap.n)
    if mode is None:
        raise RenderError(f"a rendered page has {pixmap.n} channels; expected 1, 3 or 4")
    image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    out = io.BytesIO()
    resized.save(out, format="PNG")
    return out.getvalue()

def _fitz() -> Any:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RenderError(
            'PyMuPDF is not installed. Install the [data] extra: pip install -e ".[data]"'
        ) from exc
    return fitz

def _page_count(pdf_bytes: bytes) -> int:
    """The page count; a PDF that needs a password to open is told apart from
    a broken one (``common.pdf_access``): PyMuPDF's own words for it are
    "document closed or encrypted", on the first page. One that only restricts
    printing or copying renders as it is."""
    from common.pdf_access import NEEDS_PASSWORD, UNLOCK_HINT, PdfLocked

    with _fitz().open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise PdfLocked(f"{NEEDS_PASSWORD}; {UNLOCK_HINT}")
        return doc.page_count

def _render_range(pdf_bytes: bytes, max_long_side_px: int, start: int, stop: int,
                  budget: tuple[int, int] | None = None) -> list[bytes]:
    """Pages ``start`` to ``stop - 1`` (0-based) as PNG: one process's share;
    each resized to ``budget`` when one is given (:func:`render_pdf_pages`)."""
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
            images.append(pixmap.tobytes("png") if budget is None else _resized_png(pixmap, budget))
    return images
