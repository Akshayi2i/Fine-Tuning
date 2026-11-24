"""How many tokens a page costs the model: the arithmetic training and serving share.

Training measures every staged row with it before launch
(``training.length_check``); serving holds each window's reported prompt length
to it (``inference_core.prompt_tokens``). It lives in ``common`` because the
served image (``azure_service/Dockerfile``) carries ``common/`` and not
``training/``: kept in ``training``, the served check could not import it and
fell silent in production (review R4, 23 November 2026).

Pure: reading a PNG's 24-byte header is the only I/O.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

#: Pixels per side of one visual token: 16px patches merged 2x2. The same
#: geometry ``cap_check.PIXELS_PER_VISUAL_TOKEN`` assumes (32 * 32).
RESIZE_FACTOR = 32

#: Chat-template tokens around each message (role tags, separators) and around
#: each image (vision start/end). Rounded up: over-counting refuses a row an
#: operator can look at, under-counting stops a paid run.
TOKENS_PER_MESSAGE = 8
TOKENS_PER_IMAGE_WRAPPER = 4


#: Pixels per visual token - 16px patches with a 2x2 spatial merge. Moved here from ``data_pipeline.dataset_builder.cap_check``
#: (review A1, 2026-11-24), which re-exports it: ``common.config`` checks the
#: pixel budget with it and may not import build-time code.
PIXELS_PER_VISUAL_TOKEN = 32 * 32


class LengthCheckError(RuntimeError):
    """Raised when a staged row or a page cannot be measured."""


def png_size(path: str | Path) -> tuple[int, int]:
    """``(width, height)`` from a PNG's header, without decoding the image."""
    with open(path, "rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise LengthCheckError(f"{path} is not a PNG with a readable header")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def image_tokens(
    width: int, height: int, *, min_pixels: int, max_pixels: int, factor: int = RESIZE_FACTOR
) -> int:
    """Visual tokens for one page, by Qwen's ``smart_resize`` rule.

    Each side is rounded to the factor; an area above ``max_pixels`` is scaled
    down, one below ``min_pixels`` scaled up, keeping the aspect ratio.
    """
    if width <= 0 or height <= 0:
        raise LengthCheckError(f"image has no area ({width}x{height})")
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
    return (h_bar * w_bar) // (factor * factor)
