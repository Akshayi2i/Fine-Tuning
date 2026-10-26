"""The prompt's length a server reports, held to what its pages imply.

A stock ``vllm serve`` resizes each page to the pixel budget the request names
(``mm_processor_kwargs``) - when it honours it. A server that ignores the
request's budget (an older build, a processor taking another key), or a page
that reaches it at another size, shows the model pages at a resolution it was
never trained on, and nothing fails: the answer is just worse. The server says
how long each prompt was (``usage.prompt_tokens``); this says how long it
should have been.

* **Images** are counted from each page's size by the resize rule both sides
  run (``training.length_check.image_tokens``) at the budget the request sends
  (``common.config.pixel_budget``) - exact, no pixel decoded.
* **Text** is counted exactly with the model's tokenizer when one is on this
  machine (``common.config.base_model_dir``); the prompt is then expected to
  within a few tokens of template overhead.
* With no tokenizer the text cannot be counted, only bounded: at least nothing
  and at most one token a character. The prompt must then lie between the
  image tokens alone and the image tokens plus every character - outside it,
  the pages were resized to another budget. The ratio of the prompt to the
  image tokens is recorded either way.

Pure apart from reading each page's PNG header; used by
``serving.pipeline`` per window (``window_stats``' ``prompt_check``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Exact text and image counts may still differ from the server's by the chat
#: template's own tokens; within this share (or :data:`SLACK_TOKENS`) they agree.
SLACK_SHARE = 0.03
SLACK_TOKENS = 64


@lru_cache(maxsize=4096)
def _png_size(path: str) -> tuple[int, int] | None:
    from training.length_check import LengthCheckError, png_size

    try:
        return png_size(path)
    except (OSError, LengthCheckError):
        return None


@lru_cache(maxsize=1)
def _tokenizer() -> Any:
    """The base model's tokenizer when it is on this machine, else None."""
    from common.config import base_model_dir

    try:
        model_dir = base_model_dir()
    except Exception:  # noqa: BLE001 - no config, no tokenizer: the check bounds instead
        return None
    if model_dir is None:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model_dir))
    except Exception as exc:  # noqa: BLE001 - optional: the check bounds the text instead
        log.info("no tokenizer for the prompt-length check (%s); the text is bounded, not counted", exc)
        return None


def _parts(messages: Iterable[Mapping[str, Any]]) -> tuple[list[str], list[Any]]:
    texts: list[str] = []
    images: list[Any] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
            continue
        for block in content or ():
            if block.get("type") == "text":
                texts.append(str(block.get("text") or ""))
            elif block.get("type") == "image":
                images.append(block.get("image"))
    return texts, images


def expected_prompt(messages: list[Mapping[str, Any]], *, budget: tuple[int, int] | None = None,
                    tokenizer: Any = "auto") -> dict[str, Any] | None:
    """What a prompt of ``messages`` should cost: ``{"image_tokens", "text_chars",
    "text_tokens", "expected", "low", "high"}``. ``text_tokens`` and
    ``expected`` are None without a tokenizer; ``low``/``high`` bound the
    prompt either way. None when a page's size cannot be read (bytes, a URL,
    a missing file): nothing to hold the server to."""
    from training.length_check import TOKENS_PER_IMAGE_WRAPPER, TOKENS_PER_MESSAGE, image_tokens

    if budget is None:
        from common.config import pixel_budget

        budget = pixel_budget()
    min_pixels, max_pixels = budget
    texts, images = _parts(messages)
    visual = 0
    for image in images:
        size = _png_size(str(image)) if isinstance(image, (str, Path)) else None
        if size is None:
            return None
        visual += image_tokens(*size, min_pixels=min_pixels, max_pixels=max_pixels)
    wrappers = TOKENS_PER_IMAGE_WRAPPER * len(images) + TOKENS_PER_MESSAGE * len(messages)
    chars = sum(len(text) for text in texts)
    counter = _tokenizer() if tokenizer == "auto" else tokenizer
    text_tokens: int | None = None
    if counter is not None:
        try:
            text_tokens = sum(len(counter.encode(text, add_special_tokens=False)) for text in texts if text)
        except Exception:  # noqa: BLE001 - a tokenizer that cannot count leaves the bounds
            text_tokens = None
    if text_tokens is not None:
        expected = visual + text_tokens
        slack = max(SLACK_TOKENS, int(SLACK_SHARE * expected)) + wrappers
        low, high = expected - slack, expected + slack
    else:
        expected = None
        low, high = visual, visual + chars + wrappers
    return {"image_tokens": visual, "text_chars": chars, "text_tokens": text_tokens,
            "expected": expected, "low": low, "high": high}


def check_prompt(prompt_tokens: int | None, expected: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The server's ``prompt_tokens`` against :func:`expected_prompt`:
    ``{"prompt_tokens", "image_tokens", "expected", "ratio", "plausible"}`` -
    ``ratio`` the prompt over the image tokens. None when either is unknown."""
    if prompt_tokens is None or expected is None:
        return None
    images = expected["image_tokens"]
    return {
        "prompt_tokens": int(prompt_tokens),
        "image_tokens": images,
        "expected": expected["expected"],
        "ratio": round(prompt_tokens / images, 3) if images else None,
        "plausible": expected["low"] <= prompt_tokens <= expected["high"],
    }
