"""Token estimates a window is sized by (review A1: moved from ``data_pipeline.dataset_builder.cap_check``,
which re-exports them)."""

from __future__ import annotations

from common.config import vision_for_task
from common.token_math import PIXELS_PER_VISUAL_TOKEN

#: Characters per token for English prose and Markdown tables. Real tokenizers
#: average nearer 4; 3.5 rounds against the budget, which is the safe direction.
CHARS_PER_TOKEN = 3.5

#: Fixed overhead for chat template scaffolding, role markers and image
#: delimiters — small, constant, and cheaper to over-book than to model.
TEMPLATE_OVERHEAD_TOKENS = 64

def estimate_text_tokens(text: str | None) -> int:
    """Pessimistic token count for a string.

    Digits are counted one token each, the rest at :data:`CHARS_PER_TOKEN`.
    Qwen's tokenizer splits numbers into single digits, so a flat characters-
    per-token rate UNDER-counts exactly the text that matters most here — a
    target of dates, amounts and policy numbers (``"04/01/2026"`` is ten tokens,
    not three) — which is the direction this module exists to never err in.
    """
    if not text:
        return 0
    digits = sum(ch.isdigit() for ch in text)
    return digits + int((len(text) - digits) / CHARS_PER_TOKEN) + 1

def estimate_visual_tokens(pages: int, task: str) -> int:
    """Visual tokens for ``pages`` pages at this task's pixel budget.

    Uses ``max_pixels`` — the budget, not the actual page area — because a page
    smaller than the budget costs less and a page larger is downscaled to it.
    Assuming every page fills its budget is the pessimistic direction.
    """
    if pages <= 0:
        return 0
    max_pixels = int(vision_for_task(task)["max_pixels"])
    return pages * (max_pixels // PIXELS_PER_VISUAL_TOKEN)
