"""Train/serve token parity: what is compared (arch §10.2).

The GPU parity test (``tests/test_parity.py::test_template_token_parity``) and
the Phase 0 spike encode one row through ms-swift's training template and
through the serving side (vLLM, or the model's processor), then compare the
prompt token ids and each image's token count. The comparison itself is here,
pure, so it is tested without a GPU; only the two encoders need the pod.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from training.stage_data import SWIFT_IMAGE_TAG

#: The token every visual token of a page is written as, in Qwen's chat template.
IMAGE_PAD = "<|image_pad|>"

#: Where the prompt ends: training encodes prompt + answer, serving the prompt
#: and a generation header, and the two agree only up to here.
ASSISTANT_HEADER = "<|im_start|>assistant\n"


class ParityError(ValueError):
    """Raised when a row cannot be compared as it stands."""


def prompt_ids(input_ids: Sequence[int], header_ids: Sequence[int]) -> list[int]:
    """The prompt's tokens: up to and including the last assistant header."""
    ids, header = [int(t) for t in input_ids], [int(t) for t in header_ids]
    if not header:
        raise ParityError("an empty assistant header matches everywhere")
    for start in range(len(ids) - len(header), -1, -1):
        if ids[start:start + len(header)] == header:
            return ids[:start + len(header)]
    raise ParityError("the encoded row holds no assistant header")


def image_pad_runs(ids: Sequence[int], pad_id: int) -> list[int]:
    """Each image's visual-token count: the lengths of the runs of ``pad_id``."""
    runs: list[int] = []
    length = 0
    for token in ids:
        if token == pad_id:
            length += 1
        elif length:
            runs.append(length)
            length = 0
    if length:
        runs.append(length)
    return runs


def from_swift_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    """A staged ms-swift row back in the serving shape (``to_swift_row`` undone).

    Each ``<image>`` placeholder becomes an image block holding the row's next
    local path, in order, so the serving side reads the same pages in the same
    places the trainer did.
    """
    images = list(row.get("images") or [])
    used = 0
    messages: list[dict[str, Any]] = []
    for message in row.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, str) or SWIFT_IMAGE_TAG not in content:
            messages.append({"role": message["role"], "content": content})
            continue
        blocks: list[dict[str, Any]] = []
        for index, text in enumerate(content.split(SWIFT_IMAGE_TAG)):
            if index:
                if used >= len(images):
                    raise ParityError("the row holds more <image> tags than images")
                blocks.append({"type": "image", "image": images[used]})
                used += 1
            if text:
                blocks.append({"type": "text", "text": text})
        messages.append({"role": message["role"], "content": blocks})
    if used != len(images):
        raise ParityError(f"the row lists {len(images)} images for {used} <image> tags")
    return messages


def flat_ids(encoded: Any) -> list[int]:
    """Token ids as one flat list, from a list, a batch of one, or a tensor."""
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    ids = list(encoded)
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ParityError(f"expected one sequence, got a batch of {len(ids)}")
        ids = list(ids[0])
    return [int(t) for t in ids]
