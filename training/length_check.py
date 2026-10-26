"""Every staged row's real length, measured on the pod before launch.

The corpus build sizes rows with a character-based estimate (``cap_check``),
deliberately pessimistic because it runs with no model. That is the right tool
there and the wrong last word: a row the estimate under-counts reaches the
trainer, and under ``strict`` encoding — which the run uses, so a bad row cannot
be silently swapped for another — it stops the run mid-training, hours in.

So the count is taken again on the pod, where the tokenizer exists, before
``swift sft`` starts:

* **Text** is counted with the model's own tokenizer — exact, and fast.
* **Images** are counted from each page's dimensions through Qwen's resize rule —
  the same arithmetic the image processor runs, without decoding a pixel.
* **Template overhead** is a fixed margin per message and per image, on the
  pessimistic side.

Running every row through the full processor would be exact to the token, and
would idle a paid GPU for most of an hour on image preprocessing alone. This is
the difference between those two, spent on the part that is cheap to get right.
The resize factor and margins are confirmed by the Phase 0 spike
(``visual_token_geometry``, ``swift_image_budget``).
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: Pixels per side of one visual token: 16px patches merged 2x2. The same
#: geometry ``cap_check.PIXELS_PER_VISUAL_TOKEN`` assumes (32 * 32).
RESIZE_FACTOR = 32

#: Chat-template tokens around each message (role tags, separators) and around
#: each image (vision start/end). Rounded up: over-counting refuses a row an
#: operator can look at, under-counting stops a paid run.
TOKENS_PER_MESSAGE = 8
TOKENS_PER_IMAGE_WRAPPER = 4

#: Counts tokens for a batch of strings. The model's tokenizer on the pod; a
#: stub in tests.
TokenCounter = Callable[[list[str]], list[int]]


class LengthCheckError(RuntimeError):
    """Raised when a staged row cannot be measured."""


#: Width of one bar of the row-length histogram, in tokens.
HISTOGRAM_BUCKET = 4096

#: The repeating-table groups. When their answers are more than
#: :data:`HEAVY_GROUP_SHARE` of every supervised token, the declarations'
#: fields - one value each, short answers - get a small share of the gradient.
HEAVY_GROUPS = ("arrays", "lineblk")
HEAVY_GROUP_SHARE = 0.6


@dataclass
class LengthReport:
    """What was measured, and what does not fit.

    Besides the fit check, the training rows' shape (the files ``measure``
    was told to count, the epoch files - not the validation sample): a
    row-length histogram and the supervised tokens - the answer, the
    assistant turn's text - per section group.
    """

    rows: int = 0
    longest: int = 0
    over: list[dict[str, object]] = field(default_factory=list)
    #: Rows counted in the statistics below, and their tokens.
    counted_rows: int = 0
    counted_tokens: int = 0
    #: Lower edge of each :data:`HISTOGRAM_BUCKET`-wide bar -> rows.
    histogram: dict[int, int] = field(default_factory=dict)
    #: Section group -> {"rows": n, "tokens": supervised tokens}.
    supervised: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.over

    @property
    def supervised_tokens(self) -> int:
        return sum(group["tokens"] for group in self.supervised.values())

    def heavy_share(self) -> float:
        """The share of supervised tokens in :data:`HEAVY_GROUPS`; 0 with none."""
        total = self.supervised_tokens
        heavy = sum(self.supervised.get(g, {}).get("tokens", 0) for g in HEAVY_GROUPS)
        return heavy / total if total else 0.0

    def warnings(self) -> list[str]:
        """What an operator should hear before the run: today, a supervised
        token mix the repeating tables dominate."""
        share = self.heavy_share()
        if share <= HEAVY_GROUP_SHARE:
            return []
        return [
            f"{share:.0%} of the supervised tokens are in {' and '.join(HEAVY_GROUPS)} rows (over "
            f"{HEAVY_GROUP_SHARE:.0%}): the loss is a mean over answer tokens, so the declarations' "
            "fields - short, one value each - may be under-weighted. Compare per-group accuracy "
            "before reading the overall number."
        ]

    def summary(self) -> dict[str, object]:
        """The statistics as JSON, for the run manifest and benchmark.json."""
        total = self.supervised_tokens
        return {
            "rows": self.counted_rows,
            "longest": self.longest,
            "mean_tokens": round(self.counted_tokens / self.counted_rows) if self.counted_rows else 0,
            "histogram_bucket_tokens": HISTOGRAM_BUCKET,
            "length_histogram": {str(edge): n for edge, n in sorted(self.histogram.items())},
            "supervised_tokens": total,
            "supervised_by_group": {
                group: {**counts, "share": round(counts["tokens"] / total, 4) if total else 0.0}
                for group, counts in sorted(self.supervised.items())
            },
            "heavy_group_share": round(self.heavy_share(), 4),
            "warnings": self.warnings(),
        }


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


def measure(
    files: Sequence[str | Path],
    *,
    max_length: int,
    count_tokens: TokenCounter,
    min_pixels: int,
    max_pixels: int,
    batch: int = 256,
    output_caps: dict[str, list[int]] | None = None,
    groups: dict[str, list[str]] | None = None,
    stats_files: Sequence[str | Path] | None = None,
) -> LengthReport:
    """Measure every row of the staged (ms-swift format) files.

    Against ``max_length``, and — when ``output_caps`` gives each row's task
    reservation — the target (assistant turn) against its own cap too. A target
    past its reservation is the case that clips an answer, and the corpus
    estimate under-counts exactly that text (digits tokenize one apiece).

    The rows of ``stats_files`` (default: every file) are also counted into the
    report's histogram and supervised-token totals, each row under its section
    group from ``groups`` (per file, in row order, as staging records them;
    ``unknown`` without).
    """
    report = LengthReport()
    caps_by_file = output_caps or {}
    groups_by_file = groups or {}
    counted = None if stats_files is None else {str(Path(f).resolve()) for f in stats_files}
    # One header read per image. Every epoch file names the same pages, so an
    # uncached pass read each page's header once per epoch — thousands of
    # small reads off a network volume before a launch.
    sizes: dict[str, tuple[int, int]] = {}

    def size_of(image: str) -> tuple[int, int]:
        if image not in sizes:
            sizes[image] = png_size(image)
        return sizes[image]

    for path in files:
        caps = caps_by_file.get(str(path)) or caps_by_file.get(str(Path(path).resolve()))
        row_groups = groups_by_file.get(str(path)) or groups_by_file.get(str(Path(path).resolve()))
        counts = counted is None or str(Path(path).resolve()) in counted
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            texts = [
                "".join(m["content"].replace("<image>", "") for m in row["messages"])
                for row in chunk
            ]
            targets = [
                row["messages"][-1]["content"] if row["messages"][-1]["role"] == "assistant" else ""
                for row in chunk
            ]
            # Always counted: the answer is what the run supervises, so its
            # tokens per group are the run's weighting (the length report).
            target_tokens = count_tokens(targets)
            for index, (row, text_tokens, output_tokens) in enumerate(
                zip(chunk, count_tokens(texts), target_tokens, strict=True)
            ):
                visual = sum(
                    image_tokens(*size_of(image), min_pixels=min_pixels, max_pixels=max_pixels)
                    + TOKENS_PER_IMAGE_WRAPPER
                    for image in row["images"]
                )
                total = text_tokens + visual + TOKENS_PER_MESSAGE * len(row["messages"])
                report.rows += 1
                report.longest = max(report.longest, total)
                if counts:
                    group = row_groups[start + index] if row_groups else "unknown"
                    tally = report.supervised.setdefault(group, {"rows": 0, "tokens": 0})
                    tally["rows"] += 1
                    tally["tokens"] += output_tokens
                    edge = (total // HISTOGRAM_BUCKET) * HISTOGRAM_BUCKET
                    report.histogram[edge] = report.histogram.get(edge, 0) + 1
                    report.counted_rows += 1
                    report.counted_tokens += total
                cap = caps[start + index] if caps else None
                if cap is not None and output_tokens > cap:
                    report.over.append({
                        "file": str(path), "row": start + index, "output_tokens": output_tokens,
                        "output_cap": cap, "tokens": total,
                    })
                elif total > max_length:
                    report.over.append({
                        "file": str(path), "row": start + index, "tokens": total,
                        "text_tokens": text_tokens, "visual_tokens": visual,
                        "images": len(row["images"]),
                    })
    return report


def tokenizer_counter(model_id: str, revision: str | None = None) -> TokenCounter:
    """The model's own tokenizer as a :data:`TokenCounter`. Loaded on the pod."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True
    )

    def count(texts: list[str]) -> list[int]:
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encoded]

    return count
