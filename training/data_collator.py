"""Label masking — verification, not implementation (arch §10).

**ms-swift provides the collator.** It handles interleaved image/text collation
and `-100` label masking out of the box, which is a large part of why it was
chosen as the Layer-3 entrypoint. This module deliberately does not reimplement
that; it exists to **verify** it, and to hold an override hook if a custom
masking need ever arises.

Why verification earns its own module: loss must be computed **only on the
assistant tokens**. System, image, and OCR tokens are masked with ``-100``, the
label value PyTorch's loss function ignores. If that masking breaks, the model
trains on reproducing its own prompt — and the loss curve looks entirely normal
while it happens, because reproducing a prompt is an easy objective that
converges nicely. There is no other symptom until evaluation, by which point a
training run has been spent.

That is why its guard is mutation-checked (``tests/test_training.py``,
``test_masking_out_assistant_tokens_is_caught`` and its neighbours): a guard against a silent
failure is only worth having if it actually fires.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: The label value PyTorch's cross-entropy ignores. Tokens marked with it
#: contribute nothing to the loss.
IGNORE_INDEX = -100


class MaskingError(AssertionError):
    """Raised when label masking would train the model on the wrong tokens."""


@dataclass
class MaskingReport:
    """What a batch's masking actually looks like."""

    total_tokens: int = 0
    supervised_tokens: int = 0
    masked_tokens: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def supervised_fraction(self) -> float:
        return self.supervised_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def is_correct(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        return (
            f"{self.supervised_tokens}/{self.total_tokens} tokens supervised "
            f"({self.supervised_fraction:.1%}), {len(self.problems)} problem(s)"
        )


def assert_masking_correct(
    labels: Sequence[int],
    assistant_start: int,
    assistant_end: int,
    *,
    context: str = "batch",
) -> MaskingReport:
    """Assert exactly the assistant tokens are supervised, and nothing else.

    Args:
        labels: the label sequence the trainer will compute loss against.
        assistant_start: index of the first assistant token, inclusive.
        assistant_end: index one past the last assistant token.
        context: identifier used in error messages.

    Raises:
        MaskingError: with the specific tokens at fault, not a bare assertion.
    """
    report = MaskingReport(total_tokens=len(labels))

    if not 0 <= assistant_start <= assistant_end <= len(labels):
        raise MaskingError(
            f"assistant span [{assistant_start}, {assistant_end}) does not fit a row of "
            f"{len(labels)} label(s). Spans computed before truncation do not survive it — "
            "recompute them on the truncated sequence, or the check silently examines the "
            "wrong tokens."
        )

    leaked = [i for i in range(0, assistant_start) if labels[i] != IGNORE_INDEX]
    if leaked:
        report.problems.append(
            f"{len(leaked)} PROMPT token(s) are supervised (indices {leaked[:8]}). The model "
            "would be trained to reproduce its own system prompt, OCR text, or image tokens. "
            "The loss curve will look normal while this happens."
        )

    trailing = [i for i in range(assistant_end, len(labels)) if labels[i] != IGNORE_INDEX]
    if trailing:
        report.problems.append(
            f"{len(trailing)} token(s) after the assistant turn are supervised "
            f"(indices {trailing[:8]}, token ids {[labels[i] for i in trailing[:8]]})."
        )

    unsupervised = [i for i in range(assistant_start, assistant_end) if labels[i] == IGNORE_INDEX]
    if unsupervised:
        report.problems.append(
            f"{len(unsupervised)} ASSISTANT token(s) are masked out (indices {unsupervised[:8]}). "
            "The model is not being trained on part of the JSON it is supposed to produce."
        )

    report.supervised_tokens = sum(1 for label in labels if label != IGNORE_INDEX)
    report.masked_tokens = report.total_tokens - report.supervised_tokens

    if report.supervised_tokens == 0:
        report.problems.append(
            "NO tokens are supervised — every label is -100, so the loss is constant and the "
            "model learns nothing. Training will complete successfully and produce an adapter "
            "identical to its initialisation."
        )

    if report.problems:
        raise MaskingError(
            f"label masking is wrong for {context}:\n  - " + "\n  - ".join(report.problems)
        )
    return report


def verify_batch(
    batch: dict[str, Any],
    assistant_spans: Sequence[tuple[int, int]],
    *,
    context: str = "batch",
) -> MaskingReport:
    """Verify masking across a real collated batch.

    Call this on a sample batch from the configured collator **before** launching
    a training run. It is cheap, and it is the only point at which a masking
    error is recoverable rather than costing a whole run.
    """
    labels = batch.get("labels")
    if labels is None:
        raise MaskingError(
            f"{context}: the batch has no `labels`. Without them nothing is supervised and "
            "training silently does nothing."
        )

    rows = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [rows]

    if len(rows) != len(assistant_spans):
        raise MaskingError(
            f"{context}: {len(rows)} row(s) in the batch but {len(assistant_spans)} assistant "
            "span(s) supplied — they must correspond one to one."
        )

    combined = MaskingReport()
    for index, (row, (start, end)) in enumerate(zip(rows, assistant_spans, strict=True)):
        row_report = assert_masking_correct(row, start, end, context=f"{context}[row {index}]")
        combined.total_tokens += row_report.total_tokens
        combined.supervised_tokens += row_report.supervised_tokens
        combined.masked_tokens += row_report.masked_tokens

    log.info("masking verified for %s: %s", context, combined.summary())
    return combined


def reference_labels(
    input_ids: Sequence[int],
    assistant_start: int,
    assistant_end: int,
) -> list[int]:
    """Build correctly-masked labels for a sequence.

    A **reference**, not the production path — ms-swift builds the real labels.
    It exists so tests have a known-correct baseline to compare a collator
    against, and so the intended behaviour is expressed once in code rather than
    only in prose.
    """
    return [
        token if assistant_start <= index < assistant_end else IGNORE_INDEX
        for index, token in enumerate(input_ids)
    ]


def find_assistant_span(
    input_ids: Sequence[int],
    assistant_header_ids: Sequence[int],
    end_token_id: int | None = None,
    end_suffix_ids: Sequence[int] = (),
) -> tuple[int, int]:
    """Locate the assistant turn in a tokenised sequence.

    Chat templates vary between model families, so the header token ids are
    passed in rather than hardcoded — a wrong guess here would mask the wrong
    span, which is the failure this module exists to prevent.

    ``end_suffix_ids`` are what the template writes right after the end token as
    part of the answer: ms-swift's ChatML suffix is ``<|im_end|>\n``, and it
    supervises that newline. The span takes them in only when exactly they
    follow the end token, so anything else supervised after it is still caught.
    """
    header = list(assistant_header_ids)
    if not header:
        raise MaskingError("assistant_header_ids is empty; the assistant turn cannot be located")

    ids = list(input_ids)
    for start in range(len(ids) - len(header) + 1):
        if ids[start:start + len(header)] == header:
            content_start = start + len(header)
            if end_token_id is not None:
                for end in range(content_start, len(ids)):
                    if ids[end] == end_token_id:
                        # Half-open, and the end token is INSIDE it. Excluding it
                        # meant the reference labels never supervised EOS, so a
                        # model trained against this baseline is never taught to
                        # stop — generation runs to max_new_tokens, which
                        # `Generation.truncated()` reports as dropped rows. It
                        # also made verify_batch reject a correct ms-swift batch,
                        # which does supervise it.
                        stop = end + 1
                        suffix = list(end_suffix_ids)
                        if suffix and ids[stop:stop + len(suffix)] == suffix:
                            stop += len(suffix)
                        return content_start, stop
            return content_start, len(ids)

    raise MaskingError(
        "the assistant turn was not found in the sequence. Either the chat template changed or "
        "the header token ids are wrong — both would mask the wrong span."
    )


def verify_staged_rows(
    files: Sequence[str],
    *,
    encode: Any,
    assistant_header_ids: Sequence[int],
    end_token_id: int | None,
    end_suffix_ids: Sequence[int] = (),
    sample: int = 4,
) -> MaskingReport:
    """Verify masking on real staged rows, encoded the way the trainer encodes them.

    ``encode(row) -> {"input_ids": [...], "labels": [...]}`` is ms-swift's own
    template encode on the pod. The first ``sample`` rows are checked: masking is
    a property of the template, not of any one row, so a handful proves it — and
    this is the last point at which a masking error costs minutes, not a run.
    """
    import json
    from pathlib import Path

    rows: list[dict[str, Any]] = []
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= sample:
                break
        if len(rows) >= sample:
            break
    if not rows:
        raise MaskingError(f"no staged rows in {list(files)} to verify masking on")

    combined = MaskingReport()
    for index, row in enumerate(rows):
        encoded = encode(row)
        span = find_assistant_span(encoded["input_ids"], assistant_header_ids, end_token_id,
                                   end_suffix_ids)
        report = verify_batch({"labels": [list(encoded["labels"])]}, [span],
                              context=f"staged row {index}")
        combined.total_tokens += report.total_tokens
        combined.supervised_tokens += report.supervised_tokens
        combined.masked_tokens += report.masked_tokens
    return combined


# --------------------------------------------------------------------------
# Override hook
# --------------------------------------------------------------------------

def custom_collator(*_args: Any, **_kwargs: Any):
    """Placeholder for a custom collator.

    **Not used.** ms-swift's collator handles interleaved image/text batching and
    `-100` masking, and reimplementing that would mean hand-writing exactly the
    plumbing the Layer-3 choice was made to avoid (arch §10).

    If a genuine custom masking need ever arises, implement it here and verify it
    with :func:`verify_batch` — the verification above applies to any collator,
    which is the point of keeping the two separate.
    """
    raise NotImplementedError(
        "No custom collator is needed: ms-swift provides multimodal collation and -100 masking "
        "(arch §10). This hook exists only for a genuine override, and anything implemented here "
        "must pass verify_batch() before it is used for a real run."
    )
