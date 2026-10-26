"""Sequence-budget enforcement at corpus build time (arch v2.1 §7a).

**Reject, never truncate.** That rule is the whole point of this module.

A truncated training example is not a smaller example — it is a *wrong* one. Cut
the tail off a Loss Run's assistant span and the target becomes a JSON document
that stops after eleven claims, so the model is trained to stop after eleven
claims. The row still parses, the loss still falls, and the defect surfaces months
later as unexplained row-recall loss on long documents.

v1 had no check at all. It carried a single 8192-token cap for every task while
a US-Letter page at the configured resolution costs roughly 2,400 visual tokens,
so a three-page Loss Run overflowed before a single OCR token was counted, and
nothing anywhere compared the two numbers.

Estimation, not tokenization. Running the real tokenizer over every candidate row
would need the model on a CPU-only build box. The estimate is deliberately
**pessimistic** — it rounds against the budget at every step — because the cost
of over-estimating is that a row is rejected and an operator sees it, while the
cost of under-estimating is silent truncation on the pod.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from common.config import sequence_for_task, vision_for_task

log = logging.getLogger(__name__)

#: Characters per token for English prose and Markdown tables. Real tokenizers
#: average nearer 4; 3.5 rounds against the budget, which is the safe direction.
CHARS_PER_TOKEN = 3.5

#: How far a real answer runs past its estimate, at most. Answer JSON is
#: punctuation and short keys ("raw", "parsed", "page_ref"), which the base
#: model's tokenizer splits finer than prose: over every window answer of the
#: SPEC_21 set, real tokens were 1.12x the estimate typically and 1.22x at the
#: worst (2026-10-07). Applied only where the tokenizer itself is not at hand.
TARGET_ESTIMATE_FACTOR = 1.25

#: Pixels per visual token — 16px patches with a 2x2 spatial merge. Every cap
#: in sequence.yaml derives from this number. Measured on the pod by
#: ``tests/test_parity.py::test_template_token_parity`` (``pytest -m gpu``), which
#: fails if the processor's patch x merge geometry or any page's token count
#: disagrees; record the date it first passes here.
PIXELS_PER_VISUAL_TOKEN = 32 * 32

#: Fixed overhead for chat template scaffolding, role markers and image
#: delimiters — small, constant, and cheaper to over-book than to model.
TEMPLATE_OVERHEAD_TOKENS = 64


class CapExceeded(RuntimeError):
    """Raised when a row cannot be built inside its task's budget."""


@dataclass
class TokenEstimate:
    """A row's estimated cost, itemised so a rejection says what to cut."""

    task: str
    doc_type: str | None
    prompt_tokens: int = 0
    ocr_tokens: int = 0
    visual_tokens: int = 0
    output_tokens: int = 0
    pages: int = 0

    @property
    def input_tokens(self) -> int:
        return (
            self.prompt_tokens + self.ocr_tokens + self.visual_tokens
            + TEMPLATE_OVERHEAD_TOKENS
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "doc_type": self.doc_type,
            "pages": self.pages,
            "prompt_tokens": self.prompt_tokens,
            "ocr_tokens": self.ocr_tokens,
            "visual_tokens": self.visual_tokens,
            "output_tokens": self.output_tokens,
            "input_tokens": self.input_tokens,
            "total_tokens": self.total_tokens,
        }

    def explain(self, cap: int) -> str:
        """Why this row does not fit, in the terms an operator can act on."""
        return (
            f"{self.task} needs ~{self.total_tokens:,} tokens against a {cap:,} cap "
            f"({self.pages} page(s): {self.visual_tokens:,} visual + {self.ocr_tokens:,} OCR "
            f"+ {self.prompt_tokens:,} prompt + {self.output_tokens:,} output). "
            "Route fewer pages, lower max_pixels for this task, or raise the cap — "
            "truncating is not an option, because a clipped assistant span trains the "
            "model to stop early."
        )


@dataclass
class CapReport:
    """What fit and what did not, across a corpus build."""

    accepted: int = 0
    rejected: list[dict[str, Any]] = field(default_factory=list)
    max_seen: dict[str, int] = field(default_factory=dict)
    #: The unit that is actually lost. A document with one row over budget is
    #: set aside whole, so counting rows alone reads one bad window in four as
    #: 25% rejected when 100% of the document is gone.
    documents_accepted: int = 0
    documents_rejected: int = 0
    #: Windows whose answer overran its reservation and were read as two halves
    #: instead, as serving retries them (``build_jsonl``, review item 38), and
    #: the rows those halves made. A document is set aside for an answer only
    #: when a single page's still overruns.
    windows_halved: int = 0
    rows_from_halves: int = 0

    @property
    def rejection_rate(self) -> float:
        """The share of DOCUMENTS set aside, once documents are being counted.

        The corpus build counts documents — the unit that is actually kept or
        lost — and records one rejected entry per document. Dividing those by
        accepted ROWS would mix the two units. Row-level use (``check_row`` on
        its own) keeps the row rate.
        """
        documents = self.documents_accepted + self.documents_rejected
        if documents:
            return self.documents_rejected / documents
        total = self.accepted + len(self.rejected)
        return len(self.rejected) / total if total else 0.0

    def record(self, task: str, estimate: TokenEstimate) -> None:
        self.accepted += 1
        self.max_seen[task] = max(self.max_seen.get(task, 0), estimate.total_tokens)

    def reject(
        self, source_id: str, task: str, estimate: TokenEstimate, cap: int,
        reason: str | None = None,
    ) -> None:
        self.rejected.append({
            "source_id": source_id, "task": task, "cap": cap,
            "reason": reason or estimate.explain(cap), **estimate.as_dict(),
        })

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows_accepted": self.accepted,
            "rejections_recorded": len(self.rejected),
            "documents_accepted": self.documents_accepted,
            "documents_rejected": self.documents_rejected,
            "windows_halved": self.windows_halved,
            "rows_from_halves": self.rows_from_halves,
            "rejection_rate": round(self.rejection_rate, 4),
            "max_tokens_seen_by_task": dict(sorted(self.max_seen.items())),
            "rejections": self.rejected[:20],
        }

    def warning(self) -> str | None:
        if not self.rejected:
            return None
        by_task: dict[str, int] = {}
        for row in self.rejected:
            by_task[row["task"]] = by_task.get(row["task"], 0) + 1
        documents = self.documents_rejected or len(self.rejected)
        return (
            f"{documents} document(s) set aside for exceeding a task budget "
            f"(first failing task: {by_task}). They contribute NOTHING to training — an "
            "over-budget row is dropped, not shortened, and so is the rest of its document. "
            "If the rate is material, the caps in configs/shared/sequence.yaml or the page "
            "routing are wrong, not the documents."
        )


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


@lru_cache(maxsize=1)
def _answer_tokenizer() -> Any:
    """The base model's own tokenizer where its files are on this machine (the
    pod's /workspace/models), else ``None``."""
    from common.config import base_model_dir

    local = base_model_dir()
    if local is None or not (local / "tokenizer.json").is_file():
        return None
    try:
        # As training.length_check loads it, so the two count alike.
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 - an unreadable tokenizer falls back to the padded estimate
        log.warning("could not load the tokenizer in %s (%s); answers are estimated", local, exc)
        return None


def target_tokens(text: str | None) -> int:
    """Tokens of an answer (the assistant turn).

    Counted by the base model's tokenizer when it is on this machine, so the
    corpus sets aside exactly the rows the pre-launch length check
    (training.length_check) would refuse; estimated and padded by
    :data:`TARGET_ESTIMATE_FACTOR` otherwise. The plain estimate passed answers
    of 12,954-13,492 tokens against a 12,288 reservation, and training refused
    them at launch, after the whole corpus had been built and staged.
    """
    if not text:
        return 0
    tokenizer = _answer_tokenizer()
    if tokenizer is not None:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return math.ceil(estimate_text_tokens(text) * TARGET_ESTIMATE_FACTOR)


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


def estimate_row(
    *,
    task: str,
    system_prompt: str,
    ocr_pages: list[str] | None,
    page_count: int,
    target_json: str | dict[str, Any],
    doc_type: str | None = None,
) -> TokenEstimate:
    """Estimate one candidate row's cost, itemised."""
    target = (
        target_json if isinstance(target_json, str)
        else json.dumps(target_json, ensure_ascii=False)
    )
    return TokenEstimate(
        task=task,
        doc_type=doc_type,
        pages=page_count,
        prompt_tokens=estimate_text_tokens(system_prompt),
        ocr_tokens=sum(estimate_text_tokens(p) for p in (ocr_pages or [])),
        visual_tokens=estimate_visual_tokens(page_count, task),
        output_tokens=target_tokens(target),
    )


def evaluate(estimate: TokenEstimate) -> tuple[bool, int, str | None]:
    """``(fits, cap, why not)`` for one row, without recording anything.

    The reason names the budget that actually failed. A row can fit the total
    and still overrun its reserved OUTPUT — the case that clips a target — and
    telling an operator to "raise the cap" for that would be the wrong advice.
    """
    budget = sequence_for_task(estimate.task, estimate.doc_type)
    cap = int(budget["max_seq_len"])
    output_cap = int(budget["max_output_tokens"])
    if estimate.output_tokens > output_cap:
        return False, cap, (
            f"{estimate.task} target is ~{estimate.output_tokens:,} tokens against a reserved "
            f"{output_cap:,}. The assistant span would be clipped, which trains the model to "
            "stop early — on a Loss Run, to omit claim rows. Shrink the window (§7b) rather "
            "than raising the reservation."
        )
    if estimate.total_tokens > cap:
        return False, cap, estimate.explain(cap)
    return True, cap, None


def answer_overflows(task: str, answer: str | None, doc_type: str | None = None) -> bool:
    """Whether an answer runs past its task's reserved output - the case
    :func:`evaluate` refuses first, and the one serving meets as an answer cut
    off at the token limit and retries in halves (``serving.pipeline._halves``).
    Counted as :func:`estimate_row` counts it (:func:`target_tokens`)."""
    return target_tokens(answer) > int(sequence_for_task(task, doc_type)["max_output_tokens"])


def check_row(
    estimate: TokenEstimate,
    *,
    source_id: str,
    report: CapReport | None = None,
    raise_on_exceed: bool = False,
) -> bool:
    """Whether a row fits its task budget. Records the outcome on ``report``.

    Two separate limits, and the second is the one that matters most:

    * the **total** must fit the task's ``max_seq_len``;
    * the **output** must fit the task's reserved ``max_output_tokens``, because
      a target longer than the reservation is the case that produces a clipped
      assistant span — and a clipped span is a wrong training target, not a
      short one.
    """
    fits, cap, detail = evaluate(estimate)
    if fits:
        if report is not None:
            report.record(estimate.task, estimate)
        return True

    if report is not None:
        report.reject(source_id, estimate.task, estimate, cap, reason=detail)
    log.warning("%s: %s", source_id, detail)
    if raise_on_exceed:
        raise CapExceeded(f"{source_id}: {detail}")
    return False
