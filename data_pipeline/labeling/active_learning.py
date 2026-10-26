"""Confidence-routed review queue (IMPL-04 §6, arch §7 step 6, §13 stage 11).

Once a model version exists, calibrated confidence decides how much human
attention a document gets: high-confidence extractions get a light spot-check,
low-confidence *fields* get full manual review. That is what makes labeling cost
fall across corpus versions instead of staying flat.

Two design points carry most of the value:

**Per-field, not per-document.** Flagging a whole 40-row Loss Run because one
date was uncertain wastes the reviewer's time on 39 correct rows. The queue
carries the specific fields.

**Row-completeness overrides confidence entirely.** A Loss Run whose row count
disagrees with the document goes to full review even when every extracted value
scored 0.99 — because the missing rows have no confidence score at all. Their
absence is invisible to a logprob, which is exactly why IMPL-09 computes a
separate signal for it.

**Disabled while the review requirement is "full".** Before 25 labels per type
exist, confidence routing is routing on a model with no evidence it is
trustworthy (arch §4c). The refusal is a guard, not a warning.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from artifact_registry.blob_client import BlobClient
from calibration.list_completeness import is_row_completeness_flag
from common.constants import ACTIVE_DOC_TYPES, DEFAULT_REVIEW_CONFIDENCE_THRESHOLD

log = logging.getLogger(__name__)

#: Above this, a document gets a spot-check rather than a field-by-field pass.
SPOT_CHECK_THRESHOLD = 0.90


class ActiveLearningError(RuntimeError):
    """Raised when confidence routing may not be used."""


@dataclass
class QueueItem:
    """One document's place in the review queue, and why it is there."""

    source_id: str
    doc_type: str
    overall_confidence: float
    low_confidence_fields: list[tuple[str, float]] = field(default_factory=list)
    completeness_flags: list[str] = field(default_factory=list)
    review_flags: list[str] = field(default_factory=list)

    @property
    def forced_full_review(self) -> bool:
        """Row-completeness beats confidence.

        A list whose row count disagrees with the document is missing rows, and a
        missing row has no confidence score to be low — its absence is the whole
        problem, and no per-value number can express it.
        """
        return bool(self.completeness_flags)

    @property
    def routing(self) -> str:
        if self.forced_full_review:
            return "full_review"
        if self.low_confidence_fields:
            return "field_review"
        if self.overall_confidence >= SPOT_CHECK_THRESHOLD:
            return "spot_check"
        return "field_review"

    @property
    def sort_key(self) -> tuple[int, float]:
        """Completeness-flagged documents first, then ascending confidence."""
        return (0 if self.forced_full_review else 1, self.overall_confidence)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "doc_type": self.doc_type,
            "routing": self.routing,
            "overall_confidence": round(self.overall_confidence, 4),
            "low_confidence_fields": [
                {"field": path, "confidence": round(score, 4)}
                for path, score in self.low_confidence_fields
            ],
            "completeness_flags": self.completeness_flags,
            "review_flags": self.review_flags,
            "forced_full_review": self.forced_full_review,
        }


@dataclass
class ReviewQueue:
    """The prioritized queue, plus what it would save."""

    items: list[QueueItem] = field(default_factory=list)
    model_version: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def by_routing(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.routing] = counts.get(item.routing, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def spot_check_share(self) -> float:
        """The share of documents needing only a spot-check.

        This is the number that shows labeling cost falling: it should rise with
        each corpus version, and if it does not, confidence is not improving.
        """
        if not self.items:
            return 0.0
        return self.by_routing.get("spot_check", 0) / len(self.items)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "generated_at": self.generated_at,
            "documents": len(self.items),
            "by_routing": self.by_routing,
            "spot_check_share": round(self.spot_check_share, 4),
            "queue": [item.as_dict() for item in self.items],
        }


def assert_routing_permitted(
    client: BlobClient, doc_types: Sequence[str], tenant_id: str | None = None
) -> None:
    """Refuse while any active type is still under the day-zero full-review rule.

    Per type rather than overall: routing ACORD by confidence while ACORD has
    four labels is exactly the mistake, even if Policy has three hundred.
    """
    from data_pipeline.labeling.export_golden_labels import count_golden_labels, review_requirement

    blocked = [
        f"{doc_type} ({count_golden_labels(client, doc_type, tenant_id)} labels)"
        for doc_type in doc_types
        if review_requirement(client, doc_type, tenant_id) == "full"
    ]
    if blocked:
        raise ActiveLearningError(
            "confidence routing is disabled while any document type is still under the day-zero "
            f"full-review rule: {', '.join(blocked)}. Routing on a model with no evidence it is "
            "trustworthy would send its own mistakes past a reviewer (arch §4c). Every draft gets "
            "full review until each type reaches the threshold."
        )


def build_queue_item(
    result: Any,
    *,
    review_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
) -> QueueItem:
    """Turn one ``ExtractionResult`` into a queue entry."""
    low: list[tuple[str, float]] = []
    for path, value in (result.fields or {}).items():
        if not isinstance(value, dict):
            continue
        confidence = value.get("confidence")
        # A field with no confidence is not a confident field. It means the span
        # could not be mapped, which is a reason to look at it, not to skip it.
        if confidence is None or confidence < review_threshold:
            low.append((path, float(confidence or 0.0)))

    # `is_row_completeness_flag`, not a hand-written prefix list. The prefixes
    # this used to match ("list:", "rows:", "completeness:") are emitted by
    # nothing, so the documented "row completeness outranks every confidence
    # score" override never once fired: a Loss Run where MinerU counted 8 table
    # rows and the model returned 6, every value at 0.99, was routed to a spot
    # check and sorted to the BACK of the queue.
    completeness = [
        flag for flag in (result.review_flags or []) if is_row_completeness_flag(flag)
    ]

    return QueueItem(
        source_id=result.source_id,
        doc_type=result.doc_type,
        overall_confidence=float(result.overall_confidence),
        low_confidence_fields=sorted(low, key=lambda row: row[1]),
        completeness_flags=sorted(completeness),
        review_flags=sorted(result.review_flags or []),
    )


def build_queue(
    results: Iterable[Any],
    *,
    model_version: str = "",
    review_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
) -> ReviewQueue:
    """Order extraction results into a review queue.

    Ascending confidence, with completeness-flagged documents ahead of
    everything: the reviewer's first hour goes where it changes the most.
    """
    queue = ReviewQueue(model_version=model_version)
    queue.items = sorted(
        (build_queue_item(r, review_threshold=review_threshold) for r in results),
        key=lambda item: item.sort_key,
    )
    log.info(
        "review queue: %d document(s) %s; %.0f%% need only a spot-check",
        len(queue.items), queue.by_routing, queue.spot_check_share * 100,
    )
    return queue


def write_queue(queue: ReviewQueue, client: BlobClient, tenant_id: str | None = None) -> str:
    from artifact_registry import paths

    key = (
        f"golden-labels/{paths._tenant(tenant_id)}/_review_queue/"
        f"{queue.model_version or 'unknown'}.json"
    )
    client.write_json(key, queue.as_dict())
    return key


def main(argv: Iterable[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description="Build a confidence-routed review queue")
    parser.add_argument("--model", required=True)
    parser.add_argument("--doc-types", nargs="+", default=list(ACTIVE_DOC_TYPES),
                        choices=list(ACTIVE_DOC_TYPES))
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_REVIEW_CONFIDENCE_THRESHOLD)
    parser.add_argument("--limit", type=int, default=None, help="extract at most this many documents per type")
    args = parser.parse_args(list(argv) if argv is not None else None)
    # On the pod, run detached in tmux: a closed laptop must not stop this job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed('data_pipeline.labeling.active_learning', argv):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client = BlobClient()
    assert_routing_permitted(client, args.doc_types, args.tenant)

    # The unlabelled documents, extracted by the release through the serving
    # pipeline with its calibrators (orchestration.label_noise.backlog_results).
    from inference_core.model_runner import load_model, release_model
    from orchestration.label_noise import backlog_results

    model = load_model(args.model, client)
    try:
        results = backlog_results(client, model, version=args.model, doc_types=args.doc_types,
                                  tenant_id=args.tenant, limit=args.limit)
    finally:
        release_model(model)
    queue = build_queue(results, model_version=args.model, review_threshold=args.threshold)
    key = write_queue(queue, client, args.tenant)
    print(f"review queue: {len(queue.items)} document(s) {queue.by_routing} -> {key}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
