"""Does a nearly identical PDF get the same answer, and what is the score with each policy counted once.

Near-identical PDFs stay in the exam on purpose (user, 2026-11-04): A and B print the same policy but for a page or two,
so a reader that writes one thing for A and another for B has shown something run-to-run agreement cannot show. That
reads one PDF twice; this reads two PDFs that barely differ, as a renewal or a re-scan does in production.

* :func:`near_copy_agreement`: for each near-identical pair in the exam, in each reading mode, the share of fields the
  two answers agree on, by field match's own scorer. A value only one gold holds is left out: the PDFs really differ
  there, so differing answers are right.
* :func:`per_policy_scores`: a policy on two PDFs counts twice in an average. This shows the mean with each PDF counted
  and with each policy number counted once, for reading beside the score. Neither is gated.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

log = logging.getLogger(__name__)

#: Two documents are near-identical when they share this share of the smaller one's distinct pages.
NEAR_COPY_SHARE = 0.9
#: A page with less text than this says nothing about which document it is (a blank page, a page number).
_MIN_PAGE_CHARACTERS = 40


@dataclass(frozen=True)
class NearCopyPair:
    a: str
    b: str
    share: float


def _page_keys(page_texts: Mapping[Any, str]) -> set[str]:
    """Each page's text, without spacing or case, as a hash; pages with too little text to tell are left out."""
    keys = set()
    for text in page_texts.values():
        compact = re.sub(r"\s+", "", str(text or "")).casefold()
        if len(compact) >= _MIN_PAGE_CHARACTERS:
            keys.add(hashlib.sha1(compact.encode("utf-8")).hexdigest())
    return keys


def near_copy_pairs(documents: Sequence[Any], share: float = NEAR_COPY_SHARE) -> list[NearCopyPair]:
    """The pairs of ``documents`` (each with ``source_id`` and ``page_texts``) sharing ``share`` of their page text."""
    pages = {document.source_id: _page_keys(document.page_texts) for document in documents}
    out = []
    for a, b in combinations(sorted(pages), 2):
        smaller = min(len(pages[a]), len(pages[b]))
        if smaller and len(pages[a] & pages[b]) / smaller >= share:
            out.append(NearCopyPair(a, b, round(len(pages[a] & pages[b]) / smaller, 4)))
    return out


def _gold_only_values(gold_a: Any, gold_b: Any) -> set[str]:
    """Values one gold holds and the other does not: where the two PDFs really differ."""
    from collections import Counter

    from common.canonical import values_view
    from evaluation.metrics.field_accuracy import _key_text, flatten_scalars

    def values(gold: Any) -> Counter:
        found: Counter = Counter()
        for value in flatten_scalars(values_view(gold or {})).values():
            for item in value if isinstance(value, list) else [value]:
                if _key_text(item):
                    found[_key_text(item)] += 1
        return found

    first, second = values(gold_a), values(gold_b)
    return set((first - second) | (second - first))


def _agreement(first: Any, second: Any, differ: set[str], metadata: Mapping[str, Any]) -> tuple[float, float] | None:
    """``(share of fields equal, fields compared)`` as ``run_eval.run_to_run_agreement`` reads two answers, leaving out
    a field where either answer holds a value only one gold holds."""
    from common.lob import merge_line
    from evaluation.metrics.common_model import core_for_scoring
    from evaluation.metrics.extraction_faults import _is_empty
    from evaluation.metrics.field_accuracy import _key_text, score_fields
    from evaluation.run_eval import _is_common_model_document, _scored_view

    doc_type = metadata.get("doc_type", "unknown")
    common_model = _is_common_model_document(doc_type, dict(metadata))
    line = merge_line(metadata.get("lob")) if common_model else None
    selectors = (doc_type, metadata.get("acord_form"), metadata.get("lob"))
    one, two = (_scored_view(answer or {}, selectors, line, common_model) for answer in (first, second))

    def held_by_one_gold(value: Any) -> bool:
        return any(_key_text(item) in differ for item in (value if isinstance(value, list) else [value]))

    shares, counts = [], []
    for reference, other in ((one, two), (two, one)):
        accuracy = (score_fields(*core_for_scoring(reference, other, line), skip_lists=False)
                    if common_model else score_fields(reference, other))
        compared = [r for r in accuracy.results
                    if not (_is_empty(r.expected) and _is_empty(r.got))
                    and not held_by_one_gold(r.expected) and not held_by_one_gold(r.got)]
        if compared:
            shares.append(sum(r.correct for r in compared) / len(compared))
            counts.append(len(compared))
    return (sum(shares) / len(shares), sum(counts) / len(counts)) if counts else None


def near_copy_agreement(documents: Sequence[Any], triples: Sequence[tuple[Any, Any, Mapping[str, Any]]]) -> dict[str, Any]:
    """Each near-identical pair's agreement by reading mode, and over all of them (weighted by fields compared)."""
    readings: dict[tuple[Any, Any], tuple[Any, Any, Mapping[str, Any]]] = {}
    for gold, answer, metadata in triples:
        readings[(metadata.get("source_id"), metadata.get("modality_mode"))] = (gold, answer, metadata)
    pairs = []
    weighted = total = 0.0
    for pair in near_copy_pairs(documents):
        by_mode: dict[str, dict[str, float]] = {}
        modes = {mode for source, mode in readings if source == pair.a} & {mode for source, mode in readings if source == pair.b}
        for mode in sorted(modes, key=str):
            gold_a, answer_a, metadata = readings[(pair.a, mode)]
            gold_b, answer_b, _metadata = readings[(pair.b, mode)]
            measured = _agreement(answer_a, answer_b, _gold_only_values(gold_a, gold_b), metadata)
            if measured is None:
                continue
            share, compared = measured
            by_mode[str(mode)] = {"agreement": round(share, 4), "fields_compared": round(compared, 1)}
            weighted += share * compared
            total += compared
        pairs.append({"a": pair.a, "b": pair.b, "pages_shared": pair.share, "by_mode": by_mode})
    return {"pairs": pairs, "agreement": round(weighted / total, 4) if total else None,
            "share_needed_to_pair": NEAR_COPY_SHARE}


def per_policy_scores(per_document: Mapping[str, float], documents: Sequence[Any]) -> dict[str, Any]:
    """The mean of ``per_document`` with each PDF counted, and with each policy number counted once."""
    from data_pipeline.ingestion.prepare_bundles import gold_policy_number

    def compact(text: Any) -> str:
        return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())

    policy_of = {d.source_id: compact(gold_policy_number(d.golden or {})) or d.source_id for d in documents}
    by_policy: dict[str, list[float]] = defaultdict(list)
    for source, score in per_document.items():
        by_policy[policy_of.get(source, source)].append(float(score))
    if not by_policy:
        return {}
    return {
        "documents": sum(len(scores) for scores in by_policy.values()),
        "policies": len(by_policy),
        "mean_per_document": round(sum(per_document.values()) / len(per_document), 4),
        "mean_per_policy": round(sum(sum(s) / len(s) for s in by_policy.values()) / len(by_policy), 4),
    }


def consistency_section(documents: Sequence[Any], triples: Sequence[Any], body: Mapping[str, Any]) -> dict[str, Any]:
    """What a report carries: both checks. Reported, never gated, so a failure here never costs a gate its hours: it
    is logged, and the report says the check did not run."""
    scores = (body.get("per_document") or {}).get("field_normalized_match") or {}
    try:
        return {"near_copy_agreement": near_copy_agreement(documents, triples),
                "per_policy_field_normalized_match": per_policy_scores(scores, documents)}
    except Exception as exc:  # noqa: BLE001 - reported only; the gate's own result stands
        log.warning("the near-copy consistency check did not run (%s: %s)", type(exc).__name__, exc)
        return {"near_copy_agreement": {"error": f"{type(exc).__name__}: {exc}"},
                "per_policy_field_normalized_match": {}}
