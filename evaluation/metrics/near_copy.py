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


def page_image_digests(client: Any, keys: Sequence[str], *, local: Mapping[str, str] | None = None,
                       workers: int = 16) -> list[str | None]:
    """The MD5 of each page image in ``keys``, in order: of the copy on disk when ``local`` (``{key: path}``) has
    one, else the MD5 stored with the image (``BlobClient.md5``), else of its bytes. None for a page that could not
    be read. The one page-image hash: near-copy pairing and the data health report must agree on "the same image".
    """
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    on_disk = local or {}

    def digest(key: str) -> str | None:
        try:
            path = on_disk.get(key)
            if path and Path(path).is_file():
                return hashlib.md5(Path(path).read_bytes()).hexdigest()
            if client is None:
                return None
            recorded = client.md5(key)
            return recorded.hex() if recorded else hashlib.md5(client.read_bytes(key)).hexdigest()
        except Exception as exc:  # noqa: BLE001 - one unreadable page is reported by the caller, not fatal
            log.debug("page image %s could not be read (%s: %s)", key, type(exc).__name__, exc)
            return None

    if not keys:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="page-image") as pool:
        return list(pool.map(digest, keys))


def _enough_text(document: Any) -> list[bool] | None:
    """Per page image, whether its page has text enough to tell documents apart (as :func:`_page_keys` filters);
    None when the pages and the texts do not line up one to one."""
    texts = getattr(document, "page_texts", None) or {}
    if len(texts) != len(document.image_keys):
        return None
    return [len(re.sub(r"\s+", "", str(texts[page] or ""))) >= _MIN_PAGE_CHARACTERS for page in sorted(texts)]


def page_image_keys(client: Any, documents: Sequence[Any], workers: int = 16,
                    local: Mapping[str, str] | None = None) -> dict[str, set[str]] | None:
    """Each document's distinct page images, by :func:`page_image_digests` (``local``: the copies the gate already
    downloaded, so no storage call is made for them). A page with too little text to tell documents apart (a
    blank page, a page number) is left out, as in the text comparison. None when a document has no image keys or
    a page image could not be read: the comparison then falls back to page text.

    Pages, not their OCR text, decide a near copy: 0065 and 0066 of the frozen set share 71 of 73 byte-identical page
    images, yet OCR read 18 of those 71 into different text, and by text the pair (54 of 73) was never measured
    (2026-11-07).
    """
    if not documents or any(not getattr(d, "image_keys", None) for d in documents):
        return None
    if client is None and not local:
        return None
    keys = [(document.source_id, key) for document in documents for key in document.image_keys]
    digests = page_image_digests(client, [key for _source, key in keys], local=local, workers=workers)
    unread = [key for (_source, key), image in zip(keys, digests, strict=True) if image is None]
    if unread:
        log.warning("%d page image(s) could not be read (first: %s); near copies are found by page text",
                    len(unread), unread[0])
        return None
    keep = {document.source_id: _enough_text(document) for document in documents}
    out: dict[str, set[str]] = {document.source_id: set() for document in documents}
    position: dict[str, int] = defaultdict(int)
    for (source_id, _key), image in zip(keys, digests, strict=True):
        index = position[source_id]
        position[source_id] += 1
        wanted = keep[source_id]
        if wanted is None or wanted[index]:
            out[source_id].add(image)
    return out


def near_copy_pairs(documents: Sequence[Any], share: float = NEAR_COPY_SHARE,
                    page_keys: Mapping[str, set[str]] | None = None) -> list[NearCopyPair]:
    """The pairs of ``documents`` (each with ``source_id`` and ``page_texts``) sharing ``share`` of their pages:
    by ``page_keys`` when given (:func:`page_image_keys`), else by page text."""
    pages = ({document.source_id: set(page_keys.get(document.source_id, ())) for document in documents}
             if page_keys is not None else
             {document.source_id: _page_keys(document.page_texts) for document in documents})
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


def near_copy_agreement(documents: Sequence[Any], triples: Sequence[tuple[Any, Any, Mapping[str, Any]]],
                        page_keys: Mapping[str, set[str]] | None = None) -> dict[str, Any]:
    """Each near-identical pair's agreement by reading mode, and over all of them (weighted by fields compared)."""
    readings: dict[tuple[Any, Any], tuple[Any, Any, Mapping[str, Any]]] = {}
    for gold, answer, metadata in triples:
        readings[(metadata.get("source_id"), metadata.get("modality_mode"))] = (gold, answer, metadata)
    pairs = []
    weighted = total = 0.0
    for pair in near_copy_pairs(documents, page_keys=page_keys):
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
            "share_needed_to_pair": NEAR_COPY_SHARE,
            "paired_by": "page image" if page_keys is not None else "page text"}


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


def consistency_section(documents: Sequence[Any], triples: Sequence[Any], body: Mapping[str, Any],
                        client: Any = None, local: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What a report carries: both checks. Reported, never gated, so a failure here never costs a gate its hours: it
    is logged, and the report says the check did not run. With ``client`` or ``local`` (the page images already on
    disk, ``{key: path}``), near copies are found by page image."""
    scores = (body.get("per_document") or {}).get("field_normalized_match") or {}
    try:
        page_keys = page_image_keys(client, documents, local=local)
        return {"near_copy_agreement": near_copy_agreement(documents, triples, page_keys),
                "per_policy_field_normalized_match": per_policy_scores(scores, documents)}
    except Exception as exc:  # noqa: BLE001 - reported only; the gate's own result stands
        log.warning("the near-copy consistency check did not run (%s: %s)", type(exc).__name__, exc)
        return {"near_copy_agreement": {"error": f"{type(exc).__name__}: {exc}"},
                "per_policy_field_normalized_match": {}}
