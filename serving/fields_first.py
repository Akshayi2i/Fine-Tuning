"""Fields first, full text second: one H100 for the model and MinerU (decision 2026-10-24).

The production gate is a 200-page scan extracted in 2 minutes, on ONE GPU that
serves both the model and MinerU. MinerU alone reads such a scan in 4-5
minutes (1.0-1.5 s a page), so it cannot stand before the model. And no value
may be served that the PDF does not print (decision 2026-10-24,
``serving.printed_values``), which needs page text. A document is therefore
read in two phases:

1. **Fields** (:func:`read_fields`, the gate): the pages are rendered and the
   model reads their images - ``image_only``, a mode it trains on - with the
   GPU to itself. Then MinerU reads **only the pages the fields cite** (their
   ``page_ref``: on a 200-page policy a few dozen, ~15-40 s) and the fields are
   held to that text before anything is served (:func:`verify_fields`): a value
   found there is served; one not found there is **held back**, flagged
   ``<path>:pending_full_text`` - not yet "printed nowhere", since the model
   may have cited the wrong page - and every declarations field left empty
   while those pages print its label and a value is flagged ``likely_missed``.
   A value whose cited page MinerU read empty is served flagged
   ``unverified_unread_page``, as the standard path serves it
   (``serving.printed_values``); one printed only on a read page it does not
   cite is served flagged ``printed_off_cited_page`` and sent to review, as is
   one under a label those pages print another value after (``role_mismatch``). So the fields served at the gate hold no value
   the PDF was not seen to print, nor one not said to be unverified. When no
   page the fields cite has any text, nothing could be checked: nothing is
   served before the full text. A row held back keeps the ids of the rows left
   as they are - a served row's id and links are the same at the gate and at
   the end - and a link to it is held back with it (``pending_full_text``).
2. **Full text** (:func:`add_full_text`): MinerU reads the remaining pages, and
   the fields as the model wrote them are held to the whole document's text -
   a value printed on a page it did not cite comes back; one printed nowhere is
   taken out - and the full-text tier joins them
   (``serving.pipeline.attach_full_text``). The final result is what the
   standard path serves for the same answer.

:class:`GpuTurns` keeps the order on the shared GPU: MinerU reads a run of
pages only while no document's fields are being read, so a document arriving
while another's text is read waits at most one run (``ServedOcr.run_pages``,
20 pages), not a whole document's OCR. So that full texts are not held back
for ever while fields keep arriving, a run waits at most ``text_max_wait_s``
and then goes regardless. A document's own verification is part of its
fields' turn.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: The longest a run of MinerU waits for its turn before it goes anyway.
TEXT_MAX_WAIT_S = 600.0

#: A value the pages its fields cite do not print, held back from the served
#: fields until the whole document's text has been searched.
PENDING_FLAG = "pending_full_text"

#: Fields served without being held to any page text (``verify=False``: timing
#: tests only - production serves verified fields).
UNVERIFIED_FLAG = "fields:unverified"

#: Pages read to verify fields that cite none: the declarations' usual place.
FALLBACK_PAGES = 5


class GpuTurns:
    """Whose turn the one GPU is: any document's fields before anyone's full text."""

    def __init__(self, text_max_wait_s: float | None = TEXT_MAX_WAIT_S) -> None:
        self._reading_fields = 0
        self._changed = threading.Condition()
        self.text_max_wait_s = text_max_wait_s
        #: Runs that went before their turn, having waited the longest allowed.
        self.overdue_runs = 0

    @contextmanager
    def fields(self) -> Iterator[None]:
        """Held while a document's fields are read; MinerU starts no new run meanwhile."""
        with self._changed:
            self._reading_fields += 1
        try:
            yield
        finally:
            with self._changed:
                self._reading_fields -= 1
                self._changed.notify_all()

    def wait_for_text_turn(self) -> None:
        """Return when no document's fields are being read (``ocr_via_api``'s
        gate) - or, past ``text_max_wait_s``, regardless: fields arriving one
        after another would otherwise hold every full text back for ever."""
        with self._changed:
            if self._changed.wait_for(lambda: self._reading_fields == 0, timeout=self.text_max_wait_s):
                return
            self.overdue_runs += 1
        log.warning("a run of MinerU waited %.0f s for the GPU while fields kept arriving; it goes now, "
                    "beside a document's fields", self.text_max_wait_s)


def cited_pages(extraction: Any, page_count: int) -> list[int]:
    """Every page a value of ``extraction`` cites (its ``page_ref``), within the document."""
    found: set[int] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            refs = node.get("page_ref")
            if isinstance(refs, list):
                found.update(page for page in refs if isinstance(page, int) and 1 <= page <= page_count)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(extraction)
    return sorted(found)


def _pending(flag: str) -> str:
    """A not-printed flag from a check on some pages only - on a value, or on a
    link to a row of such values - as what it means then."""
    from serving.conventions import NOT_PRINTED_FLAG
    from serving.printed_values import LINK_REMOVED_FLAG, REMOVED_FLAG

    path, _, kind = flag.rpartition(":")
    return (f"{path}:{PENDING_FLAG}" if path and kind in (REMOVED_FLAG, NOT_PRINTED_FLAG, LINK_REMOVED_FLAG)
            else flag)


#: ``calibrators``/``thresholds`` not given: those the result was scored with
#: (``ExtractionResult.calibrated_with``).
FROM_RESULT: Any = object()


def _feature_vectors(kept: Any) -> dict[str, Any]:
    """The result's feature vectors as ``FieldFeatures``: a result restored
    from a checkpoint (``dataclasses.asdict`` then JSON) carries them as dicts."""
    import dataclasses

    from calibration.features import FieldFeatures

    names = {f.name for f in dataclasses.fields(FieldFeatures) if f.init}
    out: dict[str, Any] = {}
    for path, features in (kept or {}).items():
        if isinstance(features, FieldFeatures):
            out[path] = features
        elif isinstance(features, dict):
            try:
                out[path] = FieldFeatures(**{k: v for k, v in features.items() if k in names})
            except TypeError:
                continue                            # not a feature vector: that field is not scored again
    return out


def rescore_on_text(result: Any, page_texts: Mapping[int, str], *, calibrators: Any = FROM_RESULT,
                    thresholds: Any = FROM_RESULT) -> None:
    """``result``'s confidence and review flags computed again on the page text
    that arrived after its fields were read from the images
    (``calibration.post_text.recalibrate_after_text``), in place, with the
    release calibrators and thresholds it was scored with - the same that
    ``read_fields``' ``extract_options`` gave extract - unless others are given.

    Under the fast profile the calibrators are fitted on image-only readings
    WITH their document's text (``calibration.population``): scored without it,
    a field's text agreement is missing from the vector the calibrator reads.
    A no-op without calibrators, feature vectors or text. The values a review
    flag sends to review (``serving.value_flags``) stay flagged: the second
    scoring writes each envelope's ``flagged`` from confidence alone."""
    from calibration.post_text import recalibrate_after_text
    from serving.pipeline import CalibratedWith
    from serving.value_flags import route_served

    held = getattr(result, "calibrated_with", None)
    held = held if isinstance(held, CalibratedWith) else None
    if calibrators is FROM_RESULT:
        calibrators = held.calibrators if held is not None else None
    if thresholds is FROM_RESULT:
        thresholds = held.thresholds if held is not None else None
    features = _feature_vectors(getattr(result, "confidence_features", None))
    # Given even when empty: the result's own (from a checkpoint, dicts) are never read raw.
    recalibrate_after_text(result, page_texts, calibrators=calibrators, thresholds=thresholds,
                           features=features)
    route_served(result.extraction, result.review_flags)


def verify_page_budget() -> int:
    """``fields_first.verify_page_budget`` from the serving config (0: none)."""
    from common.config import SERVING_CONFIG, ConfigError, serving_config

    budget = (serving_config().get("fields_first") or {}).get("verify_page_budget", 0)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 0:
        raise ConfigError(f"fields_first.verify_page_budget is {budget!r} in {SERVING_CONFIG.name}; "
                          "expected a whole number of pages, 0 for no budget")
    return budget


def pages_to_verify(extraction: Any, page_count: int, budget: int = 0) -> list[int]:
    """The pages MinerU reads to check the fields before they are served.

    Every value cites the pages that print it (its ``page_ref``), and a value is
    checked on any one of them: a printed value is printed on whichever page is
    searched. So not every cited page need be read - only enough that each value
    has one - and the labels' habit of citing every page that prints a value (a
    carrier name in every header) made the cited pages most of a long policy.
    With ``budget`` pages, the set is chosen greedily: each time the page that is
    cited by the most values not yet covered (the lowest page on a tie), until
    every value is covered or the budget is spent. A value left uncovered is not
    served unchecked: :func:`verify_fields` holds back a value the pages read do
    not print, and the full text settles it. ``budget`` 0 is every cited page
    (:func:`cited_pages`).
    """
    if budget <= 0:
        return cited_pages(extraction, page_count)
    weight: Counter = Counter()          # a citation set, as often as values cite it

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            refs = node.get("page_ref")
            if isinstance(refs, list):
                pages = frozenset(p for p in refs if isinstance(p, int) and not isinstance(p, bool)
                                  and 1 <= p <= page_count)
                if pages:
                    weight[pages] += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(extraction)
    uncovered = set(weight)
    chosen: list[int] = []
    while uncovered and len(chosen) < budget:
        votes: Counter = Counter()
        for pages in uncovered:
            for page in pages:
                votes[page] += weight[pages]
        page = min(votes, key=lambda p: (-votes[p], p))
        chosen.append(page)
        uncovered = {pages for pages in uncovered if page not in pages}
    return sorted(chosen)


def verify_fields(pdf: bytes, result: Any, *, name: str, lob: str, reader: Any, page_count: int) -> dict[str, Any]:
    """The fields as they may be served now: held to the text of the pages they cite.

    ``result`` is left as the model's answer, for the full text to hold to the
    whole document later; the served copy is returned. Returns ``{"served",
    "verified", "texts", "pages", "ocr", "seconds"}`` - ``texts`` by page
    number, kept so the full text does not read those pages again. ``verified``
    is False when no page read has any text: nothing was checked, and the copy
    must not be served.
    """
    from serving.pipeline import _likely_missed, _role_mismatches
    from serving.printed_values import hold_result_to_printed
    from serving.value_flags import route_served

    pages = (pages_to_verify(result.extraction, page_count, verify_page_budget())
             or list(range(1, min(page_count, FALLBACK_PAGES) + 1)))
    started = time.perf_counter()
    text = reader.ocr_text(pdf, name=name, pages=pages)
    served = copy.deepcopy(result)
    hold_result_to_printed(served, text["texts"])
    flags = {_pending(flag) for flag in served.review_flags}
    if result.doc_type == "policy" and result.mode == "image_only":
        flags.update(_likely_missed(served.extraction, text["texts"], lob))
        # A value served under a label the read pages print another value after.
        flags.update(_role_mismatches(served.extraction, text["texts"], lob))
    served.review_flags = sorted(flags)
    # A value printed only off the pages it cites - here, the pages the fields
    # cite - or under a label printing another goes to review now, not only at
    # the end (serving.value_flags).
    route_served(served.extraction, served.review_flags)
    # And its confidence computed again on the text of the pages it cites -
    # every page any field cites was read - as the full text will at the end.
    rescore_on_text(served, text["texts"])
    if served.printed_check is not None:
        served.printed_check = {**served.printed_check, "pages_read": pages, "whole_document": False}
    return {"served": served, "verified": served.printed_check is not None, "texts": text["texts"],
            "pages": pages, "ocr": text["info"], "seconds": round(time.perf_counter() - started, 2)}


def read_fields(pdf: bytes, *, name: str, lob: str, source_id: str, model: Any, reader: Any, pages_dir: Path,
                classifier: Any, calibration: Any, publish_pages: Any = None,
                turns: GpuTurns | None = None, verify: bool = True,
                extract_options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Phase 1: the pages rendered, the model reading their images, and the
    fields held to the pages they cite (:func:`verify_fields`).

    Returns ``{"result", "served", "verified", "cited_texts", "images",
    "timings_s"}``: ``result`` is the model's answer, kept for the full text;
    ``served`` the fields to serve now. Should the verification fail - or the
    pages the fields cite have no text, so that nothing could be checked -
    ``served`` is None and ``verified`` False: nothing is served before the full
    text. ``verify=False`` serves the answer unchecked, flagged
    ``fields:unverified`` - for timing tests only. ``extract_options`` go to
    :func:`serving.pipeline.extract` as they are: the release's calibrators and
    thresholds, the serving thresholds.
    """
    from serving.pipeline import ExtractionRequest, extract

    began = time.perf_counter()
    # The turn is held from the start, rendering included, though rendering is
    # the CPU's: a run of MinerU started while the pages render would still be
    # reading when the model starts, on the GPU the gate needs for the model.
    # The verification's few pages are this document's own turn too.
    with turns.fields() if turns is not None else nullcontext():
        read = reader.read(pdf, pages_dir, name=name, with_text=False, publish_pages=publish_pages)
        rendered = time.perf_counter()
        request = ExtractionRequest(
            source_id=source_id, image_paths=read["images"], page_texts={}, modality_mode="image_only",
            known_doc_type="policy", known_lob=lob, source_file_name=name,
        )
        result = extract(request, model, classifier, calibration, strict_schema=False,
                         **dict(extract_options or {}))
        answered = time.perf_counter()
        checked: dict[str, Any] | None = None
        served: Any = None
        if verify:
            try:
                checked = verify_fields(pdf, result, name=name, lob=lob, reader=reader,
                                        page_count=len(read["images"]))
            except Exception:  # noqa: BLE001 - nothing unchecked is served: the fields wait for the full text
                log.exception("%s: the fields could not be checked against their pages; they are served with "
                              "the full text", source_id)
            else:
                if checked["verified"]:
                    served = checked["served"]
                else:
                    log.warning("%s: no page the fields cite has text (pages %s); nothing was checked, so they "
                                "are served with the full text", source_id, checked["pages"])
        else:
            served = copy.deepcopy(result)
            served.review_flags = sorted({*served.review_flags, UNVERIFIED_FLAG})
    done = time.perf_counter()
    return {
        "result": result,
        "served": served,
        "verified": checked is not None and checked["verified"],
        "cited_texts": checked["texts"] if checked else {},
        "cited_pages": checked["pages"] if checked else [],
        "images": read["images"],
        "links": read.get("links"),
        "timings_s": {
            "render": round(read["timings_s"]["render"], 2),
            "links": read["timings_s"].get("links"),
            "model": round(answered - rendered, 2),
            "verify": checked["seconds"] if checked else None,
            "fields": round(done - began, 2),
        },
    }


def add_full_text(pdf: bytes, result: Any, *, name: str, lob: str, reader: Any, pages_dir: Path,
                  page_count: int, turns: GpuTurns | None = None,
                  known_texts: dict[int, str] | None = None,
                  calibrators: Any = FROM_RESULT, thresholds: Any = FROM_RESULT) -> dict[str, Any]:
    """Phase 2: MinerU reads the pages not yet read - a run of pages at a time,
    in the GPU's idle turns when ``turns`` is given - and ``result``, the
    model's answer, is held to the whole document's text and joined by the
    full-text tier; then its fields' confidence is computed again on that text
    (:func:`rescore_on_text`) with the release calibrators phase 1 scored it
    with - ``calibrators``/``thresholds``, by default those the result carries
    (a result restored from a checkpoint carries none: pass them).

    ``known_texts``: pages already read (the cited pages, by
    :func:`verify_fields`). Each page's text lands in ``<pages dir>/page_NNN.md``.
    Returns what MinerU reported and how long it took.
    """
    from serving.pipeline import attach_full_text

    began = time.perf_counter()
    texts = dict(known_texts or {})
    remaining = [page for page in range(1, page_count + 1) if page not in texts]
    info: dict[str, Any] = {}
    seconds = 0.0
    if remaining:
        text = reader.ocr_text(pdf, name=name, pages=remaining if texts else None,
                               page_count=None if texts else page_count,
                               gate=turns.wait_for_text_turn if turns is not None else None)
        texts.update(text["texts"])
        info, seconds = text["info"], text["seconds"]
    texts = dict(sorted(texts.items()))
    pages_dir.mkdir(parents=True, exist_ok=True)
    for number, page in texts.items():
        (pages_dir / f"page_{number:03d}.md").write_text(page, encoding="utf-8")
    attach_full_text(result, texts, lob=lob)
    rescore_on_text(result, texts, calibrators=calibrators, thresholds=thresholds)
    return {"ocr": info, "timings_s": {"ocr": round(seconds, 2), "pages_read": len(remaining),
                                       "full_text": round(time.perf_counter() - began, 2)}}
