"""Before training: did every labelled document reach the corpus, and is its page text sound?

Cycle 1 trained without 8 of its 84 policies (5 train, 2 val, 1 test) and nothing stopped it: one
blank "Hazardous Materials listing" form, OCR'd with 14 pictures inside its table, gave a page of
156,140 characters, a few rows went over their token budget, the build set each document aside
whole with one warning line, and 8 of 84 was under the 25% guard. Found on 2026-11-07, by hand.

This report is made by the dataset build (stage 4) before anything is written, and by
``python -m data_pipeline.data_health`` on its own. It holds:

* **documents that would not reach the corpus**, with why: labelled but not loaded (no OCR, render
  only, no pages) and set aside by the build (over the token budget, empty label, expansion failed).
  Each one STOPS the build until it is fixed or the person agrees to train without it
  (``--accept-set-aside``). This is the only check that stops anything.
* documents left out by design: the frozen eval set and its families, twins over the per-seed cap;
* pages whose stored OCR text held pictures, taken out when read (``common.ocr_text``);
* pages with no text at all, per document;
* the largest pages by characters, to see one that is out of line;
* rows over their token budget (the build's cap report);
* with ``--images`` (standalone only: one storage call per page), the same page image read into
  different OCR text - the same picture should give the same text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: How many of the largest pages the report lists.
LARGEST_PAGES = 10
#: How many entries of a long list the printed report shows (the JSON keeps them all).
SHOWN = 20
#: Storage calls made at once by the image check.
IMAGE_WORKERS = 16


@dataclass
class DataHealth:
    """What the corpus would hold, and what it would not, before it is written."""

    documents_loaded: int
    documents_kept: int
    not_loaded: list[tuple[str, str]] = field(default_factory=list)
    set_aside: list[tuple[str, str]] = field(default_factory=list)
    excluded_eval: list[str] = field(default_factory=list)
    twins_left_out: list[str] = field(default_factory=list)
    picture_pages: list[dict[str, Any]] = field(default_factory=list)
    blank_pages: dict[str, list[int]] = field(default_factory=dict)
    largest_pages: list[dict[str, Any]] = field(default_factory=list)
    cap_report: dict[str, Any] = field(default_factory=dict)
    #: None when the image check was not run.
    same_image_different_text: list[dict[str, Any]] | None = None
    accepted: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[tuple[str, str]]:
        """Every labelled document that would not reach the corpus, with why."""
        return list(self.not_loaded) + list(self.set_aside)

    def unaccepted(self, accepted: Iterable[str] = ()) -> list[tuple[str, str]]:
        agreed = set(accepted)
        return [(source_id, why) for source_id, why in self.missing if source_id not in agreed]

    def accepted_but_not_missing(self, accepted: Iterable[str]) -> list[str]:
        """Ids agreed to with ``--accept-set-aside`` that are not missing: a typo, or fixed since."""
        missing = {source_id for source_id, _why in self.missing}
        return sorted(set(accepted) - missing)

    def as_dict(self) -> dict[str, Any]:
        body = asdict(self)
        body["not_loaded"] = [{"source_id": s, "reason": r} for s, r in self.not_loaded]
        body["set_aside"] = [{"source_id": s, "reason": r} for s, r in self.set_aside]
        body["summary"] = {
            "documents_missing": len(self.missing),
            "documents_missing_not_accepted": len(self.unaccepted(self.accepted)),
            "pages_with_pictures": len(self.picture_pages),
            "pictures_removed": sum(p["pictures"] for p in self.picture_pages),
            "characters_removed": sum(p["characters_removed"] for p in self.picture_pages),
            "documents_with_blank_pages": len(self.blank_pages),
            "blank_pages": sum(len(pages) for pages in self.blank_pages.values()),
            "same_image_different_text_groups": (None if self.same_image_different_text is None
                                                 else len(self.same_image_different_text)),
        }
        return body

    def render(self) -> str:
        """The report as a person reads it."""
        summary = self.as_dict()["summary"]
        out = ["DATA HEALTH",
               f"  documents loaded: {self.documents_loaded}; reaching the corpus: {self.documents_kept}"]
        if self.missing:
            out.append(f"  NOT reaching the corpus: {len(self.missing)}")
            for source_id, why in self.missing[:SHOWN]:
                mark = " (accepted)" if source_id in set(self.accepted) else ""
                out.append(f"    {source_id}{mark}: {_short(why)}")
            if len(self.missing) > SHOWN:
                out.append(f"    ... and {len(self.missing) - SHOWN} more (data_health.json)")
        else:
            out.append("  every loaded document reaches the corpus")
        if self.excluded_eval:
            out.append(f"  left out by design, frozen eval set and its families: {len(self.excluded_eval)}")
        if self.twins_left_out:
            out.append(f"  left out by design, twins over the per-seed cap: {len(self.twins_left_out)}")
        out.append(f"  pages whose OCR text held pictures (taken out): {summary['pages_with_pictures']} "
                   f"({summary['pictures_removed']} pictures, {summary['characters_removed']:,} characters)")
        for page in self.picture_pages[:SHOWN]:
            out.append(f"    {page['source_id']} page {page['page']}: {page['pictures']} picture(s), "
                       f"{page['characters_removed']:,} characters")
        out.append(f"  pages with no text: {summary['blank_pages']} in {summary['documents_with_blank_pages']} "
                   "document(s)")
        for source_id, pages in sorted(self.blank_pages.items(), key=lambda kv: -len(kv[1]))[:SHOWN]:
            out.append(f"    {source_id}: {len(pages)} page(s): {_pages(pages)}")
        if self.largest_pages:
            out.append("  largest pages (characters, after pictures are taken out):")
            for page in self.largest_pages:
                out.append(f"    {page['source_id']} page {page['page']}: {page['characters']:,}")
        rejected = self.cap_report.get("documents_rejected") if isinstance(self.cap_report, dict) else None
        if rejected:
            out.append(f"  documents with a row over its token budget: {rejected} (largest rows by task: "
                       f"{self.cap_report.get('max_tokens_seen_by_task')})")
        if self.same_image_different_text is None:
            out.append("  same image, different text: not checked (python -m data_pipeline.data_health --images)")
        else:
            groups = self.same_image_different_text
            pages = sum(len(g["pages"]) for g in groups)
            out.append(f"  same page image read into different text: {len(groups)} image(s), {pages} page(s)")
            for group in groups[:SHOWN]:
                shown = ", ".join(f"{p['source_id']} p{p['page']}" for p in group["pages"][:6])
                out.append(f"    {group['distinct_texts']} different texts: {shown}")
        return "\n".join(out)


def _short(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _pages(pages: Sequence[int], limit: int = 15) -> str:
    shown = ", ".join(map(str, pages[:limit]))
    return shown + (f", ... ({len(pages)} in all)" if len(pages) > limit else "")


def assess(
    documents: Sequence[Any],
    built: Any,
    *,
    not_loaded: Sequence[tuple[str, str]] = (),
    picture_pages: Sequence[dict[str, Any]] = (),
    excluded_eval: Sequence[str] = (),
    twins_left_out: Sequence[str] = (),
    accepted: Iterable[str] = (),
    largest: int = LARGEST_PAGES,
) -> DataHealth:
    """The report for ``documents`` (each with ``source_id`` and ``ocr_pages``) as ``built`` would write them."""
    kept = {row["source_id"] for row in built.all_rows}
    blank: dict[str, list[int]] = {}
    sizes: list[tuple[int, str, int]] = []
    for document in documents:
        source_id = getattr(document, "source_id", None)
        pages = list(getattr(document, "ocr_pages", None) or [])
        empty = [number for number, text in enumerate(pages, start=1) if not str(text or "").strip()]
        if empty and source_id:
            blank[source_id] = empty
        sizes += [(len(text or ""), source_id, number) for number, text in enumerate(pages, start=1) if source_id]
    sizes.sort(key=lambda item: (-item[0], item[1], item[2]))
    cap = built.cap_report.as_dict() if hasattr(built.cap_report, "as_dict") else dict(built.cap_report or {})
    set_aside = [(str(s), str(r)) for s, r in built.skipped]
    # A loaded document in no row that the build never named: lost all the same, so named here.
    named = {source_id for source_id, _reason in set_aside}
    ids = [getattr(document, "source_id", None) for document in documents]
    set_aside += [(source_id, "in no row of the built corpus, and the build recorded no reason")
                  for source_id in ids if source_id and source_id not in kept and source_id not in named]
    return DataHealth(
        documents_loaded=len(documents) + len(excluded_eval) + len(twins_left_out),
        documents_kept=len(kept),
        not_loaded=[(str(s), str(r)) for s, r in not_loaded],
        set_aside=set_aside,
        excluded_eval=sorted(map(str, excluded_eval)),
        twins_left_out=sorted(map(str, twins_left_out)),
        picture_pages=sorted((dict(p) for p in picture_pages),
                             key=lambda p: (-p["characters_removed"], p["source_id"], p["page"])),
        blank_pages=blank,
        largest_pages=[{"source_id": s, "page": p, "characters": n} for n, s, p in sizes[:largest]],
        cap_report=cap,
        accepted=sorted(set(accepted)),
    )


def assess_plan(ctx: Any, plan: Any) -> DataHealth:
    """:func:`assess` for a planned corpus (``pipeline_dag.plan_corpus``) and the context that loaded it."""
    return assess(
        plan.documents, plan.built,
        not_loaded=getattr(ctx, "documents_not_loaded", ()),
        picture_pages=getattr(ctx, "ocr_pictures", ()),
        excluded_eval=getattr(plan, "excluded_eval", ()),
        twins_left_out=getattr(plan, "twins_left_out", ()),
        accepted=getattr(ctx, "accept_set_aside", ()),
    )


def stop_message(health: DataHealth, accepted: Iterable[str] = ()) -> str | None:
    """Why the build must stop, or None: a labelled document that would not reach the corpus and that the
    person has not agreed to leave out."""
    left = health.unaccepted(accepted)
    if not left:
        return None
    lines = [f"{len(left)} labelled document(s) would not reach the corpus, and training would run without "
             "them. Nothing was written."]
    lines += [f"  {source_id}: {_short(why)}" for source_id, why in left[:SHOWN]]
    if len(left) > SHOWN:
        lines.append(f"  ... and {len(left) - SHOWN} more (data_health.json beside the corpus)")
    lines.append("Fix each (the reason says how), or, to train without them knowingly, run the same command "
                 "again with --accept-set-aside " + ",".join(source_id for source_id, _why in left))
    return "\n".join(lines)


def _normalized(text: str) -> str:
    """As ``evaluation.metrics.near_copy`` compares pages: no spacing, no case."""
    return re.sub(r"\s+", "", str(text or "")).casefold()


def same_image_different_text(client: Any, documents: Sequence[Any],
                              workers: int = IMAGE_WORKERS) -> list[dict[str, Any]]:
    """Page images that are byte-identical but whose OCR text differs, across ``documents``
    (each with ``source_id``, ``image_paths`` and ``ocr_pages``).

    By the MD5 stored with each image (``BlobClient.md5``, one properties call a page), or the
    image's own bytes when none is stored.
    """
    from concurrent.futures import ThreadPoolExecutor

    pages = [(document.source_id, number, key, text)
             for document in documents
             for number, (key, text) in enumerate(zip(document.image_paths, document.ocr_pages, strict=False),
                                                  start=1)]

    def digest(key: str) -> str:
        recorded = client.md5(key)
        return recorded.hex() if recorded else hashlib.md5(client.read_bytes(key)).hexdigest()

    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="image-md5") as pool:
        digests = list(pool.map(digest, [key for _s, _n, key, _t in pages]))
    by_image: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for (source_id, number, _key, text), image in zip(pages, digests, strict=True):
        by_image[image].append((source_id, number, _normalized(text)))
    groups = []
    for image, members in by_image.items():
        texts = {text for _s, _n, text in members}
        if len(members) > 1 and len(texts) > 1:
            groups.append({"image_md5": image, "distinct_texts": len(texts),
                           "pages": [{"source_id": s, "page": n} for s, n, _t in members]})
    groups.sort(key=lambda g: (-len(g["pages"]), g["image_md5"]))
    return groups


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over plan_corpus and assess
    parser = argparse.ArgumentParser(
        prog="python -m data_pipeline.data_health",
        description="Report, before training, which labelled documents would not reach the corpus and why, "
                    "and what their page text looks like. Writes nothing to storage.")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--images", action="store_true",
                        help="also find page images that are identical but read into different text "
                             "(one storage call per page)")
    parser.add_argument("--accept-set-aside", dest="accept_set_aside", default="",
                        help="comma-separated ids agreed to stay out of the corpus")
    parser.add_argument("--out", default=None, help="write the full report here as JSON")
    args = parser.parse_args(argv)
    # It reads every page of every labelled document, and with --images asks storage about each page image:
    # minutes on a full corpus, so on the pod it runs in tmux like every long job (orchestration.detach).
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.data_health", argv):
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    from orchestration.pipeline_dag import plan_corpus
    from orchestration.run import accepted_set_aside, build_context

    ctx = build_context(argparse.Namespace(tenant=args.tenant, dry_run=False),
                        accept_set_aside=accepted_set_aside(args.accept_set_aside))
    plan = plan_corpus(ctx)
    health = assess_plan(ctx, plan)
    if args.images:
        health.same_image_different_text = same_image_different_text(ctx.client, plan.documents)
    print(health.render())
    if typo := health.accepted_but_not_missing(ctx.accept_set_aside):
        print(f"\n--accept-set-aside names document(s) that are not missing: {typo}")
    if args.out:
        Path(args.out).write_text(json.dumps(health.as_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {args.out}")
    message = stop_message(health, ctx.accept_set_aside)
    if message:
        print("\n" + message)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
