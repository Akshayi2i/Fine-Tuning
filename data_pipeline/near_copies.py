"""Documents that share their pages, found by rendering them. Read-only: nothing is moved or dropped.

Ingestion catches a PDF identical byte for byte. Two PDFs of one policy differ as files and still hold the same pages,
and one carrier's template gives different insureds the same pages but for a few. Cycle 1 held both, across its splits,
and neither its checks nor its families saw them: families join documents on insured name, policy number and mailing
address (``prepare_bundles._insured_families``), so a template is no family.

Every page is rendered small and in gray, and its pixels hashed, so a re-saved PDF matches. Each pair sharing at least
``--min-share`` of the smaller document's distinct pages is reported with what its labels say it is:

* ``same policy``: they share a policy number;
* ``same insured``: they share a name or a mailing address, not a number;
* ``template``: nothing of the insured is shared, only the pages;
* ``twin``: a synthetic document and its own source, or two twins of it, which share pages by design.

    python -m data_pipeline.near_copies --input data/bundles_c1
    python -m data_pipeline.near_copies --input data/bundles_c1 --json data/audit_report/near_copies.json

A folder is one document: ``document.pdf``, ``golden.json`` and ``metadata.json``; ``split.csv`` beside them, when
there is one, says which split each is in. Page hashes are kept in ``--cache`` by the PDF's checksum, so a run over a
grown corpus renders only what is new.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

SAME_POLICY, SAME_INSURED, TEMPLATE, TWIN = "same policy", "same insured", "template", "twin"
#: Pages rendered this small: a page's identity, not its text.
RENDER_DPI = 50
DEFAULT_MIN_SHARE = 0.5


@dataclass
class Document:
    name: str
    pages: set[str]
    page_count: int
    identity: set[str] = field(default_factory=set)
    split: str | None = None
    synthetic: bool = False
    source: str | None = None


def page_hashes(pdf: Path) -> list[str]:
    """A hash of each page's pixels, rendered small and in gray."""
    import pymupdf

    with pymupdf.open(pdf) as document:
        return [hashlib.sha1(page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csGRAY).samples).hexdigest()
                for page in document]


def pair_kind(a: Document, b: Document) -> str:
    """What two documents sharing pages are, from what their labels hold."""
    if (a.synthetic or b.synthetic) and a.source and a.source == b.source:
        return TWIN
    shared = a.identity & b.identity
    if any(key.startswith("policy:") for key in shared):
        return SAME_POLICY
    return SAME_INSURED if shared else TEMPLATE


def near_copy_pairs(documents: list[Document], min_share: float = DEFAULT_MIN_SHARE) -> list[dict[str, Any]]:
    """Every pair sharing at least ``min_share`` of the smaller document's distinct pages, most shared first."""
    holders: dict[str, list[int]] = defaultdict(list)
    for index, document in enumerate(documents):
        for page in document.pages:
            holders[page].append(index)
    shared: Counter = Counter()
    for indexes in holders.values():
        shared.update(combinations(indexes, 2))
    pairs = []
    for (i, j), count in shared.items():
        a, b = documents[i], documents[j]
        smaller = min(len(a.pages), len(b.pages))
        if smaller and count / smaller >= min_share:
            pairs.append({
                "a": a.name, "b": b.name, "pages_a": a.page_count, "pages_b": b.page_count, "shared_pages": count,
                "share": round(count / smaller, 4), "kind": pair_kind(a, b), "split_a": a.split, "split_b": b.split,
                "crosses_splits": bool(a.split and b.split and a.split != b.split),
            })
    return sorted(pairs, key=lambda p: (-p["share"], p["a"], p["b"]))


def groups_of(pairs: list[dict[str, Any]]) -> list[list[str]]:
    """Documents joined by any pair, transitively; largest group first."""
    parent: dict[str, str] = {}

    def root(name: str) -> str:
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for pair in pairs:
        parent[root(pair["a"])] = root(pair["b"])
    members: dict[str, list[str]] = defaultdict(list)
    for name in parent:
        members[root(name)].append(name)
    return sorted((sorted(group) for group in members.values()), key=lambda g: (-len(g), g))


def load_documents(root: Path, cache_path: Path | None = None, *, log=print) -> list[Document]:
    from data_pipeline.ingestion.prepare_bundles import _identity_keys

    cache: dict[str, list[str]] = {}
    if cache_path and cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    split_file = root / "split.csv"
    splits = {}
    if split_file.is_file():
        with split_file.open(encoding="utf-8", newline="") as handle:
            splits = {row["document"]: row.get("split") for row in csv.DictReader(handle)}
    documents = []
    folders = sorted(p for p in root.iterdir() if p.is_dir() and (p / "document.pdf").is_file())
    for position, folder in enumerate(folders, start=1):
        pdf = folder / "document.pdf"
        checksum = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if checksum not in cache:
            cache[checksum] = page_hashes(pdf)
            log(f"{position}/{len(folders)} rendered {folder.name}: {len(cache[checksum])} page(s)")
        gold_file, metadata_file = folder / "golden.json", folder / "metadata.json"
        gold = json.loads(gold_file.read_text(encoding="utf-8")) if gold_file.is_file() else {}
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.is_file() else {}
        documents.append(Document(
            name=folder.name, pages=set(cache[checksum]), page_count=len(cache[checksum]),
            identity=set(_identity_keys(gold)), split=splits.get(folder.name) or metadata.get("split"),
            synthetic=bool(metadata.get("synthetic")), source=metadata.get("template_id"),
        ))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return documents


def render(documents: list[Document], pairs: list[dict[str, Any]], min_share: float) -> str:
    groups = groups_of(pairs)
    in_groups = sum(len(group) for group in groups)
    out = [f"{len(documents)} document(s); {len(pairs)} pair(s) share at least {min_share:.0%} of their pages",
           f"{in_groups} document(s) sit in {len(groups)} group(s), so {len(documents) - in_groups + len(groups)} "
           "distinct group(s) of pages in all", "", "Pairs by what their labels say they are"]
    kinds = Counter(pair["kind"] for pair in pairs)
    crossing = Counter(pair["kind"] for pair in pairs if pair["crosses_splits"])
    out += [f"  {kind:14} {count:5} pair(s), {crossing.get(kind, 0)} across splits" for kind, count in kinds.most_common()]
    across = [pair for pair in pairs if pair["crosses_splits"]]
    if across:
        out += ["", "Across splits"]
        out += [f"  {p['share']:>4.0%}  {p['kind']:13} {p['split_a']:5} {p['a']}  |  {p['split_b']:5} {p['b']}"
                for p in across]
    copies = [pair for pair in pairs if pair["kind"] == SAME_POLICY]
    if copies:
        out += ["", "One policy number on more than one document"]
        out += [f"  {p['share']:>4.0%}  {p['a']}  |  {p['b']}" for p in copies]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="a folder of document folders")
    parser.add_argument("--min-share", type=float, default=DEFAULT_MIN_SHARE,
                        help="report pairs sharing at least this share of the smaller document's pages")
    parser.add_argument("--cache", default="data/audit_report/page_hashes.json",
                        help="page hashes by PDF checksum, so a later run renders only what is new")
    parser.add_argument("--json", default=None, help="also write every pair to this file")
    args = parser.parse_args(argv)
    # On the pod, run detached in tmux: a corpus of 1,000 documents renders every page once.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("data_pipeline.near_copies", argv):
        return 0
    documents = load_documents(Path(args.input), Path(args.cache) if args.cache else None)
    pairs = near_copy_pairs(documents, args.min_share)
    print(render(documents, pairs, args.min_share))
    if args.json:
        Path(args.json).write_text(json.dumps(pairs, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
