"""List the insureds whose real documents a manifest.csv delivery puts in more than one split, and propose the
split overrides that put each in one - the rows ``prepare_bundles`` reads from ``data/split_overrides.csv``.

    python -m scripts.split_leaks --input data/cycle2_combined
    python -m scripts.split_leaks --input data/cycle2_combined --write data/split_overrides_proposed.csv

Reads the golds only (no PDF, no storage). One insured is the documents printing the same insured name, policy
number or mailing address (``prepare_bundles._insured_families``, the join the corpus build refuses a split by).
Overrides already in ``--overrides`` are applied first, so a run after the owner's rows shows what is left.

The proposal, for each group: **test** when any of its documents is in test (the test set is the yardstick: its
documents stay where they are, and the insured's others leave training); otherwise the split holding most of its
real documents, a tie going to **train**. Where a move leaves a line with no real document in a split, it says so:
the proposal is a starting point, the owner decides.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from data_pipeline.ingestion.prepare_bundles import (
    DEFAULT_SPLIT_OVERRIDES,
    SPLITS,
    insured_split_groups,
    read_split_overrides,
    template_of,
)


def manifest_documents(delivery: Path, overrides: dict[str, str]) -> tuple[list[tuple], dict[str, dict]]:
    """``(planned real documents, {source: {split, lob, copies}})`` from ``<delivery>/manifest.csv``: each usable
    row (``ok`` true, gold on disk) with ``overrides`` applied, as ``prepare_bundles`` would plan it."""
    with (delivery / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    planned: list[tuple] = []
    sources: dict[str, dict] = {}
    for row in rows:
        if (row.get("ok") or "").strip().lower() != "true" or not (delivery / row["gold"]).is_file():
            continue
        source = template_of(row)
        split = overrides.get(source) or row["split"].strip().split("/")[-1].lower()
        info = sources.setdefault(source, {"split": split, "lob": row["lob"].strip(), "copies": 0})
        if row["kind"].strip().lower() == "synthetic":
            info["copies"] += 1
            continue
        planned.append((Path(row["pdf"]).stem, delivery / row["pdf"], delivery / row["gold"],
                        {"split": split, "lob": row["lob"].strip(), "synthetic": False, "source": source}))
    return planned, sources


def proposed_split(by_split: dict[str, list[str]]) -> str:
    """Test when any document is in test; else the split with most documents, a tie going to train."""
    if "test" in by_split:
        return "test"
    return max(by_split, key=lambda split: (len(by_split[split]), split == "train"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="the delivery: manifest.csv + golds")
    parser.add_argument("--overrides", type=Path, default=DEFAULT_SPLIT_OVERRIDES,
                        help="the owner's split overrides, applied first")
    parser.add_argument("--write", type=Path, default=None,
                        help="write the proposed rows (source,split,reason) here; never over an existing file")
    args = parser.parse_args(argv)

    overrides = read_split_overrides(args.overrides)
    planned, sources = manifest_documents(args.input, overrides)
    source_of = {name: metadata["source"] for name, _pdf, _gold, metadata in planned}
    groups = [by_split for by_split in insured_split_groups(planned).values() if len(by_split) > 1]
    print(f"{len(planned)} real document(s); {len(overrides)} override(s) applied from {args.overrides}")
    if not groups:
        print("No insured has real documents in more than one split.")
        return 0

    proposal: list[dict[str, str]] = []
    after = {source: info["split"] for source, info in sources.items()}
    print(f"\n{len(groups)} insured(s) in more than one split:")
    for by_split in groups:
        target = proposed_split(by_split)
        names = sorted(name for members in by_split.values() for name in members)
        print(f"\n  {', '.join(names)}")
        for split in SPLITS:
            for name in sorted(by_split.get(split, [])):
                source = source_of[name]
                move = "" if split == target else f"  -> {target}"
                print(f"    {split:5} {name} (+{sources[source]['copies']} copies){move}")
                if split != target:
                    after[source] = target
                    proposal.append({"source": source, "split": target,
                                     "reason": f"same insured as {', '.join(n for n in names if n != name)}"})

    before_counts = Counter((info["lob"], info["split"]) for source, info in sources.items()
                            if any(source_of[n] == source for n in source_of))
    after_counts = Counter((sources[source]["lob"], after[source]) for source in set(source_of.values()))
    print("\nReal documents per line, now -> after the proposal:")
    lines = sorted({lob for lob, _ in before_counts} | {lob for lob, _ in after_counts})
    empty = []
    for lob in lines:
        cells = ", ".join(f"{split} {before_counts[(lob, split)]}->{after_counts[(lob, split)]}" for split in SPLITS)
        print(f"  {lob:14} {cells}")
        empty += [f"{lob} {split}" for split in SPLITS
                  if before_counts[(lob, split)] and not after_counts[(lob, split)]]
    if empty:
        print(f"\nWATCH: the proposal leaves no real document in: {', '.join(empty)}")

    if args.write:
        if args.write.exists():
            print(f"\n{args.write} exists: not written", file=sys.stderr)
            return 1
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with args.write.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["source", "split", "reason"])
            writer.writeheader()
            writer.writerows(proposal)
        print(f"\nProposed {len(proposal)} row(s) written to {args.write}; once agreed, they go in {args.overrides}")
    by_target = defaultdict(int)
    for row in proposal:
        by_target[row["split"]] += 1
    print(f"\n{len(proposal)} move(s) proposed: " + ", ".join(f"{n} to {s}" for s, n in sorted(by_target.items())))
    return 1


if __name__ == "__main__":
    sys.exit(main())
