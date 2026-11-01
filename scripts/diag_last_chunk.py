"""Show how heavy each call of checkpoint selection's finalists' pass is, and which rows make it so.

    python scripts/diag_last_chunk.py [--corpus v3] [--scope policy] [--mode ocr_plus_image]

Rebuilds the plan the finalists' whole-document pass uses (the validation rows read
in one mode, in calls of ``configured_batch_rows()`` rows) and prints, per call, its
rows, total page-image megapixels and heaviest rows; for the last call, every row's
pages, largest page side, prompt text length and expected answer length, and any page
image that does not open. Reads the split and the cached page images; it loads no
model and needs no GPU. Written for a run whose last call stalled with the GPU idle
and one core busy, to see whether one row's input is the cause.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _row_facts(row: dict) -> dict:
    """Pages, megapixels, largest page side and text length of one row's prompt, and any unreadable page."""
    from PIL import Image

    sizes: list[tuple[int, int]] = []
    unreadable: list[str] = []
    text_chars = 0
    for message in row.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "image":
                try:
                    with Image.open(block["image"]) as image:
                        sizes.append(image.size)
                        image.verify()
                except Exception as exc:  # noqa: BLE001 - an unreadable page is the finding
                    unreadable.append(f"{block['image']}: {type(exc).__name__}")
            elif block.get("type") == "text" and message.get("role") != "assistant":
                text_chars += len(block.get("text", ""))
    return {
        "source_id": row.get("source_id"),
        "pages": len(sizes),
        "mpix": round(sum(w * h for w, h in sizes) / 1e6, 1),
        "max_side": max((max(size) for size in sizes), default=0),
        "text_chars": text_chars,
        "unreadable": unreadable,
    }


def main(argv: list[str] | None = None) -> int:
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from evaluation.checkpoint_eval import rows_in_modes
    from evaluation.validation_generation import _answer_length, configured_batch_rows, read_rows
    from training.stage_data import localize_rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default="v3")
    parser.add_argument("--scope", default="policy")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--mode", default="ocr_plus_image", help="the reading mode the finalists' pass reads")
    parser.add_argument("--top", type=int, default=12, help="heaviest rows to list for the last call")
    args = parser.parse_args(argv)

    client = BlobClient()
    split = paths.corpus_scope_eval_split(args.corpus, "val", args.scope, args.tenant)
    rows = read_rows(client.read_text(split))
    chosen = rows_in_modes(rows, [args.mode])
    size = configured_batch_rows()
    print(f"validation rows: {len(rows)}; read in {args.mode}: {len(chosen)}; rows per call: {size}")
    chosen = localize_rows(chosen, client, paths.staging_train_images_dir(args.corpus, args.tenant))

    calls = [chosen[first:first + size] for first in range(0, len(chosen), size)]
    print(f"calls: {len(calls)} of {[len(call) for call in calls]} rows")
    for number, call in enumerate(calls, 1):
        facts = sorted((_row_facts(row) | {"answer_chars": _answer_length(row)} for row in call),
                       key=lambda f: -f["mpix"])
        print(f"\ncall {number}: {len(call)} rows, {sum(f['mpix'] for f in facts):.0f} Mpix, "
              f"{sum(f['pages'] for f in facts)} pages, max page side {max(f['max_side'] for f in facts)}")
        if number == len(calls):
            for fact in facts[:args.top]:
                print(" ", {key: value for key, value in fact.items() if key != "unreadable"})
            bad = [(f["source_id"], f["unreadable"]) for f in facts if f["unreadable"]]
            print("unreadable page images:", bad or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
