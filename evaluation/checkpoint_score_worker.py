"""One GPU's share of checkpoint selection (evaluation.checkpoint_eval.ParallelScorer).

Started by the parent with ``CUDA_VISIBLE_DEVICES`` set to one GPU. Builds the
in-process scorer there - :func:`evaluation.checkpoint_eval.vllm_scorer`, so the
engine, the rows and the scoring are the ones a single-GPU selection uses - scores
each ``--checkpoint`` in turn, and writes ``{checkpoint: {"metrics": ...}}`` (or
``{"error": ...}`` for one that could not be scored) to ``--out``.

    python -m evaluation.checkpoint_score_worker --val-path <key> --sample-rows 400 \\
        --images-root <dir> --checkpoint <dir> [--checkpoint <dir> ...] --out scores.json \\
        [--full [--full-mode ocr_plus_image ...]] [--repeat]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score checkpoints on one GPU")
    parser.add_argument("--val-path", required=True, help="the scope's validation split in Blob")
    parser.add_argument("--sample-rows", type=int, default=0)
    parser.add_argument("--images-root", default=None)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--full", action="store_true", help="every validation row, not the sample")
    parser.add_argument("--full-mode", dest="full_modes", action="append", default=[],
                        help="narrow --full to this reading mode (repeatable; none = all modes)")
    parser.add_argument("--repeat", action="store_true",
                        help="read the finalists' pass twice, for run-to-run agreement")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    # Started by ParallelScorer with FIDEON_NO_DETACH=1, it scores in place; run by
    # hand on the pod, it moves into tmux like every long job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("evaluation.checkpoint_score_worker", argv):
        return 0
    from common.logging_setup import setup as setup_logging

    setup_logging(logging.INFO)

    from artifact_registry.blob_client import BlobClient
    from evaluation.checkpoint_eval import vllm_scorer

    scorer = vllm_scorer(client=BlobClient(), val_path=args.val_path, images_root=args.images_root,
                         sample_rows=args.sample_rows, full_modes=args.full_modes, repeat=args.repeat)
    target = (getattr(scorer, "full", None) or scorer) if args.full else scorer
    results: dict[str, dict] = {}
    try:
        for checkpoint in args.checkpoint:
            try:
                results[checkpoint] = {"metrics": dict(target(checkpoint))}
            except Exception as exc:  # noqa: BLE001 - recorded per checkpoint, as select_best does
                results[checkpoint] = {"error": f"{type(exc).__name__}: {exc}"}
            # What the pass cost (rows, calls, seconds), for the parent's report.
            passes = getattr(target, "passes", None)
            if passes and passes[-1].get("checkpoint") == checkpoint:
                results[checkpoint]["stats"] = {k: v for k, v in passes[-1].items() if k != "checkpoint"}
            # Written after each, so a worker that dies later keeps what it scored.
            args.out.write_text(json.dumps(results), encoding="utf-8")
    finally:
        if callable(getattr(scorer, "close", None)):
            scorer.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
