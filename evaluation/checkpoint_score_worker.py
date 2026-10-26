"""One GPU's share of checkpoint selection (evaluation.checkpoint_eval.ParallelScorer).

Started by the parent with ``CUDA_VISIBLE_DEVICES`` set to one GPU. Builds the
in-process scorer there - :func:`evaluation.checkpoint_eval.vllm_scorer`, so the
engine, the rows and the scoring are the ones a single-GPU selection uses - scores
each ``--checkpoint`` in turn, and writes ``{checkpoint: {"metrics": ...}}`` (or
``{"error": ...}`` for one that could not be scored) to ``--out``.

    python -m evaluation.checkpoint_score_worker --val-path <key> --sample-rows 400 \\
        --images-root <dir> --checkpoint <dir> [--checkpoint <dir> ...] --out scores.json \\
        [--full [--full-mode ocr_plus_image ...]] [--repeat [--repeat-mode image_only ...]] \\
        [--save-dir <dir>]

With ``--shard K --shards N`` it scores nothing: it generates shard K of N of
each checkpoint's calls and writes the answers beside ``--out``
(:func:`evaluation.checkpoint_eval.generate_shard`), for the parent to put the
pass back together and score (``generate_sharded``). ``--only-keys`` narrows
the pass to the readings a JSON list names (calibration's rows no saved answer covers).
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
    parser.add_argument("--repeat-mode", dest="repeat_modes", action="append", default=[],
                        help="read only this mode's rows twice (repeatable; none = every row)")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="save each finalist's answers and error tallies here")
    parser.add_argument("--shard", type=int, default=None, help="generate this shard of each pass's calls")
    parser.add_argument("--shards", type=int, default=None, help="...of this many")
    parser.add_argument("--only-keys", type=Path, default=None,
                        help="a JSON list of the readings to generate (with --shard)")
    parser.add_argument("--prepare-adapter", action="store_true",
                        help="a --checkpoint that is no local directory is a Blob prefix: copy it down first")
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
    from evaluation import checkpoint_eval

    if args.shards is not None:
        if args.shard is None:
            parser.error("--shards needs --shard")
        only_keys = json.loads(args.only_keys.read_text(encoding="utf-8")) if args.only_keys else None
        checkpoint_eval.generate_shard(
            client=BlobClient(), val_path=args.val_path, images_root=args.images_root,
            sample_rows=args.sample_rows, full=args.full, full_modes=args.full_modes,
            checkpoints=args.checkpoint, shard=args.shard, shards=args.shards, out=args.out,
            repeat=args.repeat, repeat_modes=args.repeat_modes, only_keys=only_keys,
            prepare_adapters=args.prepare_adapter)
        return 0

    extras = {}
    if args.repeat_modes:
        extras["repeat_modes"] = args.repeat_modes
    if args.save_dir is not None:
        extras["save_dir"] = args.save_dir
    scorer = checkpoint_eval.vllm_scorer(client=BlobClient(), val_path=args.val_path, images_root=args.images_root,
                                         sample_rows=args.sample_rows, full_modes=args.full_modes,
                                         repeat=args.repeat, **extras)
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
