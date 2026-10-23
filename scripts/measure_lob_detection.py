"""Measure how well the base model reads a policy's line of business, before detection is switched on.

    python scripts/measure_lob_detection.py --corpus v3 [--tenant T] [--mode ocr_plus_image] [--out reports/lob_detection.json]

Reads the tenant's frozen golden eval set the way the golden eval does, withholds each
policy's line, and has the base model read it zero-shot - the classifier serving
would run (``ZeroShotClassifier``) - resolved at the endpoint's own thresholds
(evaluation.lob_detection_eval). ``routing.detect_lob`` stays off until this
report clears its floor and the thresholds are fitted from its sweep. ``--corpus``
is the corpus the set was frozen from; it names the page cache the golden eval
shares. Needs the GPU to itself: stop any running pipeline first.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", required=True,
                        help="the corpus the eval set was frozen from (names the local page cache)")
    parser.add_argument("--mode", choices=["ocr_plus_image", "noisy_ocr_image", "image_only"],
                        default="ocr_plus_image")
    parser.add_argument("--format", dest="quant_format", default=None)
    parser.add_argument("--limit", type=int, default=None, help="read only the first N policies")
    parser.add_argument("--tenant", default=None,
                        help="whose frozen eval set to read; each tenant has its own")
    parser.add_argument("--out", default="reports/lob_detection.json")
    args = parser.parse_args(argv)

    # On the pod, run detached in tmux: loading vLLM takes minutes.
    from orchestration.detach import detach_script_if_needed

    if detach_script_if_needed(__file__, argv, "lob-detection"):
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from evaluation.freeze_eval_set import is_frozen
    from evaluation.golden_eval import load_golden_set
    from evaluation.lob_detection_eval import detection_cases, measure_lob_detection, serving_lob_thresholds
    from evaluation.run_eval import eval_set_keys
    from inference_core.model_runner import generate, load_model, release_model
    from serving.doc_type_classifier import ZeroShotClassifier
    from training.stage_data import localize_keys

    client = BlobClient()
    # is_frozen first: it also refuses a set still frozen at the unscoped root.
    if not is_frozen(client, tenant_id=args.tenant) and eval_set_keys(client, args.tenant):
        print("the golden eval set has documents but no manifest: finish its freeze first", file=sys.stderr)
        return 1
    documents = load_golden_set(client, ["policy"], tenant_id=args.tenant)[: args.limit]
    local = localize_keys(client, sorted({key for doc in documents for key in doc.image_keys}),
                          paths.staging_train_images_dir(f"golden-{args.corpus}", args.tenant))
    cases = detection_cases(documents, local, mode=args.mode)
    if not cases:
        print("the frozen eval set holds no policy with one known line", file=sys.stderr)
        return 1

    line_threshold, uncertain_threshold = serving_lob_thresholds()
    model = load_model("base", client, quant_format=args.quant_format)
    try:
        body = measure_lob_detection(cases, ZeroShotClassifier(model, generate),
                                     line_threshold=line_threshold, uncertain_threshold=uncertain_threshold)
    finally:
        release_model(model)
    body = {"mode": args.mode, "model": getattr(model, "tag", "base"), **body}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{body['documents']} policies: accuracy {body['overall']}, confidently wrong "
          f"{body['confidently_wrong_rate']}, below the floor: {body['below_floor'] or 'none'}; "
          f"{'clears' if body['clears_floor'] else 'does NOT clear'} the {body['floor']} floor")
    print(f"full report: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
