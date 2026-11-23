"""Read the frozen set's real documents image-only with a released adapter, through the gate's own reading code, and
score them as the gate scores served accuracy: the check that the serving fixes measured on stored answers hold on
the model itself (accuracy plan, step 4: B2), with the time it took and how fast the page groups were read.

    bash scripts/pod_run.sh start servedcheck -- /workspace/venv/bin/python -m scripts.check_served_fixes \\
        --adapter adapters/scope/policy/v4 --out /workspace/logs/served_check_v4.json

Reads nothing twice and keeps nothing in the gate's saved readings: a measurement, never a gate. Prints served
accuracy per document beside v4's official gate (``--baseline``, its eval report), the dates of any document whose
policy period printed two-digit years, and the page groups read per minute.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def use_pixel_budget(max_pixels: int) -> None:
    """Every full-resolution task read at ``max_pixels`` for this process: the window planner, the page check and the
    engine all take their budget from ``common.config.vision_for_task``, which reads ``shared_vision_config`` - so
    the override is made there, once, and they stay consistent. A budget that is not a whole number of 32x32 visual
    tokens is refused, as ``pixel_budget`` refuses one."""
    import copy

    from common import config
    from data_pipeline.dataset_builder.cap_check import PIXELS_PER_VISUAL_TOKEN

    if max_pixels <= 0 or max_pixels % PIXELS_PER_VISUAL_TOKEN:
        raise SystemExit(f"--max-pixels must be a positive multiple of {PIXELS_PER_VISUAL_TOKEN}")
    original = config.shared_vision_config()
    changed = copy.deepcopy(original)
    for block in [changed.get("defaults") or {}, *(changed.get("tasks") or {}).values(),
                  *(changed.get("profiles") or {}).values()]:
        if isinstance(block, dict) and "max_pixels" in block and int(block["max_pixels"]) > 300_000:
            block["max_pixels"] = max_pixels
    config.shared_vision_config = lambda: changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="the release's adapter: a Blob prefix or a local folder")
    parser.add_argument("--baseline", default="v4", help="the release whose gate report to compare with")
    parser.add_argument("--out", type=Path, required=True, help="where to write the result (JSON)")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--documents", nargs="*", default=None, help="only these frozen source ids")
    parser.add_argument("--max-pixels", type=int, default=None,
                        help="read the pages at this pixel budget instead of the configured one (e.g. 2483200, "
                             "the standard profile's ~2.48 MP, against the fast profile's 1254400): the sharper-"
                             "images test. Applied to every full-resolution task for this run only; vision.yaml "
                             "is not changed")
    args = parser.parse_args(argv)
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("scripts.check_served_fixes", argv):
        return 0
    from common.logging_setup import setup as setup_logging

    setup_logging(logging.INFO)
    log = logging.getLogger("check_served_fixes")
    if args.max_pixels is not None:
        use_pixel_budget(args.max_pixels)
        log.info("reading the pages at max_pixels=%d for this run (vision.yaml unchanged)", args.max_pixels)

    from artifact_registry import paths
    from artifact_registry.blob_client import BlobClient
    from common.adapter import ADAPTER
    from evaluation import golden_eval
    from evaluation.run_eval import build_report
    from inference_core.model_runner import load_model, load_runner_config, prepare_adapter, release_model

    client = BlobClient()
    began = time.monotonic()
    documents = [d for d in golden_eval.load_golden_set(client, ADAPTER.doc_types, tenant_id=args.tenant)
                 if not d.synthetic and (not args.documents or d.source_id in args.documents)]
    log.info("%d real frozen document(s), %d page(s); set loaded in %.0f s", len(documents),
             sum(len(d.image_keys) for d in documents), time.monotonic() - began)
    adapter = prepare_adapter(args.adapter, client, load_runner_config("vllm"))
    model = load_model("base", client, backend="vllm", adapter=adapter, label="served-check")
    loaded = time.monotonic()
    try:
        triples = golden_eval.evaluate(
            documents, model, client, golden_eval.gate_images_root("served-check", args.tenant),
            modes=["image_only"], run_to_run=False)
    finally:
        release_model(model)
    read_s = time.monotonic() - loaded
    body = build_report(f"served-check-{args.baseline}", list(triples)).as_dict()
    served = (body.get("per_document") or {}).get("served_field_match") or {}

    baseline: dict[str, Any] = {}
    key = paths.eval_report(args.baseline, scope=ADAPTER.name)
    if client.exists(key):
        baseline = (client.read_json(key).get("per_document") or {}).get("served_field_match") or {}
    windows = sum(len(meta.get("window_latencies_ms") or []) for _gold, _answer, meta in triples)
    dates = {}
    for _gold, answer, meta in triples:
        policy = (answer or {}).get("policy") or {}
        dates[meta.get("source_id")] = {name: (policy.get(name) or {}).get("raw") if isinstance(policy.get(name), dict)
                                        else policy.get(name) for name in ("effective_date", "expiration_date")}
    result = {
        "adapter": args.adapter, "max_pixels": args.max_pixels or "configured",
        "documents": len(documents), "readings": len(triples),
        "failed": len(getattr(triples, "infrastructure_failures", [])),
        "served_field_match": round(statistics.mean(served.values()), 4) if served else None,
        "baseline_served_field_match": (round(statistics.mean(baseline[s] for s in served if s in baseline), 4)
                                        if baseline and all(s in baseline for s in served) else None),
        "per_document": {sid: {"now": round(v, 4), "baseline": round(baseline[sid], 4) if sid in baseline else None,
                               "dates": dates.get(sid)} for sid, v in sorted(served.items())},
        "reading_seconds": round(read_s, 1), "page_groups": windows,
        "page_groups_per_minute": round(windows / (read_s / 60), 1) if read_s else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
