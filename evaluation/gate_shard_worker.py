"""One GPU's share of the gate's readings (evaluation.golden_eval.read_on_every_gpu).

Started by the gate with ``CUDA_VISIBLE_DEVICES`` set to one GPU. Loads the model as the gate loads it (the bf16
base with the release's adapter applied, under the gate's label), reads its share of the frozen documents - first
readings and repeats, through :func:`evaluation.golden_eval.evaluate`, the gate's own reading code - and keeps each
reading in the gate's saved readings, under the name the gate will look them up by. It scores nothing: the gate
then takes every kept reading and scores them all together.

    python -m evaluation.gate_shard_worker --spec <work_dir>/gate-spec.pkl --share 0 --out gate-gpu0.json

The spec (written by the gate) holds the release, the calibrators and thresholds, the adapter (a local folder),
the label, the frozen documents, the name of the saved readings and the split. The worker works the name out
again before it loads the model, and checks the loaded model's tag after, and refuses to read when either
differs: readings kept under another name are readings the gate never finds.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any


def read_share(spec: dict[str, Any], share: int, *, client: Any = None, loader: Any = None) -> dict[str, Any]:
    """Read share ``share`` of the spec's split into the gate's saved readings; what it did, as a dict."""
    from evaluation import golden_eval
    from evaluation.reading_cache import ReadingCache, default_root

    if client is None:
        from artifact_registry.blob_client import BlobClient

        client = BlobClient()
    if loader is None:
        from inference_core.model_runner import load_model

        def loader(adapter: str, label: str) -> Any:
            return load_model("base", client, backend="vllm", adapter=adapter, label=label)

    documents = spec["documents"]                       # the gate's own, as it loaded them
    wanted = set(spec["shares"][share])
    mine = [d for d in documents if d.source_id in wanted]
    if len(mine) != len(wanted):
        return {"ok": False, "error": f"{len(wanted) - len(mine)} document(s) of this share are not among the "
                                      "gate's documents"}
    settings = golden_eval.gate_settings()
    if tuple(settings) != tuple(spec["settings"]):
        return {"ok": False, "error": f"the gate's settings here {settings} are not the gate's {spec['settings']}"}
    # Before the model is loaded (minutes): the name this worker would keep its readings under. The model's tag is
    # the label it is loaded under (load_model), checked again once it is loaded.
    key = golden_eval.gate_reading_key(
        version=spec["version"], corpus_version=spec["corpus_version"], tenant_id=spec["tenant_id"],
        documents=documents, model_tag=spec["label"], adapter_checkpoint=spec["adapter_checkpoint"],
        calibrators=spec["calibrators"], thresholds=spec["thresholds"], settings=settings,
    )
    if key != spec["key"]:
        return {"ok": False, "error": f"the saved readings' name here ({key}) is not the gate's ({spec['key']}): "
                                      "the gate would not find what this worker read"}

    from inference_core.model_runner import release_model

    began = time.monotonic()
    model = loader(spec["adapter"], spec["label"])
    try:
        if getattr(model, "tag", None) != spec["label"]:
            return {"ok": False, "error": f"the model loaded as {getattr(model, 'tag', None)!r}, not "
                                          f"{spec['label']!r}: the gate would not find what this worker read"}
        repeat, repeat_mode, noisy_share, repeat_share = settings
        cache = ReadingCache(default_root(), key)
        triples = golden_eval.evaluate(
            mine, model, client, golden_eval.gate_images_root(spec["corpus_version"], spec["tenant_id"]),
            calibrators=spec["calibrators"], thresholds=spec["thresholds"], run_to_run=repeat,
            repeat_mode=repeat_mode, noisy_share=noisy_share, repeat_share=repeat_share, cache=cache,
        )
    finally:
        release_model(model)
    failures = list(getattr(triples, "infrastructure_failures", []))
    return {
        "ok": True, "documents": len(mine), "pages": sum(len(d.image_keys) for d in mine),
        "readings": len(triples), "failed": len(failures), "seconds": round(time.monotonic() - began, 1),
        "summary": (f"{len(mine)} document(s), {len(triples)} reading(s), {len(failures)} failed twice, "
                    f"{cache.summary()}, {round(time.monotonic() - began)} s"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read one GPU's share of the gate's frozen documents")
    parser.add_argument("--spec", type=Path, required=True, help="the gate's spec (gate-spec.pkl)")
    parser.add_argument("--share", type=int, required=True, help="which share of the split to read")
    parser.add_argument("--out", type=Path, required=True, help="where to write what it did (JSON)")
    args = parser.parse_args(argv)
    # Started by the gate with FIDEON_NO_DETACH=1, it reads in place; run by hand on the pod, it moves into tmux
    # like every long job.
    from orchestration.detach import detach_module_if_needed

    if detach_module_if_needed("evaluation.gate_shard_worker", argv):
        return 0
    from common.logging_setup import setup as setup_logging

    setup_logging(logging.INFO)
    with args.spec.open("rb") as handle:
        spec = pickle.load(handle)  # noqa: S301 - written by this run's own gate, on its own volume
    if not 0 <= args.share < len(spec["shares"]):
        parser.error(f"--share is 0 to {len(spec['shares']) - 1}")
    try:
        result = read_share(spec, args.share)
    except Exception as exc:  # noqa: BLE001 - reported to the gate, which reads what is left itself
        logging.getLogger(__name__).exception("the gate's share %d could not be read", args.share)
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    args.out.write_text(json.dumps(result), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
