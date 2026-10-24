"""Make the promoted release's adapter ready for a vLLM we do not run code on.

    python -m scripts.served_adapter [--tenant T] [--out served-adapters] [--pod-dir /workspace/adapters]

Production serves the base model with vLLM on RunPod, and the release's
adapter as a LoRA on it (``lora_on_base``: nothing is merged). No code of ours
runs there, so the adapter - weights, data - is copied onto the pod's network
volume and vLLM loads it by name. This:

1. finds the tenant's promoted release (``serving.release_router.build_serving_plan``);
2. refuses one trained on another MinerU's text than the one serving reads
   with (its OCR pin against ``configs/shared/ocr.yaml``): the model would be
   served input it never saw;
3. downloads its adapter from Blob into ``<out>/policy-<version>/`` and checks it
   is one vLLM can serve on the base: LoRA on the language model only (vLLM's
   LoRA on the vision tower is experimental), its rank within vLLM's limits;
4. prints how to copy it to the volume and the vLLM flags that load it.

The extraction routine then asks for it by name (``--served-model
policy-<version>``), and checks at start that vLLM serves it on our base
(``OpenAIChatBackend._check_served``). Calibrators stay in Blob: the routine
reads them from there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: Ranks vLLM's ``--max-lora-rank`` accepts.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
#: Module names that put a LoRA on the vision tower or its merger.
VISION_MODULES = ("visual", "vision", "merger", "patch_embed")


class ServedAdapterError(RuntimeError):
    """The release cannot be served as a LoRA on the pod's vLLM."""


def lora_name(release_id: str) -> str:
    """The name vLLM serves the release's adapter as, and the routine asks for."""
    return f"policy-{release_id}"


def vllm_rank(rank: int) -> int:
    """The smallest ``--max-lora-rank`` vLLM accepts that holds ``rank``."""
    for allowed in VLLM_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise ServedAdapterError(f"LoRA rank {rank} is above vLLM's largest, {VLLM_LORA_RANKS[-1]}")


def check_adapter_config(config: dict[str, Any]) -> int:
    """The adapter's rank, once its config is seen to be one vLLM can serve."""
    modules = config.get("target_modules") or []
    if isinstance(modules, str):
        modules = [modules]
    vision = [m for m in modules if any(word in str(m).lower() for word in VISION_MODULES)]
    if vision:
        raise ServedAdapterError(
            f"the adapter puts LoRA on the vision side ({vision}): vLLM serves LoRA on the language model "
            "only (its tower/connector LoRA is experimental). Train without include_vision_projector."
        )
    if config.get("modules_to_save"):
        raise ServedAdapterError(f"the adapter saves whole modules ({config['modules_to_save']}), which vLLM's "
                                 "LoRA cannot load")
    rank = int(config.get("r") or 0)
    if rank <= 0:
        raise ServedAdapterError("adapter_config.json records no rank (r)")
    return rank


def assert_ocr_pin(release_pin: dict[str, Any], pin: dict[str, str]) -> None:
    """Refuse a release whose corpus another MinerU read than the one serving reads with."""
    trained = release_pin.get("mineru_version")
    if trained != pin["version"]:
        raise ServedAdapterError(
            f"the release was trained on text from MinerU {trained or '(not recorded)'}, but serving reads "
            f"pages with MinerU {pin['version']} (configs/shared/ocr.yaml): its text is formatted otherwise, "
            "input the model never saw. Re-OCR the corpus with the served MinerU and train a new version."
        )


def vllm_flags(name: str, rank: int, pod_path: str) -> str:
    """The flags, beside vLLM's usual ones, that serve the adapter as ``name``."""
    return f"--enable-lora --max-lora-rank {vllm_rank(rank)} --max-loras 1 --lora-modules {name}={pod_path}"


def prepare(client: Any, *, tenant: str | None, out: Path, pod_dir: str) -> dict[str, Any]:
    """Steps 1-3; returns what step 4 prints."""
    from common.config import served_mineru
    from serving.release_router import build_serving_plan

    plan = build_serving_plan(client, tenant_id=tenant)
    if not plan.served:
        raise ServedAdapterError("no release is promoted for this tenant: nothing to serve but the base model")
    release = plan.served[0]
    if release.served_as != "lora_on_base" or not release.adapter_dir:
        raise ServedAdapterError(
            f"release {release.release_id} is served as {release.served_as!r}, not as an adapter on the base: "
            "only a release packaged since 2026-10-23 (lora_on_base) can be loaded into the pod's vLLM"
        )
    assert_ocr_pin(release.ocr_pin, served_mineru())
    name = lora_name(release.release_id)
    local = out / name
    client.download_dir(release.adapter_dir, local)
    config_path = local / "adapter_config.json"
    if not config_path.is_file():
        raise ServedAdapterError(f"no adapter_config.json under {release.adapter_dir}: run `package` for it")
    rank = check_adapter_config(json.loads(config_path.read_text(encoding="utf-8")))
    pod_path = f"{pod_dir.rstrip('/')}/{name}"
    files = [p for p in local.rglob("*") if p.is_file()]
    return {
        "release": release.release_id, "name": name, "local": str(local), "pod_path": pod_path, "rank": rank,
        "files": len(files), "mb": round(sum(p.stat().st_size for p in files) / 2**20, 1),
        "calibrators": release.calibrators, "flags": vllm_flags(name, rank, pod_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--out", type=Path, default=Path("served-adapters"),
                        help="where the adapter is downloaded to (default ./served-adapters)")
    parser.add_argument("--pod-dir", default="/workspace/adapters",
                        help="the folder on the pod's network volume the adapters go in")
    args = parser.parse_args(argv)

    from artifact_registry.blob_client import BlobClient

    try:
        ready = prepare(BlobClient(), tenant=args.tenant, out=args.out, pod_dir=args.pod_dir)
    except ServedAdapterError as exc:
        print(f"cannot serve the release: {exc}", file=sys.stderr)
        return 1
    print(f"""adapter {ready['name']} (release {ready['release']}, LoRA rank {ready['rank']}): {ready['files']} files, \
{ready['mb']} MB in {ready['local']}

1. Copy it to the pod's network volume, as {ready['pod_path']} - for example:
     runpodctl send {ready['local']}          (on the pod: runpodctl receive <code>, then move it there)
   or with RunPod's S3 API for network volumes, where the volume's data centre has it:
     aws s3 cp --recursive {ready['local']} s3://<volume id>/{ready['pod_path'].removeprefix('/workspace/')} \\
         --endpoint-url https://s3api-<data centre>.runpod.io

2. Start vLLM with these besides its usual flags:
     {ready['flags']}

3. Run the extraction with --served-model {ready['name']}: it checks at start that vLLM serves it on the base.
   Its calibrators stay in Blob: {ready['calibrators']}""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
