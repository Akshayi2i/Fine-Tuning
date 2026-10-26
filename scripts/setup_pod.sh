#!/usr/bin/env bash
# Install this repo's dependencies on a RunPod pod, in the order that works.
#
#   bash scripts/setup_pod.sh ocr|train|serve|quantize|dev
#
# Run from the repo root. Versions are declared in pyproject.toml; the
# requirements-*.txt files choose the groups per pod.
set -euo pipefail

role="${1:-}"
case "$role" in
  ocr)   requirements="requirements-ocr.txt" ;;
  train) requirements="requirements-train.txt" ;;
  serve) requirements="requirements-serve.txt" ;;
  quantize) requirements="requirements-quantize.txt" ;;
  dev)   requirements="requirements.txt" ;;
  *) echo "usage: bash scripts/setup_pod.sh ocr|train|serve|quantize|dev" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."

# On the pod, run detached in tmux: an install cut off halfway by a closed laptop
# leaves a broken environment. pod_run.sh installs tmux itself if it is missing.
# The pod: RUNPOD_POD_ID (not always inherited by an SSH session), RunPod's
# environment file, or on Linux /workspace mounted as a volume with a GPU present.
on_pod=0
if [ -n "${RUNPOD_POD_ID:-}" ] || [ -f /etc/rp_environment ] \
   || { [ "$(uname -s)" = Linux ] && mountpoint -q /workspace 2>/dev/null \
        && { [ -e /dev/nvidia0 ] || command -v nvidia-smi >/dev/null 2>&1; }; }; then
  on_pod=1
fi
if [ "$on_pod" = 1 ] && [ -z "${TMUX:-}" ] && [ "${FIDEON_DETACHED:-}" != "1" ] \
   && [ "${FIDEON_NO_DETACH:-}" != "1" ]; then
  name="setup-${role}-$(date -u +%Y%m%d-%H%M%S)"
  bash scripts/pod_run.sh start "$name" -- bash scripts/setup_pod.sh "$role"
  echo "Setup is running in tmux as '$name'; closing the laptop will not stop it."
  echo "  status: bash scripts/pod_run.sh status $name"
  echo "  watch:  bash scripts/pod_run.sh attach $name   (Ctrl-b then d to leave it running)"
  exit 0
fi

# tmux, for scripts/pod_run.sh: a run inside it survives the SSH session ending.
if ! command -v tmux >/dev/null 2>&1 && [ "$(id -u)" = "0" ] && command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq tmux >/dev/null
fi
# MinerU's layout models import OpenCV, which needs libGL and glib from the OS;
# a slim image lacks them and OCR then fails at import.
if [ "$role" = "ocr" ] && [ "$(id -u)" = "0" ] && command -v apt-get >/dev/null 2>&1    && ! ldconfig -p 2>/dev/null | grep -q 'libGL.so.1'; then
  apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
fi
python -m pip install --upgrade pip wheel setuptools packaging ninja
# Installed through the role's lock (scripts/lock_requirements.sh): every package,
# transitive ones included, at the version resolved for the pod, so two pods set
# up months apart run the same stack. The laptop's dev role has no lock.
lock="${requirements%.txt}.lock"
# flash-attn, the one training package not in a lock (it is compiled against the
# torch the lock installs): pinned here, checked after the install below.
FLASH_ATTN_VERSION="2.8.3"
if [ -f "$lock" ]; then
  python -m pip install -r "$requirements" -c "$lock"
else
  python -m pip install -r "$requirements"
fi

if [ "$role" = "train" ]; then
  # Built against the torch just installed. With build isolation pip compiles it
  # against a throwaway torch in a temporary env, which fails or produces a
  # library that does not load. MAX_JOBS bounds the compile's memory use.
  # --no-cache-dir: flash-attn's setup downloads its prebuilt wheel into the
  # build directory (under TMPDIR) and renames it into pip's wheel cache. With
  # the cache on the network volume and TMPDIR on the container disk that
  # rename crosses filesystems and fails ("Invalid cross-device link"); without
  # a cache both sides are under TMPDIR.
  #
  # One exact version, not a range: a range took whatever flash-attn was newest
  # on the day the pod was set up, so two pods trained on two kernels. 2.8.3 is
  # built for the stack the locks pin - torch 2.8.0 on CUDA 12.8, Python 3.11 -
  # and its release carries the prebuilt wheel for it
  # (cu12torch2.8cxx11abiTRUE-cp311); with none to fetch it compiles from source
  # against that torch. vLLM brings its own flash-attention kernels (vllm-flash-attn).
  MAX_JOBS="${MAX_JOBS:-8}" python -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --no-cache-dir
fi

python - "$role" "$FLASH_ATTN_VERSION" <<'PY'
import importlib.metadata as md
import sys

role = sys.argv[1]
wanted = {
    "ocr": ["mineru", "torch", "azure-storage-blob"],
    "train": ["torch", "transformers", "ms-swift", "peft", "vllm", "flash-attn",
              "deepspeed", "qwen-vl-utils", "azure-storage-blob"],
    "serve": ["vllm", "transformers", "azure-storage-blob"],
    "quantize": ["llmcompressor", "torch", "azure-storage-blob"],
    "dev": ["pytest", "ruff", "azure-storage-blob"],
}[role]
missing = []
for name in wanted:
    try:
        print(f"  {name:22} {md.version(name)}")
    except md.PackageNotFoundError:
        missing.append(name)
if missing:
    sys.exit(f"missing after install: {missing}")
if role == "train" and md.version("flash-attn") != sys.argv[2]:
    sys.exit(f"flash-attn {md.version('flash-attn')} is installed, not the pinned {sys.argv[2]}")
if role in ("ocr", "train", "serve", "quantize"):
    import torch

    if not torch.cuda.is_available():
        sys.exit("torch is installed but sees no CUDA device - wrong torch build for this pod?")
    print(f"  CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
PY

if [ "$role" = "ocr" ]; then
  # MinerU reads its weights from the config (~/mineru.json, or MINERU_TOOLS_CONFIG_JSON),
  # and left alone it fetches the latest: OCR runs only on the pinned ones.
  if ! python -m data_pipeline.ocr.mineru_config; then
    echo "Next: download MinerU's pinned pipeline weights: bash scripts/download_mineru_models.sh"
    echo "(it writes the config; OCR refuses to run until then)."
  fi
fi
if [ "$role" = "train" ] || [ "$role" = "serve" ]; then
  echo "Next: put the base model under /workspace/models (or set FIDEON_BASE_MODEL_DIR),"
  echo "pin its revision in configs/base_model.yaml, and run: python scripts/phase0_spike.py"
  echo "Start long runs with: bash scripts/pod_run.sh start <name> <command...> (survives disconnects)"
fi
echo "setup_pod: $role ready"
