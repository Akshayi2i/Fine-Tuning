#!/usr/bin/env bash
# One command to make a fresh H200 SXM pod ready to train.
#
#   bash scripts/pod_bootstrap.sh
#
# Run it from the repo, cloned onto the network volume (/workspace/fine-tuning).
# It is safe to re-run, and the one script that sets a pod up: every step checks what is already there and does only
# what is missing, so after a pod stop it finishes in minutes.
#
# What it does, in order:
#   1. checks: a GPU (an H200 expected), /workspace mounted, >= 150 GB free
#   2. .env: fills the pod's keys without touching any value already set
#   3. /workspace/venv      training + serving (torch 2.8, ms-swift, vLLM 0.11, flash-attn)
#   4. /workspace/venv-mineru4  the OCR server: the MinerU version and tier pinned in
#                           configs/shared/ocr.yaml (4.0.12, basic), with the torch the
#                           training lock pins, so it reads pages on the GPU. Its own
#                           environment: MinerU's dependencies are kept apart from vLLM.
#   5. that server's models for the pinned tier
#   6. the base model at the pinned revision into /workspace/models
#   7. the Phase 0 spike
#
# After it: `bash scripts/mineru_server.sh start`, read the corpus (OCR), then
# `bash scripts/mineru_server.sh stop` before training: both want the GPU.
#
# Everything lives on /workspace because the container disk is wiped when the
# pod stops. On the pod it runs detached in tmux: closing the laptop does not
# stop it. Watch it with: bash scripts/pod_run.sh attach <name it prints>
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
WORKSPACE="${FIDEON_WORKSPACE:-/workspace}"
VENV="$WORKSPACE/venv"
VENV_MINERU="${MINERU_VENV:-$WORKSPACE/venv-mineru4}"
MINERU_BACKEND="${MINERU_BACKEND:-torch}"   # torch (GPU) or onnx (CPU only: slow)
MINERU_CONFIG="$WORKSPACE/mineru.json"
MODELS="$WORKSPACE/models"
MIN_FREE_GB="${FIDEON_MIN_FREE_GB:-150}"

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "usage: bash scripts/pod_bootstrap.sh" >&2; exit 2 ;;
  esac
done

# --- detach: an install or download cut off by a closed laptop is half done ----
on_pod=0
if [ -n "${RUNPOD_POD_ID:-}" ] || [ -f /etc/rp_environment ] \
   || { [ "$(uname -s)" = Linux ] && mountpoint -q "$WORKSPACE" 2>/dev/null \
        && { [ -e /dev/nvidia0 ] || command -v nvidia-smi >/dev/null 2>&1; }; }; then
  on_pod=1
fi
if [ "$on_pod" = 1 ] && [ -z "${TMUX:-}" ] && [ "${FIDEON_DETACHED:-}" != "1" ] \
   && [ "${FIDEON_NO_DETACH:-}" != "1" ]; then
  name="bootstrap-$(date -u +%Y%m%d-%H%M%S)"
  bash scripts/pod_run.sh start "$name" -- bash scripts/pod_bootstrap.sh
  echo "Bootstrap is running in tmux as '$name'; closing the laptop will not stop it."
  echo "  status: bash scripts/pod_run.sh status $name"
  echo "  watch:  bash scripts/pod_run.sh attach $name   (Ctrl-b then d to leave it running)"
  exit 0
fi

step() { echo; echo "=== $* ==="; }
die() { echo "pod_bootstrap: $*" >&2; exit 1; }
PENDING=()

# --- 1. the pod ----------------------------------------------------------------
step "1/7 checking the pod"
[ "$(uname -s)" = Linux ] || die "run this on the RunPod pod, not the laptop"
command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi: this pod has no GPU (every stage is GPU only)"
# Every GPU's line read in full, then the first taken. `| head -n1` closed the pipe
# while nvidia-smi was still writing on a multi-GPU pod, and under pipefail its
# SIGPIPE ended the script silently at this line (4x H200, 2026-10-07).
gpus="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
gpu="${gpus%%$'\n'*}"
echo "GPU: $gpu x$(printf '%s\n' "$gpus" | grep -c .)"
case "$gpu" in
  *H200*) ;;
  *) echo "WARNING: expected an H200 SXM; the configs are sized for its 141 GB." ;;
esac
# torch 2.8 is built for CUDA 12.8; an older driver makes torch see no GPU at all.
# nvidia-smi prints the highest CUDA version the driver supports.
driver_cuda="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || true)"
echo "driver supports CUDA ${driver_cuda:-unknown}"
if [ -n "$driver_cuda" ] && ! awk -v v="$driver_cuda" 'BEGIN { split(v, a, "."); exit !(a[1] > 12 || (a[1] == 12 && a[2] >= 8)) }'; then
  die "the GPU driver supports CUDA $driver_cuda, but torch 2.8 needs 12.8 or newer. Redeploy the pod with \
RunPod's CUDA filter set to 12.8+."
fi
mountpoint -q "$WORKSPACE" || die "$WORKSPACE is not a mounted volume. Attach the network volume at $WORKSPACE: \
the container disk is wiped when the pod stops, and the environments and model must survive that."
case "$REPO/" in
  "$WORKSPACE"/*) ;;
  *) die "the repo is at $REPO, on the container disk, which is wiped on stop. Clone it under $WORKSPACE." ;;
esac
# The repo needs Python >= 3.11 (pyproject). On Ubuntu 22.04 `python3` can be the
# system 3.10 even when the image's `python` is 3.11, so choose explicitly.
PYTHON="${FIDEON_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3.11 python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
       && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
      PYTHON="$(command -v "$candidate")"; break
    fi
  done
fi
[ -n "$PYTHON" ] || die "no Python >= 3.11 on this pod (the repo requires it); set FIDEON_PYTHON"
echo "Python: $PYTHON ($("$PYTHON" -c 'import platform; print(platform.python_version())'))"
free_gb="$(df -BG --output=avail "$WORKSPACE" | tail -n1 | tr -dc '0-9')"
echo "free on $WORKSPACE: ${free_gb} GB"
[ "$free_gb" -ge "$MIN_FREE_GB" ] || die "only ${free_gb} GB free on $WORKSPACE; need ${MIN_FREE_GB} GB \
(two environments ~25 GB, the MinerU models, base model 17.5 GB, checkpoints, the staged corpus). Grow the network volume."

# --- 2. .env ---------------------------------------------------------------------
step "2/7 .env"
[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example"; }

# Set KEY to VALUE only when it is unset, empty, or still the template's default:
# a value the operator chose is never overwritten.
ensure_env() {
  local key="$1" value="$2" template_default="${3:-}" current
  if grep -qE "^${key}=" .env; then
    current="$(grep -E "^${key}=" .env | tail -n1 | cut -d= -f2-)"
    if [ -n "$current" ] && [ "$current" != "$template_default" ]; then
      [ "$current" = "$value" ] || echo "  keeping $key=$current (expected $value on this pod)"
      return
    fi
    sed -i "s|^${key}=.*|${key}=${value}|" .env
  else
    printf '%s=%s\n' "$key" "$value" >> .env
  fi
  echo "  set $key=$value"
}
ensure_env RUNPOD_VOLUME_MOUNT "$WORKSPACE" /runpod-volume
ensure_env HF_HOME "$WORKSPACE/.cache/huggingface"
# A pod set up for MinerU 1.x has magic-pdf.json here: replaced, as a template value is.
ensure_env MINERU_TOOLS_CONFIG_JSON "$MINERU_CONFIG" "$WORKSPACE/magic-pdf.json"
# MinerU 3.x on the GPU, on the pinned weights it was pointed at - never the latest.
ensure_env MINERU_DEVICE_MODE cuda
# MinerU's own default, auto, quietly picks the CPU reader (onnx) when any torch extra is missing: forced, a missing one is an error.
ensure_env MINERU_MODEL_SMALL_BACKEND torch
ensure_env MINERU_MODEL_SOURCE local
ensure_env HF_HUB_ENABLE_HF_TRANSFER 0 1
# The value .env gives KEY, read as bash reads the line when sourcing it: blanks
# before the name and an `export ` are allowed there, so they are here too. Matched
# only from the first column, `export AZURE_STORAGE_SAS_TOKEN=sv=1&sig=X` passed
# the quoting check below and was then cut at its '&' when sourced. Neither a CR
# (a .env saved on Windows) nor a ` # comment` after the value is part of it:
# `AZURE_RAW_SAS_TOKEN=   # empty: one SAS serves raw & main` sets an empty token.
env_value() {
  grep -E "^[[:space:]]*(export[[:space:]]+)?$1=" .env | tail -n1 \
    | sed -E -e "s/^[[:space:]]*(export[[:space:]]+)?$1=//" -e 's/\r$//' -e 's/[[:space:]]+#.*$//' \
    || true
}
# A value with its escaped characters (`\\`, `\&`, `\;`) set aside: bash reads each
# as itself, never as a separator, so only what is left can cut the value.
unescaped() {
  local bs='\' value
  value="${1//"$bs$bs"/}"
  value="${value//"$bs&"/}"
  printf '%s' "${value//"$bs;"/}"
}
# .env is sourced by bash (here and by pod_run.sh): an unquoted connection string
# is cut at its first ';' and every Blob call would fail with a confusing error.
case "$(unescaped "$(env_value AZURE_STORAGE_CONNECTION_STRING)")" in
  \"*|\'*) ;;
  *\;*) die "AZURE_STORAGE_CONNECTION_STRING in .env contains ';' but is not quoted. Wrap the \
value in double quotes: bash would otherwise cut it at the first ';'." ;;
esac
# A SAS token is cut the same way at each '&', and the rest run in the background.
for sas_key in AZURE_STORAGE_SAS_TOKEN AZURE_RAW_SAS_TOKEN; do
  case "$(unescaped "$(env_value "$sas_key")")" in
    \"*|\'*) ;;
    *\&*) die "$sas_key in .env contains '&' but is not quoted. Wrap the value in double \
quotes: bash would otherwise run each part after an '&' in the background." ;;
  esac
done
set -a; . ./.env; set +a
export HF_HOME MINERU_TOOLS_CONFIG_JSON MINERU_DEVICE_MODE MINERU_MODEL_SOURCE RUNPOD_VOLUME_MOUNT
# RunPod images turn on the hf_transfer downloader for every process; it is not
# installed in our environments, and Hugging Face then refuses to download.
export HF_HUB_ENABLE_HF_TRANSFER=0
# A SAS login (account URL + token) or the connection string; either one will do.
# The same rule as Python's (artifact_registry.blob_client.azure_from_environment):
# any part of a SAS login set means a SAS login, and half of one is refused there
# rather than falling back to the connection string - so it is not done here either.
# Python strips each value first, so a value of blanks - or the CR a .env saved on
# Windows leaves in `AZURE_RAW_SAS_TOKEN=` when Linux bash sources it - is empty
# there, and here too.
trimmed() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  printf '%s' "${value%"${value##*[![:space:]]}"}"
}
sas_url="$(trimmed "${AZURE_STORAGE_ACCOUNT_URL:-}")" sas_main="$(trimmed "${AZURE_STORAGE_SAS_TOKEN:-}")"
sas_raw="$(trimmed "${AZURE_RAW_SAS_TOKEN:-}")"
if [ -n "$sas_url$sas_main$sas_raw" ]; then
  if [ -z "$sas_url" ] || [ -z "$sas_main" ]; then
    PENDING+=("set both AZURE_STORAGE_ACCOUNT_URL and AZURE_STORAGE_SAS_TOKEN in $REPO/.env, or empty every SAS setting to use the connection string: half a SAS login is refused on every Blob call, not backed by the connection string")
  fi
  main_container="${AZURE_BLOB_CONTAINER:-insurance-extraction}"
  if [ -n "$sas_raw" ] && [ "${AZURE_RAW_CONTAINER:-$main_container}" = "$main_container" ]; then
    PENDING+=("AZURE_RAW_SAS_TOKEN is set in $REPO/.env but AZURE_RAW_CONTAINER names no separate raw container: set it, or empty the token (every Blob call is refused until then)")
  fi
else
  case "${AZURE_STORAGE_CONNECTION_STRING:-}" in
    ""|*AccountName=...*)
      PENDING+=("set AZURE_STORAGE_ACCOUNT_URL + AZURE_STORAGE_SAS_TOKEN, or AZURE_STORAGE_CONNECTION_STRING, in $REPO/.env (every stage reads and writes Blob)") ;;
  esac
fi

# --- 3 and 4. the two environments -------------------------------------------------
# A marker records the requirements an environment was built from; an unchanged
# environment is not reinstalled on a re-run.
make_venv() {
  local venv="$1" role="$2" requirements="$3" stamp marker
  stamp="$(cat "$requirements" "${requirements%.txt}.lock" pyproject.toml 2>/dev/null | sha256sum | cut -c1-16)"
  marker="$venv/.fideon-$role"
  if [ -f "$marker" ] && [ "$(cat "$marker")" = "$stamp" ] && [ -x "$venv/bin/python" ]; then
    echo "$venv is up to date ($role)"
    return
  fi
  if [ ! -x "$venv/bin/python" ]; then
    "$PYTHON" -m venv "$venv" 2>/dev/null || {
      apt-get update -qq && apt-get install -y -qq "python3.$("$PYTHON" -c 'import sys; print(sys.version_info[1])')-venv" >/dev/null
      "$PYTHON" -m venv "$venv"
    }
  fi
  PATH="$venv/bin:$PATH" VIRTUAL_ENV="$venv" FIDEON_DETACHED=1 bash scripts/setup_pod.sh "$role"
  echo "$stamp" > "$marker"
}
step "3/7 training + serving environment: $VENV"
make_venv "$VENV" train requirements-train.txt
# --- 4. the OCR server: MinerU 4, as pinned -------------------------------------------------
# One MinerU reads the training corpus and every served document (configs/shared/ocr.yaml); it
# runs as a server (scripts/mineru_server.sh). The pin is read from that file, not copied here.
pin() { "$PYTHON" -c "
import re
text = open('configs/shared/ocr.yaml', encoding='utf-8').read()
print(re.search(r'^\s*$1:\s*[\"]?([\w.]+)', text, re.M).group(1))"; }
MINERU_VERSION="$(pin version)"
MINERU_TIER="$(pin tier)"
TORCH_PIN="$(grep -m1 '^torch==' requirements-train.lock | cut -d' ' -f1)"
[ -n "$TORCH_PIN" ] || die "requirements-train.lock pins no torch"
step "4/7 OCR server: MinerU $MINERU_VERSION ($MINERU_BACKEND backend) in $VENV_MINERU"
# The CUDA 12 build of onnxruntime: 1.31 is built for CUDA 13 and fails to load on the pod's 12.8 (the OCR then runs on the CPU).
ORT_GPU_PIN="${ORT_GPU_PIN:-1.22.0}"
mineru_want="$MINERU_VERSION:$MINERU_BACKEND:$TORCH_PIN:ort-gpu-$ORT_GPU_PIN"
if [ -f "$VENV_MINERU/.fideon-mineru4" ] && [ "$(cat "$VENV_MINERU/.fideon-mineru4")" = "$mineru_want" ] \
   && [ -x "$VENV_MINERU/bin/mineru-api" ]; then
  echo "$VENV_MINERU is up to date (mineru $MINERU_VERSION, $MINERU_BACKEND)"
else
  # MinerU's layout models import OpenCV, which needs libGL and glib from the OS.
  if [ "$(id -u)" = "0" ] && command -v apt-get >/dev/null 2>&1 && ! ldconfig -p 2>/dev/null | grep -q 'libGL.so.1'; then
    apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
  fi
  [ -x "$VENV_MINERU/bin/python" ] || "$PYTHON" -m venv "$VENV_MINERU"
  "$VENV_MINERU/bin/pip" install --upgrade pip
  if [ "$MINERU_BACKEND" = "torch" ]; then
    # The [torch] extra (torchvision, transformers, accelerate, safetensors): without it MinerU reads on the CPU. With the torch
    # the training environment pins, so pip does not pick another; read from a wheel cache in PIP_FIND_LINKS when one is there.
    "$VENV_MINERU/bin/pip" install "mineru[torch]==$MINERU_VERSION" "$TORCH_PIN"
    # MinerU pulls the CPU onnxruntime, and the two share one folder: remove both, add the GPU build alone.
    "$VENV_MINERU/bin/pip" uninstall -y onnxruntime onnxruntime-gpu
    "$VENV_MINERU/bin/pip" install "onnxruntime-gpu==$ORT_GPU_PIN"
  else
    "$VENV_MINERU/bin/pip" install "mineru==$MINERU_VERSION"
  fi
  if [ "$MINERU_BACKEND" = "torch" ]; then
    "$VENV_MINERU/bin/python" -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
  fi
  echo "$mineru_want" > "$VENV_MINERU/.fideon-mineru4"
fi

# --- 5. the server's models ----------------------------------------------------------------
step "5/7 MinerU models ($MINERU_TIER tier, $MINERU_BACKEND backend)"
mkdir -p "$WORKSPACE/mineru-server"
models_marker="$WORKSPACE/mineru-server/models-$MINERU_VERSION-$MINERU_TIER-$MINERU_BACKEND+onnx"
mineru_ready=0
# The models live in ~/.mineru (the pod's own disk, not the volume): a marker alone, left by an earlier pod, proves nothing.
if [ -f "$models_marker" ] && [ -d "$HOME/.mineru/models/MinerU-4_models_onnx" ] && [ -d "$HOME/.mineru/models/MinerU-4_models_torch" ]; then
  echo "models present"
  mineru_ready=1
# The OCR step of a parse reads the ONNX repo whichever backend runs the layout models (found on the
# first pod: "MinerU-4_models_onnx is not ready"), so both are fetched.
elif "$VENV_MINERU/bin/mineru-models-download" --tier "$MINERU_TIER" --small-backend "$MINERU_BACKEND" -v      && "$VENV_MINERU/bin/mineru-models-download" MinerU-4_models_onnx -v; then
  touch "$models_marker"
  mineru_ready=1
else
  PENDING+=("MinerU's models: $VENV_MINERU/bin/mineru-models-download --tier $MINERU_TIER --small-backend $MINERU_BACKEND, then re-run this script")
  echo "MinerU's models are not ready yet (see the list at the end)"
fi

# --- 6. the base model ---------------------------------------------------------------
step "6/7 base model"
"$VENV/bin/python" - "$MODELS" <<'PY'
import sys
from pathlib import Path

from common.config import base_model_config

model = base_model_config()["model"]
model_id, revision = str(model["model_id"]), str(model["revision"])
if revision in ("", "PIN_ME"):
    sys.exit("configs/base_model.yaml has no pinned revision")
target = Path(sys.argv[1]) / model_id.rsplit("/", 1)[-1]
marker = target / ".fideon-revision"
if marker.is_file() and marker.read_text().strip() == revision and (target / "config.json").is_file():
    print(f"{model_id}@{revision[:12]} already in {target}")
    sys.exit(0)

# A copy put there some other way (a serving setup, a manual download) is used
# when every file is the pinned revision's, by size and, for the weights, by
# SHA-256 - not re-downloaded; a copy that differs is replaced below.
if (target / "config.json").is_file():
    import hashlib

    from huggingface_hub import HfApi

    print(f"verifying the copy in {target} against {revision[:12]} (hashes ~17.5 GB, a few minutes)")
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    problems = []
    for sibling in info.siblings:
        path = target / sibling.rfilename
        if not path.is_file():
            problems.append(f"missing {sibling.rfilename}")
            continue
        if sibling.size is not None and path.stat().st_size != sibling.size:
            problems.append(f"size of {sibling.rfilename}")
            continue
        lfs = sibling.lfs
        want = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if want:
            digest = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    digest.update(chunk)
            if digest.hexdigest() != want:
                problems.append(f"content of {sibling.rfilename}")
    if not problems:
        marker.write_text(revision + "\n")
        print(f"the existing copy is {model_id}@{revision[:12]}: using it, no download")
        sys.exit(0)
    print(f"the existing copy differs from the pinned revision ({problems[:4]}); downloading it")

from huggingface_hub import snapshot_download

print(f"downloading {model_id}@{revision} to {target} (~17.5 GB; resumes if interrupted)")
snapshot_download(repo_id=model_id, revision=revision, local_dir=str(target))
shards = sorted(target.glob("*.safetensors"))
if not (target / "config.json").is_file() or not shards:
    sys.exit(f"download incomplete: {target} has no config.json or no weight shards")
marker.write_text(revision + "\n")
print(f"ok: {len(shards)} shards, {sum(p.stat().st_size for p in shards) / 1e9:.1f} GB")
PY

# --- 7. the spike -----------------------------------------------------------------------
step "7/7 Phase 0 spike"
mkdir -p "$WORKSPACE/logs"
spike_status=0
# Each environment's bin on PATH: the spike looks for `swift` by name.
PATH="$VENV/bin:$PATH" VIRTUAL_ENV="$VENV" \
  "$VENV/bin/python" scripts/phase0_spike.py --skip-mineru --out "$WORKSPACE/logs/spike_report.json" \
  || spike_status=$?
# The MinerU checks are the server's own: scripts/mineru_server.sh check [--pdf FILE].

echo
echo "=== bootstrap summary ==="
echo "training + serving: $VENV     (source $VENV/bin/activate)"
echo "OCR server:         $VENV_MINERU  (bash scripts/mineru_server.sh start | check | stop)"
echo "base model:         $MODELS"
echo "spike report:       $WORKSPACE/logs/spike_report.json (exit $spike_status)"
if [ ${#PENDING[@]} -gt 0 ]; then
  echo "Still to do:"
  for item in "${PENDING[@]}"; do echo "  - $item"; done
fi
[ "$spike_status" = 0 ] || die "the spike reported a failed check; read $WORKSPACE/logs/spike_report.json"
if [ "$mineru_ready" = 1 ]; then
  echo "Next: bash scripts/mineru_server.sh start && bash scripts/mineru_server.sh check"
fi
echo "pod_bootstrap: done"
