#!/usr/bin/env bash
# Download MinerU 3.x's pipeline model weights at a pinned revision, on the pod's
# volume, and point MinerU's config at them.
#
#   bash scripts/download_mineru_models.sh          (pod_bootstrap.sh runs it)
#
# The same files MinerU's own `mineru-models-download -m pipeline` fetches
# (mineru.cli.models_download.download_pipeline_models), but at a fixed revision
# of opendatalab/PDF-Extract-Kit-1.0: its main branch moves on with MinerU, and
# on 2025-10-24 it deleted files MinerU 1.x loaded - a pod set up after such a
# change downloads fine and fails at the first page, or reads with other weights.
# MinerU's downloader also fetches a config template from its master branch; this
# script writes the config itself (data_pipeline.ocr.mineru_config --write).
# The weights go to HF_HOME (on the volume), the config to MINERU_TOOLS_CONFIG_JSON
# (on the volume). MINERU_MODEL_SOURCE=local, set in .env by pod_bootstrap.sh, is
# what makes MinerU read them from there (the config carries no model-source key):
# nothing is fetched at OCR time.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKSPACE="${FIDEON_WORKSPACE:-/workspace}"
VENV_OCR="$WORKSPACE/venv-ocr"
[ -x "$VENV_OCR/bin/python" ] || { echo "download_mineru_models: no OCR environment at $VENV_OCR" >&2; exit 1; }
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export HF_HOME="${HF_HOME:-$WORKSPACE/.cache/huggingface}"
# RunPod images turn on the hf_transfer downloader for every process; it is not
# installed in these environments, and Hugging Face then refuses to download.
export HF_HUB_ENABLE_HF_TRANSFER=0
# A pod set up for MinerU 1.x points this at magic-pdf.json; 3.x reads mineru.json.
case "${MINERU_TOOLS_CONFIG_JSON:-}" in
  ""|*magic-pdf.json) MINERU_TOOLS_CONFIG_JSON="$WORKSPACE/mineru.json" ;;
esac
export MINERU_TOOLS_CONFIG_JSON

version="$("$VENV_OCR/bin/python" -c 'import importlib.metadata as m; print(m.version("mineru"))')"
# The model repository's revision current for MinerU 3.4: its last commit
# (2026-06-15, the PP-OCRv6 models 3.4 reads), with no change since.
MODELS_REVISION="${MINERU_MODELS_REVISION:-ed6b654c018d742e65a17671e379c5e6ecc87ec9}"
echo "MinerU $version: pipeline weights (PDF-Extract-Kit-1.0 at $MODELS_REVISION) into $HF_HOME"
weights="$("$VENV_OCR/bin/python" - "$MODELS_REVISION" <<'PY'
import sys

from huggingface_hub import snapshot_download

# mineru.utils.enum_class.ModelPath: what the pipeline backend loads.
paths = ["models/Layout/PP-DocLayoutV2", "models/MFR/unimernet_hf_small_2503", "models/OCR/paddleocr_torch",
         "models/TabRec/SlanetPlus/slanet-plus.onnx", "models/TabRec/UnetStructure/unet.onnx",
         "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx", "models/MFR/pp_formulanet_plus_m"]
patterns = [p for path in paths for p in (path, path + "/*")]
print(snapshot_download("opendatalab/PDF-Extract-Kit-1.0", revision=sys.argv[1], allow_patterns=patterns))
PY
)"
weights="$(printf '%s\n' "$weights" | tail -n1)"
[ -d "$weights/models/OCR/paddleocr_torch" ] \
  || { echo "download_mineru_models: no OCR models under $weights" >&2; exit 1; }
"$VENV_OCR/bin/python" -m data_pipeline.ocr.mineru_config --write "$weights"
echo "download_mineru_models: done ($MINERU_TOOLS_CONFIG_JSON)"
