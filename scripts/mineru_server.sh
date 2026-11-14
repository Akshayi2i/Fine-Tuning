#!/usr/bin/env bash
# Run the MinerU 4 servers the OCR stage and serving read through (configs/shared/ocr.yaml): one per GPU.
#
#   bash scripts/mineru_server.sh start             # the APIs in tmux, a token, .env
#   bash scripts/mineru_server.sh check [--pdf FILE]
#   bash scripts/mineru_server.sh stop
#
# It is installed by scripts/pod_bootstrap.sh (the environment and the models); this
# script only runs it. Why it exists: the corpus must be read by the pinned MinerU
# (version and tier in configs/shared/ocr.yaml), and nothing in the repo started one.
# The first run used a server on a separate pod; when that pod went, so did the only
# record of how it was started (decision 2026-10-28).
#
# On the training pod, OCR comes before training and they do not overlap: `stop` the
# server before `finetune`, which needs the whole GPU.
#
# One server reads about 1.7 pages a second whether it is sent one document or four at a time (measured
# 2026-11-14 on a 4x H200 pod), so each GPU runs a server of its own: server i on GPU i, port
# MINERU_PORT+i. .env gets MINERU_URL (the first) and MINERU_URLS (all of them); the OCR stage shares
# its documents between them.
#
#   MINERU_PORT       18001   the first server's port (8001 is the pod proxy's nginx, 8000 is vLLM)
#   MINERU_SERVERS    auto    servers to run: auto = one per GPU nvidia-smi lists (1 when it lists none)
#   MINERU_CONCURRENCY 4      parse jobs at once, per server
#   MINERU_STOP_PORTS         stop: only these ports (the pipeline's, to leave servers it did not start)
#   MINERU_VENV       /workspace/venv-mineru4
set -euo pipefail

action="${1:-}"
shift || true
cd "$(dirname "$0")/.."

PORT="${MINERU_PORT:-18001}"
VENV="${MINERU_VENV:-/workspace/venv-mineru4}"
STATE="/workspace/mineru-server"
# What the server must be, from the one pin both readers share.
pin() { python3 -c "
import re
text = open('configs/shared/ocr.yaml', encoding='utf-8').read()
print(re.search(r'^\s*$1:\s*[\"]?([\w.]+)', text, re.M).group(1))"; }
VERSION="$(pin version)"
TIER="$(pin tier)"
# How many servers: MINERU_SERVERS, else one per GPU nvidia-smi lists, else 1.
servers() {
  local n="${MINERU_SERVERS:-auto}"
  if [ "$n" = auto ]; then
    n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  fi
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || n=1
  echo "$n"
}
# Server 0 keeps the names a single server always had (session mineru-server, folder $STATE).
server_name() { if [ "$1" = 0 ]; then echo "mineru-server"; else echo "mineru-server-$1"; fi; }
server_dir() { if [ "$1" = 0 ]; then echo "$STATE"; else echo "$STATE/gpu$1"; fi; }
# Ports of the MinerU APIs running here, from MINERU_PORT up (start numbers ours from it).
running_ports() {
  { pgrep -af "mineru-api --port " 2>/dev/null || true; } \
    | sed -n 's/.*mineru-api --port \([0-9][0-9]*\).*/\1/p' | sort -un \
    | awk -v lo="$PORT" '$1 >= lo && $1 < lo + 64'
}

case "$action" in
  start)
    [ -x "$VENV/bin/mineru-api" ] || { echo "no MinerU in $VENV: run bash scripts/pod_bootstrap.sh first" >&2; exit 1; }
    # GPU only: the server does not start where it would read on the CPU. MinerU's own default (auto) falls back to the CPU
    # reader without a word when a torch extra is missing, and onnxruntime lists its CUDA provider even when it cannot load
    # (the CUDA 13 build on a CUDA 12 pod): so this loads a real model.
    if ! gpu_problem="$("$VENV/bin/python" - <<'PY' 2>&1
import glob
import os
import sys

problems = []
try:
    import torch
    if not torch.cuda.is_available():
        problems.append("torch cannot see the GPU")
except Exception as exc:
    problems.append(f"torch is not usable ({exc})")
for module in ("torchvision", "transformers", "accelerate", "safetensors"):
    try:
        __import__(module)
    except Exception:
        problems.append(f"{module} is missing (MinerU[torch]): MinerU would read on the CPU")
try:
    import onnxruntime
    models = glob.glob(os.path.expanduser("~/.mineru/models/MinerU-4_models_onnx/**/*.onnx"), recursive=True)
    if models:
        session = onnxruntime.InferenceSession(
            models[0], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if session.get_providers()[0] != "CUDAExecutionProvider":
            problems.append("onnxruntime cannot run on the GPU (wrong CUDA build?)")
    elif "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        problems.append("onnxruntime is the CPU build")
except Exception as exc:
    problems.append(f"onnxruntime is not usable ({exc})")
if problems:
    print("; ".join(problems))
    sys.exit(1)
PY
)"; then
      echo "GPU is not available for the OCR server: $gpu_problem" >&2
      echo "Refusing to start: OCR on the CPU is several times slower and not what the pipeline is meant to run. bash scripts/pod_bootstrap.sh installs the GPU packages." >&2
      exit 1
    fi
    mkdir -p "$STATE"
    # A token we make ourselves: the server is on this pod and only we call it. Kept on
    # the volume, never printed, and written to .env for the OCR stage to read.
    if [ ! -s "$STATE/token.txt" ]; then
      (umask 077; python3 -c "import secrets; print(secrets.token_hex(24))" > "$STATE/token.txt")
    fi
    count="$(servers)"
    for i in $(seq 0 $((count - 1))); do
      port=$((PORT + i))
      name="$(server_name "$i")"
      dir="$(server_dir "$i")"
      if pgrep -f "mineru-api --port $port" >/dev/null 2>&1; then
        echo "a MinerU API is already running on port $port"
        continue
      fi
      # A server stopped earlier leaves its finished tmux session open (pod_run.sh keeps a run's
      # last screen), and pod_run.sh refuses to reuse a name while it is: clear it first.
      tmux kill-session -t "=fideon-$name" 2>/dev/null || true
      rm -f "/workspace/logs/$name.exit"
      mkdir -p "$dir"
      # The key is read from the file when the server starts, not put on the command line: `ps` shows a command line.
      cat > "$dir/run.sh" <<EOF
#!/usr/bin/env bash
# From a folder of its own: the server writes a lock file and a blobs/ folder into where it runs, the repo
# must stay clean (a training run refuses a modified tree), and two servers must not share them.
cd "$dir"
export MINERU_DEVICE_MODE=cuda
export MINERU_MODEL_SMALL_BACKEND=torch
# GPU $i only, numbered as nvidia-smi numbers them.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$i
exec "$VENV/bin/mineru-api" --port $port --tier $TIER --api-key "\$(cat $STATE/token.txt)" --concurrency "${MINERU_CONCURRENCY:-4}"
EOF
      chmod 700 "$dir/run.sh"
      bash scripts/pod_run.sh start "$name" -- bash "$dir/run.sh"
    done
    # .env: the addresses and the token, where the OCR stage reads them. Set without printing.
    python3 - "$PORT" "$count" "$STATE/token.txt" <<'PY'
import re
import sys
from pathlib import Path

port, count, token_file = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
token = Path(token_file).read_text().strip()
urls = [f"http://localhost:{port + i}" for i in range(count)]
env = Path(".env")
text = env.read_text(encoding="utf-8") if env.exists() else ""
for key, value in (("MINERU_URL", urls[0]), ("MINERU_URLS", ",".join(urls)), ("MINERU_TOKEN", token)):
    line = f'{key}="{value}"'
    if re.search(rf"^{key}=", text, re.M):
        text = re.sub(rf"^{key}=.*$", lambda _m: line, text, flags=re.M)
    else:
        text = text.rstrip("\n") + "\n" + line + "\n"
with open(env, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)
print(f".env: MINERU_URL={urls[0]}, MINERU_URLS ({count} server(s)) and MINERU_TOKEN set")
PY
    echo "started $count server(s). Wait for them to load, then: bash scripts/mineru_server.sh check"
    echo "  logs: bash scripts/pod_run.sh tail mineru-server (and mineru-server-1, -2, ...)"
    ;;

  check)
    pdf=""
    if [ "${1:-}" = "--pdf" ]; then pdf="${2:-}"; fi
    # The same addresses and token the OCR stage uses: every server in MINERU_URLS, else the one MINERU_URL.
    set -a; [ -f .env ] && . ./.env; set +a
    urls="${MINERU_URLS:-${MINERU_URL:-}}"
    : "${urls:?MINERU_URL is not set (run start, or set it in .env)}"
    : "${MINERU_TOKEN:?MINERU_TOKEN is not set}"
    "${MINERU_PYTHON:-/workspace/venv/bin/python}" - "$urls" "$MINERU_TOKEN" "$VERSION" "$TIER" "$pdf" <<'PY'
import json
import sys
import urllib.request

urls, token, version, tier, pdf = sys.argv[1:]
headers = {"Authorization": f"Bearer {token}"}


def get(url, path):
    req = urllib.request.Request(url + path, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.status, response.read().decode("utf-8", "replace")


def check(url):
    failed = False
    print(url)
    for path in ("/v1/health", "/v1/tiers", "/openapi.json"):
        try:
            status, body = get(url, path)
            print(f"  [{'PASS' if status == 200 else 'FAIL'}] {path}: http {status}")
            failed |= status != 200
            if path == "/openapi.json":
                paths = json.loads(body).get("paths", {})
                ok = "/v1/tasks" in paths
                print(f"  [{'PASS' if ok else 'FAIL'}] the API has /v1/tasks (what the OCR client calls)")
                failed |= not ok
            elif path == "/v1/tiers":
                ok = tier in body
                print(f"  [{'PASS' if ok else 'FAIL'}] tier '{tier}' is offered")
                failed |= not ok
        except Exception as exc:
            print(f"  [FAIL] {path}: {exc}")
            failed = True
    if pdf:
        # A real read, through the client the OCR stage uses: the version and tier the server reports.
        from data_pipeline.ocr.mineru_api import ocr_via_api

        pages, report = ocr_via_api(open(pdf, "rb").read(), url, token, filename=pdf.rsplit("/", 1)[-1], tier=tier)
        ok = report["version"] == version and report["tier"] == tier and all(text is not None for text in pages)
        chars = sum(len(text) for text in pages)
        print(f"  [{'PASS' if ok else 'FAIL'}] read {pdf}: {len(pages)} page(s), {chars} characters, "
              f"{report.get('server_seconds')} s on the server; version {report['version']}, tier {report['tier']} "
              f"(pin: {version}, {tier})")
        failed |= not ok
    return failed


servers = [url.strip().rstrip("/") for url in urls.split(",") if url.strip()]
failures = [url for url in servers if check(url)]
print(f"{len(servers) - len(failures)} of {len(servers)} server(s) pass")
sys.exit(1 if failures else 0)
PY
    ;;

  stop)
    # Every server start ran (MinerU APIs from MINERU_PORT up), not only the first - or, with MINERU_STOP_PORTS
    # (the pipeline's, when some servers were already up before it started the rest), only those ports.
    if [ -n "${MINERU_STOP_PORTS:-}" ]; then ports="$MINERU_STOP_PORTS"; else ports="$(running_ports)"; fi
    alive() { for p in $ports; do pgrep -f "mineru-api --port $p" >/dev/null 2>&1 && echo "$p"; done; true; }
    to_stop="$(echo $(alive))"
    if [ -z "$to_stop" ]; then
      echo "no MinerU API to stop (port(s) ${ports:-from $PORT})"
    else
      echo "stopping the server(s) on port(s) $to_stop"
      for port in $to_stop; do pkill -f "mineru-api --port $port" 2>/dev/null || true; done
    fi
    # The API holds the GPU models and worker processes and takes more than a moment to exit; training
    # follows straight after, so wait for them (up to 30 s), then force them, rather than report too early.
    for _ in $(seq 1 30); do [ -z "$(alive)" ] && break; sleep 1; done
    if [ -n "$(alive)" ]; then
      for port in $(alive); do pkill -9 -f "mineru-api --port $port" 2>/dev/null || true; done
      sleep 3
    fi
    # Closes the finished sessions too, so the next start can reuse the names: every server's, or those ports'.
    if [ -n "${MINERU_STOP_PORTS:-}" ]; then
      for port in $ports; do tmux kill-session -t "=fideon-$(server_name $((port - PORT)))" 2>/dev/null || true; done
    else
      for session in $(tmux ls -F '#S' 2>/dev/null | grep -E '^fideon-mineru-server(-[0-9]+)?$' || true); do
        tmux kill-session -t "=$session" 2>/dev/null || true
      done
    fi
    # The GPU memory is free once the processes are gone; say so when they are not.
    [ -n "$(alive)" ] && { echo "a server is still running: port(s) $(echo $(alive))" >&2; exit 1; } || true
    echo "stopped"
    exit 0
    ;;

  *)
    echo "usage: bash scripts/mineru_server.sh start|check [--pdf FILE]|stop" >&2
    exit 2
    ;;
esac
