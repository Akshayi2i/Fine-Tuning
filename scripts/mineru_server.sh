#!/usr/bin/env bash
# Run the MinerU 4 server the OCR stage and serving read through (configs/shared/ocr.yaml).
#
#   bash scripts/mineru_server.sh start             # the API in tmux, a token, .env
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
#   MINERU_PORT       18001   the port the API listens on (8001 is the pod proxy's nginx, 8000 is vLLM)
#   MINERU_CONCURRENCY 4      parse jobs at once
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

case "$action" in
  start)
    [ -x "$VENV/bin/mineru-api" ] || { echo "no MinerU in $VENV: run bash scripts/pod_bootstrap.sh first" >&2; exit 1; }
    mkdir -p "$STATE"
    # A token we make ourselves: the server is on this pod and only we call it. Kept on
    # the volume, never printed, and written to .env for the OCR stage to read.
    if [ ! -s "$STATE/token.txt" ]; then
      (umask 077; python3 -c "import secrets; print(secrets.token_hex(24))" > "$STATE/token.txt")
    fi
    if pgrep -f "mineru-api --port $PORT" >/dev/null 2>&1; then
      echo "a MinerU API is already running on port $PORT"
    else
      # A server stopped earlier leaves its finished tmux session open (pod_run.sh keeps a run's
      # last screen), and pod_run.sh refuses to reuse a name while it is: clear it first.
      tmux kill-session -t "=fideon-mineru-server" 2>/dev/null || true
      rm -f /workspace/logs/mineru-server.exit
      # The key is read from the file when the server starts, not put on the command line: `ps` shows a command line.
      cat > "$STATE/run.sh" <<EOF
#!/usr/bin/env bash
# From the state folder: the server writes a lock file and a blobs/ folder into where it runs, and the repo
# must stay clean (a training run refuses a modified tree).
cd "$STATE"
exec "$VENV/bin/mineru-api" --port $PORT --tier $TIER --api-key "\$(cat $STATE/token.txt)" --concurrency "${MINERU_CONCURRENCY:-4}"
EOF
      chmod 700 "$STATE/run.sh"
      bash scripts/pod_run.sh start mineru-server -- bash "$STATE/run.sh"
    fi
    # .env: the address and the token, where the OCR stage reads them. Set without printing.
    python3 - "$PORT" "$STATE/token.txt" <<'PY'
import re
import sys
from pathlib import Path

port, token_file = sys.argv[1], sys.argv[2]
token = Path(token_file).read_text().strip()
env = Path(".env")
text = env.read_text(encoding="utf-8") if env.exists() else ""
for key, value in (("MINERU_URL", f"http://localhost:{port}"), ("MINERU_TOKEN", token)):
    line = f'{key}="{value}"'
    if re.search(rf"^{key}=", text, re.M):
        text = re.sub(rf"^{key}=.*$", lambda _m: line, text, flags=re.M)
    else:
        text = text.rstrip("\n") + "\n" + line + "\n"
with open(env, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)
print(f".env: MINERU_URL=http://localhost:{port} and MINERU_TOKEN set")
PY
    echo "started. Wait for it to load, then: bash scripts/mineru_server.sh check"
    echo "  log: bash scripts/pod_run.sh tail mineru-server"
    ;;

  check)
    pdf=""
    if [ "${1:-}" = "--pdf" ]; then pdf="${2:-}"; fi
    # The same address and token the OCR stage uses.
    set -a; [ -f .env ] && . ./.env; set +a
    : "${MINERU_URL:?MINERU_URL is not set (run start, or set it in .env)}"
    : "${MINERU_TOKEN:?MINERU_TOKEN is not set}"
    "${MINERU_PYTHON:-/workspace/venv/bin/python}" - "$MINERU_URL" "$MINERU_TOKEN" "$VERSION" "$TIER" "$pdf" <<'PY'
import json
import sys
import urllib.request

url, token, version, tier, pdf = (sys.argv[1].rstrip("/"), *sys.argv[2:])
headers = {"Authorization": f"Bearer {token}"}


def get(path):
    req = urllib.request.Request(url + path, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.status, response.read().decode("utf-8", "replace")


failed = False
for path in ("/v1/health", "/v1/tiers", "/openapi.json"):
    try:
        status, body = get(path)
        print(f"  [{'PASS' if status == 200 else 'FAIL'}] {path}: http {status}")
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
sys.exit(1 if failed else 0)
PY
    ;;

  stop)
    pkill -f "mineru-api --port $PORT" 2>/dev/null && echo "stopped" || echo "no MinerU API was running on port $PORT"
    # The API holds the GPU models and worker processes and takes more than a moment to exit; training
    # follows straight after, so wait for it (up to 30 s), then force it, rather than report too early.
    for _ in $(seq 1 30); do pgrep -f "mineru-api --port $PORT" >/dev/null 2>&1 || break; sleep 1; done
    pgrep -f "mineru-api --port $PORT" >/dev/null 2>&1 && { pkill -9 -f "mineru-api --port $PORT" 2>/dev/null || true; sleep 3; }
    # Closes the finished session too, so the next start can reuse the name.
    tmux kill-session -t "=fideon-mineru-server" 2>/dev/null || true
    # The GPU memory is free once the process is gone; say so when it is not.
    pgrep -f "mineru-api --port $PORT" >/dev/null 2>&1 && { echo "the server is still running" >&2; exit 1; } || true
    ;;

  *)
    echo "usage: bash scripts/mineru_server.sh start|check [--pdf FILE]|stop" >&2
    exit 2
    ;;
esac
