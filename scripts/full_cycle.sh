#!/usr/bin/env bash
# A whole cycle on the pod in one command, from the documents already imported to the packaged release:
#
#   bash scripts/pod_run.sh start cycle -- bash scripts/full_cycle.sh [--after RUN]... [--tenant T] [-- FINETUNE_OPTIONS]
#
#   e.g. bash scripts/pod_run.sh start cycle2 -- bash scripts/full_cycle.sh --after importc2 --after ocrloop -- --skip-ingest --fresh
#
# 1. Waits for each --after run (a pod_run.sh name: the import, an OCR pass) to finish, and stops if one did not
#    exit 0 - a cycle trained on half an import is not the cycle that was asked for.
# 2. Stops the MinerU servers: training needs the GPUs. The OCR stage starts them again itself, for its length,
#    if any document is still unread.
# 3. Takes the next version and runs `finetune --version V` with the options given (the OCR stage, the corpus
#    build, training, checkpoint selection).
# 4. Freezes V's test split as the tenant's eval set when none is frozen (the gate scores the frozen set; nothing
#    in the build makes it), and copies the frozen golds' review records (the gate refuses a gold without one).
# 5. Runs `package --version V --keep-staging` (calibration, the gate on every GPU, the release, left gated).
#
# It stops at the first step that fails and says how to resume from there; nothing after a failed step runs.
#   FIDEON_PYTHON     /workspace/venv/bin/python   the interpreter the steps run with
#   FIDEON_LOG_DIR    /workspace/logs              where pod_run.sh keeps each run's .exit marker
#   FULL_CYCLE_POLL_S 60                           seconds between looks at an --after run
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${FIDEON_PYTHON:-/workspace/venv/bin/python}"
LOG_DIR="${FIDEON_LOG_DIR:-/workspace/logs}"
# Already in tmux (pod_run.sh): every step runs in place, never in a tmux session of its own.
export FIDEON_DETACHED=1

after=()
tenant=""
while [ $# -gt 0 ]; do
  case "$1" in
    --after) [ $# -ge 2 ] || { echo "full_cycle: --after needs a run name" >&2; exit 2; }; after+=("$2"); shift 2 ;;
    --tenant) [ $# -ge 2 ] || { echo "full_cycle: --tenant needs a name" >&2; exit 2; }; tenant="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "full_cycle: unknown option $1 (finetune's options go after --)" >&2; exit 2 ;;
  esac
done
tenant_args=()
[ -n "$tenant" ] && tenant_args=(--tenant "$tenant")

say() { echo "$(date -u '+%Y-%m-%d %H:%M:%S') UTC full_cycle: $*"; }
# The value a short Python expression prints last (anything a library logs on stdout before it is ignored).
value() { "$PY" -c "$1" "$tenant" | tail -1; }

# 1. What must be finished first.
for name in ${after[@]+"${after[@]}"}; do
  say "waiting for '$name' to finish"
  until [ -f "$LOG_DIR/$name.exit" ]; do sleep "${FULL_CYCLE_POLL_S:-60}"; done
  code="$(tr -d '[:space:]' < "$LOG_DIR/$name.exit")"
  if [ "$code" != "0" ]; then
    say "'$name' ended with '$code', not 0: stopping before any training. Read it: bash scripts/pod_run.sh status $name"
    exit 1
  fi
  say "'$name' finished, exit 0"
done

# 2. The GPUs, free for training.
say "stopping the MinerU servers: training needs the GPUs"
bash scripts/mineru_server.sh stop

# 3. The cycle.
version="$(value 'import sys
from artifact_registry.blob_client import BlobClient
from registry_utils.releases import next_cycle_version
print(next_cycle_version(BlobClient(), sys.argv[1] or None))')"
case "$version" in
  v[0-9]*) ;;
  *) say "could not work out the next version (got '$version'): stopping"; exit 1 ;;
esac
say "this cycle is $version: finetune $*"
if ! "$PY" -m orchestration.run finetune --version "$version" ${tenant_args[@]+"${tenant_args[@]}"} "$@"; then
  say "finetune stopped. Resume it: bash scripts/pod_run.sh start ${version}r -- $PY -m orchestration.run finetune" \
      "--version $version --from-stage <the stage that failed> $*   then freeze and package as below"
  exit 1
fi

# 4. The eval set the gate scores.
frozen="$(value 'import sys
from artifact_registry.blob_client import BlobClient
from evaluation.freeze_eval_set import is_frozen
print(is_frozen(BlobClient(), tenant_id=sys.argv[1] or None))')"
if [ "$frozen" = "True" ]; then
  say "the eval set is already frozen: the gate scores it as it is"
elif [ "$frozen" = "False" ]; then
  say "freezing $version's test split as the eval set"
  "$PY" -m orchestration.run freeze-eval-set --corpus "$version" ${tenant_args[@]+"${tenant_args[@]}"}
else
  say "could not tell whether the eval set is frozen (got '$frozen'): stopping before package"
  exit 1
fi
say "copying the frozen golds' review records"
if ! "$PY" -m orchestration.run freeze-eval-set --mark-reviewed ${tenant_args[@]+"${tenant_args[@]}"}; then
  say "a frozen gold has no review record, so the gate would refuse the set: stopping before package"
  exit 1
fi

# 5. Calibration, the gate, the release.
say "package $version"
if ! "$PY" -m orchestration.run package --version "$version" --keep-staging ${tenant_args[@]+"${tenant_args[@]}"}; then
  say "package stopped. Resume it: bash scripts/pod_run.sh start ${version}p -- $PY -m orchestration.run package" \
      "--version $version --from-stage <calibrate|evaluation_gate|package> --keep-staging"
  exit 1
fi
say "done: $version is packaged and gated, not promoted (shadow, then promote, decide that)"
