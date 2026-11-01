#!/usr/bin/env bash
# Run a pipeline command on the RunPod pod inside tmux, so it keeps running when
# the laptop shuts down, sleeps, or loses Wi-Fi.
#
#   bash scripts/pod_run.sh start <name> <orchestration.run arguments...>
#   bash scripts/pod_run.sh start <name> -- <any command...>
#   bash scripts/pod_run.sh attach <name>     watch it live (detach: Ctrl-b then d)
#   bash scripts/pod_run.sh status [<name>]   running or finished, exit code, last log lines
#   bash scripts/pod_run.sh tail <name>       follow the log (Ctrl-c stops following, not the run)
#   bash scripts/pod_run.sh list              every run this pod knows about
#   bash scripts/pod_run.sh stop <name>       the ONLY way this script ends a run
#   bash scripts/pod_run.sh env               the environment a run started now gets (env -0)
#
# Example:
#   bash scripts/pod_run.sh start finetune-v3 finetune --version v3 --corpus-version v1
#   bash scripts/pod_run.sh start package-v3 package --version v3
#   bash scripts/pod_run.sh start spike -- python scripts/phase0_spike.py
#
# You rarely need to call this yourself: on the pod, `python -m orchestration.run`,
# the Phase 0 spike and setup_pod.sh detach through it automatically
# (orchestration/detach.py).
#
# Why tmux: an SSH session that ends takes its foreground processes with it
# (SIGHUP). A tmux session belongs to the tmux server on the pod, not to the SSH
# connection, so the pipeline - and the ms-swift training it launches - carry on
# until they finish. Nothing here stops the pod or the run on its own; the session
# stays open after the run ends so its output can be read.
#
# What it does NOT survive: the pod itself stopping or restarting (a manual stop,
# RunPod host maintenance, running out of credit). Then resume with
# `--from-stage`; checkpoints and manifests on the volume and in Blob are kept.
#
# A run's settings: the repo's .env, under the caller's own. A setting .env (or
# .env.example) names that the starting shell set - inline included,
# `FIDEON_REMOTE_TOKEN=x python -m testing.extract_server` - is kept over .env's
# value, empty or not. A value the pod gives every process is not the caller's:
# .env still overrides it (HF_HUB_ENABLE_HF_TRANSFER=1 on RunPod images).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${FIDEON_LOG_DIR:-/workspace/logs}"
PREFIX="fideon-"

die() { echo "pod_run: $*" >&2; exit 1; }

require_tmux() {
  command -v tmux >/dev/null 2>&1 && return
  if [ "$(id -u)" = "0" ] && command -v apt-get >/dev/null 2>&1; then
    echo "pod_run: installing tmux"
    apt-get update -qq && apt-get install -y -qq tmux >/dev/null
  else
    die "tmux is not installed (apt-get install -y tmux)"
  fi
}

session_of() { echo "${PREFIX}$1"; }
log_of() { echo "${LOG_DIR}/$1.log"; }
exit_file_of() { echo "${LOG_DIR}/$1.exit"; }

valid_name() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || die "run name '$1' may use only letters, digits, '.', '_' and '-'"
}

running() { tmux has-session -t "=$(session_of "$1")" 2>/dev/null; }

# Every setting the repo's .env or .env.example names.
setting_names() {
  local file
  for file in "$REPO/.env" "$REPO/.env.example"; do
    if [ -f "$file" ]; then
      sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$file"
    fi
  done | sort -u
}

# The value the pod itself gives a variable - its first process's - or nothing.
pod_value() {
  local entry
  [ -r /proc/1/environ ] || return 0
  while IFS= read -r -d '' entry; do
    if [ "${entry%%=*}" = "$1" ]; then printf '%s' "${entry#*=}"; return 0; fi
  done < /proc/1/environ
}

# The caller's own settings, as export lines to source after .env. A tmux session
# takes its environment from the tmux server, not from the shell that starts the
# run, and .env was sourced over it: `FIDEON_REMOTE_TOKEN=x python -m ...` lost
# its token to .env's empty line the moment the command detached.
callers_settings() {
  local var
  for var in $(setting_names); do
    if [ -n "${!var:-}" ] && [ "${!var}" != "$(pod_value "$var")" ]; then
      printf 'export %s=%q\n' "$var" "${!var}"
    fi
  done
}

# The start of a run's script: the repo, its .env, then the caller's settings from
# $1 (owner-only, deleted once read: a token stays out of this script and the log),
# then the caller's environment.
prelude() {
  local keep="$1"
  echo '#!/usr/bin/env bash'
  echo 'set -o pipefail'
  printf 'cd %q\n' "$REPO"
  # Credentials and settings from the repo's .env, as an interactive shell would have them.
  echo 'if [ -f .env ]; then set -a; . ./.env; set +a; fi'
  printf 'if [ -f %q ]; then . %q; rm -f %q; fi\n' "$keep" "$keep" "$keep"
  echo 'export PYTHONUNBUFFERED=1'
  # The caller's PATH and virtualenv. A tmux session takes its environment from
  # the tmux server, started by whichever job came first — not from the shell
  # that starts this run — so without these a job launched from the training
  # venv would not find that venv's `swift`, and one from the OCR venv would
  # run its tools from the wrong environment.
  printf 'export PATH=%q\n' "$PATH"
  if [ -n "${VIRTUAL_ENV:-}" ]; then printf 'export VIRTUAL_ENV=%q\n' "$VIRTUAL_ENV"; fi
  # Started from a shell with no virtualenv active, a job would run the system python, which has no
  # azure or torch (the first smoke run died on that): the pod's training venv is the default.
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -x "${FIDEON_WORKSPACE:-/workspace}/venv/bin/python" ]; then
    printf 'export PATH=%q:"$PATH"\n' "${FIDEON_WORKSPACE:-/workspace}/venv/bin"
    printf 'export VIRTUAL_ENV=%q\n' "${FIDEON_WORKSPACE:-/workspace}/venv"
  fi
  # The caller's GPU selection, set or not - for the same reason. Sharded OCR
  # gives each process its own CUDA_VISIBLE_DEVICES; inherited from the tmux
  # server instead, all four shards landed on one card, and the first shard's
  # single GPU leaked into every later job, so training ran on one GPU of four
  # without saying so.
  if [ -n "${CUDA_VISIBLE_DEVICES+x}" ]; then
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
  else
    echo 'unset CUDA_VISIBLE_DEVICES'
  fi
  # Tells the command it is already detached, so it does not detach again.
  echo 'export FIDEON_DETACHED=1'
}

cmd_start() {
  [ $# -ge 2 ] || die "usage: start <name> <orchestration.run arguments...>"
  local name="$1"; shift
  valid_name "$name"
  require_tmux
  mkdir -p "$LOG_DIR"
  local session log exit_file
  session="$(session_of "$name")"; log="$(log_of "$name")"; exit_file="$(exit_file_of "$name")"

  # One run per name. A second run on the same name would write the same log and,
  # far worse, the same staging paths as the one still training.
  if running "$name"; then
    if [ -f "$exit_file" ]; then
      die "run '$name' finished (exit $(cat "$exit_file")) but its session is still open; read it with 'attach $name', then close it with 'stop $name' before reusing the name"
    fi
    die "run '$name' is still running. Watch it with: bash scripts/pod_run.sh attach $name"
  fi
  rm -f "$exit_file" "${LOG_DIR}/$name.notified"

  # POD_RUN_RETRY_STALLED=N: run the same command again, up to N times, when it stopped itself on a stall
  # (inference_core.stall_guard). Running the same command again is only right for one that resumes a named
  # cycle; a fresh `finetune` would start the next version, so that is refused here.
  local retries="${POD_RUN_RETRY_STALLED:-0}"
  case "$retries" in ''|*[!0-9]*) die "POD_RUN_RETRY_STALLED is a whole number of times to start the run again" ;; esac
  if [ "$retries" -gt 0 ] && [[ " $* " != *" --from-stage "* ]]; then
    die "POD_RUN_RETRY_STALLED runs the same command again, which is only safe for a command that resumes a cycle (--from-stage STAGE --version vN); this one would start a new cycle"
  fi

  # The caller's own settings, for the run to read over .env.
  local keep="${LOG_DIR}/$name.env"
  (umask 077; callers_settings > "$keep")
  # The command the session runs, written to a file so its quoting survives tmux.
  local runner="${LOG_DIR}/$name.run.sh"
  {
    prelude "$keep"
    # The run's name, for the pipeline's closing message to mark in the log folder (orchestration.notify).
    printf 'export FIDEON_RUN_NAME=%q\n' "$name"
    printf 'echo "=== %s started $(date -u +%%FT%%TZ) on $(hostname) ==="\n' "$name"
    # A log that has not moved for POD_RUN_STALE_MINUTES (default 15; 0 turns it off) is told once as a
    # possible hang, and again if it stalls once more after moving. A hard-killed process cannot say so
    # itself. Text messages are sent only when a channel is set up (orchestration.notify).
    printf 'stale_minutes=%q\n' "${POD_RUN_STALE_MINUTES:-15}"
    echo 'watcher='
    echo 'if [ "$stale_minutes" -gt 0 ] 2>/dev/null; then'
    echo '  ('
    echo '    told=0'
    echo '    while sleep 60; do'
    printf '      age=$(( $(date +%%s) - $(stat -c %%Y %q 2>/dev/null || date +%%s) ))\n' "$log"
    echo '      if [ "$age" -ge $((stale_minutes * 60)) ]; then'
    printf '        [ "$told" = 0 ] && python -m orchestration.notify --text "Run %s: the log has not moved for $((age / 60)) minutes, so it may be stuck." >/dev/null 2>&1\n' "$name"
    echo '        told=1'
    echo '      else told=0; fi'
    echo '    done'
    echo '  ) &'
    echo '  watcher=$!'
    echo 'fi'
    echo 'attempt=1'
    echo 'while :; do'
    # Where this attempt's output starts in the log, to read only that for a stall.
    printf '  offset=$(stat -c %%s %q 2>/dev/null || echo 0)\n' "$log"
    printf '  '
    if [ "$1" = "--" ]; then
      shift
      [ $# -ge 1 ] || die "start $name -- needs a command"
      printf '%q' "$1"; shift
    else
      printf 'python -m orchestration.run'
    fi
    printf ' %q' "$@"
    printf ' 2>&1 | tee -a %q\n' "$log"
    echo '  status=${PIPESTATUS[0]}'
    # The two lines a stalled process leaves: inference_core.stall_guard's own, and faulthandler's backstop.
    printf '  if [ "$status" != 0 ] && [ "$attempt" -le %d ] && tail -c +$((offset + 1)) %q | grep -aqE %q; then\n' \
      "$retries" "$log" '^(STALL-GUARD:|Timeout \()'
    echo '    attempt=$((attempt + 1)); echo "=== stalled; starting it again, attempt $attempt ==="; sleep 30; continue'
    echo '  fi'
    echo '  break'
    echo 'done'
    echo '[ -n "$watcher" ] && kill "$watcher" 2>/dev/null'
    # A run that failed without having told anyone (a crash, a kill): the pipeline marks its own closing
    # message in the log folder, so a run that already reported its failure is not reported twice.
    printf 'if [ "$status" != 0 ] && [ ! -e %q ]; then python -m orchestration.notify --text "Run %s ended with exit code $status and sent no closing message: it may have crashed or been killed. Log: %s" >/dev/null 2>&1; fi\n' \
      "${LOG_DIR}/$name.notified" "$name" "$log"
    printf 'echo "$status" > %q\n' "$exit_file"
    printf 'echo "=== %s finished $(date -u +%%FT%%TZ), exit $status ==="\n' "$name"
    # Keep the session open after the run, so its last screen can still be read.
    echo 'exec bash'
  } > "$runner"
  chmod +x "$runner"

  tmux new-session -d -s "$session" -x 220 -y 50 "bash $(printf %q "$runner")" \
    || { rm -f "$keep"; die "tmux could not start the session for '$name'"; }
  echo "started '$name' in tmux session '$session'"
  echo "  log:    $log"
  echo "  watch:  bash scripts/pod_run.sh attach $name   (detach with Ctrl-b then d)"
  echo "  status: bash scripts/pod_run.sh status $name"
  echo "You can close the laptop now; the run continues on the pod."
}

cmd_attach() {
  [ $# -eq 1 ] || die "usage: attach <name>"
  require_tmux
  running "$1" || die "no session for run '$1' (see: bash scripts/pod_run.sh list)"
  exec tmux attach -t "=$(session_of "$1")"
}

cmd_status() {
  if [ $# -eq 0 ]; then cmd_list; return; fi
  local name="$1" log exit_file
  log="$(log_of "$name")"; exit_file="$(exit_file_of "$name")"
  if [ -f "$exit_file" ]; then
    echo "run '$name': FINISHED, exit $(cat "$exit_file")"
  elif command -v tmux >/dev/null 2>&1 && running "$name"; then
    echo "run '$name': RUNNING"
  elif [ -f "$log" ]; then
    echo "run '$name': NOT RUNNING and no exit recorded - the pod restarted or the session was killed mid-run. Resume with --from-stage."
  else
    die "no run named '$name'"
  fi
  [ -f "$log" ] && { echo "--- last 20 lines of $log ---"; tail -n 20 "$log"; }
}

cmd_tail() {
  [ $# -eq 1 ] || die "usage: tail <name>"
  local log; log="$(log_of "$1")"
  [ -f "$log" ] || die "no log for run '$1' at $log"
  echo "following $log - Ctrl-c stops following; the run keeps going"
  exec tail -n 50 -F "$log"
}

cmd_list() {
  shopt -s nullglob
  local found=0 f name
  for f in "$LOG_DIR"/*.log; do
    found=1; name="$(basename "$f" .log)"
    if [ -f "$(exit_file_of "$name")" ]; then
      printf '  %-30s finished, exit %s\n' "$name" "$(cat "$(exit_file_of "$name")")"
    elif command -v tmux >/dev/null 2>&1 && running "$name"; then
      printf '  %-30s running\n' "$name"
    else
      printf '  %-30s interrupted (no exit recorded)\n' "$name"
    fi
  done
  [ "$found" = 1 ] || echo "no runs in $LOG_DIR"
}

cmd_env() {
  # What a run started now would have in its environment, NUL-separated: read by a
  # command that checks its settings before it detaches (testing/latency_check.py).
  local keep status=0
  keep="$(mktemp)"
  callers_settings > "$keep"
  bash -c "$(prelude "$keep"); env -0" || status=$?
  rm -f "$keep"
  return "$status"
}

cmd_stop() {
  [ $# -eq 1 ] || die "usage: stop <name>"
  require_tmux
  running "$1" || die "no session for run '$1'"
  if [ ! -f "$(exit_file_of "$1")" ]; then
    read -r -p "run '$1' is still RUNNING. Kill it? Type the run name to confirm: " answer
    [ "$answer" = "$1" ] || die "not confirmed; the run keeps going"
  fi
  tmux kill-session -t "=$(session_of "$1")"
  echo "session for '$1' closed"
}

case "${1:-}" in
  start)  shift; cmd_start "$@" ;;
  attach) shift; cmd_attach "$@" ;;
  status) shift; cmd_status "$@" ;;
  tail)   shift; cmd_tail "$@" ;;
  list)   shift; cmd_list ;;
  stop)   shift; cmd_stop "$@" ;;
  env)    shift; cmd_env ;;
  *) sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
