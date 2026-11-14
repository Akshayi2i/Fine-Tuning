#!/usr/bin/env bash
# A cycle's status in plain words, for anyone: what is running, which step, whether it is healthy, and any error.
# Reads the logs only - nothing is started, stopped or changed.
#
#   bash scripts/cycle_status.sh                          # the runs of cycle 2: cycle2, importc2, ocrloop
#   bash scripts/cycle_status.sh CYCLE [IMPORT] [OCR]     # other run names (pod_run.sh names)
#
#   FIDEON_LOG_DIR      /workspace/logs   where pod_run.sh keeps each run's log and .exit marker
#   CYCLE_STATUS_QUIET  20                minutes a running step's log may stay quiet before it is flagged
set -uo pipefail
cd "$(dirname "$0")/.."

CYCLE="${1:-cycle2}"
IMPORT="${2:-importc2}"
OCR="${3:-ocrloop}"
LOG_DIR="${FIDEON_LOG_DIR:-/workspace/logs}"
QUIET_MIN="${CYCLE_STATUS_QUIET:-20}"
problems=()
notes=()

# running / finished-ok / failed / never-started, from pod_run.sh's own markers.
state_of() {
  local exit_file="$LOG_DIR/$1.exit"
  if [ -f "$exit_file" ]; then
    [ "$(tr -d '[:space:]' < "$exit_file")" = "0" ] && echo "done" || echo "failed"
  elif [ -f "$LOG_DIR/$1.log" ]; then
    if tmux has-session -t "=fideon-$1" 2>/dev/null; then echo "running"; else echo "dead"; fi
  else
    echo "not started"
  fi
}
minutes_quiet() {
  local log="$LOG_DIR/$1.log"
  [ -f "$log" ] || { echo 0; return; }
  echo $(( ( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) ) / 60 ))
}
# Lines that mean something went wrong (not the counts of a summary such as "0 failed").
errors_in() {
  local log="$LOG_DIR/$1.log"
  [ -f "$log" ] || return 0
  # full_cycle's own stops say "stopping before ..." or "... stopped"; "stopping the MinerU servers" is a normal step.
  grep -aE " ERROR |Traceback|full_cycle: .*(stopping before|: stopping$|stopped)|stage [0-9]+ [a-z_]+: failed|failed after [0-9]+ attempt|CUDA out of memory|Killed" \
    "$log" | tail -n 5 | cut -c1-220
}

# India time by adding 5 h 30 min to UTC: a container may have no time-zone files, and then TZ=Asia/Kolkata
# quietly reads as UTC.
ist="$(date -u -d '+5 hours 30 minutes' '+%-I:%M %p IST, %d %b')"
echo "=================================================================="
echo " CYCLE STATUS  -  $ist"
echo "=================================================================="

# --- the step that is running now --------------------------------------------------------------------------
import_state="$(state_of "$IMPORT")"; ocr_state="$(state_of "$OCR")"; cycle_state="$(state_of "$CYCLE")"
# grep -c prints 0 itself when nothing matches; empty only when the log is not there yet.
imported="$(grep -acE 'ingested .* as |identical content already' "$LOG_DIR/$IMPORT.log" 2>/dev/null)"; imported="${imported:-0}"
read_docs="$(grep -ac 'processed policy_' "$LOG_DIR/$OCR.log" 2>/dev/null)"; read_docs="${read_docs:-0}"
cycle_log="$LOG_DIR/$CYCLE.log"
last_stage="$(grep -aoE 'stage [0-9]+ [a-z_]+(: [a-z_]+ in [0-9]+m [0-9]+s)?' "$cycle_log" 2>/dev/null | tail -1)"
version="$(grep -aoE 'this cycle is v[0-9.]+' "$cycle_log" 2>/dev/null | tail -1 | awk '{print $4}')"

plain() {
  case "$1" in
    *preprocessing*) echo "Checking every page is read (OCR)" ;;
    *labeling*) echo "Checking the labels" ;;
    *dataset_build*) echo "Building the training set" ;;
    *training*) echo "Training the model" ;;
    *checkpoint_eval*) echo "Picking the best checkpoint" ;;
    *calibrate*) echo "Calibrating confidence scores" ;;
    *evaluation_gate*) echo "Final exam on the test documents (gate)" ;;
    *package*) echo "Packaging the release" ;;
    *ingestion*) echo "Ingesting documents" ;;
    *) echo "$1" ;;
  esac
}

if [ "$cycle_state" = "done" ]; then
  now="FINISHED - ${version:-the version} is packaged and gated (not promoted)"
elif grep -aq "full_cycle: package" "$cycle_log" 2>/dev/null && [ -n "$last_stage" ]; then
  now="$(plain "$last_stage")"
elif grep -aqE "full_cycle: (freezing|copying the frozen)" "$cycle_log" 2>/dev/null \
     && ! grep -aq "full_cycle: package" "$cycle_log" 2>/dev/null; then
  now="Freezing the test set"
elif [ -n "$last_stage" ]; then
  now="$(plain "$last_stage")"
  case "$last_stage" in *": completed in"*) now="$now - done, starting the next step" ;; esac
elif [ "$cycle_state" = "running" ] && grep -aq "full_cycle: stopping the MinerU" "$cycle_log" 2>/dev/null; then
  now="Starting the training run"
elif [ "$import_state" = "running" ]; then
  now="Importing documents ($imported of 836) and reading their pages ($read_docs read)"
elif [ "$ocr_state" = "running" ]; then
  now="Reading pages (OCR) - final pass ($read_docs documents read)"
else
  now="Nothing running"
fi
echo
echo " NOW:      $now"
[ -n "$version" ] && echo " VERSION:  $version"

# Training progress and time left, when training is under way.
step="$(grep -aoE "global_step/max_steps'?: '?[0-9]+/[0-9]+" "$cycle_log" 2>/dev/null | tail -1 | grep -oE '[0-9]+/[0-9]+')"
left="$(grep -aoE "remaining_time'?: '?[^',}]+" "$cycle_log" 2>/dev/null | tail -1 | sed -E "s/.*: '?//")"
if [ -n "$step" ] && ! grep -aqE 'stage 5 training: [a-z_]+ in' "$cycle_log" 2>/dev/null; then
  done_steps="${step%/*}"; all_steps="${step#*/}"
  echo " TRAINING: step $done_steps of $all_steps ($(( 100 * done_steps / (all_steps > 0 ? all_steps : 1) ))%), about $left left"
fi

# --- every step ---------------------------------------------------------------------------------------------
mark() { case "$1" in done) echo "[done]   " ;; running) echo "[running]" ;; failed) echo "[FAILED] " ;; dead) echo "[STOPPED]" ;; *) echo "[waiting]" ;; esac; }
echo
echo " STEPS:"
echo "   $(mark "$import_state") 1. Import documents            ($imported of 836)"
echo "   $(mark "$ocr_state") 2. Read pages (OCR)            ($read_docs documents read)"
echo "   $(mark "$cycle_state") 3. Training run, then gate    ($CYCLE)"
for s in "4 dataset_build:Build training set" "5 training:Train the model" "6 checkpoint_eval:Pick best checkpoint" \
         "7 calibrate:Calibrate" "8 evaluation_gate:Gate (final exam)" "9 package:Package"; do
  key="${s%%:*}"; label="${s#*:}"
  if grep -aqE "stage $key: completed in" "$cycle_log" 2>/dev/null; then
    took="$(grep -aoE "stage $key: completed in [0-9]+m [0-9]+s" "$cycle_log" | tail -1 | sed 's/.*completed in //')"
    echo "         - $label: done ($took)"
  elif grep -aqE "stage $key: failed" "$cycle_log" 2>/dev/null; then
    echo "         - $label: FAILED"
  elif grep -aqE "stage $key( |$)" "$cycle_log" 2>/dev/null; then
    echo "         - $label: running"
  fi
done

# --- health -------------------------------------------------------------------------------------------------
for run in "$IMPORT" "$OCR" "$CYCLE"; do
  case "$(state_of "$run")" in
    failed) problems+=("'$run' ended with an error (exit $(tr -d '[:space:]' < "$LOG_DIR/$run.exit"))") ;;
    dead) problems+=("'$run' is not running and recorded no exit: the pod restarted or its session was killed") ;;
    running)
      q="$(minutes_quiet "$run")"
      # The cycle waiting for the import and the OCR writes nothing until they end: quiet, not stuck.
      if [ "$run" = "$CYCLE" ] && tail -n 1 "$LOG_DIR/$run.log" 2>/dev/null | grep -aq "full_cycle: waiting for"; then
        :
      elif [ "$q" -ge "$QUIET_MIN" ]; then
        problems+=("'$run' has written nothing for $q minutes: it may be stuck (training steps can be slow; check again soon)")
      fi ;;
  esac
done
for run in "$IMPORT" "$OCR" "$CYCLE"; do
  while IFS= read -r line; do [ -n "$line" ] && problems+=("$run: $line"); done < <(errors_in "$run")
done
ocr_failed="$(grep -ac 'OCR failed for' "$LOG_DIR/$OCR.log" 2>/dev/null)"; ocr_failed="${ocr_failed:-0}"
[ "$ocr_failed" -gt 0 ] && notes+=("$ocr_failed page-reading attempt(s) failed; the next pass and the training run's own check read them again")

if command -v nvidia-smi >/dev/null 2>&1; then
  gpus="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
          | awk -F', ' '{printf "GPU%s %s busy, %s used;  ", $1, $2, $3}')"
fi

echo
if [ ${#problems[@]} -eq 0 ]; then
  echo " HEALTH:   OK - no errors found"
else
  echo " HEALTH:   NEEDS ATTENTION - ${#problems[@]} problem(s):"
  for p in "${problems[@]}"; do echo "   ! $p"; done
fi
for n in ${notes[@]+"${notes[@]}"}; do echo "   note: $n"; done
[ -n "${gpus:-}" ] && { echo; echo " GPUS:     $gpus"; }
echo "=================================================================="
