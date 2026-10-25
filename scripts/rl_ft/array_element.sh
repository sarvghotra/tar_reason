#!/bin/bash
# One element of an RL run chained as short Slurm jobs (--array=0-K%1).
#
#   bash scripts/rl_ft/array_element.sh CONFIG.env
#
# CONFIG.env exports RUN_NAME, MAX_STEPS and any scripts/rl_ft/bash.sh settings
# (submit_short_array.sh freezes one; with `cluv submit --chunking` use a committed
# file). The element takes the run's training lock, exits (and cancels the rest of
# the array) when the target is reached or the chain has stalled, otherwise
# resumes from the newest checkpoint and stops before an update that would not
# finish before the time limit. Request a USR1 shortly before the limit
# (--signal=B:USR1@600) as a backup stop.
set -euo pipefail
CONFIG="$(readlink -f "${1:?Usage: array_element.sh CONFIG.env}")"
source "$CONFIG"
: "${RUN_NAME:?CONFIG must set RUN_NAME}" "${MAX_STEPS:?CONFIG must set MAX_STEPS}"
CODE="${TAR_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
export TAR_REPO_ROOT="$CODE" LAUNCHER="${LAUNCHER:-srun}" JOB_DEADLINE="${JOB_DEADLINE:-1}"
export RESUME_SAVE_STEPS="${RESUME_SAVE_STEPS:-1}"
export TAR_STAGE_DIR="${TAR_STAGE_DIR:-${SLURM_TMPDIR:+$SLURM_TMPDIR/stage}}"
if [[ -n "${TAR_ROOT:-}" ]]; then
    RESULTS_ROOT="${TAR_RESULTS:-$TAR_ROOT/results}"
else
    RESULTS_ROOT="$CODE/results"
fi
OUTPUT_DIR="$RESULTS_ROOT/models/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"
GUARD=(/usr/bin/python3 "$CODE/scripts/rl_ft/short_array.py")
cancel_rest() { [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] && scancel -t PENDING "$SLURM_ARRAY_JOB_ID" || true; }

# Only one element may train this run at a time, whatever the array throttle.
exec 9>"$OUTPUT_DIR/active_job.lock"
flock -n 9 || { echo 'Another job holds the training lock'; exit 1; }
set +e
"${GUARD[@]}" check --output_dir "$OUTPUT_DIR" --target "$MAX_STEPS"
status=$?
set -e
if (( status == 10 )); then cancel_rest; exit 0; fi
if (( status != 0 )); then cancel_rest; exit 1; fi
START_STEP=$("${GUARD[@]}" latest --output_dir "$OUTPUT_DIR")
STOP_FILE="$OUTPUT_DIR/stop-${SLURM_JOB_ID:-local}"
trap 'touch "$STOP_FILE"' USR1
echo "element ${SLURM_ARRAY_TASK_ID:-0} job ${SLURM_JOB_ID:-local}: resume from step $START_STEP at $(date -Is) on ${SLURM_JOB_NODELIST:-$(hostname)}"
# shellcheck disable=SC2086  # TRAIN_EXTRA_ARGS is a word-split list of train_grpo.py flags
bash "$CODE/scripts/rl_ft/bash.sh" --stop_file "$STOP_FILE" ${TRAIN_EXTRA_ARGS:-} &
TRAIN_PID=$!
# The USR1 trap interrupts wait; keep waiting for the safe checkpoint.
set +e
while true; do
    wait "$TRAIN_PID"; TRAIN_STATUS=$?
    kill -0 "$TRAIN_PID" 2>/dev/null || break
done
"${GUARD[@]}" record --output_dir "$OUTPUT_DIR" --job_id "${SLURM_JOB_ID:-local}" \
    --start_step "$START_STEP" --exit_code "$TRAIN_STATUS"
"${GUARD[@]}" check --output_dir "$OUTPUT_DIR" --target "$MAX_STEPS"
status=$?
set -e
if (( status != 0 )); then cancel_rest; fi
exit "$TRAIN_STATUS"
