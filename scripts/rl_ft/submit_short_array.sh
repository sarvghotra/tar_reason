#!/bin/bash
# Run scripts/rl_ft/bash.sh as a chain of short Slurm jobs: --array=0-(ARRAY_SIZE-1)%1
# lets one element run at a time; each resumes from the newest checkpoint, stops
# before an update that would not finish (JOB_DEADLINE=1), and the chain cancels
# itself when MAX_STEPS is reached or two elements in a row make no progress.
#
#   RUN_NAME=my_run OBJECTIVE=... [other bash.sh settings] bash scripts/rl_ft/submit_short_array.sh
#
# GPUs are requested per task with no node shape (--ntasks=GPUS --gpus-per-task=TYPE:1)
# under one network switch when possible (SWITCHES, e.g. 1@00:30:00), so the job can
# start on partly free nodes. Code and settings are frozen into
# $RESULTS_ROOT/launches/$RUN_NAME at submission. TAR_DRY_RUN=1 writes but does not submit.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/cluster_env.sh"
: "${RUN_NAME:?Set RUN_NAME}"
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid RUN_NAME' >&2; exit 1; }
GPUS="${GPUS:-4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-6}"
MEM_PER_GPU="${MEM_PER_GPU:-64G}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
SWITCHES="${SWITCHES-1@00:30:00}"
SIGNAL_LEAD="${SIGNAL_LEAD:-600}"   # USR1 backup stop, seconds before the time limit
ARRAY_SIZE="${ARRAY_SIZE:-60}"
MAX_STEPS="${MAX_STEPS:-500}"
ACCOUNT="${SLURM_ACCOUNT:-}"
if [[ -z "$ACCOUNT" && "${TAR_ACCOUNT_REQUIRED:-1}" == 1 ]]; then
    echo "Set SLURM_ACCOUNT for $TAR_CLUSTER (check \`sshare -U\`; put it in scripts/clusters/$TAR_CLUSTER/config.local.sh)." >&2
    exit 1
fi
# per_task: GPUs anywhere (no node shape). whole_node: clusters that only allocate
# whole nodes (Trillium, tamIA); GPUS must then be a multiple of the node's GPUs.
REQUEST_STYLE="${REQUEST_STYLE:-${TAR_GPU_REQUEST_STYLE:-per_task}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
LAUNCH_DIR="$RESULTS_ROOT/launches/$RUN_NAME"
OUTPUT_DIR="$RESULTS_ROOT/models/$RUN_NAME"
LOG_DIR="$RESULTS_ROOT/logs/$RUN_NAME"
if [[ -e "$LAUNCH_DIR" ]]; then
    echo "$LAUNCH_DIR exists; a run's code and settings are frozen once. Use a new RUN_NAME." >&2
    exit 1
fi
mkdir -p "$LAUNCH_DIR/code" "$OUTPUT_DIR" "$LOG_DIR"

# Freeze the code (tracked and untracked, not ignored) and record its provenance.
git -C "$REPO_ROOT" ls-files -z -co --exclude-standard \
    | rsync -a --from0 --files-from=- "$REPO_ROOT/" "$LAUNCH_DIR/code/"
git -C "$REPO_ROOT" rev-parse HEAD > "$LAUNCH_DIR/git_head.txt"
git -C "$REPO_ROOT" status --porcelain=v1 > "$LAUNCH_DIR/git_status.txt"
git -C "$REPO_ROOT" diff HEAD > "$LAUNCH_DIR/git_diff.patch"
if [[ -n "${TAR_ROOT:-}" ]]; then
    # Model paths are absolute in uv mode; data YAMLs still name data/... relative to the code.
    ln -s "$TAR_SHARED/data" "$LAUNCH_DIR/code/data"
    ln -s "$RESULTS_ROOT" "$LAUNCH_DIR/code/results"
else
    for name in data models sft_model reward_model results; do
        if [[ -L "$REPO_ROOT/$name" ]]; then
            ln -s "$(readlink -f "$REPO_ROOT/$name")" "$LAUNCH_DIR/code/$name"
        fi
    done
fi
if [[ -n "${TAR_ROOT:-}" ]]; then
    TAR_ENV="$(readlink -f "${TAR_ENV:-$REPO_ROOT/.venv}")"   # the immutable lock-hash venv
else
    TAR_ENV="$(readlink -f "${TAR_ENV:-$REPO_ROOT/tar}")"
fi

# Freeze every launcher setting that is currently set (bash.sh knobs plus paths).
{
    printf '# Frozen at %s by %s\n' "$(date -Is)" "$0"
    names=$(grep -ohE '^[A-Z][A-Z0-9_]*="\$\{[A-Z][A-Z0-9_]*:-' "$REPO_ROOT/scripts/rl_ft/bash.sh" | cut -d= -f1)
    names=$(printf '%s\n' $names TAR_CLUSTER TAR_ROOT TAR_SHARED TAR_RESULTS SFT_MODEL VISION_MODEL AR_MODEL \
                DECODER REWARD_MODEL_PATH REWARD_PYTHON WANDB_MODE WANDB_PROJECT HF_HUB_CACHE \
                PREV_STAGE_CHECKPOINT DATA_PATH VAL_DATA_PATH | awk '!seen[$0]++')
    for name in $names; do
        if [[ -n "${!name+x}" ]]; then printf 'export %s=%q\n' "$name" "${!name}"; fi
    done
    printf 'export RUN_NAME=%q MAX_STEPS=%q TAR_ENV=%q TAR_REPO_ROOT=%q\n' \
        "$RUN_NAME" "$MAX_STEPS" "$TAR_ENV" "$LAUNCH_DIR/code"
    printf 'export LAUNCHER=srun JOB_DEADLINE=1 RESUME_SAVE_STEPS=%q\n' "${RESUME_SAVE_STEPS:-1}"
    printf 'TRAIN_EXTRA_ARGS=%q\n' "${TRAIN_EXTRA_ARGS:-}"   # extra train_grpo.py flags, word-split
} > "$LAUNCH_DIR/config.env"

case "$REQUEST_STYLE" in
    per_task)
        # NODES (optional) pins the node count, e.g. to test training across nodes.
        RESOURCES="${NODES:+#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=$((GPUS / NODES))
}#SBATCH --ntasks=$GPUS
#SBATCH --gpus-per-task=${GPU_TYPE:+$GPU_TYPE:}1
#SBATCH --cpus-per-task=$CPUS_PER_GPU
#SBATCH --mem-per-gpu=$MEM_PER_GPU${SWITCHES:+
#SBATCH --switches=$SWITCHES}" ;;
    whole_node)
        (( GPUS % GPUS_PER_NODE == 0 )) || { echo "GPUS must be a multiple of $GPUS_PER_NODE" >&2; exit 1; }
        RESOURCES="#SBATCH --nodes=$((GPUS / GPUS_PER_NODE))
#SBATCH --ntasks-per-node=$GPUS_PER_NODE
#SBATCH --gpus-per-node=${GPU_TYPE:+$GPU_TYPE:}$GPUS_PER_NODE
#SBATCH --cpus-per-task=$CPUS_PER_GPU" ;;
    *) echo 'REQUEST_STYLE must be per_task or whole_node.' >&2; exit 1 ;;
esac
cat > "$LAUNCH_DIR/array.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=rl-${RUN_NAME:0:40}
${ACCOUNT:+#SBATCH --account=$ACCOUNT}
$RESOURCES
#SBATCH --time=$TIME_LIMIT
#SBATCH --signal=B:USR1@$SIGNAL_LEAD
#SBATCH --output=$LOG_DIR/%A_%a.out
#SBATCH --error=$LOG_DIR/%A_%a.err
set -euo pipefail
exec bash "$LAUNCH_DIR/code/scripts/rl_ft/array_element.sh" "$LAUNCH_DIR/config.env"
EOF

if [[ "${TAR_DRY_RUN:-0}" == 1 ]]; then
    echo "Dry run: wrote $LAUNCH_DIR/{config.env,array.sbatch}; not submitted."
    exit 0
fi
# Unset node-local paths of the submitting shell (a running job's TMPDIR would leak in).
JOB=$(env -u TMPDIR -u SLURM_TMPDIR sbatch --parsable --array="0-$((ARRAY_SIZE - 1))%1" "$LAUNCH_DIR/array.sbatch")
JOB="${JOB%%;*}"
printf '{"run_name": "%s", "array_job_id": "%s", "array_size": %d, "submitted": "%s"}\n' \
    "$RUN_NAME" "$JOB" "$ARRAY_SIZE" "$(date -Is)" > "$LAUNCH_DIR/launch.json"
echo "Submitted array $JOB (0-$((ARRAY_SIZE - 1))%1) for $RUN_NAME"
