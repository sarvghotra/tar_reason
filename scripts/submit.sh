#!/bin/bash
# Usage: bash scripts/submit.sh scripts/rl_ft/bash.sh [sbatch options...]
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export TAR_REPO_ROOT="$REPO_ROOT"
source "$REPO_ROOT/scripts/clusters/profile.sh"
: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT in the environment or cluster config.local.sh}"
script="${1:?Usage: bash scripts/submit.sh SCRIPT [sbatch options...]}"
shift
[[ -f "$script" ]] || { echo "Script not found: $script" >&2; exit 1; }
# Slurm opens its output files before the job shell starts.
if [[ "${TAR_DRY_RUN:-0}" != 1 ]]; then
    mkdir -p "$REPO_ROOT/results/logs/slurm"
fi
command=(sbatch --account="$SLURM_ACCOUNT" \
    --nodes=1 --ntasks=1 --gpus-per-node="${GPU_TYPE:+$GPU_TYPE:}${N_GPUS:-4}" \
    --cpus-per-task="${CPUS_PER_TASK:-24}" --mem="${JOB_MEM:-256G}" --time="${JOB_TIME:-03:00:00}" \
    --chdir="$REPO_ROOT" \
    --output="$REPO_ROOT/results/logs/slurm/%x-%j.out" \
    --error="$REPO_ROOT/results/logs/slurm/%x-%j.err" \
    "$@" "$script")
if [[ "${TAR_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"; printf '\n'
else
    exec "${command[@]}"
fi
