#!/bin/bash
# Standalone GPU integration test; submit with sbatch or scripts/submit.sh.
set -euo pipefail
cd "${TAR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}"
source scripts/cluster_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "GPU smoke test: job ${SLURM_JOB_ID:-local} on $(hostname)"
nvidia-smi
exec python -u scripts/test_qwen_reward_gpu.py
