#!/bin/bash
# torch.distributed environment for one `srun` task per GPU, possibly spread over
# several nodes (sbatch --gpus=N --ntasks=N --gpus-per-task=1, no node shape).
# Usage: srun [--kill-on-bad-exit=1] bash scripts/clusters/srun_ddp.sh python script.py ...
set -euo pipefail
: "${SLURM_PROCID:?Run this under srun}" "${SLURM_NTASKS:?Run this under srun}"
export RANK="$SLURM_PROCID" WORLD_SIZE="$SLURM_NTASKS"
# A task bound to one GPU sees it as cuda:0; unbound tasks share the node's GPUs.
visible="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "$visible" && "$visible" != *,* ]]; then
    export LOCAL_RANK=0
    # NCCL's P2P transport cannot reach GPUs hidden by per-task binding ("invalid
    # device ordinal"); shared memory and the network work (checked on Fir, job
    # 67030081). Traffic is one LoRA-gradient all-reduce per optimizer step.
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
else
    export LOCAL_RANK="${SLURM_LOCALID:?}"
fi
MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
# Unique per job (array elements included), outside the ephemeral port range.
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export MASTER_ADDR MASTER_PORT
exec "$@"
