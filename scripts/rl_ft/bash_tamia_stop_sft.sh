#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=aip-agrawal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -J stop_sft_tamia
#SBATCH -o /scratch/d/dars11/git/tar_reason_pm/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /scratch/d/dars11/git/tar_reason_pm/output_dir/slurm_logs/%x_%j.err

# TamIA port of bash_fir_stop_sft.sh (2026-10-27; account aip-agrawal, repo = worktree tar_reason_pm, extra d/ level).
# Fir port of bash_mila_stop_sft.sh (2026-10-24, jeet's run 2: critique distillation from the judge).
# Task 3a, stop-SFT: SFT of the critique on labelled images (llava/train/rl/stop_sft/).
#   Run 2 data: bash_fir_promptmix.sh MAX_STEPS=0 eval-only rollouts on a promptmix train subset with
#   EXTRA_ARGS="--init_adapter <ckpt> --eval_sample_critiques --val_dump_codes", then
#   python -m llava.train.rl.stop_sft.build_data --dumps output_dir/<run>/val_dump --targets oracle --ratio 0 --out $DATA
#   1. data: bash_mila_t7.sh eval-only rollouts with --eval_sample_critiques --val_dump_codes
#   2. labels: python -m llava.train.rl.stop_sft.build_data --dumps <dirs> --out $DATA
#   3. this script:  DATA=<jsonl> RUN_NAME=<run> sbatch scripts/rl_ft/bash_mila_stop_sft.sh
#   4. eval: bash_mila_t7.sh MAX_STEPS=0 with EXTRA_ARGS="--model_name_or_path $BASE --init_adapter
#      output_dir/<run>/final --eval_branch 1,1,1,1,1 --context_mode mdp ..."
# Optional: BASE (F merged at 250), LR (1e-5), EPOCHS (1), BATCH (16), MICRO (4).

set -uo pipefail
REPO=/scratch/d/dars11/git/tar_reason_pm
cd "$REPO"
BASE=${BASE:-/scratch/d/dars11/models/tar/sft/darshan_pm1_E_merged450}
DATA=${DATA:?set DATA (build_data.py output)}
RUN_NAME=${RUN_NAME:-stop_sft_tamia}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-1}
BATCH=${BATCH:-16}
MICRO=${MICRO:-4}
N_GPUS=${SLURM_GPUS_ON_NODE:-4}
export PYTHONPATH=${REPO}
export WANDB_PROJECT=tar_reasoning WANDB_ENTITY=darshan-singh
export WANDB_MODE=${WANDB_MODE:-offline}   # compute nodes have no internet; sync from a login node
mkdir -p output_dir/${RUN_NAME}
echo "RUN_NAME=${RUN_NAME} BASE=${BASE} DATA=${DATA} LR=${LR} EPOCHS=${EPOCHS} BATCH=${BATCH} job=${SLURM_JOB_ID:-local} host=$(hostname)"

source ~/envs/tar/bin/activate
module load cuda/12.2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node=${N_GPUS} \
    -m llava.train.rl.stop_sft.train_stop_sft \
    --model_name_or_path ${BASE} --data ${DATA} --output_dir output_dir/${RUN_NAME} \
    --learning_rate ${LR} --epochs ${EPOCHS} --batch_size ${BATCH} --micro_batch_size ${MICRO} \
    --report_to wandb --run_name ${RUN_NAME}
