#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --open-mode=append
#SBATCH -J darshan_mr_fir
#SBATCH -o /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# First job of a bash_fir_mr.sh chain whose base model is a LoRA checkpoint still to be merged (2026-11-05: Fir's 1-GPU
# queue held a separate merge job for 20+ min while 4-node jobs started at once). On the first node (CPU), merge
# MERGE_ADAPTER onto MERGE_BASE into PREV_STAGE_CHECKPOINT (util_scripts/merge_lora.py) unless that directory is complete
# (merged_from.json is written last), then run bash_fir_mr.sh unchanged in the same allocation.
#   MERGE_BASE=<sft dir> MERGE_ADAPTER=<checkpoint dir> PREV_STAGE_CHECKPOINT=<out dir> ... sbatch scripts/rl_ft/merge_then_fir_mr.sh
set -eu
cd /home/dars11/scratch/git/tar_reason
: "${MERGE_BASE:?}" "${MERGE_ADAPTER:?}" "${PREV_STAGE_CHECKPOINT:?}"
if [ ! -s "${PREV_STAGE_CHECKPOINT}/merged_from.json" ]; then
    t0=$(date +%s)
    # On CPU, as the earlier Fir merges: with a GPU visible, save_pretrained imports DeepSpeed, which wants CUDA_HOME
    # (no CUDA module loaded here) and fails ("MissingCUDAException", job 2693645, 2026-11-05).
    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= DS_ACCELERATOR=cpu ~/envs/tar/bin/python util_scripts/merge_lora.py \
        --base "${MERGE_BASE}" --adapter "${MERGE_ADAPTER}" --out "${PREV_STAGE_CHECKPOINT}" --device cpu
    echo "merged ${MERGE_ADAPTER} onto ${MERGE_BASE} -> ${PREV_STAGE_CHECKPOINT} in $(( $(date +%s) - t0 )) s"
fi
exec bash scripts/rl_ft/bash_fir_mr.sh
