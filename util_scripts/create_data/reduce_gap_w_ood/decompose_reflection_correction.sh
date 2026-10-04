#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=decompose_refl_corr
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/decompose_refl_corr_%j.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/decompose_refl_corr_%j.err

# Decompose Self-reflect / Correction text into numbered atomic steps with Qwen3.6-27B (HF Transformers, no vLLM).
# One process per GPU: rank i generates rows i, i+N, ... then rewrites shards i, i+N, ... in place
# ("conversations" -> "v1-conversations", decomposed text -> "conversations").
#
# Quick look before a full run (prints before/after, writes nothing):
#   EXTRA_ARGS="--dry_run --limit 24" bash decompose_reflection_correction.sh
# Write to a copy instead of in place:
#   EXTRA_ARGS="--output_dir ~/scratch/data/edit/decomposed" bash decompose_reflection_correction.sh

source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
# Only the rebuilt `vllm` env has a Transformers release (5.14) that loads Qwen3.6; vLLM itself is not used.
mamba activate vllm

set -euo pipefail

cd /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/util_scripts/create_data/reduce_gap_w_ood

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
WORLD_SIZE=${#GPUS[@]}
echo "Using ${WORLD_SIZE} GPU(s): ${GPUS[*]}"

EXTRA_ARGS=${EXTRA_ARGS:-}
BATCH_SIZE=${BATCH_SIZE:-12}
# Fragmentation from variable-length batches otherwise triggers OOM well below 80G.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PIDS=()
for RANK in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$RANK]} python decompose_reflection_correction.py \
        --rank ${RANK} \
        --world_size ${WORLD_SIZE} \
        --batch_size ${BATCH_SIZE} \
        ${EXTRA_ARGS} &
    PIDS+=($!)
done

STATUS=0
for PID in "${PIDS[@]}"; do
    wait ${PID} || STATUS=$?
done
exit ${STATUS}
