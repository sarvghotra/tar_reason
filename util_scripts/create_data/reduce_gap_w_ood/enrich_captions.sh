#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=enrich_captions
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/enrich_captions_%j.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/enrich_captions_%j.err

# Enrich captions with detector/depth-grounded counts, relations, attributes and negations
# (Qwen3.6-27B via HF Transformers + SAM 3, Grounding DINO, Depth Anything V2, SigLIP2; no vLLM).
# One process per GPU: rank i processes row groups i, i+N, ... then rewrites shards i, i+N, ... in place
# ("conversations" -> "v1-conversations", enriched captions -> "conversations"; same for the short column if any).
#
# Quick look before a full run (prints facts and before/after, writes no shards):
#   CUDA_VISIBLE_DEVICES=0 EXTRA_ARGS="--dry_run --limit 16 --viz_dir ~/tmp/enrich_viz" bash enrich_captions.sh
# Write to a copy instead of in place:
#   EXTRA_ARGS="--output_dir ~/scratch/data/edit/enriched" bash enrich_captions.sh
# Add a prompt-style short caption column to data that has none (the loader samples it half the time):
#   EXTRA_ARGS="--synth_short" bash enrich_captions.sh

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
BATCH_SIZE=${BATCH_SIZE:-8}
# Fragmentation from variable-length batches otherwise triggers OOM well below 80G.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PIDS=()
for RANK in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$RANK]} python enrich_captions.py \
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
