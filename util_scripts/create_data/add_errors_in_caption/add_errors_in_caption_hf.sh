#!/bin/bash

source ~/.bashrc
mamba activate tar
module load cuda/12.6.0

set -Eeuo pipefail


python \
    util_scripts/create_data/add_errors_in_caption/add_errors_in_caption_hf.py \
    --input-dir ~/scratch/data/edit/gpt-edit-simpler_hqedt_slfref_tar/train \
    --output-dir ~/scratch/data/edit/caption_error/gpt-edit-simpler_hqedt_slfref_tar/train \
    --model-path /home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-35B-A3B \
    --batch-size 16 \
    --max-errors 2


# NOTE: Qwen3.6-35B-A3B is `model_type: qwen3_5_moe`, which no env here registers yet
# (tar 4.50.2, geneval2 4.57.0) -- the script fails fast with that message. Either
# upgrade transformers to a 5.x release in this env, or point --model-path at a
# checkpoint the installed release supports, e.g. Qwen3.5-35B-A3B:
#   --model-path /network/scratch/s/sarvjeet-singh.ghotra/models/pre_train/Qwen3.5-35B-A3B
#
# Debug a couple of batches first:  --batch-size 4  on a directory with one small shard.
# Faster:                           --attn-implementation flash_attention_2 (bf16)
# Greedy instead of the card's instruct-mode sampling: --temperature 0
# Multi-job data parallel:          --rank $i --world-size $N (each rank takes files i, i+N, ...)
