#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --job-name=critic_hf
#SBATCH --partition short-unkillable
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/geneval_critique/cri_%j_%t.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/geneval_critique/cri_%j_%t.err

source ~/.bashrc
# The `vllm` env is the only one with a transformers new enough for Qwen3.6
# (`Qwen3_5ForConditionalGeneration`); inference below is plain HF, not vLLM.
mamba activate vllm
module load cuda/12.6.0

set -euo pipefail

MODEL=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-27B
INPUT_DIR=/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/conceptmix/prompts_13K_vqa_unverif_2/
OUTPUT_DIR=/home/mila/s/sarvjeet-singh.ghotra/scratch/data/syn_data_critique/conceptmix/prompts_13K_vqa_unverif_2_critic
MODE=${MODE:-Instruct}   # Instruct | Thinking (case-insensitive)

# mkdir -p "$OUTPUT_DIR"

python util_scripts/create_data/gen_critic/generate_critic_hf_parquet.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --mode "$MODE" \
    --batch_size 8 \
    --flush_rows 64 \
    --max_pixels 262144 \
    --mode Thinking \
    --seed 1119 \

# --samples 16 \

# Sampling parameters follow the Qwen3.6-27B model card and are picked by --mode:
#   Instruct: temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5
#   Thinking: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0
# Override any of them with --temperature/--top_p/--top_k/--min_p/
# --presence_penalty/--repetition_penalty/--max_new_tokens.
#
# Thinking mode (the default here) is much slower, since it generates a
# reasoning trace before the critique. For a faster, direct-answer pass:
#   MODE=Instruct sbatch util_scripts/create_data/gen_critic/critic_parquet_hf.sh
#
# Debugging: critique only N randomly drawn rows (reproducible via --seed):
#   python util_scripts/create_data/gen_critic/generate_critic_hf_parquet.py \
#       --input_dir "$INPUT_DIR" --output_dir ~/tmp/critique_dbg --model "$MODEL" \
#       --mode Instruct --samples 16 --seed 887 --batch_size 4
#
# Output is written incrementally: every --flush_rows input rows a finished
# part file (<shard>.rank<NN>.part<NNNNN>.parquet) lands in $OUTPUT_DIR and the
# per-shard <shard>.rank<NN>.progress.json is updated. A job killed at the 3 h
# wall clock therefore keeps everything generated so far, and re-submitting the
# same command resumes each shard at the first row it had not finished
# (--overwrite restarts them instead). To split the work across several jobs,
# run each with a different --shard_id and the same --num_shards, e.g.
#   --num_shards 4 --shard_id "$SLURM_PROCID"

mamba activate tar

python /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/util_scripts/sft_data_convert_format/viz/critique_rlhfv.py \
    $OUTPUT_DIR \
    $OUTPUT_DIR/viz \
