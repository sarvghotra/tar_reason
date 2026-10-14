#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --job-name=critic_vllm
#SBATCH --partition long
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/geneval_critique/cric_logs/cri_%A_%a.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/geneval_critique/cric_logs/cri_%A_%a.err

# Multi-node: submit as a job array, one node per array task, e.g.
#   sbatch --array=0-3 util_scripts/create_data/gen_critic/critic_parquet_vllm.sh
# Each task takes every NUM_SHARDS-th input file starting at SHARD_ID. Keep the
# array size fixed across resubmits, or resume will not find its progress files.

export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID   # local disk, not NFS
export TRITON_HOME=/tmp/triton_home_$SLURM_JOB_ID
PYTHON_BIN=/network/scratch/s/sarvjeet-singh.ghotra/installs/miniforge3/envs/vllm/bin/python

source ~/.bashrc
mamba activate vllm
module load cuda/12.6.0

set -euo pipefail

mkdir -p "$TRITON_CACHE_DIR" "$TRITON_HOME"

# Triton compiles launcher extensions at runtime. Stage the Python headers on
# node-local storage so concurrent workers do not depend on metadata reads from
# the network-mounted conda environment during compilation.
PYTHON_INCLUDE="$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
LOCAL_PY_INCLUDE="/tmp/python_include_${SLURM_JOB_ID}_$$"
if [ ! -f "$PYTHON_INCLUDE/Python.h" ] || [ ! -f "$PYTHON_INCLUDE/cpython/unicodeobject.h" ]; then
    echo "ERROR: incomplete Python headers in active env: $PYTHON_INCLUDE" >&2
    exit 1
fi
cp -a "$PYTHON_INCLUDE" "$LOCAL_PY_INCLUDE"
export CPATH="$LOCAL_PY_INCLUDE${CPATH:+:$CPATH}"
printf '#include <Python.h>\n' |
    gcc -E -I"$LOCAL_PY_INCLUDE" -x c - >/dev/null || {
        echo "ERROR: gcc cannot preprocess node-local Python headers: $LOCAL_PY_INCLUDE" >&2
        exit 1
    }


MODEL=/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-27B
INPUT_DIR=/home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/geneval_critique/imgs_7B/
OUTPUT_DIR=/home/mila/s/sarvjeet-singh.ghotra/scratch/data/critique/geneval_critique/crit_7B
MODE=${MODE:-Thinking}   # Instruct | Thinking (case-insensitive)

# Qwen3.6-27B in bf16 is ~54 GB, so it fits one 80 GB A100: keep one replica
# per GPU (data-parallel) instead of sharding one engine over all four.
GPUS_PER_REPLICA=${GPUS_PER_REPLICA:-1}

# Without --array these fall back to a single shard (the whole input).
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_ID=${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}

# mkdir -p "$OUTPUT_DIR"

python util_scripts/create_data/gen_critic/generate_critic_vllm_paraquet.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --mode "$MODE" \
    --gpus_per_replica "$GPUS_PER_REPLICA" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$SHARD_ID" \
    --batch_size 256 \
    --flush_rows 256 \
    --max_num_seqs 64 \
    --gpu_memory_utilization 0.90 \
    --max_model_len 8192 \
    --max_pixels 262144 \
    --seed 1119

# --samples 16 \


# Only one array task renders the viz; it shows whatever the other shards have
# written so far, so rerun it by hand once the whole array has finished.
if [ "$SHARD_ID" -ne 0 ]; then
    exit 0
fi

mamba activate tar

python /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/util_scripts/sft_data_convert_format/viz/critique_rlhfv.py \
    $OUTPUT_DIR \
    $OUTPUT_DIR/viz
