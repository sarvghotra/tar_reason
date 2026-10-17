#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0G
#SBATCH --cpus-per-task=32
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --job-name=critic_vllm
#SBATCH -o /home/jeet/scratch/data/syn_data_critique/conceptmix/cri_%A_%a.out
#SBATCH -e /home/jeet/scratch/data/syn_data_critique/conceptmix/cri_%A_%a.err

# Multi-node: submit as a job array, one node per array task, e.g.
#   sbatch --array=0-3 util_scripts/create_data/gen_critic/critic_parquet_vllm.sh
# Each task takes every NUM_SHARDS-th input file starting at SHARD_ID. Keep the
# array size fixed across resubmits, or resume will not find its progress files.

export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOB_ID   # local disk, not NFS
export TRITON_HOME=/tmp/triton_home_$SLURM_JOB_ID
# Compute nodes have no internet: fail fast on a bad local path instead of
# trying the Hub.
export HF_HUB_OFFLINE=1
PYTHON_BIN=/scratch/jeet/installs/miniforge3/envs/vllm/bin/python

type module >/dev/null 2>&1 || source /etc/profile
eval "$(/scratch/jeet/installs/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate vllm
module load cuda/12.9
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

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


MODEL=/home/jeet/scratch/models/pre_train/vlm/Qwen3.6-27B
COUNTER=/home/jeet/scratch/models/pre_train/caption_aug_models/sam3   # facebook/sam3 (SAM 3 detector counts)
INPUT_DIR=/home/jeet/scratch/data/syn_data_critique/conceptmix/all_97K_imgs
OUTPUT_DIR=/home/jeet/scratch/data/syn_data_critique/conceptmix/all_97K
# INPUT_DIR=/home/jeet/scratch/data/syn_data_critique/conceptmix/rand_samples
# OUTPUT_DIR=/home/jeet/scratch/data/syn_data_critique/conceptmix/rand_samples_critics
MODE=${MODE:-Instruct}   # Instruct | Thinking (case-insensitive)

# Qwen3.6-27B in bf16 is ~54 GB, so it fits one 80 GB H100: keep one replica
# per GPU (data-parallel) instead of sharding one engine over all four.
GPUS_PER_REPLICA=${GPUS_PER_REPLICA:-1}

# Without --array these fall back to a single shard (the whole input).
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_ID=${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}

# mkdir -p "$OUTPUT_DIR"

cd /home/jeet/scratch/git/tar_reason

python util_scripts/create_data/gen_critic/generate_critic_vllm_paraquet.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --counter_path "$COUNTER" \
    --mode "$MODE" \
    --gpus_per_replica "$GPUS_PER_REPLICA" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$SHARD_ID" \
    --batch_size 256 \
    --flush_rows 256 \
    --max_num_seqs 64 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 8192 \
    --max_pixels 262144 \
    --seed 1119

# --samples 16 \


# Only one array task renders the viz; it shows whatever the other shards have
# written so far, so rerun it by hand once the whole array has finished.

# if [ "$SHARD_ID" -ne 0 ]; then
#     exit 0
# fi

# mamba activate tar

# python util_scripts/sft_data_convert_format/viz/critique_rlhfv.py \
#     $OUTPUT_DIR \
#     $OUTPUT_DIR/viz
