#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=decompose_refl_corr_vllm
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/decompose_refl_corr_vllm_%j.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/decompose_refl_corr_vllm_%j.err

# Decompose Self-reflect / Correction text into numbered atomic steps with Qwen3.6-27B on vLLM.
# One vLLM engine per TP_SIZE GPUs (default 1, see below): rank i generates rows i, i+N, ... then rewrites shards i, i+N, ... in place
# ("conversations" -> "v1-conversations", decomposed text -> "conversations"). Shares its cache with the HF script.
#
# Quick look before a full run (prints before/after, writes nothing):
#   EXTRA_ARGS="--dry_run --limit 24" bash decompose_reflection_correction_vllm.sh
# Write to a copy instead of in place:
#   EXTRA_ARGS="--output_dir ~/scratch/data/edit/decomposed" bash decompose_reflection_correction_vllm.sh
#
# Full-size runs (hqedt ~190k, UnicEdit ~1M rows): spread generation over a job array, then rewrite once on CPU.
# Every array task takes global ranks JOB_INDEX * engines_per_job + local rank; they share the cache, and a killed or
# timed-out task resumes from it when resubmitted with the same array size.
#   sbatch --array=0-3 --export=ALL,STAGE=generate,EXTRA_ARGS="--input_dirs <dir>" decompose_reflection_correction_vllm.sh
#   STAGE=rewrite EXTRA_ARGS="--input_dirs <dir> --rewrite_workers 16" bash decompose_reflection_correction_vllm.sh
# (the rewrite stage loads no model and needs no GPU.)
#
# TP_SIZE: measured on 4x A100-80G, TP 1 (4 engines) 13.5 rows/s vs TP 2 (2 engines) 12.7 rows/s, so the default is 1.

source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
# Only the rebuilt `vllm` env loads Qwen3.6.
mamba activate vllm
module load cuda/12.6.0

set -euo pipefail

cd /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/util_scripts/create_data/reduce_gap_w_ood

# Triton / torch.compile caches on node-local disk, not NFS (one per rank below).
JOB_TAG=${SLURM_JOB_ID:-$$}
export TRITON_CACHE_DIR=/tmp/triton_cache_${JOB_TAG}
export TRITON_HOME=/tmp/triton_home_${JOB_TAG}

# The env lives on /network/scratch, whose files can lag behind on a fresh node: an import then dies with "cannot open
# shared object file". Import everything the run loads (through vLLM's own loader; a bare `import vllm._C` aborts
# on duplicate op registration) until it works, which also pulls the files into the node's cache before the ranks
# start. ~35 s when the files are already visible.
ENV_RETRIES=${ENV_RETRIES:-5}
ENV_RETRY_WAIT=${ENV_RETRY_WAIT:-60}
for (( TRY = 1; ; TRY++ )); do
    if python -c "import torch, transformers, pyarrow; from vllm import LLM; import vllm._custom_ops" >/dev/null 2>&1; then
        break
    fi
    if (( TRY >= ENV_RETRIES )); then
        echo "ERROR: the vllm env still fails to import after ${TRY} tries:" >&2
        python -c "import torch, transformers, pyarrow; from vllm import LLM; import vllm._custom_ops" 2>&1 | grep -v WARNING | tail -5 >&2
        exit 1
    fi
    echo "vllm env import failed (try ${TRY}/${ENV_RETRIES}), retrying in ${ENV_RETRY_WAIT} s" >&2
    sleep "${ENV_RETRY_WAIT}"
done
echo "vllm env imports OK"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
TP_SIZE=${TP_SIZE:-1}
if (( ${#GPUS[@]} % TP_SIZE )); then
    echo "ERROR: ${#GPUS[@]} GPU(s) is not a multiple of TP_SIZE=${TP_SIZE}" >&2
    exit 1
fi
WORLD_SIZE=$(( ${#GPUS[@]} / TP_SIZE ))
echo "Using ${#GPUS[@]} GPU(s): ${GPUS[*]} -> ${WORLD_SIZE} engine(s) x TP ${TP_SIZE}"






EXTRA_ARGS=${EXTRA_ARGS:-}
STAGE=${STAGE:-all}

EXTRA_ARGS="--input_dirs /home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt_slfref_tar/small_sample --rewrite_workers 8"
MAX_NUM_SEQS=256



if [ "${STAGE}" = "rewrite" ]; then
    # Shard rewrite from the cache only: one CPU process, all shards.
    exec python decompose_reflection_correction_vllm.py --stage rewrite --rank 0 --world_size 1 ${EXTRA_ARGS}
fi

# Job array: this task's engines take global ranks JOB_INDEX * WORLD_SIZE ... + WORLD_SIZE - 1.
NUM_JOBS=${NUM_JOBS:-${SLURM_ARRAY_TASK_COUNT:-1}}
JOB_INDEX=${JOB_INDEX:-$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))}
if (( NUM_JOBS > 1 )) && [ "${STAGE}" = "all" ]; then
    echo "WARNING: STAGE=all over ${NUM_JOBS} jobs waits until every task has generated; prefer STAGE=generate" >&2
fi
GLOBAL_WORLD_SIZE=$(( NUM_JOBS * WORLD_SIZE ))
echo "Job ${JOB_INDEX}/${NUM_JOBS}: global ranks $(( JOB_INDEX * WORLD_SIZE ))-$(( (JOB_INDEX + 1) * WORLD_SIZE - 1 )) of ${GLOBAL_WORLD_SIZE}, stage ${STAGE}"

# A rank that still dies on a missing shared object (the lag above hitting a library the preflight did not import) is
# restarted after a wait; generation resumes from the cache. Each attempt's output is also kept in RANK_LOG_DIR.
RANK_RETRIES=${RANK_RETRIES:-3}
RANK_LOG_DIR=${RANK_LOG_DIR:-/home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/decompose_refl_corr_vllm_${JOB_TAG}}
LAUNCH_STAGGER=${LAUNCH_STAGGER:-10}
mkdir -p "${RANK_LOG_DIR}"





run_rank() {
    local rank=$1 group=$2 try status log
    for (( try = 1; try <= RANK_RETRIES; try++ )); do
        log=${RANK_LOG_DIR}/rank${rank}_try${try}.log
        set +e
        CUDA_VISIBLE_DEVICES=${group} \
        TRITON_CACHE_DIR=${TRITON_CACHE_DIR}_rank${rank} \
        TRITON_HOME=${TRITON_HOME}_rank${rank} \
        LOCAL_RANK=${rank} \
        python decompose_reflection_correction_vllm.py \
            --rank $(( JOB_INDEX * WORLD_SIZE + rank )) \
            --world_size ${GLOBAL_WORLD_SIZE} \
            --tensor_parallel_size ${TP_SIZE} \
            --stage ${STAGE} \
             --keep_junk \
             --max_num_seqs ${MAX_NUM_SEQS} \
            ${EXTRA_ARGS} 2>&1 | tee "${log}"
        status=${PIPESTATUS[0]}
        set -e
        (( status == 0 )) && return 0
        # vLLM prints a harmless "libcudart.so.13: cannot open shared object file" WARNING on every start; only an
        # error line counts.
        if ! grep -v WARNING "${log}" | grep -q "cannot open shared object file"; then
            return "${status}"
        fi
        echo "rank ${rank}: missing shared object (try ${try}/${RANK_RETRIES}), restarting in ${ENV_RETRY_WAIT} s" >&2
        sleep "${ENV_RETRY_WAIT}"
    done
    return "${status}"
}



PIDS=()
for (( RANK = 0; RANK < WORLD_SIZE; RANK++ )); do
    GROUP=$(IFS=','; echo "${GPUS[*]:$(( RANK * TP_SIZE )):${TP_SIZE}}")
    mkdir -p "${TRITON_CACHE_DIR}_rank${RANK}" "${TRITON_HOME}_rank${RANK}"
    run_rank "${RANK}" "${GROUP}" &
    PIDS+=($!)
    # Ranks that start together all read the model and the env's libraries off the shared filesystem at once.
    (( RANK + 1 < WORLD_SIZE )) && sleep "${LAUNCH_STAGGER}"
done

STATUS=0
for PID in "${PIDS[@]}"; do
    wait ${PID} || STATUS=$?
done
exit ${STATUS}
