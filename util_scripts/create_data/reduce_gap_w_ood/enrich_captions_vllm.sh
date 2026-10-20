#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=hq_enrich_captions_vllm
#SBATCH --partition=long
#SBATCH -o /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/enrich_captions_vllm_%j.out
#SBATCH -e /home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/enrich_captions_vllm_%j.err

# enrich_captions.py at scale: Qwen3.6-27B behind `vllm serve`, perception in worker processes on the same GPUs.
#   - SERVERS vLLM servers, TP_SIZE GPUs each (default 2 x TP 2 on 4 GPUs), SERVER_UTIL of each GPU's memory.
#   - WORKERS_PER_GPU worker processes per GPU in the memory left over. Each is one rank of the row-group sharding
#     (global rank = JOB_INDEX * workers + local), loads only the perception models, and sends each stage's requests
#     for its whole chunk to all servers at once. While one worker runs perception the others keep the servers busy.
#
# Quick look (prints facts and before/after, writes no shards):
#   EXTRA_ARGS="--dry_run --limit 16 --viz_dir ~/tmp/enrich_viz" bash enrich_captions_vllm.sh
# Full-size runs: generate over a job array (tasks resume from the shared cache when resubmitted with the same array
# size), then rewrite once on CPU (no GPU, no servers):
#   sbatch --array=0-7 --export=ALL,STAGE=generate,EXTRA_ARGS="--input_dirs <dir>" enrich_captions_vllm.sh
#   STAGE=rewrite EXTRA_ARGS="--input_dirs <dir> --rewrite_workers 16" bash enrich_captions_vllm.sh
# Compare with the unbatched perception of the HF script: EXTRA_ARGS="... --no_batch_perception".

# STAGE=rewrite EXTRA_ARGS="--viz_diz /home/mila/s/sarvjeet-singh.ghotra/scratch/tmp_enrich_review/vllm/ --input_dirs /home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt_slfref_tar/small_sample /home/mila/s/sarvjeet-singh.ghotra/scratch/data/edit/gpt-edit-simpler_hqedt2_slfref_tar/small_sample --rewrite_workers 8" bash enrich_captions_vllm.sh

source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
# Only the rebuilt `vllm` env loads Qwen3.6.
mamba activate vllm
module load cuda/12.6.0

set -euo pipefail

cd /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/util_scripts/create_data/reduce_gap_w_ood

MODEL=${MODEL:-/home/mila/s/sarvjeet-singh.ghotra/scratch/models/pre_train/Qwen3.6-27B}
EXTRA_ARGS=${EXTRA_ARGS:-}
STAGE=${STAGE:-all}
CHUNK_SIZE=${CHUNK_SIZE:-128}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}

if [ "${STAGE}" = "rewrite" ]; then
    # Shard rewrite from the cache only: one CPU process (plus --rewrite_workers), all shards.
    exec python enrich_captions_vllm.py --stage rewrite --rank 0 --world_size 1 --v1_prefix v2- ${EXTRA_ARGS}
fi

# Triton / torch.compile caches on node-local disk, not NFS.
JOB_TAG=${SLURM_JOB_ID:-$$}
export TRITON_CACHE_DIR=/tmp/triton_cache_${JOB_TAG}
export TRITON_HOME=/tmp/triton_home_${JOB_TAG}
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${JOB_TAG}
# vLLM's compile cache (default ~/.cache/vllm) goes to node-local disk, one per server: a full home quota kills server
# startup ("Disk quota exceeded"), and servers sharing one directory on NFS read each other's half-written files.
VLLM_CACHE_BASE=/tmp/vllm_cache_${JOB_TAG}
# Fragmentation from variable-size perception batches otherwise triggers OOM well below the free memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The env lives on /network/scratch, whose files can lag behind on a fresh node: an import then dies with "cannot open
# shared object file". Import everything the run loads until it works (see decompose_reflection_correction_vllm.sh).
ENV_RETRIES=${ENV_RETRIES:-5}
ENV_RETRY_WAIT=${ENV_RETRY_WAIT:-60}
PREFLIGHT="import torch, transformers, pyarrow, openai, httpx; from vllm import LLM; import vllm._custom_ops"
# Triton also compiles a small C helper against the env's Python headers whenever a server meets a new kernel shape,
# mid-run included, and a header lagging on NFS then kills that server ("cpython/code.h: No such file or directory").
# So the env and the editable vLLM source it imports are copied to node-local disk ($SLURM_TMPDIR, ~11 GB) and run
# from there. LOCAL_ENV=0 runs straight off /network/scratch.
SCRATCH_ENV=${CONDA_PREFIX}
VLLM_SRC=/network/scratch/s/sarvjeet-singh.ghotra/installs/vllm/vllm
if [ "${LOCAL_ENV:-1}" = "1" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    LOCAL_PREFIX=${SLURM_TMPDIR}/envs/vllm
    LOCAL_VLLM_SRC=${SLURM_TMPDIR}/vllm_src/vllm
    stage_env() {
        mkdir -p "${LOCAL_PREFIX}" "${LOCAL_VLLM_SRC}"
        rsync -a "${SCRATCH_ENV}/" "${LOCAL_PREFIX}/" && rsync -a "${VLLM_SRC}/" "${LOCAL_VLLM_SRC}/" || return 1
        # The editable install finds vllm through absolute paths in its finder module; point them at the copy.
        sed -i "s#${VLLM_SRC}#${LOCAL_VLLM_SRC}#g" "${LOCAL_PREFIX}"/lib/python3.10/site-packages/__editable___vllm_*_finder.py
    }
    STAGED=${SECONDS}
    stage_env || true
    set +u; mamba activate "${LOCAL_PREFIX}"; set -u  # activation scripts read unset variables
    # The run must import vllm from the copy, not from /network/scratch.
    PREFLIGHT+="; assert vllm.__file__.startswith('${LOCAL_VLLM_SRC}/'), vllm.__file__"
    echo "vllm env copied to ${LOCAL_PREFIX} in $(( SECONDS - STAGED )) s"
else
    stage_env() { :; }
fi
# Triton builds that helper when a server starts too; a lagging header fails the build ("bltinmodule.h: No such file
# or directory") and kills the server. Preprocess Python.h (it pulls in every header) with the same compiler until it
# works; reading the headers once with cat was not enough.
PY_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")
headers_ok() { echo '#include <Python.h>' | gcc -E -I"${PY_INCLUDE}" - >/dev/null 2>&1; }
for (( TRY = 1; ; TRY++ )); do
    if python -c "${PREFLIGHT}" >/dev/null 2>&1 && headers_ok; then
        break
    fi
    if (( TRY >= ENV_RETRIES )); then
        echo "ERROR: the vllm env still fails to import or compile against its headers after ${TRY} tries:" >&2
        python -c "${PREFLIGHT}" 2>&1 | grep -v WARNING | tail -5 >&2
        echo '#include <Python.h>' | gcc -E -I"${PY_INCLUDE}" - 2>&1 >/dev/null | tail -5 >&2
        exit 1
    fi
    echo "vllm env import failed (try ${TRY}/${ENV_RETRIES}), retrying in ${ENV_RETRY_WAIT} s" >&2
    sleep "${ENV_RETRY_WAIT}"
    stage_env || true  # copies what lagged on the last try
done
echo "vllm env imports OK from ${CONDA_PREFIX}"
# Triton builds a small C helper against the env's Python headers when a server starts; read them once so they are
# visible on this node too (a lagging header fails that build with "namespaceobject.h: No such file or directory").
find "${CONDA_PREFIX}/include" -type f -exec cat {} + > /dev/null

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
TP_SIZE=${TP_SIZE:-2}
SERVERS=${SERVERS:-$(( ${#GPUS[@]} / TP_SIZE ))}
# 0.5 of an A100-80G at TP 2: 26 GB of weights per GPU and still ~250k tokens of KV cache per server (the hybrid model
# keeps KV for 16 of its 64 layers), far above what the workers keep in flight; the other ~40 GB per GPU are theirs.
SERVER_UTIL=${SERVER_UTIL:-0.5}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3}
if (( SERVERS * TP_SIZE > ${#GPUS[@]} )); then
    echo "ERROR: ${SERVERS} server(s) x TP ${TP_SIZE} needs more than the ${#GPUS[@]} GPU(s) visible" >&2
    exit 1
fi
WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
# CPU threads per worker: the default (one per core in every process) oversubscribes the node, and the image
# preprocessing of the perception models runs on the CPU.
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( CPUS / WORKERS > 1 ? CPUS / WORKERS : 1 ))}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
# Ports from the job id, so two jobs sharing a node do not collide.
BASE_PORT=${BASE_PORT:-$(( 20000 + (JOB_TAG % 2000) * 8 ))}

NUM_JOBS=${NUM_JOBS:-${SLURM_ARRAY_TASK_COUNT:-1}}
JOB_INDEX=${JOB_INDEX:-$(( ${SLURM_ARRAY_TASK_ID:-0} - ${SLURM_ARRAY_TASK_MIN:-0} ))}
GLOBAL_WORLD_SIZE=$(( NUM_JOBS * WORKERS ))
if (( NUM_JOBS > 1 )) && [ "${STAGE}" = "all" ]; then
    echo "WARNING: STAGE=all over ${NUM_JOBS} jobs waits until every task has generated; prefer STAGE=generate" >&2
fi
echo "GPUs ${GPUS[*]}: ${SERVERS} server(s) x TP ${TP_SIZE} at ${SERVER_UTIL} memory, ${WORKERS} worker(s);" \
     "job ${JOB_INDEX}/${NUM_JOBS}, global ranks $(( JOB_INDEX * WORKERS ))-$(( (JOB_INDEX + 1) * WORKERS - 1 )) of ${GLOBAL_WORLD_SIZE}, stage ${STAGE}"

LOG_DIR=${LOG_DIR:-/home/mila/s/sarvjeet-singh.ghotra/tmp/tmp/enrich_captions_vllm_${JOB_TAG}}
mkdir -p "${LOG_DIR}"

# ------------------------------------------------------------------------------------------------ servers
SERVER_PIDS=()
URLS=()
cleanup() {
    for PID in "${SERVER_PIDS[@]}"; do
        kill -- "-${PID}" 2>/dev/null || true  # the whole process group: vLLM's engine and TP workers too
    done
    sleep 10
    for PID in "${SERVER_PIDS[@]}"; do  # a server still loading ignores SIGTERM
        kill -9 -- "-${PID}" 2>/dev/null || true
    done
}
trap cleanup EXIT

# The custom all-reduce kernel fails CUDA graph capture on these nodes ("custom_all_reduce.cuh:164 'invalid
# argument'"); NCCL is used instead.
SERVER_EXTRA_ARGS=${SERVER_EXTRA_ARGS:-$( (( TP_SIZE > 1 )) && echo --disable-custom-all-reduce )}

start_server() {
    local s=$1 try=$2
    local group port
    group=$(IFS=','; echo "${GPUS[*]:$(( s * TP_SIZE )):${TP_SIZE}}")
    port=$(( BASE_PORT + s ))
    mkdir -p "${TRITON_CACHE_DIR}_server${s}" "${TRITON_HOME}_server${s}"
    CUDA_VISIBLE_DEVICES=${group} TRITON_CACHE_DIR=${TRITON_CACHE_DIR}_server${s} TRITON_HOME=${TRITON_HOME}_server${s} \
    VLLM_CACHE_ROOT=${VLLM_CACHE_BASE}_server${s} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}_server${s} \
    setsid python -m vllm.entrypoints.cli.main serve "${MODEL}" \
        --port "${port}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --gpu-memory-utilization "${SERVER_UTIL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --limit-mm-per-prompt '{"image": 8, "video": 0}' \
        --enable-prefix-caching \
        --trust-remote-code \
        --seed 0 \
        ${SERVER_EXTRA_ARGS} > "${LOG_DIR}/server${s}_try${try}.log" 2>&1 &
    SERVER_PIDS[$s]=$!
    URLS[$s]="http://localhost:${port}"
    echo "server ${s} (try ${try}): GPUs ${group}, port ${port}, log ${LOG_DIR}/server${s}_try${try}.log"
}

# VLLM_URLS="http://host:port ..." uses servers that are already up (e.g. while iterating) instead of starting any.
if [ -n "${VLLM_URLS:-}" ]; then
    read -ra URLS <<< "${VLLM_URLS}"
    SERVERS=0
fi
for (( S = 0; S < SERVERS; S++ )); do
    start_server "${S}" 1
done

# A server that dies during startup (the NFS lag again) is started again, up to SERVER_RETRIES times.
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-1800}
SERVER_RETRIES=${SERVER_RETRIES:-3}
STARTED=${SECONDS}
for (( S = 0; S < SERVERS; S++ )); do
    TRY=1
    until curl -sf "${URLS[$S]}/health" >/dev/null 2>&1; do
        if ! kill -0 "${SERVER_PIDS[$S]}" 2>/dev/null; then
            echo "server ${S} exited during startup (try ${TRY}/${SERVER_RETRIES}):" >&2
            grep -v WARNING "${LOG_DIR}/server${S}_try${TRY}.log" | grep -E "Error|error" | tail -5 >&2
            if (( TRY >= SERVER_RETRIES )); then
                echo "ERROR: server ${S} failed ${TRY} times" >&2
                exit 1
            fi
            sleep 30
            TRY=$(( TRY + 1 ))
            start_server "${S}" "${TRY}"
        fi
        if (( SECONDS - STARTED > SERVER_START_TIMEOUT )); then
            echo "ERROR: server ${S} not healthy after ${SERVER_START_TIMEOUT} s:" >&2
            tail -30 "${LOG_DIR}/server${S}_try${TRY}.log" >&2
            exit 1
        fi
        sleep 10
    done
done
(( SERVERS > 0 )) && echo "all ${SERVERS} server(s) healthy after $(( SECONDS - STARTED )) s"
# SERVERS_ONLY=1: keep the servers up for workers started elsewhere with VLLM_URLS (iterating without the ~5 min start).
if [ -n "${SERVERS_ONLY:-}" ]; then
    echo "VLLM_URLS=\"${URLS[*]}\""
    wait "${SERVER_PIDS[@]}"
    exit 0
fi

# ------------------------------------------------------------------------------------------------ workers
# A worker that dies on a missing shared object (the NFS lag above) is restarted; it resumes from the cache.
RANK_RETRIES=${RANK_RETRIES:-3}
LAUNCH_STAGGER=${LAUNCH_STAGGER:-10}

run_rank() {
    local rank=$1 gpu=$2 try status log
    for (( try = 1; try <= RANK_RETRIES; try++ )); do
        log=${LOG_DIR}/rank${rank}_try${try}.log
        set +e
        CUDA_VISIBLE_DEVICES=${gpu} \
        python enrich_captions_vllm.py \
            --rank $(( JOB_INDEX * WORKERS + rank )) \
            --world_size ${GLOBAL_WORLD_SIZE} \
            --stage ${STAGE} \
            --vllm_urls ${URLS[*]} \
            --chunk_size ${CHUNK_SIZE} \
            --max_concurrency ${MAX_CONCURRENCY} \
            --v1_prefix "v2-" \
            ${EXTRA_ARGS} > "${log}" 2>&1
        status=$?
        set -e
        (( status == 0 )) && return 0
        if ! grep -v WARNING "${log}" | grep -q "cannot open shared object file"; then
            echo "rank ${rank} failed (exit ${status}), see ${log}:" >&2
            tail -20 "${log}" >&2
            return "${status}"
        fi
        echo "rank ${rank}: missing shared object (try ${try}/${RANK_RETRIES}), restarting in ${ENV_RETRY_WAIT} s" >&2
        sleep "${ENV_RETRY_WAIT}"
    done
    return "${status}"
}

PIDS=()
for (( RANK = 0; RANK < WORKERS; RANK++ )); do
    run_rank "${RANK}" "${GPUS[$(( RANK % ${#GPUS[@]} ))]}" &
    PIDS+=($!)
    # Workers that start together all read the perception models off the shared filesystem at once.
    (( RANK + 1 < WORKERS )) && sleep "${LAUNCH_STAGGER}"
done
echo "worker logs: ${LOG_DIR}/rank*_try*.log"

STATUS=0
for PID in "${PIDS[@]}"; do
    wait ${PID} || STATUS=$?
done
exit ${STATUS}
