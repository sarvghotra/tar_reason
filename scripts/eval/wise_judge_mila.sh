#!/bin/bash
# Score iterative runs on WISE_Verified with its official open judge, unchanged: WISE's vllm_eval.py (judge prompt,
# temperature 0, binary score) + calculate_verified.py, against Qwen3.5-35B-A3B served by `vllm serve` (TP=4,
# --served-model-name Qwen3.5-35B-A3B, as WISE's README), in jeet's vllm env. For every iteration k of every seed
# dir given, the images <seed>/<k>/<NNNNN>/samples/00000.png (line NNNNN of wise_verified_prompts.jsonl = prompt_id
# NNNNN+1) are linked as <seed>/<k>/wise_images/<prompt_id>.png, WISE's eval_qwen.sh runs on that folder
# (results in wise_images/Results-qwen35/), and calculate_verified.py's report goes to <seed>/<k>/wise_scores.txt.
# Iterations with wise_scores.txt are skipped (re-submit resumes).
#
#   sbatch scripts/eval/wise_judge_mila.sh <seed dir> [<seed dir> ...]
#
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --constraint=80gb
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bj_wise_judge
#SBATCH --exclude=cn-g005,cn-g011
#SBATCH --partition=short-unkillable
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out

set -e
WISE=/network/scratch/s/singhsd/benchmarks/WISE
JUDGE=$(ls -d /network/scratch/s/singhsd/hf_home/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/* | head -1)
PORT=${PORT:-8124}
JEET=/network/scratch/s/sarvjeet-singh.ghotra
export CONDA_PREFIX=${JEET}/installs/miniforge3/envs/vllm
export PATH=${CONDA_PREFIX}/bin:${PATH} LD_LIBRARY_PATH=${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
module load cuda/12.6.0
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/network/scratch/s/singhsd/.cache/vllm TRITON_CACHE_DIR=/tmp/triton_wise_${SLURM_JOB_ID:-$$}
unset PYTHONPATH
# jeet's workaround: Triton compiles against a node-local copy of the Python headers (from our copy: reading his env
# failed on some nodes)
PY_INC=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
LOCAL_PY_INC=/tmp/py_include_wise_${SLURM_JOB_ID:-$$}
for TRY in 1 2 3 4 5; do
  rm -rf ${LOCAL_PY_INC}
  cp -r /network/scratch/s/singhsd/jeet_snapshot/py_include_vllm ${LOCAL_PY_INC} 2>/dev/null && [ -f ${LOCAL_PY_INC}/Python.h ] && break
  sleep 20
done
export CC=/tmp/triton_cc_wise_${SLURM_JOB_ID:-$$}.sh
cat > ${CC} <<EOS
#!/bin/bash
ARGS=()
for A in "\$@"; do [ "\$A" = "-I${PY_INC}" ] && A="-I${LOCAL_PY_INC}"; ARGS+=("\$A"); done
exec /usr/bin/gcc "\${ARGS[@]}"
EOS
chmod +x ${CC}

LOG=/network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/wise_judge_server_${SLURM_JOB_ID:-local}.log
for TRY in 1 2 3; do
  setsid python -m vllm.entrypoints.cli.main serve ${JUDGE} --served-model-name Qwen3.5-35B-A3B --port ${PORT} \
    --tensor-parallel-size 4 --disable-custom-all-reduce --max-model-len 16384 > ${LOG}.try${TRY} 2>&1 &
  SERVER=$!
  STARTED=${SECONDS}
  until curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; do
    if ! kill -0 ${SERVER} 2>/dev/null || (( SECONDS - STARTED > 2400 )); then break; fi
    sleep 15
  done
  curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 && break
  echo "judge server failed to start (try ${TRY}):"; grep -v shm_broadcast ${LOG}.try${TRY} | tail -15
  kill -9 -- -${SERVER} 2>/dev/null || true; sleep 30
done
curl -sf http://localhost:${PORT}/health >/dev/null 2>&1 || { echo "ERROR: judge server not up"; exit 1; }
trap 'kill -- -${SERVER} 2>/dev/null || true' EXIT
echo "$(date +%H:%M:%S) judge up"

FAILED=""
for SD in "$@"; do
  for K in $(ls -d ${SD}/[0-9]* | xargs -n1 basename | sort -n); do
    D=${SD}/${K}
    [ -s ${D}/wise_scores.txt ] && { echo "scored: $D"; continue; }
    IMG=${D}/wise_images; rm -rf ${IMG}; mkdir -p ${IMG}
    for P in ${D}/[0-9]*; do
      I=$((10#$(basename $P)))
      ln -s ${P}/samples/00000.png ${IMG}/$((I + 1)).png
    done
    echo "$(date +%H:%M:%S) judging $D ($(ls ${IMG} | wc -l) images)"
    (cd ${WISE} && IMAGE_DIR=${IMG} VLLM_API_BASE=http://127.0.0.1:${PORT}/v1 JUDGE_MODEL=Qwen3.5-35B-A3B \
        MAX_WORKERS=${MAX_WORKERS:-96} bash eval_qwen.sh > ${D}/wise_eval.log 2>&1) \
      && (cd ${WISE} && R=${IMG}/Results-qwen35 && python calculate_verified.py \
            ${R}/cultural_common_sense_scores_results.jsonl ${R}/natural_science_scores_results.jsonl \
            ${R}/spatio-temporal_reasoning_scores_results.jsonl --category all) > ${D}/wise_scores.txt \
      && tail -12 ${D}/wise_scores.txt \
      || { echo "FAILED $D"; tail -10 ${D}/wise_eval.log; FAILED="$FAILED $D"; }
  done
done
[ -z "$FAILED" ] || { echo "failed:$FAILED"; exit 1; }
echo "================ DONE ============="
