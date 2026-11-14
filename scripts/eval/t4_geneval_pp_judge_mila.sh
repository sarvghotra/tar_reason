#!/bin/bash
# Score t4's GenEval++ images (scripts/eval/t4_gen_mila.sh) with eval/geneval_pp_judge.py: Echo-4o's GenEval++
# scorer (system prompt, checklist, temperature 0, max_tokens 200) against an open judge, Qwen2.5-VL-72B-Instruct,
# served by `vllm serve` (TP=4, bf16; flags as jeet's TIIF 72B judge in iter_tiif_bench_adhoc_e2e_MDP_vllm.sh) in
# jeet's vllm env. Every iteration of every finished seed; finished iterations are skipped (re-submit resumes).
# The official judge is GPT-4.1 (user, 2026-11-13: open judge only), so these scores are not comparable to
# published GenEval++ numbers.
#
#   sbatch --export=ALL,SEEDS=13 scripts/eval/t4_geneval_pp_judge_mila.sh
#
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bj_t4_genevalpp
#SBATCH --exclude=cn-g005,cn-g011
#SBATCH --partition=long
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out

set -e
REPO=/home/mila/s/singhsd/CODE/LatentDCR/tar_reason
ROOT=${ROOT:-/network/scratch/s/singhsd/tar_reason/results_t4}/GenEval++
RUN=iter_adhoc_MDP_n${N_ITERS:-3}_E-450_ORC_512px_repeat1_Cric-SFT-t4
SEEDS=(${SEEDS:-13})
JEET=/network/scratch/s/sarvjeet-singh.ghotra
JUDGE=${JUDGE:-${JEET}/models/pre_train/Qwen2.5-VL-72B-Instruct}
PORT=${PORT:-8123}

SEED_DIRS=()
for S in "${SEEDS[@]}"; do
  D=${ROOT}/${RUN}/seed_${S}
  if [ -f ${D}/.gen_done ]; then SEED_DIRS+=(${D}); else echo "seed ${S}: not generated yet, skipping"; fi
done
[ ${#SEED_DIRS[@]} -gt 0 ] || { echo "nothing to score"; exit 0; }

export CONDA_PREFIX=${JEET}/installs/miniforge3/envs/vllm
export PATH=${CONDA_PREFIX}/bin:${PATH} LD_LIBRARY_PATH=${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
module load cuda/12.6.0
export HF_HUB_OFFLINE=1 VLLM_CACHE_ROOT=/network/scratch/s/singhsd/.cache/vllm TRITON_CACHE_DIR=/tmp/triton_judge_${SLURM_JOB_ID:-$$}
# jeet's workaround: Triton compiles against a node-local copy of the Python headers
PY_INC=$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
LOCAL_PY_INC=/tmp/py_include_judge_${SLURM_JOB_ID:-$$}
PY_INC_COPY=/network/scratch/s/singhsd/jeet_snapshot/py_include_vllm   # copy of ${PY_INC}: reading his env failed on some nodes
for TRY in 1 2 3 4 5; do
  rm -rf ${LOCAL_PY_INC}
  cp -r ${PY_INC_COPY} ${LOCAL_PY_INC} 2>/dev/null \
    && [ "$(find ${LOCAL_PY_INC} -type f | wc -l)" -eq "$(find ${PY_INC_COPY} -type f | wc -l)" ] && break
  echo "copy of Python headers failed (try $TRY)"; sleep 20
done
[ -f ${LOCAL_PY_INC}/Python.h ] || { echo "ERROR: could not copy ${PY_INC}"; exit 1; }
export CC=/tmp/triton_cc_judge_${SLURM_JOB_ID:-$$}.sh
cat > ${CC} <<EOS
#!/bin/bash
ARGS=()
for A in "\$@"; do [ "\$A" = "-I${PY_INC}" ] && A="-I${LOCAL_PY_INC}"; ARGS+=("\$A"); done
exec /usr/bin/gcc "\${ARGS[@]}"
EOS
chmod +x ${CC}

LOG=/network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/genevalpp_judge_server_${SLURM_JOB_ID:-local}.log
setsid python -m vllm.entrypoints.cli.main serve ${JUDGE} --port ${PORT} --dtype bfloat16 --tensor-parallel-size 4 \
  --max-model-len 8192 --limit-mm-per-prompt '{"image": 1, "video": 0}' --no-mm-device-do-normalize --seed 0 \
  --disable-custom-all-reduce \
  > ${LOG} 2>&1 &
SERVER=$!
trap 'kill -- -${SERVER} 2>/dev/null || true' EXIT
STARTED=${SECONDS}
until curl -sf http://localhost:${PORT}/health >/dev/null 2>&1; do
  kill -0 ${SERVER} 2>/dev/null || { echo "judge server died:"; tail -30 ${LOG}; exit 1; }
  (( SECONDS - STARTED < 2400 )) || { echo "judge server not up after 40 min"; tail -30 ${LOG}; exit 1; }
  sleep 15
done
echo "$(date +%H:%M:%S) judge up after $(( SECONDS - STARTED )) s"

python ${REPO}/eval/geneval_pp_judge.py --seed_dirs "${SEED_DIRS[@]}" --base_url http://localhost:${PORT}/v1 \
  --model ${JUDGE} --tag qwen72b --workers 64
echo "================ DONE ============="
