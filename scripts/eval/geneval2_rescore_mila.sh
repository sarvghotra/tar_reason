#!/bin/bash
# Mila (singhsd) port of jeet's scripts/eval/geneval2_rescore.sh: re-grade existing GenEval2 images with
# Qwen3.5-27B (eval/geneval2_rescore.py, copied verbatim from his checkout on 2026-11-12). No generation.
# Scores go to a mirror dir <run>_judge-${TAG} next to each source run; the source is not modified.
#
# For the paper's system-level baseline table (Mila Inbox 2026-11-12) the sources are staged copies holding
# only iteration 1 (single-pass models: image 2 = image 1): $SCRATCH/tar_reason/results_jeet_rescore/GenEval2/
# <run>/seed_*/1/{geneval2_results.json, prompt dirs symlinked to results_jeet}.
#
#   sbatch scripts/eval/geneval2_rescore_mila.sh                 # every staged run
#   sbatch scripts/eval/geneval2_rescore_mila.sh <run_dir> ...   # only these
#   POSTPROCESS_ONLY=1 bash scripts/eval/geneval2_rescore_mila.sh   # rebuild log.txt only (CPU)
#
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a100l:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=bj_geneval2_rescore
#SBATCH --partition=short-unkillable
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out

set -e
REPO=/home/mila/s/singhsd/CODE/LatentDCR/tar_reason
GENEVAL2_DIR=/network/scratch/s/singhsd/benchmarks/GenEval2   # evaluation_optz.py md5-identical to jeet's
EVAL_SET=${GENEVAL2_DIR}/geneval2_data.jsonl
STAGE=/network/scratch/s/singhsd/tar_reason/results_jeet_rescore/GenEval2
JUDGE=${JUDGE:-/network/scratch/s/singhsd/hf_home/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654}
TAG=${TAG:-Qwen3.5-27B}
BATCH=${BATCH:-32}
BUCKET=${BUCKET:-0}   # as jeet's default: plain left-padded batches (evaluation_optz.py)
BUCKET_ARG=""; [ "$BUCKET" = "1" ] || BUCKET_ARG="--no_bucket"
SCORE_PY=/home/mila/s/singhsd/envs/genai_judge/bin/python   # transformers 5.x (qwen3_5)
TAR_PY=/home/mila/s/singhsd/envs/tar/bin/python

if [ $# -gt 0 ]; then
  SRC_DIRS=("$@")
else
  SRC_DIRS=($(ls -d ${STAGE}/*/ | grep -v "_judge-" | sed 's:/$::'))
fi

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t GPU_IDS < <(seq 0 $(($(nvidia-smi --list-gpus | wc -l) - 1)))
fi
N_GPUS=${#GPU_IDS[@]}
echo "JUDGE=${JUDGE} TAG=${TAG} N_GPUS=${N_GPUS} BATCH=${BATCH} BUCKET=${BUCKET}"
printf '  %s\n' "${SRC_DIRS[@]}"

if [ "${POSTPROCESS_ONLY:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
  LOG_DIR=/network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/geneval2_rescore_${SLURM_JOB_ID:-local}
  mkdir -p $LOG_DIR
  declare -A PIDS=()
  for (( R=0; R<N_GPUS; R++ )); do
    CUDA_VISIBLE_DEVICES=${GPU_IDS[$R]} env -u PYTHONPATH $SCORE_PY ${REPO}/eval/geneval2_rescore.py \
      --src_dirs "${SRC_DIRS[@]}" --benchmark_data $EVAL_SET --model_path $JUDGE --tag $TAG \
      --geneval2_dir $GENEVAL2_DIR --rank $R --world_size $N_GPUS \
      --batch_size $BATCH $BUCKET_ARG > ${LOG_DIR}/rank${R}.log 2>&1 &
    PIDS[$R]=$!
  done
  FAILED=""
  for R in "${!PIDS[@]}"; do
    STATUS=0; wait ${PIDS[$R]} || STATUS=$?
    [ $STATUS -ne 0 ] && FAILED="$FAILED rank-${R}:${STATUS}"
  done
  if [ -n "$FAILED" ]; then
    echo "Scoring failed:$FAILED (logs in ${LOG_DIR})"; tail -20 ${LOG_DIR}/rank*.log; exit 1
  fi
fi

# Per-seed log.txt as the e2e scripts write it (eval/stats/geneval2.py reads it); prompt dirs symlinked back.
POST_FAILED=""
for SRC in "${SRC_DIRS[@]}"; do
  SRC=$(realpath $SRC)
  DST=$(dirname $SRC)/$(basename $SRC)_judge-${TAG}
  for SEED_DIR in ${DST}/seed_*; do
    SRC_SEED=${SRC}/$(basename $SEED_DIR)
    N_ITERS=$(ls -d ${SEED_DIR}/[0-9]* | wc -l)
    for ITER in $(seq 1 $N_ITERS); do ln -sfn -t ${SEED_DIR}/${ITER} ${SRC_SEED}/${ITER}/[0-9]*; done
    LOG_FILE=${SEED_DIR}/log.txt; : > $LOG_FILE
    for ITER in $(seq 1 $N_ITERS); do
      { echo " "; echo "ITER-${ITER}:"; cat ${SEED_DIR}/${ITER}/eval_log.txt; } >> $LOG_FILE
    done
    $TAR_PY ${REPO}/eval/analyze/viz_geneval2_iter_results.py --results-dir ${SEED_DIR} --delta-threshold 0.1 \
      --no_images >> $LOG_FILE || POST_FAILED="$POST_FAILED ${SEED_DIR}"
  done
  echo "== ${DST}"
  grep -h -E "^(AM|GM) Score" ${DST}/seed_*/[0-9]*/eval_log.txt 2>/dev/null | head -6
done
[ -z "$POST_FAILED" ] || { echo "log.txt summary failed for:$POST_FAILED"; exit 1; }
echo "================ DONE ============="
