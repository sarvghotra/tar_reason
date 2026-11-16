#!/bin/bash
# TamIA port of geneval2_rescore_mila.sh (2026-11-15): re-grade existing GenEval2 images with Qwen3.5-27B, the paper's
# GenEval2 judge (thinking off), for runs generated elsewhere and copied to $STAGE. Writes <run>_judge-Qwen3.5-27B
# mirrors next to each run (source unchanged). Same scorer (eval/geneval2_rescore.py) and settings as Mila.
#   sbatch scripts/eval/geneval2_rescore_tamia.sh <run_dir> ...
#SBATCH --time=2:59:00
#SBATCH --account=aip-agrawal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --job-name=ge2_rescore
#SBATCH -o /scratch/d/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.out
set -e
REPO=/scratch/d/dars11/git/tar_reason
GENEVAL2_DIR=/scratch/d/dars11/benchmarks/GenEval2   # evaluation_optz.py md5-identical to jeet's
EVAL_SET=${GENEVAL2_DIR}/geneval2_data.jsonl
STAGE=/scratch/d/dars11/tar_reason/results_rescore/GenEval2
JUDGE=${JUDGE:-/scratch/d/dars11/hf_home/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654}
TAG=${TAG:-Qwen3.5-27B}
BATCH=${BATCH:-32}
BUCKET=${BUCKET:-0}   # as jeet's default: plain left-padded batches (evaluation_optz.py)
BUCKET_ARG=""; [ "$BUCKET" = "1" ] || BUCKET_ARG="--no_bucket"
SCORE_PY=/home/d/dars11/envs/genai_judge/bin/python   # transformers 5.x (qwen3_5)
TAR_PY=/home/d/dars11/envs/tar/bin/python

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
  LOG_DIR=/scratch/d/dars11/git/tar_reason/output_dir/slurm_logs/geneval2_rescore_${SLURM_JOB_ID:-local}
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
    # the summary tables need >= 2 iterations; iteration-1-only mirrors stop at the ITER block
    # (eval/stats/export_system_level.py computes all-correct from results_geneval2.jsonl)
    [ $N_ITERS -ge 2 ] || continue
    $TAR_PY ${REPO}/eval/analyze/viz_geneval2_iter_results.py --results-dir ${SEED_DIR} --delta-threshold 0.1 \
      --no_images >> $LOG_FILE || POST_FAILED="$POST_FAILED ${SEED_DIR}"
  done
  echo "== ${DST}"
  grep -h -E "^(AM|GM) Score" ${DST}/seed_*/[0-9]*/eval_log.txt 2>/dev/null | head -6
done
[ -z "$POST_FAILED" ] || { echo "log.txt summary failed for:$POST_FAILED"; exit 1; }
echo "================ DONE ============="
