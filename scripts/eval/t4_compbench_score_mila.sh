#!/bin/bash
# Score t4's T2I-CompBench++ images (scripts/eval/t4_gen_mila.sh) with the official evaluators, through
# scripts/eval/t2icompbench_eval_mila.sh, for every category x seed x iteration whose generation is done.
# Per iteration k the images are linked into <seed>/<k>/t2icb/samples/<val prompt>_<prompt index, 6 digits>.png
# (1 image per prompt: question_id = prompt index) and scored there; a run with t2icb/score_<cat>.json is skipped,
# so re-submitting resumes. Runs on the preemptible `long` partition (each scoring pass is a few minutes).
#
#   sbatch --export=ALL,SEEDS=13 scripts/eval/t4_compbench_score_mila.sh
#
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --job-name=bj_t4_t2icb
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out

set -uo pipefail
REPO=/home/mila/s/singhsd/CODE/LatentDCR/tar_reason
ROOT=${ROOT:-/network/scratch/s/singhsd/tar_reason/results_t4}/T2I-CompBench++
RUN=iter_adhoc_MDP_n${N_ITERS:-3}_E-450_ORC_512px_repeat1_Cric-SFT-t4
DATA=/network/scratch/s/singhsd/benchmarks/T2I-CompBench/examples/dataset
SEEDS=(${SEEDS:-13})
CATS=(${CATS:-color shape texture spatial 3d_spatial non_spatial numeracy complex})

FAILED=""
for SEED in "${SEEDS[@]}"; do
  for CAT in "${CATS[@]}"; do
    SD=${ROOT}/${CAT}/${RUN}/seed_${SEED}
    [ -f ${SD}/.gen_done ] || { echo "$CAT seed $SEED: not generated yet, skipping"; continue; }
    for K in $(ls -d ${SD}/[0-9]* | xargs -n1 basename | sort -n); do
      E=${SD}/${K}/t2icb
      [ -f ${E}/score_${CAT}.json ] && { echo "$CAT seed $SEED iter $K: scored"; continue; }
      rm -rf ${E}; mkdir -p ${E}/samples
      I=0
      while IFS= read -r PROMPT || [ -n "$PROMPT" ]; do
        [ -n "$PROMPT" ] || continue
        IMG=${SD}/${K}/$(printf %05d $I)/samples/00000.png
        [ -f "$IMG" ] || { echo "missing $IMG"; exit 1; }
        ln -s "$IMG" "${E}/samples/${PROMPT}_$(printf %06d $I).png"
        I=$((I + 1))
      done < ${DATA}/${CAT}_val.txt
      echo "$(date +%H:%M:%S) $CAT seed $SEED iter $K: $I images"
      bash ${REPO}/scripts/eval/t2icompbench_eval_mila.sh ${CAT} ${E} > ${E}/eval.log 2>&1 \
        && cat ${E}/score_${CAT}.json && echo \
        || { echo "FAILED $CAT seed $SEED iter $K (log ${E}/eval.log)"; tail -5 ${E}/eval.log; FAILED="$FAILED $CAT/$SEED/$K"; }
    done
  done
done
[ -z "$FAILED" ] || { echo "failed:$FAILED"; exit 1; }
echo "================ DONE ============="
