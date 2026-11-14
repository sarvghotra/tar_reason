#!/bin/bash
# Score iterative runs on the original GenEval (Ghosh et al. 2023) with its official evaluator, unchanged:
# evaluation/evaluate_images.py (Mask2Former Swin-S detector + CLIP ViT-L-14 colour classifier) and
# evaluation/summary_scores.py, from jeet's GenEval checkout and `geneval` env (mmdet 2.28.2, mmcv 1.7.2, torch 2.1.2).
# Each iteration folder <seed>/<k>/ written by eval/iterative_generation_adhoc_MDP.py is already in GenEval's layout
# (<prompt idx>/metadata.jsonl + samples/<n>.png), so it is scored as is. Writes <seed>/<k>/geneval_results.jsonl and
# <seed>/<k>/geneval_summary.txt; iterations with a summary are skipped (re-submit resumes).
#
#   sbatch scripts/eval/geneval_score_mila.sh <seed dir> [<seed dir> ...]
#
#SBATCH --time=6:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --constraint=80gb
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --job-name=bj_geneval_score
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out

set -uo pipefail
JEET=/network/scratch/s/sarvjeet-singh.ghotra
GENEVAL=${JEET}/git/geneval
export CONDA_PREFIX=${JEET}/installs/miniforge3/envs/geneval
export PATH=${CONDA_PREFIX}/bin:${PATH} LD_LIBRARY_PATH=${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset PYTHONPATH
FAILED=""
for SD in "$@"; do
  for K in $(ls -d ${SD}/[0-9]* | xargs -n1 basename | sort -n); do
    D=${SD}/${K}
    [ -s ${D}/geneval_summary.txt ] && grep -q "^Overall score" ${D}/geneval_summary.txt && { echo "scored: $D"; continue; }
    echo "$(date +%H:%M:%S) scoring $D ($(ls -d ${D}/[0-9]* | wc -l) prompts)"
    (cd ${GENEVAL} && python evaluation/evaluate_images.py ${D} --outfile ${D}/geneval_results.jsonl \
        --model-path ${GENEVAL}/OBJECT_DETECTOR_FOLDER > ${D}/geneval_eval.log 2>&1 \
      && python evaluation/summary_scores.py ${D}/geneval_results.jsonl > ${D}/geneval_summary.txt) \
      && grep "^Overall score" ${D}/geneval_summary.txt \
      || { echo "FAILED $D"; tail -5 ${D}/geneval_eval.log; FAILED="$FAILED $D"; }
  done
done
[ -z "$FAILED" ] || { echo "failed:$FAILED"; exit 1; }
echo "================ DONE ============="
