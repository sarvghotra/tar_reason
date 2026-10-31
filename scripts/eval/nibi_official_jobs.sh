#!/bin/bash
# Nibi submissions for the official-judge re-scoring (user, 2026-10-30), driven from Fir over ssh.
#   bash scripts/eval/nibi_official_jobs.sh e450      # TIIF (Qwen2.5-VL-72B) + GenAI (CLIP-FlanT5-XXL) on E-450's images (rsynced from Mila)
#   bash scripts/eval/nibi_official_jobs.sh tar7b_gen # base Tar-7B: generate TIIF + GenAI images with jeet's pipeline, single pass, no judging
#   bash scripts/eval/nibi_official_jobs.sh tar7b     # ... then re-score them (after tar7b_gen finished)
# Preconditions (Nibi): ~/envs/{vllm_judge,clipt5_judge,tar}, $HF_HOME/hub with the 72B + clip-flant5-xxl + clip-vit-336,
# /scratch/dars11/{benchmarks/{TIIF-Bench,GenAI-Bench},git/{tar_reason,t2v_metrics_jeet,t2v_metrics_clipt5},models/tar}.
set -eu
cd /scratch/dars11/git/tar_reason
export PATH=$PATH:/opt/software/slurm/bin
mkdir -p output_dir/slurm_logs
R=/scratch/dars11/tar_reason/results_jeet
LOGS=(-o /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.out -e /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.err)
ACC=(--account=rrg-bengioy-ad_gpu)
case ${1:?e450 | tar7b_gen | tar7b} in
  e450)
    for B in tiif genai; do
      case $B in tiif) S=tiif-bench-testmini_eval ;; genai) S=genai-bench-1600 ;; esac
      BENCH=$B MODEL_NAME=darshan_pm1_E_450 N_ITERS=5 RUN_DIR=$R/$S/iter_adhoc_MDP_n5_darshan_pm1_E_450_512px_repeat1 \
        sbatch --export=ALL "${ACC[@]}" "${LOGS[@]}" -J official_${B}_e450 scripts/eval/official_judges_tamia.sh
    done ;;
  tar7b_gen)
    for B in tiif genai; do
      MODEL_NAME=7B MODEL_PATH=/scratch/dars11/models/tar/Tar-7B SINGLE_PASS=1 GEN_ONLY=1 BENCH=$B \
        sbatch --export=ALL "${ACC[@]}" "${LOGS[@]}" --mem=240G -J gen_${B}_tar7b scripts/eval/bench_jeet_tamia.sh
    done ;;
  tar7b)
    for B in tiif genai; do
      case $B in tiif) S=tiif-bench-testmini_eval ;; genai) S=genai-bench-1600 ;; esac
      BENCH=$B MODEL_NAME=7B N_ITERS=2 RUN_DIR=$R/$S/iter_adhoc_MDP_n2_7B_512px_repeat1 \
        sbatch --export=ALL "${ACC[@]}" "${LOGS[@]}" -J official_${B}_tar7b scripts/eval/official_judges_tamia.sh
    done ;;
esac
squeue -u dars11 -o "%.9i %.22j %.8T %.10M %R"
