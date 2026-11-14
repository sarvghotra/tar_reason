#!/bin/bash
# Rollback analysis of t4's TIIF run (jeet's iter_adhoc_MDP_n4_E-450_ORC_512px_repeat1_Cric-SFT-t4, Qwen2.5-VL-72B
# judge outputs in seed_*/eval_results/): eval/analyze/tiif_rollback.py once per self-score, after
# scripts/eval/tiif_self_judge_t4_mila.sh. CPU only (TIIF-Bench's summary scripts need the judge env's pandas).
#SBATCH --time=1:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=bj_tiif_rollback
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out
set -e
cd /home/mila/s/singhsd/CODE/LatentDCR/tar_reason
R=/network/scratch/s/singhsd/tar_reason/results_t4
for SCORE in approve_lp vqa_yes; do
  ~/envs/qwen_judge/bin/python eval/analyze/tiif_rollback.py \
    --run_dir /network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_MDP_n4_E-450_ORC_512px_repeat1_Cric-SFT-t4 \
    --self_dir ${R}/tiif_self_judge --out ${R}/tiif_rollback_${SCORE} --score ${SCORE} \
    --tiif_dir /network/scratch/s/singhsd/benchmarks/TIIF-Bench --judge_py /home/mila/s/singhsd/envs/qwen_judge/bin/python
done
