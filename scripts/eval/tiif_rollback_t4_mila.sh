#!/bin/bash
# Rollback analysis of an iterative TIIF run (default t4's) with eval/analyze/tiif_rollback.py, once per self-score in
# SCORES, after scripts/eval/tiif_self_judge_t4_mila.sh. CPU only (TIIF-Bench's summary scripts need the judge env's
# pandas). Override RUN_DIR / SELF_DIR / OUT_PREFIX / JUDGE_RUN_DIR (e.g. a temperature-0 re-judge).
#SBATCH --time=1:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=bj_tiif_rollback
#SBATCH -o /network/scratch/s/singhsd/tar_reason/output_dir/slurm_logs/%x_%j.out
set -e
cd /home/mila/s/singhsd/CODE/LatentDCR/tar_reason
R=/network/scratch/s/singhsd/tar_reason/results_t4
RUN_DIR=${RUN_DIR:-/network/scratch/s/sarvjeet-singh.ghotra/git/tar_reason/results/tiif-bench-testmini_eval/iter_adhoc_MDP_n4_E-450_ORC_512px_repeat1_Cric-SFT-t4}
for SCORE in ${SCORES:-approve_lp vqa_yes}; do
  ~/envs/qwen_judge/bin/python eval/analyze/tiif_rollback.py --run_dir ${RUN_DIR} ${JUDGE_RUN_DIR:+--judge_run_dir $JUDGE_RUN_DIR} \
    --self_dir ${SELF_DIR:-${R}/tiif_self_judge} --out ${OUT_PREFIX:-${R}/tiif_rollback}_${SCORE} --score ${SCORE} \
    --tiif_dir /network/scratch/s/singhsd/benchmarks/TIIF-Bench --judge_py /home/mila/s/singhsd/envs/qwen_judge/bin/python
done
