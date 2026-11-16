#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=h100:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --open-mode=append
#SBATCH -J ge2_self
#SBATCH -o /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# Self-verified selection (Mila, 2026-11-15; paper repo results/self_verified_selection/) on a Fir GenEval2 run:
# eval/geneval2_self_judge.py scores every image 1..4 of each seed with the policy's own answers (am_ans), one seed per
# GPU in parallel; resumable (re-submit the same command). Then, on a login node, the selection itself (no GPU):
#   ~/envs/tar/bin/python eval/analyze/geneval2_rollback.py --self_dir $OUT --judge_run_dir $RUN_DIR --scores am_ans am_rephr
#   MODEL_NAME=t4_e2eC300 LORA_PATH=<run>/checkpoint-300 sbatch -J ge2_self_C300 scripts/eval/geneval2_self_judge_fir.sh
set -eu
cd /home/dars11/scratch/git/tar_reason
: "${MODEL_NAME:?}" "${LORA_PATH:?}"
BASE=${BASE:-/home/dars11/scratch/models/tar/sft/darshan_E450_orc500_crisft_t4_merged}
RUN_DIR=${RUN_DIR:-/scratch/dars11/tar_reason/results_jeet/GenEval2/iter_adhoc_MDP_n4_${MODEL_NAME}_512px_repeat1}
OUT=${OUT:-/scratch/dars11/tar_reason/self_judge/GenEval2/${MODEL_NAME}}
SEEDS=${SEEDS:-2213 713 13}
GPU_OFFSET=${GPU_OFFSET:-0}   # first GPU index: two invocations in one job must not share GPUs (OOM, 2026-11-16)
mkdir -p $OUT; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "$(date) job=${SLURM_JOB_ID} MODEL_NAME=$MODEL_NAME BASE=$BASE LORA=$LORA_PATH RUN_DIR=$RUN_DIR OUT=$OUT"
g=$GPU_OFFSET
for s in $SEEDS; do
  CUDA_VISIBLE_DEVICES=$g ~/envs/tar/bin/python eval/geneval2_self_judge.py --run_dir $RUN_DIR --seeds $s \
    --model $BASE --lora_path $LORA_PATH --ta_tok /home/dars11/scratch/models/tar/ta_tok.pth \
    --siglip2 /home/dars11/scratch/models/tar/siglip2-so400m-patch14-384 --out_dir $OUT > $OUT/log_seed_$s.txt 2>&1 &
  g=$((g + 1))
done
wait
for s in $SEEDS; do echo "seed $s: $(wc -l < $OUT/seed_$s.jsonl) rows (expect 3200)"; done
