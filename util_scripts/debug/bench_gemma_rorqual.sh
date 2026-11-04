#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH -J bench_gemma
#SBATCH -o /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# Gemma 4 judges on one H100 (2026-11-03): (1) prefix-sharing equivalence + speedup for the live run's Gemma 4 12B (same
# test that waits in Fir's / TamIA's queues); (2) Gemma 4 26B-A4B (MoE, 4B active) as a candidate reward model: answer
# sanity, bench_judge.py timing on the per-row path, then prefix-sharing equivalence + speedup.
set -u
cd /home/dars11/scratch/git/tar_reason
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
V=/home/dars11/scratch/models/vlm; D=/home/dars11/scratch/data/promptmix/train_17K.jsonl; PY=~/envs/gemma_judge/bin/python
echo "host $(hostname) job ${SLURM_JOB_ID}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== 1 Gemma 4 12B prefix sharing"
$PY util_scripts/debug/test_prefix_share.py $V/gemma-4-12B-it --image_soft_tokens 280 --answer_id_mode strict 2>&1 | grep -v -i warn
echo "=== 2a Gemma 4 26B-A4B answer sanity"
$PY util_scripts/debug/test_pixel_judge.py $V/gemma-4-26B-A4B-it --image_soft_tokens 280 --answer_id_mode strict 2>&1 | grep -v -i warn
echo "=== 2b Gemma 4 26B-A4B timing (per row)"
$PY util_scripts/debug/bench_judge.py $V/gemma-4-26B-A4B-it --data $D --image_soft_tokens 280 --answer_id_mode strict 2>&1 | grep -v -i warn
echo "=== 2c Gemma 4 26B-A4B prefix sharing"
$PY util_scripts/debug/test_prefix_share.py $V/gemma-4-26B-A4B-it --image_soft_tokens 280 --answer_id_mode strict 2>&1 | grep -v -i warn
echo "=== done"
