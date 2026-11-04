#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH -J bench_judges
#SBATCH -o /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# Reward-judge timing on one H100 (user, 2026-11-03: "check how much time Qwen3.5-35B-A3B would take"): the same
# bench_judge.py load (requests of 8 images x their real promptmix questions, as the reward server sees them) for
#   1. Qwen3.5-35B-A3B (non-thinking template, strict answer ids, flash-linear-attention kernels), after an answer-sanity
#      check with test_pixel_judge.py
#   2. Gemma 4 12B it (the live run's reward judge: strict ids, 280 soft tokens, no prefix sharing)
#   3. Qwen3-VL-8B-Instruct (E-450's reward judge: geneval2 ids, shared image prefix)
# so the three numbers are directly comparable (same GPU, same images and questions).
#   sbatch util_scripts/debug/bench_judges_rorqual.sh
set -u
cd /home/dars11/scratch/git/tar_reason
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
V=/home/dars11/scratch/models/vlm; D=/home/dars11/scratch/data/promptmix/train_17K.jsonl
echo "host $(hostname) job ${SLURM_JOB_ID}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== 1a Qwen3.5-35B-A3B answer sanity"
~/envs/qwen35_judge/bin/python util_scripts/debug/test_pixel_judge.py $V/Qwen3.5-35B-A3B --answer_id_mode strict 2>&1 | grep -v -i warn
echo "=== 1b Qwen3.5-35B-A3B timing"
~/envs/qwen35_judge/bin/python util_scripts/debug/bench_judge.py $V/Qwen3.5-35B-A3B --data $D --answer_id_mode strict 2>&1 | grep -v -i "warn" \
  || ~/envs/qwen35_judge/bin/python util_scripts/debug/bench_judge.py $V/Qwen3.5-35B-A3B --data $D --answer_id_mode strict --batch_size 32
echo "=== 2 Gemma 4 12B timing"
~/envs/gemma_judge/bin/python util_scripts/debug/bench_judge.py $V/gemma-4-12B-it --data $D --answer_id_mode strict --image_soft_tokens 280 2>&1 | grep -v -i warn
echo "=== 3 Qwen3-VL-8B timing (shared prefix)"
~/envs/qwen_judge/bin/python util_scripts/debug/bench_judge.py $V/Qwen3-VL-8B-Instruct --data $D --answer_id_mode geneval2 --share_image_prefix 2>&1 | grep -v -i warn
echo "=== done"
