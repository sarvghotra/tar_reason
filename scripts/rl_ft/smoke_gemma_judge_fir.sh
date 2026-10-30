#!/bin/bash
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --exclude=fc10212
#SBATCH -J gemma_judge_smoke
#SBATCH -o /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# One-GPU smoke test of Gemma 4 12B it as the pixel judge (2026-10-30): loads it through
# pixel_reward_server.Judge with jeet's Gemma settings and scores synthetic images. Compare with
# the same script on Qwen3-VL-8B (geneval2 mode) to see how the two judges' probabilities differ.
#   sbatch scripts/rl_ft/smoke_gemma_judge_fir.sh
set -u
cd /home/dars11/scratch/git/tar_reason
export PYTHONPATH=$PWD
echo "host $(hostname) job ${SLURM_JOB_ID}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== Gemma 4 12B it (strict, 280 soft tokens)"
/home/dars11/envs/gemma_judge/bin/python util_scripts/debug/test_pixel_judge.py \
    /home/dars11/scratch/models/vlm/gemma-4-12B-it --answer_id_mode strict --image_soft_tokens 280
echo "=== Qwen3-VL-8B-Instruct (geneval2 mode, reference)"
/home/dars11/envs/qwen_judge/bin/python util_scripts/debug/test_pixel_judge.py \
    /home/dars11/scratch/models/vlm/Qwen3-VL-8B-Instruct --answer_id_mode geneval2
