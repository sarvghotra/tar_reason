#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH -J judge30b_smoke
#SBATCH -o /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# One-GPU check of Qwen3-VL-30B-A3B-Instruct (the AlphaGRPO paper's reward model, 62 GB bf16 MoE) as the pixel judge
# through pixel_reward_server.Judge, GenEval2 answer mode, on synthetic images (2026-10-30, Nibi). Must load on one H100
# and give sensible probabilities before bash_nibi_alphagrpo.sh runs with it.
#   sbatch scripts/rl_ft/smoke_judge_nibi.sh
set -u
cd /scratch/dars11/git/tar_reason
export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "host $(hostname) job ${SLURM_JOB_ID}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/home/dars11/envs/qwen_judge/bin/python util_scripts/debug/test_pixel_judge.py \
    ${JUDGE_MODEL:-/scratch/dars11/models/vlm/Qwen3-VL-30B-A3B-Instruct} --answer_id_mode geneval2 --batch_size 16
