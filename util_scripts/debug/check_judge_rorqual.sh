#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH -J judge_check
#SBATCH -o /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /home/dars11/scratch/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# One-GPU check of a candidate pixel-reward judge before an RL run (2026-11-04: gemma-4-E4B-it): answer sanity
# (test_pixel_judge.py), shared-image-prefix equivalence (test_prefix_share.py), and speed per row vs shared
# (bench_judge.py on real promptmix questions). JUDGE=<model dir> PY=<judge env python> ARGS="<judge args>".
set -u
JUDGE=${JUDGE:-/home/dars11/scratch/models/vlm/gemma-4-E4B-it}; PY=${PY:-$HOME/envs/gemma_judge/bin/python}
ARGS=${ARGS:---image_soft_tokens 280 --answer_id_mode strict}
D=/home/dars11/scratch/data/promptmix/train_17K.jsonl
cd /home/dars11/scratch/git/tar_reason; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
t0=$(date +%s); cp -r $JUDGE $SLURM_TMPDIR/judge; echo "staged in $(( $(date +%s) - t0 )) s"; M=$SLURM_TMPDIR/judge
f() { grep -v -i -E "warn|loading weights|loading checkpoint|UNEXPECTED|LOAD REPORT|^Key |^-----|can be ignored|^Notes"; }
echo "=== sanity $(date +%H:%M)";      $PY util_scripts/debug/test_pixel_judge.py $M $ARGS 2>&1 | f | cut -c1-200 | head -40
echo "=== prefix share $(date +%H:%M)"; $PY util_scripts/debug/test_prefix_share.py $M $ARGS 2>&1 | f | cut -c1-250 | head -30
echo "=== per row $(date +%H:%M)";      $PY util_scripts/debug/bench_judge.py $M --data $D $ARGS 2>&1 | f | grep -E "RESULT|Error" | head -3
echo "=== shared $(date +%H:%M)";       $PY util_scripts/debug/bench_judge.py $M --data $D $ARGS --force_share 2>&1 | f | grep -E "prompt:|RESULT|Error" | cut -c1-300 | head -4
echo "=== done $(date +%H:%M)"
