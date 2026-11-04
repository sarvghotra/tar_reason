#!/bin/bash
#SBATCH --time=02:59:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH -J genai_clipt5_baselines
#SBATCH -o /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# GenAI-Bench official judge (CLIP-FlanT5-xxl VQAScore) for the baseline image folders, on Rorqual (2026-11-04: Nibi's queue
# was a day out). Rorqual's Lustre is slow for small files and model reads, so everything is staged on node-local disk:
#   /scratch/dars11/stage/genai_judge_code.tar     clipt5_judge env + t2v_metrics repos + GenAI-Bench (tarred on Nibi, paths from /)
#   /scratch/dars11/stage/genai_baseline_images.tar the baseline folders under tar_reason/results_jeet/genai-bench-1600/
#   /scratch/dars11/hf_home/hub/models--{zhiqiulin--clip-flant5-xxl,openai--clip-vit-large-patch14-336,google--flan-t5-xxl}
# then official_judges_tamia.sh runs per folder (N_GPUS=4, one iteration per GPU) and the scores (not the images) are copied
# back to /scratch/dars11/tar_reason/results_jeet/genai-bench-1600/<run>/. RUNS="a b" limits the folders.
set -euo pipefail
T=${SLURM_TMPDIR:?}; S=/scratch/dars11/stage; HUB=/scratch/dars11/hf_home/hub
LUSTRE_OUT=/scratch/dars11/tar_reason/results_jeet/genai-bench-1600
t0=$(date +%s)
tar -C $T -xf $S/genai_judge_code.tar
tar -C $T -xf $S/genai_baseline_images.tar
mkdir -p $T/hf_home/hub
for m in models--zhiqiulin--clip-flant5-xxl models--openai--clip-vit-large-patch14-336 models--google--flan-t5-xxl; do cp -a $HUB/$m $T/hf_home/hub/; done
echo "staged in $(( $(date +%s) - t0 )) s: $(du -sh $T | cut -f1)"
export CLIPT5_ENV=$T/home/dars11/envs/clipt5_judge T2V_CLIPT5=$T/scratch/dars11/git/t2v_metrics_clipt5 \
       T2V_FORK=$T/scratch/dars11/git/t2v_metrics_jeet BENCH_DIR=$T/scratch/dars11/benchmarks HF_HOME=$T/hf_home N_GPUS=4
OUT=$T/scratch/dars11/tar_reason/results_jeet/genai-bench-1600
RUNS=(${RUNS:-$(ls $OUT)})
FAIL=0
for run in "${RUNS[@]}"; do
  d=$OUT/$run
  n=$(echo $run | sed -E 's/^.*_n([0-9]+)_.*_[0-9]+px_repeat1$/\1/'); name=$(echo $run | sed -E 's/^.*_n[0-9]+_(.*)_[0-9]+px_repeat1$/\1/')
  echo "=== $(date +%H:%M:%S) $run: MODEL_NAME=$name N_ITERS=$n seeds=$(ls -d $d/seed_* | wc -l)"
  BENCH=genai MODEL_NAME=$name N_ITERS=$n RUN_DIR=$d bash /scratch/dars11/git/tar_reason/scripts/eval/official_judges_tamia.sh || { echo "FAILED $run"; FAIL=1; }
  mkdir -p $LUSTRE_OUT/$run
  rsync -a --exclude '*.png' --exclude '*.jpg' --exclude '*.jpeg' $d/ $LUSTRE_OUT/$run/
done
[ -d $OUT/figures_clipt5 ] && rsync -a $OUT/figures_clipt5/ $LUSTRE_OUT/figures_clipt5/ || true
echo "=== $(date +%H:%M:%S) all done (fail=$FAIL)"
exit $FAIL
