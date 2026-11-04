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
# 2026-11-04: also for folders generated on Rorqual (IMAGES=lustre: copy RUNS from LUSTRE_OUT instead of the image tar) and
# for jeet's judge (JUDGES="clipt5 qwen35"; qwen35 = Qwen3.5-27B VQAScore via bench_official_tamia.sh's genai path, with
# TamIA's genai_judge env from /scratch/dars11/stage/genai_judge_env.tar; needs .gen_done in each seed folder). E.g.
#   RUNS=alphagrpo_n2_alphagrpo_rt2i_1024px_repeat1 IMAGES=lustre JUDGES="clipt5 qwen35" sbatch scripts/eval/rorqual_genai_judges.sh
T=${SLURM_TMPDIR:?}; S=/scratch/dars11/stage; HUB=/scratch/dars11/hf_home/hub
LUSTRE_OUT=/scratch/dars11/tar_reason/results_jeet/genai-bench-1600
IMAGES=${IMAGES:-tar}; JUDGES=${JUDGES:-clipt5}
OUT=$T/scratch/dars11/tar_reason/results_jeet/genai-bench-1600
t0=$(date +%s)
tar -C $T -xf $S/genai_judge_code.tar
mkdir -p $T/hf_home/hub $OUT
if [ "$IMAGES" = tar ]; then tar -C $T -xf $S/genai_baseline_images.tar
else for run in ${RUNS:?IMAGES=lustre needs RUNS}; do cp -r $LUSTRE_OUT/$run $OUT/; done; fi
case " $JUDGES " in *" clipt5 "*)
  for m in models--zhiqiulin--clip-flant5-xxl models--openai--clip-vit-large-patch14-336 models--google--flan-t5-xxl; do cp -a $HUB/$m $T/hf_home/hub/; done ;; esac
case " $JUDGES " in *" qwen35 "*)
  tar -C $T -xf $S/genai_judge_env.tar; cp -a $HUB/models--Qwen--Qwen3.5-27B $T/hf_home/hub/ ;; esac
echo "staged in $(( $(date +%s) - t0 )) s: $(du -sh $T | cut -f1)"
export CLIPT5_ENV=$T/home/dars11/envs/clipt5_judge T2V_CLIPT5=$T/scratch/dars11/git/t2v_metrics_clipt5 \
       T2V_FORK=$T/scratch/dars11/git/t2v_metrics_jeet BENCH_DIR=$T/scratch/dars11/benchmarks HF_HOME=$T/hf_home N_GPUS=4
RUNS=(${RUNS:-$(ls $OUT)})
FAIL=0
for run in "${RUNS[@]}"; do
  d=$OUT/$run
  n=$(echo $run | sed -E 's/^.*_n([0-9]+)_.*_[0-9]+px_repeat1$/\1/'); name=$(echo $run | sed -E 's/^.*_n[0-9]+_(.*)_[0-9]+px_repeat1$/\1/')
  fam=$(echo $run | sed -E 's/^(.*)_n[0-9]+_.*$/\1/'); res=$(echo $run | sed -E 's/^.*_([0-9]+)px_repeat1$/\1/')
  echo "=== $(date +%H:%M:%S) $run: MODEL_NAME=$name N_ITERS=$n seeds=$(ls -d $d/seed_* | wc -l) judges=$JUDGES"
  case " $JUDGES " in *" clipt5 "*)
    BENCH=genai MODEL_NAME=$name N_ITERS=$n RUN_DIR=$d bash /scratch/dars11/git/tar_reason/scripts/eval/official_judges_tamia.sh || { echo "FAILED clipt5 $run"; FAIL=1; } ;; esac
  case " $JUDGES " in *" qwen35 "*)
    ( cd /scratch/dars11/git/tar_reason && BENCH=genai MODEL_NAME=$name GEN_FAMILY=$fam UNI_N_ITERS=$n UNI_RES=$res \
      OUT_ROOT=$T/scratch/dars11/tar_reason/results_jeet GENAI_PY=$T/genai_judge/bin/python T2V_DIR=$T/scratch/dars11/git/t2v_metrics_jeet \
      JUDGE_HF_HOME=$T/hf_home BENCH_DIR=$T/scratch/dars11/benchmarks bash scripts/eval/bench_official_tamia.sh ) || { echo "FAILED qwen35 $run"; FAIL=1; } ;; esac
  mkdir -p $LUSTRE_OUT/$run
  rsync -a --exclude '*.png' --exclude '*.jpg' --exclude '*.jpeg' $d/ $LUSTRE_OUT/$run/
done
for f in figures_clipt5 figures; do [ -d $OUT/$f ] && rsync -a $OUT/$f/ $LUSTRE_OUT/$f/ || true; done
echo "=== $(date +%H:%M:%S) all done (fail=$FAIL)"
exit $FAIL
