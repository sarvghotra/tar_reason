#!/bin/bash
#SBATCH --time=02:59:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH -J unigen_shard
#SBATCH -o /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.out
#SBATCH -e /scratch/dars11/git/tar_reason/output_dir/slurm_logs/%x_%j.err
# One shard of a unified-model baseline's GenAI-Bench-1600 images on one GPU, Rorqual or Nibi (same /scratch/dars11 layout)
# (2026-11-04: AlphaGRPO rt2i, stuck in Mila's queue for days; run on Rorqual, user). Same generator call as bench_official_tamia.sh / bench_jeet_mila.sh for GEN_FAMILY=alphagrpo
# (eval/baselines/unified_gen.py, N_ITERS=2, repeat 1, per-prompt seed = SEED*100003 + prompt index), but the prompts are
# split over independent 1-GPU jobs with --shard k/K (+ --resume, so a timed-out shard just needs the same sbatch again).
# After all shards of a seed: SHARD= (empty) runs the merge (no model load), writing <family>_gen.json + self_reflect.json
# and .gen_done; the judges then score the folder (official_judges_tamia.sh, bench_official_tamia.sh).
#   MODEL_NAME=alphagrpo_rt2i GEN_FAMILY=alphagrpo UNI_ARGS="--mode think" SEED=13 SHARD=0/4 sbatch scripts/eval/unified_gen_shard_alliance.sh
set -euo pipefail
REPO=/scratch/dars11/git/tar_reason
GEN_FAMILY=${GEN_FAMILY:?}; MODEL_NAME=${MODEL_NAME:?}; SEED=${SEED:?}; SHARD=${SHARD:-}
UNI_RES=${UNI_RES:-1024}; N_ITERS=${N_ITERS:-2}
OUTPUT_DIR=/scratch/dars11/tar_reason/results_jeet/genai-bench-1600
GEN_MODEL=${GEN_FAMILY}_n${N_ITERS}_${MODEL_NAME}_${UNI_RES}px_repeat1/seed_${SEED}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AGRPO_REPO=/scratch/dars11/baselines/AlphaGRPO
# Model reads from Rorqual's Lustre are slow: copy the HF cache entries the backend needs to node-local disk first.
STAGE_HF_MODELS=${STAGE_HF_MODELS:-models--ByteDance-Seed--BAGEL-7B-MoT models--huangrh9--AlphaGRPO-rt2i}
export HF_HOME=/scratch/dars11/hf_home
if [ -n "${SLURM_TMPDIR:-}" ] && [ -n "$SHARD" ]; then
  t0=$(date +%s); mkdir -p $SLURM_TMPDIR/hf_home/hub
  for m in $STAGE_HF_MODELS; do cp -a $HF_HOME/hub/$m $SLURM_TMPDIR/hf_home/hub/; done
  export HF_HOME=$SLURM_TMPDIR/hf_home; echo "staged HF models in $(( $(date +%s) - t0 )) s"
fi
read -r -a UNI_EXTRA <<< "${UNI_ARGS:-}"
SHARD_ARG=(); [ -n "$SHARD" ] && SHARD_ARG=(--shard "$SHARD")
echo "$(date +%H:%M:%S) ${GEN_MODEL} shard=${SHARD:-merge} job=${SLURM_JOB_ID:-local} host=$(hostname)"
cd $REPO
PYTHONPATH=$REPO ${UNI_ENV:-$HOME/envs/${GEN_FAMILY}}/bin/python eval/baselines/unified_gen.py --family ${GEN_FAMILY} "${UNI_EXTRA[@]}" \
  --n_iters ${N_ITERS} --repeat 1 --seed ${SEED} --output_dir ${OUTPUT_DIR} --gen_model ${GEN_MODEL} \
  --root_dir /scratch/dars11/benchmarks/GenAI-Bench --eval_set_dir GenAI-Image-1600 --resume "${SHARD_ARG[@]}" ${GEN_EXTRA:-}
if [ -z "$SHARD" ]; then
  n=$(ls ${OUTPUT_DIR}/${GEN_MODEL}/${N_ITERS} | grep -c -E 'jpe?g$|png$' || true)
  [ "$n" -ge 1600 ] && touch ${OUTPUT_DIR}/${GEN_MODEL}/.gen_done && echo "merged, ${n} images: .gen_done" || { echo "only ${n} images"; exit 1; }
fi
