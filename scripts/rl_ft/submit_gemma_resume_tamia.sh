#!/bin/bash
# Continue the Gemma-4-12B-reward RL run on TamIA from the Trillium checkpoint (third copy; user 2026-11-03 "can we run this on tamia too?") (user, 2026-11-03: "run the gemma reward here",
# "only the IID evals", jeet: short jobs start on Fir, 24 h ones queue for days). Same recipe as submit_gemma_run_trillium.sh
# (bash_tamia_mr.sh PIXEL_JUDGE=gemma, oracle critiques, E-450, promptmix train_17K, 500 steps, DATASET_SEED=98381 so the prompt
# order continues Trillium's), as a chain of N_JOBS 3-hour jobs (partition gpubase_bynode_b1, 4 nodes x (3 ranks + judge)):
# each job resumes from the newest output_dir/${RUN_NAME}/checkpoint-*, SAVE_STEPS=5 bounds the loss at a time limit to 4 steps,
# and a job that finds checkpoint >= MAX_STEPS exits at once. Evals every 100 steps on val800 only (val.yaml here has one set;
# Trillium's steps 0-200 also had promptmix val_800). Trillium keeps running too (user 2026-11-03): two copies from one checkpoint,
# separate wandb runs (WANDB_RUN_ID defaults to RUN_NAME); sync each job's folder oldest first, later ones with `wandb sync --append`.
# Prerequisite: the Trillium checkpoint copied to output_dir/${RUN_NAME}/checkpoint-<N>/ (adapter + optimizer.pt + scheduler.pt + state.json).
#   bash scripts/rl_ft/submit_gemma_resume_tamia.sh            # N_JOBS=10 by default
set -eu
cd /scratch/d/dars11/git/tar_reason_pm
export PATH=$PATH:/opt/software/slurm/bin
RUN_NAME=${RUN_NAME:-darshan_mr_E450_gemma_tamia}
N_JOBS=${N_JOBS:-10}
LAST=$(ls -d output_dir/${RUN_NAME}/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
[ -n "$LAST" ] && [ -s output_dir/${RUN_NAME}/checkpoint-${LAST}/state.json ] || { echo "no resumable checkpoint in output_dir/${RUN_NAME}" >&2; exit 1; }
for f in output_dir/${RUN_NAME}/val.yaml /scratch/d/dars11/stage/envs_tar_gemmajudge.tar.zst \
         /scratch/d/dars11/models/vlm/gemma-4-12B-it/config.json /scratch/d/dars11/models/tar/sft/darshan_pm1_E_merged450/config.json \
         /scratch/d/dars11/data/promptmix/train_17K.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
grep -q promptmix output_dir/${RUN_NAME}/val.yaml && { echo "val.yaml still lists promptmix: IID (val800) only" >&2; exit 1; }
export PIXEL_JUDGE=gemma CRITIQUE_SOURCE=oracle RUN_NAME MAX_STEPS=${MAX_STEPS:-500} EVAL_STEPS=100 SAVE_STEPS=5
export DATASET_SEED=98381   # WANDB_RUN_ID defaults to RUN_NAME: the Trillium copy keeps running (user 2026-11-03), separate wandb runs
export PREV_STAGE_CHECKPOINT=/scratch/d/dars11/models/tar/sft/darshan_pm1_E_merged450
export TRAIN_JSONL=/scratch/d/dars11/data/promptmix/train_17K.jsonl
COMMON=(--export=ALL --time=02:59:00 -J darshan_mrE450gemma_tamia)
PREV=$(sbatch --parsable "${COMMON[@]}" scripts/rl_ft/bash_tamia_mr.sh); IDS=$PREV
for i in $(seq 2 ${N_JOBS}); do PREV=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} scripts/rl_ft/bash_tamia_mr.sh); IDS="$IDS $PREV"; done
echo "submitted ${N_JOBS} chained 3 h jobs: ${IDS} (${RUN_NAME}, resume from checkpoint-${LAST}, PIXEL_JUDGE=gemma, oracle critiques, to step ${MAX_STEPS}, val800 evals only)"
