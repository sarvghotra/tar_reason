#!/bin/bash
# Gemma-reward twin of E-450 (user, 2026-11-03: "We should do what we did for E-450. For fair comparison with E-450"):
# E's recipe exactly (bash_tamia_promptmix.sh: t1 recipe from SFT t21, BRANCH 8,4, hybrid critiques oracle frac 0.5, LR 1e-4,
# promptmix train_17K, one refinement round) with the pixel reward swapped to Gemma 4 12B it (PIXEL_JUDGE=gemma) and the
# validation still scored by Qwen3-VL-8B (VAL_JUDGE=qwen, second server on the judge GPU). Evals every 50 steps as E, on
# val800 only (user: IID only; val.yaml pre-created with one set). MAX_STEPS 450 = the E checkpoint we compare with.
# Chain of N_JOBS 3-hour jobs (Fir starts those within minutes; 24 h jobs wait days), SAVE_STEPS=5, each resumes from the
# newest checkpoint and exits at once when done. DATASET_SEED: set explicitly (see the user's choice in handoff/fir.md).
#   DATASET_SEED=<n> bash scripts/rl_ft/submit_gemma_E_tamia.sh
set -eu
cd /scratch/d/dars11/git/tar_reason_pm
export PATH=$PATH:/opt/software/slurm/bin
RUN_NAME=${RUN_NAME:-darshan_pm1_E_gemma_tamia}
N_JOBS=${N_JOBS:-14}
: "${DATASET_SEED:?set DATASET_SEED explicitly (E-450 used 421)}"
for f in output_dir/${RUN_NAME}/val.yaml /scratch/d/dars11/stage/envs_tar_bothjudges.tar.zst \
         /scratch/d/dars11/models/vlm/gemma-4-12B-it/config.json /scratch/d/dars11/models/vlm/Qwen3-VL-8B-Instruct/config.json \
         /scratch/d/dars11/models/tar/sft/slf_ref_edit_t21_ckpt17000/config.json /scratch/d/dars11/data/promptmix/train_17K.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
grep -q promptmix output_dir/${RUN_NAME}/val.yaml && { echo "val.yaml lists promptmix: IID (val800) only" >&2; exit 1; }
export PIXEL_JUDGE=gemma VAL_JUDGE=qwen BRANCH=8,4 RUN_NAME DATASET_SEED MAX_STEPS=${MAX_STEPS:-450} EVAL_STEPS=50 SAVE_STEPS=5
export PREV_STAGE_CHECKPOINT=/scratch/d/dars11/models/tar/sft/slf_ref_edit_t21_ckpt17000
export TRAIN_JSONL=/scratch/d/dars11/data/promptmix/train_17K.jsonl
COMMON=(--export=ALL --time=02:59:00 -J darshan_pm1Egemma_tamia)
PREV=$(sbatch --parsable "${COMMON[@]}" scripts/rl_ft/bash_tamia_promptmix.sh); IDS=$PREV
for i in $(seq 2 ${N_JOBS}); do PREV=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} scripts/rl_ft/bash_tamia_promptmix.sh); IDS="$IDS $PREV"; done
echo "submitted ${N_JOBS} chained 3 h jobs: ${IDS} (${RUN_NAME}: E-450 recipe, Gemma reward, Qwen validation judge, seed ${DATASET_SEED}, ${MAX_STEPS} steps)"
