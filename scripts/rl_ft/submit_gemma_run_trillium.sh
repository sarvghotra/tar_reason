#!/bin/bash
# Submit the Gemma-4-12B-reward RL run on Trillium (user, 2026-10-30 20:10 PDT: "launch the gemma reward run on trillium";
# Fir's copy, submit_gemma_run_fir.sh, cancelled): run 1's recipe from E-450 (bash_trillium_mr.sh, oracle critiques =
# refiner-only, promptmix train_17K, 500 steps, depth-4 mdp evals every 100 on val800 + promptmix val_800) with the pixel
# judge swapped from Qwen3-VL-8B to Gemma 4 12B it (PIXEL_JUDGE=gemma). Two chained 24 h jobs (run 1 timed out at step 423
# in 24 h on Fir). The run dir's val.yaml (two sets, logname val + promptmix) must exist; data.yaml is written by the launcher.
#   bash scripts/rl_ft/submit_gemma_run_trillium.sh       # after mr_smoke_trillium + test_pixel_judge.py passed
set -eu
cd /scratch/dars11/git/tar_reason
RUN_NAME=${RUN_NAME:-darshan_mr_E450_gemma_trillium}
for f in output_dir/${RUN_NAME}/val.yaml /scratch/dars11/stage/envs_tar_gemmajudge.tar.zst \
         /scratch/dars11/models/vlm/gemma-4-12B-it/model.safetensors /scratch/dars11/models/tar/sft/darshan_pm1_E_merged450/config.json \
         /scratch/dars11/data/promptmix/train_17K.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
grep -q 'logname: promptmix' output_dir/${RUN_NAME}/val.yaml || { echo "val.yaml lacks the promptmix set" >&2; exit 1; }
# Settings through the environment + --export=ALL (an --export list splits values at commas, see submit_distill_data_orc_fir.sh).
export PIXEL_JUDGE=gemma CRITIQUE_SOURCE=oracle RUN_NAME MAX_STEPS=${MAX_STEPS:-500} EVAL_STEPS=100 SAVE_STEPS=10
export PREV_STAGE_CHECKPOINT=/scratch/dars11/models/tar/sft/darshan_pm1_E_merged450
export TRAIN_JSONL=/scratch/dars11/data/promptmix/train_17K.jsonl
COMMON=(--export=ALL -J darshan_mrE450gemma_trillium)
J1=$(sbatch --parsable "${COMMON[@]}" scripts/rl_ft/bash_trillium_mr.sh)
J2=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${J1} scripts/rl_ft/bash_trillium_mr.sh)
echo "submitted ${J1} -> ${J2} (${RUN_NAME}, PIXEL_JUDGE=gemma, oracle critiques, 500 steps)"
