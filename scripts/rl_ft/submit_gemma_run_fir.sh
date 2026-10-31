#!/bin/bash
# Submit the Gemma-4-12B-reward RL run on Fir (user, 2026-10-30): run 1's recipe from E-450 (bash_fir_mr.sh, oracle
# critiques = refiner-only, promptmix train_17K, 500 steps, depth-4 mdp evals every 100 on val800 + promptmix val_800)
# with the pixel judge swapped from Qwen3-VL-8B (GenEval2's judge) to Gemma 4 12B it (PIXEL_JUDGE=gemma). Two chained
# 24 h jobs (the first timed out at step 423 for run 1). The run dir's data.yaml / val.yaml were copied from run 1.
#   bash scripts/rl_ft/submit_gemma_run_fir.sh            # after smoke_gemma_judge_fir.sh passed
set -eu
cd /home/dars11/scratch/git/tar_reason
export PATH=$PATH:/opt/software/slurm/bin
RUN_NAME=${RUN_NAME:-darshan_mr_E450_gemma_fir}
for f in output_dir/${RUN_NAME}/data.yaml output_dir/${RUN_NAME}/val.yaml /home/dars11/scratch/stage/envs_tar_gemmajudge.tar.zst \
         /home/dars11/scratch/models/vlm/gemma-4-12B-it/model.safetensors /home/dars11/scratch/models/tar/sft/darshan_pm1_E_merged450/config.json; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
# Settings through the environment + --export=ALL (an --export list splits values at commas, see submit_distill_data_orc_fir.sh).
export PIXEL_JUDGE=gemma CRITIQUE_SOURCE=oracle RUN_NAME MAX_STEPS=${MAX_STEPS:-500} EVAL_STEPS=100 SAVE_STEPS=10
export PREV_STAGE_CHECKPOINT=/home/dars11/scratch/models/tar/sft/darshan_pm1_E_merged450
export TRAIN_JSONL=/home/dars11/scratch/data/promptmix/train_17K.jsonl
COMMON=(--export=ALL --exclude=fc10212 -J darshan_mrE450gemma_fir)
J1=$(sbatch --parsable "${COMMON[@]}" scripts/rl_ft/bash_fir_mr.sh)
J2=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${J1} scripts/rl_ft/bash_fir_mr.sh)
echo "submitted ${J1} -> ${J2} (${RUN_NAME}, PIXEL_JUDGE=gemma, oracle critiques, 500 steps)"
