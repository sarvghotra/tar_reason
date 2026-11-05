#!/bin/bash
# Run 1's recipe from the Gemma-reward E-twin's checkpoint-450 (user, 2026-11-05: "run this Gemma 12b experiment 450
# checkpoint for iterations, exactly same as run-1"). Run 1 = darshan_mr_E450_orc_fir: bash_fir_mr.sh (N=4 chain mode,
# MDP, children_r1, LR 5e-5, fresh LoRA r64, KL 0.05), oracle critiques (refiner only), promptmix train_17K, 500 steps,
# depth-4 mdp evals every 100 (incl. step 0) on val800 + promptmix val_800 (own + oracle), DATASET_SEED=81556.
# Differences, all chosen by the user: base = darshan_pm1_E_gemma_fir checkpoint-450 merged onto SFT t21 (run 1: E-450
# merged); reward = Gemma 4 12B it (PIXEL_JUDGE=gemma, shared image prefix with the soft-cap fix, checked 2026-11-05);
# validation still scored by Qwen3-VL-8B (VAL_JUDGE=qwen), as run 1. Same prompt order as run 1 (its DATASET_SEED).
#   bash scripts/rl_ft/submit_gemma450_run1_fir.sh
set -eu
cd /home/dars11/scratch/git/tar_reason
export PATH=$PATH:/opt/software/slurm/bin
RUN_NAME=${RUN_NAME:-darshan_mr_Egemma450_fir}
M=/home/dars11/scratch/models/tar/sft
BASE=$M/darshan_pm1_E_gemma_merged450
ADAPTER=output_dir/darshan_pm1_E_gemma_fir/checkpoint-450
N_JOBS=${N_JOBS:-5}
TIME=${TIME:-11:59:00}
for f in $ADAPTER/adapter_model.safetensors $M/slf_ref_edit_t21_ckpt17000/config.json output_dir/darshan_mr_E450_orc_fir/data.yaml \
         output_dir/darshan_mr_E450_orc_fir/val.yaml /home/dars11/scratch/stage/envs_tar_gemmajudge.tar.zst \
         /home/dars11/scratch/stage/envs_tar_qwenjudge.tar.zst /home/dars11/scratch/models/vlm/gemma-4-12B-it/config.json \
         /home/dars11/scratch/models/vlm/Qwen3-VL-8B-Instruct/config.json; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
mkdir -p output_dir/${RUN_NAME}
cp -n output_dir/darshan_mr_E450_orc_fir/data.yaml output_dir/darshan_mr_E450_orc_fir/val.yaml output_dir/${RUN_NAME}/   # run 1's data / val sets
# 1) the adapter is merged onto SFT t21 (as E-450 was for run 1) by merge_then_fir_mr.sh inside the first job of the chain
#    (Fir's 1-GPU queue held a separate merge job for 20+ min on 2026-11-05); every chain job goes through the wrapper,
#    which skips the merge once merged_from.json exists, so a failed merge is retried by the next job.
export MERGE_BASE=$M/slf_ref_edit_t21_ckpt17000 MERGE_ADAPTER=$PWD/$ADAPTER
# 2) the run: chain of TIME jobs; each resumes from the newest checkpoint and exits at once when done
export PIXEL_JUDGE=gemma VAL_JUDGE=qwen CRITIQUE_SOURCE=oracle RUN_NAME DATASET_SEED=81556 MAX_STEPS=500 EVAL_STEPS=100 SAVE_STEPS=10
export PREV_STAGE_CHECKPOINT=$BASE TRAIN_JSONL=/home/dars11/scratch/data/promptmix/train_17K.jsonl
export EXTRA_ARGS="--save_total_limit 60"   # keep every checkpoint (every 10 steps, ~1.9 GB each)
COMMON=(--export=ALL --exclude=fc10212 --time=${TIME} -J ${JOB_NAME:-darshan_mrEgemma450_fir})
PREV=$(sbatch --parsable "${COMMON[@]}" scripts/rl_ft/merge_then_fir_mr.sh); IDS=$PREV
for i in $(seq 2 ${N_JOBS}); do PREV=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} scripts/rl_ft/merge_then_fir_mr.sh); IDS="$IDS $PREV"; done
echo "submitted ${N_JOBS} chained ${TIME} jobs: ${IDS} (${RUN_NAME})"
