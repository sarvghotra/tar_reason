#!/bin/bash
# Gemma-reward twin of E-450 (user, 2026-11-03: "We should do what we did for E-450. For fair comparison with E-450"):
# E's recipe exactly (bash_fir_promptmix.sh: t1 recipe from SFT t21, BRANCH 8,4, hybrid critiques oracle frac 0.5, LR 1e-4,
# promptmix train_17K, one refinement round) with the pixel reward swapped to Gemma 4 12B it (PIXEL_JUDGE=gemma) and the
# validation still scored by Qwen3-VL-8B (VAL_JUDGE=qwen, second server on the judge GPU). Evals every 50 steps as E, on
# val800 only (user: IID only; val.yaml pre-created with one set). Evals at 450 (= E-450) and 500 only (user: no time for every-50).
# Chain of N_JOBS 3-hour jobs (Fir starts those within minutes; 24 h jobs wait days), SAVE_STEPS=5, each resumes from the
# newest checkpoint and exits at once when done. DATASET_SEED: set explicitly (see the user's choice in handoff/fir.md).
#   DATASET_SEED=<n> bash scripts/rl_ft/submit_gemma_E_fir.sh
# Smaller reward model (user 2026-11-05, moved from Rorqual during its maintenance): PIXEL_JUDGE=gemma_e4b
#   RUN_NAME=darshan_pm1_E_gemmaE4B_fir JOB_NAME=darshan_pm1EgemmaE4B_fir DATASET_SEED=87002 bash scripts/rl_ft/submit_gemma_E_fir.sh
set -eu
cd /home/dars11/scratch/git/tar_reason
export PATH=$PATH:/opt/software/slurm/bin
RUN_NAME=${RUN_NAME:-darshan_pm1_E_gemma_fir}
PIXEL_JUDGE=${PIXEL_JUDGE:-gemma}
case ${PIXEL_JUDGE} in gemma) JM=gemma-4-12B-it ;; gemma_e4b) JM=gemma-4-E4B-it ;; *) echo "PIXEL_JUDGE must be gemma or gemma_e4b" >&2; exit 1 ;; esac
N1=${N1:-13}   # phase 1: to step 450 (eval at 450 only)
N2=${N2:-3}    # phase 2: 450 -> 500 (eval at 500); extra jobs exit at once
: "${DATASET_SEED:?set DATASET_SEED explicitly (E-450 used 421)}"
mkdir -p output_dir/${RUN_NAME}
[ -s output_dir/${RUN_NAME}/val.yaml ] || printf 'datasets:\n    - json_path:\n      - %s\n      logname: val\n      ratio: 1\n' /home/dars11/scratch/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl > output_dir/${RUN_NAME}/val.yaml   # IID set only
for f in output_dir/${RUN_NAME}/val.yaml /home/dars11/scratch/stage/envs_tar_qwenjudge.tar.zst /home/dars11/scratch/stage/envs_tar_gemmajudge.tar.zst \
         /home/dars11/scratch/models/vlm/${JM}/config.json /home/dars11/scratch/models/vlm/Qwen3-VL-8B-Instruct/config.json \
         /home/dars11/scratch/models/tar/sft/slf_ref_edit_t21_ckpt17000/config.json /home/dars11/scratch/data/promptmix/train_17K.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
grep -q promptmix output_dir/${RUN_NAME}/val.yaml && { echo "val.yaml lists promptmix: IID (val800) only" >&2; exit 1; }
# User 2026-11-03: no time for intermediate evals: IID (val800) evals only at steps 450 and 500, no step-0 eval; checkpoints
# every 5 steps (a 3 h limit then loses < 25 min at ~5 min/step), all kept (--save_total_limit via EXTRA_ARGS overrides the launcher's 5).
export PIXEL_JUDGE VAL_JUDGE=qwen BRANCH=8,4 RUN_NAME DATASET_SEED SAVE_STEPS=5
export EXTRA_ARGS="--no-eval_on_start --save_total_limit 120"
export PREV_STAGE_CHECKPOINT=/home/dars11/scratch/models/tar/sft/slf_ref_edit_t21_ckpt17000
export TRAIN_JSONL=/home/dars11/scratch/data/promptmix/train_17K.jsonl
# TIME: per-job limit (2026-11-05 probes: Fir starts 4-node 6-12 h jobs as soon as 3 h ones, 24 h ones wait days).
COMMON=(--export=ALL --exclude=fc10212 --time=${TIME:-02:59:00} -J ${JOB_NAME:-darshan_pm1Egemma_fir})
PREV=$(MAX_STEPS=450 EVAL_STEPS=450 sbatch --parsable "${COMMON[@]}" scripts/rl_ft/bash_fir_promptmix.sh); IDS=$PREV
for i in $(seq 2 ${N1}); do PREV=$(MAX_STEPS=450 EVAL_STEPS=450 sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} scripts/rl_ft/bash_fir_promptmix.sh); IDS="$IDS $PREV"; done
for i in $(seq 1 ${N2}); do PREV=$(MAX_STEPS=500 EVAL_STEPS=500 sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} scripts/rl_ft/bash_fir_promptmix.sh); IDS="$IDS $PREV"; done
echo "submitted $((N1 + N2)) chained 3 h jobs: ${IDS} (${RUN_NAME}: E-450 recipe, ${PIXEL_JUDGE} reward, Qwen validation judge, seed ${DATASET_SEED}; evals at 450 and 500 only)"
