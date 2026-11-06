#!/bin/bash
# E-450 critique ablations (user, 2026-11-06, both priority): E's recipe exactly (t1 recipe from SFT t21, fresh LoRA, LR 1e-4,
# BRANCH 8,4 = 8 drafts x 4 refines, one refinement round ("2 iter" = draft + 1 refine), std floor 0.04, BREAK_WEIGHT 2, Qwen3-VL-8B
# reward, promptmix train_17K, DATASET_SEED=87002 = E's prompt order, evals every 50 on val800 + promptmix val_400, own + oracle)
# with only the training critiques changed:
#   1) CRITIQUE_SOURCE=policy  every child samples its own critique and the critique tokens are trained (no oracle critiques)
#   2) CRITIQUE_SOURCE=mixed   per parent a coin: 50% oracle (as E), 50% own critiques sampled per child and trained
#      (E's "hybrid" used the frozen policy's critique, untrained, for that half)
# Step 0 = SFT t21 = E's step 0, so no step-0 eval. Chain of N_JOBS x TIME jobs, each resumes from the newest checkpoint.
#   CRITIQUE_SOURCE=policy RUN_NAME=darshan_pm1_E_policy_fir  bash scripts/rl_ft/submit_E_critique_ablation.sh   (Fir)
#   CRITIQUE_SOURCE=mixed  RUN_NAME=darshan_pm1_E_mixed_tamia bash scripts/rl_ft/submit_E_critique_ablation.sh   (TamIA)
#   CRITIQUE_SOURCE=policy RUN_NAME=darshan_pm1_E_policy_rorqual bash scripts/rl_ft/submit_E_critique_ablation.sh   (Rorqual copy of the Fir one)
set -eu
: "${CRITIQUE_SOURCE:?policy | mixed}" "${RUN_NAME:?}"
export PATH=$PATH:/opt/software/slurm/bin
if [ -d /scratch/d/dars11 ]; then   # TamIA
  cd /scratch/d/dars11/git/tar_reason_pm; S=/scratch/d/dars11; LAUNCHER=scripts/rl_ft/bash_tamia_promptmix.sh; EXCL=()
elif [[ "${CC_CLUSTER:-}" == rorqual* || "$(hostname)" == rorqual* ]]; then   # Rorqual (same paths and account as Fir)
  cd /home/dars11/scratch/git/tar_reason; S=/home/dars11/scratch; LAUNCHER=scripts/rl_ft/bash_rorqual_promptmix.sh; EXCL=()
else                               # Fir
  cd /home/dars11/scratch/git/tar_reason; S=/home/dars11/scratch; LAUNCHER=scripts/rl_ft/bash_fir_promptmix.sh; EXCL=(--exclude=fc10212)
fi
N_JOBS=${N_JOBS:-4}
TIME=${TIME:-11:59:00}
for f in $S/stage/envs_tar_qwenjudge.tar.zst $S/models/vlm/Qwen3-VL-8B-Instruct/config.json \
         $S/models/tar/sft/slf_ref_edit_t21_ckpt17000/config.json $S/data/promptmix/train_17K.jsonl $S/data/promptmix/val_400.jsonl \
         $S/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
grep -q '"mixed"' llava/train/rl/train_grpo.py || { echo "this checkout has no --critique_source mixed" >&2; exit 1; }
export CRITIQUE_SOURCE RUN_NAME HYBRID_ORACLE_FRAC=0.5 PIXEL_JUDGE=qwen BRANCH=8,4 DATASET_SEED=87002
export MAX_STEPS=${MAX_STEPS:-450} EVAL_STEPS=50 SAVE_STEPS=10
export PREV_STAGE_CHECKPOINT=$S/models/tar/sft/slf_ref_edit_t21_ckpt17000 TRAIN_JSONL=$S/data/promptmix/train_17K.jsonl
export EXTRA_ARGS="--no-eval_on_start --save_total_limit 60"   # keep every checkpoint (every 10 steps)
COMMON=(--export=ALL "${EXCL[@]}" --time=${TIME} -J ${JOB_NAME:-${RUN_NAME}})
PREV=$(sbatch --parsable "${COMMON[@]}" ${LAUNCHER}); IDS=$PREV
for i in $(seq 2 ${N_JOBS}); do PREV=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} ${LAUNCHER}); IDS="$IDS $PREV"; done
echo "submitted ${N_JOBS} chained ${TIME} jobs: ${IDS} (${RUN_NAME}, critique_source=${CRITIQUE_SOURCE})"
