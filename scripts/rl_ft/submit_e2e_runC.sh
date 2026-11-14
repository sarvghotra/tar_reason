#!/bin/bash
# Stage 3, run C (2026-11-14): run A's setup (t4-merged start + KL reference, fresh LoRA r64, own critiques trained,
# --critique_reward stop, STOP_PENALTY 0, MDP, children_r1, break 2, KL 0.05) with jeet's substage-2 fixes
# (his rl_ft_sstg2_t1 README): 168 prompts/step (PROMPTS_PER_GPU=14), 2 refinement rounds (BRANCH=2,1,1), LR 1e-4,
# promptmix v1 train_17K + promptmix v2 (jeet's 19K build minus its 800 v2 val prompts, 18,635 rows), 300 steps,
# depth-3 evals (EVAL_BRANCH=1,1,1,1) at 0/100/200/300 on GenEval2 val800 + promptmix val_800 + promptmix v2 val_800.
# Writes this run's data.yaml / val.yaml with the cluster's paths, then calls submit_e2e_fir.sh (which picks the launcher).
#
#   Fir / Rorqual:  bash scripts/rl_ft/submit_e2e_runC.sh
#   TamIA:          ACCOUNT=<your allocation> bash scripts/rl_ft/submit_e2e_runC.sh     (default 15 chained 2:59 jobs)
# Overridable: RUN_NAME, JOB_NAME, ACCOUNT, TIME, N_JOBS, SAVE_STEPS. Keep DATASET_SEED (prompt order) as is.
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
S=/home/dars11/scratch
TAMIA=0; [[ "${CC_CLUSTER:-}" == tamia* || "$(hostname)" == tamia* ]] && { S=/scratch/d/dars11; TAMIA=1; }
export RUN_NAME=${RUN_NAME:-darshan_e2e_stop_C_fir}
export DATASET_SEED=${DATASET_SEED:-57602}   # = the value submit/launcher derive from RUN_NAME=darshan_e2e_stop_C_fir: same prompt order on every cluster
O=output_dir/$RUN_NAME; mkdir -p $O
for f in $S/data/promptmix/train_17K.jsonl $S/data/promptmix_v2/train_19K_minus_val800.jsonl $S/data/promptmix/val_800.jsonl \
         $S/data/promptmix_v2/val_800.jsonl $S/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
cat > $O/data.yaml <<EOF
# Run C: promptmix v1 train_17K + promptmix v2 (18,635 rows, v2 val removed); pooled, shuffled by DATASET_SEED.
datasets:
    - json_path:
      - $S/data/promptmix/train_17K.jsonl
      ratio: 1
    - json_path:
      - $S/data/promptmix_v2/train_19K_minus_val800.jsonl
      ratio: 1
EOF
cat > $O/val.yaml <<EOF
datasets:
    - json_path:
      - $S/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl
      logname: val
      ratio: 1
    - json_path:
      - $S/data/promptmix/val_800.jsonl
      logname: promptmix
      ratio: 1
    - json_path:
      - $S/data/promptmix_v2/val_800.jsonl
      logname: promptmix_v2
      ratio: 1
EOF
export CRITIQUE_REWARD=stop PROMPTS_PER_GPU=14 BRANCH=2,1,1 EVAL_BRANCH=1,1,1,1 LR=1e-4 MAX_STEPS=300 EVAL_STEPS=100
if [ $TAMIA = 1 ]; then export TIME=${TIME:-02:59:00} N_JOBS=${N_JOBS:-15} SAVE_STEPS=${SAVE_STEPS:-5}
else export TIME=${TIME:-11:59:00} N_JOBS=${N_JOBS:-4} SAVE_STEPS=${SAVE_STEPS:-10}; fi
export JOB_NAME=${JOB_NAME:-${RUN_NAME}}
bash scripts/rl_ft/submit_e2e_fir.sh
