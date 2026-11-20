#!/bin/bash
# 4-round ORC run with an in-loop rollback verifier (user 2026-11-19): E-450_ORC's recipe (run 1, darshan_mr_E450_orc_fir:
# start + KL reference = E-450 merged, fresh LoRA r64, oracle critiques, chain mode to depth 4 with 4 siblings at a random
# round, MDP context, children_r1 draft reward, break weight 2, KL 0.05, LR 5e-5) plus:
#   --verdict_coef 0.1   after every refined image the policy answers "Does this image match the prompt better than the
#                        previous image?" with the 8B judge's label (better / worse by > 0.02 AM, ties skipped): a CE term
#                        next to GRPO. Training rollouts are otherwise unchanged (no rollback in training).
#   --eval_rollback      extra validation pass val_rb*/: own critiques + own verdicts, a "No" goes back to the previous image.
# Taken from later runs: 168 prompts/step (PROMPTS_PER_GPU=14, run C / jeet's sstg2_t1: more sibling groups per depth),
# promptmix v1 + v2 data and three val sets (run C), LR kept at 5e-5 (run C's 1e-4 diverged after step 300).
# Why (self-selection analysis, 2026-11-19): 38-44% of stage-3 chains end below an earlier image; keeping the better
# image is worth +2.9-4.2 AM (self-scored) to +5.6-7.2 (judge), mostly edits that broke an image that still had errors.
#
#   smoke:  SMOKE=1 bash scripts/rl_ft/submit_orc_verdict.sh           (1 node, 2 steps, evals on 24 prompts)
#   run:    bash scripts/rl_ft/submit_orc_verdict.sh                   (Fir / Rorqual / Nibi / TamIA, picked by host)
# Overridable: RUN_NAME, NODES, MAX_STEPS, PROMPTS_PER_GPU, VERDICT_COEF, ACCOUNT, TIME, N_JOBS, SAVE_STEPS, AFTER.
set -eu
export PATH=$PATH:/opt/software/slurm/bin
export REPO=$(cd "$(dirname "$0")/../.." && pwd); cd $REPO
S=/home/dars11/scratch; TAMIA=0
[[ "${CC_CLUSTER:-}" == tamia* || "$(hostname)" == tamia* ]] && { S=/scratch/d/dars11; TAMIA=1; }
BASE=${BASE:-$S/models/tar/sft/darshan_pm1_E_merged450}
for f in $BASE/config.json $S/stage/envs_tar_qwenjudge.tar.zst $S/models/vlm/Qwen3-VL-8B-Instruct/config.json \
         $S/data/promptmix/train_17K.jsonl $S/data/promptmix_v2/train_19K_minus_val800.jsonl $S/data/promptmix/val_800.jsonl \
         $S/data/promptmix_v2/val_800.jsonl $S/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
grep -q -- '--verdict_coef' llava/train/rl/train_grpo.py || { echo "this checkout has no --verdict_coef" >&2; exit 1; }
if [ "${SMOKE:-0}" = 1 ]; then
  RUN_NAME=${RUN_NAME:-orc_verdict_smoke}
  export MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24 PROMPTS_PER_GPU=${PROMPTS_PER_GPU:-2}
  SB=(--nodes=1 --time=01:00:00); N_JOBS=1
else
  RUN_NAME=${RUN_NAME:-darshan_orc_verdict_E450}
  export MAX_STEPS=${MAX_STEPS:-300} EVAL_STEPS=100 PROMPTS_PER_GPU=${PROMPTS_PER_GPU:-14}
  if [ $TAMIA = 1 ]; then TIME=${TIME:-02:59:00}; N_JOBS=${N_JOBS:-15}; export SAVE_STEPS=${SAVE_STEPS:-5}
  else TIME=${TIME:-11:59:00}; N_JOBS=${N_JOBS:-5}; export SAVE_STEPS=${SAVE_STEPS:-10}; fi
  SB=(--nodes=${NODES:-4} --time=$TIME)   # NODES=2 with PROMPTS_PER_GPU=28 keeps 168 prompts/step
fi
O=output_dir/$RUN_NAME; mkdir -p $O output_dir/slurm_logs
cat > $O/data.yaml <<EOF
# promptmix v1 train_17K + promptmix v2 (18,635 rows, v2 val removed), as run C; pooled, shuffled by DATASET_SEED.
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
# Launcher defaults kept (= run 1): BRANCH=2,1,1,1,1, CHAIN_SIBLINGS=4, CONTEXT_MODE=mdp, EVAL_BRANCH=1,1,1,1,1 (depth 4),
# LR=5e-5, KL 0.05, DRAFT_REWARD=children_r1, BREAK_WEIGHT=2, --eval_oracle. DATASET_SEED derives from RUN_NAME.
export RUN_NAME PREV_STAGE_CHECKPOINT=$BASE PIXEL_JUDGE=qwen CRITIQUE_SOURCE=oracle
export EXTRA_ARGS="--verdict_coef ${VERDICT_COEF:-0.1} --verdict_margin 0.02 --eval_rollback --save_total_limit 100"
if [[ "${CC_CLUSTER:-}" == rorqual* || "$(hostname)" == rorqual* ]]; then LAUNCHER=scripts/rl_ft/bash_rorqual_mr.sh; EXCL=()
elif [ $TAMIA = 1 ]; then LAUNCHER=scripts/rl_ft/bash_tamia_mr.sh; EXCL=(); ACCOUNT=${ACCOUNT:-aip-agrawal}
elif [[ "${CC_CLUSTER:-}" == nibi* || "$(hostname -f)" == *nibi* ]]; then
  LAUNCHER=scripts/rl_ft/bash_fir_mr.sh; EXCL=(); export SBATCH_MEM_PER_NODE=${SBATCH_MEM_PER_NODE:-400G} SRUN_MEM=""
else LAUNCHER=scripts/rl_ft/bash_fir_mr.sh; EXCL=(--exclude=${EXCLUDE:-fc10212}); fi
COMMON=(--export=ALL "${EXCL[@]}" --account=${ACCOUNT:-rrg-bengioy-ad_gpu} "${SB[@]}" -J ${JOB_NAME:-${RUN_NAME}}
        -o $REPO/output_dir/slurm_logs/%x_%j.out -e $REPO/output_dir/slurm_logs/%x_%j.err)
PREV=$(sbatch --parsable "${COMMON[@]}" ${AFTER:+--dependency=afterany:${AFTER}} ${LAUNCHER}); IDS=$PREV
for i in $(seq 2 ${N_JOBS}); do PREV=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} ${LAUNCHER}); IDS="$IDS $PREV"; done
echo "submitted ${N_JOBS} job(s): ${IDS} (${RUN_NAME}, base=${BASE}, ${PROMPTS_PER_GPU} prompts/GPU, EXTRA_ARGS=${EXTRA_ARGS})"
