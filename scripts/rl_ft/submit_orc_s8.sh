#!/bin/bash
# Two E-450 refinement-RL runs with 8 siblings per group (user 2026-11-22), Rorqual / Fir / TamIA (picked by host):
#   VARIANT=jeet  "build on sstg2_t1": jeet's code (commit 1105427, copied from his Fir checkout to
#                 ~/scratch/git/tar_reason_jeet) and his rl_ft_sstg2_t1 settings (1 draft per prompt, drafts frozen with
#                 their own KL, siblings at round 1 or 2 weighted 1:4, hybrid 50/50 critiques with the critic anchored,
#                 KL ratio cap 20, LR 1e-4, 24 prompts/GPU = 288 chains/step), with CHAIN_SIBLINGS 4 -> 8.
#   VARIANT=r1    "E-450_ORC recipe, round 1": our code, oracle critiques, 2 drafts, 168 prompts/step, LR 5e-5, every
#                 sibling group at round 1 (BRANCH=2,1) with 8 siblings.
# Both: start + KL reference = E-450 merged, fresh LoRA r64, Qwen3-VL-8B pixel judge, KL 0.05, children_r1, break 2, MDP;
# our leak-free data (promptmix v1 train_17K + v2 minus its val 800; jeet's k1-4_one30 / AlphaGRPO files are not
# readable from our account) and our 3 val sets with depth-4 own + oracle chains every 100 steps; 300 steps.
#   smoke: SMOKE=1 VARIANT=jeet bash scripts/rl_ft/submit_orc_s8.sh      (1 node, 2 steps, 24 eval prompts)
#   run:   VARIANT=jeet bash scripts/rl_ft/submit_orc_s8.sh              (AFTER=<smoke id> to start only if it passed)
# Overridable: LR, MORE_ARGS (appended trainer flags, e.g. --no-eval_on_start), RUN_NAME, JOB_NAME, MAX_STEPS, PROMPTS_PER_GPU, TIME, N_JOBS, SAVE_STEPS, ACCOUNT, AFTER.
set -eu
export PATH=$PATH:/opt/software/slurm/bin
OURS=$(cd "$(dirname "$0")/../.." && pwd); cd $OURS
S=/home/dars11/scratch; HOSTF=$(hostname -f)
if [[ "${CC_CLUSTER:-}" == tamia* || "$HOSTF" == *tamia* ]]; then
  S=/scratch/d/dars11; LAUNCHER=scripts/rl_ft/bash_tamia_mr.sh; EXCL=(); DEF_ACCOUNT=aip-agrawal
elif [[ "${CC_CLUSTER:-}" == rorqual* || "$HOSTF" == *rorqual* ]]; then
  LAUNCHER=scripts/rl_ft/bash_rorqual_mr.sh; EXCL=(); DEF_ACCOUNT=rrg-bengioy-ad_gpu
else
  LAUNCHER=scripts/rl_ft/bash_fir_mr.sh; EXCL=(--exclude=${EXCLUDE:-fc10212}); DEF_ACCOUNT=rrg-bengioy-ad_gpu
fi
BASE=$S/models/tar/sft/darshan_pm1_E_merged450
case ${VARIANT:?VARIANT=jeet or r1} in
  jeet) export REPO=$S/git/tar_reason_jeet RUN_NAME=${RUN_NAME:-darshan_sstg2_s8} BRANCH=1,1,1 LR=${LR:-1e-4} \
               CRITIQUE_SOURCE=hybrid HYBRID_ORACLE_FRAC=0.5 VAL_DUMP=0 PPG=24
        grep -q -- '--chain_round_weights' $REPO/llava/train/rl/train_grpo.py || { echo "$REPO is not jeet's 1105427 tree" >&2; exit 1; }
        export EXTRA_ARGS="--adv_rounds 1,2 --frozen_kl_coef 0.2 --chain_round_weights 1,4 --kl_ratio_max 20 --anchor_critiques --critique_anchor_weight 1.0 --save_keep_every 100 ${MORE_ARGS:-}" ;;
  r1)   export REPO=$OURS RUN_NAME=${RUN_NAME:-darshan_orc_r1s8} BRANCH=2,1 LR=${LR:-5e-5} CRITIQUE_SOURCE=oracle VAL_DUMP=1 PPG=14
        export EXTRA_ARGS="--save_total_limit 100 ${MORE_ARGS:-}" ;;
  *) echo "VARIANT must be jeet or r1" >&2; exit 1 ;;
esac
export CHAIN_SIBLINGS=8 PREV_STAGE_CHECKPOINT=$BASE PIXEL_JUDGE=qwen
for f in $BASE/config.json $S/data/promptmix/train_17K.jsonl $S/data/promptmix_v2/train_19K_minus_val800.jsonl \
         $S/data/promptmix/val_800.jsonl $S/data/promptmix_v2/val_800.jsonl $S/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
if [ "${SMOKE:-0}" = 1 ]; then
  RUN_NAME=${RUN_NAME}_smoke
  export MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24 PROMPTS_PER_GPU=2
  SB=(--nodes=1 --time=01:30:00); N_JOBS=1
else
  export MAX_STEPS=${MAX_STEPS:-300} EVAL_STEPS=100 SAVE_STEPS=${SAVE_STEPS:-10} PROMPTS_PER_GPU=${PROMPTS_PER_GPU:-$PPG}
  SB=(--nodes=4 --time=${TIME:-11:59:00}); N_JOBS=${N_JOBS:-5}   # short jobs: TIME=02:59:00 SAVE_STEPS=5 N_JOBS=15
fi
export RUN_NAME
O=$REPO/output_dir/$RUN_NAME; mkdir -p $O $OURS/output_dir/slurm_logs
cat > $O/data.yaml <<EOF
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
COMMON=(--export=ALL "${EXCL[@]}" --account=${ACCOUNT:-$DEF_ACCOUNT} "${SB[@]}" -J ${JOB_NAME:-${RUN_NAME}}
        -o $OURS/output_dir/slurm_logs/%x_%j.out -e $OURS/output_dir/slurm_logs/%x_%j.err)
PREV=$(sbatch --parsable "${COMMON[@]}" ${AFTER:+--dependency=afterok:${AFTER}} ${LAUNCHER}); IDS=$PREV
for i in $(seq 2 ${N_JOBS}); do PREV=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} ${LAUNCHER}); IDS="$IDS $PREV"; done
echo "submitted ${N_JOBS} job(s): ${IDS} (${RUN_NAME}, REPO=${REPO}, branch=${BRANCH} siblings=${CHAIN_SIBLINGS} ${PROMPTS_PER_GPU}/GPU lr=${LR} critiques=${CRITIQUE_SOURCE})"
