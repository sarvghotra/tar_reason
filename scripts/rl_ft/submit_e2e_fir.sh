#!/bin/bash
# Stage 3, end-to-end RL (2026-11-10; advisor's suggestion, agreed with jeet and Xiaofeng): after run 2 (jeet's critique SFT
# t4), RL on everything (drafts, own critiques incl. the stop decision, refinements) with only the model's OWN critiques.
#   Start = run 1 step 500 merged (darshan_mr_E450_orc_merged500) + jeet's t4 critique-SFT adapter, merged
#           -> models/tar/sft/darshan_E450_orc500_crisft_t4_merged (base AND KL reference; fresh LoRA r64).
#   Recipe = run 1's (bash_fir_mr.sh: chain mode to depth 4, 4 siblings at a random round, MDP context, children_r1 draft
#            reward, break weight 2, KL 0.05, LR 5e-5, promptmix train_17K) with
#              CRITIQUE_SOURCE=policy   own critiques, sampled per child and trained (no oracle critiques in rollouts)
#              STOP_PENALTY=0           the stop signal comes from the critique reward only
#              --critique_reward stop | full (llava/train/rl/critique_reward.py), lambda 0.25 (full: beta 0.5)
#   Run A (Fir, Darshan): CRITIQUE_REWARD=stop  -> +1 / -1 for the right / wrong stop decision.
#   Run B (Xiaofeng):     CRITIQUE_REWARD=full  -> as A, but editing a failing image earns recall - 0.5 * false alarms
#                                                  (checks named by the critique, judged text-only by Qwen3-VL-8B).
#   Evals: depth-4 MDP chains (own + oracle) on GenEval2 val800 + promptmix val_800 at step 0 and every 100 steps.
#   Fir or Rorqual (same paths and account; the launcher is picked by cluster).
#   Step 0 doubles as the check that our code reproduces jeet's t4 numbers (his d3: val800 own am_final 0.826,
#   promptmix 0.792 = our val/am_upto_3).
#
#   smoke: SMOKE=1 CRITIQUE_REWARD=full bash scripts/rl_ft/submit_e2e_fir.sh
#   run A: CRITIQUE_REWARD=stop bash scripts/rl_ft/submit_e2e_fir.sh
set -eu
: "${CRITIQUE_REWARD:?stop | full}"
export PATH=$PATH:/opt/software/slurm/bin
# Runs from the checkout this script is in (Darshan: /home/dars11/scratch/git/tar_reason; Xiaofeng: his own clone of
# branch darshan-rl-test). Outputs, logs and checkpoints go to that checkout's output_dir/. Inputs (models, data, the
# packed Python envs) are read from Darshan's scratch (S), shared read-only with xiaofeng via ACLs.
export REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd $REPO
S=/home/dars11/scratch
BASE=${BASE:-$S/models/tar/sft/darshan_E450_orc500_crisft_t4_merged}
for f in $BASE/merged_from.json $S/stage/envs_tar_qwenjudge.tar.zst $S/models/vlm/Qwen3-VL-8B-Instruct/config.json \
         $S/data/promptmix/train_17K.jsonl $S/data/promptmix/val_800.jsonl $S/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl; do
  [ -s "$f" ] || { echo "missing $f" >&2; exit 1; }
done
grep -q -- '--critique_reward' llava/train/rl/train_grpo.py || { echo "this checkout has no --critique_reward" >&2; exit 1; }
if [ "${SMOKE:-0}" = 1 ]; then
  RUN_NAME=${RUN_NAME:-e2e_${CRITIQUE_REWARD}_smoke_fir}
  export MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24
  SB=(--nodes=1 --time=01:00:00)
else
  RUN_NAME=${RUN_NAME:-$( [ "$(id -un)" = dars11 ] && echo darshan || id -un )_e2e_${CRITIQUE_REWARD}_fir}
  export MAX_STEPS=${MAX_STEPS:-500} EVAL_STEPS=${EVAL_STEPS:-100} SAVE_STEPS=${SAVE_STEPS:-10}   # evals every 100 (user, 2026-11-10)
  SB=(--nodes=${NODES:-4} --time=${TIME:-23:59:00})
fi
mkdir -p output_dir/${RUN_NAME}
[ -f output_dir/${RUN_NAME}/val.yaml ] || printf 'datasets:\n    - json_path:\n      - %s\n      logname: val\n      ratio: 1\n    - json_path:\n      - %s\n      logname: promptmix\n      ratio: 1\n' \
  $S/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl $S/data/promptmix/val_800.jsonl > output_dir/${RUN_NAME}/val.yaml
export RUN_NAME PREV_STAGE_CHECKPOINT=$BASE TRAIN_JSONL=$S/data/promptmix/train_17K.jsonl PIXEL_JUDGE=qwen
export CRITIQUE_SOURCE=policy STOP_PENALTY=0
export EXTRA_ARGS="--critique_reward ${CRITIQUE_REWARD} --critique_reward_coef ${LAMBDA:-0.25} --critique_fa_beta ${BETA:-0.5} --save_total_limit 100"
N_JOBS=${N_JOBS:-$([ "${SMOKE:-0}" = 1 ] && echo 1 || echo 3)}
mkdir -p output_dir/slurm_logs
if [[ "${CC_CLUSTER:-}" == rorqual* || "$(hostname)" == rorqual* ]]; then LAUNCHER=scripts/rl_ft/bash_rorqual_mr.sh; EXCL=()
else LAUNCHER=scripts/rl_ft/bash_fir_mr.sh; EXCL=(--exclude=fc10212); fi
COMMON=(--export=ALL "${EXCL[@]}" --account=${ACCOUNT:-rrg-bengioy-ad_gpu} "${SB[@]}" -J ${JOB_NAME:-${RUN_NAME}}
        -o $REPO/output_dir/slurm_logs/%x_%j.out -e $REPO/output_dir/slurm_logs/%x_%j.err)
PREV=$(sbatch --parsable "${COMMON[@]}" ${AFTER:+--dependency=afterany:${AFTER}} ${LAUNCHER}); IDS=$PREV
for i in $(seq 2 ${N_JOBS}); do PREV=$(sbatch --parsable "${COMMON[@]}" --dependency=afterany:${PREV} ${LAUNCHER}); IDS="$IDS $PREV"; done
echo "submitted ${N_JOBS} job(s): ${IDS} (${RUN_NAME}, critique_reward=${CRITIQUE_REWARD}, base=${BASE})"
