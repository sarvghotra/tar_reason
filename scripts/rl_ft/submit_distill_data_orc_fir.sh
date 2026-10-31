#!/bin/bash
# Run-2 data, oracle-driven (jeet, 2026-10-30: "generate the rollouts with the oracle itself, don't involve policy
# critiques at all"). Eval-only job on run 1's merged step-500 model: depth-4 MDP chains on all 17,259 promptmix train
# prompts where every critique is the oracle's (--eval_oracle --no-eval_own), so a chain refines until its image
# passes every judge question (at most 4 rounds); every image (draft + refinements) is dumped with its codes and
# per-question scores (val_oracle_step0_rank*.jsonl). Then, on the login node:
#   ~/envs/tar/bin/python -m llava.train.rl.stop_sft.build_data --dumps output_dir/pm_run1_distill_data_orc_fir/val_dump \
#       --examples images --ratio 0 --draft_replay 1.0 --out ~/scratch/data/stop_sft/distill_run1_orc_replay.jsonl
# Replaces the policy-driven dump of 672319 (pm_run1_distill_data_fir, own sampled critiques + oracle relabelling).
#   bash scripts/rl_ft/submit_distill_data_orc_fir.sh
set -eu
cd /home/dars11/scratch/git/tar_reason
export PATH=$PATH:/opt/software/slurm/bin
RUN_NAME=pm_run1_distill_data_orc_fir
mkdir -p output_dir/${RUN_NAME}
# train_17K as the single validation set (logname val -> the oracle pass dumps as val_oracle_*)
[ -s output_dir/${RUN_NAME}/val.yaml ] || printf 'datasets:\n    - json_path:\n      - /home/dars11/scratch/data/promptmix/train_17K.jsonl\n      logname: val\n      ratio: 1\n' > output_dir/${RUN_NAME}/val.yaml
# Settings go through the environment (--export=ALL): a comma inside --export's list is a separator even when
# escaped, which turned EVAL_BRANCH=1,1,1,1,1 into "1" (job 1499041, 2026-10-30: drafts only, 27 min).
J=$(RUN_NAME=${RUN_NAME} MAX_STEPS=0 EVAL_BRANCH=1,1,1,1,1 CONTEXT_MODE=mdp \
    PREV_STAGE_CHECKPOINT=/home/dars11/scratch/models/tar/sft/darshan_mr_E450_orc_merged500 \
    EXTRA_ARGS="--no-eval_own --val_dump_codes" \
    sbatch --parsable --export=ALL --exclude=fc10212 --time=06:00:00 -J darshan_run1_distdata_orc scripts/rl_ft/bash_fir_promptmix.sh)
echo "submitted ${J} (${RUN_NAME}: oracle-driven depth-4 chains on train_17K, dump with codes)"
