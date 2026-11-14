# Start stage-3 run C on TamIA (for Xiaofeng or his agent)

Written 2026-11-14 by Darshan. Self-contained: follow these steps in order. Ask Darshan before changing anything not listed here.

## What run C is
End-to-end RL of the final model (run 2 = jeet's critique-SFT t4, merged) on drafts, the model's own critiques (including the
"looks good" stop decision) and refinements, with the stop-only critique reward, plus jeet's substage-2 fixes:
168 prompts per step, 2 refinement rounds, LR 1e-4, promptmix v1 + v2 data, 300 steps, evals at steps 0/100/200/300 on
GenEval2 val800, promptmix val_800 and promptmix v2 val_800 (own and oracle critiques, depth 3).
Every setting is fixed by `scripts/rl_ft/submit_e2e_runC.sh`; do not override any of them.

## Steps
1. On a TamIA login node, in your clone of `git@github.com:sarvghotra/tar_reason.git`, branch `darshan-rl-test`
   (e.g. `/scratch/x/xiaofeng/tar_reason_e2e`; clone it with `git clone -b darshan-rl-test ...` if you do not have it):
   ```bash
   cd /scratch/x/xiaofeng/tar_reason_e2e
   git checkout darshan-rl-test && git pull
   git log -1 --oneline        # must include commit 121e02a or later
   ```
2. Submit (your allocation; 15 chained 2 h 59 min jobs, 4 nodes x 4 H100 each, checkpoints every 5 steps, each job resumes
   from the newest checkpoint):
   ```bash
   ACCOUNT=aip-courvill bash scripts/rl_ft/submit_e2e_runC.sh
   squeue -u $USER --start
   ```
   The script checks that all inputs exist and prints the job ids. It writes `output_dir/darshan_e2e_stop_C_fir/{data,val}.yaml`
   in your clone. Logs go to `output_dir/slurm_logs/darshan_e2e_stop_C_fir_<jobid>.{out,err}`.
3. Tell Darshan the job ids. **Only one copy of run C may run**: Darshan also has copies queued on Fir, Rorqual and TamIA
   (same run name). If one of his starts first, cancel yours: `scancel --name=darshan_e2e_stop_C_fir`. If yours starts
   first, he cancels his.

## Inputs (read-only, from Darshan's scratch; nothing to copy)
- Start model / KL reference: `/scratch/d/dars11/models/tar/sft/darshan_E450_orc500_crisft_t4_merged`
- Judge: `/scratch/d/dars11/models/vlm/Qwen3-VL-8B-Instruct`; packed envs: `/scratch/d/dars11/stage/envs_tar_qwenjudge.tar.zst`
- Data: `/scratch/d/dars11/data/promptmix/train_17K.jsonl`, `/scratch/d/dars11/data/promptmix_v2/train_19K_minus_val800.jsonl`
- Val: `/scratch/d/dars11/data/geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl`, `.../promptmix/val_800.jsonl`,
  `.../promptmix_v2/val_800.jsonl`
If the script reports a missing or unreadable file, stop and tell Darshan (read access is by ACL for user `xiaofeng`).

## Checks once it runs
- First lines of the `.out` log: `RUN_NAME=darshan_e2e_stop_C_fir ... DATASET_SEED=57602`, and
  `multi-round: critique_source=policy branch=2,1,1 ... lr=1e-4 eval_branch=1,1,1,1`, `prompts/step 168`.
- The step-0 eval runs first (about 1 h), then training, about 7 min per step.
- Per-step JSON lines: `train/chain_groups_1`, `train/chain_groups_2` should be roughly 130 / 115; `train/kl` should stay
  below ~0.2; `train/crit_stop_acc` around 0.8.
- Report to Darshan: any `Traceback`, `out of memory` or NCCL timeout in the `.err` log, and the eval lines
  (`step=0`, `step=100`, ... containing `val/am_upto_3`).

## Background
Recipe, metrics and the A/B runs: `scripts/rl_ft/README_e2e_stage3.md` (section "Run C").
