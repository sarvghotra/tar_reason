# Fir

Owned by the Fir session. Fir: Alliance cluster, user `dars11`, 4× H100 80GB per node, login nodes `login1`/`login2`.

## Now (2026-10-11 15:50 PDT)
- **Nothing running on Fir.** `darshan_t7_fir` finished 300 steps (`Done.` at 00:22 PDT 10-11; job 63858144 hit its
  24 h limit at ~step 250 as planned, 63858145 resumed from checkpoint-250 and finished). Results in `runs.md`.
- Kept (verified identical): `~/scratch/tar_reason_kept/darshan_t7_fir/checkpoint-{100,200,300}` (LoRA adapter +
  optimizer, on top of `sft/slf_ref_edit_t21_ckpt17000`). The run folder also has checkpoint-280…300.
  checkpoint-150 (highest draft AM, 0.742) was pruned.
- Headline: at step 300 own-critique final − draft +0.064, fix 38%, final AM 0.762, oracle fix 51%. **But** the
  draft got worse after step 150 (0.742 → 0.698), so final AM barely beat step 150 (0.756). Possible reward
  hacking (weaker drafts leave more to fix) as KL rose to 0.4–0.7; not verified.
- wandb synced: https://wandb.ai/darshan-singh/tar_reasoning/runs/darshan_t7_fir (steps 1–300 except 250, lost at
  the time-limit kill; all 7 val800 evals, step-250's at 251). Smoke test `t7_smoke_fir` not synced.
- **Next (user to decide):** evaluate
  checkpoint-100/200/300 on official GenEval2/TIIF by *final* AM; if continuing RL, consider lower LR / higher KL
  or a draft-quality term so the draft can't be sacrificed.
- Logs: `~/scratch/git/tar_reason/output_dir/slurm_logs/darshan_t7_fir_6385814{3,4,5}.{out,err}`.

## Setup
- Accounts: `rrg-bengioy-ad_gpu` (member since 2026-10-09; fair-share ~0.27, the allocation is over its share),
  also `def-agrawal_gpu`, `def-bengioy_gpu`. Not in `aip-bengioy`.
- Paths: `~/scratch` → `/scratch/dars11` (symlink). Notes dir `~/scratch/git/` (`CLAUDE.md` → `tar_reason/handoff/CLAUDE.md`,
  `papers/`). Repo `~/scratch/git/tar_reason` on `darshan-rl-test`; `output_dir/` is a real directory (gitignored).
- Envs: `~/envs/tar`, `~/envs/qwen_judge` (versions in CLAUDE.md). Exact package lists:
  `~/scratch/tamia_transfer/{tar_env_freeze,qwen_judge_env_freeze}.txt`. Install fixes: flash-attn from the
  prebuilt GitHub wheel (source build fails), `setuptools<70` (torch 2.1 imports `pkg_resources`).
- Weights `~/scratch/models/`: `tar/{Tar-7B (HF csuhan/Tar-7B-v0.1), ta_tok.pth, ar_dtok_lp_512px.pth,
  vq_ds16_t2i.pt, siglip2-so400m-patch14-384/config.json}`, `tar/sft/{slf_ref_edit_t21_ckpt17000,
  slf_ref_edit_t20_ckpt16000, slf_ref_edit_t6_repro_w_corr_t2_ckpt20000}`, `vlm/Qwen3-VL-8B-Instruct`.
- Data `~/scratch/data/geneval2_50K_v2/` (verified md5 + no overlap); old `geneval2_50K/` also present (don't use with val800).
- wandb logged in; compute nodes have no internet, so runs are offline.
- Launchers: `scripts/rl_ft/bash_fir_t7.sh` (t7; overridable `RUN_NAME`, `MAX_STEPS`, `EVAL_STEPS`, `SAVE_STEPS`,
  `EVAL_MAX_PROMPTS`, `PROMPTS_PER_GPU`, `BRANCH`; `--nodes` on the sbatch line), `bash_darshan.sh` (latent reward).
- Quirks:
  - `output_dir/slurm_logs/` must exist before `sbatch` (SLURM opens `-o/-e` before the script runs).
  - Scratch reads are slow: policy shards load in ~5.5 min, judge in ~2.5–4 min.
  - Queue: whole 4×H100 nodes waited ~75 min (1 h limit) on 2026-10-09.
  - Claude: tmux session `claude` on `login2` (when away); conversation `2f24f487-5bff-422b-bf9d-8947af5a8a9f`,
    resumed from `/scratch/dars11/git`.
  - tmux sessions are per login node; Ctrl-C in a pane where `salloc` waits cancels the request.
  - `scontrol update job <id> Account=...` moves a pending job to another account without losing its place.

## Inbox
- 2026-10-12 04:35 EDT from TamIA: t11 queue estimates (`--test-only`, 12 h, at 04:29 EDT): 4 nodes → 2026-10-13 04:56 EDT, 2 nodes → 2026-10-13 02:42 EDT, 1 node → 2026-10-12 23:21 EDT. TamIA's `--test-only` has run ~6 h optimistic, and t7's 516261 (2 nodes, `squeue --start` 21:15 EDT today) is ahead in our own queue. Worktree ready; smoke test waits for the user's copy of `rl_ft_oracle_t8_merged500`.

## Log (newest first)
- 2026-10-11 17:10: smoke test 64442582 (1 node, 32 min) passed for `--init_adapter` (from kept ckpt-100: val final AM 0.755 on
  24 prompts, i.e. not SFT), `--val_dump_dir` (val/ + val_oracle/ jsonl per rank) and `--image_token_weight 0` (with hybrid
  critiques: loss/grad exactly 0, weights unchanged). `critic_diagnostic.py` runs on the dump.
- 2026-10-11 15:55: synced the 3 offline pieces of `darshan_t7_fir` to wandb; the 3rd needed `--append` (history was dropped otherwise).
- 2026-10-11 15:50: run done (300 steps); copied checkpoint-300 to kept (watcher had died with the session before step 300).
  Step 200/250/300 val800 recorded; draft AM dropped after step 150 (see Now).
- 2026-10-10 17:10: job 63858144 hit the time limit (~step 250); 63858145 resumed from checkpoint-250.
- 2026-10-10 10:00: KL trending up (median 0.53 over steps 181–188); decision deferred to the step-200 eval.
- 2026-10-10 08:28: one-step KL spike at step 168 (kl 1.37, grad 1.21, clip_frac 0.17; image tokens only); recovered at 169 (kl 0.15) and 170 (0.19).
- 2026-10-10 07:10: step-150 val800: own fix flat 24.4%, oracle fix 38.2%, final AM 0.756; KL 0.2–0.4 late.
- 2026-10-10 01:30: step-100 val800 positive (+0.021, fix 24.5%, brk 7.2%). Watcher v1/v2 had a Python f-string syntax
  error, so val/spike/error events never fired (job states and checkpoint copies worked); fixed in v3 and tested.
- 2026-10-09 23:30: step-50 val800 positive; job 63858143 SIGBUS at step 41 (17:10), resumed by 63858144 from ckpt-40; watcher armed.
- 2026-10-09 13:40: step-0 val800 matched Mila's within noise; steps 1–2 at ~4.7–5.0 min/step.
- 2026-10-09: Inbox from TamIA handled: `darshan_t7_tamia` (2 nodes, 36/step) queued at the user's request alongside `darshan_t7_fir`.
- 2026-10-09: moved notes into `tar_reason/handoff/`; old `FIR_HANDOFF.md` frozen in `archive/`.
- 2026-10-09: first `darshan_t7_fir` chain (63847016–18) failed in minutes: `RUN_NAME` not exported into the
  `srun bash -c` block → empty `--run_name`. Fixed `6c8f5e8` (and `bash_tamia_t7.sh`, `b0c8c31`); resubmitted as 63858143–45.
- 2026-10-09: smoke test `t7_smoke_fir` passed (job 63829085), ~4.8 min/step.
- 2026-10-09: judge env, Qwen3-VL-8B, SigLIP2 config, leak-free data set up; added to `rrg-bengioy-ad`.
- 2026-10-06: setup paused, user moved to Mila (no GPU job scheduled on Fir under `def-agrawal_gpu`).
- 2026-10-05/06: training env, Tar weights, SFT checkpoints + data copied from `/scratch/jeet/tmp`; branch `darshan-rl-test` created.
