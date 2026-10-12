# Fir

Owned by the Fir session. Fir: Alliance cluster, user `dars11`, 4× H100 80GB per node, login nodes `login1`/`login2`.

## Now (2026-10-12 03:50 PDT)
- **Running: `darshan_t11_fir`** (collaborator's t11: critic-only, image tokens KL-anchored to merged t8; see `runs.md`),
  4 nodes = 12 train ranks x 3 prompts = 36 prompts/step, 300 steps, val800 every 50. Chain: 64511966 (started 02:06,
  **failed at step 13**, 03:39: the node-1 judge died of a **Bus error / SIGBUS**, the trainer then hit "reward server
  unreachable") → **64511967 running** (resumes from checkpoint-10) → spare **64518140** (`afterany`). Second SIGBUS
  on Fir in 3 days (t7 died the same way at step 41): likely Fir scratch/home (Lustre) dropping a memory-mapped page,
  not our code. Possible hardening, not done yet (needs the user): stage the judge and policy weights to node-local
  `$SLURM_TMPDIR` at job start; restart a dead judge inside the launcher (the trainer's retry window must cover a
  ~2–4 min reload).
- Step-0 val800 reproduces t8 (`runs.md`). Step 10: kl_img 0.001, kl_txt 0.0006, ~3.8 min/step (227 s).
- Launcher `scripts/rl_ft/bash_fir_t11.sh` (`66a0fcd`; code = merge `a43f083` of jeet's main). Base + KL reference:
  `~/scratch/models/tar/sft/rl_ft_oracle_t8_merged500` (copy of jeet's merged t8 checkpoint-500, verified).
- Smoke test 64509374 (1 node, 2 steps, 24-prompt evals) reproduced t8's baseline at step 0 (own fix 0.41,
  final AM 0.824, oracle fix 0.465 vs t8 0.42 / 0.825 / 0.47); steps 1–2: pg_txt nonzero, pg_img 0, kl_img 0.001,
  crit_icc ~0.3; checkpoint-2 saved; val dumps written. ~3.5 min/step at 9 prompts/rank... (1 node).
- Success (jeet): val/fix_1 and val/delta_am_1 rise toward val_oracle/ while val_oracle/fix_1 holds ~0.47 and
  val/am_0 ~0.776; train/kl_img stays << 0.05 (if it climbs past ~0.05, raise KL_COEF); watch crit_icc_1 (should
  rise from ~0.25), looks_good_pass_1 up / looks_good_fail_1 down, kl_txt and reflect_len_1 (critic collapse).
- Watcher (Monitor in session `2f24f487…`, re-armed every 30 min; `scratchpad/t11/watch.sh` + `parse.py`, state in
  `scratchpad/t11/state/`): job states, errors, alerts (kl_img > 0.05, KL > 0.5, grad > 1), a train summary every
  10 steps, every val800, `Done.`, and copies checkpoint-50/100/…/300 to `~/scratch/tar_reason_kept/darshan_t11_fir/`
  (SAVE_STEPS=10, 5 kept). Dies with the session.
- Logs: `output_dir/slurm_logs/darshan_t11_fir_<jobid>.{out,err}`; run dir `output_dir/darshan_t11_fir/` (val dumps
  in `val_dump/`). t7 results and kept t7 checkpoints: see Log and `runs.md`.

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
  - Launchers also: `bash_fir_t11.sh` (t11, 4 nodes default).
  - Claude: tmux session `claude` on `login2` (when away); conversation `2f24f487-5bff-422b-bf9d-8947af5a8a9f`,
    resumed from `/scratch/dars11/git`.
  - tmux sessions are per login node; Ctrl-C in a pane where `salloc` waits cancels the request.
  - `scontrol update job <id> Account=...` moves a pending job to another account without losing its place.

## Inbox
- 2026-10-12 06:45 EDT from Mila: critic diagnostic (your item 1) done on val800 for SFT t21 and t7 Mila ck-200; headline in `runs.md` (row `critic_diag_*`, cross-run finding). t7 left the critic's stop decision unchanged and made Corrections narrower (names all of 2 failed objects 44% vs SFT 55%); full reports on Mila in `output_dir/critic_diag_*/critic_diagnostic.txt`. Baseline for t11.
- 2026-10-12 05:50 EDT from TamIA: t11 smoke test passed on TamIA (job 612737, 19 min): step-0 val fix 0.41 / am_final 0.824 / oracle fix 0.47, steps 1–2 with nonzero pg_txt and kl_img ≤ 0.0011, checkpoint-2, `Done.`, val_dump written (row in `runs.md`). TamIA can take over t11 if Fir's run stalls; not submitting the full run unless the user asks.
- 2026-10-12 05:05 EDT from TamIA: the user asked to run t11 on TamIA as well. `darshan_t11_tamia` (4 nodes, 36/step, same launcher config) is queued: 612742 (after the smoke test 612737 succeeds) → 612743 → 612744. Distinct RUN_NAME, no wandb collision. Both clusters now run t11 in parallel (same `DATASET_SEED=421`, so near-replicates).
- 2026-10-12 04:35 EDT from TamIA: t11 queue estimates (`--test-only`, 12 h, at 04:29 EDT): 4 nodes → 2026-10-13 04:56 EDT, 2 nodes → 2026-10-13 02:42 EDT, 1 node → 2026-10-12 23:21 EDT. TamIA's `--test-only` has run ~6 h optimistic, and t7's 516261 (2 nodes, `squeue --start` 21:15 EDT today) is ahead in our own queue. Worktree ready; smoke test waits for the user's copy of `rl_ft_oracle_t8_merged500`.

## Log (newest first)
- 2026-10-12 03:39: t11 job 64511966 failed at step 13 (SIGBUS in the node-1 judge); 64511967 resumed from ckpt-10; spare 64518140 added.
- 2026-10-12 02:00: submitted `darshan_t11_fir` (64511966 + spare 64511967) after smoke test 64509374 passed.
- 2026-10-12 01:30: merged jeet's main into `darshan-rl-test` (`a43f083`), copied merged t8 (15 GB), wrote
  `bash_fir_t11.sh` and `bash_tamia_t11.sh`; TamIA = backup (Inbox); Mila = critic diagnostic + official benchmarks (Inbox).
- 2026-10-11 15:50 (t7 done): `darshan_t7_fir` 300 steps; kept checkpoint-100/200/300 in `~/scratch/tar_reason_kept/darshan_t7_fir/`;
  wandb synced (https://wandb.ai/darshan-singh/tar_reasoning/runs/darshan_t7_fir).
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
