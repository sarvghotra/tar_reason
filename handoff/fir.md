# Fir

Owned by the Fir session. Fir: Alliance cluster, user `dars11`, 4× H100 80GB per node, login nodes `login1`/`login2`.

## Now (2026-10-13 08:00 PDT)
- **Nothing running on Fir. Fir GPU outage from today 10:00 PDT** (reservation `GPUoutage`, end not given).
- **t11 (`darshan_t11_fir`) done**, 300 steps (`runs.md`): final AM 0.828 at step 0 and at 300, own-critique fix 42.6% → 46.4%,
  stop decision oscillates between evals (looks_good on failing drafts 0.09 … 0.22 … 0.09). Kept: checkpoint-50…300 in
  `~/scratch/tar_reason_kept/darshan_t11_fir/`. wandb complete: https://wandb.ai/darshan-singh/tar_reasoning/runs/darshan_t11_fir
- **Tasks C and E (code) done**: `--context_mode mdp` (eval-only, `b0e0e34`), `--chain_siblings` chain-mode multi-round
  training (`a011791`), launchers `bash_fir_mr.sh` / `bash_tamia_mr.sh`; Fir multi-round smoke 64622474 passed.
  Task F (N=4 from t8) waits for the user + jeet; with Fir down it would run on TamIA (`bash_tamia_mr.sh`).
- **MDP training now supported** (`7ff8bee`: one row per node, own segment only; CPU tests pass); launchers default
  `CONTEXT_MODE=mdp` because TamIA's B/C showed full history degrades t8 after round 1 and mdp does not.
  GPU smoke of MDP training (64740914, 1 node) **passed** (30 min, no errors, checkpoint-2, `Done.`).
- Hardened launcher (node-local staging of envs + weights, `STAGE_LOCAL=1`) in `bash_fir_t11.sh` and `bash_fir_mr.sh`;
  the env archive `~/scratch/stage/envs_tar_qwenjudge.tar.zst` must be rebuilt after any env change (command in the launcher).
- Watcher: none needed (nothing running).

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
(empty)

## Log (newest first)
- 2026-10-13: asked TamIA (`t8_depth4_full/mdp_tamia`) and Mila (`depth4_t8_geneval2_part1/2`) to backfill `am_upto_d` into their wandb summaries.
- 2026-10-13: Inbox from TamIA handled: tasks B and C done (full history degrades t8 after round 1, mdp does not) -> F defaults to mdp; TamIA multi-round smoke passed.
- 2026-10-13 08:22: MDP-training smoke 64740914 passed.
- 2026-10-13 07:50: MDP training support (`7ff8bee`), CONTEXT_MODE in the mr launchers; smoke 64740914 submitted.
- 2026-10-13 07:55: t11 done (step-300 val800 final 0.828, own fix 0.464); checkpoint-300 kept; spare 64620283 cancelled; wandb synced.
- 2026-10-13 04:25: t11 step-250 val800: own fix 0.477, final 0.824, looks_good pass/fail 0.71/0.22; checkpoint-250 kept.
- 2026-10-13 00:50: t11 step-200 val800: final 0.831, own fix 0.441, looks_good pass/fail 0.58/0.09; checkpoint-200 kept.
- 2026-10-12 23:05: Inbox from Mila handled: t8 official GenEval2 76.8 / TIIF long 68.6; depth on official GenEval2 best at 2 rounds
  (76.4) then down to 74.3 at 4, critic almost never stops (626/800 still refining at round 4). Supports task F.
- 2026-10-12 21:35: t11 step-150 val800 ≈ step 0 (own fix 0.422, final 0.825, looks_good_fail back to 0.09); checkpoint-150 kept.
- 2026-10-12 18:56: t11 job 64620282 started (15 h 03 min limit, ends 09:59), resumed from checkpoint-120.
- 2026-10-12 18:53: multi-round smoke 64622474 passed.
- 2026-10-12 19:00: chain-mode rollouts (`a011791`) + `bash_fir_mr.sh`; smoke 64622474 queued.
- 2026-10-12 18:05: Inbox cleared: TamIA cancelled its t11 (612742–44) and queued task B (t8 depth-4 full, job 614114, est. 03:15 EDT;
  SFT eval cancelled per the trimmed plan); Mila verified the t8 copy and queued task D (11222563–65); Mila critic diagnostic noted earlier.
- 2026-10-12 17:50: hardened launcher committed (`b00c50e`); t11 resubmitted 64620282/83 (TimeMin 3 h, before the 10-13 outage).
- 2026-10-12 11:45: t11 job 64511967 failed at step 123: SIGBUS in a training rank (local_rank 0, fc10114); spare 64518140 queued (Priority).
- 2026-10-12 10:20: step-100 val800: looks_good on failing drafts 0.20 (from 0.09); checkpoint-50/100 kept.
- 2026-10-12 06:00: step-50 val800: anchor holds (draft 0.770, oracle fix 0.486), own fix 0.448.
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
