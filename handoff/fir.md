# Fir

Owned by the Fir session. Fir: Alliance cluster, user `dars11`, 4× H100 80GB per node, login nodes `login1`/`login2`.

## Now (2026-10-17 21:30 PDT)
- **F stopped** (jeet, 2026-10-17) at step ~855 (job 65300507 cancelled). Best checkpoint **F-800**: val800 depth-4 mdp own 0.858
  (0.866 after 3 rounds), all-correct 44.1% at 3, oracle 0.905. Checkpoints 100–800 kept in `~/scratch/tar_reason_kept/darshan_mr_fir/`.
- **New: jeet's promptmix run** (`rl_ft_promptsmix_oracle_t1` recipe): t8 recipe from SFT t21, one round (BRANCH 2,4), hybrid critiques,
  LR 1e-4, 500 steps, eval every 50 on GenEval2 val800 (`val/`) + promptmix val400 (`val_promptmix/`). Data copied to
  `~/scratch/data/promptmix/` (train_17K 17,259 rows md5 5a7d8259…, val_800 md5 90eb656a…, val_400 = first 400 of val_800;
  0 overlap with each other and with GenEval2 val800). Launcher `bash_fir_promptmix.sh` (`8377344`). **Smoke 65708006 submitted**;
  full run (4 nodes, 2 chained 24 h jobs, RUN_NAME `darshan_pm_fir`) after it passes and the user says go.
  Open with jeet: his notes say train_18K (17.7K) and val 400; the shared folder has train_17K and val_800.
- Stop decision: 3a (stop-SFT, Mila) and 3b (stop-RL, here) both negative (`runs.md`).
- Watcher: `mr/watch2.sh` pattern (self-terminating, 29 min); none running for F now.

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
- 2026-10-16 from Mila: **F-250 official benchmarks done** (`runs.md`). Draft → final (1 round): GenEval2 AM 72.5 → 77.4 (t8 70.3 → 76.8), TIIF short 68.7 → 69.8 / long 65.7 → 66.0 (t8 70.5 / 68.6 final), GenAI-Bench 40.4 → 41.8 (t8 39.3 → 40.7). Official GenEval2 depth-4 MDP: 72.6 / 76.8 / 77.5 / 78.1 / **78.4** (t8 70.8 → 77.9), all-correct 19.0 → 23.8% (t8 17.6 → 24.5%). F beats t8 on GenEval2 and GenAI, mostly via the draft; TIIF long refinement is weaker. Ready for F-300 / G-300 (or the 1000-step checkpoints) once copied to Mila.
- 2026-10-15 23:00 EDT from TamIA: **G → 1000 submitted** (`darshan_mr_sft_tamia`, MAX_STEPS=1000, resumes from checkpoint-300; jobs 621301 → … → 621306, 4 nodes × 12 h). F-full (621218–20) stays queued too (user: both run, least total time).
- 2026-10-15 21:45 EDT from TamIA: **F-full queued**: `darshan_mr_full_tamia` = `bash_tamia_mr.sh` with `CONTEXT_MODE=full` (defaults otherwise: merged t8, LR 5e-5, children_r1, micro-batch 2), jobs 621218 → 621219 → 621220 (4 nodes × 12 h). Will report the step-0/100/200/300 full-history evals and the final mdp eval.
(empty)

## Log (newest first)
- 2026-10-17 21:30: jeet: stop F and G, switch to promptmix. Cancelled F's last job (65300507; F ended at ~855); TamIA Inbox: stop G. Promptmix launcher + smoke 65708006.
- 2026-10-17 17:40: F step-800 eval: best so far (own 0.858 at 4, 0.866 at 3, all-correct 44.1% at 3, draft 0.816); the 500–700 dip was noise. Checkpoint-800 kept.
- 2026-10-17 09:40: F step-700 eval: own 0.854, oracle 0.902, draft 0.792: plateau since 400 (draft slowly down); checkpoint-700 kept. Best checkpoint so far: 400 (0.857, draft 0.812).
- 2026-10-17 00:30: F step-600 eval: own 0.855 at depth 4 (0.862 at 3), oracle 0.905: plateau; job 65300505 TIMEOUT at ~585, 65300506 resumed from 580.
- 2026-10-16 15:20: F step-500 eval: own 0.855 (400: 0.857), oracle 0.901, all-correct 39.3%: plateau. Watcher now on login3 (session moved).
- 2026-10-16 06:30: F step-400 eval: own 0.857 (step 300: 0.848), draft 0.812 (0.787), oracle 0.900, all-correct 39.9%; checkpoint-350 (KL bump to 0.27) and -400 kept.
- 2026-10-15 21:45: 3b step-100 eval flat → cancelled 65224187/88 (user's rule). Stop decision: SFT (3a) and RL (3b) both fail to improve discrimination.
- 2026-10-15 19:00: jeet asked to continue F and G to 1000 steps: F resubmitted here (3 × 24 h chained), G to TamIA's Inbox (priority over F-full).
- 2026-10-15 18:10: F done (step-300 own 0.848, oracle 0.896), wandb synced (2 pieces, second with --append), checkpoint-300 kept. 3b step 50: no change vs step 0. Watchers restarted after the tmux session ended.
- 2026-10-15 06:10: Inbox handled: Mila 3a negative (balanced SFT stops more on everything; natural-ratio = baseline 0.842; discrimination unchanged), so for 3b watch looks_good_fail too; TamIA G done (step 300 own 0.757, oracle 0.803), TamIA idle. F job 2 and 3b est. start 15:00.
- 2026-10-15 01:20: 3b smoke passed; full 3b submitted from F-250 (65120761/62). Mila Inbox: jeet's Mila scratch not readable (user to ask jeet); Mila 3a well underway.
- 2026-10-14 21:05: F job 1 TIMEOUT at step ~265 (as planned); job 2 limit cut to 7 h. F-250 merged (CPU job, def-bengioy).
- 2026-10-14 21:00: user's go on 3a/3b without waiting for jeet; 3b code + launcher; 3a sent to TamIA; plots: https://claude.ai/artifact/KfZqPa7tjqYWFuR9oJHBTZ
- 2026-10-14 19:30 PDT: Inbox handled (TamIA: L backfill, M split, G steps 100/200; Mila: H, I, J); all in `runs.md`. Watcher restarted (tmux session had ended).
- 2026-10-14 16:20: F step-200 eval: final 0.840 (+0.005), depth-3 0.845, draft 0.785, oracle 0.890; checkpoint-200 kept. 13:00: KL 0.37 at step 162, recovered; checkpoint-150 kept as a fallback.
- 2026-10-14 07:40: F step-100 eval: final AM flat (0.834), draft −0.016, round 2 now gains (+0.025); checkpoint-100 kept.
- 2026-10-13 21:05: F (64862883) started; step-0 depth-4 eval matches TamIA's t8 mdp eval (own 0.835 / oracle 0.882 at depth 4).
- 2026-10-13: user approved F (children_r1), G now on TamIA, GenAI-bench on Mila. `d14d6e6` children_r1 + test; N done; F submitted 64862883/84; Inbox to TamIA (G, M) and Mila (H, I, J).
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
