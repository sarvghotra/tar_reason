# Fir

Owned by the Fir session. Fir: Alliance cluster, user `dars11`, 4× H100 80GB per node, login nodes `login1`/`login2`.

## Now (2026-10-12 19:00 PDT)
- **Task E code done:** `--chain_siblings` (commit `a011791`) + launcher `scripts/rl_ft/bash_fir_mr.sh` (t8 recipe, chain mode,
  merged-t8 base, LR 5e-5, depth-4 evals every 100). CPU tests pass (`util_scripts/debug/test_chain_mode.py`). GPU smoke
  64622474 **passed** (micro-batch 2 fits, ~4.5 min/step on 1 node). 24-prompt hint: t8 depth-4 am_final 0.764 vs 0.824 at depth 1. Also `--context_mode mdp`
  (task C, `b0e0e34`) is eval-only.
- **Plan (user + jeet, 2026-10-12; `runs.md` "Plan"):** Fir = A (t11 to step 300), writes C (MDP eval mode) and E (N=4
  multi-round training rollout), later runs F (N=4 from t8). TamIA = B (t8/SFT depth-4 evals), then C's eval, later G.
  Mila = D (official benchmarks of t8, then t11 checkpoints). TamIA's duplicate t11 is cancelled (user).
- **Fir GPU outage from 2026-10-13 10:00 PDT** (reservation `GPUoutage`, 160 GPU nodes fc101xx–fc110xx, end not given).
  No Fir GPU work after that until it lifts; plan tasks F (and anything else 4-node) move to TamIA unless it ends soon.
- **t11 resubmitted with the hardened launcher** (commit `b00c50e`; smoke 64608933: staging 13 s, judge ready 10 s,
  step-0 val identical): 64620282 + spare 64620283, `TimeLimit=16:00:00 TimeMin=03:00:00` so they can start with a
  shortened limit that ends before the outage. Resumes from checkpoint-120; ~13.5 h needed to reach 300. Old spare
  64518140 cancelled. Watcher restarted (scratchpad `t11/watch.sh`).
- **Running: `darshan_t11_fir`** (collaborator's t11: critic-only, image tokens KL-anchored to merged t8; see `runs.md`),
  4 nodes = 12 train ranks x 3 prompts = 36 prompts/step, 300 steps, val800 every 50. Chain: 64511966 (started 02:06,
  **failed at step 13**, 03:39: the node-1 judge died of a **Bus error / SIGBUS**, the trainer then hit "reward server
  unreachable") → **64511967 running** (resumes from checkpoint-10) → spare **64518140** (`afterany`). Second SIGBUS
  on Fir in 3 days (t7 died the same way at step 41): likely Fir scratch/home (Lustre) dropping a memory-mapped page,
  not our code. Possible hardening, not done yet (needs the user): stage the judge and policy weights to node-local
  `$SLURM_TMPDIR` at job start; restart a dead judge inside the launcher (the trainer's retry window must cover a
  ~2–4 min reload).
- **64511967 failed at step 123 (11:45) with SIGBUS (exit −7) in a training rank on fc10114** — third SIGBUS on Fir
  (t7 @41, t11 judge @13, now a trainer rank). Spare **64518140** pending on Priority since 11:46 (no start estimate);
  it will resume from checkpoint-120. Kept: checkpoint-50/100. TamIA's `darshan_t11_tamia` is still queued there
  (`squeue --start` 20:30 EDT).
- Before the crash (step ~100 at 10:20; ~3.8 min/step). Evals 0/50/100 in `runs.md`. **Step 100: the critic
  stops on failing drafts twice as often (looks_good_fail 0.09 → 0.11 → 0.20)** — jeet's pre-set trigger to raise
  STOP_PENALTY; final AM 0.828 → 0.820, own fix flat 0.447, oracle fix holds 0.489, kl_img ~0.004. Decision with the user.
- Watcher: the tmux session ended ~10:15 and its scratchpad (watch scripts) is gone; no watcher is running.
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
- 2026-10-13 00:40 EDT from Mila: t8 ck-500 depth on the **official** GenEval2 800 prompts (full-history chain to depth 4, greedy own critic, Qwen3-VL-8B in GenEval2 mode): final AM after 0/1/2/3/4 rounds = 70.9 / 75.7 / **76.4** / 75.5 / 74.3; GM 26.7 / 32.1 / 32.7 / 30.5 / 28.3; 626 of 800 prompts still refining at round 4. Best at 2 rounds, deeper rounds hurt. Row `depth4_t8_geneval2_*` in `runs.md`; compare with task B on val800.
- 2026-10-12 22:10 EDT from Mila: task D part 1 done: t8 ck-500 (merged) on the official benchmarks, same pipeline/seed as t7 Mila: GenEval2 AM draft 70.3 → final **76.8** (t7 ck-200: 68.9 → 71.7), GM final 33.5, all-correct 22.6%; TIIF short 68.3 → 70.5, long 65.4 → **68.6** (refining now helps on long). Row `bench_mila` in `runs.md`. Depth (draft + 4 rounds on official GenEval2) is running next.
(empty)

## Log (newest first)
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
