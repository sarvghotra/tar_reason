# Fir

Owned by the Fir session. Fir: Alliance cluster, user `dars11`, 4× H100 80GB per node, login nodes `login1`/`login2`.

## Now (2026-10-13, evening PDT)
- **Task F submitted: `darshan_mr_fir`** (64862883 → 64862884 `afterany`, 4 nodes × 24 h, 300 steps): N=4 chain-mode MDP training
  from merged t8, `--draft_reward children_r1` (`d14d6e6`, user-approved; each draft judged after one refinement), LR 5e-5,
  depth-4 mdp val800 evals every 100 steps. **Running since 21:05 PDT** (4 nodes staged + judges ready in seconds). Step-0 val800 depth-4 mdp: own am_upto
  0.779/0.825/0.829/0.830/0.835, oracle 0.774/0.844/0.867/0.874/0.882 (= TamIA's t8 mdp eval within noise).
  **Step-100** val800: own am_upto 0.763/0.813/0.833/0.830/0.834 (final flat vs 0.835; draft −0.016; round-2 gain up: delta_am_2
  +0.025 vs +0.005, fix_2 0.35 vs 0.27; all-correct after 2 rounds 35.0% vs 32.5%); oracle 0.756/…/0.881 (flat). KL 0.10–0.18. checkpoint-100 kept.
  **Step-200** val800: own am_upto 0.785/0.840/0.844/**0.845**/0.840 (final +0.005 vs step 0, best depth 3 +0.015; draft recovered to
  0.785; all-correct 37.2% at depth 3 vs 33.0%); oracle 0.786/0.856/0.873/0.884/**0.890**. Gain is in the draft and round 1 (delta_am_1
  +0.055); rounds 2–4 ≈ 0 again. KL spike to 0.37 at step 162 recovered in 1 step; checkpoint-150 and -200 kept. Watch for: step time (~5 min/step expected), KL, `train/chain_*`.
- Plan F–N in `runs.md`: G (from SFT) on TamIA in parallel; H (official GenEval2 depth-4 MDP), I (base Tar), J (GenAI-bench) on
  Mila; L (am_upto backfill) TamIA + Mila; M (stop vs content split) TamIA; N done here (no unrecognised stop phrases).
- Fir's GPU outage is over. t11 done (results in `runs.md`); kept checkpoints in `~/scratch/tar_reason_kept/`.
- Hardened launcher (`STAGE_LOCAL=1`) in `bash_fir_t11.sh` and `bash_fir_mr.sh`; rebuild the env archive after any env change.
- Watcher: Monitor on `darshan_mr_fir` (scratchpad `mr/watch.sh`, state in `mr/state`), started 2026-10-13 19:40 in the tmux session on login2; re-armed every 30 min; keeps checkpoint-100/200/300.

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
- 2026-10-14 21:40 EDT from TamIA: **G step 200** (val800 depth-4 mdp, AM after 0..4 rounds): own 0.723/0.743/0.749/0.742/**0.744** (step 0: 0.684/…/0.675; step 100: …/0.697), oracle 0.728/0.759/0.775/0.784/**0.789** (step 0: 0.697). Draft +0.039, final +0.069 since step 0; fix_1 13.1 → 24.7%; oracle gains in every round. KL bump 161–171 (≤ 0.37) recovered. Checkpoints 50–200 kept.
- 2026-10-14 11:50 EDT from TamIA: **G step 100** (val800 depth-4 mdp, AM after 0..4 rounds): own 0.686/0.698/0.705/0.702/0.697 (step 0: …/0.675), oracle 0.701/…/0.728 (step 0: 0.697). Small gains from refinement (fix_1 13.1 → 17.7%), draft flat; far below F/t8. KL 0.05, no spikes. Checkpoints 50/100 kept.
- 2026-10-14 from Mila: tasks H, I, J done (rows in `runs.md`). **H**: t8 depth-4 with `--context_mode mdp` on the official GenEval2 800: AM after 0–4 rounds 70.8 / 76.1 / 77.5 / 77.3 / **77.9** (full history 70.9 / 75.7 / 76.4 / 75.5 / 74.3), all-correct 17.6 → 24.5% (full 18.1 → 18.5%): your val800 MDP result holds on the official prompts. **I**: base Tar-7B single pass: GenEval2 AM 64.7, TIIF 69.2 / 65.2. **J**: GenAI-Bench 1,600 prompts, Qwen3.5-27B VQAScore: base 37.7, t8 draft → final 39.3 → 40.7 (advanced 37.4 → 40.8). Copy F/G checkpoints to Mila when ready; I'll run GenEval2 + TIIF + GenAI (+ MDP depth).
- 2026-10-13 20:30 EDT from TamIA: **task M done** (`runs.md` cross-run findings). Depth 4, MDP, final AM / all-correct: own 0.833 / 0.329 → own with perfect stopping 0.860 / 0.494 (lower bound) → oracle 0.885 / 0.595: stopping ≈ 53% of the AM gap and 62% of the all-correct gap, rest ≈ critique content. Critic recall (looks good | image passes) 0.50 / 0.16 / 0.08 / 0.07 by round; false stops 0.08 / 0.04 / 0.04 / 0.02; 132 chains reach an all-correct image and then lose it. Best image per own chain 0.891 (> oracle). **Task G: ready but not submitted yet** (Claude Code auto mode blocked the 4-node sbatch; waiting for the user).
- 2026-10-13 19:45 EDT from TamIA: depth-curve backfill done for `t8_depth4_full_tamia` and `t8_depth4_mdp_tamia` (`val/` and `val_oracle/` `am_upto_0..4`, `allpass_upto_0..4` in the wandb summaries; checks am_upto_0 = am_0, am_upto_4 = am_final exact). allpass_upto own full 0.236/0.316/0.311/0.295/0.264, own mdp 0.249/0.306/0.347/0.325/0.329; oracle full 0.274→0.552, oracle mdp 0.245→0.595.
(empty)

## Log (newest first)
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
