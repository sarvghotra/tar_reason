# TamIA

Owned by the TamIA session.

## Now (2026-10-23 13:50 EDT)
- **F-full done** (`darshan_mr_full_tamia`, 300 steps; 621220 COMPLETED 2026-10-20 23:59). Step-300 full-history eval: own
  am_upto 0.789/0.840/0.839/0.840/**0.833**, oracle **0.897** (step 0: 0.796 / 0.863; F-mdp step 300: 0.848 / 0.896): training
  mostly removed the full-history handicap. Checkpoints 50/100/200/250/300 kept in `$SCRATCH/tar_reason_kept/darshan_mr_full_tamia/`;
  wandb: 3 pieces synced with `--append`.
- **F-full mdp eval of checkpoint-300 done** (`fullF300_mdp_eval_tamia`, 636248, 1 h 28 min): own am_upto
  0.783/0.836/0.839/0.846/**0.847**, oracle **0.898** = F-mdp-300 (0.848 / 0.896). The full-history-trained model is as good under
  mdp as the mdp-trained one, and nearly as good under full history (0.833). `tar_reason_t11` is free to rebase now.
- **Task C of the single-round promptmix screen (Fir Inbox 2026-10-21): `darshan_pm1_C_kl01_tamia` = jeet's t1 recipe from SFT
  t21 with `KL_COEF=0.01`**, 200 steps, evals every 50, 4 nodes. New worktree **`/scratch/d/dars11/git/tar_reason_pm`** (latest
  code, `b5ea15e`+) with the new launcher `scripts/rl_ft/bash_tamia_promptmix.sh` (port of `bash_fir_promptmix.sh`: account,
  12 h jobs, paths, STAGE_LOCAL off; trainer settings identical). Data checked: `$SCRATCH/data/promptmix/train_17K.jsonl` (md5
  5a7d8259) + `val_400.jsonl` (= first 400 rows of val_800), 0 overlap with either val set (no new copy needed).
  **Smoke `pm_smoke_tamia` passed** (636249, 19 min: judge 40 s, both val sets, steps 1–2 at ~1.3 min/step on 1 node,
  checkpoint-2, val_dump, `Done.`; DATASET_SEED from RUN_NAME). **C submitted 2026-10-23 13:45: jobs 639316 → 639317** (4 nodes
  × 12 h, `afterany`): `KL_COEF=0.01 MAX_STEPS=200 EVAL_STEPS=50 RUN_NAME=darshan_pm1_C_kl01_tamia … bash_tamia_promptmix.sh`. Report every eval (val + val_promptmix: am_0, am_final, oracle, fix/brk/looks_good) to Fir's Inbox and the
  `runs.md` row; decision at step 200 (rule in `fir.md` Now). Fir runs A, Rorqual B.
- G stopped at ~433 (last eval step 400: own 0.757, oracle 0.791); promptmix t1 full-history eval done (collapse at depth 3–4);
  earlier tasks: see Log and `runs.md`. Mila asked for G's checkpoints (copy needs the user).
- Notes are edited in the main checkout `tar_reason/`. Watcher: background `wait_event.sh` on C + Inbox (restarted 2026-10-23 13:50; the earlier one stopped on a job's first PENDING line, now ignored).

## Setup
- User `dars11`, single account `aip-agrawal`. `$SCRATCH=/scratch/d/dars11` (2 TB quota), `~/scratch` → `$SCRATCH`,
  home `/home/d/dars11` (25 GB). Paths differ from Fir by the extra `d/` level.
- Layout: repo `/scratch/d/dars11/git/tar_reason` (git identity `Darshansingh11` set repo-locally; GitHub SSH key
  `~/.ssh/id_ed25519` added), models `$SCRATCH/models/{tar,vlm}` (Tar-7B-v0.1, TA-Tok, de-tokenizers, SigLIP2 config,
  `sft/slf_ref_edit_t21_ckpt17000` from Fir, Qwen3-VL-8B-Instruct), data `$SCRATCH/data/geneval2_50K_v2` (md5 prefixes
  and 0 prompt overlap verified), envs `~/envs/tar` and `~/envs/qwen_judge` built with uv from the Fir freeze files
  (`$SCRATCH/tamia_transfer/`; imports verified). uv in `~/.local/bin`, `UV_CACHE_DIR=$SCRATCH/.uv_cache`.
  wandb logged in (2026-10-09).
- GPU partitions are whole-node only: `gpubase_bynode_b1/b2/b3` (3 h / 12 h / 24 h limits) and `gpubase_interac`
  (6 h); `--time` picks the partition. H100 nodes: 4× H100 80GB SXM (NVLink), 48 cores, 500 GB. H200 nodes: 8 GPUs,
  64 cores, 1 TB. CUDA modules `cuda/12.2`, `cuda/12.6` (launcher keeps 12.2).
- Login nodes have internet (GitHub, HF); compute nodes assumed not, so wandb is offline (`wandb sync` from a login node).
- Queueing: `aip-agrawal` is over its fair share (effective usage ~3.8% vs 1% share, 2026-10-09), so jobs wait on
  `Priority`. 1 vs 2 nodes changes the estimate by minutes; 24 h jobs wait much longer than 12 h; one H200 node
  is no faster. `--test-only` estimates were ~6 h more optimistic than `squeue --start` for the real job.
- Launcher `scripts/rl_ft/bash_tamia_t7.sh`: copy of `bash_fir_t7.sh` with only account, job name and paths changed.
- Session: login node `tamia1`, session id `a2cf8ab9-d921-49b4-af79-f29b7522648d`; resume with
  `cd ~/scratch/git && claude --resume a2cf8ab9-d921-49b4-af79-f29b7522648d` (in tmux `claude` when away). Watcher: Monitor running `watch_job.py <name> <log_prefix> <jobids…>` (generic; in this session's scratchpad `/tmp/claude-3165746/.../a2cf8ab9.../scratchpad/`, state `watch_state_<name>.json` there), re-armed every 30 min; reports job state changes, startup lines, steps 1–2 and every 10th, spikes (KL > 0.5 / grad > 1), warnings (grad > 0.3, clip_frac > 0.1, kl_img > 0.05), val results, checkpoints, errors, `Done.`. Dies with the session. 
- Claude Code's auto mode blocks multi-node `sbatch` ("Shared Cluster Mutation"); the user switched mode to approve.

## Inbox
- 2026-10-23 from Fir: status check: no note since 2026-10-22 02:50 (C smoke 636249 queued, F-full mdp eval 636248 queued). Please report where C (`darshan_pm1_C_kl01_tamia`) stands (smoke result, full run job ids, any evals) and the F-full mdp eval result. For context: B (BRANCH 4,4) won the screen and beats jeet's t1 at 500 steps (val800 0.783, promptmix 0.740); A/D changed nothing; E (BRANCH 8,4) is ahead of B per step. C's 200-step result still matters (KL axis).

## Log (newest first)
- 2026-10-23 13:45: smoke 636249 passed; F-full mdp eval 636248 done (own 0.847, oracle 0.898 = F-mdp); promptmix C submitted (639316 → 639317). Both jobs had finished on 10-22 unreported: the watcher had exited on the first PENDING line.
- 2026-10-22 02:50: F-full done (step-300 own 0.833, oracle 0.897; checkpoint-300 kept; wandb synced); its mdp eval submitted (636248, old worktree). Fir Inbox (single-round promptmix screen, variant C) handled: new worktree `tar_reason_pm`, launcher `bash_tamia_promptmix.sh` (`b5ea15e`), data verified, smoke 636249 submitted.
- 2026-10-20 19:15: Fir Inbox (jeet's latest main merged; SIGLIP2_PATH must be exported; new-run settings) handled: `bash_tamia_mr.sh` defaults updated in the main checkout and pushed; worktree untouched (F-full 621220 still queued on `cb1c831`).
- 2026-10-20 03:50: user resumed in tmux; watcher on F-full (621220) and Inbox started.
- 2026-10-20 02:45: F-full 621219 TIMEOUT at step ~275; checkpoint-250 kept; 621220 pending. Inbox handled (Mila TA-Tok fix + scorer alignment; Fir merge of jeet's main; dataset-seed rule): decided to keep the worktree at `cb1c831` until F-full is done (rule 4), then rebase before any new run.
- 2026-10-19 13:30: caught up after the tmux session ended: promptmix full-history eval done (collapse at depth 3–4); F-full at step ~200 (step-200 eval own 0.819); checkpoint-150 lost to pruning, checkpoint-200 kept.
- 2026-10-18 00:30: promptmix files verified; LoRA merged (626094); full-history depth-4 eval submitted (626099).
- 2026-10-18 00:20: Fir Inbox handled: G stopped (621301–06 cancelled, last eval step 400 flat vs 300; checkpoint-400 kept; wandb piece synced); F-full kept (step-100 eval: own 0.817 vs 0.796 at step 0; checkpoint-100 kept); promptmix full-history eval waits for the user's copy from Fir.
- 2026-10-17 12:57: F-full step-0 eval (full history) reproduces t8: own am_upto 0.778/0.821/0.820/0.811/0.796, oracle → 0.863.
- 2026-10-17 12:45: G-1000 621301 started on tg[10701,10802,10805-10806], resumed from checkpoint-300. Both chains running.
- 2026-10-17 12:05: F-full 621218 started on tg[10606,10808,10908,11302]; startup checks pass.
- 2026-10-15 22:46: user resumed in tmux (auto mode); background wait on F-full/G start and Inbox started.
- 2026-10-15 23:00: G → 1000 submitted (621301–06) in manual mode; F-full left queued (both run).
- 2026-10-15 22:50: Fir Inbox (G to 1000 steps, priority over F-full) taken into Now; resume checks done; submit blocked by auto mode.
- 2026-10-15 21:45: Fir Inbox (F-full) handled: worktree rebased to `cb1c831` (nothing was queued), submitted `darshan_mr_full_tamia` 621218 → 19 → 20.
- 2026-10-15 06:40: G done (616485 COMPLETED 06:28, `Done.`); step-300 eval own 0.757, oracle 0.803; checkpoint-300 kept; wandb synced (3 pieces with `--append`, verified lastHistoryStep 300 and 4 evals); worktree rebased.
- 2026-10-15 02:24: G 616485 started, resumed from checkpoint-260.
- 2026-10-15 02:21: G 616484 TIMEOUT at step ~260 (checkpoint-260); checkpoint-250 kept. Steps 200–260 stable, KL 0.04 → 0.21 (gentle rise), grad ≤ 0.07.
- 2026-10-14 23:25: user resumed in tmux; watcher on G restarted (step 223).
- 2026-10-14 23:25: Fir Inbox (3a stop-SFT moved to Mila) handled: no 3a work had been started on TamIA. Pulled notes into the main checkout only; the worktree stays at its pre-`--stop_reward` code until G's last job (616485) has started/finished (rule 4; the change is default-off anyway).
- 2026-10-14 21:33: G step-200 eval: own am_upto 0.723/0.743/0.749/0.742/0.744, oracle → 0.789; draft +0.039, final +0.069 vs step 0. Checkpoint-200 kept.
- 2026-10-14 14:21: G 616484 started, resumed from checkpoint-120.
- 2026-10-14 14:04: G job 616483 hit its 12 h limit at step ~128; 616484 queued to resume from checkpoint-120.
- 2026-10-14 11:48: G step-100 eval: own final 0.675 → 0.697 (best depth 2: 0.705), oracle 0.697 → 0.728 (`runs.md`). Checkpoints 50 and 100 kept in `$SCRATCH/tar_reason_kept/darshan_mr_sft_tamia/`.
- 2026-10-14 04:20: Mila Inbox (tasks H/I/J done; MDP holds on official GenEval2: 77.9 vs 74.3 full at depth 4) noted; G-checkpoint copy request moved to Now.
- 2026-10-14 02:56: G step-0 eval (SFT baseline, depth-4 mdp): am_upto own 0.684 → 0.675, oracle 0.681 → 0.697 (`runs.md`).
- 2026-10-14 02:10: pulled Mila's benchmark-script commits into the worktree while G runs (rule 4 check done after the fact): only `eval/*` and `scripts/eval/bench_mila.sh` changed, which the RL trainer and `bash_tamia_mr.sh` don't use, so 616484/85 are unaffected.
- 2026-10-14 02:03: task G 616483 started (4 nodes); startup checks pass.
- 2026-10-13 22:30: user resumed in tmux; watcher on G (616483) started. No new commits; Inbox empty.
- 2026-10-13 20:40: task G `darshan_mr_sft_tamia` submitted (616483 → 84 → 85, 4 nodes × 12 h, afterany) after the user switched to manual mode.
- 2026-10-13 20:30: pulled `c39b934` (plan F–N; children_r1). Task M done (CPU, `t8_depth4_mdp_tamia` val_dump) → `runs.md`, Fir's Inbox. Task G submit blocked by auto mode.
- 2026-10-13 19:45: Fir Inbox handled: backfilled `val{,_oracle}/am_upto_{0..4}` and `allpass_upto_{0..4}` into the wandb summaries of `t8_depth4_full_tamia` and `t8_depth4_mdp_tamia` (from val_dump, resume="must", no new run; script `backfill_depth.py` in this session's scratchpad). Checks: am_upto_0 = am_0 and am_upto_4 = am_final to 1e-15; history unchanged. Oracle drafts without a child count as all-pass (oracle gives no critique only when no question fails). New metrics noted; pull done.
- 2026-10-13 19:25: wandb synced `t8_depth4_full_tamia` and `t8_depth4_mdp_tamia` (one offline folder each, in `tar_reason_t11/wandb/`); verified via the API (finished, step 0, summaries match the logs).
- 2026-10-13 19:15: Fir Inbox (MDP training landed) handled: pulled `5fe658e` into both checkouts (nothing queued).
- 2026-10-13 06:15: task C (`t8_depth4_mdp_tamia`, 614753) and multi-round smoke (614754) done; results to `runs.md` and Fir's Inbox. CPU tests `test_context_mode.py` and `test_chain_mode.py` passed on the t8 tokenizer.
- 2026-10-13 05:00: Inbox handled: Mila's official-GenEval2 depth result noted; task C (MDP eval) submitted as 614753; multi-round smoke 614754 submitted (F itself waits for the user). Pulled `--context_mode`/`--chain_siblings` code into the worktree (nothing was queued).
- 2026-10-13 04:45: `t8_depth4_full_tamia` done (614114, 02:54–04:27); results to `runs.md` and Fir's Inbox. Final AM by depth computed from val_dump.
- 2026-10-12 20:55: Fir's trimmed plan: cancelled 614104 (`t8_depth4_tamia`) and 614105 (SFT); resubmitted t8 as `t8_depth4_full_tamia` (614114). text/mdp evals wait for Fir's context-mode code.
- 2026-10-12 20:50: Fir Inbox (plan A–G) handled: cancelled `darshan_t11_tamia` (612742–44); added `PREV_STAGE_CHECKPOINT` override to `bash_tamia_t11.sh` (`28f6d3b`); submitted task B evals 614104 (t8) and 614105 (SFT), 2 nodes × 8 h.
- 2026-10-12 13:20: user back in VSCode; tmux session's watchers gone. 612742 still pending, est. 20:30.
- 2026-10-12 05:47: t11 smoke test passed (612737); 612742 released from its dependency.
- 2026-10-12 05:46: t11 smoke test passed (job 612737); details in Now and `runs.md`.
- 2026-10-12 05:00: user asked to run t11 on TamIA too; submitted `darshan_t11_tamia` 612742 (afterok smoke) → 43 → 44.
- 2026-10-12 04:58: user switched to tmux; generic watcher `watch_job.py` started on the t11 smoke test.
- 2026-10-12: t8 merged model verified; t11 smoke test submitted (job 612737).
- 2026-10-12 04:45: user cancelled 516261; t7 stops at step 250. Main checkout pulled.
- 2026-10-12 04:35: Fir Inbox items (t11 setup; t11 runs on Fir, TamIA backup) taken into Now: worktree `tar_reason_t11` at `0e32009`, queue estimates sent to Fir's Inbox; waiting for the t8 model copy. Main checkout not pulled (516261 queued).
- 2026-10-11 18:45: caught up after the tmux session ended (~2026-10-10 14:00; watcher gone). 516259 TIMEOUT at step ~120, 516260 resumed 120 → 250, TIMEOUT during step-250 eval. Val 100/150/200 in `runs.md`. Checkpoints 150/200 lost to pruning. 516261 pending, est. 2026-10-12 17:07.
- 2026-10-10 12:52: spike at step 88 (KL 1.63, grad 39, clipped to 1) after build-up at 86–87; step 89 KL 0.73 / grad 1.4; recovered at 90 (KL 0.14, grad 0.20). Train fix_1 stayed ~25%. Pre-spike `checkpoint-85` copied to `$SCRATCH/tar_reason_kept/darshan_t7_tamia/`.
- 2026-10-10 11:45: steps 67–71 unstable (KL peak 0.34 @69, grad 0.67 @70, clip_frac 0.28 @70–71), recovered by step 72 (KL ~0.09, grad ~0.05, clip ≤0.005). Watcher now also reports any step with grad > 0.3 or clip_frac > 0.1.
- 2026-10-10 04:13: job 516259 started (queued since 15:00; estimates had slipped 23:45 → 01:45 → 05:30).
- 2026-10-09 15:30: pulled `839ccc2` (handoff-only); recorded login node and session id.
- 2026-10-09: pulled `ba65f42` (handoff/ folder); `~/scratch/git/CLAUDE.md` is now a symlink to `handoff/CLAUDE.md`;
  local `TAMIA_HANDOFF.md` deleted (contents are here; original in `archive/`). Inbox from Fir handled: RUN_NAME fix
  pulled (line 125 verified), notes moved, Fir smoke reference noted in `runs.md`.
- 2026-10-09: submitted `darshan_t7_tamia` (2 nodes, 516259 → 60 → 61) at the user's request.
- 2026-10-09: pulled `b0c8c31` (RUN_NAME export fix from Fir) before any full-run submit.
- 2026-10-09: smoke test `t7_smoke_tamia` passed (job 515432, 1 node, 21 min): judge ready 40 s (load 28 s), steps
  0–2, checkpoint-2 (1.9 GB), `Done.` Rollout ~155 s, reward ~75 s, train 33–63 s ≈ 4.5–5 min/step at 18/step.
  Peak GPU memory not reported by `seff`/`sacct` on TamIA.
- 2026-10-09: setup done (envs, weights, data, launcher `f473e8c`).
