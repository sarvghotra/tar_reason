# TamIA

Owned by the TamIA session.

## Now (2026-10-20 19:15 EDT)
- **G stopped** (user, via Fir 2026-10-17, after the meeting with jeet): `darshan_mr_sft_tamia` cancelled at step ~433 (621301
  cancelled 2026-10-18 00:15; 621302–06 cancelled before starting). Last eval, step 400 (depth-4 mdp, val800): own am_upto
  0.734/0.748/0.762/0.759/**0.757**, oracle **0.791** — flat vs step 300 (0.757 / 0.803). Kept: checkpoints 50–400 in
  `$SCRATCH/tar_reason_kept/darshan_mr_sft_tamia/`; 390–430 still in the run dir. wandb: last piece synced with `--append`.
- **F-full running/queued** (user: keep it): `darshan_mr_full_tamia`, 621218 ran 2026-10-17 12:05–00:05 (TIMEOUT at step ~133,
  checkpoint-130); 621219 ran 2026-10-19 06:35–18:35 (TIMEOUT at step ~275, checkpoint-270); **621220 pending (est. 2026-10-20 21:28)**: steps 271–300 + step-300 eval. Step-100 full-history eval: own am_upto
  0.785/0.826/0.833/0.820/**0.817** (step 0: …/0.796), oracle **0.882** (0.863); the depth-1 → depth-4 drop shrank from
  −0.025 to −0.009. Step 200: own **0.819**, oracle **0.880** (flat vs 100). Checkpoints 50/100/200/250 kept in `$SCRATCH/tar_reason_kept/darshan_mr_full_tamia/` (150 was pruned while no watcher ran). After step 300: mdp eval
  of checkpoint-300 (`bash_tamia_t11.sh`, `MAX_STEPS=0`, `EXTRA_ARGS="--init_adapter <ckpt-300> --eval_branch 1,1,1,1,1 --context_mode mdp"`,
  base = merged t8, worktree still `cb1c831`). Watcher: background `wait_event.sh` (F-full + Inbox), restarted 2026-10-20 19:15.
- **Promptmix t1 full-history depth-4 eval done** (`pm_t1_depth4_full_tamia`, 626099, 3 h 14 min): full history collapses at
  depth 3–4 (GenEval2 own 0.750 at d1 → **0.577** at d4, oracle → 0.696; promptmix own 0.718 → **0.589**) vs mdp 0.761 / 0.809 and
  0.721 / 0.778 (Fir step 0). Reported to `runs.md` and Fir's Inbox. Merged model `$SCRATCH/models/tar/sft/promptmix_t1_merged500`.
- **Code: the worktree stays at `cb1c831` until F-full is finished** (its last job 621220 and the final mdp eval), so all of F-full
  runs one code version. Upstream now has jeet's main merged (`168f38a`: fast RL, string `--stop_reward`, faster AR decoding) +
  Mila's TA-Tok fix (`395fcb4`, needed after that merge) + per-run `--dataset_seed` from RUN_NAME (`d71ee10`). **Before any new
  run after F-full: rebase the worktree.** `bash_tamia_mr.sh` now defaults to jeet's settings (REWARD_QUESTION_BATCH=64,
  GEN_BATCH_SIZE=48, REWARD_DECODE_BATCH=32, TRAIN_MICRO_BATCH=2, EVAL_STEPS=100; latest merge `7e3cbda`/`59c8261` needs
  `SIGLIP2_PATH` exported, which our launchers do). Notes are edited in the main checkout (already at the latest).
- Earlier tasks (B, C, M, G-300, smoke tests, t7, t11): see Log and `runs.md`.
- Mila asked for G's checkpoints (benchmarks); the TamIA → Mila copy needs the user (suggested: checkpoint-300 or -400).

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
- 2026-10-21 from Fir (user + jeet; replaces the earlier line): **single-round promptmix first; hyper-parameter screen.** After F-full finishes (step-300 eval + its mdp eval), rebase the worktree and run **variant C = KL 0.01**: port Fir's `scripts/rl_ft/bash_fir_promptmix.sh` (jeet's t1 recipe from **SFT t21** `slf_ref_edit_t21_ckpt17000`, fresh LoRA, BRANCH 2,4, one round, hybrid critiques 0.5, DRAFT_REWARD children, LR 1e-4, std floor 0.04, EVAL_BRANCH 1,1, speed settings) to TamIA as `bash_tamia_promptmix.sh`, then `KL_COEF=0.01 MAX_STEPS=200 EVAL_STEPS=50 RUN_NAME=darshan_pm1_C_kl01_tamia`, 4 nodes, smoke first. Data: promptmix `train_17K.jsonl` + `val_400.jsonl` (= first 400 rows of the `val_800.jsonl` you already have): the user copies Fir's `~/scratch/data/promptmix/` to `$SCRATCH/data/promptmix/` (`rsync -av fir:scratch/data/promptmix/ tamia:...`). Val = GenEval2 val800 (`val`) + promptmix val_400 (`promptmix`) as in the launcher. Report every eval (val + val_promptmix: am_0, am_final, oracle, fix/brk/looks_good) to Fir's Inbox and your `runs.md` row; decision at step 200 (rule in `fir.md` Now). Fir runs A (oracle frac 0.75), Rorqual B (BRANCH 4,4).

## Log (newest first)
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
