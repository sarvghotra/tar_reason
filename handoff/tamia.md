# TamIA

Owned by the TamIA session.

## Now (2026-10-15 22:50 EDT)
- **G → 1000 steps (user + jeet via Fir, 2026-10-15; priority over F-full):** ready, **not submitted yet** (auto mode blocked the
  multi-node sbatch; waiting for the user). Checked: `checkpoint-300` in the run dir (`state.json` step 300); launcher unchanged since
  G ran; `train_grpo.py` changes since then are default-off (`--stop_reward`, `--val_dump_codes`, `--eval_sample_critiques`); warmup =
  5% of 1000 = 50 updates < 300, so the LR stays 1e-4. Command (from the worktree, 6 jobs chained with `afterany`):
  `MAX_STEPS=1000 PREV_STAGE_CHECKPOINT=/scratch/d/dars11/models/tar/sft/slf_ref_edit_t21_ckpt17000 LR=1e-4 RUN_NAME=darshan_mr_sft_tamia
  sbatch -J darshan_mr_sft_tamia scripts/rl_ft/bash_tamia_mr.sh`. Open question for the user: hold F-full (621218, queued first,
  est. 2026-10-16 14:39) so G goes first, or let both run.
- **Plan A–G** (`runs.md` "Plan"). Code is written by the Fir session; TamIA pulls it into the worktree
  `/scratch/d/dars11/git/tar_reason_t11` (at `16f2cf1`+, includes `--context_mode` and `--chain_siblings`).
  Fir's GPUs go down 2026-10-13 10:00 PDT, so **task F (N=4 multi-round training from t8) will likely run on TamIA**;
  it waits for the user's go (and jeet's input on `draft_reward`). Don't submit F before that.
- **Task B done:** `t8_depth4_full_tamia` (614114). With full history, own critique is best at depth 1 (final AM 0.827 →
  0.795 at depth 4), oracle best at depth 3 (0.867). Mila's official GenEval2 run: own critique best at depth 2
  (76.4), also declining after. Details in `runs.md`.
- **Task C done:** `t8_depth4_mdp_tamia` (614753). With MDP context t8 does not get worse with depth: own critique
  final AM 0.826 (depth 1) → 0.833 (depth 4), oracle 0.848 → 0.885; full history at depth 4 was 0.795 / 0.866. Details in `runs.md`.
- **Multi-round smoke `mr_smoke_tamia` passed** (614754, 30 min, ~4.2 min/step on 1 node): TamIA is ready for task F.
- **Task G done** (`darshan_mr_sft_tamia`, 616483–85, finished 2026-10-15 06:28, 300 steps, wandb synced). Depth-4 mdp eval,
  final AM own 0.675 → **0.757** (best depth 2: 0.761), oracle 0.697 → **0.803**, draft 0.684 → 0.740; still below t8/F
  (`runs.md`). Checkpoints 50–300 kept in `$SCRATCH/tar_reason_kept/darshan_mr_sft_tamia/`.
  **Mila asked for G's checkpoints** (GenEval2, TIIF, GenAI, MDP depth): copying TamIA → Mila needs the user's Mila login;
  ask the user which checkpoints (300, maybe 200) and run the copy in tmux.
- **Task F-full queued (user's go via Fir, 2026-10-15): `darshan_mr_full_tamia`**, F's recipe with `CONTEXT_MODE=full` (merged t8
  base, LR 5e-5, `children_r1`, chain mode, micro-batch 2, 300 steps, depth-4 full-history evals every 100). Jobs **621218 → 621219
  → 621220** (4 nodes × 12 h, `afterany`), worktree at `cb1c831`. Expected step 0 ≈ `t8_depth4_full_tamia` (own 0.795, oracle 0.866);
  F (mdp) reached 0.848. After step 300: eval the final checkpoint with `--context_mode mdp` (`MAX_STEPS=0`). Report evals to
  `runs.md` and Fir's Inbox. Copy checkpoints at multiples of 50 to `$SCRATCH/tar_reason_kept/darshan_mr_full_tamia/`.
  **While F-full runs, edit notes in the main checkout** and don't rebase the worktree onto code changes (rule 4).
- Optional later: F continued 300 → 500 on TamIA if the user copies F-300 here.
- **Task M done** (CPU): own-vs-oracle gap at depth 4 is ~half stopping, ~half critique content (`runs.md` cross-run findings).
- t11 on TamIA cancelled; t7 stopped at 250 (results in `runs.md`).
- Checkouts: `tar_reason/` (main) and the worktree (queued jobs use it). Before pulling with jobs queued, check rule 4.

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

## Log (newest first)
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
