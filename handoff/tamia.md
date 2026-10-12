# TamIA

Owned by the TamIA session.

## Now (2026-10-12 04:35 EDT)
- **Two checkouts.** `tar_reason/` stays at `caff9b6` (pre-merge code) for t7's job 516261: do **not** pull there
  until 516261 has finished (the merge `a43f083` changes reward scoring of "No"-answer atoms). Notes and t11 live
  in the worktree `/scratch/d/dars11/git/tar_reason_t11` (detached HEAD on `origin/darshan-rl-test`; push notes
  with `git push origin HEAD:darshan-rl-test` after `git pull --rebase origin darshan-rl-test`).
- **t7:** `darshan_t7_tamia` at step 250/300; job 516261 pending (Priority), `squeue --start` 2026-10-12 21:15
  (needs ~5.5 h: eval 250, steps 251–300, eval 300). Same draft-decline-after-150 pattern as Fir (`runs.md`).
  Checkpoints kept: 85, 100 only (150/200 were pruned). No watcher running.
- **t11 (from Fir's Inbox, 2026-10-12):** critic-only GRPO from the merged t8 base, launcher
  `scripts/rl_ft/bash_tamia_t11.sh` (4 nodes, 36 prompts/step, 12 h jobs). Status:
  - worktree created, `output_dir/slurm_logs` made; launcher checked (paths, account, RUN_NAME export).
  - **waiting on the user** to rsync `rl_ft_oracle_t8_merged500` (15 GB) from Fir to
    `/scratch/d/dars11/models/tar/sft/`; then verify (4 shards, `config.json` sha256 `fbe9de40…`,
    `model.safetensors.index.json` sha256 `f20f5984…`) and run the 1-node smoke test `t11_smoke_tamia`.
  - `--test-only` estimates at 04:29 EDT for 12 h jobs: 4 nodes 2026-10-13 04:56, 2 nodes 2026-10-13 02:42,
    1 node 2026-10-12 23:21 (TamIA's `--test-only` has run ~6 h optimistic; 516261 also needs 2 nodes at ~21:15).
  - **Decision (user, via Fir 2026-10-12 01:45 PDT): t11 runs on Fir (4 nodes); TamIA is the backup.** Do setup +
    smoke test so TamIA can take over; **no full `darshan_t11_tamia` run unless the user asks** (e.g. Fir's run fails).

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
  `cd ~/scratch/git && claude --resume a2cf8ab9-d921-49b4-af79-f29b7522648d` (in tmux `claude` when away). Watcher: Monitor running `watch_t7_tamia.py` (in this session's scratchpad, `/tmp/claude-3165746/.../a2cf8ab9.../scratchpad/`; state in `watch_state.json` there), re-armed every 30 min (when a session is running it); reports job starts, startup checks, every 10th step, KL > 0.5 / grad > 1 spikes, val results, ckpts at multiples of 50, errors, `Done.`. Dies with the session.
- Claude Code's auto mode blocks multi-node `sbatch` ("Shared Cluster Mutation"); the user switched mode to approve.

## Inbox

## Log (newest first)
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
