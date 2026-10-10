# TamIA

Owned by the TamIA session.

## Now (2026-10-10 09:40 EDT)
- **Running:** `darshan_t7_tamia`: t7 recipe on **2 nodes × 4 H100** (GPUs 0–2 train, GPU 3 judge per node),
  36 prompts/step (t7's batch), 300 steps, val800 every 50. Job 516259 started 04:13 on `tg[10906,10908]`
  (12 h, `gpubase_bynode_b2`); follow-ups 516260 → 516261 (`afterany`; each resumes from the latest checkpoint).
  Code at `b0c8c31`. Runs alongside the 1-node `darshan_t7_fir` (18/step): the pair tests batch size.
- Startup passed: both judges ready after 30 s; `train prompts: 49000 (per rank/epoch 8167), val prompts per rank: 134`.
  Step-0 val800 (04:54): own final − draft +0.008, fix 14.9%, brk 7.0%; oracle +0.011, fix 14.9%, brk 7.7% (see `runs.md`).
  Step-50 val800 (09:36): own final − draft +0.001, fix 18.7%, brk 9.4%; oracle +0.017, fix 19.3% (see `runs.md`).
  KL 0.09 at step 50, no spikes; ~4.9 min/step. Job 516259 ends ~16:13 (~step 125); 516260 continues.
- **Next:** record step-0 val800 vs Mila/Fir and s/step; watch KL/grad spikes; copy checkpoint-100/200/300 before
  pruning (only 5 newest kept) to `$SCRATCH/tar_reason_kept/darshan_t7_tamia/`; `wandb sync` from a login node.
- Queued jobs run whatever is checked out when they start: check `git log HEAD..origin/darshan-rl-test --stat`
  before pulling code changes (CLAUDE.md rule 4).

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
  `cd ~/scratch/git && claude --resume a2cf8ab9-d921-49b4-af79-f29b7522648d` (in tmux `claude` when away). Watcher: Monitor running `watch_t7_tamia.py` (in this session's scratchpad, `/tmp/claude-3165746/.../a2cf8ab9.../scratchpad/`; state in `watch_state.json` there), re-armed every 30 min; reports job starts, startup checks, every 10th step, KL > 0.5 / grad > 1 spikes, val results, ckpts at multiples of 50, errors, `Done.`. Dies with the session.
- Claude Code's auto mode blocks multi-node `sbatch` ("Shared Cluster Mutation"); the user switched mode to approve.

## Inbox

## Log (newest first)
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
