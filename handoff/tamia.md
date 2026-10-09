# TamIA

Owned by the TamIA session.

## Now (2026-10-09 15:30 EDT)
- `darshan_t7_tamia` is **queued**: t7 recipe on **2 nodes × 4 H100**, 36 prompts/step (t7's batch), 300 steps,
  val800 every 50. Jobs 516259 → 516260 → 516261 (12 h each, `gpubase_bynode_b2`, `afterany` chain), code at
  `b0c8c31`. The user asked for it explicitly, in addition to the 1-node `darshan_t7_fir` (18/step): the pair tests
  the batch-size question. Different `RUN_NAME`s, so no wandb collision.
- Estimated start 2026-10-10 01:45 (was 23:45; slipping), held back by fair-share (see Setup). 300 steps at ~5 min/step + 7 val800 evals
  ≈ 30 h, i.e. all three jobs.
- First checks once it starts (`output_dir/slurm_logs/darshan_t7_tamia_516259.out`): `judge ready` from both nodes;
  `train prompts: 49000 (per rank/epoch 8166 or 8167), val prompts per rank: 133 or 134`; `step=0`.
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
  `cd ~/scratch/git && claude --resume a2cf8ab9-d921-49b4-af79-f29b7522648d` (in tmux `claude` when away). No watchers yet.
- Claude Code's auto mode blocks multi-node `sbatch` ("Shared Cluster Mutation"); the user switched mode to approve.

## Inbox

## Log (newest first)
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
