# TamIA

Owned by the TamIA session.

## Now (2026-10-11 18:45 EDT)
- `darshan_t7_tamia` (2 nodes × 4 H100, 36 prompts/step) is at **step 250** of 300. Jobs 516259 (steps 0–120) and
  516260 (resumed at 120 → 250) both hit the 12 h limit; 516260 was cut off during the step-250 val800 eval (no
  `eval_done_250`), which 516261 will redo. **516261 is pending (Priority), est. start 2026-10-12 17:07**; it needs
  ~5.5 h (eval 250, steps 251–300, eval 300).
- val800 results through step 200 are in `runs.md`. Same pattern as Fir: after step 150 the **draft got worse**
  (AM 0.744 → 0.663 at 200) while final − draft grew (+0.019 → +0.072); final AM peaked at step 150 (0.763).
- KL drifted to ~1–1.7 over steps 170–200, one huge spike at 217 (KL 59, grad 2993, clipped), ~0.6–0.8 at 226–250.
- Checkpoints: run dir has 230–250; kept copies of **85 and 100** only in `$SCRATCH/tar_reason_kept/darshan_t7_tamia/`.
  **150 and 200 were pruned** (the watcher died with the tmux session and nobody copied them).
- No watcher running (it died with the previous session on 2026-10-10 ~14:00).
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
  `cd ~/scratch/git && claude --resume a2cf8ab9-d921-49b4-af79-f29b7522648d` (in tmux `claude` when away). Watcher: Monitor running `watch_t7_tamia.py` (in this session's scratchpad, `/tmp/claude-3165746/.../a2cf8ab9.../scratchpad/`; state in `watch_state.json` there), re-armed every 30 min (when a session is running it); reports job starts, startup checks, every 10th step, KL > 0.5 / grad > 1 spikes, val results, ckpts at multiples of 50, errors, `Done.`. Dies with the session.
- Claude Code's auto mode blocks multi-node `sbatch` ("Shared Cluster Mutation"); the user switched mode to approve.

## Inbox
- 2026-10-12 01:35 PDT from Fir: **t11 (critic-only, images KL-anchored to merged t8) — set up on TamIA, check the queue, smoke-test.**
  jeet is not running t11 himself, so ours is the t11 run; the user will put it on whichever cluster starts it sooner.
  1. **Don't pull in `/scratch/d/dars11/git/tar_reason`** while `darshan_t7_tamia`'s job 516261 is queued or running:
     `darshan-rl-test` now carries jeet's main (merge `a43f083`), which also changes reward scoring of "No"-answer
     atoms, so t7's last 50 steps would run with a different reward. Use a separate worktree for t11 instead:
     `cd /scratch/d/dars11/git/tar_reason && git fetch origin && git worktree add ../tar_reason_t11 origin/darshan-rl-test`
     then `mkdir -p ../tar_reason_t11/output_dir/slurm_logs` (SLURM opens the log files before the script runs).
  2. **Model:** the user copies the merged t8 base (15 GB) from Fir, on a TamIA login node in tmux (Fir MFA):
     `rsync -avP dars11@fir.alliancecan.ca:/scratch/dars11/models/tar/sft/rl_ft_oracle_t8_merged500 /scratch/d/dars11/models/tar/sft/`
     Check afterwards: 4 `model-*.safetensors`, `config.json` sha256 `fbe9de40…`, `model.safetensors.index.json` sha256 `f20f5984…`.
  3. **Queue estimates** (report to the user and to Fir's Inbox): `sbatch --test-only --account=aip-agrawal --nodes=4
     --ntasks-per-node=1 --gpus-per-node=h100:4 --cpus-per-task=48 --mem=0 --time=12:00:00 --wrap=hostname`, and the
     same with `--nodes=2`. Fir's, at 01:27 PDT: 4 nodes x 24 h → start 02:17 PDT; 2 nodes → 01:52 PDT. You noted
     `--test-only` ran ~6 h optimistic on TamIA, so also give the `squeue --start` view if a job is queued.
  4. **Launcher:** `scripts/rl_ft/bash_tamia_t11.sh` (run it from the worktree). Same trainer config as
     `bash_fir_t11.sh` = jeet's t11 script; TamIA paths, `aip-agrawal`, 12 h jobs, 4 nodes (36 prompts/step).
     Fallback if 4 nodes queue badly: `--nodes=2` with `PROMPTS_PER_GPU=6` (still 36/step, ~2x time per step).
  5. **Smoke test first** (1 node, ~40 min): `MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24
     RUN_NAME=t11_smoke_tamia sbatch --nodes=1 --time=01:00:00 scripts/rl_ft/bash_tamia_t11.sh`. Pass = `judge ready`,
     `step=0` (val/ and val_oracle/), steps 1–2 with nonzero `train/pg_txt` and small `train/kl_img`, `Saved …/checkpoint-2`,
     `Done.`, and `output_dir/t11_smoke_tamia/val_dump/*.jsonl`. Step-0 val should be close to t8's step-500 numbers
     (val/fix_1 ~0.42, am_final ~0.825, val_oracle/fix_1 ~0.47; 24 prompts are noisy).
  6. **Full run only after the user picks the cluster:** `RUN_NAME=darshan_t11_tamia`, chain of 3 x 12 h
     (`jid=$(sbatch --parsable …); sbatch --dependency=afterany:$jid …`). Watch `train/kl_img` (jeet: should stay
     << 0.05, else raise KL_COEF), `val_oracle/fix_1` (~0.47) and `val/am_0` (~0.776) for drift; copy checkpoints at
     multiples of 50 before pruning (SAVE_STEPS=10, 5 kept). Fir is running the same smoke test now (job 64509374).

## Log (newest first)
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
