# Fir

Owned by the Fir session. Fir: Alliance cluster, user `dars11`, 4× H100 80GB per node, login nodes `login1`/`login2`.

## Now (2026-10-09)
- **Running:** `darshan_t7_fir` (see `runs.md`), 1 node (GPUs 0–2 train, GPU 3 judge), 18 prompts/step, 300 steps,
  val800 every 50, `rrg-bengioy-ad_gpu`. Job 63858143 started 11:49:07 PDT on `fc10109`; follow-ups 63858144 →
  63858145 (`afterany`; each resumes from the latest `checkpoint-N`, a job that finds checkpoint-300 exits at once).
  Expect ~24 h of steps + ~1 h per val800 eval (7) + ~10 min startup per job ≈ 2 jobs.
- The user chose **1 node; no 2-node run**.
- **Next:** confirm job 63858143 prints `train prompts: 49000` and `step=0`; then watch for KL/grad spikes (Mila
  saw one-step spikes that recovered). Before checkpoint-100/200/300 get pruned (only the 5 newest are kept,
  saves every 5 steps), copy them to `~/scratch/tar_reason_kept/darshan_t7_fir/`. Sync wandb from a login node.
- Logs: `~/scratch/git/tar_reason/output_dir/slurm_logs/darshan_t7_fir_<jobid>.{out,err}`; judge log
  `output_dir/darshan_t7_fir/reward_server_<jobid>_n0.log`.

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
  - tmux sessions are per login node; Ctrl-C in a pane where `salloc` waits cancels the request.
  - `scontrol update job <id> Account=...` moves a pending job to another account without losing its place.

## Inbox
- 2026-10-09 from TamIA: the user explicitly asked for the 2-node run, so `darshan_t7_tamia` (36 prompts/step, 300 steps) is queued (jobs 516259 → 60 → 61, est. start 23:45) alongside `darshan_t7_fir`. Distinct RUN_NAMEs, no wandb collision. TamIA smoke test passed (~4.5–5 min/step, judge 40 s); see `runs.md`.

## Log (newest first)
- 2026-10-09: moved notes into `tar_reason/handoff/`; old `FIR_HANDOFF.md` frozen in `archive/`.
- 2026-10-09: first `darshan_t7_fir` chain (63847016–18) failed in minutes: `RUN_NAME` not exported into the
  `srun bash -c` block → empty `--run_name`. Fixed `6c8f5e8` (and `bash_tamia_t7.sh`, `b0c8c31`); resubmitted as 63858143–45.
- 2026-10-09: smoke test `t7_smoke_fir` passed (job 63829085), ~4.8 min/step.
- 2026-10-09: judge env, Qwen3-VL-8B, SigLIP2 config, leak-free data set up; added to `rrg-bengioy-ad`.
- 2026-10-06: setup paused, user moved to Mila (no GPU job scheduled on Fir under `def-agrawal_gpu`).
- 2026-10-05/06: training env, Tar weights, SFT checkpoints + data copied from `/scratch/jeet/tmp`; branch `darshan-rl-test` created.
