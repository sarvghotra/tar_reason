# Fir

Owned by the Fir session. Fir: Alliance cluster, user `dars11`, 4× H100 80GB per node, login nodes `login1`/`login2`.

## Now (2026-10-10 07:15 PDT)
- **Running:** `darshan_t7_fir` (see `runs.md`), 1 node (GPUs 0–2 train, GPU 3 judge), 18 prompts/step, 300 steps,
  val800 every 50, `rrg-bengioy-ad_gpu`. Job **63858144** (started 17:11 PDT on `fc10109`, resumed from
  checkpoint-40, 24 h limit → ends ~17:10 on 10-10) reached step 150 at 05:35; spare **63858145** queued behind it.
- 63858143 died at step 41 (17:10:52) with SIGBUS (exit −7) on all 3 ranks at once, mid-step: likely a scratch
  (Lustre) hiccup under memory-mapped files, not a code bug. The chain resumed it 16 s later. If it recurs, only
  one spare job remains: queue another `afterany` follow-up.
- Step-150 val800: draft and final AM keep rising (0.742 → 0.756); own-critique fix flat at ~24.4% since step 50
  while oracle-critique fix rose to 38.2% (critic now limits refinement; numbers in `runs.md`). KL drifted up to
  0.2–0.4 in steps 145–149 (Mila late baseline 0.1–0.2), one-step spike to 1.37 at step 168 that recovered
  next step; watch it. checkpoint-100 kept (verified identical).
- Remaining: ~205 steps × 5.1 min + 5 evals × 80 min ≈ 24 h, so 63858145 will be needed (~17:10 on 10-10).
- Watcher (Monitor in session `2f24f487…`, re-armed every 30 min; `watch_t7_fir_v3.sh` + `watch_parse.py` in that
  session's scratchpad, state in `watch_state/`) reports job state changes, log errors, KL > 0.5 or grad > 1 spikes,
  val800 results, `Done.`, and copies checkpoint-100/200/300 to `~/scratch/tar_reason_kept/darshan_t7_fir/`.
  It dies with the session; restart it after switching clients.
- **Next:** checkpoint-200 copy + step-200 val800 (~11:30–13:00 PDT); job 63858144 hits its time limit ~17:10,
  then 63858145 resumes for steps ~225–300.
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
  - Claude: tmux session `claude` on `login2` (when away); conversation `2f24f487-5bff-422b-bf9d-8947af5a8a9f`,
    resumed from `/scratch/dars11/git`.
  - tmux sessions are per login node; Ctrl-C in a pane where `salloc` waits cancels the request.
  - `scontrol update job <id> Account=...` moves a pending job to another account without losing its place.

## Inbox
(empty)

## Log (newest first)
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
