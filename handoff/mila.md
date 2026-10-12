# Mila

Owned by the Mila session. Mila: user `singhsd`, project folder `/home/mila/s/singhsd/CODE/LatentDCR/`
(`CLAUDE.md` there → `tar_reason/handoff/CLAUDE.md`, old copy `CLAUDE.md.old`; `papers/` and `tools/` next to `tar_reason/`).

## Now (2026-10-12 04:50 EDT)
- **Working on the Inbox items from Fir (2026-10-12 01:45 PDT).**
  1. Critic diagnostic: two eval-only jobs (`MAX_STEPS=0 EVAL_ORACLE=0`, val800, own critic, `--val_dump_dir`):
     **11209613** `critic_diag_t7mila200` (`--init_adapter kept/darshan_t7_mila/checkpoint-200`), then **11209614**
     `critic_diag_sft` (no adapter), chained `afterany`, `short-unkillable` 4× A100, ~1 h each. Then
     `critic_diagnostic.py` on `output_dir/<run>/val_dump`; results go here and to `runs.md`.
  2. Official benchmarks: surveying what in the repo (and `xiaofeng`) can evaluate a LoRA checkpoint iteratively;
     a setup proposal goes to the user before any GPU use.
- `bash_mila_t7.sh` now takes `EXTRA_ARGS` and `EVAL_ORACLE` (`bd82b85`).
- `darshan_t7_mila` is done (200 steps); checkpoints 100/200 in `$SCRATCH/tar_reason/kept/darshan_t7_mila/`.

## Setup
- Session: Claude Code session `830290ea-cb02-4188-9cf2-1c5fb3768fa2`, opened from the VSCode plugin via
  `mila code CODE/LatentDCR --alloc …` (runs on a compute node, not a login node). Resume:
  `cd ~/CODE/LatentDCR && claude --resume 830290ea-cb02-4188-9cf2-1c5fb3768fa2`. No tmux login node chosen yet
  (Mila's `login.server.mila.quebec` round-robins over several login nodes; pick one with `ssh login-N`).
- Envs: `~/envs/tar` (training), `~/envs/qwen_judge` (judge; freeze in `~/envs/qwen_judge.freeze.txt`).
- Weights: `$SCRATCH/models/tar/` (Tar-7B, `ta_tok.pth`, `ar_dtok_lp_512px.pth`, `vq_ds16_t2i.pt`,
  `siglip2-so400m-patch14-384/config.json`, `sft/slf_ref_edit_t21_ckpt17000`), `$SCRATCH/models/vlm/Qwen3-VL-8B-Instruct`.
- Data: `$SCRATCH/data/geneval2_50K_v2/` (train 49,000 + val800 + val256). `$SCRATCH/data/geneval2_50K/` is the
  old 49,488-row copy (288 val800 prompts inside); `val800_train_overlap.tsv` there lists them.
- `tar_reason/output_dir` → `$SCRATCH/tar_reason/output_dir`; SLURM logs in `output_dir/slurm_logs/`.
  Artifacts can't be published from there (resolves outside the allowed roots): copy out first.
- Partitions: `main` caps a user at 2 GPUs / 8 CPUs / 48 GB; `long` has no cap but is preemptible (requeued jobs
  resume); `short-unkillable` allows 4 GPUs for 3 h, not preempted (QoS: 4 GPUs per user). Use `a100l` (80 GB);
  the 2 `h100` nodes are usually full. A full `a100l` node with 4 free GPUs is rare on `long`.
- On A100, t7 is ~7.3 min/step (rollout ~225 s, decode+judge ~130 s, train ~95 s); a val800 validation with
  `--eval_oracle` takes 1 h 46 min on 3 ranks. Judge loads in ~50 s.
- sbatch copies the script at submission: editing a launcher does not change already-queued jobs.
- Launchers: `bash_mila.sh` (latent reward, `BRANCH` from env), `bash_mila_oracle.sh` (latent + oracle critiques,
  std floor 0.04, untruncated sampling), `bash_mila_t7.sh` (t7). Untracked `scripts/rl_ft/bash_t7_sarv.sh` = the
  collaborator's original t7 script.
- Tools outside the repo, Mila only: `tools/walkthrough/build.py` (generates the code-walkthrough artifact; run
  with `~/envs/tar/bin/python`). In the repo: `scripts/rl_ft/viz/trajectory_gallery.py` (gallery of logged val
  strips; gallery of the first run: https://claude.ai/artifact/RH2zJjLUTSsTtt6PKw3GhR).

## Inbox
- 2026-10-12 01:45 PDT from Fir: **two jobs for Mila while t11 runs on Fir** (details: `runs.md`, collaborator t8–t11).
  Code: pull `darshan-rl-test` (merge `a43f083` brings jeet's main: critique groups, `--train_images`,
  `--image_adv_scale`, `--init_adapter`, merge_lora, negation-atom scoring; ours: `--val_dump_dir`,
  `scripts/rl_ft/critic_diagnostic.py`). Nothing runs on Mila, so pulling is safe.
  1. **Critic diagnostic** (baseline for t11): val800, eval only (`MAX_STEPS=0`), own critic, for
     (a) `--init_adapter $SCRATCH/tar_reason/kept/darshan_t7_mila/checkpoint-200` and (b) the SFT model (no adapter),
     each with `--val_dump_dir output_dir/<run>/val_dump` and a distinct `RUN_NAME`. `bash_mila_t7.sh` needs an
     `EXTRA_ARGS` pass-through like `bash_fir_t7.sh`. `--eval_oracle` is optional (off halves the time). Then
     `python scripts/rl_ft/critic_diagnostic.py output_dir/<run>/val_dump --step 0`; results to `mila.md`, headline
     numbers (stop table, error recall, stop precision, mention rate, fix rate mentioned vs not) to `runs.md`.
     First look on Fir (24 prompts, t7 ckpt-100): stop decisions fine; critiques usually name ONE problem while
     drafts fail 2–5 questions — check that on 800.
  2. **Official benchmarks** (GenEval2, TIIF short/long, GenAI-bench): none has been run on any RL checkpoint yet.
     First find what in the repo can evaluate a LoRA checkpoint iteratively (`eval/`, `tts/eval/`,
     `eval/iterative_generation_adhoc.py`, ...) and what each benchmark needs (data, judge, env; Mila compute nodes
     have internet). **Propose a setup to the user before using GPUs.** Then: SFT t21, t7 Mila checkpoint-100/200,
     later t11's; report draft and final scores separately (final is what matters; t7's draft dropped late on
     Fir/TamIA).

## Log (newest first)
- 2026-10-12 04:45: pulled the t8–t11 merge (`a43f083` … `f1e40b9`); started the critic diagnostic jobs.
- 2026-10-10 05:02: `darshan_t7_mila` done (step-200 val800 in `runs.md`); `keep_t7_ckpts` done (100 + 200 kept).
- 2026-10-10 03:17: checkpoint-200 saved and copied. A leftover watcher from the previous session ate one event:
  after a session restart, check `ps` for old `watch_t7_mila.sh` processes before starting a new one.
- 2026-10-10 02:25: session restarted; rebuilt the watcher in `tools/watch/` (the scratchpad copy was lost).
- 2026-10-09: moved notes to `handoff/` (Fir's scheme): rewrote this file and the Mila rows of `runs.md`;
  `LatentDCR/CLAUDE.md` is now a symlink to `tar_reason/handoff/CLAUDE.md` (old file kept as `CLAUDE.md.old`);
  deleted `LatentDCR/MILA_HANDOFF.md` and `LatentDCR/FIR_HANDOFF.md` (frozen copies in `handoff/archive/`).
- 2026-10-09 12:13: step-100 val800 done; checkpoint-100 copied to `kept/`. Started `keep_t7_ckpts`.
- 2026-10-09: rewrote the code walkthrough for the t7 path (artifact KPqjbeJ5iZMWsFuAyVmUHk, commit `e535421`).
- 2026-10-08 23:10: rechained `darshan_t7_mila` as `MAX_STEPS=200 EVAL_STEPS=100` (12 jobs after 11143217).
- 2026-10-08: `darshan_t7_mila` started (job 11143217: step-0 eval + steps 1–11). Earlier attempts cancelled within
  minutes to switch data: val256 → val800 → val800 minus 288 → the collaborator's leak-free v2 data.
- 2026-10-08: added `--dataset_seed` and redo-of-interrupted-validation to `train_grpo.py` (`a533714`).
- 2026-10-08: built `~/envs/qwen_judge`, downloaded Qwen3-VL-8B; t7 smoke test passed (job 11141217, 29 min).
- 2026-10-08: cancelled `darshan_rl_oracle_b25_mila` at step ~54 after its step-0 diagnosis.
- 2026-10-07: `darshan_rl_2gpu_mila` finished (500 steps, 19 h 20 min). Built the trajectory gallery tool.
- 2026-10-06: latent-reward smoke test passed (1 A100: 42.5 GB peak, ~125 s/step); SigLIP2 path fix `39a6672`.
- 2026-10-05: Mila setup (GitHub SSH key, clone, `~/envs/tar`, HF weights).
