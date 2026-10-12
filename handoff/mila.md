# Mila

Owned by the Mila session. Mila: user `singhsd`, project folder `/home/mila/s/singhsd/CODE/LatentDCR/`
(`CLAUDE.md` there → `tar_reason/handoff/CLAUDE.md`, old copy `CLAUDE.md.old`; `papers/` and `tools/` next to `tar_reason/`).

## Now (2026-10-12 05:30 EDT; the user is away until 2026-10-13, the Mila session runs unattended in tmux)
- **Critic diagnostic done** (results in `runs.md`, row `critic_diag_*`, and the Log below).
- **Inbox item 2, official benchmarks:** the user approved GenEval2 + TIIF (GenAI-bench later), official GenEval2
  `evaluation.py` scoring (+ re-score SFT ourselves), one seed (713). New launcher `scripts/eval/bench_mila.sh`
  (adhoc generators: draft → one greedy critique → one refinement; draft and final scored separately; resumable
  stage markers). Assets: `$SCRATCH/benchmarks/{GenEval2,TIIF-Bench}` (official repos), judges in
  `HF_HOME=$SCRATCH/hf_home` (Qwen3-VL-8B for GenEval2, Qwen2.5-VL-7B for TIIF). The official 800 GenEval2 prompts
  share none with our train / val800 / val256. Smoke test 11209834 passed. Queued on `short-unkillable` 4× A100, each model
  with a spare resubmission: SFT t21 11209951/52 (running), t7 Mila ck-200 11209953/54, ck-100 11209955/56.
  Results: `$SCRATCH/tar_reason/bench/<name>/seed_713/summary.json` (+ TIIF report `tiif/eval_results/`).
- `darshan_t7_mila` is done (200 steps); checkpoints 100/200 in `$SCRATCH/tar_reason/kept/darshan_t7_mila/`.

## Setup
- Session: Claude Code session `830290ea-cb02-4188-9cf2-1c5fb3768fa2`; when the user is away it runs in tmux
  session `claude` on **`login-4`** (`ssh login-4`, `tmux attach -t claude`).
- Earlier note on the same session, opened from the VSCode plugin via
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
- 2026-10-12 01:45 PDT from Fir (item 2 of 2; item 1, the critic diagnostic, is done, see Log): **Official
  benchmarks** (GenEval2, TIIF short/long, GenAI-bench) on SFT t21, t7 Mila checkpoint-100/200, later t11's; report
  draft and final separately. In progress: the user approved GenEval2 + TIIF (GenAI later); runs queued (see Now).

## Log (newest first)
- 2026-10-12 08:50: official benchmarks for SFT t21 and t7 Mila ck-200 done (numbers in `runs.md`, row `bench_mila`):
  GenEval2 final AM 61.8 → 71.7; TIIF short final 63.6 → 71.5, long 60.3 → 61.7 (refine hurts on long).
- 2026-10-12 06:40: critic diagnostic done (Fir inbox item 1). Full reports: `output_dir/critic_diag_{t7mila200,sft}/critic_diagnostic.txt`.
  Stop decision unchanged by t7; Corrections narrower than SFT's (name all of 2 failed objects 44% vs 55%); fix rate
  when mentioned 12% → 26%. Extra breadth script: `tools/watch/critic_breadth.py` (outside the repo).
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
