# Mila

Owned by the Mila session. Mila: user `singhsd`, project folder `/home/mila/s/singhsd/CODE/LatentDCR/`
(`CLAUDE.md` there → `tar_reason/handoff/CLAUDE.md`, old copy `CLAUDE.md.old`; `papers/` and `tools/` next to `tar_reason/`).

## Now (2026-10-13 00:40 EDT)
- **Nothing running on Mila.** Task D (plan): t8 ck-500 done on the official benchmarks (GenEval2 final 76.8, TIIF
  short/long final 70.5/68.6) and on depth (official GenEval2, full history: best at 2 rounds, 76.4; rounds 3–4 lower
  it to 74.3). Both in `runs.md`.
- **Open items of task D:** t11 checkpoints 150/200/300 once Fir keeps them (t11 step-150 val800 is flat vs step 0,
  per `runs.md`), and the GenAI-bench setup (needs the user's go-ahead: third env with transformers ≥ 5, 54 GB judge).
- To benchmark another checkpoint: `MODEL_NAME=<label> LORA_PATH=<ckpt> sbatch scripts/eval/bench_mila.sh`
  (merged models: `BASE_MODEL=<dir>`, no `LORA_PATH`), ~66 min on 4× A100. Depth: `bash_mila_t7.sh` with
  `MAX_STEPS=0 EVAL_ORACLE=0 VAL_JSONL=<split> EXTRA_ARGS="--model_name_or_path <model> --eval_branch 1,1,1,1,1
  --val_dump_dir output_dir/<run>/val_dump"` (~70–80 min per 400 prompts), then `tools/watch/depth_curve.py <dumps>`.

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
- 2026-10-13 from Fir (user's go), three tasks in this order (plan rows H, I, J in `runs.md`):
  **H**: t8 depth-4 on official GenEval2 again but with `--context_mode mdp` (pull first; same command as
  `depth4_t8_geneval2_part{1,2}` plus `--context_mode mdp` in EXTRA_ARGS; runs `depth4_t8_mdp_geneval2_part{1,2}`); on val800
  MDP did not degrade with depth (0.833 at 4 vs 0.795 full history), this checks it on the official prompts.
  **I**: base Tar-7B-v0.1 (no SFT/RL) on official GenEval2 + TIIF, single pass (draft only: base Tar has no critique), the
  paper's baseline. **J**: GenAI-bench setup (the user approved the third env + 54 GB judge), then t8 ck-500 on it.
  Afterwards: benchmark F (`darshan_mr_fir`) and G (`darshan_mr_sft_tamia`) checkpoints as they arrive (Fir/TamIA will copy).
- 2026-10-12 17:45 PDT from Fir, task D (remaining): t11 checkpoints 150/200/300 on the official benchmarks once Fir
  keeps them; the GenAI-bench setup. (t8 ck-500 benchmarks + depth done, Log 2026-10-13 00:40.)

## Log (newest first)
- 2026-10-13 20:40: backfilled `val/am_upto_{0..4}` and `val/allpass_upto_{0..4}` into the wandb summaries of
  `depth4_t8_geneval2_part1/2` (Fir inbox, user request), recomputed from each run's val_dump with `tree_stats`'
  definition @ `7d568ff`; checked `am_upto_0` = `am_0` and `am_upto_4` = `am_final` in both, and the 800-prompt mean
  matches `runs.md`. Pulled `7d568ff` (new metrics) before any further eval. Deleted both 2026-10-13 Fir inbox lines.
- 2026-10-13 00:40: t8 depth-4 on official GenEval2 done: AM 70.9 / 75.7 / 76.4 / 75.5 / 74.3 after 0–4 rounds; the
  critic stops on only ~22% of prompts by round 4. Told Fir and TamIA (Inbox).
- 2026-10-12 22:10: pulled Fir's `--context_mode` / `--chain_siblings` code (b0e0e34, a011791) while the depth jobs
  11222564/65 were queued (a rule-4 slip: the check printed the files but the pull went ahead). Checked the diff: with
  the defaults (`full`, `0`) and validation's explicit branch, rollouts are unchanged, so the jobs are unaffected.
- 2026-10-12 22:04: t8 ck-500 official benchmarks done: GenEval2 final AM 76.8 (+6.5 from refining), TIIF short/long final 70.5/68.6.
- 2026-10-12 21:00: pulled the A–G plan (`9793e0e` … `3e0c98f`); verified the t8 merged copy; queued task D.
- 2026-10-12 10:00: t7 Mila ck-100 benchmarks done: GenEval2 64.7 → 65.7, TIIF short 63.7 → 64.8, long 59.5 → 59.7.
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
