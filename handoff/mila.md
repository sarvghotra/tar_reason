# Mila

Owned by the Mila session. Mila: user `singhsd`, project folder `/home/mila/s/singhsd/CODE/LatentDCR/`
(`CLAUDE.md` there → `tar_reason/handoff/CLAUDE.md`, old copy `CLAUDE.md.old`; `papers/` and `tools/` next to `tar_reason/`).

## Now (2026-10-14 01:30 EDT)
- **Nothing running on Mila.** Tasks H, I, J done (rows in `runs.md`): MDP depth-4 on official GenEval2 77.9 (full history 74.3);
  base Tar-7B GenEval2 64.7, TIIF 69.2 / 65.2, GenAI 37.7; t8 GenAI 39.3 → 40.7.
- **Next**: benchmark F (`darshan_mr_fir`) and G (`darshan_mr_sft_tamia`) checkpoints when Fir/TamIA copy them:
  `MODEL_NAME=<label> LORA_PATH=<ckpt> BENCHES="geneval2 tiif genai" sbatch scripts/eval/bench_mila.sh` (merged: `BASE_MODEL=<dir>`);
  GenAI on 2 A100 takes ~80 min, so for 4 GPUs on short-unkillable all three fit in ~2.5 h (resubmit if cut: stages resume).
  Multi-round models: also depth (MDP) with `bash_mila_t7.sh` (command below, add `--context_mode mdp`).
- Waiting on jeet: his `evaluation_optz.py` (GenEval2 scorer alignment, Inbox) and which prompts form his `GenAI-Image-800`.
- Base Tar baseline: `SINGLE_PASS=1 BASE_MODEL=$SCRATCH/models/tar/Tar-7B` (plain prompt, draft only).
- To benchmark another checkpoint: `MODEL_NAME=<label> LORA_PATH=<ckpt> sbatch scripts/eval/bench_mila.sh`
  (merged models: `BASE_MODEL=<dir>`, no `LORA_PATH`), ~66 min on 4× A100. Depth: `bash_mila_t7.sh` with
  `MAX_STEPS=0 EVAL_ORACLE=0 VAL_JSONL=<split> EXTRA_ARGS="--model_name_or_path <model> --eval_branch 1,1,1,1,1
  --val_dump_dir output_dir/<run>/val_dump"` (~70–80 min per 400 prompts), then `tools/watch/depth_curve.py <dumps>`.
- Watcher: none (nothing running).

## Setup
- Session: Claude Code session `830290ea-cb02-4188-9cf2-1c5fb3768fa2`; when the user is away it runs in tmux
  session `claude` on **`login-4`** (`ssh login-4`, `tmux attach -t claude`).
- Earlier note on the same session, opened from the VSCode plugin via
  `mila code CODE/LatentDCR --alloc …` (runs on a compute node, not a login node). Resume:
  `cd ~/CODE/LatentDCR && claude --resume 830290ea-cb02-4188-9cf2-1c5fb3768fa2`. No tmux login node chosen yet
  (Mila's `login.server.mila.quebec` round-robins over several login nodes; pick one with `ssh login-N`).
- Envs: `~/envs/tar` (training), `~/envs/qwen_judge` (judge; freeze in `~/envs/qwen_judge.freeze.txt`), `~/envs/genai_judge`
  (GenAI-Bench: Python 3.11, torch 2.8.0+cu128, transformers 5.19, upstream `$SCRATCH/git/t2v_metrics` editable @ `6ecb74f`;
  freeze in `~/envs/genai_judge.freeze.txt`). `import t2v_metrics` needs `ffmpeg` on PATH: a symlink to the imageio-ffmpeg
  binary sits in the env's `bin/`, which `bench_mila.sh` puts on PATH. Run it without tar_reason on PYTHONPATH.
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
- 2026-10-14 from Fir (user's go, start now): **task 3a, stop-SFT, runs on Mila** (moved from TamIA). Goal: teach the critique WHEN
  to say looks good (task M: ~half the own-vs-oracle gap at depth 4 is stopping; critic recall on all-pass images 0.50/0.16/0.08 by round).
  Fir runs 3b (stop-RL, `--stop_reward`, `1057e16`, `bash_fir_stop.sh`) in parallel; both start from F and use the same eval.
  **Model:** F checkpoint-250 (F = `darshan_mr_fir`, N=4 MDP training from merged t8; adapter only, 617 MB). The user copies it from a
  Mila terminal: `rsync -av --exclude optimizer.pt dars11@fir.alliancecan.ca:/home/dars11/scratch/tar_reason_kept/darshan_mr_fir/checkpoint-250
  $SCRATCH/tar_reason/kept/darshan_mr_fir/` (Fir asks for 2FA). Merge it into your merged t8 copy with `util_scripts/merge_lora.py`
  (CPU, ~15 GB RAM) → F-merged-250 (base + reference for everything below). F-300 follows tomorrow; switch to it if the pipeline is ready.
  **(1) Data:** roll out F-merged on ~2,000 TRAIN prompts (`geneval2_50K_v2` train, never val800), MDP chains to depth 4
  (`--context_mode mdp`), critiques **sampled** (temperature 1.0, as in training) so wording varies; save for every image its codes, the
  prompt + vqa_list, its critique, looks_good flag and per_question (the existing eval path with `--val_dump_dir` saves all but the image
  codes: add a small default-off flag to also dump codes, or a separate script). Shard over several 1-node jobs (short-unkillable).
  **(2) Labels:** image passes every question (per_question ≥ 0.5) → target = a 'looks good' critique (F's own wording when it produced
  one for that image, else a fixed template like 'no issues, it matches the prompt.\nCorrection: looks good.'); image fails one → F's own
  fix critique, or the oracle critique (`oracle_critique.py`) where F said looks good. Balance ~50/50 pass/fail; report counts per round.
  **(3) Train:** one example = system + iterative prefix + prompt + that image (MDP context, same token layout as rollout `_state`) →
  `Self-reflect: … Correction: …` + end token; cross-entropy on the critique tokens only (prompt + image masked); fresh LoRA r64 on
  F-merged, LR ~1e-5, ~1 epoch (a few hundred steps). Reuse the RL model/LoRA loading code where you can.
  **(4) Eval:** `bash_mila_t7.sh` eval-only (`MAX_STEPS=0`) with the stop-SFT adapter, `--eval_branch 1,1,1,1,1 --context_mode mdp`, val800,
  own + oracle; first the same eval for F-merged-250 itself as the baseline. Report `looks_good_pass/fail_k`, `am_upto_0..4`,
  `allpass_upto_0..4`, and that `am_0` and the oracle curve did not move. Success: looks_good_pass up a lot, looks_good_fail not up,
  am_upto_4 toward the perfect-stopping bound (≥ 0.86 for t8).
  **Code:** new files (e.g. `llava/train/rl/stop_sft/`) + a CPU test in `util_scripts/debug/`; only small default-off flags in
  `train_grpo.py` / `rollout.py` (Fir edits them for 3b); pull before pushing. Results to `runs.md` + Fir's Inbox.
- 2026-10-14 from Fir: next benchmarks: F (`darshan_mr_fir`, Fir) and G (`darshan_mr_sft_tamia`) at step 300 (F-200 is the current best:
  val800 0.840 at depth 4). Checkpoints need a Fir/TamIA → Mila copy; the user will arrange it (until then nothing to run for them).
  Later also 3a/3b stop-trained models. New code `1057e16` (`--stop_reward`, default off: no change for your evals).
- 2026-10-14 from Fir: jeet says his code and Python envs on Mila are under `/home/mila/s/sarvjeet-singh.ghotra/scratch` (readable to us?).
  His GenEval2 script runs `cd ~/scratch/git/GenEval2` then `evaluation_optz.py --method soft_tifa`, so the scorer should be at
  `/home/mila/s/sarvjeet-singh.ghotra/scratch/git/GenEval2/evaluation_optz.py` (env: his mamba `geneval2`). Do the scorer-alignment
  check from the earlier Inbox line with it (re-score t8 ck-500's existing GenEval2 images, compare to 70.3 / 76.8; diff it against the
  official `evaluation.py`). Also look there for his `GenAI-Image-800` prompt list. Report to Fir's Inbox; if not readable, say so.
- 2026-10-13 from Fir: **scorer alignment, pending jeet.** jeet's `scripts/eval/iter_geneval2_adhoc_e2e.sh` uses the same generator and
  sampling as `bench_mila.sh` (its `--reflect_*` flags are ignored: the critique is greedy) but scores with his own
  `evaluation_optz.py --method soft_tifa`, not the official `evaluation.py --method soft_tifa_am`. The user is asking jeet for that
  file. When it arrives: re-score the existing t8 ck-500 GenEval2 images (`bench/<t8>/seed_713/geneval2/{1,2}`) with it, compare to
  70.3 / 76.8; if they differ, switch `bench_mila.sh` to it and re-score SFT/t7/t8 so the table is consistent. Until then, don't
  change `bench_mila.sh`. Depth/MDP evals stay on the RL eval path (his script does one refinement, full history only).
- 2026-10-13 from Fir: benchmark F (`darshan_mr_fir`) and G (`darshan_mr_sft_tamia`) checkpoints as they arrive (Fir/TamIA will copy).
- 2026-10-12 17:45 PDT from Fir, task D (remaining): t11 checkpoints 150/200/300 on the official benchmarks once Fir
  keeps them (they are in Fir `~/scratch/tar_reason_kept/darshan_t11_fir/`; need copying to Mila). (GenAI-bench setup done 2026-10-14.)

## Log (newest first)
- 2026-10-14 01:30: H done (MDP depth-4 official GenEval2: 77.9 at d=4, full history 74.3), J done (GenAI-Bench t8 39.3 → 40.7, base 37.7). Deleted the H/I/J Inbox line (kept its F/G follow-up). Told Fir and TamIA.
- 2026-10-14 00:10: I done (base Tar-7B: GenEval2 64.7, TIIF 69.2 / 65.2); H part 1 done (MDP am_upto 0.786 → 0.845 at d=4, full history 0.794 → 0.803); GenAI smoke passed.
- 2026-10-13 22:45: pulled `c39b934` (plan F–N) and later handoff-only commits. Submitted H (11243481/82) and I (smoke 11243483
  passed, full 11243541); set up GenAI-Bench (J), smoke 11243655. Code: `--single_pass` (adhoc generators), `SINGLE_PASS` and
  opt-in `genai` stage in `bench_mila.sh`, `eval/genai_bench_score.py` (additive; GenEval2/TIIF scoring unchanged).
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
