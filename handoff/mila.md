# Mila

Owned by the Mila session. **Seeded from Fir on 2026-10-09** from the Fir copy of CLAUDE.md; the Mila session
should rewrite this file from its own notes (its `MILA_HANDOFF.md` and CLAUDE.md), then delete this line.

## Now (as last known on Fir, 2026-10-09 ~11:00)
- **Running:** `darshan_t7_mila` (see `runs.md`): chain of 12 jobs 11145380 … 11145391 (`afterany`),
  `MAX_STEPS=200 EVAL_STEPS=100` (evals at 0/100/200; extend to 300 later if promising, the LR is constant).
  ~23 steps per 3 h job. Was at step 95 on 2026-10-09 11:00.
- CPU job `keep_t7_ckpts` copies checkpoint-100/200 to `$SCRATCH/tar_reason/kept/darshan_t7_mila/`.
- If refinement improves but slowly: rerun with `PROMPTS_PER_GPU=12` (36/step, t7's batch, ~2× time per step).

## Setup
- User `singhsd`. Env `~/envs/tar`, `~/envs/qwen_judge`. Weights in `$SCRATCH/models/tar/` (Tar-7B, TA-Tok,
  de-tokenizers, `siglip2-so400m-patch14-384` config, `sft/slf_ref_edit_t21_ckpt17000`) and
  `$SCRATCH/models/vlm/Qwen3-VL-8B-Instruct`. Data `$SCRATCH/data/geneval2_50K_v2/`.
- `tar_reason/output_dir` is a symlink to `$SCRATCH/tar_reason/output_dir`; SLURM logs in its `slurm_logs/`.
- Partitions: `main` caps a user at 2 GPUs / 8 CPUs / 48 GB, so multi-GPU runs go on `long` (preemptible; the
  trainer resumes from the latest checkpoint). Use `a100l` or `h100` (80 GB).
- Launchers: `bash_mila.sh` (latent reward), `bash_mila_oracle.sh`, `bash_mila_t7.sh` (t7). Untracked
  `scripts/rl_ft/bash_t7_sarv.sh` = the collaborator's original t7 script, for reference.
- `--dataset_seed` was added on our branch; the collaborator's version is not pushed.

## Inbox
- 2026-10-09 from Fir: notes moved to `tar_reason/handoff/` (see CLAUDE.md "How the sessions coordinate").
  Merge your `MILA_HANDOFF.md` and any CLAUDE.md edits into `mila.md` / `runs.md` / `handoff/CLAUDE.md`, then
  replace your `~/scratch/git`-level `CLAUDE.md` with a symlink to `tar_reason/handoff/CLAUDE.md`.
- 2026-10-09 from Fir: `darshan_t7_fir` (same recipe and batch, 1 node of H100) is running on Fir, ~4.8 min/step.
  Compare against `darshan_t7_mila` in `runs.md`.

## Log (newest first)
- 2026-10-08: started `darshan_t7_mila`; cancelled `darshan_rl_oracle_b25_mila` at step ~54.
- 2026-10-07: `darshan_rl_2gpu_mila` finished (500 steps).
- 2026-10-06: latent-reward smoke test passed (1 A100: 42.5 GB peak, ~125 s/step); SigLIP2 path fix `39a6672`.
