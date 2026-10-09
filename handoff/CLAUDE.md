# LatentDCR

Research project: **LatentDCR — Iterative Image Generation via Self-Critique and Refinement in Discrete
Visual Latents**. It is a unified MLLM, built on **Tar** (Qwen2.5 + text-aligned discrete visual tokens),
that generates an image as a draft → text critique → refined draft loop, entirely in discrete visual-latent
and text tokens, with no external verifier and no pixel decoding at inference.

## How the sessions coordinate (read first)
Claude sessions run on three clusters (Mila, Fir, TamIA) and share notes through this folder,
`tar_reason/handoff/` on branch `darshan-rl-test`. On each cluster, `~/scratch/git/CLAUDE.md` (or the Mila
equivalent) is a symlink to this file.

| File | Written by | Contents |
|---|---|---|
| `CLAUDE.md` | any session, small edits | Stable project facts and these rules. No job IDs or progress. |
| `runs.md` | any session, only its own rows | One row per run across all clusters, plus cross-run findings. |
| `fir.md`, `mila.md`, `tamia.md` | only that cluster's session | **Now** (rewritten, always current), **Setup**, **Inbox**, **Log** (newest first). |
| `archive/` | nobody | Frozen copies of the old `*_HANDOFF.md` files (2026-10-05 … 10-09). |

Rules:
1. `git pull` at session start, and before editing these files or submitting a job. Commit and push note
   changes promptly, with a message starting `handoff:`.
2. Rewrite your cluster's **Now** whenever the state changes, so it is always true. Move superseded details
   to **Log** as one dated line.
3. Edit only your own cluster file and your own rows in `runs.md`. To tell another session something, add a
   dated line to *its* **Inbox** (`- 2026-10-09 from Fir: ...`). The owner deletes the line once acted on,
   noting it in its Log.
4. **A pull can change code under queued jobs.** A queued or chained job runs whatever code is checked out when
   it *starts*. If jobs are queued, check `git log HEAD..origin/darshan-rl-test --stat` first; if anything outside
   `handoff/` changed, decide whether the queued jobs should get it before pulling. Never switch branches in a
   checkout that running or queued jobs use.
5. Work and push only on `darshan-rl-test`; never push to `main`. Commit with the user's identity (GitHub
   `Darshansingh11`). If this branch is ever merged into `main`, leave `handoff/` out.

## Project status
- SFT is done on Tar-7B (`slf_ref_edit_t20` @16K; newer `slf_ref_edit_t21` @17K is the RL starting point). It
  iterates on edit-style prompts but barely improves OOD (GenEval2, TIIF, GenAI-bench), and its single-pass
  quality is below base Tar. Diagnosed causes: the critique decision is keyed on caption style, and the refine
  step can't execute count, spatial, or text edits.
- RL (GRPO, `llava/train/rl/`) is underway; every run and its results are in `runs.md`. Short version: latent-reward
  RL improved drafting, not refinement; diagnostics say the refiner, not the critic, is the bottleneck; the
  collaborator's t7 recipe (pixel judge, oracle/hybrid critiques) is being reproduced on Mila and Fir.
- Next (from the paper plan): rebuild the SFT data (image-conditioned critiques for all caption families; refine
  pairs from the model's own drafts), run the diagnostic experiments, RL with reward `α·AM + (1−α)·GM`.
- No checkpoint has been run on the official GenEval2 / TIIF / GenAI-bench yet; in-training validation is
  GenEval2-style IID prompts (val256 / val800).

## Reference material
Read the notes first; open the PDFs only for exact numbers. In `~/scratch/git/papers/` (outside the repo):
- `publishing_plan_notes.md`: the paper plan, status, SFT root-cause analysis, fix plan (`paper_publishing_plan.pdf`).
- `tar_notes.md`: the Tar base model (`Tar_paper.pdf`, arXiv:2506.18898).
PDFs can't be rendered where poppler is missing; extract text with `uv run --no-project --with pypdf python -c ...`.

## Key conventions
- Chat template is Qwen2.5 `<|im_start|>` / `<|im_end|>`. Image tokens are `<im_start><S0><I…>…<im_end>`.
  Don't confuse the two pairs of tags.
- Iterative prompt prefix: `Generate an image iteratively by self-reflecting and correcting.`
- Critique format: `Self-reflect: …` then `Correction: …`. A correction of "looks good" ends generation.
  "no edit" / "no change" should also count as a stop signal.
- Main metrics: GenEval2-style AM and GM. Also report the all-atoms-correct rate, critic recall and
  specificity, and the fix rate of named atoms.
- Benchmarks: GenEval2, GenAI-bench, TIIF-bench (short/long). IID val sets are built in GenEval2 style.

## Data
- Use the leak-free GenEval2-style data: `geneval2_50K_v2/` — train 49,000 rows (md5 `317971cc1d9e…`) + val800
  (md5 `1d3feb3f9425…`), no prompt overlap. Same data as the collaborator's t7.
- The older `geneval2_50K/` train file (49,488 rows) contains 288 val800 prompts: never pair it with val800.

## Code
Repo `git@github.com:sarvghotra/tar_reason.git` (branches `main`, `orig_code`, `xiaofeng`, `darshan-rl-test`),
checked out as `tar_reason/` next to the cluster's `CLAUDE.md` symlink. The collaborator (`jeet` on Fir, GitHub
`sarvghotra`) pushes to `main`; their private paths (`/home/jeet/...`, `/scratch/jeet/...`) appear in their
scripts. On Fir, the user can read only `/scratch/jeet/tmp/` (via `/scratch/jeet/...`, not `/home/jeet/...`).

RL code: `llava/train/rl/` (`train_grpo.py` entry point, `rollout.py`, `reward.py`, `grpo.py`, `dataset.py`,
`oracle_critique.py`, `pixel_reward_server.py`). Launchers in `scripts/rl_ft/`: `bash.sh` (collaborator's),
`bash_mila.sh` / `bash_darshan.sh` (latent reward, Mila / Fir), `bash_mila_t7.sh` / `bash_fir_t7.sh` /
`bash_tamia_t7.sh` (t7 recipe). Line-by-line walkthrough of the RL code (commit `e535421`):
https://claude.ai/artifact/KPqjbeJ5iZMWsFuAyVmUHk. Regenerate after code changes with
`~/envs/tar/bin/python tools/walkthrough/build.py`, then republish `tools/walkthrough/out/tar_grpo_walkthrough.html`
to the same URL.

## Clusters (details in each cluster's file)
- **Mila** (user `singhsd`): A100 80GB (`a100l`) / H100; internet on compute nodes. See `mila.md`.
- **Fir** (Alliance, user `dars11`, 4× H100 80GB per node, account `rrg-bengioy-ad_gpu`): no internet on compute
  nodes. See `fir.md`.
- **TamIA** (Alliance, user `dars11`, H100 4/node, account `aip-agrawal`; paths have an extra `d/` level). See `tamia.md`.
- All: Python envs `~/envs/tar` (training: uv, Python 3.10, torch 2.1.2+cu121, transformers 4.50.0, flash-attn
  2.5.9.post1 prebuilt wheel, `setuptools<70`) and `~/envs/qwen_judge` (torch 2.6.0+cu124, transformers 4.57.6).
  RL needs 80 GB GPUs (peak ~42.5 GB/GPU). The user prefers uv over conda and is new to the Alliance clusters:
  explain cluster specifics briefly.

## Looking at RL results
- Metrics: wandb, entity `darshan-singh`, project `tar_reasoning`, run name = wandb run id = `RUN_NAME`. Key
  panels: `val/am_0` vs `val/am_final`, `val/improved_1`, `val/degraded_1`, `val/fix_1`, `val/looks_good_1`.
  Alliance clusters log offline: `wandb sync wandb/offline-run-*` from a login node, in the repo.
- Trajectories: each validation logs the first `--log_images` (8) val prompts as draft|refined strips with the
  critique in the caption; the same prompts every time. Training rollouts are not saved.
- Gallery of those strips (no GPU, seconds): from `tar_reason/`, run
  `python scripts/rl_ft/viz/trajectory_gallery.py wandb/run-<date>-<RUN_NAME>`. It writes
  `output_dir/<RUN_NAME>/gallery/index.html` + `img/`. To publish it as an artifact, copy the folder out of
  `output_dir` first. The first run's gallery: https://claude.ai/artifact/RH2zJjLUTSsTtt6PKw3GhR
