# Trillium-GPU

Owned by the Trillium session (to be opened by the user; until then the Fir session writes the Inbox). Trillium-GPU:
SciNet / Alliance, user `dars11`, login `trillium-gpu.scinet.utoronto.ca` (**SSH keys only**, uploaded through CCDB; no
passwords), H100 nodes (4 per node, 80 GB), account **`rrg-bengioy-ad`** (no `_gpu` suffix; also `def-bengioy`, `def-agrawal`).
Compute nodes have no internet; the login node `trig-login01` has internet and 4 H100s (short tests only).

## Now
- **2026-10-30 22:45 EDT: set up (Inbox steps 1–5), 1-node smoke `mr_smoke_trillium` = job 1203913 on `debug` (est. start
  22:57).** Smoke = `bash_trillium_mr.sh` with `PIXEL_JUDGE=gemma CRITIQUE_SOURCE=oracle`, E-450, promptmix train_17K.
- After the smoke: **idle until the user assigns the first training job** (candidates: seed twin of Fir's Gemma-reward run, or
  the run-2 replay-SFT ablations). Ask the user before launching anything that trains. The AlphaGRPO-30B run went to Nibi;
  `bash_trillium_alphagrpo.sh` (30B judge default) stays as an unrun port, `vlm/Qwen3-VL-30B-A3B-Instruct` is downloaded here too.
- Have here: envs `~/envs/{tar,qwen_judge,gemma_judge}` + archives `/scratch/dars11/stage/envs_tar_{qwen,gemma}judge.tar.zst`;
  models `Tar-7B, ta_tok, ar_dtok_lp_512px, vq_ds16_t2i, siglip2, vlm/{Qwen3-VL-8B-Instruct, gemma-4-12B-it}`,
  `sft/{darshan_pm1_E_merged450, slf_ref_edit_t21_ckpt17000}` (not `darshan_mr_E450_orc_merged500`); data `geneval2_50K_v2`
  (md5s match), `promptmix/`.

## Cluster facts (2026-10-30)
- **`~/scratch` was a plain directory on `/home`** (100 GiB quota, **read-only on compute nodes**). Everything now lives in the
  real scratch `$SCRATCH=/scratch/dars11` (25 TiB, 10M files) and `~/scratch -> /scratch/dars11` is a symlink, so Fir-style
  paths still resolve. Launchers use `/scratch/dars11/...` directly (sbatch rejects `-o` under `/home`). Transfers: send to
  `/scratch/dars11/...`. `/project/rrg-bengioy-ad/dars11` also exists. `/scratch/dars11/sage/` is another project: leave it.
- GPU nodes `trig[0002-0061]`: 4× H100 80GB, 96 CPUs, 755 GB RAM. Partitions are picked from the request: **never pass
  `-p compute` or `--mem`** (rejected); a 4-GPU job lands in `compute_full_node` (770 GB, 24 h max). `-p debug`: 187.5 GB,
  2 h max wall but **QOS `debug` caps a job at 180 GPU-minutes (4 GPUs → 45 min)**, 1 job/user. QOS `normal`: max 100 GPUs/job.
  Also: `compute_s4h` (1-GPU nodes), `compute_h200` (1 node), `compute_b200` (52 × 8 B200).
- `$SLURM_TMPDIR` = `/dev/shm/slurm.<user>.<job>` (tmpfs, 566 GB, RAM-backed: staged files count as job memory).
- Modules: none loaded by default; `module load StdEnv/2023 gcc/12.3 cuda/12.2`. Driver 580 (CUDA 13.0).
- Envs: uv in `~/.local/bin`, Pythons `cpython-3.10.22` / `3.12.15` in `~/.local/share/uv/python`, `UV_CACHE_DIR=~/scratch/.uv_cache`.
  `tar` from `fir_requirements.txt` (+ flash-attn wheel, `setuptools<70`), `qwen_judge` torch 2.6.0+cu124 / tf 4.57.6,
  `gemma_judge` torch 2.15.0+cu132 / tf 5.22.1 (works on the 13.0 driver: matmul checked). deepspeed import needs the cuda module.
- `vq_ds16_t2i.pt` is not in `csuhan/TA-Tok`: it comes from `peizesun/llamagen_t2i`.

## Setup
- Session: `e02f49ad-0196-4b16-b3af-aa1d35c8ca0d` on `trig-login01`, tmux `claude`
  (`cd ~/scratch/git && claude --resume e02f49ad-0196-4b16-b3af-aa1d35c8ca0d`).
- Repo: clone `git@github.com:sarvghotra/tar_reason.git` to `~/scratch/git/tar_reason`, branch `darshan-rl-test`; symlink
  `~/scratch/git/CLAUDE.md -> tar_reason/handoff/CLAUDE.md`.

## Inbox
- 2026-10-30 20:10 PDT from Fir (user: "launch the gemma reward run on trillium"): **run the Gemma-4-12B-reward RL run here.** Recipe = jeet's run 1 (`darshan_mr_E450_orc_fir`, see `fir.md` / `runs.md`) with only the judge changed: `bash_trillium_mr.sh` with `PIXEL_JUDGE=gemma CRITIQUE_SOURCE=oracle PREV_STAGE_CHECKPOINT=<E-450 merged, models/tar/sft/darshan_pm1_E_merged450> TRAIN_JSONL=<promptmix/train_17K.jsonl> MAX_STEPS=500 EVAL_STEPS=100 SAVE_STEPS=10`, 4 nodes × 4 H100 (3 train ranks + Gemma judge per node; judge flags `--answer_id_mode strict --image_soft_tokens 280`, as `bash_fir_mr.sh PIXEL_JUDGE=gemma` sets), `RUN_NAME=darshan_mr_E450_gemma_trillium`, `DATASET_SEED` left to the RUN_NAME default. **val.yaml = two sets like run 1**: `geneval2_50K_v2/evaluation_metadata_shuf_val800.jsonl` (logname `val`) + `promptmix/val_800.jsonl` (logname `promptmix`) — write it into `output_dir/<RUN_NAME>/val.yaml` before the first sbatch (run 1's file is quoted in `fir.md`; the launcher only creates a single-set one). Two chained 24 h jobs (`--dependency=afterany`; run 1 timed out at step 423 in 24 h on Fir). Before the full run: (1) your `mr_smoke_trillium` must have passed, (2) `util_scripts/debug/test_pixel_judge.py models/vlm/gemma-4-12B-it --answer_id_mode strict --image_soft_tokens 280` on one GPU prints sensible yes/no probabilities (red square: red ≫ blue). Fir's own copy of this run (`submit_gemma_run_fir.sh`, smoke 1475184 queued 5.5 h) is cancelled; Fir may later run a second seed. **Do not change any setting of the recipe without asking the user.** Report: step-0 (= E-450 depth-4) and every eval's `am_upto_0..4` (own + oracle, val800 + promptmix) in a `runs.md` row `darshan_mr_E450_gemma_trillium` and a line in Fir's Inbox; sync wandb pieces oldest-first (`--append` for later ones).
- 2026-10-30 from Fir (user): **set Trillium up as the second RL cluster** — DONE 2026-10-30 (see Now / Cluster facts); kept
  until the smoke passes. Steps:
  1. Cluster facts: `sinfo -o "%P %D %G %l"`, `sacctmgr show assoc user=$USER format=account`, `diskusage_report`, CPU
     count per GPU node, local disk (`$SLURM_TMPDIR`) size, whether `module load cuda/12.2` exists, MFA / key policy.
  2. Envs with uv (same recipe as the other clusters, `CLAUDE.md` → Clusters): `~/envs/tar` (Python 3.10, torch 2.1.2+cu121,
     transformers 4.50.0, flash-attn 2.5.9.post1 prebuilt wheel, `setuptools<70`), `~/envs/qwen_judge` (torch 2.6.0+cu124,
     transformers 4.57.6), `~/envs/gemma_judge` (Python 3.12, latest torch + transformers ≥ 5.22, accelerate, pillow,
     safetensors; for `google/gemma-4-12B-it` as the RL reward model, see `bash_fir_mr.sh PIXEL_JUDGE=gemma`).
  3. Public weights into `~/scratch/models`: `csuhan/Tar-7B-v0.1` → `models/tar/Tar-7B`, `csuhan/TA-Tok` (`ta_tok.pth`,
     `ar_dtok_lp_512px.pth`, `vq_ds16_t2i.pt`) → `models/tar/`, `google/siglip2-so400m-patch14-384` → `models/tar/siglip2-so400m-patch14-384`,
     `Qwen/Qwen3-VL-8B-Instruct` → `models/vlm/Qwen3-VL-8B-Instruct`, `google/gemma-4-12B-it` → `models/vlm/gemma-4-12B-it`
     (flat copy, `cp -rL` of the HF snapshot).
  4. Data: `geneval2_50K_v2/` (train 49,000 + val800) and `promptmix/` (`train_17K.jsonl`, `val_400.jsonl`, `val_800.jsonl`) are
     on Fir `~/scratch/data/`; our merged checkpoints (`models/tar/sft/{darshan_pm1_E_merged450, slf_ref_edit_t21_ckpt17000,
     darshan_mr_E450_orc_merged500}`) are on Fir / TamIA. **These are not public: Fir pushes them over ssh once the user's
     Alliance key (`~/.ssh/id_ed25519_alliance` on Fir, public key uploaded to CCDB) opens `ssh trillium-gpu`;** otherwise ask the
     user for a Globus transfer Fir → Trillium. Say in Fir's Inbox when the destination dirs are ready.
  5. Port the launcher: copy `scripts/rl_ft/bash_fir_mr.sh` to `bash_trillium_mr.sh` with Trillium paths, account, GPU gres and
     the staging-archive step (`STAGE_LOCAL`, see the header: the archive packs `envs/tar` + the judge env + uv's Pythons). Smoke it
     with `MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24 RUN_NAME=mr_smoke_trillium sbatch --nodes=1 --time=01:00:00 …`
     once the checkpoints are there. First real job (decided with the user): either the seed twin of Fir's Gemma-reward run
     (`PIXEL_JUDGE=gemma CRITIQUE_SOURCE=oracle PREV_STAGE_CHECKPOINT=…/darshan_pm1_E_merged450`, promptmix train_17K, 500 steps,
     `DATASET_SEED` different from Fir's) or the run-2 replay-SFT ablations (`--draft_loss_weight 0.5 / 2.0`, `bash_*_stop_sft.sh`).
  Rules as in `CLAUDE.md`: pull first, edit only this file and your own `runs.md` rows, commit `handoff: …` as Darshansingh11,
  push only `darshan-rl-test`.

## Log (newest first)
- 2026-10-30 20:10 PDT (Fir): user assigned the Gemma-reward RL run to Trillium (Inbox).
- 2026-10-30 22:50 EDT: Inbox: AlphaGRPO-30B run reassigned to Nibi (acted on: not started here, line removed).
- 2026-10-30 20:00 PDT (Fir): the AlphaGRPO-with-30B-reward task moved to Nibi; Trillium keeps the setup task only, first job to be assigned by the user.
- 2026-10-30 22:45 EDT: setup: envs + archives built, public weights downloaded, `~/scratch` moved to `/scratch/dars11` (home
  quota hit by a VLM rsync; the VLMs were already downloaded from HF), launchers `bash_trillium_{mr,alphagrpo}.sh`, smoke 1203913.
- 2026-10-30: file created by the Fir session (user's request: Trillium gets an Inbox and a session of its own).
