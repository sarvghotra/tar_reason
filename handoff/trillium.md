# Trillium-GPU

Owned by the Trillium session (to be opened by the user; until then the Fir session writes the Inbox). Trillium-GPU:
SciNet / Alliance, user `dars11`, login `trillium-gpu.scinet.utoronto.ca` (**SSH keys only**, uploaded through CCDB; no
passwords), H100 nodes (4 per node, 80 GB), account **`rrg-bengioy-ad`** (no `_gpu` suffix; also `def-bengioy`, `def-agrawal`).
Compute nodes have no internet; the login node `trig-login01` has internet and 4 H100s (short tests only).

## Now
- **2026-10-31 15:20 EDT: `darshan_mr_E450_gemma_trillium` still queued: 1204109 (est. start slipped 11:36 → 2026-10-31 22:27 EDT; reason Priority, account fair-share 0.002) → 1204110
  (afterany).** Inbox recipe unchanged (`submit_gemma_run_trillium.sh`: `bash_trillium_mr.sh PIXEL_JUDGE=gemma CRITIQUE_SOURCE=oracle`,
  E-450, promptmix train_17K, 500 steps, evals every 100 on val800 + promptmix val_800, `DATASET_SEED=98381`). Run dir
  `output_dir/darshan_mr_E450_gemma_trillium/` (val.yaml written by hand: two sets). To report: step-0 and every eval's
  `am_upto_0..4` (own + oracle, both sets) in `runs.md` + Fir's Inbox; wandb sync oldest first (`--append` later).
- Setup done; smoke `mr_smoke_trillium` 1203913 passed on `debug` (26 min: staging 44 s, Gemma judge ready 30 s, 2 steps, 2 evals,
  checkpoint). `test_pixel_judge.py` on gemma-4-12B-it (login GPU) passed (yes/no + counts confident; open colour answer fails
  under strict ids, irrelevant for our yes/no + number data).
- Have here: envs `~/envs/{tar,qwen_judge,gemma_judge}` + archives `/scratch/dars11/stage/envs_tar_{qwen,gemma}judge.tar.zst`;
  models `Tar-7B, ta_tok, ar_dtok_lp_512px, vq_ds16_t2i, siglip2, vlm/{Qwen3-VL-8B-Instruct, gemma-4-12B-it, Qwen3-VL-30B-A3B-Instruct}`,
  `sft/{darshan_pm1_E_merged450, slf_ref_edit_t21_ckpt17000}` (not `darshan_mr_E450_orc_merged500`); data `geneval2_50K_v2`
  (md5s match), `promptmix/`. `bash_trillium_alphagrpo.sh` (30B judge) is an unrun port (that run went to Nibi).

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
(empty)

## Log (newest first)
- 2026-10-31 15:20 EDT: 1204109 not started (est. 11:36 → 22:27, low fair-share); pulled 25 commits, none touch the queued run's code.
- 2026-10-30 23:25 EDT: Inbox acted on and removed: setup task (smoke 1203913 passed) and "launch the gemma reward run" (1204109 → 1204110).
- 2026-10-30 20:10 PDT (Fir): user assigned the Gemma-reward RL run to Trillium (Inbox).
- 2026-10-30 22:50 EDT: Inbox: AlphaGRPO-30B run reassigned to Nibi (acted on: not started here, line removed).
- 2026-10-30 20:00 PDT (Fir): the AlphaGRPO-with-30B-reward task moved to Nibi; Trillium keeps the setup task only, first job to be assigned by the user.
- 2026-10-30 22:45 EDT: setup: envs + archives built, public weights downloaded, `~/scratch` moved to `/scratch/dars11` (home
  quota hit by a VLM rsync; the VLMs were already downloaded from HF), launchers `bash_trillium_{mr,alphagrpo}.sh`, smoke 1203913.
- 2026-10-30: file created by the Fir session (user's request: Trillium gets an Inbox and a session of its own).
