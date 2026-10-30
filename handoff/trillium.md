# Trillium-GPU

Owned by the Trillium session (to be opened by the user; until then the Fir session writes the Inbox). Trillium-GPU:
SciNet / Alliance, user `dars11`, login `trillium-gpu.scinet.utoronto.ca` (**SSH keys only**, uploaded through CCDB; no
passwords), H100 nodes (4 per node, 80 GB), account `rrg-bengioy-ad_gpu` (check with `sacctmgr show assoc user=$USER`).
Compute nodes are assumed to have no internet (Alliance norm): download on the login node.

## Now
- Not set up yet (2026-10-30). Fill in after the Inbox setup task: node layout, partitions, quotas, where things live.

## Setup
- Session: (record the Claude Code session id, login node and tmux name here)
- Repo: clone `git@github.com:sarvghotra/tar_reason.git` to `~/scratch/git/tar_reason`, branch `darshan-rl-test`; symlink
  `~/scratch/git/CLAUDE.md -> tar_reason/handoff/CLAUDE.md`.

## Inbox
- 2026-10-30 from Fir (user): **set Trillium up as the second RL cluster**, then report in **Now** and in Fir's Inbox. Steps:
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
- 2026-10-30: file created by the Fir session (user's request: Trillium gets an Inbox and a session of its own).
