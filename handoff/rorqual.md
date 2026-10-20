# Rorqual

Owned by the Rorqual session. Rorqual: Alliance cluster, user `dars11`, 4× H100 80GB per node (like Fir).

## Now (2026-10-20)
- **`darshan_mr_f800pm_rorqual`** (Fir Inbox task, user's go): F's recipe (`bash_rorqual_mr.sh`: N=4 chain mode, MDP, hybrid critiques,
  children_r1, LR 5e-5) on promptmix train_17K from **F-800** merged, jeet's speed settings (GEN_BATCH_SIZE 48, REWARD_DECODE_BATCH 32,
  TRAIN_MICRO_BATCH 4, launcher defaults since `86d3c78`; checkout has the TA-Tok fix `395fcb4`), dataset seed from RUN_NAME,
  500 steps, evals every 100 on GenEval2 val800 (`val/`) + promptmix val800 (`val_promptmix/`), depth-4 mdp.
  Chain (submitted 2026-10-20 ~04:00): **merge 23887481** (CPU, `rrg-bengioy-ad_cpu`, → `models/tar/sft/darshan_mr_fir_merged800`)
  → **smoke 23887482** (`f800pm_smoke_rorqual`, 1 node, afterok) → **full 23887483** (4 nodes, afterok) → 23887484 → 23887485 (afterany).
  Yamls pre-created in `output_dir/{f800pm_smoke_rorqual,darshan_mr_f800pm_rorqual}/`. All jobs are named `darshan_mr_rorqual`
  (logs `output_dir/slurm_logs/darshan_mr_rorqual_<job>.*`).
  To do: report the smoke's per-step time (rollout + reward + train) to Fir's Inbox, then each eval (val800 + promptmix) to Fir's Inbox
  and my `runs.md` row. Pair: Fir's `darshan_mr_pm_fir`.
- The t8 smoke 23864852 was cancelled before it ran (replaced by the f800pm smoke).
- Accounts (user, `sacctmgr`): `rrg-bengioy-ad_gpu` (fair share 0.32 on 2026-10-19, same RAC as Fir), `def-bengioy_gpu`,
  `def-agrawal_gpu` (+ matching `_cpu`). MFA is mandatory for every ssh login (Fir cannot script into Rorqual without an open
  ControlMaster connection the user starts).

## Setup checklist (for the first Rorqual session; follow fir.md "Setup" as the model)
1. Notes folder `~/scratch/git/` with the repo `tar_reason/` on `darshan-rl-test` (git identity `Darshansingh11`, GitHub SSH key), and
   `~/scratch/git/CLAUDE.md` → symlink to `tar_reason/handoff/CLAUDE.md`. Record the login node + Claude session id here.
2. **Envs: first try the ready-made archive** copied from Fir (`~/scratch/stage/envs_tar_qwenjudge.tar.zst`, 4.4 GB: `envs/tar`,
   `envs/qwen_judge` + uv's base Python stored as `uvpy/cpython-3.10.22-linux-x86_64-gnu`; see how `bash_fir_mr.sh` unpacks it with
   STAGE_LOCAL). Home is `/home/dars11` on both clusters, so extracting `envs/` into `~` and `uvpy/<python>` into
   `~/.local/share/uv/python/` should give working envs; verify `~/envs/tar/bin/python -c 'import torch, flash_attn, transformers'` and the
   judge env on a GPU node. Fallback: envs with uv (`~/envs/tar`, `~/envs/qwen_judge`) from the freeze files (Fir: `~/scratch/tamia_transfer/{tar_env_freeze,
   qwen_judge_env_freeze}.txt`; TamIA built from the same files). flash-attn: prebuilt wheel 2.5.9.post1; `setuptools<70`.
   Then the node-local staging archive (`bash_fir_mr.sh` header has the command) if Rorqual's scratch shows SIGBUS like Fir's.
3. Weights (~95 GB) and data from Fir: **the user runs one `rsync --relative` from Rorqual** (one MFA prompt), into `~/scratch/`:
   Tar-7B, ta_tok, ar_dtok_lp_512px, vq_ds16_t2i, siglip config, sft/{slf_ref_edit_t21_ckpt17000, rl_ft_oracle_t8_merged500,
   rl_ft_promptsmix_t1_merged500}, vlm/Qwen3-VL-8B-Instruct, data/{geneval2_50K_v2, promptmix}, stage/ env archive, tamia_transfer/
   freeze files, tar_reason_kept/darshan_mr_fir/checkpoint-800. Expected layout: `~/scratch/models/tar/{Tar-7B, ta_tok.pth,
   ar_dtok_lp_512px.pth, vq_ds16_t2i.pt, siglip2-so400m-patch14-384/config.json, sft/<needed merged models>}`, `~/scratch/models/vlm/
   Qwen3-VL-8B-Instruct`, `~/scratch/data/{geneval2_50K_v2, promptmix}` (check md5s in CLAUDE.md / fir.md).
4. `wandb login` (entity darshan-singh); compute nodes probably have no internet → offline + sync from a login node.
5. Launcher `scripts/rl_ft/bash_rorqual_mr.sh` = copy of `bash_fir_mr.sh` with paths and `#SBATCH --account=rrg-bengioy-ad_gpu`;
   1-node smoke (`MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24`), then report to Fir's Inbox.

## Setup
- Login node `rorqual3`; tmux session `claude`; Claude session id `f80c7887-914b-4a2c-a958-99786c0b63a8`
  (`cd ~/scratch/git && claude --resume f80c7887-914b-4a2c-a958-99786c0b63a8`). Notes folder `~/scratch/git/` (= `/scratch/dars11/git`),
  repo `tar_reason/` on `darshan-rl-test`, identity `Darshansingh11`, GitHub key `~/.ssh/id_ed25519_github` works.
- Weights + data from Fir (one `rsync --relative`, 96.7 GB, finished 2026-10-20 ~03:15, no errors) at the paths in the checklist;
  md5s match: geneval2_50K_v2 train `317971cc…` / val800 `1d3feb3f…`, promptmix train_17K `5a7d8259…` / val_800 `90eb656a…`.
- Envs unpacked from `~/scratch/stage/envs_tar_qwenjudge.tar.zst` into `~/envs/{tar,qwen_judge}` + `~/.local/share/uv/python/
  cpython-3.10.22-linux-x86_64-gnu`. **The venvs point at uv's alias `cpython-3.10-linux-x86_64-gnu`, which the archive lacks: added it
  as a symlink** to the 3.10.22 dir. Imports OK on the login node (torch 2.1.2+cu121 / tf 4.50.0; judge torch 2.6.0+cu124 / tf 4.57.6);
  flash_attn/GPU checked by the smoke. Jobs stage the envs to `$SLURM_TMPDIR` themselves (STAGE_LOCAL=1, rewrites the link).
- Launcher `scripts/rl_ft/bash_rorqual_mr.sh` = `bash_fir_mr.sh` with job/run name `darshan_mr_rorqual` (same paths, account,
  `gpu:h100:4`, `module load cuda/12.2` exists here). Rorqual also has MIG slices (`h100_3g.40gb` …): always request `h100:4`.
- Compute-node internet: not checked yet; assume none (wandb offline, sync from the login node).

## Inbox

## Log (newest first)
- 2026-10-20 04:00: Inbox (Fir, F-800 → promptmix) handled: launcher regenerated from `bash_fir_mr.sh` @`86d3c78` (speed settings, REPO overridable); t8 smoke 23864852 cancelled; merge 23887481 → smoke 23887482 → full 23887483–85 submitted.
- 2026-10-20: setup (repo, rsync from Fir, envs, launcher, wandb login); smoke 23864852 submitted.
- 2026-10-19: file created by Fir (user decided to add Rorqual).
