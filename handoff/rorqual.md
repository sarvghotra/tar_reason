# Rorqual

Owned by the Rorqual session (none yet). Rorqual: Alliance cluster, user `dars11`, 4× H100 80GB per node (like Fir).

## Now (2026-10-19)
- **Not set up yet.** Accounts (user, `sacctmgr`): `rrg-bengioy-ad_gpu` (fair share 0.32 on 2026-10-19, same RAC as Fir),
  `def-bengioy_gpu`, `def-agrawal_gpu` (+ `_cpu`). MFA is mandatory for every ssh login (Fir cannot script into Rorqual without an
  open ControlMaster connection the user starts).

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

## Inbox

## Log (newest first)
- 2026-10-19: file created by Fir (user decided to add Rorqual).
