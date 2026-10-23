# Rorqual

Owned by the Rorqual session. Rorqual: Alliance cluster, user `dars11`, 4× H100 80GB per node (like Fir).

## Now (2026-10-23)
- **Variant R `darshan_pm1_R_br48_rorqual`** (Fir Inbox 2026-10-23): B's recipe with **BRANCH 4,8** (4 drafts × 8 refines; tests the refine axis,
  E on Fir tested drafts 8,4), from SFT t21, `MAX_STEPS=200 EVAL_STEPS=50`, 4 nodes: **25023367 → 25023368** (afterany spare), submitted 11:00.
  Expect ~1.6× B's rollout; if a step > ~5 min or memory is tight, resubmit with PROMPTS_PER_GPU=2. Copy each eval's checkpoint to
  `~/scratch/tar_reason_kept/darshan_pm1_R_br48_rorqual/`; evals 0–200 to Fir + my `runs.md` row.
- Done 2026-10-23: B-450 / B-500 merged onto SFT t21 (`~/scratch/models/tar/sft/darshan_pm1_B_merged450`, `_merged500`; **merged450 waits for
  the user to copy it to Mila**); depth-4 MDP eval of B-450: no own-critique gain past round 1, oracle +0.011 / +0.024 (critic-limited).
- B `darshan_pm1_B_br44_rorqual` done at 500 (2026-10-23 07:10): val800 own 0.783, promptmix 0.740 (t1 at 500: 0.751 / 0.722); best step 450
  (0.787 / 0.744). Checkpoints 50–500 in `~/scratch/tar_reason_kept/darshan_pm1_B_br44_rorqual/`, wandb synced; details in `runs.md`.
  Launcher `scripts/rl_ft/bash_rorqual_promptmix.sh` (copy of `bash_fir_promptmix.sh`); jobs named `darshan_pm_rorqual`.
  **Copy each eval's checkpoint to `~/scratch/tar_reason_kept/<run>/` right after the eval: `--save_total_limit 5` deletes it 50 steps later.**
- Done: `darshan_mr_f800pm_rorqual` (F-800 → promptmix, 500 steps, 2026-10-21): final val800 0.849 (−0.009), promptmix 0.730 (+0.018);
  details in `runs.md`; checkpoint-500 in `~/scratch/tar_reason_kept/darshan_mr_f800pm_rorqual/`; wandb synced.
- `bash_rorqual_mr.sh` re-synced with `bash_fir_mr.sh` (jeet's settings: REWARD_QUESTION_BATCH 64, TRAIN_MICRO_BATCH 2; SIGLIP2_PATH exported).
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
- 2026-10-23 from Fir (user + jeet, paper baseline; **after R, or instead of it if R has not started: this has priority**): **AlphaGRPO on base Tar-7B**, 380 steps, 4 nodes. (1) `git pull` (code `d07e3a0`: `--prompt_style plain`, `--reward_combine geomean`, `quality_list` rows). (2) Data: on a login node, `mkdir -p ~/scratch/data/alphagrpo && cd ~/scratch/data/alphagrpo && for f in train test; do curl -sL -o $f.jsonl https://media.githubusercontent.com/media/huangrh99/AlphaGRPO/main/alpha_grpo/dataset/alphagrpo20k/$f.jsonl; done` (144 MB + 7.6 MB; Fir's train.jsonl md5 starts c70edf19a8dd), then from the repo `python util_scripts/convert_alphagrpo.py ~/scratch/data/alphagrpo/train.jsonl ~/scratch/data/alphagrpo/train_19.5K_ours.jsonl` and `... test.jsonl val_400_ours.jsonl --limit 400` (19,500 rows, ~14 semantic + ~21 quality questions each). (3) Port `scripts/rl_ft/bash_fir_alphagrpo.sh` to `bash_rorqual_alphagrpo.sh` (paths/account as your other launchers; base = `models/tar/Tar-7B`). Smoke first (1 node, MAX_STEPS=2, EVAL_MAX_PROMPTS=24) unless Fir's smoke result is in your Inbox by then, then `RUN_NAME=darshan_agrpo_tar_rorqual` with an afterany spare; evals every 50 (val800 AM + alphagrpo val_400, draft only); copy each eval's checkpoint. Judge load is ~5× a promptmix step (14 images × 35 questions per prompt): if a step exceeds ~6 min, say so (we can drop PROMPTS_PER_GPU to 2). Report evals to Fir's Inbox + a `runs.md` row (placeholder row exists; make it yours). After 380: merge onto Tar-7B (CPU) → the user copies to Mila for jeet's single-pass evals.

## Log (newest first)
- 2026-10-23 11:00: Inbox (Fir: variant R, BRANCH 4,8) handled: 25023367 → 25023368 submitted.
- 2026-10-23 09:25: depth-4 eval of B-450 done (own: no gain past round 1; oracle +0.011 / +0.024); wandb synced; sent to Fir. Idle.
- 2026-10-23 07:52: merges of B-450 / B-500 onto SFT t21 done (`darshan_pm1_B_merged450`, `_merged500`); depth-4 eval 25011837 pending.
- 2026-10-23 07:50: Inbox (Fir: merge B-450/500, depth-4 eval of B-450) handled: merges 25011835/36, eval 25011837 submitted.
- 2026-10-23 07:15: B done at 500: val800 0.783, promptmix 0.740 (> t1 by +0.032 / +0.018); ckpts 50–500 kept; wandb synced; spare cancelled; sent to Fir.
- 2026-10-23 05:20: B step 450: val800 own 0.787, promptmix 0.744 (best so far); ckpt-450 kept; sent to Fir.
- 2026-10-23 03:30: B step 400: val800 own 0.760, promptmix 0.734 (refinement +0.037); KL spikes to 0.70 at 395 (recovered); ckpt-400 kept; sent to Fir.
- 2026-10-23 01:40: B step 350: val800 own 0.766, promptmix 0.727; ckpt-350 kept; sent to Fir.
- 2026-10-22 23:45: B step 300: val800 own 0.762, promptmix 0.736; refinement positive; ckpt-300 kept; sent to Fir.
- 2026-10-22 21:55: B step 250: val800 own 0.760, promptmix 0.732 (> t1@500); ckpt-250 kept; sent to Fir.
- 2026-10-22 19:00: Inbox (Fir: extend B to 500) handled: 24981206 → 24981207 submitted (resume from 200); critic-limit note added to runs.md.
- 2026-10-22 18:40: B done: step 200 val800 0.735, promptmix 0.684 (beats A); ckpts 50–200 kept, wandb synced, spare 24904809 cancelled; sent to Fir.
- 2026-10-22 16:45: B step 150: val800 own 0.714, promptmix 0.686, oracle pm 0.715; ckpt-150 kept; sent to Fir.
- 2026-10-22 14:55: B step 100: val800 own 0.722 (+0.053), promptmix 0.668 (+0.064), mostly draft; ckpt-100 kept; sent to Fir.
- 2026-10-22 13:05: B step 50: val800 own 0.695 (+0.026), promptmix 0.638 (+0.034), oracle pm 0.667; 2 min/step; ckpt-50 kept; sent to Fir.
- 2026-10-22 11:10: full B 24904808 up (10:52, staging 107 s); step-0 eval (SFT t21) val800 own 0.669, promptmix 0.604; sent to Fir.
- 2026-10-22 03:45: B smoke passed (≈ 2 min/step, PROMPTS_PER_GPU stays 3); full B 24904808 → 24904809 submitted.
- 2026-10-22 02:50: Inbox handled: f800pm had finished 500 steps on 2026-10-21 20:21 (23887485, resumed at 350); spare 24413404 cancelled; wandb synced (2 pieces); step 300/400/500 evals to Fir + runs.md; only checkpoint-500 survived (save_total_limit 5). `bash_rorqual_mr.sh` re-synced (Fir's 2026-10-20 merge Inbox: jeet's settings, SIGLIP2_PATH). Variant B: `bash_rorqual_promptmix.sh`, smoke 24902055.
- 2026-10-21 09:05: f800pm job 1 (23887484) TIMEOUT at ~350; 23887485 resumed (11:07).
- 2026-10-20 23:10: step-200 eval: val800 own 0.845 (−0.013), promptmix 0.694 (−0.018), drafts −0.04; flagged to Fir (continue or stop?).
- 2026-10-20 19:10: step-100 eval: val800 own 0.850 (−0.007), promptmix own 0.712 (=), all-correct 18.3% (+2.3); drafts down on both; KL 0.16. Sent to Fir. (Session restarted; watchers restarted.)
- 2026-10-20 10:10: full run up (23887484); step-0 eval (= F-800) val800 own 0.858 / oracle 0.905, promptmix own 0.712 / oracle 0.766; 3.7 min/step; sent to Fir's Inbox.
- 2026-10-20 09:05: full job 23887483 cancelled by Slurm at start (CANCELLED by 0, signal 54, no logs); 23887484 takes over, spare 24413404 chained.
- 2026-10-20 05:15: smoke 23887482 passed (both val sets, ckpt-2, staging 79 s); per-step time sent to Fir's Inbox; smoke wandb synced. Full run 23887483 pending (Priority).
- 2026-10-20 04:16: F-800 merge 23887481 done (`darshan_mr_fir_merged800`, 12 GB, LoRA r64 on all 7 proj); reading the base from scratch took 27 min.
- 2026-10-20 04:00: Inbox (Fir, F-800 → promptmix) handled: launcher regenerated from `bash_fir_mr.sh` @`86d3c78` (speed settings, REPO overridable); t8 smoke 23864852 cancelled; merge 23887481 → smoke 23887482 → full 23887483–85 submitted.
- 2026-10-20: setup (repo, rsync from Fir, envs, launcher, wandb login); smoke 23864852 submitted.
- 2026-10-19: file created by Fir (user decided to add Rorqual).
