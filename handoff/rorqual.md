# Rorqual

Owned by the Rorqual session. Rorqual: Alliance cluster, user `dars11`, 4× H100 80GB per node (like Fir).

## Now (2026-10-25)
- **AlphaGRPO baseline `darshan_agrpo_tar_rorqual`** (Fir Inbox 2026-10-23, user + jeet, paper baseline; priority over R): base Tar-7B, fresh LoRA,
  `scripts/rl_ft/bash_rorqual_alphagrpo.sh` (= `bash_fir_alphagrpo.sh`, names changed), 380 steps, evals every 50 (val800 AM + alphagrpo val_400, draft only).
  **25175710 → 25175711** (afterany spare), submitted 2026-10-25 22:35, still pending 2026-10-26 11:40. **Rorqual power outage `POWEROUT20261027` 2026-10-27 05:00 → 11-03 04:00 (all nodes):** 25175710's limit cut 24 h → 12 h on 10-26 11:40 so it can start before then (it was blocked with ReqNodeNotAvail); **Backups: Fir 566575 → 566577 and TamIA also run AlphaGRPO; whichever starts first keeps it.** If 25175710 starts before 10-27 05:00, tell Fir's Inbox (they cancel theirs); if not, cancel 25175710/11 at 05:00 (don't let it start after 11-03 on top of the others). Fir's smoke passed (~65 s/step at 9 prompts; expect ~70–90 s/step at 36);
  say so if a step > ~6 min. Data `~/scratch/data/alphagrpo/` (train.jsonl md5 c70edf19…; `train_19.5K_ours.jsonl` 9a4a5587…, `val_400_ours.jsonl` 6037c1d5…).
  Copy each eval's checkpoint to `~/scratch/tar_reason_kept/darshan_agrpo_tar_rorqual/`; evals to Fir + the `runs.md` row.
  **After 380: merge onto Tar-7B (CPU, `merge_lora.py --base ~/scratch/models/tar/Tar-7B`)** → the user copies it to Mila for jeet's single-pass evals.
- **Next, after 11-03 (lower priority since Fir's oracle-only run 1 was nearly flat): `darshan_mr_E450_hyb_rorqual`** (Fir Inbox 2026-10-25; hybrid twin of Fir's run 1 `darshan_mr_E450_orc_fir`):
  `bash_rorqual_mr.sh` (re-synced 2026-10-25: CRITIQUE_SOURCE / HYBRID_ORACLE_FRAC overridable, default hybrid 0.5), `PREV_STAGE_CHECKPOINT=
  ~/scratch/models/tar/sft/darshan_pm1_E_merged450 TRAIN_JSONL=~/scratch/data/promptmix/train_17K.jsonl MAX_STEPS=500 EVAL_STEPS=100`.
  `output_dir/darshan_mr_E450_hyb_rorqual/val.yaml` pre-created (val800 + promptmix val_800). **Blocked on the user:** copy E-450 merged
  (16 GB) from Fir `scratch/models/tar/sft/darshan_pm1_E_merged450/` (ssh to Fir needs MFA). Keep every eval's checkpoint; evals to Fir + runs.md.
- R (`darshan_pm1_R_br48_rorqual`) dropped (Fir 2026-10-25: draft axis won the screen).
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

## Log (newest first)
- 2026-10-26 11:50: Inbox handled (Fir/user): AlphaGRPO backups on Fir (566575) and TamIA; Rorqual keeps 25175710 only if it starts before the outage, else cancel at 10-27 05:00. Hybrid twin waits for after 11-03.
- 2026-10-26 11:40: AlphaGRPO 25175710 still pending, blocked by the 10-27 → 11-03 power-outage reservation (24 h limit did not fit); limit cut to 12 h. E-450 merged not yet copied from Fir.
- 2026-10-25 23:45: Inbox handled (Fir): R dropped; hybrid twin `darshan_mr_E450_hyb_rorqual` queued as next after AlphaGRPO; `bash_rorqual_mr.sh` re-synced (CRITIQUE_SOURCE overridable), val.yaml pre-created; E-450 merged needs copying from Fir (user, MFA). AlphaGRPO 25175710 still pending (est. start 03:27).
- 2026-10-25 22:35: missed Fir's two AlphaGRPO Inbox lines (10-23) for 2.5 days (watchers died with the session). Handled now: R 25023367/68 cancelled (never started); data downloaded + converted; `bash_rorqual_alphagrpo.sh`; full run 25175710 → 25175711 (no smoke: Fir's passed).
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
