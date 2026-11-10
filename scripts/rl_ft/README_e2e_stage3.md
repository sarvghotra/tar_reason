# Stage 3: end-to-end RL with a critique reward (runs A and B)

Written 2026-11-10 (Darshan) for Xiaofeng, who runs B on Fir. A runs on Fir under Darshan's account.

## Quick start (Xiaofeng, run B)

1. Clone the branch anywhere in your Fir scratch:
   `git clone -b darshan-rl-test git@github.com:sarvghotra/tar_reason.git tar_reason_e2e && cd tar_reason_e2e`
2. Optional smoke test (1 node, 2 steps, ~30 min):
   `SMOKE=1 CRITIQUE_REWARD=full bash scripts/rl_ft/submit_e2e_fir.sh`
3. The run (4 nodes, 3 chained 24 h jobs on `rrg-bengioy-ad_gpu`; it resumes on its own):
   `CRITIQUE_REWARD=full bash scripts/rl_ft/submit_e2e_fir.sh`
   Shorter jobs often start much sooner on Fir: `TIME=11:59:00 N_JOBS=5` (5 × 12 h) instead of the default 3 × 24 h.
   Check with `squeue -u $USER --start`.
4. Check step 0 (the first eval, before training starts). `val/am_upto_3` should be ≈ 0.826 and `val_promptmix/am_upto_3` ≈ 0.792
   (jeet's t4 numbers). If they are far off, stop and ping Darshan.

**Inputs.** No setup is needed. The models, data and packed Python envs are read from Darshan's scratch: your account
has read-only access. Outputs go to your clone's `output_dir/`.

**Settings.** Please don't change any. A (Darshan, stop-only reward) and B must differ only in `CRITIQUE_REWARD`.

**Logs and wandb.** Logs are in `output_dir/slurm_logs/`. wandb runs offline: see "How to run B" below for syncing,
and ask Darshan for an invite to `darshan-singh/tar_reasoning` so A and B sit side by side.

The rest of this file explains what the runs do, the details of running B, and which metrics to watch.

## What the runs do

The pipeline so far:

1. E-450 (RL on drafts and refinements).
2. Run 1 (refiner RL under oracle critiques).
3. Run 2: jeet's critique SFT `rl_ft_s2ce_pmix_oracle_t4`.

Stage 3 runs RL on everything from that model: drafts, the model's **own** critiques (including the decision to stop with
"looks good") and refinements. No oracle critiques are used in the rollouts.

| | Run A (Darshan) | Run B (Xiaofeng) |
|---|---|---|
| `CRITIQUE_REWARD` | `stop` | `full` |
| Image parent passes every check, critique says "looks good" | +1 | +1 |
| Parent passes every check, critique asks for edits | −1 | −1 |
| Parent fails some check, critique says "looks good" | −1 | −1 |
| Parent fails some check, critique asks for edits | +1 | recall(failed checks named) − 0.5 · false alarm(passed checks named) |

In both runs:

- **Normalisation.** The critique reward is normalised among the siblings of the same parent image.
- **Critique tokens.** They train with `adv_image + 0.25 · adv_critique`.
- **Image tokens.** They keep the usual GRPO advantage.
- **Which checks a critique names (B only).** Decided by the reward judge (Qwen3-VL-8B) in a text-only call, through
  `pixel_reward_server.py` `/critique_mentions`.
- **Code.** `llava/train/rl/critique_reward.py`, flags `--critique_reward`, `--critique_reward_coef`, `--critique_fa_beta`.
- **CPU test.** `util_scripts/debug/test_critique_reward.py`.

Everything else is run 1's recipe (`scripts/rl_ft/bash_fir_mr.sh`):

- **Rollouts.** Chain mode to depth 4, with 4 siblings at a random round per draft. MDP context.
- **Draft reward.** `children_r1`.
- **Optimisation.** Break weight 2, LR 5e-5 constant, fresh LoRA r64, KL 0.05 to the start model.
- **Data and length.** promptmix `train_17K`, 500 steps.
- **Critiques and stopping.** `CRITIQUE_SOURCE=policy`. `STOP_PENALTY=0`: the stop signal comes only from the critique
  reward.
- **Start model.** Base and KL reference are `models/tar/sft/darshan_E450_orc500_crisft_t4_merged`, i.e. run 1 step 500
  merged plus jeet's t4 adapter, merged.
- **Evals.** Depth-4 chains with own and oracle critiques, on GenEval2 val800 and promptmix val_800, at step 0 and
  every 100 steps.
- **Reproduction check.** Step 0 also checks that our code reproduces jeet's t4 numbers: val800 / promptmix own
  `am_final` at depth 3 = 0.826 / 0.792. Compare these with `val/am_upto_3` and `val_promptmix/am_upto_3`.

## How to run B on Fir (Xiaofeng)

Darshan's models, data and packed Python envs are readable to `xiaofeng` (read-only ACLs). The job unpacks the envs on
each node, so no environment setup is needed.

```bash
cd /scratch/xiaofeng     # or wherever you keep code
git clone -b darshan-rl-test git@github.com:sarvghotra/tar_reason.git tar_reason_e2e
cd tar_reason_e2e
SMOKE=1 CRITIQUE_REWARD=full bash scripts/rl_ft/submit_e2e_fir.sh   # optional: 1 node, 2 steps, ~30 min
CRITIQUE_REWARD=full bash scripts/rl_ft/submit_e2e_fir.sh           # the run: 4 nodes, 3 chained 24 h jobs
```

- **Run name and account.** The run name defaults to `xiaofeng_e2e_full_fir`. The account is `rrg-bengioy-ad_gpu`;
  override with `ACCOUNT=...`.
- **Outputs.** Logs, checkpoints (every 10 steps, all kept), val dumps and the offline wandb folder all go to
  `output_dir/` of your clone.
- **Resume.** Resubmit the same command; it resumes from the newest checkpoint.
- **Cancel.** `scancel --name=xiaofeng_e2e_full_fir`.
- **wandb.** Compute nodes are offline. From a login node, in the clone, sync each job's folder oldest first:

  ```bash
  wandb sync wandb/offline-run-*-<RUN_NAME>      # first piece
  wandb sync --append <later pieces>
  ```

  Use the same project as A (`tar_reasoning`) or your own.
- **Settings.** Don't change them without telling Darshan and jeet. A and B should differ only in `CRITIQUE_REWARD`.

## What to watch

| wandb key | Meaning |
|---|---|
| `train/crit_stop_acc` | Share of own critiques whose stop decision is right. Should rise. |
| `train/crit_recall`, `train/crit_false_alarm` | B only: on failing parents, how many failed checks the critique names, and how many passed checks it wrongly names. |
| `train/crit_reward`, `train/crit_abs_adv`, `train/crit_frac_zero_adv_groups` | Size of the critique signal. |
| `val/am_upto_d`, `val/allpass_upto_d` | val800, own critiques, after at most d rounds; the main curves. Also `val_promptmix/*`. |
| `val/looks_good_pass_k`, `val/looks_good_fail_k` | Stop rate on passing / failing images at round k. The first should rise and the second stay low, especially for k ≥ 2. |
| `val/am_0` | Draft score. Must not drop: the safeguard against the drift seen in Abl-1 and Abl-2. |
| `train/kl` | Should stay ~0.05–0.1. |

Never compare `val/delta_am_k` with `val_oracle/delta_am_k`: they average different sets of nodes.
