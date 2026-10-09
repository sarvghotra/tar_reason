# Fir handoff: run the collaborator's t7 recipe on Fir H100s (2026-10-09)

Context for Claude on Fir, written from the Mila session. Read `CLAUDE.md` first; it has the project, the Mila
runs and their results. The user (`dars11` on Fir, GitHub `Darshansingh11`) now has GPU access on Fir under
`def-agrawal_gpu`. Explain Fir specifics briefly when they come up; the user prefers uv over conda.

## Goal
Run `scripts/rl_ft/bash_fir_t7.sh` on branch `darshan-rl-test`: the collaborator's `rl_ft_oracle_t7` on
**2 nodes × 4 H100** (per node: GPUs 0–2 train, GPU 3 runs a Qwen3-VL-8B pixel judge), i.e. 6 training ranks ×
6 prompts = **36 prompts/step**, t7's batch. The Mila port (`bash_mila_t7.sh`, 1 node, 18 prompts/step) is
running in parallel; this run tests the same recipe at t7's batch size and ~2× the speed.
Trainer flags are identical to `bash_mila_t7.sh` (pixel judge, hybrid critiques, `BRANCH=2,4`, LR 1e-4 constant,
KL 0.05, 2 PPO epochs, `draft_reward children`, stop penalty 0.5@0.8, 300 steps, val800 every 50 steps).
`bash_fir_t7.sh` was written on Mila and has **not been run yet**; the multi-node `srun` + `torchrun` part is new.

Reference results on Mila (same recipe, 18 prompts/step), for sanity checks:
- step 0, val800: `val/` final−draft −0.010, fix 12.0%; `val_oracle/` final−draft +0.009, fix 14.9%, brk 7.4%
  (the collaborator's t7 notes: oracle fix 15.3%, brk 7.0%).
- training: `train/fix_1` 13.7% (steps 1–12) → 22.5% (steps 73–95), `train/fix_oracle_1` 12% → 26%,
  `train/reward_1` +0.01…+0.03, ~7.3 min/step on A100. One-step KL spikes (1.2 at step 14, 3.5 at step 55,
  grad norm 87, clipped to 1) that recovered the next step.

## Already on Fir (from 2026-10-05/06)
- Repo at `~/scratch/git/tar_reason`; GitHub SSH key works (known_hosts has github.com).
- Training env `~/envs/tar` (uv, Python 3.10, torch 2.1.2+cu121, transformers 4.50.0, flash-attn 2.5.9.post1
  wheel, `setuptools<70`). Imports verified; CUDA never tested on a Fir GPU.
- `~/scratch/models/tar/`: `Tar-7B/`, `ta_tok.pth`, `ar_dtok_lp_512px.pth`, `vq_ds16_t2i.pt`,
  `sft/slf_ref_edit_t21_ckpt17000` (and t20 / t6 SFT checkpoints).
- `~/scratch/data/geneval2_50K/`: the OLD 49,488-row train file. It contains 288 val800 prompts: do not use it
  with val800. Use the leak-free copy below.
- wandb is logged in (entity `darshan-singh`).

## Steps (1–4 on a login node, inside tmux: compute nodes have no internet)
1. Code: `cd ~/scratch/git/tar_reason && git fetch origin && git checkout darshan-rl-test && git pull`.
   Expect `scripts/rl_ft/bash_fir_t7.sh` (commit `90c1ac9` or later). There is a local untracked
   `scripts/rl_ft/bash_t7_sarv.sh` on Mila only; it is the collaborator's original, for reference.
2. Judge env (separate from `~/envs/tar`; Qwen3-VL needs transformers 4.57):
   ```
   export UV_CACHE_DIR=$HOME/scratch/.uv_cache UV_LINK_MODE=copy
   uv venv ~/envs/qwen_judge --python 3.10
   uv pip install --python ~/envs/qwen_judge/bin/python --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
   uv pip install --python ~/envs/qwen_judge/bin/python transformers==4.57.6 accelerate==1.13.0 huggingface_hub==0.36.2 pillow==11.2.1
   ~/envs/qwen_judge/bin/python -c "import torch, transformers; from transformers import Qwen3VLForConditionalGeneration; print(torch.__version__, transformers.__version__)"
   ```
   Expect `2.6.0+cu124 4.57.6`. (Same recipe as Mila and as Xiaofeng's `scripts/prepare_qwen_reward.sh`.)
3. Weights:
   ```
   uvx --from huggingface_hub hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ~/scratch/models/vlm/Qwen3-VL-8B-Instruct
   uvx --from huggingface_hub hf download google/siglip2-so400m-patch14-384 config.json --local-dir ~/scratch/models/tar/siglip2-so400m-patch14-384
   ```
   Expect 17 GB and 4 `model-*.safetensors` shards for Qwen3-VL. `tok/ta_tok.py` reads the SigLIP2 config via
   `SIGLIP2_PATH` (the script exports it); without it TA-Tok tries `/scratch/jeet/...` and fails.
4. Leak-free data (the collaborator's updated copy; dars11 can read `/scratch/jeet/...` but not `/home/jeet/...`):
   ```
   mkdir -p ~/scratch/data/geneval2_50K_v2
   cp /scratch/jeet/tmp/data/T2I_datasets/geneval2_50K/evaluation_metadata_shuf_{train,val800}.jsonl ~/scratch/data/geneval2_50K_v2/
   wc -l ~/scratch/data/geneval2_50K_v2/*.jsonl
   ```
   Expect 49,000 train rows and 800 val rows, no prompt overlap. On Mila the md5s are train `317971cc1d9e…`,
   val800 `1d3feb3f9425…`; check with `md5sum` that Fir's copy matches.
5. Smoke test, 1 node (~30 min):
   ```
   MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24 RUN_NAME=t7_smoke_fir \
     sbatch --nodes=1 --time=01:00:00 scripts/rl_ft/bash_fir_t7.sh
   ```
   Log: `output_dir/slurm_logs/t7_smoke_fir_<jobid>.out/.err`. Pass = `judge ready after …s`, `step=0`
   (val/ and val_oracle/ lines), `step=1`, `step=2`, `Saved output_dir/t7_smoke_fir/checkpoint-2`, `Done.`
   The judge's own log is `output_dir/t7_smoke_fir/reward_server_<jobid>_n0.log`.
   On Mila the same smoke test took 29 min: judge load 50 s, ~7.6 min/step.
6. Full run, 2 nodes: `sbatch scripts/rl_ft/bash_fir_t7.sh` (RUN_NAME `darshan_t7_fir`, 23:59 limit).
   Expect ~4 min/step on H100 (the collaborator's estimate: ~20 h of steps + ~3 h of evals for 300 steps), so
   1–2 jobs. Queue a follow-up in advance with
   `jid=$(sbatch --parsable scripts/rl_ft/bash_fir_t7.sh); sbatch --dependency=afterany:$jid scripts/rl_ft/bash_fir_t7.sh`;
   a job that finds checkpoint ≥ MAX_STEPS exits at once. If 2 nodes queue too long, `sbatch --nodes=1 …`
   runs the same recipe at 18 prompts/step. First things to check once running: both nodes print
   `judge ready`, rank 0 prints `train prompts: 49000 (per rank/epoch 8166 or 8167), val prompts per rank:
   133 or 134`, and `step=0` appears.
7. wandb is offline on compute nodes. Upload from a login node, in the repo: `wandb sync wandb/offline-run-*`.

## Result 2026-10-09: 1-node smoke test PASSED (job 63829085, `rrg-bengioy-ad_gpu`, node fc10106)
Steps 1–4 done (judge env `~/envs/qwen_judge` torch 2.6.0+cu124 / transformers 4.57.6; Qwen3-VL-8B 17 GB;
SigLIP2 config; leak-free data md5s match Mila, no train/val800 overlap). `output_dir/slurm_logs/` had to be
created before `sbatch`. Queued 75 min, ran 38 min: judge ready after 260 s, policy shards load 5.5 min (slow
Fir scratch), step 0 / 1 / 2, `checkpoint-2` (1.9 GB), `Done.`; no errors. Per step: rollout ~170 s, reward
~81 s, train ~35 s = **~4.8 min/step** (Mila A100: ~7.6). KL 0.015, grad norm 0.1.
Projection for the 2-node run: 300 × 4.8 min ≈ 24 h of steps + val800 evals + ~10 min startup per job → 2 jobs.

## Things to watch / known pitfalls
- Multi-node: `torchrun` must not use `--standalone` (it would confine the job to one node); the script uses
  `--nnodes/--node_rank/--master_addr` from SLURM. Each node runs its own judge on the same port because the
  trainer reaches it at 127.0.0.1. If node 1 fails with "reward server unreachable", its judge did not start.
- The judge is started before `module load cuda/12.2` so its torch 2.6 finds its own CUDA 12.4 libraries (the
  collaborator hit `undefined symbol: __nvJitLinkComplete_12_4` when the module came first).
- Only the 5 newest checkpoints are kept (`--save_total_limit 5`, saves every 5 steps). Copy any checkpoint
  needed for later benchmarking (e.g. 100, 200, 300) out of `output_dir/darshan_t7_fir/` before it is pruned.
- A validation interrupted by a time limit is redone by the next job (`eval_done_<step>` markers), logged to
  wandb one step later.
- Don't switch git branches in the checkout a running job uses. Work and push only on `darshan-rl-test`;
  never push to `main`. Commit with the user's git identity.
- tmux sessions live on one login node; a new SSH session may land on another node.

## Earlier history (2026-10-05/06), kept for reference

Context for Claude on Fir, carried over from a Mila session. Read `CLAUDE.md` first.

## Goal
Run the collaborator's GRPO RL script `tar_reason/scripts/rl_ft/bash.sh` as a **test run** under the user's own
account (`dars11`). Results don't matter yet; the aim is to show it loads, rolls out, scores, steps and saves.

## What the script does
Launches `llava/train/rl/train_grpo.py` with torchrun on 1 node × 4 H100. Per step and per GPU: 2 GenEval2 prompts
× 4 drafts, each draft gets a critique (≤128 tokens) and 2 refined children (`BRANCH="4,2"`). The reward is a
frozen Tar-7B answering the prompt's VQA questions in latent space; score = `ALPHA·AM + (1−ALPHA)·GM`, with
`ALPHA=1.0` (AM only). Refinement reward = child score − parent score, with a penalty for stopping ("looks good")
too early. LoRA-only update (r=64, LR 1e-5, KL 0.01). 500 steps, eval and save every 25 steps.
The comment in the script says the partition kills jobs after 3 h and training auto-resumes from the latest
`checkpoint-N`, so resubmit (or chain with `--dependency=afterany`) with the same `WANDB_RUN_ID`.

## Status (updated 2026-10-06): paused, user moved experiments to Mila
Everything for the latent-reward smoke test is in place on Fir; no GPU job has run yet (queued requests under
`def-agrawal_gpu` were cancelled before starting). Plan: debug on Mila, come back here to submit the real run.

Done:
- Env `~/envs/tar` verified (imports OK): torch 2.1.2+cu121, transformers 4.50.0, peft 0.18.1, wandb 0.26.0,
  flash-attn 2.5.9.post1 from the prebuilt GitHub wheel (`...cu122torch2.1cxx11abiFALSE-cp310...whl`);
  needs `setuptools<70` (torch 2.1 imports `pkg_resources`). CUDA not yet tested on a GPU.
- `~/scratch/models/tar/`: `Tar-7B/` (HF `csuhan/Tar-7B-v0.1`, identical to jeet's), `ta_tok.pth`,
  `ar_dtok_lp_512px.pth`, `vq_ds16_t2i.pt`, and `sft/` with `slf_ref_edit_t21_ckpt17000` (used),
  `slf_ref_edit_t20_ckpt16000`, `slf_ref_edit_t6_repro_w_corr_t2_ckpt20000` (weights only, 17G each).
- `~/scratch/data/geneval2_50K/`: train (49,488) + val256 jsonl. Copied from `/scratch/jeet/tmp`
  (access via `/scratch/jeet/...`, not `/home/jeet/...`).
- wandb logged in, entity `darshan-singh`, offline by default in the script.
- Branch `darshan-rl-test` (not main; main has a revert of the script): jeet's latest code (`ff96ea9`) +
  `scripts/rl_ft/bash_darshan.sh` (t21 ckpt, `def-agrawal_gpu`, overridable `N_GPUS`, `MAX_STEPS`,
  `EVAL_STEPS`, `SAVE_STEPS`, `EVAL_MAX_PROMPTS`). Run dir with data/val yaml: `output_dir/darshan_rl_test/`.
- dars11 is NOT in `rrg-bengioy-ad` (only `def-agrawal_{cpu,gpu}`); membership needs a CCDB role request.

Smoke test command (1 GPU):
`N_GPUS=1 MAX_STEPS=3 EVAL_STEPS=3 SAVE_STEPS=3 EVAL_MAX_PROMPTS=16 sbatch --gres=gpu:h100:1 --cpus-per-task=8 --mem=128G --time=1:00:00 scripts/rl_ft/bash_darshan.sh`

Still needed for jeet's t7 recipe (pixel reward, oracle/hybrid critiques; script pasted in chat, not in repo):
t7 `data.yaml`/`val.yaml` + 800-prompt val jsonl, `pip freeze` of his `geneval2` judge env,
`Qwen/Qwen3-VL-8B-Instruct` weights.

## User preferences
- New to Fir; explain cluster specifics briefly when they come up.
- Prefers uv over conda.

## Update 2026-10-06 (from Mila): smoke test passes; one fix needed on Fir
1-GPU smoke test on Mila (A100 80GB, `bash_mila.sh`, 3 steps + eval + save) passed: peak GPU memory 42.5 GB,
~125 s/step (rollout ~102 s, reward ~9 s, train ~15 s), checkpoint-3 = 1.9 GB (LoRA adapter + optimizer).
Fix pushed to `darshan-rl-test` (`39a6672`): `tok/ta_tok.py` hardcoded `/scratch/jeet/.../siglip2-so400m-patch14-384`
(only its config.json is needed). It now reads `SIGLIP2_PATH`. Before running on Fir:
1. `git pull` on `darshan-rl-test`.
2. On a login node: `uvx --from huggingface_hub hf download google/siglip2-so400m-patch14-384 config.json --local-dir ~/scratch/models/tar/siglip2-so400m-patch14-384`
3. Add `export SIGLIP2_PATH=/home/dars11/scratch/models/tar/siglip2-so400m-patch14-384` to `bash_darshan.sh`
   (next to `ENCODER=`), then commit and push.

## Full run submitted 2026-10-09 (1 node, by the user's choice; no 2-node run)
`RUN_NAME=darshan_t7_fir`, `sbatch --nodes=1 scripts/rl_ft/bash_fir_t7.sh` under `rrg-bengioy-ad_gpu`: 3 train
ranks × 6 = 18 prompts/step (same batch as `darshan_t7_mila`), 300 steps, val800 every 50. Jobs 63858143 →
63858144 → 63858145 (`afterany` chain; a job that finds checkpoint-300 exits at once). Estimate ~24 h of steps
+ ~1 h per val800 eval on 3 H100 ranks (7 evals) + ~10 min startup per job ≈ 2 jobs. Logs:
`output_dir/slurm_logs/darshan_t7_fir_<jobid>.out/.err`. wandb offline: `wandb sync wandb/offline-run-*`.

First chain 63847016–18 failed in minutes: `RUN_NAME` was not exported into the `srun bash -c` block, so
`--run_name` was empty (the smoke test passed only because `RUN_NAME` was set at submit). Fixed in `6c8f5e8`
(and the same bug in `bash_tamia_t7.sh`, `b0c8c31`), then resubmitted as 63858143–45.
