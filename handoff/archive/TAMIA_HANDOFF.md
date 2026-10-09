# TamIA handoff: run the t7 RL recipe on TamIA (2026-10-09)

Context for Claude on TamIA, written from the Fir session. Read `CLAUDE.md` first (project, Mila runs and
results); `FIR_HANDOFF.md` has the Fir setup this copies. The user (`dars11`, GitHub `Darshansingh11`) is new to
TamIA: explain cluster specifics briefly when they come up. Prefers uv over conda.

## Why TamIA
On Fir the 1-node smoke test (job 63829085, `rrg-bengioy-ad_gpu`) has been queued since 08:53 PDT on 2026-10-09
with no start estimate: whole 4×H100 nodes are scarce there and the Bengio allocation is over its fair share.
On TamIA, `sbatch --test-only --account=aip-agrawal --nodes=1 --gpus-per-node=h100:4 --time=1:00:00` estimated a
start ~30 min out (partition `gpubase_bynode_b1`). Plan: set TamIA up in parallel; run the 2-node job on whichever
cluster gives GPUs first. **If one cluster starts the full run, don't start it on the other as well** (same
`RUN_NAME` and wandb run id would collide in wandb); ask the user.

## Goal
Run the collaborator's `rl_ft_oracle_t7` recipe on **2 nodes × 4 H100**: per node GPUs 0–2 train, GPU 3 runs a
Qwen3-VL-8B pixel judge; 6 training ranks × 6 prompts = **36 prompts/step**. Start from
`scripts/rl_ft/bash_fir_t7.sh` (branch `darshan-rl-test`) and write `scripts/rl_ft/bash_tamia_t7.sh`.
Trainer flags must stay identical to `bash_fir_t7.sh` / `bash_mila_t7.sh` (pixel judge, hybrid critiques,
`BRANCH=2,4`, LR 1e-4 constant, KL 0.05, 2 PPO epochs, `draft_reward children`, stop penalty 0.5@0.8, 300 steps,
val800 every 50 steps, `DATASET_SEED=19`). Only paths, account, modules and SLURM lines change.
Neither `bash_fir_t7.sh` nor its multi-node `srun` + `torchrun` part has run anywhere yet.

Reference results on Mila (same recipe, 18 prompts/step) for sanity checks: step-0 val800 `val/` final−draft
−0.010, fix 12.0%; `val_oracle/` final−draft +0.009, fix 14.9%, brk 7.4%. Mila smoke test: judge load 50 s,
~7.6 min/step on A100; H100 expected ~4 min/step.

## Steps
Ask the user first: TamIA username (likely `dars11`), where they want the repo, and which account to use
(`sacctmgr -nP show assoc user=$USER format=account`; expected `aip-agrawal`). Check whether compute nodes have
internet (assume not; wandb offline, download everything on a login node, in tmux).

1. **Cluster facts** (record in this file when known): `sinfo -o "%P %G %D %c %m" | sort -u` for GPU types per
   partition (H100 4/node; H200 8/node may exist), CUDA modules (`module avail cuda`), `$SCRATCH` path and
   purge policy, whether whole-node requests are mandatory, max wall time.
2. **Code:** `git clone git@github.com:sarvghotra/tar_reason.git` (add an SSH key to GitHub first if needed),
   `git checkout darshan-rl-test`. Expect commit `7c9d7ba` or later. Work and push only on `darshan-rl-test`;
   never push to `main`. Commit with the user's git identity.
3. **Envs** from the exact Fir package lists, in `~/scratch/tamia_transfer/` on Fir (see step 5 to copy):
   - Training env `~/envs/tar` (Python 3.10; torch 2.1.2+cu121, transformers 4.50.0, peft 0.18.1, flash-attn
     2.5.9.post1 prebuilt wheel, `setuptools==69.5.1`, i.e. <70 because torch 2.1 imports `pkg_resources`):
     `uv venv ~/envs/tar --python 3.10` then
     `uv pip install --python ~/envs/tar/bin/python -r tar_env_freeze.txt --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match`
   - Judge env `~/envs/qwen_judge` (torch 2.6.0+cu124, transformers 4.57.6): same, with
     `qwen_judge_env_freeze.txt` and the cu124 index.
   - Set `UV_CACHE_DIR` on scratch and `UV_LINK_MODE=copy`.
   - Check: `~/envs/tar/bin/python -c "import torch, transformers, peft, flash_attn, wandb"` and
     `~/envs/qwen_judge/bin/python -c "from transformers import Qwen3VLForConditionalGeneration"`.
   - The user runs `wandb login` (API key); entity `darshan-singh`, project `tar_reasoning`.
4. **Public weights** (`uvx --from huggingface_hub hf download ...`), into `$SCRATCH/models/`:
   | Repo | Files | Target |
   |---|---|---|
   | `csuhan/Tar-7B-v0.1` | all | `tar/Tar-7B/` (plain `csuhan/Tar-7B` does not exist) |
   | `csuhan/TA-Tok` | `ta_tok.pth ar_dtok_lp_512px.pth` | `tar/` |
   | `peizesun/llamagen_t2i` | `vq_ds16_t2i.pt` | `tar/` |
   | `google/siglip2-so400m-patch14-384` | `config.json` | `tar/siglip2-so400m-patch14-384/` |
   | `Qwen/Qwen3-VL-8B-Instruct` | all (17 GB, 4 shards) | `vlm/Qwen3-VL-8B-Instruct/` |
   `tok/ta_tok.py` reads the SigLIP2 config from `SIGLIP2_PATH`; without it it falls back to `/scratch/jeet/...`.
5. **From Fir** (rsync from a TamIA login node, in tmux; Alliance asks for MFA; Globus if slow):
   - `dars11@fir.alliancecan.ca:/scratch/dars11/models/tar/sft/slf_ref_edit_t21_ckpt17000/` (17 GB) →
     `$SCRATCH/models/tar/sft/slf_ref_edit_t21_ckpt17000/`
   - `dars11@fir.alliancecan.ca:/scratch/dars11/data/geneval2_50K_v2/` → `$SCRATCH/data/geneval2_50K_v2/`.
     Check: 49,000 train + 800 val rows; md5 prefixes train `317971cc1d9e`, val800 `1d3feb3f9425`; no prompt in
     both files. Don't use the older `geneval2_50K/` train file (it holds 288 val800 prompts).
   - `dars11@fir.alliancecan.ca:/scratch/dars11/tamia_transfer/` (the two env freeze files).
6. **Write `scripts/rl_ft/bash_tamia_t7.sh`** as a copy of `bash_fir_t7.sh`. Change:
   - `#SBATCH --account` → TamIA account; `-o/-e` → a TamIA `output_dir/slurm_logs/`, which must exist before
     `sbatch` (SLURM opens the log files before the script's own `mkdir` runs).
   - `REPO`, `MODELS`, `DATA`, `JUDGE_MODEL`, `JUDGE_PY` → TamIA paths.
   - `module load cuda/12.2` → TamIA's equivalent. Keep it *after* the judge starts: the judge's torch 2.6 must
     find its own CUDA 12.4 libraries (`undefined symbol: __nvJitLinkComplete_12_4` otherwise).
   - `--cpus-per-task` to the node's core count if whole nodes are mandatory.
   - If you use **H200 nodes (8 GPUs)**: `N_GPUS=8`, `JUDGE_GPU=7`, `N_TRAIN_GPUS=7`, `TRAIN_GPUS=0,…,6`, and
     1 node gives 7 ranks (42 prompts/step at 6 prompts/GPU). That changes the batch from t7's 36: ask the user
     before doing that, or set `PROMPTS_PER_GPU` / train ranks to keep 36.
   Commit and push to `darshan-rl-test`.
7. **Smoke test, 1 node** (~30 min):
   `MAX_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=2 EVAL_MAX_PROMPTS=24 RUN_NAME=t7_smoke_tamia sbatch --nodes=1 --time=01:00:00 scripts/rl_ft/bash_tamia_t7.sh`
   Pass = `judge ready after …s`, `step=0` (val/ and val_oracle/ lines), `step=1`, `step=2`,
   `Saved output_dir/t7_smoke_tamia/checkpoint-2`, `Done.` Judge log: `output_dir/<RUN_NAME>/reward_server_<jobid>_n0.log`.
   Record s/step and peak GPU memory (`seff <jobid>` or `nvidia-smi` in the log). **Tell the user the result
   before submitting the 2-node run.**
8. **Full run, 2 nodes**, only after the user agrees: `sbatch scripts/rl_ft/bash_tamia_t7.sh` with
   `RUN_NAME=darshan_t7_tamia`. Chain a follow-up: `jid=$(sbatch --parsable ...); sbatch --dependency=afterany:$jid ...`
   (a job that finds checkpoint ≥ MAX_STEPS exits at once). First checks: both nodes print `judge ready`, rank 0
   prints `train prompts: 49000 (per rank/epoch 8166 or 8167), val prompts per rank: 133 or 134`, and `step=0`.
   wandb sync from a login node: `wandb sync wandb/offline-run-*` in the repo.

## Pitfalls (from Fir and Mila)
- `torchrun` must not use `--standalone` on multi-node; each node runs its own judge on the same port (the trainer
  reaches it at 127.0.0.1). "reward server unreachable" on node 1 = its judge didn't start.
- Only the 5 newest checkpoints are kept (saves every 5 steps). Copy checkpoints 100/200/300 elsewhere before they
  are pruned.
- A validation interrupted by a time limit is redone by the next job (`eval_done_<step>` markers).
- tmux sessions live on one login node. Ctrl-C in a pane where `salloc` waits cancels the request; suggest
  `tmux attach -r` to watch.
- Don't switch git branches in the checkout a running job uses.
