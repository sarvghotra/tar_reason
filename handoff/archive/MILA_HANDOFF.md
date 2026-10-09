# Mila handoff: RL smoke test (2026-10-06)

Context for Claude on Mila, carried over from a Fir session. Read `CLAUDE.md` first. Note that CLAUDE.md lists the
Fir SLURM account as `rrg-bengioy-ad_gpu`: the user is not a member of that allocation (only `def-agrawal_gpu`).

## Why the user is here
The goal is a **test run** of the collaborator's GRPO RL training (`llava/train/rl/train_grpo.py`): show that it
loads, rolls out, scores, steps and saves. Everything was set up on Fir, but no GPU job got scheduled there, so the
user wants to debug on Mila first and then return to Fir to submit the real run. Results don't matter yet.

## What the RL run does
torchrun launches `llava/train/rl/train_grpo.py`. Per step and per GPU: `PROMPTS_PER_GPU=2` GenEval2 prompts × 4
drafts, each draft gets a critique (≤128 tokens) and 2 refined children (`BRANCH="4,2"`). Reward (`--reward_kind
latent`, the default): a frozen Tar-7B answers the prompt's VQA questions in latent space; score =
`ALPHA·AM + (1−ALPHA)·GM` with `ALPHA=1.0`. LoRA-only update (r=64, LR 1e-5, KL 0.01). Each GPU holds a full
replica (policy + frozen reward model + de-tokenizers), so 1 GPU runs the same code path as 4, with a smaller batch.
Expected memory: ~40 GB+ per GPU, so use 80 GB GPUs only.

## Code
- Repo `git@github.com:sarvghotra/tar_reason.git`, branch **`darshan-rl-test`**. Work and push only there. The user
  asked not to push to `main` (main has a revert commit removing the script).
- `darshan-rl-test` = collaborator's latest code (`ff96ea9`, adds oracle/hybrid critiques, constant LR, advantage
  std floor) + `scripts/rl_ft/bash_darshan.sh` (the Fir launch script). Every flag it passes exists in
  `train_grpo.py`; new options default to the old behaviour (`--critique_source policy`, `--adv_std_floor 0`,
  cosine LR).
- `bash_darshan.sh` reads `N_GPUS`, `MAX_STEPS`, `EVAL_STEPS`, `SAVE_STEPS`, `EVAL_MAX_PROMPTS` from the environment
  (defaults 4 / 500 / 25 / 25 / full val set), so a smoke test needs no edits.
- Commit with the user's git identity (GitHub `Darshansingh11`).

## Assets
| Item | Source |
|---|---|
| Tar-7B (policy base + frozen reward model) | HF `csuhan/Tar-7B-v0.1` → `Tar-7B/` (the plain `csuhan/Tar-7B` repo does not exist) |
| `ta_tok.pth`, `ar_dtok_lp_512px.pth` | HF `csuhan/TA-Tok` |
| `vq_ds16_t2i.pt` | HF `peizesun/llamagen_t2i` |
| SFT policy `slf_ref_edit_t21` checkpoint-17000 (17 GB, safetensors + tokenizer) | Fir only: `/scratch/dars11/models/tar/sft/slf_ref_edit_t21_ckpt17000` |
| GenEval2 train (49,488) + val256 jsonl | Fir only: `/scratch/dars11/data/geneval2_50K/` |

On Fir there are also `slf_ref_edit_t20_ckpt16000` and `slf_ref_edit_t6_repro_w_corr_t2_ckpt20000`; the user
chose **t21** (the collaborator's latest RL runs start from it).

## Pending (in order)
1. **Clone or update the repo** on Mila and check out `darshan-rl-test`. Ask the user where their Mila checkout is.
2. **Python env** with uv (user prefers uv over conda), Python 3.10, same recipe as Fir:
   - `torch==2.1.2` from the cu121 index, then `fir_requirements.txt` with the conda-only `@ file://` pins removed
     and torch / flash-attn / triton / nvidia-* excluded.
   - `uv pip install "setuptools<70"`: torch 2.1's `cpp_extension` imports `pkg_resources`, which newer
     setuptools dropped.
   - flash-attn: install the prebuilt wheel instead of building from source:
     `https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
   - Verify: `python -c "import torch, transformers, peft, flash_attn, wandb"`. On Fir this gave torch
     2.1.2+cu121, transformers 4.50.0, peft 0.18.1, flash-attn 2.5.9.post1, wandb 0.26.0.
   - `wandb login` must be run by the user (API key). wandb entity: `darshan-singh`.
3. **Weights:** download the HF items into `$SCRATCH/models/tar/` (`uvx --from huggingface_hub hf download ...`).
4. **Fir-only items:** the user runs, in tmux, from Mila (Alliance asks for MFA):
   `rsync -avP dars11@fir.alliancecan.ca:/scratch/dars11/models/tar/sft/slf_ref_edit_t21_ckpt17000 $SCRATCH/models/tar/sft/`
   and the same for `/scratch/dars11/data/geneval2_50K`. Fall back to Globus if too slow.
5. **Write `scripts/rl_ft/bash_mila.sh`** from `bash_darshan.sh` (keep `bash.sh` and `bash_darshan.sh` untouched):
   - `#SBATCH`: remove `--account`; use a Mila partition and an 80 GB GPU type (`a100l` or `h100`; check `sinfo`).
     `-o/-e` → a Mila output dir that exists before submission.
   - Paths (`PREV_STAGE_CHECKPOINT`, `REWARD_MODEL_PATH`, `AR_MODEL`, `DECODER`, `ta_tok.pth` source, `cd`,
     `PYTHONPATH`) → Mila copies. Use `$SLURM_TMPDIR` instead of `/tmp` for `ta_tok.pth`.
   - `module load cuda/...` → the Mila module name (check `module avail cuda`).
   - `WANDB_MODE` can be `online` (Mila compute nodes have internet).
   - The script reads `output_dir/${RUN_NAME}/data.yaml` and `val.yaml` (not the ones in `scripts/rl_ft/`); write
     them there, pointing at the Mila jsonl paths. `output_dir` is gitignored.
   Commit and push to `darshan-rl-test`.
6. **Smoke test** on 1 GPU, interactively (`salloc`) or as a batch job:
   `N_GPUS=1 MAX_STEPS=3 EVAL_STEPS=3 SAVE_STEPS=3 EVAL_MAX_PROMPTS=16 bash scripts/rl_ft/bash_mila.sh`
   Check it loads, rolls out, scores, takes a training step, evaluates and saves `checkpoint-3`; record peak GPU
   memory and seconds per step. Then try 2–4 GPUs to exercise the gradient all-reduce.
7. **Fix bugs** on Mila, push fixes to `darshan-rl-test`, then tell the user what to bring back to Fir. On Fir:
   `git pull`, then submit `bash_darshan.sh` (smoke test command is in `FIR_HANDOFF.md`).

## Later: the collaborator's t7 recipe
The collaborator's best current run (`rl_ft_oracle_t7`) uses a pixel-space reward: a Qwen3-VL-8B-Instruct judge
server (`llava/train/rl/pixel_reward_server.py`) in a separate env with transformers 4.57, oracle/hybrid critiques,
2 nodes × 4 H100 with one judge GPU per node. Its launch script was pasted in chat and is not in the repo (ask the
user for it). The code for it is on `darshan-rl-test`. Still missing from the collaborator: t7 `data.yaml`/`val.yaml`
plus the 800-prompt val jsonl, and a `pip freeze` of his `geneval2` judge env. Do the latent-reward smoke test first.

## Gotchas from Fir
- tmux sessions live on one login node; a new SSH session may land on another node and not see them.
- Pressing Ctrl-C (or killing tmux) in a pane where `salloc` is waiting cancels the request. Suggest
  `tmux attach -r` for watching.
- Don't switch git branches in the checkout a running job uses. The job reads the code from those files.

## User preferences
- Prefers uv over conda.
- Briefly explain cluster specifics when they come up.
- Don't push to `main`.
