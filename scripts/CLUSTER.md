For portable links and per-cluster submission commands, start with [the cluster README](clusters/README.md). This file retains detailed Fir environment and RL implementation notes.

# Running Tar on the Alliance cluster

Run commands below from `/home/xiaofeng/tar_reason`. The launchers load
`StdEnv/2023 python/3.10 cuda/12.2`, activate the repo's `tar/` environment
(Python 3.10.13), and set `PYTHONPATH` automatically. Conda is not needed.

## Inputs verified on September 9, 2026

The shared root is `/project/rrg-bengioy-ad/jeet`.

| Input | Path relative to the shared root | Status |
|---|---|---|
| Base model | `models/Tar-7B` | Weight shards readable |
| SFT policy | `models/sft_ckpt/slf_ref_edit_t6_repro_w_corr_t2/checkpoint-24000` | Config and all four weight shards readable |
| Vision tokenizer | `models/tar/ta_tok.pth` | Present |
| AR de-tokenizer | `models/tar/ar_dtok_lp_512px.pth` | Present; produces 512px images |
| Image decoder | `models/tar/vq_ds16_t2i.pt` | Present |
| RL training data | `data/T2I_datasets/geneval2_50K/evaluation_metadata_shuf_train.jsonl` | Present |
| RL validation data | `data/T2I_datasets/geneval2_50K/evaluation_metadata_shuf_val256.jsonl` | Present |

The SFT shard permission issue was resolved and all four shards were opened
successfully during the RL fix verification. The launchers check shard
permissions before starting Python training. Setting a different model path is
supported, but using Tar-7B instead of the SFT model changes the experiment.

`bash scripts/prepare_cluster.sh` initializes the output directories and caches
the small SigLIP configuration needed by TA-Tok. This preparation has been run;
repeat it if you move or clear the cache. It does not download model weights.
Launchers default to `HF_HUB_OFFLINE=1`; set it to `0` explicitly if a run needs
network access, such as fetching additional evaluation assets on a login node.
For a separate local SigLIP config directory, set `TAR_SIGLIP_MODEL`.

## Outputs

`results/` is your existing symlink to `/scratch/xiaofeng/tar_reason_results/`.

| Contents | Location |
|---|---|
| Trained models, adapters, optimizer state | `results/models/$RUN_NAME/` |
| Console logs | `results/logs/$RUN_NAME/` |
| Slurm stdout and stderr | `results/logs/slurm/` |
| W&B runs, media and staged artifacts | `results/wandb/$RUN_NAME/` |
| Validation metric JSON files and generated evaluation images | `results/evaluations/$RUN_NAME/` |
| Hugging Face, Triton, extension and W&B caches | `results/cache/` |

W&B defaults to `WANDB_MODE=offline`. To upload runs, explicitly set
`WANDB_MODE=online` and provide your own W&B authentication and, if needed,
`WANDB_ENTITY`. No API keys or another user's W&B entity are embedded in scripts.
Temporary files use `SLURM_TMPDIR` inside jobs and `results/tmp/` otherwise.

## RL training

Check inputs and print the full launch command without allocating GPUs or
loading model weights:

```bash
TAR_DRY_RUN=1 bash scripts/rl_ft/bash.sh
```

To check the command using the base model instead:

```bash
TAR_DRY_RUN=1 \
PREV_STAGE_CHECKPOINT=/project/rrg-bengioy-ad/jeet/models/Tar-7B \
bash scripts/rl_ft/bash.sh
```

Submit the default RL experiment:

```bash
RUN_NAME=rl_pixel_sft24k N_GPUS=4 \
bash scripts/submit.sh scripts/rl_ft/bash.sh --job-name=tar-rl
```

`submit.sh` defaults to account `rrg-bengioy-ad_gpu`, one node, one task,
four GPUs, 24 CPUs, 256 GB RAM and three hours. Change resource requests with
trailing `sbatch` options, for example `--time=12:00:00 --gpus-per-node=h100:4`.
Set `SLURM_ACCOUNT` if your allocation uses another account. Allocation access
and resource availability have not been tested by submitting a job.

Within an existing suitable GPU allocation, use:

```bash
RUN_NAME=rl_pixel_sft24k N_GPUS=4 bash scripts/rl_ft/bash.sh
```

Use a new `RUN_NAME` for a new experiment. Reusing it resumes the latest RL
checkpoint under `results/models/$RUN_NAME` and keeps the W&B run ID stable.
RL checkpoints contain LoRA adapters; they are not standalone full models.
The one-shot/iterative scripts below take full models, not these adapter folders.

RL settings such as `MAX_STEPS`, `LR`, `LOG_IMAGES`, `PROMPTS_PER_GPU`,
`GEN_BATCH_SIZE`, `NUM_ROLLOUTS`, `MAX_REFINEMENTS`, `MAX_SEQ_LEN`,
`DATA_PATH` and `VAL_DATA_PATH` accept environment
overrides. `REWARD_MODEL_PATH` defaults to
`reward_model/` (linked to `results/pretrained/Qwen3-VL-8B-Instruct` on Fir). `REWARD_PYTHON` defaults to
`$REPO_ROOT/qwen_reward/bin/python`; point it at a separate environment with
Qwen3-VL support (Transformers 4.57+ and its compatible PyTorch/vision stack).
The current `tar` environment uses Transformers 4.50 and cannot load this judge.
The checkpoint is now downloaded under `results/pretrained/`, and the separate
Python 3.11 environment is installed under `results/environments/qwen_reward`
with the repo's `qwen_reward` symlink pointing to it. Installed versions are
PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, and Transformers 4.57.6.
`results/environments/qwen_reward.freeze.txt` records the full environment;
`download_provenance.json` in the model directory records its pinned HF revision.
There is no latent-reward fallback.
Direct `bash` launch also accepts trailing trainer arguments.

`scripts/Tar_1.5B_finetune_RL.sh` forwards to the current RL launcher: its old
DeepSpeed/GRPO argument list no longer matched `train_grpo.py`. The current
shared policy is 7B despite that legacy filename.

RL sampling uses temperature `1.0`, top-k `0`, and top-p `1.0` for both image
tokens and reflections. Only the image/text vocabulary mask changes the
softmax, matching the probabilities in the PPO loss. Incompatible sampling
overrides and nonzero policy dropout are rejected; checkpoint decoding defaults
cannot silently override the RL sampler. Standalone inference scripts retain
their own decoding settings. The trainer initializes all ranks with the same
seed and broadcasts trainable adapters from rank 0 before optimizer creation;
rollout seeds still differ by rank and step.

Use a new run name when comparing against runs made before these fixes. A new
run starts from the SFT weights; resuming an older RL adapter cannot undo its
previous updates under the mismatched sampler or divergent rank policies.

CPU regression checks (tiny local models, no shared weights or GPU required):

```bash
source scripts/cluster_env.sh
TMPDIR=/tmp \
python -m unittest discover -s tests -p 'test_rl_*.py' -v
```

The distributed test starts two local Gloo processes. These checks do not
replace a GPU smoke test with the actual SFT checkpoint.

## Final-image episode objective

The RL launcher now defaults to `RUN_NAME=rl_pixel_qwen`. Every prompt gets
`NUM_ROLLOUTS=4` independent episodes, without shared drafts or correction
branches. Tar generates a complete semantic image, then reflection text.
The checkpoint's tokenizer EOS (`<|im_end|>`, ID 151645) ends the episode.
A sampled `<im_start>` instead requests another image; the scale token is
inserted and the next complete image is sampled. The phrase "looks good"
has no special control behavior. The sampled EOS and image-transition tokens
are retained as trainable text actions; inserted prefixes and scale tokens are
not trained.

Only the last complete image in each episode is rendered and scored by frozen
`Qwen/Qwen3-VL-8B-Instruct`, the official GenEval2 judge. Reward is soft-TIFA AM
on pixels: for each VQA question, append `Answer in one word.`, obtain the
first generated token's distribution with greedy one-token generation, sum
accepted answer-variant scores, and take their arithmetic mean. GM is logged
only as a diagnostic. The first-token variant sum matches the official
[GenEval2 evaluator](https://github.com/facebookresearch/GenEval2/blob/main/evaluation.py),
including duplicate token IDs and space-leading numeric variants. It does not
apply the legacy latent scorer's deduplication; benchmark-compatible scores
can therefore exceed one. Values are kept on the evaluator's per-image scale,
without multiplying by 100.

The second AR renderer uses CFG 4, temperature 1, top-k 0, top-p 1 with
multinomial sampling. Its sampled pixels now affect the reward. All renderer
and Qwen weights stay frozen; only first-model LoRA is optimized. One persistent
Qwen subprocess per training rank loads on that rank's GPU. It can use a
separate Python environment without upgrading `tar`. Images cross the process
boundary as lossless temporary PNGs, removed after scoring; worker errors stop
the run rather than inventing rewards. Decoding and Qwen evaluation now incur
compute and memory costs even with `LOG_IMAGES=0`. GPU memory/runtime have not
been validated with the full models.

There are no intermediate image scores or improvement rewards.
`ADV_NORM=mean` uses final reward minus the mean final reward across the same
prompt's episodes. `ADV_NORM=std` additionally divides by group standard
deviation. Every sampled token receives that episode advantage, including the
initial image and terminal EOS. Token losses are averaged within each episode
(respecting `REFLECT_TOKEN_WEIGHT`), then across episodes.

EOS can stop at any reflection. Safety limits default to
`MAX_REFINEMENTS=3` (up to four complete images), `REFLECT_TOKENS=128` per
reflection, and `MAX_SEQ_LEN=4096` including the prompt. The architectural
context limit is also respected. A reflection that reaches its token limit
without EOS or `<im_start>` is truncated; it does not force another image.
If an episode hits a safety limit, its last complete image is scored without
an extra truncation penalty, and no synthetic EOS is added. Monitor
`train/eos_rate`, `train/truncated_rate`, `train/max_refinements_rate`,
`train/max_seq_len_rate`, and `train/reflection_limit_rate`; raise the relevant
budget if truncations are frequent. This scores capped episodes as well as
EOS-completed episodes, rather than silently discarding difficult rollouts.

Validation uses the same stochastic episode policy with one rollout per prompt
and reports final scores and stop reasons. Intermediate-score/delta metrics
were removed because intermediate images are no longer scored. Standalone
legacy evaluation scripts still use their separate generation protocols.

Visual evaluation runs before training (step 0) and every `EVAL_STEPS=25`
updates by default, using the same validation prompts and sampling seed.
`LOG_IMAGES=8` saves exemplars from rank 0's fixed validation shard; aggregate
metrics cover all validation shards. Under
`results/evaluations/<run>/step-<N>/`, `evaluation.json` records exemplar
prompts, reflections, semantic codes, per-question scores, final rewards, stop reasons, and aggregate
metrics. Separate PNGs retain each draft/refinement, and a horizontal trajectory
PNG shows them in order with the final image at the right. The final PNG is
exactly the image scored by Qwen, retained without re-rendering. Earlier images
are decoded for inspection only, with a fixed exemplar seed. Validation uses a
fixed seed for rollout generation and final-image rendering. Compare step 0 with later steps to
inspect changes from RL; these stochastic examples are illustrative, while
aggregate validation rewards provide the broader comparison.

W&B receives the aggregate `val/*` metrics and `val/images` trajectory strips
with prompts, reflection text, final scores, and stop reasons in their captions.
Local visual artifacts are saved even with `--report_to none`. `LOG_IMAGES=0`
disables exemplar logging; reward images are still decoded and numeric validation JSON is saved. On resume,
an existing step-0 baseline is preserved rather than regenerated from a trained
adapter. Intermediate images are visual diagnostics, not additional rewards.

Start this objective with a new run name. `--branch`, `--group`, and the
`BRANCH`/`GROUP` environment variables are obsolete and rejected. New
checkpoints record `objective=final_pixel_qwen3vl_soft_tifa_am_v1`; resuming an old
local-reward or final-semantic-reward checkpoint is rejected before model loading. The underlying LoRA
adapter file format is unchanged.

## Evaluation

The current `tts/eval/1shot_gen.py`, `tts/eval/iterative_generation.py`, and
`tts/generate_step_by_step.py` import helpers that are absent from
`llava/train/rl/grpo.py`. Their generation paths require a separate repair
before the TTS commands below can run, even after weight access is granted.
Dry runs and `--help` do not exercise those imports. This is separate from
the RL trainer's `EpisodeRollout` path.

For one-shot image generation with the readable base model:

```bash
RUN_NAME=tar7b_one_shot N_GPUS=1 \
MODEL_PATH=/project/rrg-bengioy-ad/jeet/models/Tar-7B \
bash scripts/submit.sh tts/eval/run_1shot.sh --job-name=tar-eval
```

For iterative generation with the SFT model after repairing the missing helpers:

```bash
RUN_NAME=sft24k_iterative N_GPUS=1 \
bash scripts/submit.sh tts/eval/run_iterative.sh --job-name=tar-iter
```

Both use the shared 256-prompt validation file and 512px de-tokenizer. Override
`MODEL_PATH` and `PROMPTS_FILE` as needed. `TAR_DRY_RUN=1 bash <script>` checks
inputs and prints the command for either launcher. These scripts generate
images; they do not compute external benchmark scores.

## Other scripts and outstanding inputs

| Launcher | Still needed |
|---|---|
| `scripts/Tar_1.5B_pretrain_demo.sh` | Qwen2.5-1.5B-Instruct and the datasets in `scripts/data_demo.yaml` |
| `scripts/Tar_1.5B_finetune_demo.sh` | Tar-1.5B and the demo datasets |
| `scripts/Tar_1.5B_finetune_viscot.sh` | Tar-1.5B, Visual-CoT JSON and images |
| `scripts/img_gen/sft.sh` | The three critique datasets in `scripts/img_gen/sft.yaml`; defaults to the available Tar-7B |
| `scripts/eval/Tar_1.5B_pretrain_demo_gen_eval.sh` | Original DPG Bench and GenEval prompts; override `DPG_PROMPTS` / `GENEVAL_PROMPTS` |
| `scripts/eval/Tar_1.5B_pretrain_demo_und_eval.sh` | A compatible `lmms_eval` installation and task data, e.g. MME |
| `tts/eval/reward.sh` | `tts/rl_w_vlm/vlm_reward.py` (absent from this checkout), decomposition data and Qwen2.5-VL weights |

These launchers now share the environment and output conventions, but the
missing inputs were not downloaded or substituted. GenEval2 VQA prompt files
are not a substitute for the original GenEval scoring metadata. The generation
benchmark launcher produces images; external DPG/GenEval scoring remains separate.

The pinned Diffusers 0.39.0 / PyTorch 2.1.2 import incompatibility remains for
the SANA/Lumina diffusion paths. The AR-based launchers here do not import
Diffusers. No GPU training, image generation or benchmark scoring was executed
during setup; path checks, parser checks and imports do not verify GPU runtime
compatibility or memory requirements.


## Qwen judge environment checks

`scripts/prepare_qwen_reward.sh` recreates/updates the isolated environment using
uv; `tar/bin/python scripts/download_qwen_reward.py` downloads the official
checkpoint (requires network access with `HF_HUB_OFFLINE=0`). Both use scratch
through `results/` to avoid filling the home quota.

Inside a suitable interactive GPU allocation:

```bash
cd /home/xiaofeng/tar_reason
source scripts/cluster_env.sh
python -u scripts/test_qwen_reward_gpu.py
```

This first tests the real Qwen judge on red/blue image fixtures and repeat
scoring. It then samples two SFT draft/reflection episodes, decodes their final
images, scores pixels with Qwen, checks episode advantages, and saves the exact
scored images. No optimizer updates are performed. Evidence is written under
`results/evaluations/qwen_reward_smoke/`.

The cron monitor `scripts/monitor_gpu_test.py` checks allocation 59153811 once
per minute. On RUNNING it uses `srun --jobid=59153811 --overlap` to execute the
GPU smoke test inside that allocation, leaving the interactive shell open.
`results/logs/gpu-test-59153811/status.json` records waiting/testing/passed/failed;
`test.log` holds test output and `cron.log` holds cron errors. A file lock avoids
overlapping invocations, and an `attempted` marker prevents automatic reruns even
after failure. When the allocation disappears, monitoring becomes a no-op.
The crontab entry is tagged `tar-gpu-test-59153811` for removal with `crontab -e`
when it is no longer needed. This monitor runs tests, not automatic code fixes.
Cron installation was verified, but scheduled execution was not observed on the
login host. A fallback CPU job, 59161662 on `def-bengioy`, is submitted with
`--dependency=after:59153811`. It invokes the same monitor and uses the same
exclusive attempt marker, so cron and Slurm cannot launch duplicate tests.
