# Agent instructions for this repository

## Read first

- `scripts/clusters/README.md`: portable filesystem layout and job submission.
- `scripts/CLUSTER.md`: detailed environment and RL notes; historical job IDs are
  not evidence that a job is currently running.
- `docs/AGENT_HANDOFF.md`: implementation status and outstanding checks.

## Working expectations

Plan before substantial edits. Trace the relevant execution path and do not
change uncertain behavior based on assumptions. Run meaningful adversarial
checks, particularly for reward attribution, EOS/padding, sampling likelihoods,
checkpoint compatibility, file replacement, and distributed updates.

Preserve unrelated working-tree edits. Do not commit environments, credentials,
linked datasets/models, or run artifacts. Do not copy another author's Git
identity. Update the handoff when behavior or validation status changes.

## Intended RL objective

The trainable first autoregressive model samples independent complete episodes
per prompt. Each starts with semantic image tokens, then reflection text and
optional refinements. Tokenizer EOS ends an episode; sampled `<im_start>` requests
another image. The phrase "looks good" is not a termination condition. EOS and
image-transition tokens are trainable actions; forced prefixes/splices are not.

Render only the final complete image for reward. The frozen second AR model
samples VQ codes and the frozen VQ decoder produces pixels. A frozen
Qwen3-VL-8B-Instruct judge computes GenEval2 soft-TIFA AM on those pixels. Do not
restore semantic-token rewards or local improvement credit. All sampled actions
in an episode receive its final reward minus the same prompt's mean reward
(optional standard-deviation normalization remains supported). Loss weights
are normalized within episodes, then averaged across episodes.

Safety caps currently default to 3 refinements, 128 tokens per reflection, and
4096 total tokens subject to the architectural context limit. Capped episodes
score their last complete image and report truncation separately; no fake EOS
or extra penalty is inserted.

First-model sampling uses temperature 1, top-k 0, top-p 1, and phase-specific
vocabulary masks to match the training likelihood. Keep policy dropout zero.
The second AR renderer currently uses CFG 4, temperature 1, no top-k/top-p
truncation, and stochastic sampling. Its settings affect pixel rewards.

The Qwen scoring protocol intentionally matches official GenEval2's first-token
answer-variant summation, including duplicate IDs and space-leading numeric
variants. This differs from unique-token probability sums and can exceed one.
Do not silently alter it while claiming benchmark parity.

## Entry points

- `llava/train/rl/rollout.py`: episodes, termination, action masks and row weights.
- `llava/train/rl/train_grpo.py`: final-pixel scoring, advantages, training, logs.
- `llava/train/rl/grpo.py`: restricted log-probabilities and clipped objective.
- `llava/train/rl/pixel_reward.py`: persistent external Qwen judge client.
- `llava/train/rl/qwen_reward_worker.py`: official pixel VQA protocol.
- `scripts/rl_ft/bash.sh`: shared RL launcher.
- `scripts/clusters/fir/`: configured cluster profile, setup and submission.
- `scripts/clusters/generic/`: unconfigured other-cluster placeholder.

Only LoRA on the first model trains. The renderer and judge remain frozen.
Evaluation exemplars must reuse the exact final image scored by Qwen; rendering
it again could show different pixels. Save per-question scores, reflections,
stop reasons and images alongside aggregate metrics. Existing checkpoints from
other reward objectives must not silently resume under the pixel objective.

## Filesystem and environments

Use repo-local `data/`, `models/`, `sft_model/`, `reward_model/`, and `results/`
symlinks. Put cluster-specific paths/accounts/modules in cluster profiles or
ignored local configuration, not the shared trainer. `.cluster` selects the
local profile. Do not overwrite real directories during symlink setup.

Recreate environments on each cluster; do not transfer compiled environments.
`tar/` runs the first model and renderer; `qwen_reward/` runs the newer Qwen3-VL
stack in a separate process. Keep outputs, W&B files, caches and logs in
`results/`; use node-local temporary storage when available.

## Validation and jobs

CPU regressions, when the environment and temporary filesystem are healthy:

```bash
PYTHONPATH="$PWD:$PWD/tests" tar/bin/python -m unittest discover -s tests -p 'test_rl_*.py' -v
PYTHONPATH="$PWD:$PWD/tests" python3 -m unittest test_cluster_layout test_gpu_monitor -v
```

The distributed CPU test needs local Gloo sockets. Run scheduler commands and
checks requiring those sockets with the environment's appropriate permissions.
Do not work around access controls or destroy files to address storage errors.

Prefer standalone `sbatch` tests over unattended `salloc`: an interactive
allocation may be cancelled when its owning terminal disappears. Inspect
`squeue`, `sacct`, and both stdout/stderr before reporting success. The GPU smoke
test writes reports under `results/evaluations/qwen_reward_smoke/` and performs
no optimizer updates. Passing it is not equivalent to validating a full RL run.
