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
default to normalization within episodes, then averaging across episodes.
`--length_normalization constant` instead divides sampled token weights by the
configured `max_seq_len` before averaging episodes. `--kl_gradient_correction`
enables the differentiable, unclipped policy ratio on K3 KL. Both are opt-in;
legacy defaults are `episode` and correction off. Checkpoint method metadata
must match on resume; missing metadata means legacy settings.

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

Opt-in `--reset_context_per_turn` (`RESET_CONTEXT_PER_TURN=1` in the launcher)
rebuilds each subsequent refinement context from the prompt and latest image.
All exact-context segments are trained with final episode advantage; copied
images are masked, and normalization/optimizer averaging remain per episode.
The logical episode budget is not reset. This flag is checkpoint method
metadata; missing means false, and standalone evaluation inherits it.

Opt-in `--tree_branch "R[xM],..."` (`TREE_BRANCH`) samples the episodes as a
tree: after image k, R reflections per image and M images per reflection that
requests one; unlisted rounds do not branch. Rewards remain final images only
(one render/score per distinct final image). A node's value is the mean final
reward of its subtree; its advantage is value minus its siblings' mean, and an
only child shares its nearest branch point's advantage. Drafts have mass 1 and
children split it equally; weights use mass (not descendant counts) and the
denominator counts drafts. Opt-in `--text_policy_gradient transitions`
removes reflection prose from the policy gradient but keeps its KL; EOS,
`<im_start>` and image tokens keep it. Both are checkpoint method metadata
(missing means `[]` / `all`); validation and evaluation stay single-chain.

Evaluation-only `--protocol looks_good` (`eval_geneval2`,
`generate_instruction_benchmarks`) runs a draft plus up to `--max_refinements`
reflect -> refine rounds, stopping on a "looks good" verdict (otherwise forcing a
refinement); it needs an explicit context flag and supports `--decoding greedy`.

- `llava/train/rl/rollout.py`: episodes, termination, action masks and row weights.
- `llava/train/rl/train_grpo.py`: final-pixel scoring, advantages, training, logs.
- `llava/train/rl/grpo.py`: restricted log-probabilities and clipped objective.
- `llava/train/rl/pixel_reward.py`: persistent external Qwen judge client.
- `llava/train/rl/qwen_reward_worker.py`: official pixel VQA protocol.
- `scripts/rl_ft/bash.sh`: shared RL launcher.
- `scripts/clusters/fir/`: configured cluster profile, setup and submission.
- `scripts/clusters/generic/`: unconfigured other-cluster placeholder.

Opt-in `--objective janus_one_iteration` (requires `--max_refinements 1`) follows
Janus-Pro-R1: draft -> reflection -> one refinement unless the Correction says
"looks good" (`--verdict_rule correction|anywhere`). Rewards AM(draft), the
verdict reward and AM(final) train the draft, reflection and final image (the
draft again when kept) as three segments. Verdict reward `--verdict_target
all_pass` (default): 1[verdict == every draft question >= 0.5]; `soft_am` is
the Janus formula `1 - |AM(draft) - verdict|`, which collapsed to "looks good"
(soft AM > 0.5 on most failing drafts) and is kept only to reproduce that run; each is normalized per
prompt group with the unbiased std, and the loss averages the three per-sequence
token means. Every draft and refined image is rendered and judged once. A stop
token that disagrees with the verdict keeps KL but no PG. Default objective
remains `final_episode`; both are checkpoint method metadata.
`--render_batch_size` (renderer rows sample independently) and
`--reward_question_batch_size` (left-padded judge forwards, same variant sums)
only change throughput; default 1 is the legacy path, and benchmark evaluation
keeps batch 1. Batched judging failed the GPU equivalence check (per-question
diffs up to 0.108 for a 1.4x speedup), so keep the judge at 1.

Opt-in `--objective teacher_tree` (`llava/train/rl/teacher.py`,
`EpisodeRollout.run_teacher_tree`): after each image the judge model, acting
as a teacher that sees only the prompt and the rendered image, writes the
reflection. It is parsed into the policy's own format (a leading space,
`<critique>\nCorrection: <instruction>\n` + `<im_start>`, or exactly
`no issues, it matches the prompt.\nCorrection: looks good.` + EOS) and spliced:
"looks good" stops, otherwise `--tree_branch` gives M policy images per teacher
reflection (every entry needs R = 1), up to `--max_refinements`. Teacher tokens
are off-policy: no PG and no KL, only cross-entropy (`--teacher_ce_coef` x the
mean per-token NLL over distilled teacher reflections; `--teacher_ce_filter agree`
distills only those whose verdict equals the judge's all_pass target of the image
they critique, default `all`). Image tokens get final-reward tree
credit (an intermediate image is credited with the final rewards it led to). After
every image the policy samples `--shadow_reflections` reflections from the teacher
reflection's context; they are never continued, and their reward is verdict
correctness against the judge's all_pass target (prompt groups, unbiased std;
`--shadow_coef` x the mean of per-shadow token means; stop-token mismatch keeps
KL, no PG). Each image is rendered once; the judge scores those exact pixels.
Validation is the teacher-free looks_good protocol. Unparseable teacher replies
stop the episode (`teacher_malformed`). All settings are checkpoint method metadata.

Opt-in `--objective janus_multi_iteration` (`EpisodeRollout.run_janus_multi`): the
Janus episode with up to `--max_refinements` reflect -> refine rounds and the
policy's own reflections (no teacher, no tree). "Looks good" stops; a refine verdict
forces the next image with `run_one_iteration`'s splice rule; the image at the cap is
still reflected on. Every image span is trained with the episode's final-image AM
advantage; every reflection span with verdict correctness against the judge's
all_pass target of the image it critiques (pooled per prompt group); both use the
unbiased group std; the loss averages per-span token means per episode. A stop token
that disagrees with its verdict keeps KL, no PG. Validation: teacher-free looks_good.

Benchmark scripts default to `--protocol native` (the trained policy). Opt-in
`--protocol main_adhoc` reproduces origin/main's forced one-refinement generation
(`llava/train/rl/main_protocol.py`) for cross-branch comparison; keep it
byte-faithful to main and label results by protocol.

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

uv mode (set `TAR_ROOT=$SCRATCH/tar_reason`) is the portable layout, one folder
per cluster:
- `shared/` holds the inputs (`MANIFEST.json`, `PROVENANCE.json`;
  `scripts/clusters/shared_inputs.py`). cluv pushes it from Fir without `--delete`.
- `results/` holds the outputs. They are fetched back to Fir; on Fir this is a
  link to `$SCRATCH/tar_reason_results`.
- `env/venvs/<uv.lock hash>` holds the one environment (`pyproject.toml`/`uv.lock`:
  Python 3.11, torch 2.8 cu128, transformers 5.17, flash-attn 2.8.3).
  `scripts/clusters/setup_uv.sh` builds it and links `.venv` to it. `$HOME` quotas
  are too small for it.

HEAD must keep running under transformers 4.50 (`tar/`), 4.57 (`qwen_reward/`) and
5.x. Go through `llava/train/rl/hf_compat.py` (`dtype_kwargs`,
`without_model_generation_defaults`); never pass `torch_dtype` or
`use_model_defaults` directly. 5.x merges the checkpoint's generation_config.json
(temperature 0.7, top-k 20, top-p 0.8, repetition penalty 1.05) into `generate`
unless blanked. Legacy environments stay for frozen snapshots until the parity gates in
`docs/AGENT_HANDOFF.md` pass; judge numbers can shift between environments, so
compare configurations only within one environment and one GPU model.

## Validation and jobs

CPU regressions, when the environment and temporary filesystem are healthy:

```bash
PYTHONPATH="$PWD:$PWD/tests" tar/bin/python -m unittest discover -s tests -p 'test_rl_*.py' -v
PYTHONPATH="$PWD:$PWD/tests" .venv/bin/python -m unittest discover -s tests -p 'test_rl_*.py' -v
PYTHONPATH="$PWD:$PWD/tests" python3 -m unittest test_cluster_layout test_gpu_monitor test_cluster_launch test_shared_inputs -v
PYTHONPATH="$PWD:$PWD/tests" .venv/bin/python -m unittest test_eval_suite -v
```

The distributed CPU test needs local Gloo sockets. Run scheduler commands and
checks requiring those sockets with the environment's appropriate permissions.
Do not work around access controls or destroy files to address storage errors.

Prefer standalone `sbatch` tests over unattended `salloc`: an interactive
allocation may be cancelled when its owning terminal disappears. Inspect
`squeue`, `sacct`, and both stdout/stderr before reporting success. The GPU smoke
test writes reports under `results/evaluations/qwen_reward_smoke/` and performs
no optimizer updates. Passing it is not equivalent to validating a full RL run.

Request GPUs by count, not node shape, wherever the cluster allows it:
`--ntasks=N --gpus-per-task=TYPE:1 --mem-per-gpu=...`, with `--switches=1@<wait>`
for multi-GPU training. Trillium and tamIA accept only whole nodes.

Train as 3-hour `--array=0-K%1` chains (`scripts/rl_ft/submit_short_array.sh`):
- `LAUNCHER=srun` with `scripts/clusters/srun_ddp.sh`;
- `JOB_DEADLINE=1`, so an update is started only if it, its validation and the
  save fit before the end;
- `RESUME_SAVE_STEPS` saves to `output_dir/resume`, outside the milestone rotation;
- `TAR_STAGE_DIR` stages inputs to node-local disk.
Seeds depend on the step and rank, so a chain samples what one long job would with
the same GPU count.

Evaluation suites come from `scripts/eval/looks_good_suite.py`: one GPU per task,
light replays packed per GPU, finalize steps on a small GPU, summaries on CPU.
Measured needs are in `docs/AGENT_HANDOFF.md`. Training needs 80 GB GPUs; most
evaluation stages fit 40-48 GB.

## W&B preference

Use online W&B logging for future runs by default (`WANDB_MODE=online`).
Keep credentials outside version control. Do not restart existing offline runs
just to change logging mode; job 59261866 was explicitly left offline.
