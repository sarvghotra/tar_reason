# Continuing training across allocations

Resume requires the adapter, `state.json`, `optimizer.pt` and `scheduler.pt`.
The trainer restores the optimizer and original cosine schedule; `--max_steps`
is the total target, so resuming step 200 with `--max_steps 500` adds 300 updates.
Method flags must match the saved checkpoint. Use a copied checkpoint in a new
output directory when the original run must remain intact.

`--stop_at_unix` and `--stop_file` request a coordinated stop at an optimizer
boundary. Complete checkpoints are published atomically, and a final
`training_status_<job>.json` records the saved step and target. The prepared Fir
continuation wrapper stops 45 minutes before each 24-hour allocation expires.
After a successful segment, `resubmit_until_complete.py` submits at most one
successor with an `afterok` dependency on its parent. It stops at the target or
on errors/no progress; ambiguous submissions require checking Slurm before retrying.

The current concrete script is
`results/launches/rl_continue_corrected_lr1e4_ck200_to500_20260916/resume.sbatch`.
Its `launch.json` records the initial job; successor ledgers and per-job training
status files are under the corresponding `results/models/` run directory.

# Full GenEval2 evaluation

`eval_geneval2.py` evaluates a frozen RL adapter on all **800 official GenEval2
prompts**, using the exact SFT base recorded in its adapter configuration. It
samples one complete episode per prompt, renders its final complete image once,
and scores those pixels with the existing frozen Qwen3-VL judge. There are no
optimizer updates. All models must use the same seed, generation batch size,
GPU count, backend and sampling settings for a comparable run.

To evaluate a full SFT checkpoint directly, pass `--checkpoint_type sft`
(`CHECKPOINT_TYPE=sft` for either evaluation launcher). The default remains
`rl_adapter`, which requires valid final-pixel RL metadata. SFT mode loads no
adapter and hashes the full weight shards, model configuration and tokenizer
files into the evaluation manifest. Both modes freeze the policy and use the
same episode/rendering protocol. SFT evaluation can run independently while
RL trains a new adapter from that checkpoint.

The September 15 comparison uses benchmark revision
`a6e82d2289e8d418f27f0adee77908b07060eea3` from
[the official repository](https://github.com/facebookresearch/GenEval2).
Its JSONL SHA256 is
`09233adbc9f877fba205f1a20efa4a8640e7e62a91dfd55893562bed8a928e43`.
No benchmark prompt exactly matches a prompt in the RL training set.

Each prompt saves its exact scored PNG, question scores, reflections, semantic
codes, sampled sequence and stop reason. Complete runs write `metrics.json`,
`score_lists.json` in benchmark order, and `image_filepath_data.json`. Metrics
include mean prompt AM/GM, pooled per-skill AM and per-atomicity AM/GM. Online W&B
receives aggregate scores and eight final-image exemplars at completion.

Manifest checks prevent mixing checkpoints, benchmark versions, environments,
backends or sampling settings when resuming. Re-submit the **same** job script
to retain verified prompt results and regenerate incomplete batches with their
original seed. A partial run is never presented as a full benchmark score.

The general launcher is `scripts/rl_ft/eval_geneval2.sh`; set `RL_CHECKPOINT`,
`GENEVAL2_BENCHMARK`, `EVAL_CODE_ROOT` (a source snapshot with `manifest.json`),
and a unique `RUN_NAME`, then submit with `JOB_TIME=02:59:00`. Defaults retain
the training environment and FlashAttention 2. `EVAL_PYTHON` and
`ATTN_IMPLEMENTATION` explicitly select another tested evaluation setup.

For the current Fir comparison, source/environment I/O failures in `/home`
required a verified scratch snapshot and the healthy scratch Python 3.11,
PyTorch 2.6 / Transformers 4.57.6 environment with SDPA and isolated extra
dependencies. The original training and judge environments were not modified.
The renderer's EasyDict checkpoint configuration is explicitly allowlisted for
PyTorch's weights-only loader. This backend may produce different sampled pixels
from historical FlashAttention evaluations; it is identical across these four
new evaluations and recorded in every manifest.

The concrete Fir batch scripts and checkpoint mapping are under
`results/evaluations/geneval2_full_20260915/`. Each GPU script requests four
H100s, 24 CPUs, 256 GB RAM and **02:59:00**. `comparison.json` records job IDs;
a dependent CPU report job writes `RESULTS.md` and `results.json` there.
Per-model results are in the `output_dir` recorded for that model.

# RL method flags

Two independent opt-in settings preserve the previous defaults:

| CLI | Launcher environment | Default |
|---|---|---|
| `--length_normalization episode\|constant` | `LENGTH_NORMALIZATION=episode\|constant` | `episode` |
| `--kl_gradient_correction` / `--no-kl_gradient_correction` | `KL_GRADIENT_CORRECTION=1\|0` (also `true\|false`) | `0` |

In `episode` mode, sampled token weights sum to one per episode. In `constant`
mode, image-token weights are `1/MAX_SEQ_LEN` and text-token weights are
`REFLECT_TOKEN_WEIGHT/MAX_SEQ_LEN`, then losses are averaged across episodes.
The denominator is the configured fixed `MAX_SEQ_LEN` (default 4096), not
sampled length, padded batch width, or remaining context. Forced tokens and
padding have zero weight; sampled EOS and transition tokens remain trainable.
This removes episode-length normalization bias; nonunit reflection weights
still deliberately change the relative weighting of text and image actions.
Both PG and KL use the selected normalization, which changes gradient scale.
Learning rate is not automatically adjusted.

KL correction multiplies K3 by the differentiable, unclipped
`exp(logp - old_logp)`. Its derivative matters even when the ratio equals one
in the first PPO epoch. Reward, sampling, group advantages and trainable
parameters are unchanged. `pg_loss` and `kl` report objective-normalized terms
(`loss = pg_loss + KL_COEF * kl`, up to rounding); ratio diagnostics remain
weighted means. Constant normalization no longer forces centered PG loss to
cancel to zero. The previously documented per-modality metric aggregation
limitation remains; these are not exact modality loss contributions.

Enable both for a **fresh experiment** on Fir:

```bash
LENGTH_NORMALIZATION=constant KL_GRADIENT_CORRECTION=1 \
RUN_NAME=rl_pixel_t20_constant_corrected_kl \
  bash scripts/clusters/fir/submit_rl.sh --job-name=tar-rl
```

For a preview, set the same environment variables and run
`TAR_DRY_RUN=1 bash scripts/rl_ft/bash.sh` instead. W&B config and new
`state.json` checkpoint metadata record both flags; constant mode also records
its fixed denominator. Resume requires matching method settings. Old pixel
checkpoints without this metadata resume with legacy defaults. A method change
requires a fresh run name/output directory, which starts from the chosen SFT
checkpoint; the flags do not implicitly migrate old adapters or optimizer state.

# Tree credit and reflection-text policy gradient

Two further opt-ins; empty/`all` defaults run the unchanged episode code path.

| CLI | Launcher environment | Default |
|---|---|---|
| `--tree_branch "R[xM],..."` | `TREE_BRANCH` | empty (independent episodes) |
| `--max_tree_leaves_per_prompt N` | `MAX_TREE_LEAVES_PER_PROMPT` | `64` |
| `--text_policy_gradient all\|transitions` | `TEXT_POLICY_GRADIENT` | `all` |

**Tree rollouts.** Entry k applies after image k (image 0 is the draft): sample R
reflections per image, and M images for each reflection that samples
`<im_start>` and passes the caps. Unlisted rounds do not branch, so depth is
still set by `MAX_REFINEMENTS` and sampled EOS. `NUM_ROLLOUTS` drafts per prompt;
at most `NUM_ROLLOUTS x prod(R*M)` leaves, fewer when branches stop early. With
`NUM_ROLLOUTS=4`: `2` -> 8, `2,2` -> 16, `2,2,2` -> 32, `2x2` -> 16. More rounds
than `MAX_REFINEMENTS` are rejected (the last reflection can only stop).

Only final images are rendered and scored, once per distinct final image (a
child that emits EOS reuses its parent's exact pixels). value(leaf) = final
reward, value(node) = mean of its children; advantage = value minus the mean of
its siblings (drafts: the prompt's drafts). An only child shares its nearest
branch point's advantage, so unbranched trees reproduce episode advantages.
`ADV_NORM=std` divides a prompt's advantages by the mass-weighted std of its leaf
rewards. Drafts have mass 1 and children split their parent's mass; token
weights are kind weight x sum over leaves under the node of mass/episode length
(`episode`) or kind weight x mass / `MAX_SEQ_LEN` (`constant`). Weighting by
descendant counts would bias the stop/refine gradient (a CPU exact-expectation
test checks this). Each node's actions train once; the denominator counts drafts.
Sibling means include the node itself, shrinking each level by (K-1)/K as in
standard GRPO. Validation and benchmark evaluation still use single chains.
New `train/tree_*` metrics report leaves/nodes per prompt, mean |advantage| of
drafts, reflections and refined images, and the mean advantage of stopping vs
continuing reflections that have siblings.

**Text policy gradient.** `transitions` zeroes the policy-gradient term of
reflection prose (sampled text other than EOS/`<im_start>`), keeping its KL
term and its normalization weight; EOS, `<im_start>` and image tokens keep their
policy gradient. This is a deliberate partial gradient: prose is still sampled
from the shared LoRA policy and anchored only by KL.

Main-branch-like first run (4 GPUs; 8 leaves per prompt, 16 per rank):

```bash
NUM_ROLLOUTS=4 TREE_BRANCH=2 PROMPTS_PER_GPU=2 RESET_CONTEXT_PER_TURN=1 \
TEXT_POLICY_GRADIENT=transitions RUN_NAME=rl_tree2_transitions \
  bash scripts/clusters/fir/submit_rl.sh
```

Unlike main, leaves continue as chains up to `MAX_REFINEMENTS` and stop on
sampled EOS. Both settings are checkpoint method metadata; old checkpoints mean
`[]` / `all`, and mismatches refuse to resume.

# One-iteration Janus-style objective

`--objective janus_one_iteration --max_refinements 1` (`OBJECTIVE`, `VERDICT_RULE`,
`ADV_STD_FLOOR` in the launcher): draft -> reflection -> one refinement unless the
Correction says "looks good". Rewards, each group-normalized (unbiased std):

| Segment (tokens) | Reward |
|---|---|
| draft image | AM(draft) |
| reflection text (incl. stop token) | `all_pass` (default): 1[looks good == every draft question >= 0.5]; `soft_am`: 1 - abs(AM(draft) - 1[looks good]) |
| final image (refined image, or the draft again when kept) | AM(final) |

Loss = mean over the three segments of per-sequence token means (KL in each), as
Janus-Pro-R1 `grpo_trainer_t2iv1.py`. Differences: exact log-probs (T=1, no CFG),
sampled draft tokens kept in context, optional PPO epochs, corrected KL. If the
reflection did not end with a sampled `<im_start>` but must refine, a trailing EOS is
dropped and `<im_start><S0>` spliced (newline-aware); a stop token that disagrees with
the verdict keeps KL but no PG. Stats: `train/val am_draft, am_final, looks_good_rate,
delta_am_refined, looks_good_on_pass/fail, mismatch_*_rate, rule_disagreement_rate`,
per-segment `pg_/kl_{draft,reflection,final}`. Cost: every draft and refined image is
rendered and judged (about 1.8 per episode); use `RENDER_BATCH_SIZE=24` (12.6 -> 1.07 s/image
in `check_batched_reward.py`, job 64727589). Keep `REWARD_QUESTION_BATCH_SIZE=1`: batched judging
saved only 0.28 -> 0.20 s/image and moved per-question scores by up to 0.108 (left-padded bf16). The
`soft_am` target collapsed in run `..._janus1iter_g8p8_..._20261012`: "looks good" rose
from 22% to 92% (90% on failing drafts) in 37 steps, because it pays whenever soft AM > 0.5.

# Benchmark generation protocols

`eval_geneval2` / `generate_instruction_benchmarks` default to `--protocol
native` (the trained EOS policy: temperature 1, no truncation, up to
`--max_refinements` 3, context mode from checkpoint metadata).
`--protocol main_adhoc --max_refinements 1` reproduces the main branch's
`eval/iterative_generation_adhoc.py` generation: one forced refinement unless the
greedy reflection says "looks good", image top-k 1200 / top-p 0.95, reflection
re-encoded as text, full history, LoRA merged. Use it for cross-branch numbers;
judges and scoring are shared. Outputs record the protocol in their manifest.

# Fixed periodic validation subset

`val.yaml` selects `val32.jsonl`: 32 unchanged rows from the original validation
set. Eight rows each have 2–3, 4–5, 6–7 and 8–10 compositional atoms. These are
complexity proxies, not measured difficulty. Selection favors skill and question
word/count diversity with deterministic hash tie-breaking; no rewards are used.
This balanced diagnostic subset is not an unbiased estimate of the full-set mean.

Four-GPU stride partitioning gives each rank two prompts from each bin, including
rank 0's eight saved exemplars. Other world sizes retain the same global subset.

Regenerate with `python3 scripts/rl_ft/select_val32.py` from the repo root.
Source SHA256: `beb1b6eed5b855fa587f68adb545773c7c114ddde55e33ada9ad0716dba33d98`.

Use `VAL_DATA_PATH=scripts/rl_ft/val256.yaml` for full-set comparisons. The previous
step-0 baseline used 256 prompts; establish a matching 32-prompt baseline before
comparing scores. Sampling, reward, episode limits and interval are unchanged.

| Index | Atoms | Prompt |
|---|---|---|
| 0 | 3 | an elephant in front of a candle |
| 1 | 3 | a monkey playing with a flamingo |
| 2 | 3 | seven red motorcycles |
| 3 | 3 | a monkey jumping over a penguin |
| 4 | 3 | a croissant behind a trumpet |
| 5 | 2 | a wooden monkey |
| 6 | 3 | six brown trumpets |
| 7 | 3 | a dog under a rabbit |
| 8 | 5 | a purple chair on top of a striped bicycle |
| 9 | 5 | a white zebra jumping over a sparkling penguin |
| 10 | 5 | four brown bicycles to the left of an umbrella |
| 11 | 5 | three pink elephants playing with a flamingo |
| 12 | 5 | a checkered flower in front of six giraffes |
| 13 | 5 | five cows jumping over a black elephant |
| 14 | 5 | a blue mushroom to the left of a spotted horse |
| 15 | 4 | a plastic kangaroo chasing a rabbit |
| 16 | 6 | four white chairs to the left of a stone flamingo |
| 17 | 7 | two glass monkeys, and a purple koala jumping over a raccoon |
| 18 | 6 | a yellow donut on top of three plastic toys |
| 19 | 7 | a brown monkey chasing four blue bears, and a car |
| 20 | 7 | seven sparkling mushrooms on top of five pink backpacks |
| 21 | 7 | four horses, and a zebra playing with six blue kangaroos |
| 22 | 7 | a white raccoon to the right of a bicycle, and three red trucks |
| 23 | 6 | a striped turtle chasing four purple cows |
| 24 | 9 | a striped zebra, and two brown bagels on top of seven blue kangaroos |
| 25 | 9 | a stone horse, and six striped zebras playing with five blue flamingos |
| 26 | 10 | six green cows to the right of three brown trucks, and four black chairs |
| 27 | 10 | five red flowers, and seven glass sheeps chasing six black dogs |
| 28 | 9 | two white turtles on top of three blue suitcases, and a wooden cat |
| 29 | 8 | a sparkling monkey chasing four sparkling turtles, and three backpacks |
| 30 | 8 | a brown bird in front of five trucks, and two yellow cars |
| 31 | 8 | a pink flamingo behind seven plastic croissants, and three giraffes |

# Teacher-guided tree (`teacher_tree`)

`OBJECTIVE=teacher_tree TREE_BRANCH=1x2 MAX_REFINEMENTS=3 RESET_CONTEXT_PER_TURN=1 ADV_NORM=std`
(plus `SHADOW_REFLECTIONS`, `SHADOW_COEF`, `TEACHER_CE_COEF`, `TEACHER_MAX_TOKENS`,
`TEACHER_BATCH_SIZE`). Each rollout round samples images from the policy, renders them, and
asks the judge model (Qwen3-VL-8B, same worker process; image + prompt only) for a reflection
(`llava/train/rl/teacher.py`). The reply is rewritten into the policy's reflection format and
spliced: "looks good" stops the episode; a correction is followed by M policy images
(`TREE_BRANCH` entries `1xM`; R must be 1), until `MAX_REFINEMENTS`.

| Tokens | Credit | Loss |
|---|---|---|
| draft / refined images | tree value = mean final AM of the subtree, minus siblings' mean | PG + KL, episode weights per draft subtree |
| teacher reflection (incl. EOS / `<im_start>`) | none (not a policy sample) | `TEACHER_CE_COEF` x mean per-token NLL |
| shadow reflections (`SHADOW_REFLECTIONS` per image, not continued) | 1[verdict == all judge questions >= 0.5], prompt groups, unbiased std | `SHADOW_COEF` x mean of per-shadow token means, PG + KL |

Stats (`train/`): `am_draft`, `am_final`, `delta_am_refined`, `stop_*_rate`,
`teacher_verdict_accuracy/precision/recall`, `teacher_looks_good_on_fail`,
`teacher_malformed_reply_rate`, `shadow_verdict_accuracy`, `shadow_looks_good_on_pass/fail`,
`shadow_mismatch_rate`, `ce_loss`, `teacher_nll`, `pg_/kl_{draft,shadow,refined}`, and
`time/teacher_render`, `time/teacher_generate` (both inside `time/rollout`). Validation runs
the teacher-free looks_good protocol (the policy's own reflections; `val/am_upto_k`,
`verdict_accuracy`, `looks_good_on_fail`). `check_teacher_reflections.py` checks the teacher's
format compliance and agreement with the judge on saved SFT drafts before a run.
