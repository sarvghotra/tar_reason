# Full GenEval2 evaluation

`eval_geneval2.py` evaluates a frozen RL adapter on all **800 official GenEval2
prompts**, using the exact SFT base recorded in its adapter configuration. It
samples one complete episode per prompt, renders its final complete image once,
and scores those pixels with the existing frozen Qwen3-VL judge. There are no
optimizer updates. All models must use the same seed, generation batch size,
GPU count, backend and sampling settings for a comparable run.

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
