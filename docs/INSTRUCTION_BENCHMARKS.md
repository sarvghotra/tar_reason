# TIIF-Bench and GenAI-Bench RL evaluation

The implementation adapts the benchmark utilities in upstream Tar `main`
(`28c823c`) to the trained RL policy. It loads the LoRA checkpoint's recorded
SFT base and uses the existing EOS-driven `EpisodeRollout`: one independent
episode per prompt, at most three refinements, 128 reflection tokens, context
limit 4096, unrestricted temperature-1 sampling, and CFG-4 rendering at 512px.
Only the final complete image is rendered. No optimizer or reward-based image
selection runs. This differs from upstream's forced draft/reflection/revision
generation; generation provenance is recorded explicitly.

## Benchmarks and judges

- **TIIF-Bench testmini:** 277 cases, each with short and long wording, for
  **554 final images** across all 39 dimensions. Generation and evaluation
  JSONL files are checked for matching file sets, dimension, and row count.
  Source: [official TIIF-Bench](https://github.com/A113N-W3I/TIIF-Bench), revision
  `cca586f40f057c1590d819ae3807ea4e27620e84`.
- **TIIF judge:** Qwen2.5-VL-7B-Instruct, using the upstream Tar local judge's
  exact question templates and yes/no parsing. Greedy first pass, four sampled
  retries, then the same single-question fallback. This wrapper uses batch size
  one and a deterministic seed per image. It retains raw outputs and retries.
  Unparseable results fail rather than silently dropping questions. Scores
  measure agreement with ground-truth yes/no answers, including negative ones.
- **TIIF aggregation:** question-weighted accuracy within each dimension;
  dimension means within the nine official groups; equal-weight mean across
  those nine groups. Short and long prompts remain separate. Basic, advanced,
  real-world, group, and dimension scores are saved. Computation keeps full
  precision; the official Excel workflow rounds dimensions to four decimals
  before grouping and may therefore differ slightly. `eval_results/final/`
  also preserves the schema accepted by the official Excel scripts.
- **GenAI-Bench:** the full official **1,600 image prompts** from
  `BaiqiL/GenAI-Bench-1600`. Upstream Tar references a private/local
  `GenAI-Image-800` folder, which is absent from its repository. The official
  `genai_bench.evaluate` supports 527/1600, so no arbitrary 800-row subset was
  substituted. The user was informed of this distinction.
- **GenAI judge:** Qwen3.5-27B, as selected in upstream Tar. Uses the unmodified
  Qwen evaluator from [t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) at
  `6ecb74f92028f42c7e64546d3a71e98c8c73068f`: probability of `Yes` for
  `Does this figure show "<prompt>"? Please answer Yes or No.`, greedy one-token
  generation, thinking disabled. Overall is the union of tagged IDs; skill
  membership is joined by ID rather than assuming IDs are array positions.

These scores depend on the specified judges. They are not interchangeable
with published results obtained using GPT-4o or a different VQAScore model.
Images remain lossless PNGs for both benchmarks; unlike upstream GenAI's JPEG
export, no lossy re-encoding changes the evaluated pixels.

## Entry points

1. `scripts/prepare_instruction_benchmarks.py` downloads judge snapshots and
   official GenAI prompt/skill files. It pins the resolved HF revision, checks
   LFS SHA256 and sizes, and writes file-level provenance. Existing training
   environments are not modified.
2. `python -m llava.train.rl.instruction_eval_io tiif|genai` prepares validated
   benchmark JSON. Pass `--prompts`, `--questions` (TIIF) or `--skills` (GenAI),
   and `--output`.
3. `scripts/rl_ft/eval_instruction_benchmarks.sh` runs `STAGE=generate|score|all`
   in an allocation. It requires explicit checkpoint, benchmark, output, judge,
   renderer, Python and dependency paths. The code snapshot must have a
   `manifest.json` recording its source hashes.
4. `python -m llava.train.rl.instruction_eval_io summary --output <run>` verifies
   all records and computes metrics. `--allow_partial` writes explicitly
   incomplete results. Complete metrics require every expected image/score.
5. `scripts/rl_ft/report_instruction_benchmarks.py <launch_dir>` records scheduler
   states, coverage and verified full scores in `RESULTS.md` and `results.json`.

Generation saves image hashes, prompts, semantic codes, sampled token sequences,
reflections and stop reasons. Generation/scoring have separate manifests;
incompatible settings, changed data, changed images and malformed scores fail
resumption. A resumed generation batch uses its original random seed and keeps
already saved records. Scoring can restart without rerendering.

## September 16, 2026 submission

Requested model: `ge2full_corrected_lr1e4_ck200_20260915`, resolved to
`results/models/rl_pixel_t20_val32_p4_const_kl_lr1e4_20260911_170923/checkpoint-200`.
One sample per prompt, seed 421, matching the previous GenEval2 generation
settings. This is a single-seed evaluation, not upstream's three-seed average.

Artifacts: `results/evaluations/instruction_benchmarks_20260916/`.
This includes the immutable code snapshot, prepared inputs, launch metadata,
all batch scripts and the final report. Logs are under
`results/logs/instruction_benchmarks_20260916/`.

| Job | Purpose | Dependency |
|---|---|---|
| 60094710 | 1-H100 smoke: generate and score 2 TIIF + 2 GenAI images | none |
| 60094722 | 4-H100 TIIF generation | none (smoke dependency removed by user request) |
| 60094723 | 4-H100 GenAI generation | none (smoke dependency removed by user request) |
| 60094727 | 4-H100 TIIF scoring | TIIF generation succeeds |
| 60094728 | 4-H100 GenAI scoring | GenAI generation succeeds |
| 60095067 | CPU completion report | both scoring jobs terminate |

Smoke limit: 00:59:00. Each full GPU stage: 02:59:00. All jobs were pending at
the last submission check; no new GPU results were available. Dependency
failures cancel downstream GPU jobs rather than running invalid evaluations.
For timeout recovery, re-submit the same stage script and update downstream
dependencies; do not change the submitted code/manifest in place.

Per-benchmark outputs:
`results/evaluations/tiif_corrected_lr1e4_ck200_20260916/` and
`results/evaluations/genai_corrected_lr1e4_ck200_20260916/`.

Generation uses the verified scratch Python 3.11 / Torch 2.6 / Transformers
4.57.6 stack and earlier recovered renderer source. TIIF uses that same judge
stack. GenAI prepends isolated `instruction_judge_deps` (Transformers 5.17.0,
tokenizers 0.23.1, HF Hub 1.7.1, qwen-vl-utils 0.0.14) to support Qwen3.5.
Only the four unmodified t2v_metrics Qwen/base modules are vendored, with license
and checksums, avoiding unrelated API/audio/model imports.

Validation: seven CPU tests cover alignment, noncontiguous IDs, generation and
score integrity, incomplete coverage, full durable aggregation, and TIIF's
question weighting / nine-group average. Python/shell syntax and real
generator/renderer/TIIF imports passed. Qwen3.5 model imports, local processor
loading and the single `Yes` token (9175) passed. GPU validation remains pending
until the smoke job actually completes successfully.
