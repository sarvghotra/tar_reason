# Agent handoff — September 11, 2026

## September 16: TIIF / GenAI-Bench implementation and submission

Completed results verified subsequently: TIIF generation/scoring
**60094722/60094727** and GenAI generation/scoring **60094723/60094728** all
COMPLETED with exit 0:0. TIIF covered 554/554 images: nine-group overall
0.6135394113 (short), 0.5970945121 (long). GenAI covered 1600/1600 images:
VQAScore overall 0.3222131458, basic 0.3526821729, advanced 0.2969563863.
Exact metrics are in each benchmark's `metrics.json`. These supersede the
submission-time pending status below. Source, tests, configuration and docs
are being published to `origin/xiaofeng`; environments, checkpoints, dataset
links, credentials and generated run outputs remain excluded.

User subsequently requested bypassing the smoke dependency. At 10:32:39
cluster time, `scontrol update ... Dependency=` cleared dependencies on
**60094722** and **60094723**; both were verified eligible with
`Dependency=(null)`. Scoring retains its necessary generation dependency, and
the final report retains its scoring dependencies. Smoke 60094710 remains
independently queued and no longer blocks either full evaluation.

See [INSTRUCTION_BENCHMARKS.md](INSTRUCTION_BENCHMARKS.md) for the complete
protocol, entry points, validation, dependencies and output paths. Fetched
upstream `main` at `28c823c`; adapted its TIIF local Qwen2.5-VL judge and the
official t2v_metrics Qwen3.5-27B VQAScore to the existing EOS-driven RL policy.
The requested `ge2full_corrected_lr1e4_ck200_20260915` resolves to the corrected
LR=1e-4 run's checkpoint-200, with the adapter's recorded SFT base.

TIIF: all 277 testmini cases in both registers (554 images). GenAI: the full
official 1,600 image prompts; the author's referenced local 800-prompt split
was not published with the repo. User was asked about that split and informed
that 1,600 would be used absent a supplied path. One sample/prompt, seed 421.
Generation and scoring are resumable, image/manifest integrity checked, and
incomplete runs are never presented as full benchmark results.

Submitted smoke **60094710**, TIIF generation/scoring **60094722/60094727**,
GenAI generation/scoring **60094723/60094728**, final CPU report **60095067**.
Full generation requires a successful smoke; scoring requires successful
generation. Last check: smoke PENDING/Priority with no start estimate, all
others PENDING/Dependency. GPU runtime validation and evaluation scores remain
outstanding; re-query Slurm and inspect both stdout/stderr before claiming
success. Limits: 59-minute one-H100 smoke, 02:59 for each four-H100 full stage.

Immutable launch snapshot and inputs:
`results/evaluations/instruction_benchmarks_20260916/`.
Seven new CPU checks and Python/shell syntax passed; actual imports and the
local Qwen3.5 processor passed. Both judge downloads were SHA256-verified.
Training environments are unchanged; GenAI uses isolated Transformers 5.17.0
dependencies. Do not overwrite submitted snapshots or previous evaluation
artifacts. No upstream training changes were merged into the local RL code.

## September 15: full GenEval2 comparison submitted

The user requested all trained RL models' latest saved checkpoints on the full
official benchmark, then changed each job's limit to **02:59:00**. Submitted:

| GPU job | Run | Latest saved checkpoint |
|---|---|---|
| 59949648 | rl_pixel_sft_t20_16k_20260911 | checkpoint-25 |
| 59949649 | rl_pixel_sft_t20_val32_p4_20260911 | checkpoint-200 |
| 59949650 | rl_pixel_sft_t20_val32_p4_const_kl_20260911_134347 | checkpoint-200 |
| 59949651 | rl_pixel_t20_val32_p4_const_kl_lr1e4_20260911_170923 | checkpoint-200 |

Each requests one node, four H100s, 24 CPUs, 256 GB RAM, online W&B and 02:59:00.
CPU report job **59949652** uses `afterany` on all four jobs and writes a
comparison even if some evaluations fail or time out. The last scheduler check
on September 15 around 15:04 UTC showed all four evaluations RUNNING and the
report waiting on its dependency. Each evaluation had saved 131–136 of 800
prompt results. All four connected to online W&B; no runtime errors were found.
The first allocation's live GPUs used approximately 39–43 GiB each at about
60% utilization. These are partial evaluations, not final benchmark results.

At the user's request, CPU monitor job **59951777** now polls accounting,
saved-prompt counts and log errors every 60 seconds. It was confirmed RUNNING
with repeated successful updates and empty stderr. Its script, `MONITOR.md`,
`monitor_status.json` and append-only `monitor_history.jsonl` are beside
`comparison.json`; monitor logs share the evaluation log directory. It stops
when all four evaluations and the report are terminal, or at its 02:59:00
limit. It records status only; it does not restart jobs or send messages.

`results/evaluations/geneval2_full_20260915/comparison.json` records exact paths,
job IDs and the source snapshot; the same directory contains all five concrete
`.sbatch` files and the report `RESULTS.md` / `results.json`. Per-model output
directories are specified by `output_dir`; logs are under
`results/logs/geneval2_full_20260915/`. Do not overwrite the submitted snapshot.

The new `llava/train/rl/eval_geneval2.py` uses the production episode policy,
one episode per official prompt, final-image-only rendering/scoring and no
optimizer. It loads each adapter's recorded SFT base, records environment and
sampling provenance, saves exact scored PNGs and question/trajectory records,
and resumes only compatible, integrity-checked results. Full results require
all 800 prompts. AM/GM, per-skill AM and per-atomicity metrics are saved; W&B
receives aggregates and eight final-image exemplars at completion. The official
benchmark revision and SHA256 are documented in `scripts/rl_ft/README.md`.
Exact prompt overlap with the RL training set was zero.

Operational workaround: `/home` returned I/O/transport errors for the renderer
source, `scripts/submit.sh`, and modules in `tar/` (Torch import failed). The
original files were left untouched. The scratch code snapshot contains a
recovered `tok/ar_dtok/ar_model.py` whose SHA256 matches the earlier review
manifest (`89a443f4fc987ce873ac742ebfe337a15cd7c89b9183c26e2c795b2303f4a8b6`).
Jobs run entirely from that snapshot using the healthy scratch Qwen Python
3.11 / Torch 2.6 / Transformers 4.57.6 environment, SDPA for the first AR,
and isolated PEFT/einops/EasyDict/W&B dependencies; no existing environment was
modified. W&B dependencies also use the healthy scratch `wandb_sync` site-packages.
The judge still runs as a separate frozen process. Backend changes can change
sampled pixels versus historical evaluations, but all four new jobs match.
EasyDict is explicitly allowlisted for the renderer's weights-only load.

Validation: all 44 CPU RL tests passed in this isolated setup (42 initially;
the two distributed tests passed after rerunning through `python -m unittest`
instead of stdin, which multiprocessing cannot restart). Two extra SDPA
probability/EOS checks passed. All three renderer checkpoint structures loaded
with weights-only + mmap/meta tensors; this checks compatibility without
running the models. Imports, seven new aggregation/resumption/integrity tests,
Python/shell syntax, latest-checkpoint selection and adapter file structure
checks passed. The running evaluations have now completed and saved real
GPU-generated, Qwen-scored images; full 800-prompt completion remains outstanding.

Re-submit the same per-model `.sbatch` file to resume an interrupted evaluation;
update comparison job IDs and submit a new dependent report job as needed.
Never present partial prompt scores as the full benchmark.

## Implemented

- Python/uv cluster setup with a separate Qwen judge environment.
- Independent EOS-driven episodes and whole-episode GRPO credit.
- Frozen Qwen3-VL-8B-Instruct soft-TIFA AM reward on the final rendered image.
- Exact scored-image reuse in saved/W&B evaluation exemplars.
- Step-0 and periodic validation, persistent result files and checkpoint-objective guards.
- Policy initialization synchronization, matching sampler/loss distributions,
  and zero-dropout validation.
- Portable repo-local links, a configured Fir folder and a generic placeholder.

The complete filesystem and submission instructions are in
[`scripts/clusters/README.md`](../scripts/clusters/README.md). No settings for
the other cluster have been guessed; configure the generic placeholder there.

## Validation completed

CPU regression suites have covered final reward credit, EOS versus padding,
independent multi-round episodes, truncation, restricted log-probabilities,
LoRA synchronization and resume, microbatch consistency, pixel-judge transport,
GenEval2 scoring, exact final-image logging, and monitor duplicate prevention.
Four symlink safety tests and cluster submission dry runs passed after the
portable layout changes. Shell/Python syntax checks passed for those changes.

These results were obtained in separate runs during development, not a fresh
all-tests pass on the final tree. Full-model GPU inference and RL training have
not yet been verified. The real GPU smoke test is
`scripts/test_qwen_reward_gpu.py`, submitted through
`scripts/test_qwen_reward_batch.sh`.

## Fir model and environment

Qwen3-VL-8B-Instruct was downloaded from the official HF repository at revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. Four complete safetensors shards and
750 indexed tensors passed structural checks. The isolated judge environment
has Python 3.11, PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124 and Transformers
4.57.6. Imports, processor loading, image preprocessing and dependency checks
passed. The full package freeze and model provenance are under `results/`.

Tar uses its existing Python 3.10 environment with PyTorch 2.1.2+cu121 and
Transformers 4.50.0. Do not upgrade it implicitly to install the Qwen judge.

## Outstanding issues / operational context

- Fir has recently produced filesystem I/O/transport errors. The final launcher
  dry run failed while Python read a `.pth` file in the existing `tar` environment.
  This is unresolved and is distinct from the symlink/submission checks.
- `tts/eval/reward.sh` and `scripts/download_qwen_reward.py` became unreadable
  during branch preparation. The scratch recovery checkout reconstructs the download helper from its exact
  known session contents; the original file still needs filesystem recovery. The
  unreadable local changes to `tts/eval/reward.sh` cannot be recovered confidently,
  so its previously committed version is retained. Neither unreadable working
  file was overwritten. Preserve those files for recovery.
- Some legacy TTS evaluation scripts import helpers absent from the current
  `grpo.py`. They are not the current trainer's episode/evaluation path.
- The old interactive allocation 59153811 was cancelled before allocation;
  its trigger 59161662 completed without running the test. Cron was installed
  on a login host but scheduled execution was never verified. Login-host cron
  state is not portable.
- Standalone GPU smoke job 59256440 was submitted afterward; its last observed
  status was pending. Re-query Slurm rather than treating this note as live status.
- Full RL defaults to four full H100 80 GB GPUs, with policy/renderer/judge
  replicas per rank. GPU memory/runtime and full training remain unverified.
- The legacy final-checkpoint behavior with `save_steps=0` was identified earlier
  but was not changed as part of the reward/cluster work.

## Conditional full training submission

Full training job `59261866` (`tar-rl-t20`) was submitted with
`afterok:59256440` and `--kill-on-invalid-dep=yes`. At submission, the smoke job
was still running; the Qwen color/repeatability test had passed, but the complete
integration test had not yet passed. The batch script also requires its final
integration-pass log message before switching the repo's `sft_model` symlink.

The new checkpoint is
`/project/rrg-bengioy-ad/jeet/models/sft_ckpt/slf_ref_edit_t20/checkpoint-16000`.
Its configuration matches the prior architecture; JSON and all four indexed
safetensors shard headers/sizes passed checks. This does not establish inference
correctness for the new checkpoint. The login-node launcher dry run still hit
the known Python `.pth` I/O error.

Job settings: four H100 80 GB GPUs, 24 CPUs, 256 GB host RAM, three-hour limit,
500 training steps, evaluation/checkpoint every 25 steps, run name
`rl_pixel_sft_t20_16k_20260911`. Outputs remain under `results/`. The exact batch
script is `results/launches/rl_pixel_sft_t20_16k_20260911.sh`. It pins the new SFT
path independently of later symlink changes. W&B retains the configured default
(offline unless overridden). Re-query Slurm for live status.

Do not claim a successful GPU test based only on a completed trigger job or an
empty error log.

### Verified startup update

Smoke job `59256440` completed with exit `0:0` after 16m40s. Its saved
`results/evaluations/qwen_reward_smoke/integration_smoke.json` identifies that
job and records final-image rewards approximately 0.500277 and 1.0, with
advantages -0.249862 and +0.249862. The final integration-pass message is present.
This smoke test used the previous SFT checkpoint and performed no optimizer updates.

Full job `59261866` subsequently started on `fc10105`. Its startup log confirms
four GPUs, the new `slf_ref_edit_t20/checkpoint-16000`, and successful input/shard
permission checks. The repo's `sft_model` link now points to that checkpoint.
At this check, no optimizer step was yet reported and Slurm stderr was empty;
full distributed training correctness remains to be verified from later logs.

### W&B default for future runs

At the user's request, `scripts/cluster_env.sh` now defaults `WANDB_MODE` to
`online`, while preserving explicit overrides. Job 59261866 was confirmed
offline from its process environment and was not restarted or modified.

### Full run outcome: 59261866

Slurm reports TIMEOUT after 03:00:08, ending September 11 at 10:13:28
(cluster timestamp). The run logged 25 of the intended 500 updates and saved
`results/models/rl_pixel_sft_t20_16k_20260911/checkpoint-25`, including adapter,
optimizer, scheduler and objective/step metadata. Only step-0 validation is
complete (AM 0.6354); no step-25 validation metrics were saved before timeout.
There is therefore no matched before/after validation comparison yet.

Five-update training AM means were 0.64626, 0.65644, 0.64822, 0.65840 and
0.60286. This late dip uses changing prompt groups and does not establish
validation degradation. Across 25 updates AM ranged 0.5341–0.7547, mean 0.64244.
Average recorded rollout/reward/update time was 264.28 seconds per update.
The appropriate next diagnostic is fixed-set evaluation of checkpoint-25 with
the same policy/renderer/judge settings as step 0, before tuning hyperparameters.
No follow-up evaluation or resumed training has been submitted by this log review.

### Next-run batch preference

The user requested four prompts per GPU for the next run. Both the RL launcher
and direct trainer CLI now default to 4; four rollouts per prompt, generation
batch 16 and training microbatch 2 remain configured. On four GPUs this means
16 prompts and 64 episodes per update. This larger training batch has not yet
been GPU-tested or submitted. Evaluation settings remain unchanged.

### Fixed 32-prompt periodic validation

The user approved 32 varied prompts. `scripts/rl_ft/val.yaml` now uses
`val32.jsonl`: eight unchanged source rows per atom bin (2–3, 4–5, 6–7, 8–10),
with skill/question diversity. Each of four ranks gets two per bin; the eight
rank-0 exemplars span all bins. See `scripts/rl_ft/README.md` for provenance and
prompts. Reproducibility across processes, source-order invariance, exact row/VQA
preservation, uniqueness, rank balance and actual dataset loading passed. The
full set remains in `val256.yaml`. Its earlier baseline is not directly comparable
to the 32-prompt mean. No new job has been submitted.

### Next RL submission duration and logging

The user requested a seven-day limit and online W&B for the next run. Fir's
live sinfo advertises seven-day H100 partitions. Use
`RUN_NAME=<fresh-name> bash scripts/clusters/fir/submit_rl.sh --job-name=tar-rl`.
The new wrapper defaults JOB_TIME=7-00:00:00, WANDB_MODE=online and
PROMPTS_PER_GPU=4, preserving explicit overrides. Generic/smoke submission
defaults remain unchanged. MAX_STEPS is still 500, and default validation is the
fixed 32-prompt subset. No real job was submitted by this configuration change.

### Submitted next run: 59311752

User authorized submission. Job `59311752` (`tar-rl-t20-p4`) was submitted
with a seven-day limit, four H100s, 24 CPUs and 256 GB host RAM. Run name:
`rl_pixel_sft_t20_val32_p4_20260911`. Explicit submission environment pins online
W&B, four prompts/GPU, four rollouts/prompt, generation batch 16, train microbatch
2, 500 updates, evaluation/save every 25 updates and `scripts/rl_ft/val.yaml`
(the fixed 32 prompts). SFT_MODEL and PREV_STAGE_CHECKPOINT both explicitly point
to `slf_ref_edit_t20/checkpoint-16000`; this is a fresh SFT-initialized run,
not a resume of checkpoint-25. W&B credential presence was checked without
printing credentials; online initialization must still be verified from job logs.
Slurm logs: `results/logs/slurm/tar-rl-t20-p4-59311752.{out,err}`.

### Full RL review

See `docs/RL_REVIEW_2026-09-11.md` for the review findings and scope. The full
28-test RL suite passed in both the actual Tar environment and an alternate
scratch stack. Additional probes reproduced wrong-base resume, truncated
checkpoint-marker failure, missing final save with save_steps=0, empty-shard
collective timeout, microbatch-dependent per-modality metrics, and incomplete
worker-read timeout coverage. The existing W&B unequal-strip warning is also
recorded. Same-step W&B logging was checked and retains both train/val metrics.
No production implementation or pending job was changed. Review fixtures and
source hashes are under `results/reviews/rl-review-20260911/`.

### Method-focused RL review

See `docs/RL_METHOD_REVIEW_2026-09-11.md`. The rollout/grouping/clipping/update
path matches an outcome-supervised GRPO variant, but two mathematical concerns
were reproduced: per-episode length normalization can change the expected
final-reward gradient's direction, and direct differentiation of the K3 KL
estimator does not give the documented conditional KL(policy || reference)
gradient. The first follows the current explicit episode-normalization rule;
neither was silently changed. Six new CPU method probes passed, including a
202-action, 16-episode generation/logprob comparison with maximum error 2.38e-7
and old-policy snapshot persistence across optimizer epochs. Artifacts:
`results/reviews/rl-review-20260911/method_probes.{py,json}`. Full GPU BF16/FA2
likelihood parity and renderer-only reward variance remain unmeasured. These
findings do not establish the cause of the previous reward trend. No production
code or submitted job was changed.

### Opt-in GRPO method corrections implemented

User requested backward-compatible flags for both method findings. Trainer CLI
now accepts `--length_normalization episode|constant` (default `episode`) and
`--kl_gradient_correction` / `--no-kl_gradient_correction` (default off). Shared
launcher environment equivalents are `LENGTH_NORMALIZATION` and
`KL_GRADIENT_CORRECTION=1|0` (also true/false). Constant mode divides raw sampled
token weights by configured `max_seq_len`, then averages episodes; forced
positions remain masked. The KL option applies the differentiable, unclipped
policy/old-policy ratio to K3. See `scripts/rl_ft/README.md` for usage, gradient
scale and diagnostic semantics.

New checkpoints record `rl_method` with both settings and the fixed denominator
when applicable. The main resume path validates them before loading models.
Missing method metadata means legacy settings; old pixel checkpoints still
resume with defaults. Changing either setting (or a constant denominator)
requires a fresh experiment rather than silently resuming incompatible state.

Post-implementation review checked flags from shell through CLI, rollout weights,
loss gradients, accumulation, rank averaging, W&B config and checkpoint save /
resume. It caught and addressed the need for global weight normalization of
ratio diagnostics when ranks have different sampled lengths. No further new
blocking finding was identified. Known pre-existing modality-metric and other
operational findings in the earlier review remain outside this change.

Validation: all 37 CPU RL tests passed in the actual Tar environment, including
nine new method-flag tests. These cover bit-identical legacy loss/gradient,
all four combinations, EOS/padding/caps, the exact expected-reward direction
counterexample, on/off-policy KL gradients, microbatch invariance, a two-rank
corrected update against a combined batch with unequal rank token counts, and
checkpoint compatibility. Bash/Python syntax, diff whitespace checks, and
actual launcher previews for legacy/enabled settings passed. Log:
`results/reviews/rl-review-20260911/method_flags_tests.log`. Production GPU
training with enabled flags is not tested. Existing jobs were not restarted or
resubmitted; their default behavior remains legacy.

The same 37 tests also passed in the existing alternate Torch 2.6 / Transformers
4.57 stack with review-local PEFT (`method_flags_tests_alternate.log`). Neither
Python environment was modified.

### Submitted two-day corrected-method run: 59335407

User authorized a two-day job using both new corrections. Submitted job
`59335407` (`tar-rl-const-kl`) on September 11, 2026 at 13:44:18 cluster time.
Run/W&B ID: `rl_pixel_sft_t20_val32_p4_const_kl_20260911_134347`.
Explicit exported settings: LENGTH_NORMALIZATION=constant,
KL_GRADIENT_CORRECTION=1, MAX_SEQ_LEN=4096, WANDB_MODE=online, four H100s,
24 CPUs, 256 GB RAM, JOB_TIME=2-00:00:00, four prompts/GPU, four rollouts/prompt,
generation batch 16, train microbatch 2, one PPO epoch, LR=1e-5, KL=0.01,
mean-only advantages, reflection weight 1, three refinements, 128 reflection
tokens, 500 updates, evaluation/save every 25 updates, eight exemplars and
fixed val32. SFT_MODEL and PREV_STAGE_CHECKPOINT are pinned to
`/project/6004852/jeet/models/sft_ckpt/slf_ref_edit_t20/checkpoint-16000`.
Fresh output directory: this starts from SFT, not a legacy RL adapter.

Input/launcher previews passed before submission. `squeue`, `scontrol` and
`sacct` confirmed PENDING, four H100s and a two-day limit; no start time was
available. Both stdout/stderr had not yet been created. Logs will be
`results/logs/slurm/tar-rl-const-kl-59335407.{out,err}`. Online W&B is requested;
actual online initialization must be checked after allocation. Prior job
`59311752` was also pending and already showed a two-day limit at this check;
it was neither modified nor cancelled by this submission.

### Submitted LR 1e-4 comparison: 59359932

User authorized a fresh job at LR=1e-4 with a 23:59:00 limit. Submitted
`59359932` (`tar-rl-lr1e4`) at 2026-09-11 17:10:03 cluster time. Run/W&B ID:
`rl_pixel_t20_val32_p4_const_kl_lr1e4_20260911_170923`.
Both corrections remain enabled (constant denominator 4096, corrected K3 KL).
Explicit settings otherwise match the previous corrected run: four H100s on
one node, 24 CPUs, 256 GB RAM; four prompts/GPU and four rollouts/prompt;
gen batch 16, train microbatch 2, one PPO epoch, mean advantages, KL=0.01,
clip=0.2, reflection weight 1, LoRA rank 64/alpha 128, seed 421, 3% warmup
and cosine decay, 500 updates, evaluation/save every 25, val32/eight exemplars,
online W&B, three refinements and 128 tokens per reflection. Both SFT path
variables pin `slf_ref_edit_t20/checkpoint-16000` under
`/project/6004852/jeet/models/sft_ckpt/`. Fresh output directory starts from SFT.

Input checks and trainer/submission previews passed. Slurm verified PENDING,
four H100s and 23:59:00; eligible partitions are
`gpubase_bynode_b3,gpubackfill`. No start estimate or stdout/stderr existed at
verification. Logs: `results/logs/slurm/tar-rl-lr1e4-59359932.{out,err}`.
Actual online W&B startup and training remain to be checked after allocation.
No earlier job was modified or cancelled; the default launcher LR remains 1e-5.
