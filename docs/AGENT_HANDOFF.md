# Agent handoff — September 11, 2026

## October 19: `--objective teacher_tree` implemented (CPU validated; GPU checks next)

User design: tree rollouts with large groups, up to 3 refinements with early stop,
final-image reward for final and intermediate images, reflection-correctness reward,
and teacher-VLM reflections in the rollout. User choices: distill teacher reflections
and add K policy "shadow" reflections rewarded for verdict correctness vs judge
all_pass; teacher = Qwen3-VL-8B in the judge worker, image + prompt only; scale 4
prompts x 8 drafts x `TREE_BRANCH=1x2` per GPU. Code: `teacher.py` (prompt
`teacher_reflect_v1`, parser into the measured policy format: a leading space,
`...\nCorrection: ...\n` + `<im_start>`, or the canonical looks-good text + EOS,
from ~2.4k sampled reflections), `run_teacher_tree`, `build_teacher_tree_rows`
(`_pack_tree_nodes` refactor of the tree row builder), worker op `generate`, client
`generate_texts`, `grpo_loss(ce_weight=...)`, trainer scoring/advantages/stats,
teacher-free looks_good validation, method metadata, launcher variables, docs.
Judge batching v2 dropped: the judge is 0.28 s/image at batch 1 vs 1.07 s/image
rendering at batch 24 (job 64727589 report), so rendering dominates reward time.
CPU: 136 `test_rl_*` tests pass (14 new in `tests/test_rl_teacher_tree.py`, incl.
an exact SGD check that the CE step equals -lr x coef x grad(mean teacher NLL));
cluster tests pass. A check + smoke job (65918236) was cancelled while pending: at
the user's request the full run was submitted directly as job 65919413
(`launches/teacher_tree_t21_20261019/full/`, run
`rl_pixel_sft_t21_ck17000_teachertree_b1x2_p4g8_r3_ep2_kl001_lr1e4_20261019`): 24 h
segments with the Janus-style safe stop and afterok continuation to 500 steps;
4 GPUs x 4 prompts x 8 drafts, `1x2`, <= 3 refinements, reset context, 1 shadow,
teacher CE 0.1, `ADV_NORM=std`, KL 0.01 (corrected), LR 1e-4, 2 PPO epochs,
`GEN_BATCH_SIZE=32`, render batch 24; frozen snapshot passes all 136 CPU tests and
the flags were parsed by the frozen trainer before submission. Its first steps are
the first GPU test of this code (teacher format/agreement, memory, step time).
`scripts/rl_ft/check_teacher_reflections.py` and the smoke script remain in the launch
directory if needed.

## October 19: checkpoint-300 looks_good evaluation submitted

User request: evaluate the Janus all-pass r24 run's checkpoint-300 like checkpoint-150.
Root `results/evaluations/janus_allpass_r24_ck300_looksgood_5iter_20261019/`
{janus300_greedy, janus300_sample}; SFT not rerun (compare with the ck150 suite's
sft_* RESULTS.md: same code and seed). Checkpoint copied from `models/<run>_kept/`
(sha-equal to the run directory's copy; rl_method and adapter config equal to
checkpoint-150's up to LoRA target-module order). Code is the ck150 suite's frozen
snapshot, byte-identical (same manifest sha). `prepare_and_submit.py` derives every
job script from the ck150 suite's GPU-validated whole-node scripts (SFT configs as
source) by path/name substitution; 26 of 34 equal the same substitution of the
janus150 scripts, and all six generation commands match ck150's original module
arguments. Same shards, packing and limits (< 3 h); report `--mem=32G`. `--validate`
(frozen-code import + checkpoint metadata) passed. Submitted 34 jobs 65912785–65912818
(reports 65912801 greedy / 65912818 sample); job ids in `<config>/launch.json`.

## October 15: Janus all-pass run stopped at step 312; checkpoint-300 is the candidate

Run `rl_pixel_sft_t21_ck17000_janus1iter_allpass_r24_g8p8_ep2_kl001_lr1e4_20261013`
(64742381 steps 1-154, 65011109 steps 155-312, both COMPLETED 0:0). User asked to stop;
pending successor 65202971 was cancelled before starting. Train AM (10-step means):
draft 0.66 -> 0.84, final tracks draft (refined delta ~0 throughout); verdict accuracy
~0.80-0.86, "looks good" on failing drafts 0.03-0.08 except a transient 0.18 window at
steps 261-270 that reversed; KL ~0.13-0.15 with one-step K3 spikes (67, 162: 11.3, 176
in draft, 216) that all recovered. val32: draft 0.644 -> 0.800-0.823 (steps 250-300),
final 0.639 -> ~0.80; val300 verdict accuracy 0.844, failing-draft keep 0.043.
`--save_total_limit 3` pruned checkpoints <= 250 (checkpoint-250 is gone; another
session's suite holds a copy of 150 under `evaluations/janus_allpass_r24_ck150_*`).
Kept copies (sha256-verified): `models/<run>_kept/checkpoint-{275,300,312}`.
Suggested next: per-token KL cap; refinement-specific credit (delta advantage or M
refinements per draft); benchmark checkpoint-300 with the main protocol.

## October 13: 118-job benchmark suites complete (incl. retries)

`results/evaluations/t21_methods_native_and_main_20261012/*/RESULTS.md` (x100, one seed).
Main protocol (draft -> final): SFT GenEval2 61.9 -> 61.0 (main reports 62.0 -> 61.8),
TIIF short/long 63.3/61.1 -> 63.1/60.3, GenAI 33.3 -> 33.2; Sept reset 64.2 -> 64.8,
65.8/62.1 -> 65.4/62.6, 35.2 -> 36.0; textpg 64.7 -> 65.5, 66.7/61.2 -> 67.3/59.9,
35.2 -> 35.6; textpg_tree2 64.6 -> 64.0, 66.6/61.4 -> 66.6/60.8, 34.3 -> 35.1; tree2
64.1 -> 64.3, 67.6/61.7 -> 67.0/61.5, 34.7 -> 36.3. Native final: textpg 64.0,
66.2/59.3, 34.8; textpg_tree2 63.3, 63.6/60.1, 34.6; tree2 63.4, 63.9/62.8, 34.4.
All RL variants are +3 to +4.5 GenEval2 over SFT; method differences are within ~1.5
(single seed). Running: Janus all-pass job 64742381 (healthy at step 30, ~9 min/step);
user chose to keep it rather than restart with a smaller batch.

## October 13: Janus run collapsed; pass/fail verdict + batched reward

Run 64576010 (`..._janus1iter_g8p8_ep2_kl001_lr1e4_20261012`) stopped at its 16 h limit at
step 37 (checkpoint-37 kept). Mechanics were healthy: finite, KL <= 0.17, mismatches
<= 0.2%. But the soft-AM verdict reward drove "looks good" from 22% to 92%, and from 15% to
90% on failing drafts. Val32 at step 25: every logged reflection was "no issues ... looks
good"; draft and final AM rose only through the draft reward. Steps took 20-30 min, mostly
reward. Continuation 64715201 was cancelled at the user's request.

Fixes (CPU-tested, 130 tests pass):
- `--verdict_target all_pass` (default; recorded in metadata; old Janus checkpoints read as
  `soft_am`).
- Verdict accuracy/precision/recall stats.
- Batched renderer (`--render_batch_size`) and batched judge forward
  (`--reward_question_batch_size`, worker `--question_batch_size`). Both default to 1, the
  legacy path; checkpoint state records them under `reward_batching`.
- `scripts/rl_ft/check_batched_reward.py`: GPU timing plus judge equivalence (fails if the
  max per-question |diff| > 2e-3 or the AM |diff| > 1e-3). The fixed run depends on it via
  afterok.

Submitted from launch root `results/launches/janus_allpass_t21_20261013/` (snapshot tests and
CLI method == preflight METHOD verified):
- Check job **64727589** (1 H100, 2 h): report in
  `results/evaluations/batched_reward_check_20261013/report.json`.
- Training **64727590** (`afterok`): run
  `rl_pixel_sft_t21_ck17000_janus1iter_allpass_g8p8_ep2_kl001_lr1e4_20261013`, same settings
  as the collapsed run plus `--verdict_target all_pass --render_batch_size 24
  --reward_question_batch_size 32`.
- The training job pends on the GPUoutage reservation from 2026-10-13 10:00.

Check 64727589 ran: rendering 12.6 -> 1.07 s/image at batch 24 (adopted); judging 0.28 -> 0.20
s/image at batch 32, but per-question |diff| up to 0.108 (mean 0.002; 16/232 > 2e-3; 1 flip at
0.5). The check failed as designed, which cancelled 64727590. Relaunched with
`--reward_question_batch_size 1`: **job 64742381**, run
`rl_pixel_sft_t21_ck17000_janus1iter_allpass_r24_g8p8_ep2_kl001_lr1e4_20261013`, launch
`results/launches/janus_allpass_r24_t21_20261013/` (snapshot tests and method == preflight
verified). The unused `janus_allpass_t21_20261013` launch and its empty run directories are
left in place. The job pends on the GPU outage.

## October 12: submitted Janus one-iteration full run

User authorized the full run directly (no smoke test). Job **64576010**
(`rl-t21-janus1`), run `rl_pixel_sft_t21_ck17000_janus1iter_g8p8_ep2_kl001_lr1e4_20261012`,
launch root `results/launches/janus_t21_20261012/` (frozen snapshot; 119-test suite and
CLI parse passed in it). Settings: t21 SFT ck17000, `--objective janus_one_iteration
--verdict_rule correction --max_refinements 1 --num_rollouts 8 --prompts_per_gpu 8`
(32 prompts, 256 episodes/step), `--num_ppo_epochs 2 --adv_norm std --kl_coef 0.01
--kl_gradient_correction --length_normalization episode`, LR 1e-4 cosine, 500 steps,
val32, seed 421, automatic continuation. A `GPUoutage` maintenance reservation covers
nearly all H100 nodes from 2026-10-13 10:00 PDT, so the first segment's limit was
lowered to 16:00:00 (`scontrol update`; recorded in launch.json); successors keep 24 h
and will wait for the outage to end.

## October 12: eval retry after scratch I/O error

`t21_tree2_main-genai-img1` (64544249) failed at 11:45 with `OSError: [Errno 5]
Input/output error` writing to /scratch (dependents 64544250/51 cancelled). Its 534
written records are valid JSON (atomic writes); one 0-byte `prompt-0542.png.tmp`
remains and is overwritten on rerun. Scratch writes were healthy afterwards; no other
job logged EIO. Resubmitted as 64567697 -> 64567698 -> 64567699 (recorded under
`retry_job_ids` in `tree2_main/launch.json`, markers `*.retry_io_error_20261012.json`;
a first attempt was rejected by sbatch because afterok on an already-completed job is
invalid, marker renamed `*.sbatch_rejected.json`, no job created).

## October 12: Janus-style one-iteration objective (CPU only)

Added opt-in `--objective janus_one_iteration` (see `scripts/rl_ft/README.md`):
`EpisodeRollout.run_one_iteration`, `build_janus_rows` (packed segment rows,
`pos_segment`, stop-token `pg_mask`), `score_one_iteration`,
`assign_janus_advantages` (unbiased std), `janus_stats`; `normalize_groups` gained
`unbiased` and `adv_std_floor` (legacy bitwise unchanged); metadata keys `objective`,
`verdict_rule`, `adv_std_floor` (missing -> legacy). Design compared with
Janus-Pro-R1 (`../Janus-Pro-R1/janus-rl`): same rewards/normalization/segment loss;
deliberate differences: exact likelihoods, sampled draft tokens in context, optional
2 PPO epochs, corrected KL. CPU: 10 new tests (rollout paths, rewards vs Janus formula,
loss vs direct Janus compute_loss, real-model likelihood on packed and forced paths,
microbatch invariance, 2-rank kept-vs-refined equivalence, metadata) plus all prior
tests pass. No GPU run yet. Main-protocol GenEval2 final AM x100 (800 prompts):
SFT 61.0 (main reports 61.8), Sept reset 64.8, textpg 65.5, textpg_tree2 64.0, tree2 64.3.

## October 12: main-branch evaluation protocol; 118 benchmark jobs submitted

Status since October 8: the CUDA failure was node-side (some nodes lacked
`/usr/lib64/nvidia`; driver libs only in `/usr/lib64`, invisible to the CVMFS
loader). It was fixed upstream; the three method runs were resubmitted
(63819860/61/62 + continuations) and all reached step 500 (`target_reached`),
and both SFT t21-17k all-images suites (63820261-63820302) completed.
val32 AM x100 at step 500 (step 0 = 67.38): textpg 66.12 (best 73.90@375),
textpg_tree2 72.84 (best@500), tree2 67.67 (best 71.35@350), Sept reset 72.23.
SFT 17k final: reset GenEval2 59.41 / TIIF 62.42/58.41 / GenAI 33.79; full
57.60 / 60.70/58.47 / 31.95 (images 1-2 identical across modes, as expected).

Compared evaluation with origin/main f289996. Judges/scoring already match
(Qwen3-VL-8B GenEval2 first-token variant sums; identical TIIF judge file;
Qwen3.5-27B VQAScore via t2v_metrics code). Generation differs, so added
opt-in `--protocol main_adhoc` (`llava/train/rl/main_protocol.py`, port of
main's `eval/iterative_generation_adhoc.py::generate_batch`): one forced
refinement, greedy 128-token reflection, "looks good" skips refinement,
image top-k 1200 / top-p 0.95 with no vocab mask, reflection decoded and
re-encoded as `" " + text + "\n<im_start><S0>"`, full history, checkpoint
generation_config defaults, LoRA merged. Requires `--max_refinements 1` and no
context flag; recorded as `generation_protocol` in output manifests. An ad hoc
check ran main's own generate_batch and the port on a tiny random model with the
real t21 tokenizer: every generate call matched exactly for 3 seeds (the
looks-good path is covered by unit test only). Main's GenAI-Image-800 is a
Mila-only file (not public; the public 800 set is video prompts), so main mode
uses our GenAI-1600 at seed 13; GenEval2 seed 713, TIIF 13. Main's RNG streams
(global seed + rank) are not reproducible here. CPU: 109 tests passed.

Submitted (all PENDING at submission), root
`results/evaluations/t21_methods_native_and_main_20261012/` (job IDs in each
`launch.json` and `job_ids.txt`): native protocol (21 jobs each) for textpg,
textpg_tree2, tree2 step-500; main protocol (11 jobs each: image 1 + final) for
those three, the Sept reset RL step-500 and the SFT 17k. Snapshot tests 101/103
(2 cluster-script modules intentionally absent) and `validate.py` passed.

## October 14: looks_good evaluation protocol; Janus all-pass r24 checkpoint-150

New evaluation-only protocol (training paths unchanged; 122 `test_rl_*` CPU tests
pass, incl. new `tests/test_rl_looks_good.py`): `EpisodeRollout.run_looks_good`
and `--protocol looks_good` in `eval_geneval2` / `generate_instruction_benchmarks`.
A draft, then up to `--max_refinements` reflect -> refine rounds; an episode stops
when the reflection's verdict is "looks good" (`--verdict_rule correction`,
as in Janus training), otherwise a refinement is forced with
`run_one_iteration`'s splice rule. EOS alone never stops. The context flag must be
explicit (it is not inherited: the Janus run trained full-context, one
iteration; reset puts the latest image in the one-iteration draft slot).
`--decoding greedy` = argmax image tokens and reflections (new
`RolloutConfig.img_sample`, rejected for training); renderer/judges unchanged.
Records add per-reflection `verdicts`. `intermediate_eval`/report accept image
ordinals up to 8 (`MAX_ORDINAL`); the report takes an optional `final_note`.

User request: evaluate `rl_pixel_sft_t21_ck17000_janus1iter_allpass_r24_g8p8_ep2_kl001_lr1e4_20261013`
checkpoint-150 (copied + hashed before rotation) and the t21 ck17000 SFT, reset
context, <= 5 refinements ("5 iterations" = 5 reflect -> refine rounds, up to 6
images), greedy and sampled, seed 421 only. Root
`results/evaluations/janus_allpass_r24_ck150_looksgood_5iter_20261014/`
{janus150,sft}_{greedy,sample}: gen (GenEval2 scored inline), replay of images
1–5 (image 6 = final at the cap), TIIF/GenAI scoring of all six stages, report.
`validate.py` passed. Submitted 36 jobs: janus150_greedy 65045809–65045817,
janus150_sample 65045824–65045833, sft_greedy 65045834–65045842, sft_sample
65045843–65045851 (reports 65045817 / 65045833 / 65045842 / 65045851).

Oct 15 resharding (user request: 1-GPU jobs, each < 3 h). Completed and reused:
janus150_greedy gen-{geneval2,tiif,genai}, janus150_sample gen-geneval2
(GenEval2 final AM 55.1 greedy / 71.4 sampled). The 32 still-PENDING 4-GPU jobs
were cancelled and replaced by `sharded_1gpu/prepare_and_submit_1gpu.py`: each
work command becomes items x shards 1-GPU array tasks running the frozen module
with RANK=shard, WORLD_SIZE=shards (gen 8/8/12 shards for GenEval2/TIIF/GenAI,
replay 4/4/8 per ordinal, score 2 per directory; limits 2.5 h / 1.5 h / 1.25 h).
Outputs are shard-count invariant (global batch seeds, per-prompt renderer and
judge seeds); new manifests record the actual world size. Generation runs through
`sharded_1gpu/run_shard.py`, which no-ops the process group, recomputes the
broadcast checkpoint metadata on non-zero ranks, exits workers at the final
barrier, and runs the frozen rank-0 summary in a 1-GPU finalize job; replay and
score summaries (`--summary_only`) run as CPU jobs; original report.sbatch reused.
Record per config: `<config>/launch_1gpu.json`; first new job 65157778, reports
65157800 / 65157815 / 65157832 / 65157850. CPU check of the wrapper used a stand-in
module only; the first real tasks are the GPU validation.

Oct 16 back to whole-node jobs (user request): none of the 1-GPU arrays started in
29 h because Fir's per-GPU partitions reach only 30 of the 100 full-H100 nodes (14
of them drained) while whole-node jobs reach all 100; the 60 PENDING 1-GPU entries
were cancelled. `sharded_fullnode/prepare_and_submit_fullnode.py` turns each 1-GPU
array of T tasks into ceil(T/4) whole-node tasks (4 x H100, 24 CPUs, mem 0, same
time limits < 3 h): node task j runs 1-GPU tasks 4j..4j+3 concurrently with
CUDA_VISIBLE_DEVICES = one GPU and the unchanged RANK/WORLD_SIZE/case commands
(per-slot logs `<name>-<array>_<task>-slot<k>.{out,err}`); any failed slot fails
the task. Gen finalize jobs are whole-node restricted to the first GPU; CPU
summaries and report.sbatch are unchanged. CPU stub test checked the 40-pair
replay mapping, multi-item score packing, failure propagation and the 4-GPU guard;
prelude and case blocks are identical to the 1-GPU scripts. Record:
`<config>/launch_fullnode.json`; reports 65435569 / 65435584 / 65435601 / 65435618.
Oct 17: whole-node replay, generation (run_shard work + finalize) and scoring all
ran on GPU without errors (per-GPU placement and SHARD_DONE/SHARD_FINALIZED checked).
report.sbatch's 4G limit OOMed (janus150_greedy peak 12.7G): resubmitted as
65673790 with `--mem=32G` (completed, RESULTS.md written); the three pending
reports were raised to 32G with scontrol. janus150_greedy is complete.
Oct 19: all four configs complete (RESULTS.md per config; report peaks ~13G).
Draft -> final (looks_good, reset, <= 5 refinements), sampled: GenEval2 AM SFT
58.9 -> 59.1, ck150 74.2 -> 71.4 (best 74.3 after one refinement); TIIF short/long
SFT 61.4/58.0 -> 63.1/59.8, ck150 66.4/64.8 -> 67.9/64.4; GenAI SFT 30.9 -> 34.0,
ck150 39.1 -> 39.9. RL's gain is mostly in the draft; its refinements help less
than SFT's (and hurt GenEval2 attributes), so the RL-over-SFT margin shrinks
with refinement rounds. Greedy drafts are much weaker for both models.

## October 11: the three method runs reached step 500

All three runs COMPLETED at step 500 with no traceback, via two automatic
continuations each (first jobs 63819860/61/62; then 64076554/64076711/64076555
from steps 204/205/205; then 64325721/64327296/64325720 from 408/410/410;
"Target 500 reached; no successor submitted"). `checkpoint-{450,475,500}`
remain in each output directory. No benchmark evaluation of these adapters yet.

val32 `am_final` (single 32-prompt sample per checkpoint; one-checkpoint
swings of 5–9 points are common, so only averages are informative):

| Run | step 500 AM / GM | mean AM, steps 25–500 (20 ckpts) | mean AM, steps 400–500 |
|---|---|---|---|
| baseline reset (20260921) | 0.722 / 0.294 | 0.693 | 0.714 |
| textpg (transitions) | 0.661 / 0.220 | 0.692 | 0.681 |
| textpg + tree `2x1` | 0.728 / 0.342 | 0.684 | 0.693 |
| tree `2x1` (all-text PG) | 0.677 / 0.328 | 0.677 | 0.678 |

val32 does not separate the methods from the baseline. Prose-masked runs had
about half the text KL of all-text PG (~0.0015 vs ~0.0030).

October 13: the user prioritises the final-checkpoint benchmark evaluation.
Prepared (not yet submitted) `results/evaluations/rl500_method_runs_all_images_20261013/`
{textpg, textpg_tree2, tree2}: 21 jobs each, copied and hashed `checkpoint-500`,
the SFT suite's frozen code snapshot reused byte for byte (same manifest hash),
same seeds/judges/layout as the SFT and RL500 suites, driver-library block
included, explicit `--reset_context_per_turn` checked against metadata.
`validate.py` passed. Submitted at user request with afterok dependencies:
textpg **64815832–64815852**, textpg_tree2 **64815853–64815877**, tree2
**64815878–64815898** (report jobs 64815852 / 64815877 / 64815898).
All 63 were then CANCELLED (user-approved) before running: they duplicated
`results/evaluations/t21_methods_native_and_main_20261012/*_native` (submitted
2026-10-12 by another session; same adapter hashes, same 21 jobs and commands,
reset context from metadata), which had already COMPLETED. Use those results;
the 20261013 root is prepared-only. Step-500 native finals (GenEval2 AM / TIIF
short/long / GenAI): textpg 64.00 / 66.18/59.30 / 34.83; textpg_tree2 63.28 /
63.58/60.12 / 34.62; tree2 63.41 / 63.88/62.81 / 34.43; reset baseline 63.20 /
63.55/61.58 / 34.69. Checkpoints before step 450 were rotated away
(`save_total_limit 3`), so per-iteration benchmarks exist only at step 500.

## October 9: diagnosed CUDA failure; resubmitted the three method runs

Root cause (diagnostics 63718163, 63719511, 63719727 under
`logs/cuda_diag_20261008/`): some Fir GPU nodes rebooted 2026-10-08 15:08
(fc10414/15/16/18) have no `/usr/lib64/nvidia`; the driver is only at
`/usr/lib64/libcuda.so.1`, which the cvmfs Gentoo loader does not search, so
torch reports no driver (both `tar` torch 2.1 and `qwen_reward` torch 2.6).
fc10405 still has `/usr/lib64/nvidia`. `LD_LIBRARY_PATH=/usr/lib64` breaks
Python (host glibc). Workaround: a node-local symlink directory containing only
`/usr/lib64/libcuda*`, `libnvidia-*` and `libcudadebugger*`, prepended to
`LD_LIBRARY_PATH`; verified on fc10405 (CUDA + 2-GPU NCCL all-reduce in both
environments), not yet on a node lacking `/usr/lib64/nvidia`.

User requested direct resubmission with a full node each. Each `resume.sbatch`
now adds the symlink block, `--exclusive` and 48 CPUs (originals kept as
`resume.sbatch.first_attempt_63717xxx`; `launch.json` keeps the first job ID,
adds `retry_job_ids`, new `script_sha256`). Preflights passed; no checkpoints
existed. Resubmitted: **63819860** (textpg), **63819861** (textpg_tree2),
**63819862** (tree2), PENDING at submission. All three started 2026-10-09
(fc10112, fc10417 — a node from the affected group — and fc10215), passed NCCL
init, and logged step-0 val32 AM 0.6738 (identical to the prior reset baseline)
and step 1 with no OOM or traceback. Step 1 times ≈ baseline (rollout ~150 s,
render+judge ~215 s, train ~35 s); tree runs: 8 leaves/prompt, 37k trained
tokens vs 43k for episodes. Runs 2/3 share identical step-1 rollouts (same seed
and initial policy); only `pg_txt` differs, confirming the prose mask is active.

SFT all-images suites (user-requested): the same symlink block was added to all
42 scripts (originals in `{reset,full}/original_scripts/`, `launch.json`
records both hashes); `validate.py` re-passed; submitted with afterok
dependencies: reset **63820261–63820281**, full **63820282–63820302** (report
jobs 63820281 / 63820302). All PENDING at submission.

All 42 COMPLETED (sacct). `RESULTS.md` in each mode directory; full benchmark,
"up to image N" (scores x100; RL step 500 from
`rl500_four_models_all_images_20260930/t21_17k_{reset,full}`):

| Model | GenEval2 AM img1 / img2 / img3 / final | TIIF short/long final | GenAI img1 / final |
|---|---|---|---|
| SFT 17k reset | 58.88 / 59.15 / 59.51 / 59.41 (GM 16.11) | 62.42 / 58.41 | 30.93 / 33.79 |
| SFT 17k full | 58.88 / 59.15 / 59.18 / 57.60 (GM 15.91) | 60.70 / 58.47 | 30.93 / 31.95 |
| RL500 reset | 62.50 / 63.54 / 63.68 / 63.20 (GM 17.18) | 63.55 / 61.58 | 32.66 / 34.69 |
| RL500 full | 63.05 / 63.48 / 64.10 / 62.57 (GM 18.11) | 63.30 / 61.23 | 33.44 / 33.01 |

SFT images 1–2 are identical across modes (same seeds; the first refinement
context is the same), so the modes differ only from image 3. Full-context SFT
loses 1.6 GenEval2 AM and 0.9 GenAI from image 3 to final; reset does not.
RL adds ~3.6 AM to image 1 and +3.8 (reset) / +5.0 (full) AM at final.

## October 8: method runs failed at CUDA init; SFT all-images suites prepared

All three jobs (63717015/16/18) FAILED after ~2.5 min, before any training:
every rank raised `ProcessGroupNCCL is only supported with GPUs, no GPUs
found!` at `dist.init_process_group`, although `nvidia-smi` in the same job
listed four H100s. Driver 580.159.03 equals the working September runs; the
`tar` environment (torch 2.1.2+cu121) is unchanged since September 11; the
sbatch differs from the working reset run only in the new method flags. Under
diagnosis with one-GPU job 63718163 (`logs/cuda_diag_20261008/`). Not resubmitted.

Evaluation: added opt-in `--reset_context_per_turn` / `--no-...` to
`eval_geneval2` and `generate_instruction_benchmarks` via
`evaluation_resets_context`: full SFT checkpoints may opt in (default stays
full context); RL adapters must match their metadata or fail. CPU tests: 92 RL
regressions passed. Prepared, not yet submitted:
`results/evaluations/sft_t21_ck17000_all_images_20261008/{reset,full}` (21 jobs
each, same protocol/judges/seed/layout as the step-500 suite; frozen snapshot
tests and `validate.py` preflight passed).

## October 8: submitted three t21 ck17000 reset-context method runs

User authorized three fresh runs from t21 SFT checkpoint-17000, all with reset
context and the prior reset baseline's settings (LR1e-4, constant
denominator4096, corrected KL0.01, LoRA64/128, seed421, 500 steps, val32,
eval/save every25, keep3, four H100s, 24-hour segments with automatic
continuation, online W&B). Launch root (frozen code snapshot overlaying the
current RL source, preflight, launch.json, submission markers):
`results/launches/tree_textpg_t21_20261008/`.

| Job | Run name | Method | Prompts/GPU x drafts |
|---|---|---|---|
| 63717015 | `rl_pixel_sft_t21_ck17000_reset_textpg_const_kl_lr1e4_20261008` | text PG transitions, episodes | 4 x 4 |
| 63717016 | `rl_pixel_sft_t21_ck17000_reset_textpg_tree2_const_kl_lr1e4_20261008` | transitions + `tree_branch 2x1` | 2 x 4 |
| 63717018 | `rl_pixel_sft_t21_ck17000_reset_tree2_const_kl_lr1e4_20261008` | all-text PG + `tree_branch 2x1` | 2 x 4 |

Compare against `rl_pixel_sft_t21_ck17000_reset_const_kl_lr1e4_20260921`.
Before submission: 91 RL CPU tests passed inside the snapshot, each run's
exact CLI parsed to the intended `rl_method`, SFT fingerprint preflight
passed. All three verified PENDING immediately after submission; GPU start
and tree-mode memory/runtime are not yet verified.

## October 8: opt-in tree credit and reflection-prose PG masking (CPU only)

Added two opt-in RL methods; defaults run the unchanged episode code path.
`--tree_branch "R[xM],..."` (`TREE_BRANCH`) samples episodes as a tree via
`EpisodeRollout.run_tree` (`TreeNode` records shared action spans). Rewards
remain final images only, rendered/scored once per distinct final image node.
`assign_tree_advantages`: value = mean final reward of the subtree, advantage =
value minus siblings' mean, only children inherit their branch point; std mode
uses the per-prompt mass-weighted leaf std. `build_tree_training_rows` trains
each node once (merging only-child chains so unbranched rows equal episode
rows/segments), weights by mass, denominator = drafts. `--text_policy_gradient
transitions` adds a `pg_mask`: reflection prose keeps KL but no PG.
`rl_method` metadata adds `tree_branch` (canonical list, missing `[]`) and
`text_policy_gradient` (missing `all`). Launcher: `TREE_BRANCH`,
`MAX_TREE_LEAVES_PER_PROMPT`, `TEXT_POLICY_GRADIENT`. Docs:
`scripts/rl_ft/README.md`.

Validation: **91 RL CPU regressions passed** (76 existing unchanged, 12 tree,
3 text-PG) plus 8 cluster/monitor checks. New tests cover exact equivalence
with episode mode when unbranched (trajectories, advantages, rows, updates;
full and reset context; both normalizations), clone independence, exact node
contexts, leaf/terminal bijection under room/cap/reflection limits, masses
with image branching, shared final-image scoring, tied-draft std handling, an
exact-expectation unbiasedness test in constant mode (a leaf-count weighting
mutant fails it), real tiny-model sampling log-probs equal tree training
log-probs, microbatch invariance, metadata/resume guards and a two-rank Gloo
update. Launcher `bash -n` and `TAR_DRY_RUN=1` previews passed. No GPU run or
job submission yet; full-model memory/runtime in tree mode are unvalidated.

## September 30: submitted four step-500 models, three benchmarks, images 1–4

Latest refresh (after 14:05 cluster time): 28/84 stages COMPLETED 0:0;
subsequent squeue shows 7 RUNNING and 49 PENDING, no failures in accounting.
All four final GenEval2 evaluations are complete (800 each); t21/14k full
AM/GM x100 is 63.5336/18.9864. GenEval2 images 1–3 are also complete
for t20 reset, t21/17k reset and t21/17k full (800 each). Their image1/2/3/final
AM x100: 60.2967/63.3034/64.2388/64.9750,
62.4953/63.5439/63.6795/63.2043, and
63.0528/63.4825/64.1008/62.5743 respectively.
TIIF final scoring now also complete for t21/17k full (554, short/long x100
63.2959/61.2267). Both reset models have TIIF images 1–3 generation complete,
but intermediate scoring remains pending. No GenAI score or combined report
yet. Both stdout/stderr of all 14 newly completed jobs inspected; saved
metrics confirm scored completions. No jobs changed.

Subsequent live accounting/log check (through 12:58 cluster time): 14 stages
COMPLETED 0:0. Completed IDs: 62271521, 62271525–62271527, 62271529,
62271531, 62271533, 62271542, 62271546–62271547, 62271554, 62271563,
62271567, 62271588. Both stdout/stderr inspected. Final TIIF metrics are now
complete for t20 reset and t21/17k reset (554/554 each): short/long x100
60.1783/57.7326 and 63.5453/61.5788 respectively. TIIF final generation
is complete for all four models; t20 reset intermediate images 1–3 generation
also completed. GenAI final generation completed for both reset models
(1600 each), with scoring pending. Three final GenEval2 results below remain
the completed scored results; t21/14k GenEval2 and both full-context GenAI
generation jobs were still running. No complete all-images suite yet.

Later status check: 6 jobs COMPLETED 0:0, 8 RUNNING, 70 PENDING; no failed
states in sacct. GenEval2 final-image evaluation is complete (800 prompts)
for t20 reset (AM/GM x100 64.9750/19.2316), t21/17k reset
(63.2043/17.1754), and t21/17k full (62.5743/18.1096). TIIF generation
also completed for those three, but scoring remains outstanding. Both
stdout/stderr of the six completed jobs were inspected. No combined report
exists yet; intermediate-image results, GenAI scoring and t21/14k evaluation
remain outstanding. The initial-check script's queue-membership assertion
now fails because completed jobs leave squeue; fresh sacct evidence was read
from its saved `submission_verification.json` instead.

User requested submissions for both reset models and full-context t21 models
at RL step500. Scope includes both t21 SFT bases (14000 and 17000).
Root: `results/evaluations/rl500_four_models_all_images_20260930/`.
Each model subdirectory contains a SHA256-verified checkpoint-500 adapter copy,
frozen code, `launch.json`, `comparison.json`, 21 sbatch files and preparation
script. Logs: `results/logs/rl500_four_models_all_images_20260930/<model>/`.

| Model subdirectory | GenEval2 generation | TIIF generation | GenAI generation | Report |
|---|---|---|---|---|
| t20_16k_reset | 62271521 | 62271525 | 62271533 | 62271541 |
| t21_17k_reset | 62271542 | 62271546 | 62271554 | 62271562 |
| t21_17k_full | 62271563 | 62271567 | 62271575 | 62271583 |
| t21_14k_full | 62271584 | 62271588 | 62271597 | 62271605 |

All **84 jobs** (80 GPU stages + 4 CPU reports) verified PENDING in both
squeue/sacct immediately after submission. Both stdout/stderr were absent
for all jobs. GPU stages request four H100s, 24 CPUs, full-node memory
(`--mem=0`); GenEval2/TIIF limit 02:59:00, GenAI limit 12:00:00. The latter
was increased because prior GenAI image1 scoring 61155777 timed out at six
hours. Reports depend on all required stages with afterok and
kill-on-invalid-dep. Online W&B requested. Exact IDs (not a contiguous overall
range), resources and initial states are in `submission_verification.json`.

The shared snapshot retains the prior verified renderer/judges and overlays
the current reset-aware RL/evaluation code. Relevant rollout, GRPO, trainer,
evaluation-policy and generation modules match the reset training snapshot
byte-for-byte. Reset mode is read from each checkpoint, not forced by CLI.
Same benchmarks (800 GenEval2, 554 TIIF, 1600 GenAI prompts), judges, seed421,
CFG4, SDPA, three refinements, 128 reflection tokens, total4096-token cap.
Intermediate stages replay saved semantic images. Early stops retain their
last complete image; actual ordinal coverage and matched subsets are reported.
Up-to-image4 aliases exact final PNGs/scores. No additional rounds are forced.

Validation: **35 CPU regressions passed** in the frozen evaluation environment,
including reset likelihoods/masks/caps, two-rank unequal-segment updates,
checkpoint loading, resumption/integrity, exact final-image reuse and reports.
Imports, source hashes, all 84 shell scripts, adapter spans (392 tensors/model),
base shard structures/read probes (866 tensors/base), and checkpoint base/mode
checks passed. See `validation_passed.json`, `preflight_passed.json`, `validate.py`.
`prepare_all.py --submit` skips recorded job IDs; unresolved submission-attempt
markers require reconciliation, never blind resubmission. GPU startup and
benchmark completion remain outstanding. No training jobs were changed.

## September 30: verified completed RL runs and evaluation status

Read-only artifact/log review and live sacct confirm the three September 21
runs reached 500/500 and completed 0:0 on September 25: t21 ck17000 full
context (61485829), t21 ck17000 reset (61483901), and t20 ck16000 reset
(61487393). Each has checkpoint-500, step-500 validation, a target_reached
status file, and stdout confirming no successor was submitted. Both stdout
and stderr were inspected. Earlier t20 continuation (60377786) and t21
ck14000 (60520816) also have COMPLETED 0:0 accounting records.

Fixed val32 AM x100, baseline -> step500; best observed validation:
- t21 ck17000 full: 61.44 -> 69.29; 74.08 at step100.
- t21 ck17000 reset: 67.38 -> 72.23; 74.48 at step425.
- t20 ck16000 reset: 66.72 -> 72.44; 74.52 at step250.
- Earlier t21 ck14000 full: 63.11 -> 69.29; 71.28 at step300.
- Earlier t20 full continuation: step500 68.81; best across the continuation
  70.71 at step425 (original step0 baseline 64.88).
These small-set validation peaks are not full benchmark results. Reset-run
peak checkpoints 425/250 are absent from their current retained checkpoint
directories; the t21 full step100 permanent copy is documented below.

The t21 ck17000 RL100 evaluation has complete final-image metrics for GenEval2
(800 prompts, AM/GM x100 62.8045/18.8261), TIIF (554, short/long
61.2623/57.6453), and GenAI (1600, 32.9564). Its combined RESULTS.md is absent,
GenAI image1 metrics.json is absent, and dependent report 61155782 is CANCELLED;
do not describe the entire all-images suite as completed. The older t21
ck14000 RL400 combined report now exists, including the completed GenAI
image2 score 34.4104. No full benchmark artifacts were found for the newest
step500 checkpoints or either reset run. No jobs were submitted or changed.

## September 23: t21 ck17000 RL step-100 permanent copy and image 1–4 evaluation

The user chose step 100 of `rl_pixel_sft_t21_ck17000_const_kl_lr1e4_20260921`
(the full-context run, job 61100222). This is **t21 SFT checkpoint-17000 + RL step
100**. It is not the checkpoint-14000 base that the step-400 evaluation used.
Copies of the complete checkpoint (adapter, optimizer, scheduler), all
SHA256-verified against the training output:
- Persistent, read-only, group `rrg-bengioy-ad`:
  `/project/rrg-bengioy-ad/xiaofeng/tar_reason_checkpoints/rl_pixel_sft_t21_ck17000_const_kl_lr1e4_20260921/checkpoint-100`,
  with hashes in `checkpoint-100.sha256` beside it.
- Scratch backup: `results/launches/t21_rl_eval_candidates_20260923/checkpoint-100`.
Note that the rrg-bengioy-ad project's file count is 4995K of its 5000K quota.

Evaluation root: `results/evaluations/t21_ck17k_rl_ck100_all_images_20260923/`
(`prepare_and_submit.py`, `launch.json`, sbatch scripts, and later
`RESULTS.md`). Logs are in `results/logs/t21_ck17k_rl_ck100_all_images_20260923/`.
It reuses the step-400 evaluation code snapshot byte-for-byte (manifest
`33d2bb41…`), with the same inputs, judges, seed 421, caps and CFG 4. The stages
are the same as before: image 1, up to image 2, up to image 3, and final; up to
image 4 aliases final. Changes: GenAI GPU jobs get 06:00:00 (GenEval2/TIIF keep
02:59:00), and every 4-GPU job requests `--mem=0` (Slurm grants 1125G).
Jobs: GenEval2 **61155762–61155765**; TIIF **61155766–61155773**; GenAI
**61155774–61155781**; report **61155782**, with afterok dependencies and
kill-on-invalid-dep. All were PENDING at submission. Results are pending until
sacct, logs and the report are checked.

## September 23: final state of the three September 21 training runs

sacct now reports **NODE_FAIL** (1:0) for 60768995, 60783514 and 60783515, all
ending 2026-09-21 13:18:16. None is fully trained (target 500 steps): the last
console steps were 59, 28 and 28. The latest checkpoints remain 50 (t21 full
context) and 25 (both reset runs). No continuation job was queued and nothing
was resubmitted. Leftover `active_job.lock` files are harmless: `flock` is
released when the process dies.

All three W&B runs were online (`xiaofeng-zhang/tar_reasoning/<run name>`) and
uploading with HTTP 200 until about 12:04 PDT. The server has history for steps
0–58 / 0–27 / 0–27 and validation at steps 0/25/50, 0/25 and 0/25; its state is
`crashed`. The local `.wandb` logs contain only steps 0–57 / 0–27 / 0–27, so
`wandb sync` would add nothing. The final console steps (59, 28, 28) were never
committed to W&B because a row is committed only when the next step is logged;
their scalars survive only in `files/output.log`. On resume with the same run ID,
re-logged steps at or below the server's last step (51–58 and 26–27) would be
dropped by W&B.

**Resubmitted (user-authorized):** 61100222 (t21 full context, resumes
checkpoint-50), 61100223 (t21 reset, checkpoint-25), 61100224 (t20 reset,
checkpoint-25). All are PENDING, and the preflight passed on the login node for
each. The only change is `#SBATCH --mem=256G` -> `--mem=0` in each run's frozen
`resume.sbatch`: four H100s is a full node, so Slurm now requests 1125G. Automatic
continuations reuse that script. The code snapshot, trainer flags, W&B run ID and
output directory are unchanged. Hashes before and after the edit are in each launch
root's `retry_node_fail_20260923.json`; `launch.json` keeps the original job ID
and hash.

## September 21: three training allocations on unresponsive nodes

User reported the three recent runs failed. Live checks found all three still
RUNNING in squeue/sacct, but `scontrol show node` reports
`MIXED+NOT_RESPONDING` on each allocated node: job60768995 on fc10407,
job60783514 on fc10108, job60783515 on fc10216. SSH to fc10108 returned
`No route to host`; bounded SSH diagnostics to the other two timed out.
This supports a compute-node/cluster connectivity incident, not a demonstrated
Python/RL exception; underlying infrastructure cause is not yet established.
Both stdout/stderr show no traceback, OOM or timeout error. W&B internal logs
also showed no error at inspection. Last completed logged steps were 59,28,28
respectively, with output stale roughly15-20 minutes at the initial check.
Saved checkpoints were25/50 for the full-context run and25 for both reset runs.
No jobs were cancelled, restarted or resubmitted during diagnosis. Scheduler
RUNNING is not proof these unresponsive allocations are making progress.

## September 21: submitted t21/t20 context-reset experiments

User authorized both new runs. Submitted **60783514** (`rl-t21-reset-lr1e4`)
from t21 SFT checkpoint-17000 and **60783515** (`rl-t20-reset-lr1e4`) from t20
SFT checkpoint-16000. Both verified PENDING in squeue/sacct immediately after
submission; neither stdout nor stderr existed yet. Existing full-context t21
job **60768995** was RUNNING at the pre-submission check and was left untouched.

Shared frozen current-code snapshot:
`results/launches/reset_context_t21_t20_20260921/code`.
Each of `{t21,t20}/` beneath that launch root contains `resume.sbatch`,
`preflight.py`, `base_manifest.json`, `launch.json` with job ID, and a submission
attempt marker. `prepare_and_submit.py` preserves the preparation/submission
procedure and skips recorded job IDs; do not repeat ambiguous submissions.

Run names:
`rl_pixel_sft_t21_ck17000_reset_const_kl_lr1e4_20260921` and
`rl_pixel_sft_t20_ck16000_reset_const_kl_lr1e4_20260921`.
Each has separate models/evaluations/logs/W&B directories under `results/`.
Settings: reset-context enabled, constant denominator4096, corrected KL,
LR1e-4, KL0.01, LoRA64/128, four prompts/GPU, four independent rollouts,
four H100s, 24CPUs, 256GB, 24-hour allocations, fresh step0 toward500 with
progress-checked automatic continuation. Evaluation/save every25, val32,
eight exemplars, keep3 checkpoints, online W&B. Same final-pixel Qwen reward
and renderer CFG4. Includes the complete-response reward timeout fix.

Both SFT bases passed metadata/index and all shard-header/span/read checks
(866 tensors each). Source SHA manifests, batch syntax and launch flags checked;
isolated preflight probes rejected full-context resume states. Preflight checks
base metadata/shard fingerprints, source hashes, exact base and all four method
metadata fields on each segment. The frozen trainer passed import/CLI checks
in the actual Tar environment after a shared-filesystem wait. GPU startup and W&B initialization remain
outstanding until allocation; prior CPU regression results are recorded below.

## September 21: reward worker complete-response timeout

Fixed review P2 in `pixel_reward.py`: readiness of the first stdout byte no
longer leads to an unbounded blocking `readline()`. Stdout is nonblocking and
raw chunks accumulate in a byte buffer until newline under one monotonic
deadline per response, including startup. Partial/dripping output cannot
restart the timeout. Trailing bytes are preserved for subsequent responses;
JSON is decoded only after a complete line, including split UTF-8. EOF before
newline raises a worker-exit error; timeouts close/reap the worker and raise
TimeoutError, with no substitute reward. Existing termination/kill cleanup
grace remains unchanged.

Validation: nine targeted transport/pixel-reward tests passed in Tar's Python,
including silent/partial/dripping output, partial startup cleanup, fragmented
UTF-8, buffered multiple lines, early EOF, malformed JSON, PNG transport and
official reward semantics. The four new transport tests also passed with
system Python. Whitespace checks passed. No GPU run or frozen job snapshot
was changed.

## September 21: optional context reset for final-reward RL

Implemented `--reset_context_per_turn` / `--no-reset_context_per_turn`, default
off, exposed as `RESET_CONTEXT_PER_TURN=1` in the shared RL launcher. After
each corrected image, the next reflection starts with the original prompt and
latest semantic image. Reflection and its requested correction share that
context. Initial draft/first correction can share one segment because their
conditioning is identical to the previous implementation.

`Trajectory.seq/kinds` retain the logical full episode for safety accounting
and metrics; `segments` retain exact independent model inputs and action masks.
Copied image/prefix tokens are observations, not new actions. All original
sampled actions (including draft, transitions and EOS) receive the episode's
final-image advantage exactly once. Episode normalization spans segments;
constant normalization still divides by MAX_SEQ_LEN. GPU microbatches bound
segment rows; optimizer denominator still counts episodes. Metric weight sums
now handle partial-episode segments below weight one. Logical total token,
architectural and refinement caps remain unchanged; resets do not replenish
the episode budget. No cross-call KV cache is carried across resets.

Checkpoint method metadata includes the flag; missing means false for both
old legacy and corrected-method checkpoints. Mismatched resume flags fail.
In-training evaluation shares the rollout mode. Standalone GenEval2/TIIF/GenAI
generation inherits it from checkpoint state, fingerprints enabled mode in
the manifest and saves exact context segments alongside logical sequences.
Full SFT evaluation defaults to the historical context mode. No change to
the final-pixel reward, group assignment, renderer or trainable parameters.

Validation: all **72 RL CPU regressions passed**, including seven new context
reset tests: real generation-vs-training token likelihoods, exact-once action
coverage/copy masking, final-only reward and earlier-action credit, EOS equal
to padding, global caps, legacy/new resume handling, microbatch invariance
and two-rank updates with unequal segment counts in both normalization modes.
Eight cluster/layout/monitor checks passed. Launcher opt-in dry run, shell
syntax and whitespace checks passed. Initial sandbox Gloo socket failure was
resolved by running the suite with authorized local-socket permissions.
Review traced sampler -> segments -> likelihood/KL -> episode denominator ->
distributed averaging and checkpoint/evaluation propagation; no outstanding
blocker found in these tested paths. Full-model GPU validation remains pending.

Usage and limits are documented in `scripts/CLUSTER.md`. No new job was
submitted or existing job modified: **60768995** retains its original frozen
full-context source snapshot. Use a fresh run name and updated source snapshot
for a context-reset experiment; do not resume full-context checkpoints under
the new flag.

## September 21: fresh RL from t21 SFT checkpoint-17000

User requested a new RL run from
`/project/rrg-bengioy-ad/jeet/models/sft_ckpt/slf_ref_edit_t21/checkpoint-17000`
(canonical project alias `/project/6004852`). Submitted job **60768995**
(`rl-t21-17k-lr1e4`), verified **PENDING / Priority** in squeue and sacct;
stdout/stderr had not yet been created. Run
`rl_pixel_sft_t21_ck17000_const_kl_lr1e4_20260921`, fresh step 0 toward 500,
using the previous t21 settings: LR 1e-4, LoRA 64/128, constant denominator
4096, corrected KL, KL 0.01, four prompts/GPU, four independent rollouts,
generation batch 16 and training microbatch 2. Same final-pixel Qwen reward,
three refinement cap, renderer CFG 4, fixed val32, evaluation/save every 25
steps, eight exemplars and online W&B. Retention stays at three checkpoints.

Launch snapshot and batch script:
`results/launches/sft_t21_ck17000_20260921/{code,resume.sbatch,launch.json}`.
Models, evaluations, logs and W&B use this fresh run name beneath `results/`.
Uses the previous proven t21 trainer snapshot without changing RL behavior;
shared `sft_model` link and prior runs are untouched. Allocation is four H100s,
24 CPUs, 256 GB, 24 hours, with safe checkpointing before timeout and automatic
progress-checked continuation until 500. `--resume` starts fresh in the empty
new output directory and resumes only its own later segments.

Validation: all four new SFT shard headers, tensor byte spans/sizes and reads
passed; all 866 tensor keys/shapes/dtypes match t21 checkpoint-14000. Model
configuration, tokenizer and special/generation-token files are identical.
Inherited source hashes, shell syntax, precise training options, fresh output
and continuation paths checked. `base_manifest.json` fingerprints metadata,
shard sizes and headers; the preflight enforces these on each segment. Isolated
adversarial checks confirmed changed SFT metadata and wrong-base RL resume are
rejected. The actual Tar environment also passed trainer import and CLI
validation (`python -m llava.train.rl.train_grpo --help`, exit 0), after a
shared-filesystem wait. GPU startup, new-weight inference and online W&B
initialization remain outstanding until allocation.

## September 20 UTC: t21 RL checkpoint-400, four image stages and GenAI categories

September 21 check: all evaluation stages completed 0:0 except GenAI image-2
scoring **60551619**, which TIMEOUTed after 02:59:20; report **60551622** was
CANCELLED by its failed dependency. Source generation/replay is complete for
all benchmarks and images. GenAI image-2 has 1209/1600 valid scores, distributed
400/400/400/9 across ranks; no full score is reported for this stage. Stderr
contains repeated Slurm NVML errors on fc10501, but the cause of rank-3
slowness is not established.

Validated all 1600 saved image records and all 1209 saved scores, frozen code
hashes and original scripts. Resubmitted unchanged scoring as **60767771**,
same four H100s / 02:59:00, excluding fc10501; it will skip verified existing
scores and compute the missing 391. Dependent report **60767772** uses afterok.
Both PENDING at verification, no logs yet. `launch.json` now identifies retry
jobs and preserves previous IDs; `retry_genai_image2_20260921.json` records
recovery details. Do not submit duplicate retries.

Completed full-set metrics x100 (image1 / up-to2 / up-to3 / final): GenEval2
AM 62.5042 / 63.2412 / 64.0343 / 63.2484; GM
18.3270 / 18.8991 / 19.5262 / 19.7952. TIIF short
65.4065 / 65.3954 / 64.6369 / 63.4696; long
59.3823 / 59.6136 / 59.5603 / 59.0215. GenAI all
32.9277 / pending / 33.4898 / 32.9115; basic
36.7049 / pending / 36.2864 / 34.9541; advanced
29.7967 / pending / 31.1717 / 31.2183. Aggregate report and its W&B upload
remain outstanding until the retry succeeds. These supersede startup status below.

User requested t21 after 300 RL steps, then chose 350, then explicitly chose
**400** after ongoing training pruned the older checkpoints. This evaluation
is **t21 SFT checkpoint-14000 + corrected LR 1e-4 RL step 400**, not the earlier
t20 step-400 run. No evaluation of t21 step 300/350 was submitted.

Preserved candidate adapters 400 and 425 under
`results/launches/t21_rl_eval_candidates_20260920/`, with hash provenance.
Only 400 is evaluated; 425 is a backup candidate. The selected adapter/config/
state were copied and verified again into
`results/evaluations/t21_rl_ck400_all_images_20260920/checkpoint-400`.
Base config points to `/project/6004852/jeet/models/sft_ckpt/slf_ref_edit_t21/checkpoint-14000`.
392 tensors passed safetensors byte-span/shape checks. Adapter SHA256:
`041fd3081d5ce377e27a129f3ac4e8903141d2ffa70f50b6dbb2df0bd82c6faf`.

Same full 800 GenEval2 / 554 TIIF / 1600 GenAI inputs, judges, seed 421,
batch 16, SDPA, CFG 4 and rollout caps as t20 step 500. Each GPU stage uses
four H100s, 24 CPUs, 256 GB and 02:59:00; online W&B. Images 1–3 replay saved
semantic codes from the same episodes; up-to-four reuses final pixels/scores.
The report includes actual image coverage and identical-prompt subsets.

GenAI reporting now explicitly separates **simple = basic** and
**complex = advanced**, using saved benchmark tags: 722 basic, 871 advanced,
zero overlap and 7 untagged prompts. Untagged prompts are generated/scored but
excluded from both categories and from the existing official tagged-union
aggregate. No difficulty is inferred from prompt length. `per_skill` scores
already exist in JSON; the report now adds readable full-set and matched-subset
tables plus flat W&B basic/advanced scalars and counts for each image stage.
Judging and benchmark aggregation code remain unchanged.

Launch root (snapshot, scripts, `launch.json`, `comparison.json`, eventual
`RESULTS.md` / `results.json`):
`results/evaluations/t21_rl_ck400_all_images_20260920/`.
Logs: `results/logs/t21_rl_ck400_all_images_20260920/`.
Jobs: GenEval2 final **60551602**, images 1/2/3 **60551603–60551605**;
TIIF final generation/scoring **60551606/60551607**, image stages
**60551608–60551613**; GenAI final generation/scoring **60551614/60551615**,
image stages **60551616–60551621**; combined CPU/W&B report **60551622**.
Dependencies use afterok and kill invalid dependent jobs. Last check: GenEval2
final RUNNING on fc10410; other jobs PENDING on priority/dependencies. Squeue,
sacct and both available log streams checked; only startup output exists,
so successful generation/scoring remains outstanding.

Validation: all 24 CPU benchmark/ordinal tests passed in the working tree and
the frozen snapshot using the actual evaluation Python. New checks cover
category tag membership, unlabelled exclusion, matched subset counts, missing
categories and inconsistent cross-stage counts; end-to-end report assertions
cover both simple/complex labels. Checkpoint/copy hashes, inherited code,
benchmark hashes, all 21 batch scripts and dependency ordering were verified.

## September 19: continued t20 checkpoint-500, all four image ordinals

Completion verified: all 21 jobs 60464510–60464530 COMPLETED with exit 0:0.
Full scores exist for all 800 GenEval2, 554 TIIF and 1600 GenAI prompts at
every requested image budget. Combined report stdout and W&B sync confirmed;
this supersedes running/pending and outstanding validation statements below.
Results x100 (up to images 1 / 2 / 3 / 4=final): GenEval2 AM
62.8774 / 64.0485 / 63.3021 / 62.3786, GM
18.9318 / 18.4267 / 17.0028 / 17.2662; TIIF short
63.1503 / 64.0956 / 61.4804 / 60.5562, long
57.7723 / 57.9543 / 58.1225 / 58.2446; GenAI VQAScore
33.0161 / 34.6079 / 33.7610 / 32.7083.
Actual image 1/2/3/4 counts: GenEval2 800/674/611/343;
TIIF 554/178/175/106; GenAI 1600/904/903/599. Actual-ordinal
matched-subset metrics and prompt indices are in `results.json`; full report
is `results/evaluations/ck500_all_images_20260919/RESULTS.md`.
W&B comparison/visuals: `xiaofeng-zhang/tar_reasoning/ck500_all_images_20260919`.

User requested all three benchmarks for the completed continued t20 LR 1e-4
run, plus images 1, 2, 3 and 4 within the same episodes. Training status
`training_status_60377786.json` confirms step 500 / target reached. Preserved
adapter/config/state from
`results/models/rl_continue_corrected_lr1e4_ck200_to500_20260916/checkpoint-500`
in `results/evaluations/ck500_all_images_20260919/checkpoint-500`.
All copied hashes match; 392 adapter tensors passed safetensors header/shape/
byte-span checks. Recorded base is t20 checkpoint-16000, constant denominator
4096 and corrected KL enabled. Adapter SHA256:
`93ec0295781eedafaea33ed659cc925d34885d07ba8553d3eaf9cd2018148010`.

Extended `intermediate_eval.py` to ordinals 1–4, retaining the last complete
image after early termination. `report_intermediate_eval.py` now accepts an
ordinal mapping and checkpoint-step guard, reports actual ordinal coverage,
eligible prompt indices and all-stage scores on matched subsets. Older
first/second configurations remain supported. Four images is the maximum:
draft + three refinements. Therefore up-to-four reuses the original final PNGs
and scores directly; actual-fourth scores are restricted to episodes with four
complete images. The report rejects this reuse if the declared cap differs or
an episode exceeds it. No extra rounds are forced and no policy code changed.

Launch root: `results/evaluations/ck500_all_images_20260919/` contains the
frozen code, preserved checkpoint, `comparison.json`, `launch.json`, all batch
scripts and `prepare_and_submit.py`. Outputs are `<benchmark>_ck500_final` and
`<benchmark>_ck500_image1|2|3`; image 4 references final as explained above.
Logs: `results/logs/ck500_all_images_20260919/`.
Each GPU job: four H100s, 24 CPUs, 256 GB, 02:59:00; online W&B. Sampling,
seed 421, benchmark hashes, first-AR batch 16 and judges match step 400.
Original final generation/scoring implementation is copied unchanged from the
successful instruction-evaluation snapshot. Existing snapshots are untouched.

Submitted:
- GenEval2 final **60464510**; image 1/2/3 **60464511 / 60464512 / 60464513**.
- TIIF final generation/scoring **60464514 / 60464515**; image 1 **60464516 /
  60464517**, image 2 **60464518 / 60464519**, image 3 **60464520 / 60464521**.
- GenAI final generation/scoring **60464522 / 60464523**; image 1 **60464524 /
  60464525**, image 2 **60464526 / 60464527**, image 3 **60464528 / 60464529**.
- Combined report + W&B metrics/visuals **60464530**, CPU, 20 minutes.
Dependencies use afterok, with invalid dependent jobs cancelled. Replay waits
for its source episodes; instruction judging waits for replay; report waits for
all scores. Do not resubmit these while they remain queued or running.

Latest verification after the user's continue: GenEval2 final COMPLETED 0:0
in 1:45:26, all 800 prompts scored (AM 0.6237864901664825,
GM 0.17266202210845086), W&B sync confirmed. TIIF final generation COMPLETED
0:0 in 1:07:35 with 554 images; final scoring and image-1 replay RUNNING.
GenAI final generation COMPLETED 0:0 in 2:46:14 with 1600 images; final
scoring queued. Other per-image jobs and report remain pending. Both log
streams, squeue and sacct were checked; no reported failed job at that check.

Validation: 23 CPU benchmark/ordinal tests passed in the working tree and
frozen snapshot using the actual scratch evaluation Python. Tests cover all
1–4 image/ordinal combinations, early termination, unchanged source/image
integrity, invalid caps, partial results, paired ordinal subsets, legacy
report configs and end-to-end four-image reporting. All 21 batch scripts
passed syntax, dependency-order, checkpoint path and resource checks;
inherited code and benchmark hashes were verified. Full four-image report
and new image-3 replay completion remain outstanding.

## September 18: checkpoint-400 draft / first-refinement comparison

Completion verified: all 11 intermediate evaluation/report jobs 60389092–60389102
COMPLETED with exit 0:0. Full final/first/up-to-two coverage is 800 GenEval2,
554 TIIF and 1600 GenAI prompts each. Saved metrics, both log streams and report
W&B sync were checked. This supersedes the pending/unverified status below.
Results (x100, first / up to two / final): GenEval2 AM
63.2675 / 64.3188 / 62.3025, GM 19.3994 / 19.0437 / 17.1193;
TIIF short 63.8441 / 63.9272 / 62.0289, long 60.6168 / 61.3519 / 61.0521;
GenAI VQAScore 33.8596 / 35.1465 / 32.2372. Actual second-image coverage:
669/800, 175/554 and 882/1600 respectively. On this evaluation, later
refinements reduce aggregate AM/TIIF/GenAI scores relative to up-to-two;
this is one rollout per prompt, not a repeated-seed significance analysis.
Full and matched-subset results are in
`results/evaluations/ck400_intermediate_images_20260918/RESULTS.md` and `results.json`.
Matched-subset TIIF long overall is unavailable because not all nine groups
are represented. W&B run `xiaofeng-zhang/tar_reasoning/ck400_intermediate_images_20260918`
successfully synced the comparison metrics and visual table.

User requested first-image and second-image scores alongside final-episode
scores on GenEval2, TIIF and GenAI-Bench for the **continued t20 LR 1e-4** model.
`llava/train/rl/intermediate_eval.py` replays saved semantic codes from the
existing step-400 evaluations. It never loads/resamples the first AR or forces
generation past EOS. Ordinal 1 is the draft; ordinal 2 is the first refinement,
carrying the draft forward if no second image exists. The report also shows
actual-second coverage and first/second/final scores on that identical subset.
Do not describe the conditional subset as the full benchmark.

Renderer settings and per-prompt seeds match the source evaluations. If the
selected image is already final, its original PNG is copied byte-for-byte;
GenEval2 also reuses its exact saved question scores. TIIF/GenAI use the same
unchanged judges and scoring seeds. Source manifests, episode JSON hashes,
benchmark hashes, selected ordinals and image hashes are checked on replay
and report. Incomplete outputs cannot produce a full comparison.

`scripts/rl_ft/report_intermediate_eval.py` writes `results.json`, `RESULTS.md`
and online W&B metrics plus eight first/second/final visual triplets per
benchmark. New output root and frozen snapshot:
`results/evaluations/ck400_intermediate_images_20260918/`.
`launch.json` records each batch script, dependency and submitted job ID;
`comparison.json` identifies the original final and both intermediate outputs.
GPU stages use four H100s and 02:59:00, depending on successful source jobs.
Active source snapshots, final artifacts and training are not modified.

Submitted jobs: GenEval2 first/second **60389092 / 60389093**;
TIIF first render/score **60389094 / 60389095**, second **60389096 / 60389097**;
GenAI first render/score **60389098 / 60389099**, second **60389100 / 60389101**;
combined CPU/W&B report **60389102**. `squeue` and `sacct` confirmed all
PENDING/Dependency with the requested limits; no stdout/stderr files exist yet.
Replay waits for source generation/evaluation success; scoring waits for replay;
report waits for all intermediate and original final scores (`afterok` throughout).
Invalid dependencies cancel the downstream job. The CPU report has 20 minutes.

Validation: 21 CPU tests passed against the working tree and frozen snapshot
(snapshot used repo tests on PYTHONPATH); seven new tests cover early EOS,
caps, malformed codes, source mutation, exact final-pixel reuse, incomplete
outputs, matched subsets and end-to-end reporting. Bash syntax and hashes of
all inherited snapshot files passed. Actual scratch-environment renderer/judge
imports passed (Torch 2.6.0+cu124, Transformers 4.57.6, PEFT 0.18.1,
torchvision 0.21.0+cu124), after a temporary Lustre read wait. Real source-record
validation also passed for 127 GenEval2, 128 TIIF and 69 GenAI episodes available
at that check; 65/81/39 respectively had no second image. These are partial
generation counts, not benchmark results. GPU replay remains unverified until its
jobs execute. A first attempt to create `tests/test_intermediate_eval.py`
hit the existing `/home` transport error; that path was left untouched. The
working tests are in `tests/test_rl_intermediate_images.py`.

## September 18: continued t20 checkpoint-400 GenEval2 evaluation

User also requested TIIF and GenAI-Bench for this same checkpoint. Submitted
TIIF generation/scoring **60386251 / 60386252**, GenAI generation/scoring
**60386253 / 60386254**, and CPU report **60386255**. Each GPU stage requests
four H100s, 24 CPUs, 256 GB and 02:59:00; online W&B. Scoring depends on its
generation succeeding (`afterok`, invalid dependencies cancel scoring); the
report uses `afterany` on both scoring jobs. Last check: all PENDING, scoring
and report waiting on dependencies, no stdout/stderr yet.

These jobs reuse the preserved step-400 adapter below and the validated,
hash-verified September 16 instruction-evaluation snapshot, with identical
seed 421, sampling and judges as checkpoint 200. TIIF: 277 testmini cases in
short/long registers (554 images); GenAI: all 1600 official image prompts.
Checkpoint hashes, benchmark counts and shell syntax were verified. The report
helper was copied from the previous successful launch and its SHA256 recorded;
no evaluator or training code was changed.

TIIF/GenAI scripts, launch metadata, report helper and eventual `RESULTS.md`:
`results/evaluations/instruction_t20_ck400_20260918/`.
Individual outputs: `results/evaluations/tiif_t20_corrected_lr1e4_ck400_20260918/`
and `results/evaluations/genai_t20_corrected_lr1e4_ck400_20260918/`.
Logs: `results/logs/instruction_t20_ck400_20260918/`.

User explicitly confirmed the continued t20 LR 1e-4 run, not the fresh t21 run.
Submitted **60386005** to evaluate
`results/models/rl_continue_corrected_lr1e4_ck200_to500_20260916/checkpoint-400`
on all 800 GenEval2 prompts. Four H100s, 24 CPUs, 256 GB, **02:59:00**, online
W&B; same seed 421, batch 16, SDPA, episode caps and CFG-4 renderer as the
completed step-200 evaluation. Reuses the verified September 15 evaluator snapshot.

To prevent pruning by active training, copied the adapter/config/state into
`results/launches/geneval2_t20_corrected_lr1e4_ck400_20260918/checkpoint-400`.
All three SHA256 hashes match the source; adapter header/size, step, reward
objective, recorded t20 SFT base and snapshot hashes passed checks. No training
files or active launch snapshots were changed.

Job **60386006** runs after the evaluation terminates and compares its complete
results with the previous corrected LR 1e-4 step-200 result. Batch scripts,
launch provenance, comparison metadata and eventual `RESULTS.md` live in that
same launch directory. Per-prompt outputs are under
`results/evaluations/geneval2_t20_corrected_lr1e4_ck400_20260918/`; stdout/stderr
under the corresponding `results/logs/` directory. Last Slurm/accounting check:
evaluation PENDING, report PENDING/Dependency, no logs yet. GPU completion and
scores remain outstanding.

## September 17: new t21 SFT, fresh RL and three benchmark evaluations

Evaluation completion verified September 18: all six evaluation/report jobs
COMPLETED with exit 0:0. Full coverage: GenEval2 800/800, TIIF 554/554,
GenAI 1600/1600. T21 SFT results: GenEval2 AM 0.567582861963, GM 0.141690321054;
TIIF short 0.586298165931, long 0.591478113816; GenAI VQAScore
0.300628116078 (basic 0.322017593691, advanced 0.282897688022).
`results/evaluations/sft_t21_ck14000_20260917/BENCHMARK_RESULTS.md` contains the
combined report. Stdout completion records, saved metrics and W&B sync logs
were checked. This supersedes the historical evaluation startup status below.

No matching standalone evaluations of the unmodified t20 SFT checkpoint were
found. The earlier three-benchmark scores are for **t20 + corrected LR 1e-4 RL
at step 200**, not pure SFT: GenEval2 AM/GM 0.625910943658/0.181373328724;
TIIF short/long 0.613539411291/0.597094512101; GenAI 0.322213145814.
Their generation settings, benchmark hashes and judge settings match t21's
evaluations, but comparing these models does not isolate an SFT improvement.

Verified readable new SFT checkpoint:
`models/sft_ckpt/slf_ref_edit_t21/checkpoint-14000` (project alias resolves to
`/project/6004852/jeet/models/sft_ckpt/slf_ref_edit_t21/checkpoint-14000`).
All four safetensors shards are present, header offsets fit file sizes, and all
first-AR tensor keys/shapes match Qwen2. Tokenizer vocabulary 217207, EOS 151645,
image vocabulary starts at 151671 with 65536 codes; production rollout special
tokens and prefixes passed. The existing `sft_model` symlink was not retargeted.

Submitted these independent jobs:

| Job | Purpose | Limit / dependency |
|---|---|---|
| 60238112 | Fresh corrected GRPO from t21, LR 1e-4 | 24h, automatic continuation to 500 |
| 60238114 | SFT GenEval2, all 800 prompts | 02:59, independent |
| 60238115 | SFT TIIF generation, 554 images | 02:59, independent |
| 60238116 | TIIF scoring | 02:59, afterok 60238115 |
| 60238119 | SFT GenAI generation, 1600 prompts | 02:59, independent |
| 60238123 | GenAI scoring | 02:59, afterok 60238119 |
| 60238124 | CPU benchmark report | 10m, afterany 60238114/60238116/60238123 |

Each GPU job requests four H100s, 24 CPUs and 256 GB. Last verification:
training RUNNING on fc10210, GenEval2 on fc10114, TIIF generation on fc10109,
GenAI generation on fc10110; scoring/report pending dependencies. Logs exist
and no startup error was present at the 32-second check, but optimizer updates
and complete GPU evaluations had not yet been verified. W&B is configured online.
Dependency failures cancel downstream scoring instead of scoring partial data.

Training starts fresh at RL step 0, target 500, new LoRA r64/alpha128. Same
constant length normalization (4096), corrected KL, KL 0.01, four prompts/GPU,
four rollouts/prompt, cosine LR with 3% warmup, 32-prompt validation/checkpoints
every 25 steps and eight exemplars. The 24-hour wrapper stops/checkpoints with
a 45-minute buffer and submits an afterok successor until 500, stopping on error
or no progress. Prior t20 continuation is independent and was not modified.

Training run: `rl_pixel_sft_t21_ck14000_const_kl_lr1e4_20260917`, with its own
`results/{models,evaluations,logs,wandb}/<run>` directories. All seven batch
scripts, job IDs/settings and immutable source snapshot are under
`results/launches/sft_t21_ck14000_20260917/` (`launch.json`, `code/`). Do not edit
that submitted snapshot. Benchmark metadata/reports are under
`results/evaluations/sft_t21_ck14000_20260917/`; final combined report is
`BENCHMARK_RESULTS.md`. Individual benchmark outputs use
`results/evaluations/{geneval2,tiif,genai}_sft_t21_ck14000_20260917/`.

Added explicit `--checkpoint_type sft` / `CHECKPOINT_TYPE=sft` support to both
evaluation entry points via `evaluation_policy.py`. SFT mode loads and freezes
the complete first AR directly, with SHA256 provenance for all model shards,
config and tokenizer files; RL-adapter mode remains default and retains the
objective/base-model checks. No SFT optimizer state is loaded into fresh RL.
Evaluations use the established one-episode, final-image protocol, seed 421,
four GPUs, generation batch 16, SDPA evaluation environment and existing judges.
TIIF is testmini short+long; GenAI is the official 1600-prompt dataset.

Validation: 18 relevant tests passed (SFT/adapter loading and freezing, weight
fingerprints, missing/misidentified checkpoints, out-of-directory shard guards,
GenEval2 and instruction-benchmark integrity/aggregation). The four new loader
tests also passed in the exact scratch Torch 2.6 / Transformers 4.57.6 evaluation
environment. Real checkpoint shape/token checks, script/Python syntax, snapshot
hashes and fresh-initialization preflight passed. Training code is unchanged
from the continuation snapshot validated by 51 tests earlier on September 17.

## September 17: corrected LR 1e-4 continuation submitted

User authorized finishing the previously prepared continuation from checkpoint
200 to total step 500 (300 additional updates). Submitted **60174686**
(`rl-continue-lr1e4`); verified PENDING, four H100s, 24 CPUs, 256 GB RAM,
24:00:00, account `rrg-bengioy-ad_gpu`. Subsequently verified RUNNING on
`fc10212`, elapsed 2h21m, with training logs through step 218 (18 resumed
updates). Online W&B is syncing and no fatal error markers were found. The
next scheduled checkpoint and validation are at step 225; only the seed
checkpoint-200 existed at this check. Step 500 remains outstanding.

Run name: `rl_continue_corrected_lr1e4_ck200_to500_20260916`.
`results/launches/<run>/launch.json` records the first job ID, source checkpoint,
snapshot and settings; `resume.sbatch` is the submitted script. Checkpoints,
evaluations, logs and online W&B files use `results/{models,evaluations,logs,wandb}/<run>`.
The copied seed checkpoint is independent of the original run; the original
checkpoint-200 is preserved. Do not overwrite the submitted code snapshot.

The original Python 3.10 / Torch 2.1.2 / Transformers 4.50 / FlashAttention 2
training environment passed imports and regressions. Optimizer entries and
scheduler are all at step 200; saved LR is 6.819524684817438e-5, base LR 1e-4.
Resume restores optimizer and cosine scheduler without new warmup. Method flags
remain constant normalization (4096) and corrected KL. Same four prompts/GPU,
four rollouts/prompt, KL 0.01, EOS/cap policy and frozen pixel judge/renderer.

The prepared trainer supports coordinated stopping at optimizer boundaries,
atomic checkpoint publication, and final checkpoint/status publication. The
batch wrapper requests a safe stop 45 minutes before the 24-hour limit. A
successful segment below 500 submits one successor; no-progress, missing-state,
training-error and ambiguous-submission cases stop automatic continuation.
Fixed a successor race by adding `afterok:<parent-job-id>` to sbatch, so a new
segment cannot start before its parent releases the training-directory lock.
Successor IDs are in `results/models/<run>/successor_<parent-job>.json`; each
segment writes `training_status_<job>.json`. Reaching 500 submits no successor.

Validation: 51 CPU RL tests passed against the snapshot in the original training
environment, including both distributed checks, checkpoint integrity and the
new parent-dependency / duplicate-submission / ambiguous-submission checks.
Snapshot hashes (55 files), shell syntax, SFT/method preflight, and saved
optimizer/scheduler state passed. The first test invocation used shared `/tmp`
as CWD and hit another `profile.py` shadowing the standard library; rerunning
from the actual snapshot job directory resolved it without changing that file.

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

Completion verified September 17: evaluation jobs 59949648–59949651,
report job 59949652 and monitor job 59951777 all COMPLETED with exit 0:0.
Each evaluation covered 800/800 prompts in approximately 1h38–1h40, within
02:59:00. Stdout final metrics match the saved comparison; stderr confirms W&B
sync completed. AM/GM percentages: early step25 56.3984/14.5812; legacy LR1e-5
58.3426/15.3610; corrected LR1e-5 57.2754/14.1300; corrected LR1e-4
62.5911/18.1373. The corrected LR1e-4 run scored highest among these four.
This completion supersedes the historical running/partial status below.

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
