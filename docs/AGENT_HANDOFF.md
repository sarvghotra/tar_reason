# Agent handoff — September 11, 2026

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

No full RL training run has been submitted during this work. Do not claim a
successful GPU test based only on a completed trigger job or an empty error log.
