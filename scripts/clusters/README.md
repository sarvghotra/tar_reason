# Running this repository across clusters

Keep code in the checkout, large inputs in shared/project storage, and outputs
in scratch or persistent experiment storage. Each checkout selects a cluster
profile and provides the same local links. Training code should never need a
cluster's absolute storage path.

## Filesystem layout

| Repo path | Purpose | Fir target |
|---|---|---|
| `data/` | Dataset root | `/project/rrg-bengioy-ad/jeet/data` |
| `models/` | Base model and visual tokenizer/decoder weights | `/project/rrg-bengioy-ad/jeet/models` |
| `sft_model/` | Exact initial SFT checkpoint | `models/sft_ckpt/slf_ref_edit_t6_repro_w_corr_t2/checkpoint-24000` in shared storage |
| `reward_model/` | Frozen Qwen3-VL-8B-Instruct checkpoint | `results/pretrained/Qwen3-VL-8B-Instruct` |
| `results/` | All run outputs and caches | `/scratch/$USER/tar_reason_results` |
| `tar/` | Python environment for Tar | Recreate on each cluster; existing Fir environment is Python 3.10 |
| `qwen_reward/` | Separate Python environment for Qwen3-VL | Fir link to `results/environments/qwen_reward` |

The project alias may resolve to `/project/6004852` on Fir; these are the same
shared files. The setup utility checks every target before modifying links,
leaves matching links untouched, and never replaces real files/directories.
Use `--replace-links` to deliberately retarget existing symlinks. This changes
links only; it never moves or deletes their targets.

Links, `.cluster`, environments, outputs, and `config.local.sh` are ignored by
Git. They must be configured separately in each checkout. Do not copy a virtual
environment between clusters: Python, CUDA, modules and compiled extensions can
differ. Scratch files may be purged according to the site's policy; keep durable
checkpoints in suitable persistent storage.

Dataset YAMLs use `data/...` relative to the repo. Launchers change into the
repo before reading them. Existing absolute paths *inside* dataset JSON/JSONL
records are not automatically rewritten by creating a symlink.

## Cluster folders

- `fir/`: configured Alliance Fir modules, GPU account, link targets, and submit wrapper.
- `generic/`: placeholder for the other cluster. No account or module assumptions.
- `profile.sh`, `setup_links.py`: shared profile selection and safe link setup.

`.cluster` records the selected profile; `TAR_CLUSTER` overrides it. Without
either, the generic profile is selected and submission requires an account.
A profile's optional `config.local.sh` can override its settings without putting
machine-specific configuration in Git.

## Set up Fir

From the repo root:

```bash
bash scripts/clusters/fir/setup.sh
TAR_DRY_RUN=1 bash scripts/rl_ft/bash.sh
```

Existing Fir data, model, reward-model and results targets are required.
Override targets with `DATA_TARGET`, `MODELS_TARGET`, `SFT_TARGET`,
`REWARD_TARGET`, and `RESULTS_TARGET`. Create/download missing targets first.
The setup command does not download large inputs or create environments.

The Tar launchers load `StdEnv/2023 python/3.10 cuda/12.2` and activate `tar/`.
The judge uses `qwen_reward/bin/python`, independently of Tar's environment.
See [the detailed environment/reward guide](../CLUSTER.md) for installation,
Qwen preparation, reward semantics and legacy evaluation limitations.
`TAR_ENV` and `REWARD_PYTHON` can override the environment locations.

## Submit jobs

Standalone batch jobs survive terminal disconnections. There is no need for an
interactive allocation, a cron monitor, or a dependent trigger for these jobs.

Check the submission command without allocating resources:

```bash
TAR_DRY_RUN=1 bash scripts/clusters/fir/submit.sh scripts/rl_ft/bash.sh
```

Queue the GPU integration smoke test (one full H100 80 GB; no optimizer updates):

```bash
N_GPUS=1 CPUS_PER_TASK=8 JOB_MEM=64G JOB_TIME=01:00:00 \
  bash scripts/clusters/fir/submit.sh scripts/test_qwen_reward_batch.sh \
  --job-name=tar-qwen-smoke
```

Queue RL training (default: one node, four full H100 80 GB GPUs, 24 CPUs,
256 GB host RAM, three-hour limit):

```bash
RUN_NAME=rl_pixel_sft24k \
  bash scripts/clusters/fir/submit.sh scripts/rl_ft/bash.sh --job-name=tar-rl
```

Override `N_GPUS`, `GPU_TYPE`, `CPUS_PER_TASK`, `JOB_MEM`, `JOB_TIME`, or
`SLURM_ACCOUNT` as needed. Prefer `N_GPUS` to changing only sbatch's GPU flag,
so allocation and torchrun process counts agree. Trailing arguments are sbatch
options, not trainer arguments. For trainer settings use the launcher's environment
variables, such as `MAX_STEPS`, `EVAL_STEPS`, `NUM_ROLLOUTS`, and `MAX_REFINEMENTS`.
The four-GPU configuration is the intended training configuration; successful
full-model GPU memory/runtime validation must be established by actual tests.

Use a new `RUN_NAME` for a new experiment. Reusing a name resumes compatible
checkpoints in that run's results directory. Old semantic-reward checkpoints
cannot resume under the new pixel Qwen reward objective.

## Outputs and status

| Output | Location |
|---|---|
| Adapters, optimizer state, checkpoints | `results/models/<run>/` |
| Run console logs | `results/logs/<run>/` |
| Slurm stdout/error logs | `results/logs/slurm/<job-name>-<job-id>.out` and `.err` |
| Evaluations, final-image rewards, visual exemplars | `results/evaluations/<run>/` |
| GPU smoke-test evidence | `results/evaluations/qwen_reward_smoke/` |
| W&B files | `results/wandb/<run>/` |
| HF, Triton, uv and other caches | `results/cache/` |

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Start,End
# Logs normally appear once the batch job starts:
tail -f results/logs/slurm/tar-rl-JOB_ID.out
tail -f results/logs/slurm/tar-rl-JOB_ID.err
```

W&B defaults to offline mode. Set `WANDB_MODE=online` and configure your own
credentials to upload. Temporary files use `SLURM_TMPDIR` in allocations,
otherwise `results/tmp/`. An empty error log alone is not proof of success;
check the job exit state and the test's saved report.

## Placeholder: configure the other cluster with Codex

On the other cluster, ask Codex to inspect available storage, modules, Python,
CUDA, Slurm accounts and GPU types, then configure this placeholder. No settings
for that cluster have been guessed here.

1. Copy `scripts/clusters/generic/` to `scripts/clusters/<cluster-name>/`.
2. Create `config.local.sh` from `config.local.sh.example` and fill in the site's
   account/modules/GPU type. Keep secrets out of it.
3. Copy or download data and weights to that cluster's storage. Preserve the
   relative layout under `data/` and `models/`, or explicitly override the
   relevant launcher variables. Create the results target directory.
4. Configure the links:

   ```bash
   DATA_TARGET=/path/to/data MODELS_TARGET=/path/to/models \
   SFT_TARGET=/path/to/checkpoint-24000 REWARD_TARGET=/path/to/Qwen3-VL-8B-Instruct \
   RESULTS_TARGET=/path/to/results \
     bash scripts/clusters/<cluster-name>/setup.sh
   ```

5. Recreate the Tar and Qwen environments for that site. Use compatible pinned
   dependencies as a starting point, not precompiled environments from Fir.
6. Run input checks and submission dry runs, then the one-GPU smoke test before
   starting a full training run.

Transfer code through Git. Symlinks and ignored input/output trees are not
transferred with it; recreate links on the destination. Copy actual datasets
and checkpoints separately when they are not already available there.
