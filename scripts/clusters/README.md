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

Queue RL training (one node, four full H100 80 GB GPUs, 24 CPUs,
256 GB host RAM, **seven-day limit**, online W&B, four prompts per GPU):

```bash
RUN_NAME=rl_pixel_sft_t20_val32 \
  bash scripts/clusters/fir/submit_rl.sh --job-name=tar-rl
```

Authenticate with W&B before submission. The wrapper preserves explicit
`JOB_TIME`, `WANDB_MODE`, and `PROMPTS_PER_GPU` overrides. It uses the current
SFT symlink and the fixed 32-prompt validation YAML. Use a fresh run name unless
intentionally resuming. The general submit wrapper retains its three-hour default
for other workloads. A seven-day allocation is a maximum runtime; the trainer
still stops at its configured `MAX_STEPS` (default 500).

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

W&B defaults to online mode; configure your own credentials to upload.
Set `WANDB_MODE=offline` explicitly when needed. Temporary files use `SLURM_TMPDIR` in allocations,
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

## uv mode and cluv (portable layout)

Every cluster gets one folder, `$SCRATCH/tar_reason` (`TAR_ROOT`):

| Path | Content | Synced |
|---|---|---|
| `shared/` | Inputs: GenEval2 training prompts, benchmarks, renderer, SFT base, judges, SigLIP2 config. `MANIFEST.json` holds sizes and sha256; `PROVENANCE.json` maps each input to its original path | Pushed from Fir to every cluster by `cluv sync` (rsync, never `--delete`) |
| `results/` | Outputs (`models/`, `evaluations/`, `logs/`, `wandb/`, `launches/`) | Fetched back to Fir by `cluv sync`; on Fir a link to `$SCRATCH/tar_reason_results` |
| `env/` | `venvs/<uv.lock sha256[:12]>`, one immutable venv per lock, and `uv_cache/` | Rebuilt per cluster (never copied) |
| `code/<commit>` | Checkouts made by `cluv_job.sh` | No |
| `cache/` | wandb, triton, torch caches | No |

Outputs only flow back to the hub, so another cluster's file quota never sees
Fir's ~800K result files. A run moves to another cluster by publishing its
checkpoint: `shared_inputs.py publish --dest $TAR_ROOT/shared CKPT checkpoints/RUN/checkpoint-N`.

### One-time setup

On Fir (the hub):
1. `TAR_ROOT=$SCRATCH/tar_reason bash scripts/clusters/setup_uv.sh`. This writes
   `~/.config/uv/uv.toml` with `cache-dir` on scratch, builds the lock-hash venv,
   and links `.venv` to it.
2. `python3 scripts/clusters/shared_inputs.py build --dest $SCRATCH/tar_reason/shared --sources scripts/clusters/fir/shared_sources.txt`.
   Run it as a CPU job: it copies about 23 GB and hashes 112 GB.
3. Install cluv (`uv tool install git+https://github.com/mila-iqia/cluv@e17dd88`),
   then `cluv login <cluster>` for each cluster (SSH ControlMaster with MFA).

On each other cluster:
1. Clone the repository to the path cluv expects. On Trillium and Killarney that is
   `$SCRATCH/tar_reason/project`: jobs cannot be submitted from `/home`.
2. Run `setup_uv.sh` there **before** the first `cluv sync`. Otherwise cluv's own
   `uv sync` creates `.venv` in `$HOME`.
3. Confirm the allocation with `sshare -U`. Put `SLURM_ACCOUNT=...` (and
   `TAR_CPU_ACCOUNT=...`) in `scripts/clusters/<cluster>/config.local.sh`.
4. From Fir, run `cluv sync`. Then, on the cluster:
   `python3 scripts/clusters/shared_inputs.py verify --dest $SCRATCH/tar_reason/shared [--full]`.

### Using it

- **cluv submission:** `cluv submit <cluster> [sbatch flags] -- bash scripts/rl_ft/bash.sh`
  runs at the pushed commit through `scripts/clusters/cluv_job.sh`.
- **Training with cluv chunking:** `cluv submit <cluster> --chunking 3 --time=<total> --ntasks=4
  --gpus-per-task=<type>:1 --cpus-per-task=6 --mem-per-gpu=64G --switches=1@00:30:00
  --signal=B:USR1@600 -- bash scripts/rl_ft/array_element.sh scripts/rl_ft/configs/<run>.env`.
  cluv turns this into `--time=03:00:00 --array=0-K%1`. `array_element.sh` resumes, stops
  before the limit, and cancels the rest once the run is complete or stalled. Run
  settings live in a committed `.env` file (see `scripts/rl_ft/configs/`), because
  cluv does not forward the local environment.
- **Direct submission:** the repo submitters (`scripts/rl_ft/submit_short_array.sh`,
  `scripts/eval/looks_good_suite.py`) work on any cluster with `TAR_ROOT` set. They
  freeze code and settings into `results/launches/` or the suite directory.
- **cluv skips unchanged pushes:** it skips a push when `shared/`'s *top-level*
  mtime predates its last push. `shared_inputs.py build`/`publish` touch it; after
  manual changes, `touch $SCRATCH/tar_reason/shared`. The same applies after a
  scratch purge (DRAC: files unused for about 60 days); `verify` reports it.
- **W&B on clusters without internet:** compute nodes on Rorqual, Narval, Trillium
  and tamIA have none, so jobs log offline. Upload with `wandb sync` from Fir after
  the results are fetched.
