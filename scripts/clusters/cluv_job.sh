#!/bin/bash
# cluv job script ([tool.cluv] job_script_path): run a command at the submitted commit.
#
#   cluv submit <cluster> [sbatch flags] -- bash scripts/rl_ft/eval_geneval2.sh ...
#
# Unlike cluv's safe_job.sh, the commit is checked out once per cluster under
# $TAR_ROOT/code/<commit> (immutable, on shared storage, visible to every node)
# and runs with the venv of its uv.lock at $TAR_ROOT/env/venvs/<lock hash>. No
# per-job copy of the venv is made, so start-up stays short. The command runs
# once on the batch node; multi-GPU launchers start their own `srun` steps
# (LAUNCHER=srun). Resource flags come from cluv's sbatch_args or the CLI.
set -euo pipefail
: "${GIT_COMMIT:?GIT_COMMIT is not set; submit this script with cluv submit}"
: "${TAR_ROOT:?Set TAR_ROOT (the [tool.cluv.env] default is \$SCRATCH/tar_reason)}"
[[ "$GIT_COMMIT" =~ ^[0-9a-f]{7,40}$ ]] || { echo "Bad GIT_COMMIT: $GIT_COMMIT" >&2; exit 1; }
PROJECT="${SLURM_SUBMIT_DIR:?}"        # cluv runs `cd <project_dir> && sbatch ...`
CODE="$TAR_ROOT/code/$GIT_COMMIT"
mkdir -p "$TAR_ROOT/code" "$TAR_ROOT/env/venvs"

# One checkout per commit; concurrent jobs wait for the first.
exec 8>"$TAR_ROOT/code/.lock"
flock 8
if [[ ! -f "$CODE/.tar_ready" ]]; then
    rm -rf "$CODE.incoming"
    git clone --quiet --no-checkout "$PROJECT" "$CODE.incoming"
    git -C "$CODE.incoming" checkout --quiet --detach "$GIT_COMMIT"
    ln -s "$TAR_ROOT/shared/data" "$CODE.incoming/data"   # data YAMLs name data/...
    ln -s "$TAR_ROOT/results" "$CODE.incoming/results"
    mv "$CODE.incoming" "$CODE"
    touch "$CODE/.tar_ready"
fi
flock -u 8

# One venv per uv.lock, built offline from the uv cache that `cluv sync` filled.
VENV="$TAR_ROOT/env/venvs/$(sha256sum "$CODE/uv.lock" | cut -c1-12)"
exec 7>"$TAR_ROOT/env/venvs/.lock"
flock 7
if [[ ! -f "$VENV/.tar_ready" ]]; then
    UV_PROJECT_ENVIRONMENT="$VENV" UV_CACHE_DIR="${UV_CACHE_DIR:-$TAR_ROOT/env/uv_cache}" \
        uv --directory "$CODE" sync --frozen --offline
    touch "$VENV/.tar_ready"
fi
flock -u 7

export TAR_ENV="$VENV" VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" TAR_REPO_ROOT="$CODE"
export TAR_STAGE_DIR="${TAR_STAGE_DIR:-${SLURM_TMPDIR:+$SLURM_TMPDIR/stage}}"
cd "$CODE"
echo "commit $GIT_COMMIT  venv $VENV  node $(hostname)  start $(date -Is)"
exec "$@"
