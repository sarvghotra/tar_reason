# TamIA

Owned by the TamIA session. **Seeded from Fir on 2026-10-09**; the TamIA session should rewrite this file from
its own progress (and its `TAMIA_HANDOFF.md`, archived in `archive/`), then delete this line.

## Now (as last known on Fir, 2026-10-09)
- Setting up the t7 recipe; `scripts/rl_ft/bash_tamia_t7.sh` pushed (`f473e8c`), `RUN_NAME` export fixed by Fir (`b0c8c31`).
- Goal: 1-node smoke test (`RUN_NAME=t7_smoke_tamia`), report to the user (s/step, peak memory), then stop.
  **No full run on TamIA unless the user asks**: `darshan_t7_fir` already runs the same recipe on Fir.

## Setup
- User `dars11`; `$SCRATCH` = `/scratch/d/dars11`; `~/scratch` → `$SCRATCH` (symlink, created 2026-10-09);
  home `/home/d/dars11`. Paths differ from Fir by the extra `d/` level.
- Account `aip-agrawal` (not in `aip-bengioy`). `sbatch --test-only` for 1 node × 4 H100, 1 h estimated a start
  ~30 min out (partition `gpubase_bynode_b1`).
- Setup recipe (envs from the Fir freeze files, weights, data from Fir, script changes): `archive/TAMIA_HANDOFF.md` steps 1–8.

## Inbox
- 2026-10-09 from Fir: pull `darshan-rl-test` before submitting anything. `b0c8c31` fixes `bash_tamia_t7.sh`:
  `RUN_NAME` was not exported into the `srun bash -c` block, so `--run_name` reached `train_grpo.py` empty (it
  killed Fir's first full run; the smoke test can't catch it because it sets `RUN_NAME` at submit). Check line
  125 reads `export RUN_NAME REPO MODELS ...`.
- 2026-10-09 from Fir: notes moved to `tar_reason/handoff/` (see CLAUDE.md "How the sessions coordinate"). Replace
  your `~/scratch/git`-level `CLAUDE.md` with a symlink to `tar_reason/handoff/CLAUDE.md` and keep this file current.
- 2026-10-09 from Fir: Fir smoke-test reference (`t7_smoke_fir`): judge ready 260 s, policy load ~5.5 min, 38 min
  total; rollout ~170 s, reward ~81 s, train ~35 s = ~4.8 min/step; KL ~0.015, grad norm ~0.1.

## Log (newest first)
- 2026-10-09: `bash_tamia_t7.sh` added (`f473e8c`).
