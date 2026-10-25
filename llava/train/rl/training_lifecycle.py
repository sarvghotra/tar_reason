"""CPU-only deadline and complete-checkpoint publication helpers."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
import time


def stop_requested(stop_at_unix=0, stop_file=None, now=None):
    now = time.time() if now is None else now
    return bool((stop_at_unix and now >= stop_at_unix) or
                (stop_file and Path(stop_file).exists()))


@contextmanager
def checkpoint_directory(output_dir, step):
    """A partial save never appears as a resumable checkpoint-N directory."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'checkpoint-{step}'
    if target.exists():
        raise FileExistsError(f'Refusing to replace existing checkpoint: {target}')
    temporary = Path(tempfile.mkdtemp(prefix=f'.checkpoint-{step}-', dir=root))
    try:
        yield str(temporary)
        required = ['state.json', 'adapter_config.json', 'adapter_model.safetensors',
                    'optimizer.pt', 'scheduler.pt']
        for name in required:
            path = temporary/name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f'Incomplete checkpoint: missing/empty {name}')
        # Flush all regular files before atomically publishing the directory.
        for path in temporary.iterdir():
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        os.rename(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


# ---------------------------------------------------------------------------
# Short (e.g. 3-hour) jobs: step-aware deadline and resume checkpoints
# ---------------------------------------------------------------------------

RESUME_SUBDIR = 'resume'


class StepClock:
    """Recent step, validation and save durations (seconds) for deadline decisions.

    Carried in each checkpoint's state.json so the first step after a resume
    already has an estimate.
    """

    def __init__(self, timing=None, window=5):
        timing = timing or {}
        self.window = window
        self.steps = [float(t) for t in timing.get('step_seconds', [])][-window:]
        self.validation = timing.get('validation_seconds')
        self.save = timing.get('save_seconds')

    def add_step(self, seconds):
        self.steps = (self.steps + [float(seconds)])[-self.window:]

    def to_dict(self):
        return dict(step_seconds=self.steps, validation_seconds=self.validation, save_seconds=self.save)


def deadline_allows_step(now, job_end_unix, clock, validation_due=False, margin=1.25,
                         default_save_seconds=120.0):
    """Whether one more step, its validation (if due) and the exit save fit before the job ends.

    The step estimate is the longest of the recent steps; an unmeasured
    validation is assumed to take as long. No job end, or no step measured
    yet, never stops.
    """
    if not job_end_unix or not clock.steps:
        return True
    step = max(clock.steps)
    need = margin * step
    if validation_due:
        need += margin * (step if clock.validation is None else clock.validation)
    need += default_save_seconds if clock.save is None else margin * clock.save
    return now + need <= job_end_unix


def checkpoint_dirs(output_dir):
    """(step, path) of every published checkpoint, milestone or resume, oldest first."""
    found = []
    for root in (Path(output_dir), Path(output_dir)/RESUME_SUBDIR):
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.name.startswith('checkpoint-') and (path/'state.json').is_file():
                try:
                    found.append((int(path.name.split('-', 1)[1]), str(path)))
                except ValueError:
                    pass
    # A milestone and a resume copy of the same step are equivalent; prefer the milestone.
    found.sort(key=lambda item: (item[0], RESUME_SUBDIR + os.sep not in item[1]))
    return found


def prune_resume_checkpoints(output_dir, keep_step):
    """Remove resume checkpoints other than `keep_step`; milestones are never touched."""
    root = Path(output_dir)/RESUME_SUBDIR
    if not root.is_dir():
        return
    for path in root.iterdir():
        if path.name.startswith('checkpoint-') and path.name != f'checkpoint-{keep_step}':
            shutil.rmtree(path, ignore_errors=True)
