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
