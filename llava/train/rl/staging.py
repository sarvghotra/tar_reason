"""Node-local copies of frozen model inputs.

Every rank on a node used to read the same ~37 GB (policy, judge, renderer)
from network storage at start-up. With TAR_STAGE_DIR set (for example
$SLURM_TMPDIR/stage), the first process on the node to take the node-local
lock copies each input once; the others wait and then load the copy, which is
also shared through the page cache. Recorded paths (checkpoints, adapter
configs, W&B) keep using the canonical paths; only loading uses the copies.
"""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

MARKER = '.staged.json'


# A Hugging Face model directory needs only these; SFT checkpoints also hold
# optimizer/RNG state that from_pretrained never reads.
MODEL_DIR_SUFFIXES = ('.json', '.safetensors', '.txt', '.model', '.jinja', '.tiktoken')


def _files(source):
    """(relative path, absolute source, size) of the files to stage, following symlinks."""
    source = Path(source)
    if source.is_file():
        return [(source.name, source, source.stat().st_size)]
    out = []
    for root, _, names in os.walk(source, followlinks=True):
        for name in sorted(names):
            path = Path(root)/name
            if path.is_file() and name.endswith(MODEL_DIR_SUFFIXES):
                out.append((str(path.relative_to(source)), path, path.stat().st_size))
    if not any(rel.endswith('.safetensors') for rel, _, _ in out):
        raise ValueError(f'No safetensors weights to stage in {source}')
    return sorted(out)


def _destination(stage_root, source):
    digest = hashlib.sha1(str(Path(source).resolve()).encode()).hexdigest()[:12]
    return Path(stage_root)/digest/Path(source).name


def _complete(destination, files):
    marker = destination.parent/MARKER
    if not marker.is_file() or not destination.exists():
        return False
    recorded = json.loads(marker.read_text())
    expected = {rel: size for rel, _, size in files}
    if recorded != expected:
        return False
    root = destination if destination.is_dir() else destination.parent
    names = {rel: (root/rel if destination.is_dir() else destination) for rel in expected}
    return all(path.is_file() and path.stat().st_size == expected[rel] for rel, path in names.items())


def _copy(source, destination, files, workers):
    staging = Path(tempfile.mkdtemp(prefix='.incoming-', dir=destination.parent))
    try:
        if Path(source).is_file():
            targets = [(path, staging/Path(source).name) for _, path, _ in files]
        else:
            targets = [(path, staging/Path(source).name/rel) for rel, path, _ in files]
        for _, target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max(1, workers)) as pool:
            list(pool.map(lambda pair: shutil.copyfile(*pair), targets))
        for (_, _, size), (_, target) in zip(files, targets):
            if target.stat().st_size != size:
                raise OSError(f'Staged copy size mismatch: {target}')
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        os.rename(staging/Path(source).name, destination)
        (destination.parent/MARKER).write_text(json.dumps({rel: size for rel, _, size in files}))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def stage_inputs(paths, stage_root=None, workers=8, log=print):
    """Map each canonical input path to a verified node-local copy.

    Returns {canonical: path to load}. Without a stage root, or when local disk
    lacks space, inputs map to themselves (loaded from shared storage).
    """
    stage_root = stage_root or os.environ.get('TAR_STAGE_DIR')
    paths = [str(p) for p in dict.fromkeys(p for p in paths if p)]
    if not stage_root:
        return {p: p for p in paths}
    root = Path(stage_root)
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    mapping, copied = {}, 0
    with open(root/'.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one copier per node; the rest wait, then reuse
        try:
            for path in paths:
                files = _files(path)
                destination = _destination(root, path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not _complete(destination, files):
                    need = sum(size for _, _, size in files)
                    if shutil.disk_usage(root).free < need * 1.05 + (1 << 30):
                        log(f'[stage] not enough local space for {path}; loading it from shared storage')
                        mapping[path] = path
                        continue
                    _copy(path, destination, files, workers)
                    copied += need
                mapping[path] = str(destination)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    log(f'[stage] {len(paths)} input(s) ready under {root} in {time.monotonic() - started:.1f}s '
        f'({copied / 1e9:.1f} GB copied by this process)')
    return mapping
