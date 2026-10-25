"""Build, verify and extend the synced input folder ($TAR_ROOT/shared).

cluv pushes this folder from the hub cluster to every other cluster (rsync
without --delete). MANIFEST.json records the size and sha256 of every file so
each cluster can check its copy (truncated or purged files included).

  build   --dest D --sources scripts/clusters/<hub>/shared_sources.txt
  verify  --dest D [--full]          sizes always; --full also re-hashes
  publish --dest D SRC REL [--model] add one input (e.g. a checkpoint) and update the manifest

Sources file lines: "<mode> <source path> <relative destination>", where mode is
copy | link (hard link, same filesystem only) and a "-model" suffix keeps only
the Hugging Face loading files of a model directory. Publishing or building
touches the folder itself, because cluv skips a push when the top-level mtime
predates its last push.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

MANIFEST = 'MANIFEST.json'
PROVENANCE = 'PROVENANCE.json'   # relative destination -> original source (base-model remapping)
MODEL_SUFFIXES = ('.json', '.safetensors', '.txt', '.model', '.jinja', '.tiktoken')
MODES = ('copy', 'link', 'copy-model', 'link-model')


def parse_sources(text):
    entries = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3 or parts[0] not in MODES:
            raise ValueError(f'line {number}: expected "<{"|".join(MODES)}> <source> <relative destination>"')
        mode, source, rel = parts
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError(f'line {number}: destination must be relative and inside the folder')
        entries.append((mode, Path(os.path.expandvars(source)), rel))
    return entries


def source_files(source, model_only):
    """(relative path, absolute path) of the files to place, following symlinks."""
    if source.is_file():
        return [('', source)]
    files = []
    for root, _, names in os.walk(source, followlinks=True):
        for name in sorted(names):
            path = Path(root)/name
            if path.is_file() and (not model_only or name.endswith(MODEL_SUFFIXES)):
                files.append((str(path.relative_to(source)), path))
    if not files:
        raise ValueError(f'No files to place from {source}')
    if model_only and not any(rel.endswith('.safetensors') for rel, _ in files):
        raise ValueError(f'No safetensors weights in model directory {source}')
    return sorted(files)


def place(mode, source, destination):
    """Place source at destination atomically; existing complete copies are kept."""
    model_only = mode.endswith('-model')
    link = mode.startswith('link')
    files = source_files(source, model_only)
    target = lambda root, rel: root/rel if rel else root
    if destination.exists():
        if all(target(destination, rel).is_file() and target(destination, rel).stat().st_size == path.stat().st_size
               for rel, path in files):
            return 'kept'
        raise FileExistsError(f'{destination} exists but differs from {source}; remove it explicitly first')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.incoming-', dir=destination.parent))
    try:
        built = staging/destination.name
        for rel, path in files:
            out = target(built, rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            if link:
                if os.stat(path).st_dev != os.stat(staging).st_dev:
                    raise ValueError(f'Cannot hard-link across filesystems: {path}; use copy')
                os.link(path, out)
            else:
                shutil.copyfile(path, out)
            if out.stat().st_size != path.stat().st_size:
                raise OSError(f'Size mismatch after placing {path}')
        os.rename(built, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 'linked' if link else 'copied'


def sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while block := stream.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def tree(dest):
    dest = Path(dest)
    return sorted(str(p.relative_to(dest)) for p in dest.rglob('*')
                  if p.is_file() and p.name not in (MANIFEST, MANIFEST + '.tmp')
                  and not any(part.startswith('.incoming-') for part in p.parts))


def write_manifest(dest, workers=8, previous=None):
    """Hash every file (reusing unchanged entries of `previous` by size and mtime)."""
    dest = Path(dest)
    previous = previous or {}
    def entry(rel):
        stat = (dest/rel).stat()
        old = previous.get(rel)
        if old and old['size'] == stat.st_size and old.get('mtime_ns') == stat.st_mtime_ns:
            return rel, old
        return rel, dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256(dest/rel))
    with ThreadPoolExecutor(workers) as pool:
        files = dict(pool.map(entry, tree(dest)))
    temporary = dest/(MANIFEST + '.tmp')
    temporary.write_text(json.dumps(dict(version=1, files=files), indent=1, sort_keys=True))
    os.replace(temporary, dest/MANIFEST)
    os.utime(dest)  # cluv only pushes when the top-level mtime is newer than its last push
    return files


def load_manifest(dest):
    path = Path(dest)/MANIFEST
    return json.loads(path.read_text())['files'] if path.is_file() else {}


def verify(dest, full=False, workers=8):
    """Problems (missing, extra, size or hash mismatch) relative to MANIFEST.json."""
    dest = Path(dest)
    expected = load_manifest(dest)
    if not expected:
        return [f'{dest/MANIFEST} is missing or empty']
    present = set(tree(dest))
    problems = [f'missing: {rel}' for rel in sorted(set(expected) - present)]
    problems += [f'not in manifest: {rel}' for rel in sorted(present - set(expected))]
    def check(rel):
        size = (dest/rel).stat().st_size
        if size != expected[rel]['size']:
            return f'size {size} != {expected[rel]["size"]}: {rel}'
        if full and sha256(dest/rel) != expected[rel]['sha256']:
            return f'sha256 mismatch: {rel}'
        return None
    with ThreadPoolExecutor(workers) as pool:
        problems += [p for p in pool.map(check, sorted(present & set(expected))) if p]
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--sources', required=True, type=Path)
    v = sub.add_parser('verify')
    v.add_argument('--full', action='store_true')
    u = sub.add_parser('publish')
    u.add_argument('source', type=Path)
    u.add_argument('rel')
    u.add_argument('--model', action='store_true', help='keep only Hugging Face loading files')
    u.add_argument('--link', action='store_true', help='hard-link instead of copying')
    for q in (b, v, u):
        q.add_argument('--dest', required=True, type=Path)
        q.add_argument('--workers', type=int, default=8)
    args = p.parse_args(argv)
    if args.command == 'verify':
        problems = verify(args.dest, args.full, args.workers)
        for problem in problems:
            print(problem)
        print(f'{args.dest}: ' + ('OK' if not problems else f'{len(problems)} problem(s)'))
        return 1 if problems else 0
    args.dest.mkdir(parents=True, exist_ok=True)
    if args.command == 'build':
        entries = parse_sources(args.sources.read_text())
    else:
        entries = [(('link' if args.link else 'copy') + ('-model' if args.model else ''), args.source, args.rel)]
    provenance_path = args.dest/PROVENANCE
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    for mode, source, rel in entries:
        print(f'{place(mode, source, args.dest/rel):>7}  {rel}  <-  {source}', flush=True)
        provenance[rel] = dict(source=str(source), resolved=str(source.resolve()))
    provenance_path.write_text(json.dumps(provenance, indent=1, sort_keys=True))
    files = write_manifest(args.dest, args.workers, load_manifest(args.dest))
    print(f'{args.dest/MANIFEST}: {len(files)} files, {sum(f["size"] for f in files.values()) / 1e9:.1f} GB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
