"""Download pinned public benchmark inputs/judges without changing training envs."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


def download_repo(repo, destination, dataset=False, names=None):
    kind = 'datasets' if dataset else 'models'
    with urllib.request.urlopen(f'https://huggingface.co/api/{kind}/{repo}?blobs=true') as response:
        info = json.load(response)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'download_provenance.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['revision'] != info['sha']:
            raise ValueError(f'{destination} has a different revision; choose a fresh directory')
    revision = info['sha']
    prefix = 'datasets/' if dataset else ''
    files = [f for f in info['siblings'] if (f['rfilename'] in names if names else
             '/' not in f['rfilename'] and f['rfilename'].endswith(('.json', '.safetensors', '.txt', '.jinja')))]
    if not files:
        raise ValueError('No matching files')

    def fetch(item):
        name = item['rfilename']
        path = destination / name
        size = item.get('size', item.get('lfs', {}).get('size'))
        if not path.exists() or (size is not None and path.stat().st_size != size):
            temporary = path.with_name(path.name + '.partial')
            subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--retry-delay', '5',
                            '--connect-timeout', '30', '--silent', '--show-error', '-C', '-',
                            '-o', str(temporary),
                            f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}'], check=True)
            if size is not None and temporary.stat().st_size != size:
                raise ValueError(f'Wrong download size: {name}')
            os.replace(temporary, path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        expected = item.get('lfs', {}).get('sha256')
        if expected and digest.hexdigest() != expected:
            raise ValueError(f'Checksum mismatch: {name}')
        print(f'Verified {repo}/{name}', flush=True)
        return dict(name=name, size=path.stat().st_size, sha256=digest.hexdigest())

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(fetch, files))
    marker.write_text(json.dumps(dict(repo=repo, revision=revision, files=verified), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--asset', choices=['tiif-judge', 'genai-judge', 'genai-data'], required=True)
    args = p.parse_args()
    if args.asset == 'tiif-judge':
        download_repo('Qwen/Qwen2.5-VL-7B-Instruct', args.root / 'Qwen2.5-VL-7B-Instruct')
    elif args.asset == 'genai-judge':
        download_repo('Qwen/Qwen3.5-27B', args.root / 'Qwen3.5-27B')
    else:
        download_repo('BaiqiL/GenAI-Bench-1600', args.root / 'GenAI-Image-1600', dataset=True,
                      names=['genai_image.json', 'genai_skills.json'])
