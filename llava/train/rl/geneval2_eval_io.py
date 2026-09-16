"""CPU-only validation, durable prompt results, and GenEval2 aggregation."""
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_benchmark(path, expected_count=800):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if len(rows) != expected_count or len({r['prompt'] for r in rows}) != len(rows):
        raise ValueError('Benchmark must contain the expected number of unique prompts')
    for r in rows:
        if not r['prompt'].strip() or not r['vqa_list'] or len(r['skills']) != len(r['vqa_list']):
            raise ValueError('Malformed benchmark prompt/questions/skills')
        if any(len(qa) != 2 for qa in r['vqa_list']):
            raise ValueError('Malformed VQA pair')
        if not isinstance(r['atom_count'], int):
            raise ValueError('Missing integer atom_count')
    return rows


def protocol_id(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def ensure_manifest(directory, manifest):
    path = Path(directory) / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Evaluation manifest differs: use a new output directory')
    else:
        if any(Path(directory).glob('prompt-*.json')):
            raise ValueError('Prompt records exist without a manifest')
        atomic_json(path, manifest)
    return protocol_id(manifest)


def score_means(scores):
    if not scores or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in scores):
        raise ValueError('Invalid question scores')
    return sum(scores) / len(scores), (0.0 if 0 in scores else
                                     math.exp(sum(math.log(v) for v in scores) / len(scores)))


def read_record(directory, index, row, identity):
    path = Path(directory) / f'prompt-{index:04d}.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record['index'] != index or record['protocol_id'] != identity or
            record['prompt'] != row['prompt'] or record['vqa_list'] != row['vqa_list']):
        raise ValueError(f'Incompatible prompt record: {path}')
    if len(record['question_scores']) != len(row['vqa_list']):
        raise ValueError(f'Wrong number of scores: {path}')
    am, gm = score_means(record['question_scores'])
    if not math.isclose(am, record['am'], abs_tol=1e-12) or not math.isclose(gm, record['gm'], abs_tol=1e-12):
        raise ValueError(f'Incorrect aggregate scores: {path}')
    # Only accept the expected local image, never an arbitrary path in a record.
    expected = f'prompt-{index:04d}.png'
    if record['image'] != expected or sha256_file(Path(directory) / expected) != record['image_sha256']:
        raise ValueError(f'Missing or changed scored image: {path}')
    return record


def summarize(directory, rows, manifest, require_complete=True):
    identity = protocol_id(manifest)
    records = [r for i, row in enumerate(rows)
               if (r := read_record(directory, i, row, identity)) is not None]
    if require_complete and len(records) != len(rows):
        raise ValueError(f'Incomplete benchmark: {len(records)}/{len(rows)} prompts')
    skills, atoms, stops = defaultdict(list), defaultdict(list), defaultdict(int)
    for r in records:
        row = rows[r['index']]
        for skill, score in zip(row['skills'], r['question_scores']):
            skills[skill].append(score)
        atoms[row['atom_count']].append(r)
        stops[r['stop_reason']] += 1
    n = len(records)
    result = dict(complete=n == len(rows), n_prompts=n, expected_prompts=len(rows),
                  am=sum(r['am'] for r in records)/n if n else None,
                  gm=sum(r['gm'] for r in records)/n if n else None,
                  per_skill_am={s: {'score': sum(v)/len(v), 'n_questions': len(v)} for s, v in skills.items()},
                  per_atomicity={str(a): {'am': sum(r['am'] for r in v)/len(v),
                                         'gm': sum(r['gm'] for r in v)/len(v), 'n_prompts': len(v)}
                                 for a, v in atoms.items()},
                  stop_counts=dict(stops), protocol_id=identity)
    atomic_json(Path(directory) / ('metrics.json' if result['complete'] else 'partial_metrics.json'), result)
    if result['complete']:
        # Exact benchmark order, compatible with official soft_tifa_analysis.py.
        atomic_json(Path(directory)/'score_lists.json', [r['question_scores'] for r in records])
        atomic_json(Path(directory)/'image_filepath_data.json',
                    {r['prompt']: str((Path(directory)/r['image']).resolve()) for r in records})
    return result
