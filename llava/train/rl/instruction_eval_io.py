"""Validated TIIF/GenAI inputs, final-image records, and benchmark aggregation."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from llava.train.rl.geneval2_eval_io import atomic_json, protocol_id, sha256_file


TIIF_GROUPS = {
    'attribute': ['shape+color', 'color+texture', 'texture+color', 'shape+texture'],
    'relation': ['2d_spatial_relation', '3d_spatial_relation', 'action+2d', 'action+3d'],
    'reasoning': ['numeracy', 'negation', 'differentiation', 'comparison'],
    'attribute_relation': ['action+color', 'action+texture', 'color+2d', 'color+3d',
                           'shape+2d', 'shape+3d', 'texture+2d', 'texture+3d'],
    'attribute_reasoning': [a+'+'+b for a in ['numeracy', 'comparison', 'differentiation', 'negation']
                            for b in ['color', 'texture']],
    'relation_reasoning': [a+'+'+b for a in ['numeracy', 'comparison', 'differentiation', 'negation']
                           for b in ['2d', '3d']],
    'text': ['text'], 'style': ['style'], 'real_world': ['real_world'],
}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prepare_tiif(prompts, questions):
    rows, sources, dimensions = [], {}, set()
    files = sorted(Path(prompts).glob('*_prompts.jsonl'))
    if not files:
        raise ValueError('No TIIF prompt files')
    expected_files = {p.name.replace('_prompts.jsonl', '_eval_prompts.jsonl') for p in files}
    if expected_files != {p.name for p in Path(questions).glob('*.jsonl')}:
        raise ValueError('TIIF generation/evaluation file sets differ')
    for path in files:
        qpath = Path(questions) / path.name.replace('_prompts.jsonl', '_eval_prompts.jsonl')
        generated, evaluated = jsonl(path), jsonl(qpath)
        if not generated or len(generated) != len(evaluated):
            raise ValueError(f'TIIF line alignment differs: {path}')
        dimension = generated[0]['type']
        if dimension in dimensions:
            raise ValueError('Duplicate TIIF dimension')
        dimensions.add(dimension)
        sources[str(path.resolve())] = sha256_file(path)
        sources[str(qpath.resolve())] = sha256_file(qpath)
        for index, (g, q) in enumerate(zip(generated, evaluated)):
            if g['type'] != dimension or q['type'] != dimension:
                raise ValueError('TIIF dimension alignment differs')
            qs, answers = q['yn_question_list'], q['yn_answer_list']
            if not qs or len(qs) != len(answers) or any(a.strip().lower() not in ('yes', 'no') for a in answers):
                raise ValueError('Invalid TIIF yes/no questions/answers')
            for register in ('short_description', 'long_description'):
                if not isinstance(g[register], str) or not g[register].strip():
                    raise ValueError('Empty TIIF prompt')
                rows.append(dict(id=f'{dimension}/{register}/{index}', prompt=g[register],
                                 dimension=dimension, register=register, line_idx=index,
                                 jsonl_file=qpath.name, questions=qs, answers=answers))
    return dict(benchmark='tiif', rows=rows, sources=sources)


def prepare_genai(prompts, skills):
    data, tags = json.loads(Path(prompts).read_text()), json.loads(Path(skills).read_text())
    if not data:
        raise ValueError('Empty GenAI data')
    ids = {int(k): k for k in data}
    if len(ids) != len(data):
        raise ValueError('Ambiguous GenAI IDs')
    for tag, members in tags.items():
        if not members or len(set(members)) != len(members) or any(i not in ids for i in members):
            raise ValueError(f'Invalid GenAI skill membership: {tag}')
    if not {'basic', 'advanced'} <= tags.keys():
        raise ValueError('Missing GenAI basic/advanced skills')
    rows = []
    for key in sorted(data):
        if not data[key]['prompt'].strip():
            raise ValueError('Empty GenAI prompt')
        rows.append(dict(id=key, prompt=data[key]['prompt'],
                         skills=[tag for tag, members in tags.items() if int(key) in members]))
    return dict(benchmark='genai', rows=rows,
                sources={str(Path(p).resolve()): sha256_file(p) for p in (prompts, skills)})


def load_benchmark(path):
    data = json.loads(Path(path).read_text())
    rows = data['rows']
    if data['benchmark'] not in ('tiif', 'genai') or not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Invalid benchmark/duplicate IDs')
    if any(not r['prompt'].strip() for r in rows):
        raise ValueError('Empty prompt')
    return rows


def read_record(directory, index, row, identity):
    path = Path(directory) / f'prompt-{index:04d}.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record['index'] != index or record['protocol_id'] != identity or record['row'] != row:
        raise ValueError(f'Incompatible generation record: {path}')
    expected = f'prompt-{index:04d}.png'
    if record['image'] != expected or sha256_file(Path(directory)/expected) != record['image_sha256']:
        raise ValueError(f'Missing or changed image: {path}')
    return record


def check_score(score, row, generated, identity):
    if (score['protocol_id'] != identity or score['id'] != row['id'] or
            score['image_sha256'] != generated['image_sha256']):
        raise ValueError('Stale or incompatible score')
    if 'questions' in row:
        if (score['questions'] != row['questions'] or score['gt_answers'] != row['answers'] or
                len(score['model_pred']) != len(row['questions']) or
                any(p not in ('yes', 'no') for p in score['model_pred'])):
            raise ValueError('Malformed TIIF predictions')
    elif not isinstance(score['score'], (int, float)) or not math.isfinite(score['score']) or not 0 <= score['score'] <= 1:
        raise ValueError('Invalid VQAScore')


def aggregate(benchmark, rows, scores, require_complete=True):
    if len(scores) != len(rows) or not rows:
        raise ValueError('Row/score count mismatch')
    pairs = [(r, s) for r, s in zip(rows, scores) if s is not None]
    complete = len(pairs) == len(rows)
    if require_complete and not complete:
        raise ValueError(f'Incomplete evaluation: {len(pairs)}/{len(rows)}')
    result = dict(benchmark=benchmark, complete=complete, n_scored=len(pairs), n_expected=len(rows))
    mean = lambda values: sum(values) / len(values) if values else None
    if benchmark == 'genai':
        by_skill = defaultdict(list)
        for row, score in pairs:
            for skill in row['skills']:
                by_skill[skill].append(score['score'])
        # Official 'all' is the union of tagged prompt IDs, never array offsets.
        result['per_skill'] = {k: dict(mean=mean(v), n=len(v)) for k, v in by_skill.items()}
        result['all'] = mean([s['score'] for r, s in pairs if r['skills']])
    else:
        buckets = defaultdict(lambda: [0, 0])
        for row, score in pairs:
            key = (row['register'].split('_')[0], row['dimension'])
            for gt, pred in zip(row['answers'], score['model_pred']):
                buckets[key][0] += gt.strip().lower() == pred
                buckets[key][1] += 1
        result['registers'] = {}
        for register in ('short', 'long'):
            values = {d: correct / total for (r, d), (correct, total) in buckets.items() if r == register}
            groups = {k: mean([values[d] for d in dims if d in values]) for k, dims in TIIF_GROUPS.items()}
            result['registers'][register] = dict(per_dimension=values, groups=groups,
                # All nine groups must exist for an overall score.
                overall=mean(list(groups.values())) if all(v is not None for v in groups.values()) else None,
                basic=mean([values[d] for k in ('attribute', 'relation', 'reasoning') for d in TIIF_GROUPS[k] if d in values]),
                advanced=mean([values[d] for k in ('attribute_relation', 'attribute_reasoning', 'relation_reasoning', 'text', 'style') for d in TIIF_GROUPS[k] if d in values]),
                real_world=values.get('real_world'))
    return result


def summarize(directory, require_complete=True):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    source = json.loads(Path(manifest['benchmark']).read_text())
    if sha256_file(manifest['benchmark']) != manifest['benchmark_sha256']:
        raise ValueError('Benchmark changed since generation')
    score_manifest = json.loads((directory/'scores/manifest.json').read_text())
    if score_manifest['generation_protocol_id'] != protocol_id(manifest):
        raise ValueError('Scoring used different generations')
    scores = []
    for index, row in enumerate(source['rows']):
        generated = read_record(directory, index, row, protocol_id(manifest))
        path = directory / 'scores' / f'prompt-{index:04d}.json'
        score = json.loads(path.read_text()) if path.exists() else None
        if score is not None:
            if generated is None:
                raise ValueError('Score exists without generation')
            check_score(score, row, generated, protocol_id(score_manifest))
        scores.append(score)
    result = aggregate(source['benchmark'], source['rows'], scores, require_complete)
    result['judge'] = score_manifest
    atomic_json(directory/('metrics.json' if result['complete'] else 'partial_metrics.json'), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['tiif', 'genai', 'summary'])
    p.add_argument('--prompts')
    p.add_argument('--questions')
    p.add_argument('--skills')
    p.add_argument('--output', required=True)
    p.add_argument('--allow_partial', action='store_true')
    args = p.parse_args()
    if args.action == 'summary':
        print(json.dumps(summarize(args.output, not args.allow_partial), indent=2))
    else:
        data = prepare_tiif(args.prompts, args.questions) if args.action == 'tiif' else prepare_genai(args.prompts, args.skills)
        atomic_json(args.output, data)
        print(f"Prepared {len(data['rows'])} {args.action} prompts")


if __name__ == '__main__':
    main()
