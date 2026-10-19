"""Compare saved image ordinals and final on identical episodes, with coverage."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from llava.train.rl import geneval2_eval_io as ge
from llava.train.rl import instruction_eval_io as ins
from llava.train.rl.intermediate_eval import MAX_ORDINAL, source_data, validate_outputs, checked_source, selection


def aggregate(kind, rows, scores):
    if not rows:
        return None
    if kind != 'geneval2':
        return ins.aggregate(kind, rows, scores)
    skills = defaultdict(list)
    for row, score in zip(rows, scores):
        for skill, value in zip(row['skills'], score['question_scores']):
            skills[skill].append(value)
    return dict(n_scored=len(scores), am=sum(s['am'] for s in scores)/len(scores),
                gm=sum(s['gm'] for s in scores)/len(scores),
                per_skill_am={k: sum(v)/len(v) for k, v in skills.items()})


def compare(kind, rows, score_sets, available):
    if len(available) != len(rows) or any(len(v) != len(rows) or any(s is None for s in v)
                                        for v in score_sets.values()):
        raise ValueError('Incomplete paired evaluation')
    subset = [i for i, exists in enumerate(available) if exists]
    return dict(n_prompts=len(rows), actual_second_count=len(subset),
                actual_second_coverage=len(subset)/len(rows),
                full_benchmark={k: aggregate(kind, rows, v) for k, v in score_sets.items()},
                actual_second_matched_subset={k: aggregate(kind, [rows[i] for i in subset],
                                                          [v[i] for i in subset])
                                              for k, v in score_sets.items()},
                subset_note='Conditional subset, not a full-benchmark result. All three stages use the same subset.')


def load_scores(directory, kind, rows):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if kind == 'geneval2':
        ge.summarize(directory, rows, manifest)
        return [ge.read_record(directory, i, row, ge.protocol_id(manifest)) for i, row in enumerate(rows)]
    ins.summarize(directory)
    return [json.loads((directory/'scores'/f'prompt-{i:04d}.json').read_text()) for i in range(len(rows))]


def headline(kind, metrics):
    if metrics is None:
        return 'N/A'
    if kind == 'geneval2':
        values = [('AM', metrics['am']), ('GM', metrics['gm'])]
    elif kind == 'genai':
        values = [('VQAScore', metrics['all'])]
    else:
        values = [(key, metrics['registers'][key]['overall']) for key in ('short', 'long')]
    return ', '.join(f'{key}={100*value:.4f}' if value is not None else f'{key}=N/A' for key, value in values)


def capped_final_records(records, manifest, ordinal):
    """At the declared image cap, up-to-N and final are exactly the same pixels."""
    if ordinal != manifest.get('max_refinements', -1) + 1:
        raise ValueError('Only the declared maximum image ordinal can alias final')
    for record in records:
        if not selection(record, ordinal)['reused_final_image']:
            raise ValueError('Episode exceeds the declared cap')
    return records


def compare_ordinals(kind, rows, score_sets, counts, ordinals):
    if len(counts) != len(rows) or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('Invalid episode image counts')
    result = compare(kind, rows, score_sets, [n >= 2 for n in counts])
    result['actual_image_subsets'] = {}
    for ordinal in ordinals.values():
        indices = [i for i, n in enumerate(counts) if n >= ordinal]
        result['actual_image_subsets'][str(ordinal)] = dict(
            n_prompts=len(indices), coverage=len(indices)/len(rows), prompt_indices=indices,
            metrics={key: aggregate(kind, [rows[i] for i in indices], [values[i] for i in indices])
                     for key, values in score_sets.items()})
    return result


def genai_split_lines(result, stage_names, labels):
    """Report official basic/advanced tags, without inferring difficulty."""
    lines = ['', '## GenAI-Bench simple / complex prompts', '',
             'Simple = benchmark basic label; complex = benchmark advanced label.',
             'Unlabelled prompts are not assigned a difficulty. Scores are multiplied by 100.',
             'Matched subsets contain only episodes that reached the specified image.', '',
             '| Population | Category | Prompts | ' + ' | '.join(labels) + ' |',
             '|---|---|---|' + '---|' * len(labels)]
    populations = [('Full benchmark', result['full_benchmark'])] + [
        (f'Reached image {ordinal}', subset['metrics'])
        for ordinal, subset in result['actual_image_subsets'].items()]
    for population, metrics in populations:
        for tag, label in (('basic', 'Simple (basic)'), ('advanced', 'Complex (advanced)')):
            values = [(metrics[key] or {}).get('per_skill', {}).get(tag) for key in stage_names]
            counts = [value['n'] if value else 0 for value in values]
            if len(set(counts)) != 1:
                raise ValueError('GenAI category counts differ across image stages')
            scores = [f"{100*value['mean']:.4f}" if value else 'N/A' for value in values]
            lines.append('| ' + ' | '.join([population, label, str(counts[0])] + scores) + ' |')
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--report_to', choices=('wandb', 'none'), default='wandb')
    args = p.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    ordinals = config.get('image_ordinals', {'first': 1, 'up_to_two': 2})
    if (not ordinals or any(type(n) is not int or not 1 <= n <= MAX_ORDINAL for n in ordinals.values())
            or len(set(ordinals.values())) != len(ordinals) or 'final' in ordinals):
        raise ValueError('Invalid image ordinal mapping')
    ordinals = dict(sorted(ordinals.items(), key=lambda item: item[1]))
    stage_names = list(ordinals) + ['final']
    results, examples = {}, []
    for kind, paths in config['benchmarks'].items():
        loaded_kind, rows, source_manifest, hashes = source_data(paths['final'])
        if loaded_kind != kind:
            raise ValueError('Benchmark kind mismatch')
        if ('checkpoint_step' in config and source_manifest.get('checkpoint_step') != config['checkpoint_step']):
            raise ValueError('Wrong checkpoint step')
        source_records = [checked_source(paths['final'], i, row, source_manifest, hashes[i], kind)
                          for i, row in enumerate(rows)]
        score_sets = {'final': load_scores(paths['final'], kind, rows)}
        for name, ordinal in ordinals.items():
            if Path(paths[name]).resolve() == Path(paths['final']).resolve():
                capped_final_records(source_records, source_manifest, ordinal)
                score_sets[name] = score_sets['final']
                continue
            k, r, manifest, records = validate_outputs(paths[name])
            if (k != kind or r != rows or manifest['source_protocol_id'] != ge.protocol_id(source_manifest)
                    or manifest['source_record_sha256'] != hashes or manifest['image_ordinal'] != ordinal):
                raise ValueError('Unpaired intermediate evaluations')
            score_sets[name] = load_scores(paths[name], kind, rows)
        counts = [len(record['semantic_codes']) for record in source_records]
        results[kind] = compare_ordinals(kind, rows, score_sets, counts, ordinals)
        results[kind]['paths'] = paths
        for i in range(min(8, len(rows))):
            examples.append((kind, i, rows[i]['prompt'], counts[i],
                             [str(Path(paths[key])/f'prompt-{i:04d}.png') for key in stage_names]))
    ge.atomic_json(config_path.parent/'results.json', results)
    labels = [f'Image {ordinal} (up to)' for ordinal in ordinals.values()] + ['Final']
    lines = ['# ' + config.get('model', 'Saved-episode image comparison'), '',
             'Scores below are multiplied by 100. Image 1 is the initial draft.',
             'Up to N retains the last complete image when the episode stops before image N.',
             config.get('final_note', 'Final is the original EOS/cap-terminated episode result. No extra rounds are forced.'),
             'Intermediate rendering uses the source settings and per-prompt seed.',
             'At the image cap, the final PNG and score are reused without any re-rendering or judging.', '',
             '| Benchmark | ' + ' | '.join(labels) + ' |',
             '|---|' + '---|' * len(labels)]
    for kind, result in results.items():
        lines.append('| ' + ' | '.join([kind] + [headline(kind, result['full_benchmark'][key])
                                                 for key in stage_names]) + ' |')
    lines += ['', '## Actual ordinal coverage', '',
              '| Benchmark | ' + ' | '.join(f'Image {n}' for n in ordinals.values()) + ' |',
              '|---|' + '---|' * len(ordinals)]
    for kind, result in results.items():
        lines.append('| ' + ' | '.join([kind] + [str(result['actual_image_subsets'][str(n)]['n_prompts'])
                       + '/' + str(result['n_prompts']) for n in ordinals.values()]) + ' |')
    lines += ['', '## Actual-image matched subsets', '',
              'Conditional diagnostics, not full-benchmark results. Each row compares identical eligible prompts.',
              'Unavailable TIIF macro scores mean that some of the nine benchmark groups are absent.', '',
              '| Benchmark | Eligible episodes | ' + ' | '.join(labels) + ' |',
              '|---|---|' + '---|' * len(labels)]
    for kind, result in results.items():
        for ordinal in ordinals.values():
            subset = result['actual_image_subsets'][str(ordinal)]
            lines.append('| ' + ' | '.join([kind, f"Reached image {ordinal}: {subset['n_prompts']}"] +
                         [headline(kind, subset['metrics'][key]) for key in stage_names]) + ' |')
    if 'genai' in results:
        lines += genai_split_lines(results['genai'], stage_names, labels)
    (config_path.parent/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if args.report_to == 'wandb':
        import wandb
        run = wandb.init(project='tar_reasoning', name=config_path.parent.name,
                         id=config_path.parent.name, resume='allow', job_type='intermediate_comparison', config=config)
        table = wandb.Table(columns=['benchmark', 'index', 'prompt', 'actual_image_count'] + stage_names)
        for kind, i, prompt, count, paths in examples:
            table.add_data(kind, i, prompt, count, *(wandb.Image(path) for path in paths))
        split_metrics = {}
        if 'genai' in results:
            for stage, metrics in results['genai']['full_benchmark'].items():
                for tag in ('basic', 'advanced'):
                    value = metrics['per_skill'].get(tag)
                    if value:
                        split_metrics[f'genai/{stage}/{tag}'] = value['mean']
                        split_metrics[f'genai/{stage}/{tag}_n'] = value['n']
        run.log({'comparison': results, 'examples': table, **split_metrics})
        run.finish()
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
