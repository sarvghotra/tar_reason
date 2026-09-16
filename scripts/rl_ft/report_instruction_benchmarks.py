"""Write a completion-aware report after both instruction benchmark pipelines."""
import argparse
import json
from pathlib import Path
import subprocess

from llava.train.rl.geneval2_eval_io import atomic_json
from llava.train.rl.instruction_eval_io import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('launch_dir', type=Path)
    args = p.parse_args()
    launch = json.loads((args.launch_dir/'launch.json').read_text())
    results = {}
    for benchmark, spec in launch['evaluations'].items():
        directory = Path(spec['output_dir'])
        result = dict(benchmark=benchmark, complete=False, expected=spec['n_prompts'],
                      n_images=len(list(directory.glob('prompt-*.json'))),
                      n_scores=len(list((directory/'scores').glob('prompt-*.json'))),
                      jobs=spec['jobs'])
        try:
            result['accounting'] = subprocess.check_output(
                ['sacct', '-j', ','.join(map(str, spec['jobs'])), '-X', '--parsable2', '--noheader',
                 '--format=JobID,State,ExitCode,Elapsed'], text=True).strip()
        except (OSError, subprocess.CalledProcessError) as error:
            result['accounting_error'] = str(error)
        try:
            result['metrics'] = summarize(directory, require_complete=False)
            result['complete'] = result['metrics']['complete']
        except (OSError, ValueError, KeyError) as error:
            result['validation_error'] = str(error)
        results[benchmark] = result
    atomic_json(args.launch_dir/'results.json', results)
    lines = ['# TIIF / GenAI-Bench evaluation', '', f"Checkpoint: `{launch['checkpoint']}`", '',
             f"Seed: {launch['seed']}; one final image per prompt.", '',
             '| Benchmark | Generated | Scored | Status |', '|---|---:|---:|---|']
    for name, result in results.items():
        lines.append(f"| {name} | {result['n_images']}/{result['expected']} | "
                     f"{result['n_scores']}/{result['expected']} | "
                     f"{'Complete' if result['complete'] else 'Incomplete'} |")
    for name, result in results.items():
        lines += ['', f'## {name}', '']
        if result['complete']:
            metrics = result['metrics']
            if name == 'tiif':
                for register, values in metrics['registers'].items():
                    lines.append(f"- {register}: overall={values['overall']:.6f}, "
                                 f"basic={values['basic']:.6f}, advanced={values['advanced']:.6f}, "
                                 f"real-world={values['real_world']:.6f}")
            else:
                lines.append(f"- VQAScore all={metrics['all']:.6f}, "
                             f"basic={metrics['per_skill']['basic']['mean']:.6f}, "
                             f"advanced={metrics['per_skill']['advanced']['mean']:.6f}")
        else:
            lines.append('No complete benchmark score is available.')
        if result.get('validation_error'):
            lines.append(f"Validation: {result['validation_error']}")
        lines += ['', '```', result.get('accounting', result.get('accounting_error', '')), '```']
    (args.launch_dir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
