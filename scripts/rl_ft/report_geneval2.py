"""Write a comparison of full GenEval2 evaluations; never label partial scores final."""
import argparse
import json
from pathlib import Path
from llava.train.rl.geneval2_eval_io import atomic_json, load_benchmark, sha256_file, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', required=True)
    args = parser.parse_args()
    path = Path(args.comparison)
    comparison = json.loads(path.read_text())
    rows = load_benchmark(comparison['benchmark'])
    results = []
    for model in comparison['models']:
        directory = Path(model['output_dir'])
        manifest_path = directory/'manifest.json'
        result = {**model, 'status': 'not_started'}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                if (manifest['benchmark_sha256'] != sha256_file(comparison['benchmark']) or
                        manifest['checkpoint'] != str(Path(model['checkpoint']).resolve())):
                    raise ValueError('Report checkpoint/benchmark differs from evaluation manifest')
                metrics = summarize(directory, rows, manifest, require_complete=False)
                result.update(metrics)
                result['status'] = 'complete' if metrics['complete'] else 'partial'
            except (ValueError, OSError, KeyError) as exc:
                result.update(status='error', error=str(exc))
        results.append(result)
    atomic_json(path.parent/'results.json', results)
    lines = ['# Full GenEval2 comparison', '',
             '800 official prompts; one final image per prompt, identical rollout/renderer settings.', '',
             '| Model | Checkpoint | Status | Prompts | Soft-TIFA AM (%) | Soft-TIFA GM (%) |',
             '|---|---:|---|---:|---:|---:|']
    for r in results:
        complete = r['status'] == 'complete'
        am = f"{100*r['am']:.4f}" if complete else '—'
        gm = f"{100*r['gm']:.4f}" if complete else '—'
        lines.append(f"| {r['label']} | {r['step']} | {r['status']} | {r.get('n_prompts',0)}/800 | {am} | {gm} |")
    lines += ['', 'Partial per-prompt results are retained for resumption; partial scores are not benchmark results.']
    report = path.parent/'RESULTS.md'
    report.write_text('\n'.join(lines)+'\n')
    print(report.read_text())


if __name__ == '__main__':
    main()
