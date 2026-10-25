"""looks_good suite launcher: stages, dependencies, placement and frozen commands."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts/eval'))
import looks_good_suite as lgs  # noqa: E402


def prepare(tmp, *extra):
    ck = Path(tmp)/'checkpoint-300'
    ck.mkdir(exist_ok=True)
    (ck/'state.json').write_text(json.dumps({'step': 300}))
    env = {k: v for k, v in os.environ.items() if not k.startswith(('TAR_', 'SLURM_'))}
    env.update(TAR_ROOT=tmp, SLURM_ACCOUNT='acct_gpu', TAR_CPU_ACCOUNT='acct_cpu', TAR_GPU_TINY='mig_1g')
    out = subprocess.run([sys.executable, str(REPO/'scripts/eval/looks_good_suite.py'), '--suite', 'unit',
                          '--config', f'sft={tmp}/sft:sft', '--config', f'ck300={ck}', '--max_refinements', '4',
                          *extra], env=env, capture_output=True, text=True)
    return out, Path(tmp)/'results/evaluations/unit'


def task_lines(script):
    import re
    return [l for l in script.read_text().splitlines() if re.match(r'\s+\d+\) ', l)]


class SuiteTest(unittest.TestCase):
    def test_stage_graph_matches_the_derived_suites(self):
        stages = lgs.stage_list(['geneval2', 'tiif', 'genai'])
        self.assertEqual(stages[:4], ['gen-geneval2', 'gen-geneval2-finalize', 'replay-geneval2', 'replay-geneval2-summary'])
        self.assertEqual(sorted(lgs.dependencies('report', ['geneval2', 'tiif', 'genai'])),
                         sorted(['gen-geneval2-finalize', 'replay-geneval2-summary', 'score-tiif-summary',
                                 'score-genai-summary']))
        self.assertEqual(lgs.dependencies('score-tiif', ['tiif']), ['gen-tiif-finalize', 'replay-tiif-summary'])
        self.assertEqual(lgs.dependencies('replay-genai', ['genai']), ['gen-genai-finalize'])
        placement = {s: lgs.profile_of(s) for s in stages}
        self.assertEqual(placement['gen-geneval2'], 'gen-large')
        self.assertEqual(placement['score-genai'], 'gen-large')
        self.assertEqual(placement['replay-tiif'], 'light')
        self.assertEqual(placement['replay-geneval2'], 'gen')
        self.assertEqual(placement['gen-tiif-finalize'], 'tiny')
        self.assertEqual(placement['report'], 'cpu')

    def test_prepare_freezes_code_and_writes_single_gpu_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, root = prepare(tmp)
            self.assertEqual(out.returncode, 0, out.stderr)
            launch = json.loads((root/'launch.json').read_text())
            counts = {k: v['tasks'] for k, v in launch['stages'].items() if k.startswith('ck300/')}
            self.assertEqual(counts['ck300/gen-geneval2'], 8)
            self.assertEqual(counts['ck300/gen-genai'], 12)
            self.assertEqual(counts['ck300/replay-geneval2'], 16)   # images 1-4 x 4 shards; image 5 is the final cap
            self.assertEqual(counts['ck300/score-tiif'], 10)       # (final + images 1-4) x 2 shards
            self.assertEqual(launch['stages']['ck300/replay-tiif']['array_elements'], 4)  # 4 tasks per GPU
            gen = (root/'ck300/jobs/gen-geneval2.sbatch').read_text()
            for flag in ('--ntasks=1', '--gpus-per-task=h100:1', '--array=0-7', '--account=acct_gpu'):
                self.assertIn(flag, gen)
            self.assertNotIn('--mem=0', gen)
            self.assertNotIn('--gpus-per-node', gen)
            self.assertIn('--gpus-per-task=mig_1g:1', (root/'ck300/jobs/gen-geneval2-finalize.sbatch').read_text())
            summary = (root/'ck300/jobs/replay-geneval2-summary.sbatch').read_text()
            self.assertIn('--account=acct_cpu', summary)
            self.assertNotIn('gpus', summary)
            argv = shlex.split(task_lines(root/'ck300/jobs/gen-geneval2.sbatch')[3].split(')', 1)[1].rstrip(' ;'))
            self.assertIn('RANK=3', argv)
            for flag, value in (('--seed', '421'), ('--gen_batch_size', '16'), ('--attn_implementation', 'sdpa'),
                                ('--decoding', 'sample'), ('--max_refinements', '4'), ('--verdict_rule', 'correction')):
                self.assertEqual(argv[argv.index(flag) + 1], value)
            self.assertIn('--reset_context_per_turn', argv)
            comparison = json.loads((root/'ck300/comparison.json').read_text())
            self.assertEqual(comparison['benchmarks']['tiif']['image5'], comparison['benchmarks']['tiif']['final'])
            self.assertEqual(comparison['checkpoint_step'], 300)
            self.assertTrue((root/'code/llava/train/rl/eval_geneval2.py').is_file())
            self.assertIn('files', json.loads((root/'code/manifest.json').read_text()))
            for script in (root/'ck300/jobs').glob('*.sbatch'):
                subprocess.run(['bash', '-n', str(script)], check=True)

    def test_one_gpu_model_for_every_gpu_stage_and_no_second_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, root = prepare(tmp, '--gpu_type', 'l40s')
            self.assertEqual(out.returncode, 0, out.stderr)
            for name in ('gen-geneval2', 'replay-tiif', 'score-genai'):
                self.assertIn('--gpus-per-task=l40s:1', (root/f'ck300/jobs/{name}.sbatch').read_text())
            again, _ = prepare(tmp)
            self.assertNotEqual(again.returncode, 0)


if __name__ == '__main__':
    unittest.main()
