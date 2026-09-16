import json
import tempfile
import unittest
from pathlib import Path

from llava.train.rl.geneval2_eval_io import (
    atomic_json, ensure_manifest, load_benchmark, protocol_id, read_record,
    score_means, sha256_file, summarize)


class FullGenEval2Tests(unittest.TestCase):
    def rows(self):
        return [dict(prompt='a cat', atom_count=3, vqa_list=[['q1', 'Yes'], ['q2', 'one']],
                     skills=['object', 'count']),
                dict(prompt='a dog', atom_count=4, vqa_list=[['q3', 'Yes']], skills=['object'])]

    def record(self, directory, index, row, manifest, scores):
        image = Path(directory)/f'prompt-{index:04d}.png'
        image.write_bytes(b'fixture-scored-pixels')
        am, gm = score_means(scores)
        r = dict(index=index, protocol_id=protocol_id(manifest), **row,
                 question_scores=scores, am=am, gm=gm, image=image.name,
                 image_sha256=sha256_file(image), stop_reason='eos')
        atomic_json(Path(directory)/f'prompt-{index:04d}.json', r)
        return r

    def test_aggregate_is_mean_per_prompt_and_official_skill_weighting(self):
        with tempfile.TemporaryDirectory() as d:
            rows, manifest = self.rows(), {'model': 'A'}
            ensure_manifest(d, manifest)
            # A duplicate-ID official score > 1 is deliberately valid.
            self.record(d, 0, rows[0], manifest, [1.2, 0.0])
            self.record(d, 1, rows[1], manifest, [0.2])
            out = summarize(d, rows, manifest)
            self.assertAlmostEqual(out['am'], .4)  # not the question-weighted mean
            self.assertAlmostEqual(out['gm'], .1)
            self.assertAlmostEqual(out['per_skill_am']['object']['score'], .7)
            self.assertEqual(out['per_atomicity']['3']['n_prompts'], 1)
            self.assertEqual(json.loads((Path(d)/'score_lists.json').read_text()), [[1.2, 0.0], [.2]])

    def test_partial_results_never_report_full_benchmark_success(self):
        with tempfile.TemporaryDirectory() as d:
            rows, manifest = self.rows(), {'model': 'A'}
            ensure_manifest(d, manifest)
            self.record(d, 0, rows[0], manifest, [.2, .4])
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                summarize(d, rows, manifest)
            result = summarize(d, rows, manifest, require_complete=False)
            self.assertFalse(result['complete'])
            self.assertEqual(result['n_prompts'], 1)
            self.assertFalse((Path(d)/'metrics.json').exists())
            self.assertFalse((Path(d)/'score_lists.json').exists())

    def test_resume_rejects_wrong_model_seed_code_or_benchmark(self):
        for field in ('checkpoint', 'seed', 'code_sha256', 'benchmark_sha256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                manifest = {field: 'a'}
                identity = ensure_manifest(d, manifest)
                self.assertEqual(identity, ensure_manifest(d, manifest))
                with self.assertRaisesRegex(ValueError, 'manifest differs'):
                    ensure_manifest(d, {field: 'b'})

    def test_score_image_and_prompt_integrity_are_verified_on_resume(self):
        for kind in ('image', 'scores', 'identity', 'prompt', 'path'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                row, manifest = self.rows()[0], {'model': 'A'}
                identity = ensure_manifest(d, manifest)
                r = self.record(d, 0, row, manifest, [.2, .4])
                self.assertIsNotNone(read_record(d, 0, row, identity))
                if kind == 'image':
                    (Path(d)/r['image']).write_bytes(b'different pixels')
                else:
                    key, value = {'scores': ('question_scores', [.3]), 'identity': ('protocol_id', 'B'),
                                  'prompt': ('prompt', 'different'), 'path': ('image', '../outside.png')}[kind]
                    r[key] = value
                    atomic_json(Path(d)/'prompt-0000.json', r)
                with self.assertRaises(ValueError):
                    read_record(d, 0, row, identity)

    def test_orphan_image_without_record_is_not_reused(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'prompt-0000.png').write_bytes(b'partial image')
            self.assertIsNone(read_record(d, 0, self.rows()[0], 'identity'))

    def test_atomic_failed_write_preserves_previous_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'result.json'
            atomic_json(path, {'valid': 1})
            with self.assertRaises(ValueError):
                atomic_json(path, {'invalid': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'valid': 1})

    def test_benchmark_size_uniqueness_and_question_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'benchmark.jsonl'
            rows = self.rows()
            def write(rs):
                path.write_text('\n'.join(json.dumps(r) for r in rs))
            write(rows)
            self.assertEqual(load_benchmark(path, 2), rows)
            with self.assertRaises(ValueError):
                load_benchmark(path, 800)
            write([rows[0], rows[0]])
            with self.assertRaises(ValueError):
                load_benchmark(path, 2)
            rows[0]['skills'] = []
            write(rows)
            with self.assertRaises(ValueError):
                load_benchmark(path, 2)


if __name__ == '__main__':
    unittest.main()
