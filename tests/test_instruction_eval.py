import json
from pathlib import Path
import tempfile
import unittest

from llava.train.rl.geneval2_eval_io import atomic_json, ensure_manifest, sha256_file
from llava.train.rl.instruction_eval_io import (
    TIIF_GROUPS, aggregate, check_score, prepare_genai, prepare_tiif, read_record, summarize)


class InstructionEvaluationTest(unittest.TestCase):
    def test_tiif_alignment_and_two_registers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'p').mkdir()
            (root/'q').mkdir()
            g = dict(type='color', short_description='short', long_description='long')
            q = dict(type='color', yn_question_list=['Red?'], yn_answer_list=['yes'])
            (root/'p/color_prompts.jsonl').write_text(json.dumps(g)+'\n')
            qp = root/'q/color_eval_prompts.jsonl'
            qp.write_text(json.dumps(q)+'\n')
            result = prepare_tiif(root/'p', root/'q')
            self.assertEqual([r['prompt'] for r in result['rows']], ['short', 'long'])
            self.assertEqual(len(result['sources']), 2)
            qp.write_text(json.dumps(q)+'\n'+json.dumps(q)+'\n')
            with self.assertRaisesRegex(ValueError, 'alignment'):
                prepare_tiif(root/'p', root/'q')

    def test_genai_noncontiguous_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, s = Path(tmp)/'prompts.json', Path(tmp)/'skills.json'
            atomic_json(p, {'00042': {'prompt': 'B'}, '00007': {'prompt': 'A'}})
            atomic_json(s, {'basic': [7], 'advanced': [42]})
            rows = prepare_genai(p, s)['rows']
            result = aggregate('genai', rows, [{'score': .25}, {'score': .75}])
            self.assertEqual(result['all'], .5)
            self.assertEqual(result['per_skill']['basic']['mean'], .25)
            atomic_json(s, {'basic': [0], 'advanced': [42]})
            with self.assertRaisesRegex(ValueError, 'membership'):
                prepare_genai(p, s)

    def test_image_integrity_and_row_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = ensure_manifest(root, {'seed': 1})
            image = root/'prompt-0000.png'
            image.write_bytes(b'exact scored pixels')
            row = {'id': 'x', 'prompt': 'a'}
            atomic_json(root/'prompt-0000.json', dict(index=0, protocol_id=identity,
                        row=row, image=image.name, image_sha256=sha256_file(image)))
            self.assertIsNotNone(read_record(root, 0, row, identity))
            with self.assertRaises(ValueError):
                read_record(root, 0, dict(row, id='y'), identity)
            image.write_bytes(b'changed pixels')
            with self.assertRaisesRegex(ValueError, 'changed image'):
                read_record(root, 0, row, identity)

    def test_no_partial_score_presented_as_complete(self):
        rows = [{'skills': ['basic']}, {'skills': ['advanced']}]
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            aggregate('genai', rows, [{'score': .5}, None])
        self.assertFalse(aggregate('genai', rows, [{'score': .5}, None], False)['complete'])

    def test_tiif_question_weighting_then_nine_group_macro_average(self):
        rows, scores = [], []
        for group, dims in TIIF_GROUPS.items():
            for dimension in dims:
                for register in ('short_description', 'long_description'):
                    rows.append(dict(dimension=dimension, register=register, answers=['yes', 'yes']))
                    scores.append({'model_pred': ['yes', 'yes'] if group == 'text' else ['no', 'no']})
        result = aggregate('tiif', rows, scores)
        self.assertAlmostEqual(result['registers']['short']['overall'], 1/9)
        self.assertAlmostEqual(result['registers']['long']['overall'], 1/9)
        # Extra questions within a dimension are question-weighted, not image-weighted.
        rows.append(dict(dimension='text', register='short_description', answers=['yes']*6))
        scores.append(dict(model_pred=['no']*6))
        result = aggregate('tiif', rows, scores)
        self.assertEqual(result['registers']['short']['per_dimension']['text'], .25)

    def test_score_validation(self):
        row = dict(id='x', questions=['Q1', 'Q2'], answers=['yes', 'no'])
        generated = dict(image_sha256='abc')
        score = dict(id='x', protocol_id='pid', image_sha256='abc', questions=row['questions'],
                     gt_answers=row['answers'], model_pred=['yes', 'no'])
        check_score(score, row, generated, 'pid')
        for field, bad in [('image_sha256', 'changed'), ('model_pred', ['yes']), ('protocol_id', 'old')]:
            with self.assertRaises(ValueError):
                check_score(dict(score, **{field: bad}), row, generated, 'pid')
        for bad in (float('nan'), float('inf'), -.1, 1.1):
            with self.assertRaises(ValueError):
                check_score(dict(score, score=bad), {'id': 'x'}, generated, 'pid')

    def test_end_to_end_durable_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(id='00042', prompt='A cat', skills=['basic', 'advanced'])
            benchmark = root/'benchmark.json'
            atomic_json(benchmark, dict(benchmark='genai', rows=[row]))
            manifest = dict(benchmark=str(benchmark), benchmark_sha256=sha256_file(benchmark))
            identity = ensure_manifest(root, manifest)
            image = root/'prompt-0000.png'
            image.write_bytes(b'pixels')
            atomic_json(root/'prompt-0000.json', dict(index=0, protocol_id=identity, row=row,
                        image=image.name, image_sha256=sha256_file(image)))
            (root/'scores').mkdir()
            score_id = ensure_manifest(root/'scores', dict(generation_protocol_id=identity))
            partial = summarize(root, False)
            self.assertFalse(partial['complete'])
            self.assertFalse((root/'metrics.json').exists())
            atomic_json(root/'scores/prompt-0000.json', dict(id=row['id'], protocol_id=score_id,
                        image_sha256=sha256_file(image), score=.75))
            self.assertEqual(summarize(root)['all'], .75)
            self.assertTrue((root/'metrics.json').exists())
            benchmark.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Benchmark changed'):
                summarize(root)


if __name__ == '__main__':
    unittest.main()
