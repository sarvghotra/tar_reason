import tempfile
import unittest
import json
import contextlib
import io
from unittest.mock import patch
from pathlib import Path

from llava.train.rl import geneval2_eval_io as ge
from llava.train.rl.intermediate_eval import selection, source_data, checked_source, validate_outputs, check_selection
from scripts.rl_ft.report_intermediate_eval import compare, compare_ordinals, capped_final_records, genai_split_lines


class IntermediateTests(unittest.TestCase):
    def test_genai_difficulty_uses_tags_and_matched_counts(self):
        rows = [{'skills': ['basic', 'count']}, {'skills': ['advanced']}, {'skills': []}]
        sets = {'first': [{'score': .2}, {'score': .8}, {'score': 1.}],
                'final': [{'score': .4}, {'score': .6}, {'score': 0.}]}
        result = compare_ordinals('genai', rows, sets, [1, 4, 4], {'first': 1, 'up_to_four': 4})
        lines = '\n'.join(genai_split_lines(result, ['first', 'final'], ['First', 'Final']))
        self.assertIn('| Full benchmark | Simple (basic) | 1 | 20.0000 | 40.0000 |', lines)
        self.assertIn('| Full benchmark | Complex (advanced) | 1 | 80.0000 | 60.0000 |', lines)
        self.assertIn('| Reached image 4 | Simple (basic) | 0 | N/A | N/A |', lines)
        self.assertIn('| Reached image 4 | Complex (advanced) | 1 | 80.0000 | 60.0000 |', lines)
        # The unlabelled row must not enter either group or the official tagged union.
        self.assertEqual(result['full_benchmark']['first']['all'], .5)
        result['full_benchmark']['final']['per_skill']['basic']['n'] = 2
        with self.assertRaises(ValueError):
            genai_split_lines(result, ['first', 'final'], ['First', 'Final'])

    def test_selection_eos_and_cap(self):
        for count in (1, 2, 3, 4):
            for ordinal in (1, 2, 3, 4):
                result = selection({'semantic_codes': [[i]*729 for i in range(count)]}, ordinal)
                self.assertEqual(result['selected_ordinal'], min(count, ordinal))
                self.assertEqual(result['reused_final_image'], count <= ordinal)
                self.assertEqual(result['actual_second_available'], count >= 2)

    def test_invalid_codes(self):
        for codes in ([], [[1]*728], [[-1]*729], [[65536]*729], [[True]*729]):
            with self.assertRaises(ValueError):
                selection({'semantic_codes': codes}, 2)

    def fixture(self, root):
        source, out = root/'source', root/'out'
        source.mkdir()
        out.mkdir()
        row = dict(id='0', prompt='red chair', skills=['basic'])
        benchmark = root/'benchmark.json'
        ge.atomic_json(benchmark, dict(benchmark='genai', rows=[row]))
        manifest = dict(protocol='instruction_benchmark_final_pixel_v1', benchmark=str(benchmark),
                        benchmark_sha256=ge.sha256_file(benchmark), n_prompts=1, max_refinements=3)
        ge.atomic_json(source/'manifest.json', manifest)
        (source/'prompt-0000.png').write_bytes(b'exact final PNG pixels')
        record = dict(index=0, row=row, protocol_id=ge.protocol_id(manifest),
                      semantic_codes=[[5]*729], image='prompt-0000.png',
                      image_sha256=ge.sha256_file(source/'prompt-0000.png'))
        ge.atomic_json(source/'prompt-0000.json', record)
        kind, rows, original, hashes = source_data(source)
        derived = dict(manifest, source_dir=str(source), source_protocol_id=ge.protocol_id(original),
                       source_record_sha256=hashes, image_ordinal=2)
        ge.atomic_json(out/'manifest.json', derived)
        (out/'prompt-0000.png').write_bytes((source/'prompt-0000.png').read_bytes())
        output = dict(record, protocol_id=ge.protocol_id(derived), selection=selection(record, 2),
                      source_record_sha256=hashes[0])
        ge.atomic_json(out/'prompt-0000.json', output)
        return source, out, row, manifest, record, hashes[0], output

    def test_exact_final_and_no_fake_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out, row, manifest, record, digest, output = self.fixture(Path(tmp))
            validate_outputs(out)
            output['selection']['actual_second_available'] = True
            with self.assertRaises(ValueError):
                check_selection(output, record, digest, 2)
            output['selection'] = selection(record, 2)
            output['image_sha256'] = 'rerendered pixels'
            with self.assertRaises(ValueError):
                check_selection(output, record, digest, 2)

    def test_changed_source_codes_or_pixels(self):
        for mutation in ('codes', 'pixels'):
            with tempfile.TemporaryDirectory() as tmp:
                source, out, row, manifest, record, digest, output = self.fixture(Path(tmp))
                if mutation == 'codes':
                    record['semantic_codes'][0][0] = 9
                    ge.atomic_json(source/'prompt-0000.json', record)
                else:
                    (source/'prompt-0000.png').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    checked_source(source, 0, row, manifest, digest, 'genai')
                with self.assertRaises(ValueError):
                    validate_outputs(out)

    def test_partial_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out, *_ = self.fixture(Path(tmp))
            (out/'prompt-0000.json').unlink()
            with self.assertRaises(ValueError):
                validate_outputs(out)

    def test_fourth_alias_requires_cap_and_complete_images(self):
        records = [{'semantic_codes': [[n]*729 for n in range(count)]} for count in (1, 2, 3, 4)]
        self.assertIs(capped_final_records(records, {'max_refinements': 3}, 4), records)
        for cap, ordinal in ((4, 4), (3, 3)):
            with self.assertRaises(ValueError):
                capped_final_records(records, {'max_refinements': cap}, ordinal)
        with self.assertRaises(ValueError):
            capped_final_records([{'semantic_codes': [[0]*729]*5}], {'max_refinements': 3}, 4)
        with self.assertRaises(ValueError):
            capped_final_records([{'semantic_codes': [[0]*728]}], {'max_refinements': 3}, 4)

    def test_ordinal_subsets_use_identical_prompt_indices(self):
        rows = [{'skills': ['basic']}]*4
        scores = [{'score': value} for value in (.1, .2, .3, .4)]
        ordinals = {'first': 1, 'up_to_two': 2, 'up_to_three': 3, 'up_to_four': 4}
        report = compare_ordinals('genai', rows, {k: scores for k in list(ordinals)+['final']}, [1, 2, 3, 4], ordinals)
        for ordinal in range(1, 5):
            subset = report['actual_image_subsets'][str(ordinal)]
            self.assertEqual(subset['prompt_indices'], list(range(ordinal-1, 4)))
            self.assertEqual(subset['n_prompts'], 5-ordinal)
            self.assertEqual(subset['metrics']['up_to_four'], subset['metrics']['final'])
        self.assertEqual(report['actual_image_subsets']['4']['metrics']['first']['all'], .4)

    def test_paired_subset_not_confounded_by_early_eos(self):
        rows = [dict(skills=['basic'])]*2
        sets = {key: [dict(score=a), dict(score=b)] for key, a, b in
                [('first', .2, .9), ('up_to_two', .2, .6), ('final', .2, .4)]}
        report = compare('genai', rows, sets, [False, True])
        self.assertEqual(report['actual_second_coverage'], .5)
        self.assertEqual(report['full_benchmark']['up_to_two']['all'], .4)
        subset = report['actual_second_matched_subset']
        self.assertEqual([subset[k]['all'] for k in ('first', 'up_to_two', 'final')], [.9, .6, .4])
        report = compare('genai', rows, sets, [False, False])
        self.assertTrue(all(v is None for v in report['actual_second_matched_subset'].values()))
        sets['first'][0] = None
        with self.assertRaises(ValueError):
            compare('genai', rows, sets, [False, True])

    def test_report_end_to_end_with_single_image_episodes(self):
        from scripts.rl_ft.report_intermediate_eval import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, second, row, _, record, digest, output = self.fixture(root)
            first = root/'first'
            first.mkdir()
            manifest = json.loads((second/'manifest.json').read_text())
            manifest['image_ordinal'] = 1
            ge.atomic_json(first/'manifest.json', manifest)
            (first/'prompt-0000.png').write_bytes((second/'prompt-0000.png').read_bytes())
            ge.atomic_json(first/'prompt-0000.json', dict(output, protocol_id=ge.protocol_id(manifest),
                                                        selection=selection(record, 1)))
            for directory in (source, first, second):
                generation = json.loads((directory/'manifest.json').read_text())
                judge = dict(generation_protocol_id=ge.protocol_id(generation))
                ge.atomic_json(directory/'scores/manifest.json', judge)
                ge.atomic_json(directory/'scores/prompt-0000.json', dict(id=row['id'],
                               protocol_id=ge.protocol_id(judge), image_sha256=record['image_sha256'], score=.7))
            config = root/'comparison.json'
            ge.atomic_json(config, dict(benchmarks={'genai': dict(final=str(source), first=str(first), up_to_two=str(second))}))
            with patch('sys.argv', ['report', '--config', str(config), '--report_to', 'none']), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((root/'results.json').read_text())['genai']
            self.assertEqual(result['actual_second_count'], 0)
            self.assertEqual(result['full_benchmark']['first']['all'], .7)
            self.assertIn('70.0000', (root/'RESULTS.md').read_text())
            self.assertIn('Simple (basic)', (root/'RESULTS.md').read_text())
            self.assertIn('Complex (advanced) | 0 | N/A', (root/'RESULTS.md').read_text())
            # Third image retains the last complete image; fourth aliases the
            # final record exactly, without another renderer or judge call.
            third = root/'third'
            third.mkdir()
            manifest['image_ordinal'] = 3
            ge.atomic_json(third/'manifest.json', manifest)
            (third/'prompt-0000.png').write_bytes((second/'prompt-0000.png').read_bytes())
            ge.atomic_json(third/'prompt-0000.json', dict(output, protocol_id=ge.protocol_id(manifest),
                                                        selection=selection(record, 3)))
            judge = dict(generation_protocol_id=ge.protocol_id(manifest))
            ge.atomic_json(third/'scores/manifest.json', judge)
            ge.atomic_json(third/'scores/prompt-0000.json', dict(id=row['id'], protocol_id=ge.protocol_id(judge),
                                                              image_sha256=record['image_sha256'], score=.7))
            ge.atomic_json(config, dict(model='Checkpoint 500', image_ordinals={
                'first': 1, 'up_to_two': 2, 'up_to_three': 3, 'up_to_four': 4}, benchmarks={'genai': dict(
                final=str(source), first=str(first), up_to_two=str(second), up_to_three=str(third), up_to_four=str(source))}))
            with patch('sys.argv', ['report', '--config', str(config), '--report_to', 'none']), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((root/'results.json').read_text())['genai']
            self.assertEqual(result['full_benchmark']['up_to_four'], result['full_benchmark']['final'])
            self.assertEqual(result['actual_image_subsets']['4']['n_prompts'], 0)
            self.assertIsNone(result['actual_image_subsets']['4']['metrics']['up_to_four'])
            self.assertIn('# Checkpoint 500', (root/'RESULTS.md').read_text())


if __name__ == '__main__':
    unittest.main()
