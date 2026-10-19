"""Checks for the main-branch evaluation protocol port (--protocol main_adhoc)."""
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llava.train.rl.grpo import KIND_IMG, KIND_NONE, KIND_TXT
from llava.train.rl.main_protocol import MainAdhocRollout

SPECIAL = ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<im_start>', '<im_end>', '<S0>']
IMAGES = [f'<I{i}>' for i in range(8)]


class CharTokenizer:
    """Specials and <I*> are single tokens; everything else is one token per character."""

    def __init__(self):
        chars = [chr(c) for c in range(32, 127)] + ['\n']
        self.vocab = {t: i for i, t in enumerate(SPECIAL + IMAGES + chars)}
        self.inverse = {i: t for t, i in self.vocab.items()}
        self.eos_token_id = self.vocab['<|im_end|>']
        self.pad_token_id = self.vocab['<|endoftext|>']
        self.pattern = re.compile('|'.join(re.escape(t) for t in SPECIAL + IMAGES) + r'|.', re.S)

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[self.vocab[t] for t in self.pattern.findall(text)])

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ('<|im_start|>assistant\n' if add_generation_prompt else '')

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(t for t in (self.inverse[i] for i in ids)
                       if not (skip_special_tokens and (t in SPECIAL or t in IMAGES)))


class ScriptedGenerate(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs, self.calls = list(outputs), []

    def generate(self, input_ids, **kwargs):
        rows = self.outputs.pop(0)
        self.calls.append(dict(kwargs, input_ids=input_ids.clone()))
        assert len(rows) == len(input_ids)
        out = torch.full((len(rows), max(map(len, rows))), kwargs['pad_token_id'], dtype=torch.long)
        for i, row in enumerate(rows):
            out[i, :len(row)] = torch.tensor(row)
        return torch.cat((input_ids, out), dim=1)


class MainProtocolTests(unittest.TestCase):
    def test_forced_schedule_looks_good_skip_reencoding_and_codes(self):
        tok = CharTokenizer()
        I = lambda k: tok.vocab[f'<I{k}>']
        enc = lambda s: tok(s).input_ids
        eos, im_start, pad = tok.eos_token_id, tok.vocab['<im_start>'], tok.pad_token_id
        rollout = MainAdhocRollout(tok, 'cpu', reflect_tokens=8, gen_seq_len=3)
        model = ScriptedGenerate([
            # Phase 1: a full image; one cut by EOS (padded with code 0); one with a text token kept in context.
            [[I(1), I(2), I(3)], [I(4), eos, I(5)], [I(6), enc('x')[0], I(7)]],
            # Phase 2: greedy reflections (stop ids end them).
            [enc('Looks good.') + [eos], enc(' fix') + [im_start], enc('add dog') + [eos]],
            # Phase 3 for rows 1 and 2 only.
            [[I(0), I(1), I(2)], [I(3), I(3), I(3)]],
        ])
        batch = rollout.run(model, [{'prompt': 'a'}, {'prompt': 'b'}, {'prompt': 'c'}])
        a, b, c = batch.trajectories
        self.assertEqual([t.stop_reason for t in batch.trajectories],
                         ['looks_good', 'forced_refinement', 'forced_refinement'])
        self.assertEqual(a.images, [[1, 2, 3]])
        self.assertEqual(b.images, [[4, 0, 0], [0, 1, 2]])
        self.assertEqual(c.images, [[6, 7, 0], [3, 3, 3]])
        self.assertEqual([t.reflections for t in batch.trajectories], [['Looks good.'], ['fix'], ['add dog']])
        image_call, reflect_call, refine_call = model.calls
        for call in (image_call, refine_call):
            self.assertEqual((call['do_sample'], call['temperature'], call['top_k'], call['top_p'],
                              call['repetition_penalty'], call['max_new_tokens']), (True, 1.0, 1200, 0.95, 1.0, 3))
            self.assertNotIn('eos_token_id', call)      # main relies on the checkpoint generation_config
            self.assertNotIn('logits_processor', call)  # no image-vocabulary mask
        self.assertFalse(reflect_call['do_sample'])
        self.assertEqual(reflect_call['max_new_tokens'], 8)
        self.assertEqual(set(reflect_call['eos_token_id']), {eos, pad, im_start})
        # Phase 3 context: phase-2 row + " " + decoded reflection + "\n<im_start><S0>"; draft ids kept raw.
        prefix = enc(rollout.make_prefix('c'))
        phase2 = prefix + [I(6), enc('x')[0], I(7)] + enc('<im_end>\nSelf-reflect:')
        expected = phase2 + enc(' add dog') + enc('\n<im_start><S0>')
        row = refine_call['input_ids'][1]
        self.assertEqual(row[row != pad].tolist()[-len(expected):], expected)
        self.assertEqual(c.seq, expected + [I(3)] * 3)
        self.assertEqual(len(c.kinds), len(c.seq))
        self.assertEqual(c.kinds[-3:], [KIND_IMG] * 3)
        self.assertEqual(c.kinds[len(prefix) + 1], KIND_TXT)  # non-image token inside the draft run
        self.assertEqual(a.seq[-len(enc('Looks good.')):], enc('Looks good.'))
        self.assertTrue(all(k == KIND_NONE for k in a.kinds[:len(prefix)]))
        with self.assertRaises(ValueError):
            rollout.run(model, [{'prompt': 'a'}], num_rollouts=2)

    def test_eval_cli_requires_one_refinement_and_no_context_flag(self):
        from llava.train.rl import eval_geneval2, generate_instruction_benchmarks
        required = ['x', '--checkpoint', 'c', '--benchmark', 'b', '--output_dir', 'o', '--ar_path', 'a',
                    '--encoder_path', 'e', '--decoder_path', 'd', '--code_manifest', 'm']
        for module, extra in ((eval_geneval2, ['--reward_model', 'r', '--reward_python', 'p']),
                              (generate_instruction_benchmarks, [])):
            with patch('sys.argv', required + extra + ['--protocol', 'main_adhoc', '--max_refinements', '1']):
                self.assertEqual(module.parse_args().protocol, 'main_adhoc')
            with patch('sys.argv', required + extra):
                self.assertEqual(module.parse_args().protocol, 'native')
            for bad in (['--protocol', 'main_adhoc'],
                        ['--protocol', 'main_adhoc', '--max_refinements', '1', '--reset_context_per_turn']):
                with self.subTest(module=module.__name__, bad=bad), patch('sys.argv', required + extra + bad), \
                        patch('sys.stderr'), self.assertRaises(SystemExit):
                    module.parse_args()


if __name__ == '__main__':
    unittest.main()
