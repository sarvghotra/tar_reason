"""Checks for the evaluation-only looks_good protocol (EpisodeRollout.run_looks_good)."""
import unittest
from unittest.mock import patch

import torch

from llava.train.rl.grpo import KIND_IMG, KIND_NONE, KIND_TXT
from llava.train.rl.intermediate_eval import MAX_ORDINAL, selection
from llava.train.rl.rollout import RolloutConfig
from test_rl_episodes import ScriptedModel
from test_rl_janus import PROMPTS, janus_rollout

P = [1, 7]          # prompt prefix
R = [4, 6]          # reflection splice
# 16 "Correction: looks good." | 17 "Correction: add a dog" | 18 "Self-reflect: looks good but"
# 19 ".\n" | 7 " x" | 2 EOS | 3 <im_start> | 5 <S0> | 6 "\n"
CALLS = [('image', [[8] * 3, [9] * 3, [10] * 3, [11] * 3]),
         # e0 stops at once; e1 packed <im_start>; e2 EOS after newline; e3 verdict only in Correction
         ('text', [[16, 2], [17, 3], [17, 19, 2], [18, 17, 3]]),
         ('image', [[12] * 3, [13] * 3, [14] * 3]),
         # e1 "looks good" + <im_start> still stops; e2 capped (no stop token); e3 packed
         ('text', [[16, 3], [17, 7, 7, 7], [17, 3]]),
         ('image', [[15] * 3, [8, 9, 10]]),
         ('text', [[17, 3], [17, 2]]),
         ('image', [[9] * 3, [10] * 3]),
         # cap reached: e2 still records its verdict, e3 stops on max_refinements
         ('text', [[16, 2], [17, 3]])]


class Recording(ScriptedModel):
    """Also records each row's exact unpadded context and decoding flags."""

    def __init__(self, calls):
        super().__init__(calls)
        self.contexts, self.flags = [], []

    def generate(self, **kwargs):
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        self.contexts.append([row[m.bool()].tolist() for row, m in zip(ids, mask)])
        self.flags.append({k: kwargs.get(k) for k in ('do_sample', 'temperature', 'top_k', 'top_p')})
        return super().generate(**kwargs)


def run(reset=True, calls=CALLS, max_refinements=3, **cfg):
    rollout = janus_rollout(num_rollouts=4)
    rollout.cfg.max_refinements = max_refinements
    rollout.cfg.reset_context_per_turn = reset
    for key, value in cfg.items():
        setattr(rollout.cfg, key, value)
    model = Recording(calls)
    batch = rollout.run_looks_good(model, PROMPTS)
    assert model.calls == []
    return rollout, model, batch


class LooksGoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_stops_only_on_verdict_or_cap(self):
        for reset in (True, False):
            with self.subTest(reset=reset):
                _, _, batch = run(reset)
                e = batch.trajectories
                self.assertEqual([t.stop_reason for t in e], ['looks_good'] * 3 + ['max_refinements'])
                self.assertEqual([len(t.images) for t in e], [1, 2, 4, 4])
                self.assertEqual([t.verdicts for t in e],
                                 [[True], [False, True], [False, False, False, True], [False] * 4])
                self.assertEqual(e[2].images, [[2] * 3, [5] * 3, [7] * 3, [1] * 3])
                self.assertEqual(e[3].images[2], [0, 1, 2])
                for t in e:
                    self.assertEqual(len(t.seq), len(t.kinds))
                    self.assertEqual(len(t.reflections), len(t.verdicts))

    def test_reset_contexts_are_one_iteration_rounds(self):
        _, model, batch = run(True)
        e2 = batch.trajectories[2]
        rounds = [P + [10] * 3 + R + [17, 19] + [3, 5] + [13] * 3,          # EOS dropped, no extra newline
                  P + [13] * 3 + R + [17, 7, 7, 7] + [6, 3, 5] + [15] * 3,  # capped: newline + splice
                  P + [15] * 3 + R + [17, 3, 5] + [9] * 3,                  # packed sampled <im_start>
                  P + [9] * 3 + R + [16, 2]]
        self.assertEqual([s.seq for s in e2.segments], rounds)
        for segment in e2.segments[1:]:  # copied latest image is an observation
            self.assertEqual(segment.kinds[:len(P) + 3], [KIND_NONE] * (len(P) + 3))
        self.assertEqual(e2.segments[0].kinds[len(P):len(P) + 3], [KIND_IMG] * 3)
        # Exact model inputs: the final reflection of e2 sees only round 4's prefix.
        self.assertEqual(model.contexts[-1][0], P + [9] * 3 + R)
        self.assertEqual(model.contexts[2][1], rounds[0][:-3])  # e2's first refinement input
        # The logical episode keeps every sampled action once.
        self.assertEqual(sum(k != KIND_NONE for k in e2.kinds),
                         sum(k != KIND_NONE for s in e2.segments for k in s.kinds))

    def test_full_context_keeps_history(self):
        _, model, batch = run(False)
        e2 = batch.trajectories[2]
        expected = (P + [10] * 3 + R + [17, 19] + [3, 5] + [13] * 3 + R + [17, 7, 7, 7] + [6, 3, 5]
                    + [15] * 3 + R + [17, 3, 5] + [9] * 3 + R + [16, 2])
        self.assertEqual(e2.seq, expected)
        self.assertEqual(e2.segments, [])
        self.assertEqual(model.contexts[-1][0], expected[:-2])
        kinds = dict(zip(range(len(expected)), e2.kinds))
        self.assertEqual(kinds[len(P) + 3 + 2], KIND_TXT)       # reflection token
        self.assertEqual(kinds[len(P) + 3 + 2 + 2], KIND_NONE)  # forced <im_start>

    def test_zero_refinements_and_context_limit(self):
        _, _, batch = run(True, calls=[CALLS[0], CALLS[1]], max_refinements=0)
        self.assertEqual([t.stop_reason for t in batch.trajectories],
                         ['looks_good', 'max_refinements', 'max_refinements', 'max_refinements'])
        self.assertTrue(all(len(t.images) == 1 for t in batch.trajectories))
        # Full context runs out of room before the 3rd refinement; reset never does.
        short = CALLS[:5]
        _, _, batch = run(False, calls=short, max_seq_len=27)  # e2: 26 + 6 > 27, e3: 22 + 6 > 27
        self.assertEqual([t.stop_reason for t in batch.trajectories],
                         ['looks_good', 'looks_good', 'max_seq_len', 'max_seq_len'])

    def test_greedy_and_sampled_decoding_flags(self):
        _, model, _ = run(True, img_sample=False, reflect_sample=False)
        self.assertTrue(all(f == dict(do_sample=False, temperature=None, top_k=None, top_p=None)
                            for f in model.flags))
        _, model, _ = run(True)
        self.assertTrue(all(f == dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
                            for f in model.flags))
        for field in ('img_sample', 'reflect_sample'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                RolloutConfig(**{field: False}).validate_for_training()
        with self.assertRaises(ValueError):
            janus_rollout().run_looks_good(Recording([]), PROMPTS, verdict_rule='typo')

    def test_selection_up_to_six_images(self):
        record = {'semantic_codes': [[i] * 729 for i in range(6)]}
        for ordinal in range(1, 7):
            self.assertEqual(selection(record, ordinal)['selected_ordinal'], ordinal)
        self.assertTrue(selection(record, 6)['reused_final_image'])
        for bad in (0, MAX_ORDINAL + 1):
            with self.assertRaises(ValueError):
                selection(record, bad)

    def test_eval_cli(self):
        from llava.train.rl import eval_geneval2, generate_instruction_benchmarks
        required = ['x', '--checkpoint', 'c', '--benchmark', 'b', '--output_dir', 'o', '--ar_path', 'a',
                    '--encoder_path', 'e', '--decoder_path', 'd', '--code_manifest', 'm']
        for module, extra in ((eval_geneval2, ['--reward_model', 'r', '--reward_python', 'p']),
                              (generate_instruction_benchmarks, [])):
            ok = ['--protocol', 'looks_good', '--max_refinements', '5', '--reset_context_per_turn',
                  '--decoding', 'greedy']
            with patch('sys.argv', required + extra + ok):
                args = module.parse_args()
                self.assertEqual((args.decoding, args.verdict_rule, args.reset_context_per_turn), ('greedy', 'correction', True))
            for bad in (['--protocol', 'looks_good'],                    # context must be explicit
                        ['--decoding', 'greedy'],                        # native stays sampled
                        ['--verdict_rule', 'anywhere'],
                        ['--protocol', 'looks_good', '--no-reset_context_per_turn', '--decoding', 'beam']):
                with self.subTest(module=module.__name__, bad=bad), patch('sys.argv', required + extra + bad), \
                        patch('sys.stderr'), self.assertRaises(SystemExit):
                    module.parse_args()


if __name__ == '__main__':
    unittest.main()
