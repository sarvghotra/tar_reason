"""Pass/fail verdict target and batched reward rendering/judging."""
import copy
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace

import torch

from llava.train.rl import qwen_reward_worker as worker
from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig
from llava.train.rl.rollout import RolloutBatch, Trajectory
from test_rl_janus import CALLS, fixture
from test_rl_policy import LoraConfig, get_peft_model, policy_args, tiny_base

AMS = [.9, .3, .5, .2, .6, .8, .4, .1, .7, .35]


def judge_with(question_scores):
    """Fake judge returning the given per-question rows, in render order."""
    queue = [list(q) for q in question_scores]

    def score_images(images, vqa_lists):
        assert len(images) == len(queue)
        am = [sum(q) / len(q) for q in queue]
        return am, am, queue
    return SimpleNamespace(score_images=score_images)


class TaggingDecoder:
    """Returns one tagged image per code row and records the call sizes."""

    def __init__(self):
        self.calls = []

    def decode_codes(self, codes):
        self.calls.append(len(codes))
        return [('img', tuple(c)) for c in codes]


class VerdictTargetTests(unittest.TestCase):
    def test_all_pass_reward_cases_and_threshold_edges(self):
        self.assertTrue(trainer.verdict_target([.5, .9], .5))
        self.assertFalse(trainer.verdict_target([.49999, 1.3], .5))
        self.assertTrue(trainer.verdict_target([1.2, 1.0], .5))  # duplicate-variant scores can exceed 1
        _, batch = fixture()
        # Render order: e0 d | e1 d,r | e2 d,r | e3 d | e4 d,r | e5 d,r. Verdicts: 1,0,0,1,0,0.
        rows = [[.9, .9], [.2, .9], [.5, .5], [.9, .6], [.5, .5], [.8, .3], [.4, .45], [.6, .9],
                [.9, .9], [.1, .2]]
        trainer.score_one_iteration(batch, judge_with(rows), TaggingDecoder(), target='all_pass')
        drafts_pass = [True, False, True, False, False, True]
        verdicts = [True, False, False, True, False, False]
        for t, passed, verdict in zip(batch.trajectories, drafts_pass, verdicts):
            self.assertEqual(t.one_iter.verdict, verdict)
            self.assertEqual(t.one_iter.reward_verdict, float(passed == verdict))
        # soft_am keeps the Janus formula.
        _, soft = fixture()
        trainer.score_one_iteration(soft, judge_with(rows), TaggingDecoder(), target='soft_am')
        for t in soft.trajectories:
            self.assertAlmostEqual(t.one_iter.reward_verdict, 1 - abs(t.one_iter.am_draft - t.one_iter.verdict))
        with self.assertRaises(ValueError):
            trainer.score_one_iteration(fixture()[1], judge_with(rows), TaggingDecoder(), target='typo')

    def test_collapse_incentive_removed(self):
        # Failing drafts with high AM: soft_am pays "looks good" more; all_pass pays only refining.
        calls = [('image', [[8] * 3, [9] * 3]), ('text', [[16, 2], [17, 3]]), ('image', [[10] * 3])]
        rows = [[.95, .95, .4], [.95, .95, .4], [.9, .9, .9]]
        rewards = {}
        for target in ('soft_am', 'all_pass'):
            _, batch = fixture(calls=calls, num_rollouts=2)
            trainer.score_one_iteration(batch, judge_with(rows), TaggingDecoder(), target=target)
            rewards[target] = [t.one_iter.reward_verdict for t in batch.trajectories]
        kept, refined = rewards['soft_am']
        self.assertGreater(kept, refined)                    # the old reward prefers "looks good"
        self.assertEqual(rewards['all_pass'], [0.0, 1.0])    # the new one only rewards refining

    def test_stats_verdict_accuracy_precision_recall(self):
        _, batch = fixture()
        rows = [[.9, .9], [.2, .9], [.5, .5], [.9, .6], [.5, .5], [.8, .3], [.4, .45], [.6, .9],
                [.9, .9], [.1, .2]]
        trainer.score_one_iteration(batch, judge_with(rows), TaggingDecoder(), target='all_pass')
        f = trainer.finalize_janus_stats(trainer.janus_stats(batch, 'x/', .5), 'x/')
        # pass = [T,F,T,F,F,T], looks good = [T,F,F,T,F,F]: correct on e0, e1, e4 -> 3/6.
        self.assertAlmostEqual(f['x/verdict_accuracy'], .5)
        self.assertAlmostEqual(f['x/verdict_precision'], .5)  # 1 of 2 "looks good" drafts pass
        self.assertAlmostEqual(f['x/verdict_recall'], 1 / 3)  # 1 of 3 passing drafts kept
        self.assertAlmostEqual(f['x/verdict_accuracy'],
                               sum(t.one_iter.reward_verdict for t in batch.trajectories) / 6)

    def test_metadata_old_janus_checkpoint_resumes_only_as_soft_am(self):
        with tempfile.TemporaryDirectory() as directory:
            new = policy_args(output_dir=directory, save_total_limit=3, max_seq_len=32,
                              objective='janus_one_iteration', verdict_rule='correction', adv_std_floor=0.0,
                              verdict_target='all_pass', verdict_pass_threshold=.5,
                              render_batch_size=24, reward_question_batch_size=32)
            model = get_peft_model(tiny_base(), LoraConfig(r=2, target_modules=['q_proj'], task_type='CAUSAL_LM'))
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            trainer.save_checkpoint(model, opt, scheduler, 1, new, torch.device('cpu'))
            ck = Path(directory) / 'checkpoint-1'
            state = json.loads((ck / 'state.json').read_text())
            self.assertEqual(state['reward_batching'], dict(render_batch_size=24, reward_question_batch_size=32))
            self.assertEqual(state['rl_method']['verdict_target'], 'all_pass')
            trainer.validate_resume_checkpoint(ck, new)
            faster = copy.copy(new)
            faster.render_batch_size = 1  # throughput settings are not resume-guarded
            trainer.validate_resume_checkpoint(ck, faster)
            for key, value in (('verdict_target', 'soft_am'), ('verdict_pass_threshold', .6)):
                changed = copy.copy(new)
                setattr(changed, key, value)
                with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                    trainer.validate_resume_checkpoint(ck, changed)
            for key in ('verdict_target', 'verdict_pass_threshold'):
                del state['rl_method'][key]
            (ck / 'state.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                trainer.validate_resume_checkpoint(ck, new)
            old = copy.copy(new)
            old.verdict_target = 'soft_am'
            trainer.validate_resume_checkpoint(ck, old)
            # A non-Janus checkpoint never gains verdict keys.
            legacy = policy_args(max_seq_len=32)
            self.assertIsNone(trainer.rl_method_settings(legacy)['verdict_target'])

    def test_cli_defaults_and_refusals(self):
        from unittest.mock import patch
        required = ['train', '--model_name_or_path', 'm', '--data_path', 'd', '--output_dir', 'o']
        with patch('sys.argv', required):
            args = trainer.parse_args()
        self.assertEqual((args.verdict_target, args.render_batch_size, args.reward_question_batch_size),
                         ('all_pass', 1, 1))
        for bad in (['--render_batch_size', '0'], ['--reward_question_batch_size', '0'],
                    ['--verdict_pass_threshold', 'nan']):
            with self.subTest(bad=bad), patch('sys.argv', required + bad), patch('sys.stderr'), \
                    self.assertRaises(SystemExit):
                trainer.parse_args()


class BatchedRenderTests(unittest.TestCase):
    def test_render_chunks_preserve_order_and_owner_mapping(self):
        for size in (1, 3, 4, 24):
            _, batch = fixture()
            decoder = TaggingDecoder()
            trainer.score_one_iteration(batch, judge_with([[a] for a in AMS]), decoder, render_batch_size=size)
            self.assertEqual(sum(decoder.calls), 10)
            self.assertEqual(len(decoder.calls), -(-10 // size))
            self.assertTrue(all(c <= size for c in decoder.calls))
            for t in batch.trajectories:
                self.assertEqual([img[1] for img in t.scored_images], [tuple(c) for c in t.images])
                self.assertIs(t.final_image, t.scored_images[-1])
            kept = batch.trajectories[0]
            self.assertEqual(len(kept.scored_images), 1)
            self.assertAlmostEqual(kept.am, .9)

    def test_score_rollouts_batched_dedups_and_maps(self):
        trajectories = [Trajectory(i % 2, [], [], images=[[i, i]]) for i in range(5)]
        prompts = [{'vqa_list': [('q', 'yes')]}, {'vqa_list': [('q', 'yes')]}]
        batch = RolloutBatch(prompts, trajectories)
        decoder = TaggingDecoder()
        trainer.score_rollouts(batch, judge_with([[.1 * (i + 1)] for i in range(5)]), decoder, render_batch_size=2)
        self.assertEqual(decoder.calls, [2, 2, 1])
        for i, t in enumerate(trajectories):
            self.assertEqual(t.final_image, ('img', (i, i)))
            self.assertAlmostEqual(t.am, .1 * (i + 1))

    def test_render_count_mismatch_raises(self):
        broken = SimpleNamespace(decode_codes=lambda codes: [object()])
        with self.assertRaises(RuntimeError):
            trainer.render_codes(broken, [[1], [2]], 2)


VOCAB = 64


class FakeTokenizer:
    def __init__(self):
        self.padding_side = 'right'
        # Duplicate first tokens on purpose (" yes" == "yes"), as GenEval2 sums them twice.
        self.table = {'Yes': [3], 'yes': [4], ' yes': [4], ' Yes': [5], 'two': [6], 'Two': [7],
                      ' two': [8], ' Two': [8], '2': [9], ' 2': [1, 9]}

    def encode(self, text):
        return self.table[text]


def row_token(image, text):
    return 10 + zlib.crc32(f'{image}|{text}'.encode()) % (VOCAB - 10)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.batch_sizes = []

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                            return_tensors='pt', padding=False):
        single = isinstance(messages[0], dict)
        conversations = [messages] if single else messages
        rows = []
        for conv in conversations:
            image, text = conv[0]['content'][0]['image'], conv[0]['content'][1]['text']
            rows.append([2] * (1 + len(text) % 5) + [row_token(image, text)])
        self.batch_sizes.append(len(rows))
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):
            if self.tokenizer.padding_side == 'left':
                ids[i, width - len(row):] = torch.tensor(row)
            else:
                ids[i, :len(row)] = torch.tensor(row)
        return Inputs(input_ids=ids)


class Inputs(dict):
    def to(self, device):
        return self


class FakeJudgeModel:
    device = 'cpu'

    @staticmethod
    def logits_for(last):
        return torch.sin(last.double()[:, None] * torch.arange(VOCAB, dtype=torch.float64)[None]) * 4

    def __call__(self, input_ids, logits_to_keep=1):
        return SimpleNamespace(logits=self.logits_for(input_ids[:, -1])[:, None, :].float())

    def generate(self, input_ids, **kwargs):
        assert input_ids.shape[0] == 1 and kwargs['max_new_tokens'] == 1 and not kwargs['do_sample']
        return SimpleNamespace(scores=[self.logits_for(input_ids[:, -1]).float()])


class BatchedJudgeTests(unittest.TestCase):
    def setUp(self):
        self.images = ['a.png', 'b.png', 'c.png']
        self.vqa = [[('Is there a dog?', 'yes'), ('How many cats?', 'two')],
                    [('How many cats?', 'two'), ('Is it red?', 'yes'), ('Is it big?', 'yes')],
                    [('Is there a cup?', 'yes')]]

    def test_batched_matches_legacy_across_chunks_with_duplicates(self):
        model, processor = FakeJudgeModel(), FakeProcessor()
        legacy = worker.score_request(model, processor, self.images, self.vqa, 1)
        self.assertEqual(processor.batch_sizes, [1] * 6)  # batch 1 = per-question generate
        for size in (2, 3, 4, 32):
            processor = FakeProcessor()
            got = worker.score_request(model, processor, self.images, self.vqa, size)
            self.assertEqual([len(r) for r in got], [2, 3, 1])
            for a, b in zip(legacy, got):
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y, places=6)
            self.assertEqual(sum(processor.batch_sizes), 6)
            self.assertEqual(processor.tokenizer.padding_side, 'right')  # restored
        # Duplicate first tokens are summed twice ("yes" and " yes" both map to id 4).
        probs = torch.softmax(FakeJudgeModel.logits_for(
            torch.tensor([row_token('a.png', 'Is there a dog? Answer in one word.')])).float(), -1)[0]
        expected = float(probs[3] + 2 * probs[4] + probs[5])
        self.assertAlmostEqual(legacy[0][0], expected, places=6)

    def test_right_padding_would_be_wrong(self):
        model, processor = FakeJudgeModel(), FakeProcessor()
        original = worker.score_batched
        legacy = worker.score_request(model, processor, self.images, self.vqa, 1)

        class RightPadProcessor(FakeProcessor):
            def apply_chat_template(self, *a, **k):
                self.tokenizer.padding_side = 'right'
                return super().apply_chat_template(*a, **k)
        broken = worker.score_request(model, RightPadProcessor(), self.images, self.vqa, 6)
        self.assertNotEqual([round(x, 6) for r in legacy for x in r], [round(x, 6) for r in broken for x in r])
        self.assertIs(worker.score_batched, original)

    def test_client_command_keeps_legacy_for_batch_one(self):
        from llava.train.rl import pixel_reward
        seen = []

        class Stop(Exception):
            pass

        def popen(command, **kwargs):
            seen.append(command)
            raise Stop
        from unittest.mock import patch
        with patch.object(pixel_reward.subprocess, 'Popen', side_effect=popen):
            for size in (1, 8):
                with self.assertRaises(Stop):
                    pixel_reward.QwenPixelReward('py', 'model', 'cuda:0', 10, question_batch_size=size)
        self.assertNotIn('--question_batch_size', seen[0])
        self.assertEqual(seen[1][-2:], ['--question_batch_size', '8'])
        with self.assertRaises(ValueError):
            pixel_reward.QwenPixelReward('py', 'model', 'cuda:0', 10, question_batch_size=0)


if __name__ == '__main__':
    unittest.main()
