"""Adversarial checks for --objective teacher_tree (teacher reflections, shadow verdicts)."""
import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_NONE, KIND_TXT, grpo_loss, masked_token_logprobs
from llava.train.rl.pixel_reward import QwenPixelReward
from llava.train.rl.qwen_reward_worker import generate_request
from llava.train.rl.rollout import SEGMENT_TEACHER
from llava.train.rl.teacher import LOOKS_GOOD_TEXT, parse_teacher_text, teacher_instruction
from test_rl_episodes import ScriptedModel
from test_rl_policy import small_rollout, policy_args, tiny_base, get_peft_model, LoraConfig
from test_rl_reset_context import new_model

# id -> text. 2 = EOS, 3 = <im_start>, 4 = <im_end>, 5 = <S0>, 8..15 image tokens.
REFINE_TEXT = " no dog\nCorrection: add a dog\n"
TEXT = {0: '', 1: '<p>', 2: '<|im_end|>', 3: '<im_start>', 4: '<im_end>', 5: '<S0>', 6: '\n', 7: ' x',
        16: ' Correction: looks good.', 17: ' Correction: add a dog', 20: LOOKS_GOOD_TEXT, 21: ' no dog',
        22: '\nCorrection: add a dog\n'}
SPECIAL = {0, 2, 3, 4, 5}
ENCODE = {LOOKS_GOOD_TEXT: [20], REFINE_TEXT: [21, 22]}
PROMPTS = [{'prompt': 'a dog', 'vqa_list': [('Is there a dog?', 'Yes'), ('Is the dog red?', 'Yes')]}]
LOOKS_GOOD_REPLY = 'Self-reflect: no issues, it matches the prompt.\nCorrection: looks good.'
REFINE_REPLY = 'no dog\nCorrection: add a dog'


class FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return ''.join('' if (skip_special_tokens and i in SPECIAL) else TEXT.get(i, f'<I{i}>') for i in ids)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(ENCODE[text]))


def teacher_rollout(reset=False, max_refinements=2, reflect_tokens=4, num_rollouts=2):
    rollout = small_rollout()
    rollout.cfg.num_rollouts = num_rollouts
    rollout.cfg.max_seq_len = 100
    rollout.cfg.max_refinements = max_refinements
    rollout.cfg.reflect_tokens = reflect_tokens
    rollout.cfg.reset_context_per_turn = reset
    rollout.reflect_splice = [4, 6]
    rollout.image_splice = [5]
    rollout.make_prefix = lambda prompt: [1, 7]
    rollout.tok = FakeTok()
    return rollout


class FakeTeacher:
    """Scripted teacher_fn: one list of replies per round; records its inputs."""

    def __init__(self, rounds):
        self.rounds, self.calls, self.rendered = list(rounds), [], 0

    def __call__(self, prompts, codes, ask):
        self.calls.append((list(prompts), [list(c) for c in codes], list(ask)))
        replies = self.rounds.pop(0)
        images = []
        for _ in codes:
            images.append(f'pixels-{self.rendered}')
            self.rendered += 1
        return list(zip(images, replies))


# Draft A: teacher correction -> 2 images A1, A2. Draft B: teacher looks good (stop).
# A1: teacher looks good (stop). A2: correction -> A2'. A2' is at the cap (not asked).
CALLS = [('image', [[8] * 3, [9] * 3]),          # drafts A, B
         ('text', [[17, 3], [16, 2]]),            # shadows after A, B
         ('image', [[10] * 3, [11] * 3]),         # A1, A2
         ('text', [[17, 2], [16, 3]]),            # shadows after A1 (EOS mismatch), A2 (<im_start> mismatch)
         ('image', [[12] * 3]),                   # A2'
         ('text', [[7, 7, 7, 7]])]                # shadow after A2' (token cap)
ROUNDS = [[REFINE_REPLY, LOOKS_GOOD_REPLY], ['Correction: looks good.', REFINE_REPLY], [None]]
AMS = {'pixels-0': .4, 'pixels-1': .9, 'pixels-2': .8, 'pixels-3': .3, 'pixels-4': .6}  # A B A1 A2 A2'


def fixture(reset=False, calls=CALLS, rounds=ROUNDS, **kwargs):
    rollout = teacher_rollout(reset=reset, **{k: v for k, v in kwargs.items() if k in
                                              ('max_refinements', 'reflect_tokens', 'num_rollouts')})
    model = ScriptedModel(calls)
    teacher = FakeTeacher(rounds)
    batch = rollout.run_teacher_tree(model, PROMPTS, kwargs.get('branch', '1x2'), teacher,
                                     num_shadows=kwargs.get('num_shadows', 1))
    assert model.calls == [], model.calls
    assert teacher.rounds == [], teacher.rounds
    return rollout, batch, teacher


class FakeJudge:
    def __init__(self, ams):
        self.ams, self.requests = ams, []

    def score_images(self, images, vqa_lists):
        self.requests.append(list(images))
        am = [self.ams[i] for i in images]
        return am, [a / 2 for a in am], [[a] * len(q) for a, q in zip(am, vqa_lists)]


# A1 now fails all_pass although the teacher said "looks good" there: the only disagreement.
AMS_DISAGREE = dict(AMS, **{'pixels-2': .3})


def scored(reset=False, norm='mean', ams=AMS, ce_filter='all'):
    rollout, batch, teacher = fixture(reset)
    judge = FakeJudge(ams)
    trainer.score_teacher_tree(batch, judge, ce_filter=ce_filter)
    cfg = GRPOConfig(adv_norm=norm)
    trainer.assign_tree_advantages(batch, cfg)
    trainer.assign_shadow_advantages(batch, cfg)
    return rollout, batch, teacher, judge


class TeacherParsingTests(unittest.TestCase):
    def test_policy_format_and_malformed_replies(self):
        self.assertEqual(parse_teacher_text(LOOKS_GOOD_REPLY), (LOOKS_GOOD_TEXT, True))
        self.assertEqual(parse_teacher_text('The image looks good.'), (LOOKS_GOOD_TEXT, True))
        self.assertEqual(parse_teacher_text('  there are  seven giraffes\n\nCorrection:  remove one giraffe. '),
                         (' there are seven giraffes\nCorrection: remove one giraffe.\n', False))
        self.assertEqual(parse_teacher_text('the dog looks good but no cat\nCorrection: add a cat'),
                         (' the dog looks good but no cat\nCorrection: add a cat\n', False))
        for bad in (None, '', '  ', 'no correction clause', 'x\nCorrection:', 'Correction: add a cat',
                    'a\nCorrection: b\nCorrection: c', 'x <im_start>\nCorrection: y', 'x\nCorrection: <I3>'):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_teacher_text(bad))
        self.assertIn('"a red cup"', teacher_instruction(' a red cup '))


class TeacherRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tree_shape_stops_tokens_and_teacher_inputs(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                rollout, batch, teacher = fixture(reset)
                nodes = batch.nodes
                self.assertEqual([n.kind for n in nodes], ['image', 'image', 'reflection', 'reflection', 'image',
                                                           'image', 'reflection', 'reflection', 'image'])
                self.assertEqual([n.teacher for n in nodes], [False, False, True, True, False, False, True, True,
                                                              False])
                A, B, tA, tB, A1, A2, tA1, tA2, A2b = nodes
                self.assertEqual(tA.actions, [21, 22, 3])
                self.assertEqual(tB.actions, [20, 2])
                self.assertEqual(tA1.actions, [20, 2])
                self.assertEqual(tA2.actions, [21, 22, 3])
                self.assertTrue(all(n.action_kinds == [KIND_TXT] * len(n.actions) for n in nodes if n.teacher))
                self.assertEqual(tA.children, [A1, A2])
                self.assertEqual([t.stop_reason for t in batch.trajectories],
                                 ['teacher_looks_good', 'max_refinements', 'teacher_looks_good'])
                self.assertEqual([t.nodes[-1] for t in batch.trajectories], [tA1, A2b, tB])
                self.assertEqual([t.mass for t in batch.trajectories], [.5, .5, 1.0])
                self.assertEqual([len(t.images) for t in batch.trajectories], [2, 3, 1])
                self.assertEqual(batch.trajectories[1].reflections,
                                 ['no dog\nCorrection: add a dog', 'no dog\nCorrection: add a dog'])
                # Teacher inputs: semantic codes, ask only below the image cap; exact pixels kept.
                self.assertEqual([c[2] for c in teacher.calls], [[True, True], [True, True], [False]])
                self.assertEqual(teacher.calls[0][1], [[0] * 3, [1] * 3])
                self.assertEqual(teacher.calls[0][0], ['a dog', 'a dog'])
                self.assertEqual([n.image for n in (A, B, A1, A2, A2b)], [f'pixels-{i}' for i in range(5)])
                self.assertIsNone(A2b.teacher_reply)
                self.assertEqual([n.teacher_verdict for n in (A, B, A1, A2, A2b)], [False, True, True, False, None])
                # The teacher reflection extends exactly the context its image's shadows saw.
                for image, reflection in ((A, tA), (B, tB), (A1, tA1), (A2, tA2)):
                    self.assertEqual(len(image.shadows), 1)
                    self.assertEqual(image.shadows[0].context, reflection.context)
                if reset:
                    self.assertEqual(tA2.context, [1, 7, 11, 11, 11, 4, 6])
                    self.assertEqual(A2b.context, [1, 7, 11, 11, 11, 4, 6, 21, 22, 3, 5])
                    self.assertEqual(A2b.shadows[0].context, [1, 7, 12, 12, 12, 4, 6])
                    self.assertEqual([len(t.segments) for t in batch.trajectories], [2, 2, 1])
                else:
                    self.assertEqual(A2b.context, [1, 7, 8, 8, 8, 4, 6, 21, 22, 3, 5, 11, 11, 11, 4, 6,
                                                   21, 22, 3, 5])
                    self.assertEqual(A2b.shadows[0].context, A2b.context + [12] * 3 + [4, 6])
                    t = batch.trajectories[1]
                    self.assertEqual(t.seq, A2b.context + [12] * 3)
                    self.assertEqual(len(t.seq), len(t.kinds))
                shadows = [n.shadows[0] for n in (A, B, A1, A2, A2b)]
                self.assertEqual([s.verdict for s in shadows], [False, True, False, True, False])
                self.assertEqual([s.stop_token_agrees for s in shadows], [True, True, False, False, False])
                self.assertEqual([s.capped for s in shadows], [False, False, False, False, True])

    def test_malformed_long_contract_and_spec_errors(self):
        calls = [('image', [[8] * 3, [9] * 3]), ('text', [[17, 3], [16, 2]])]
        _, batch, _ = fixture(calls=calls, rounds=[['garbage', LOOKS_GOOD_REPLY]], branch='1')
        self.assertEqual([t.stop_reason for t in batch.trajectories], ['teacher_malformed', 'teacher_looks_good'])
        self.assertEqual([n.kind for n in batch.nodes], ['image', 'image', 'reflection'])
        self.assertIs(batch.trajectories[0].nodes[-1], batch.nodes[0])  # the leaf ends on its image
        # REFINE_TEXT + <im_start> is 3 tokens: too long for a 2-token reflection budget.
        _, batch, _ = fixture(calls=[('image', [[8] * 3, [9] * 3]), ('text', [[17, 3], [16, 2]])],
                              rounds=[[REFINE_REPLY, REFINE_REPLY]], reflect_tokens=2)
        self.assertEqual([t.stop_reason for t in batch.trajectories], ['teacher_malformed'] * 2)
        self.assertEqual(len(batch.nodes), 2)
        # No shadows: no text generation at all.
        _, batch, _ = fixture(calls=[('image', [[8] * 3, [9] * 3])], rounds=[[LOOKS_GOOD_REPLY] * 2],
                              num_shadows=0)
        self.assertTrue(all(not n.shadows for n in batch.nodes))
        self.assertEqual([t.stop_reason for t in batch.trajectories], ['teacher_looks_good'] * 2)
        for rounds in ([[None, LOOKS_GOOD_REPLY]],        # asked but no reply
                       [['x']]):                         # wrong count
            with self.subTest(rounds=rounds), self.assertRaises(RuntimeError):
                fixture(calls=[('image', [[8] * 3, [9] * 3]), ('text', [[17, 3], [16, 2]])], rounds=rounds)
        with self.assertRaises(RuntimeError):  # replied for a cap image
            fixture(calls=[('image', [[8] * 3, [9] * 3]), ('text', [[17, 3], [16, 2]])],
                    rounds=[[LOOKS_GOOD_REPLY] * 2], max_refinements=0)
        with self.assertRaisesRegex(ValueError, 'R = 1'):
            fixture(branch='2')


class TeacherCreditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_judging_rewards_values_and_advantages(self):
        for reset in (False, True):
            rollout, batch, teacher, judge = scored(reset)
            A, B, tA, tB, A1, A2, tA1, tA2, A2b = batch.nodes
            self.assertEqual(judge.requests, [[f'pixels-{i}' for i in range(5)]])  # one call, teacher's pixels
            self.assertEqual([t.reward for t in batch.trajectories], [.8, .6, .9])
            self.assertEqual(batch.trajectories[1].scored_images, ['pixels-0', 'pixels-3', 'pixels-4'])
            self.assertEqual(batch.trajectories[1].final_image, 'pixels-4')
            # Intermediate images are credited with the final rewards they led to.
            for node, value in ((tA1, .8), (A1, .8), (A2b, .6), (tA2, .6), (A2, .6), (tA, .7), (A, .7),
                                (tB, .9), (B, .9)):
                self.assertAlmostEqual(node.value, value, places=12)
            for node, adv in ((A, -.1), (B, .1), (tA, -.1), (A1, .1), (A2, -.1), (tA1, .1), (tA2, -.1),
                              (A2b, -.1), (tB, .1)):
                self.assertAlmostEqual(node.adv, adv, places=12)
            # Shadows: correct verdict = all_pass of the image they follow (A fail, B pass, A1 pass, A2 fail, A2' pass).
            shadows = [n.shadows[0] for n in (A, B, A1, A2, A2b)]
            self.assertEqual([s.reward for s in shadows], [1.0, 1.0, 0.0, 0.0, 0.0])
            for s, adv in zip(shadows, [.6, .6, -.4, -.4, -.4]):
                self.assertAlmostEqual(s.adv, adv, places=12)
        _, batch, _, _ = scored(norm='std')
        std = math.sqrt(1.2 / 4)  # unbiased std of [1, 1, 0, 0, 0]
        self.assertAlmostEqual(batch.nodes[0].shadows[0].adv, .6 / (std + 1e-4), places=9)
        leaf_std = math.sqrt((.5 * 0 + .5 * .2 ** 2 + 1 * .1 ** 2) / 2)  # mass-weighted
        self.assertAlmostEqual(batch.nodes[0].adv, -.1 / (leaf_std + 1e-4), places=9)

    def test_stats(self):
        _, batch, _, _ = scored()
        s = trainer.finalize_teacher_tree_stats(trainer.teacher_tree_stats(batch, 'x/'), 'x/')
        expected = {'am_final': .8, 'teacher_looks_good_rate': .5, 'stop_max_refinements_rate': .25,
                    'stop_teacher_looks_good_rate': .75, 'stop_teacher_malformed_rate': 0.0,
                    'am_draft': .65, 'draft_pass_rate': .5, 'delta_am_refined': .2, 'improved_rate': 2 / 3,
                    'teacher_verdict_accuracy': 1.0, 'teacher_malformed_reply_rate': 0.0,
                    'shadow_verdict_accuracy': .4, 'shadow_looks_good_on_fail': .5,
                    'shadow_looks_good_on_pass': 1 / 3, 'shadow_mismatch_rate': .4, 'shadow_capped_rate': .2,
                    'images_per_draft': 2.5, 'refinements': .75}
        for key, value in expected.items():
            self.assertAlmostEqual(s[f'x/{key}'], value, places=12, msg=key)


def row_by_tokens(rows, tokens):
    for i in range(len(rows['input_ids'])):
        n = int(rows['attention_mask'][i].sum())
        if rows['input_ids'][i, :n].tolist() == tokens:
            return {k: rows[k][i, :n] for k in ('pos_kind', 'adv', 'weight', 'ce_weight', 'pg_mask', 'pos_segment')}
    raise AssertionError(f'no row {tokens}')


class TeacherRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_row_weights_masks_and_segments(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                rollout, batch, _, _ = scored(reset)
                c, s = .3, .7
                rows = rollout.build_teacher_tree_rows(batch, c, s)
                kinds, weight, ce, pg = rows['pos_kind'], rows['weight'], rows['ce_weight'], rows['pg_mask']
                segment, train = rows['pos_segment'], rows['train_mask']
                drafts = 2
                img = kinds == KIND_IMG
                self.assertAlmostEqual(float(weight[img].sum()), drafts, places=5)    # one per draft subtree
                self.assertAlmostEqual(float(ce.sum()), c * drafts, places=5)          # c x mean teacher NLL
                shadow = segment == 2
                self.assertAlmostEqual(float(weight[shadow].sum()), s * drafts, places=5)
                teacher = segment == SEGMENT_TEACHER
                self.assertEqual(int(teacher.sum()), 3 + 2 + 2 + 3)
                self.assertTrue(bool((weight[teacher] == 0).all() and (pg[teacher] == 0).all()))
                self.assertTrue(bool((kinds[teacher] == KIND_TXT).all() and (ce[~teacher] == 0).all()))
                self.assertTrue(bool((train == (kinds != KIND_NONE)).all()))
                self.assertEqual(int(((pg == 0) & train & ~teacher).sum()), 2)  # two shadow stop mismatches
                # Exact rows: B + its teacher reflection; one shadow row with a stop mismatch.
                b = row_by_tokens(rows, [1, 7, 9, 9, 9, 4, 6, 20, 2])
                self.assertEqual(b['pos_kind'].tolist(), [0, 0, 1, 1, 1, 0, 0, 2, 2])
                torch.testing.assert_close(b['weight'], torch.tensor([0, 0] + [1 / 3] * 3 + [0] * 4))
                torch.testing.assert_close(b['ce_weight'], torch.tensor([0.] * 7 + [c * drafts / (4 * 2)] * 2))
                torch.testing.assert_close(b['adv'], torch.tensor([0, 0] + [.1] * 3 + [0] * 4))
                self.assertEqual(b['pos_segment'].tolist(), [0, 0, 1, 1, 1, 0, 0, 4, 4])
                a2 = batch.nodes[5]
                m = row_by_tokens(rows, a2.shadows[0].context + [16, 3])
                self.assertEqual(m['pg_mask'][-2:].tolist(), [1.0, 0.0])
                torch.testing.assert_close(m['weight'][-2:], torch.tensor([s * drafts / (5 * 2)] * 2))
                torch.testing.assert_close(m['adv'][-2:], torch.tensor([-.4, -.4]))
                # Image weights exclude teacher tokens from each leaf's normalizer.
                expected = {0: .5 / 6 + .5 / 9, 4: .5 / 6, 5: .5 / 9, 8: .5 / 9, 1: 1 / 3}
                for node_id, w in expected.items():
                    node = batch.nodes[node_id]
                    found = [float(weight[i, j]) for i in range(len(weight)) for j in range(weight.shape[1])
                             if kinds[i, j] == KIND_IMG and int(rows['input_ids'][i, j]) == node.actions[0]
                             and train[i, j]]
                    self.assertTrue(found and all(abs(x - w) < 1e-6 for x in found), (node_id, found, w))
        with self.assertRaises(ValueError):
            rollout.build_teacher_tree_rows(batch, -1.0, 1.0)

    def test_zero_distillation_weight_keeps_teacher_tokens_out_of_the_objective(self):
        rollout, batch, _, _ = scored(reset=True)
        rows = rollout.build_teacher_tree_rows(batch, 0.0, .7)
        self.assertEqual(float(rows['ce_weight'].sum()), 0.0)
        self.assertEqual(int(rows['teacher_mask'].sum()), 10)
        self.assertTrue(bool((rows['weight'][rows['teacher_mask']] == 0).all()))
        params, metrics = [], {}
        for coef in (0.0, 1e-12):  # coef 0 still reports teacher NLL; tiny coef ~ same update
            model = new_model()
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            args = policy_args(train_micro_batch=2, max_grad_norm=1e6, num_ppo_epochs=1,
                               objective='teacher_tree', teacher_ce_coef=coef, shadow_coef=.7)
            metrics[coef] = trainer.train_on_rollouts(model, rollout, batch, args, GRPOConfig(), opt, scheduler,
                                                      torch.device('cpu'))
            params.append([p.detach().clone() for p in model.parameters() if p.requires_grad])
        self.assertEqual(metrics[0.0]['ce_loss'], 0.0)
        self.assertGreater(metrics[0.0]['teacher_nll'], 0.0)
        self.assertAlmostEqual(metrics[0.0]['teacher_nll'], metrics[1e-12]['teacher_nll'], places=6)
        self.assertAlmostEqual(metrics[0.0]['kl_txt'], metrics[1e-12]['kl_txt'], places=6)
        for p, q in zip(*params):
            torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-9)
        # Text-phase metrics exclude teacher tokens whatever their CE weight.
        torch.manual_seed(1)
        logp, old, ref, adv = (torch.randn(5) for _ in range(4))
        kinds, weight = torch.tensor([2, 2, 2, 2, 1]), torch.tensor([.2, .2, 0, 0, .2])
        teacher = torch.tensor([False, False, True, True, False])
        _, m = grpo_loss(logp, old, ref, adv, weight, kinds, 1.0, GRPOConfig(), ce_weight=torch.zeros(5),
                         teacher_mask=teacher)
        d = ref[:2] - logp[:2]
        self.assertAlmostEqual(m['kl_txt'], float((d.exp() - d - 1).mean()), places=6)
        self.assertEqual(m['sum_teacher_tokens'], 2.0)

    def test_loss_without_ce_weight_is_unchanged(self):
        torch.manual_seed(3)
        logp, old, ref, adv = (torch.randn(7) for _ in range(4))
        weight, kinds = torch.rand(7), torch.tensor([1, 1, 2, 2, 2, 1, 2])
        base, base_metrics = grpo_loss(logp, old, ref, adv, weight, kinds, 3.0, GRPOConfig())
        zero, zero_metrics = grpo_loss(logp, old, ref, adv, weight, kinds, 3.0, GRPOConfig(),
                                       ce_weight=torch.zeros(7))
        self.assertEqual(float(base), float(zero))
        self.assertEqual({k: v for k, v in zero_metrics.items() if not k.startswith('sum_')}, base_metrics)
        ce = torch.tensor([0, 0, 0, .5, .25, 0, 0])
        with_ce, metrics = grpo_loss(logp, old, ref, adv, weight, kinds, 3.0, GRPOConfig(), ce_weight=ce)
        self.assertAlmostEqual(float(with_ce - base), float((ce * -logp).sum() / 3.0), places=6)
        self.assertEqual(metrics['sum_teacher_tokens'], 2.0)
        self.assertAlmostEqual(metrics['sum_teacher_nll'], float(-(logp[3] + logp[4])), places=6)

    def test_agreement_filter_distills_only_verdicts_matching_the_judge(self):
        for ce_filter, distilled in (('all', [True] * 4), ('agree', [True, True, False, True])):
            with self.subTest(ce_filter=ce_filter):
                rollout, batch, _, _ = scored(ams=AMS_DISAGREE, ce_filter=ce_filter)
                teachers = [n for n in batch.nodes if n.teacher]  # tA, tB, tA1, tA2
                self.assertEqual([n.distill for n in teachers], distilled)
                c = .3
                rows = rollout.build_teacher_tree_rows(batch, c, .7)
                self.assertAlmostEqual(float(rows['ce_weight'].sum()), c * 2, places=5)  # mean over distilled
                b = row_by_tokens(rows, [1, 7, 9, 9, 9, 4, 6, 20, 2])
                torch.testing.assert_close(b['ce_weight'][-2:], torch.tensor([c * 2 / (sum(distilled) * 2)] * 2))
                a1 = batch.nodes[4]
                ids, ce = rows['input_ids'], rows['ce_weight']
                tA1 = [float(ce[i, j]) for i in range(len(ids)) for j in range(ids.shape[1] - 1)
                       if ids[i, j] == 10 and ids[i, j + 1] == 4 and rows['pos_kind'][i, j] == KIND_IMG]
                self.assertTrue(tA1)  # A1's row continues with its teacher reflection
                if ce_filter == 'agree':
                    for i in range(len(ids)):
                        n = int(rows['attention_mask'][i].sum())
                        if ids[i, :n].tolist()[-2:] == [20, 2] and 10 in ids[i, :n].tolist():
                            self.assertEqual(rows['ce_weight'][i, n - 2:n].tolist(), [0.0, 0.0])
                            self.assertEqual(rows['weight'][i, n - 2:n].tolist(), [0.0, 0.0])
                s = trainer.finalize_teacher_tree_stats(trainer.teacher_tree_stats(batch, 'x/'), 'x/')
                self.assertAlmostEqual(s['x/teacher_distill_rate'], sum(distilled) / 4)
                self.assertAlmostEqual(s['x/teacher_distilled_looks_good_share'],
                                       (1 + distilled[2]) / sum(distilled))
                self.assertAlmostEqual(s['x/teacher_verdict_accuracy'], .75)
        with self.assertRaises(ValueError):
            scored(ce_filter='typo')

    def test_teacher_cross_entropy_gradient_is_exact(self):
        """SGD step difference between teacher_ce_coef c and 0 = -lr * c * grad(mean NLL of the
        distilled teacher reflections). The PG/KL terms are identical in both runs and cancel."""
        for reset, ams, ce_filter in ((False, AMS, 'all'), (True, AMS, 'all'), (True, AMS_DISAGREE, 'agree')):
            with self.subTest(reset=reset, ce_filter=ce_filter):
                rollout, batch, _, _ = scored(reset, ams=ams, ce_filter=ce_filter)
                lr, c = .2, .5
                params = []
                metrics = {}
                for coef in (0.0, c):
                    model = new_model()
                    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
                    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
                    args = policy_args(train_micro_batch=2, max_grad_norm=1e6, num_ppo_epochs=1,
                                       objective='teacher_tree', teacher_ce_coef=coef, shadow_coef=.7)
                    metrics[coef] = trainer.train_on_rollouts(model, rollout, batch, args,
                                                              GRPOConfig(kl_gradient_correction=True), opt,
                                                              scheduler, torch.device('cpu'))
                    params.append([p.detach().clone() for p in model.parameters() if p.requires_grad])
                model = new_model()
                nll, nll_sum, tokens = [], 0.0, 0
                for node in (n for n in batch.nodes if n.teacher):
                    seq = node.context + node.actions
                    ids = torch.tensor([seq])
                    train = torch.zeros_like(ids, dtype=torch.bool)
                    train[0, len(node.context):] = True
                    kinds = torch.where(train, torch.tensor(KIND_TXT), torch.tensor(KIND_NONE))
                    logp = masked_token_logprobs(model, ids, torch.ones_like(ids), train, kinds, 8, 16)
                    if node.distill:
                        nll.append(-logp.mean())
                    nll_sum += float(-logp.sum())  # teacher_nll covers every teacher token
                    tokens += len(node.actions)
                loss = torch.stack(nll).mean()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                for p0, pc, g in zip(params[0], params[1], grads):
                    torch.testing.assert_close(pc - p0, -lr * c * g, rtol=1e-4, atol=1e-7)
                self.assertAlmostEqual(metrics[c]['ce_loss'], c * float(loss), places=5)
                self.assertEqual(tokens, 10)
                self.assertAlmostEqual(metrics[c]['teacher_nll'], nll_sum / tokens, places=5)
                for key in ('loss', 'pg_loss', 'kl', 'pg_draft', 'pg_shadow', 'pg_refined', 'kl_shadow'):
                    self.assertTrue(math.isfinite(metrics[c][key]), key)
                self.assertAlmostEqual(metrics[0.0]['pg_shadow'], metrics[c]['pg_shadow'], places=6)


class TeacherIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_teacher_reflector_renders_all_and_asks_where_requested(self):
        rendered, asked = [], []
        decoder = SimpleNamespace(decode_codes=lambda codes: [rendered.append(c) or f'img{len(rendered)}'
                                                              for c in codes])
        reward = SimpleNamespace(generate_texts=lambda images, texts, max_new_tokens, batch_size:
                                 asked.append((images, texts, max_new_tokens, batch_size)) or ['r1', 'r3'])
        reflector = trainer.TeacherReflector(reward, decoder, render_batch_size=2, max_new_tokens=50, batch_size=4)
        out = reflector(['p1', 'p2', 'p3'], [[1], [2], [3]], [True, False, True])
        self.assertEqual(out, [('img1', 'r1'), ('img2', None), ('img3', 'r3')])
        self.assertEqual(rendered, [[1], [2], [3]])
        self.assertEqual(asked, [(['img1', 'img3'], [teacher_instruction('p1'), teacher_instruction('p3')], 50, 4)])
        self.assertEqual(reflector([], [], []), [])

    def test_validation_uses_self_reflections_and_judges_every_image(self):
        rollout = teacher_rollout(reset=True)
        model = ScriptedModel([('image', [[8] * 3]), ('text', [[17, 3]]), ('image', [[9] * 3]), ('text', [[16, 2]])])
        judge = SimpleNamespace(score_images=lambda images, vqa: ([.3, .7], [.1, .2], [[.3, .3], [.7, .7]]))
        decoder = SimpleNamespace(decode_codes=lambda codes: [object() for _ in codes])
        args = SimpleNamespace(seed=1, gen_batch_size=4, objective='teacher_tree', verdict_rule='correction',
                               verdict_pass_threshold=.5, render_batch_size=2, log_images=0)
        out = trainer.run_validation(model, rollout, judge, PROMPTS, args, torch.device('cpu'), 0, decoder)
        self.assertEqual(model.calls, [])
        expected = {'val/am_draft': .3, 'val/am_final': .7, 'val/am_upto_1': .3, 'val/am_upto_2': .7,
                    'val/am_upto_3': .7, 'val/looks_good_rate': 1.0, 'val/verdict_accuracy': 1.0,
                    'val/refinements': 1.0, 'val/looks_good_on_fail': 0.0, 'val/verdict_recall': 1.0}
        for key, value in expected.items():
            self.assertAlmostEqual(out[key], value, places=12, msg=key)

    def test_worker_generate_request_batches_and_restores_padding(self):
        class Inputs(dict):
            def to(self, device):
                return self
        seen = []
        tokenizer = SimpleNamespace(padding_side='right')
        def template(messages, **kwargs):
            seen.append((messages, kwargs, tokenizer.padding_side))
            return Inputs(input_ids=torch.ones(len(messages), 3, dtype=torch.long))
        def generate(**kwargs):
            self.assertFalse(kwargs['do_sample'])
            self.assertEqual(kwargs['max_new_tokens'], 9)
            n = len(kwargs['input_ids'])
            return torch.cat([kwargs['input_ids'], torch.arange(2 * n).view(n, 2)], dim=1)
        processor = SimpleNamespace(apply_chat_template=template, tokenizer=tokenizer,
                                    batch_decode=lambda ids, skip_special_tokens: [str(r.tolist()) for r in ids])
        replies = generate_request(SimpleNamespace(device='cpu', generate=generate), processor,
                                   ['a.png', 'b.png', 'c.png'], ['t1', 't2', 't3'], 9, 2)
        self.assertEqual(replies, ['[0, 1]', '[2, 3]', '[0, 1]'])
        self.assertEqual([len(m) for m, _, _ in seen], [2, 1])
        self.assertEqual(seen[0][0][1][0]['content'], [{'type': 'image', 'image': 'b.png'},
                                                       {'type': 'text', 'text': 't2'}])
        self.assertTrue(all(kw['padding'] and side == 'left' for _, kw, side in seen))
        self.assertEqual(tokenizer.padding_side, 'right')

    def test_client_generate_texts_over_a_real_pipe(self):
        real_popen = subprocess.Popen
        code = '''import sys,json
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
 r=json.loads(line)
 assert r["op"]=="generate" and r["max_new_tokens"]==7 and r["batch_size"]==3
 assert open(r["images"][0],"rb").read(8)==b"\\x89PNG\\r\\n\\x1a\\n"
 n=len(r["texts"]) if r["texts"][0]!="short" else 1
 print(json.dumps({"texts": [t.upper() for t in r["texts"]][:n]}),flush=True)
'''
        with patch('llava.train.rl.pixel_reward.subprocess.Popen',
                   side_effect=lambda *a, **k: real_popen([sys.executable, '-u', '-c', code], **k)):
            judge = QwenPixelReward(sys.executable, 'fake', 'cpu', timeout=30)
        try:
            images = [Image.new('RGB', (2, 2)), Image.new('RGB', (2, 2))]
            self.assertEqual(judge.generate_texts(images, ['ab', 'cd'], 7, 3), ['AB', 'CD'])
            self.assertEqual(judge.generate_texts([], [], 7, 3), [])
            with self.assertRaises(ValueError):
                judge.generate_texts(images, ['short', 'x'], 7, 3)
            with self.assertRaises(ValueError):
                judge.generate_texts(images, ['only one'], 7, 3)
        finally:
            judge.close()

    def test_cli_requirements(self):
        required = ['train_grpo', '--model_name_or_path', 'unused', '--data_path', 'unused',
                    '--output_dir', 'unused', '--objective', 'teacher_tree', '--num_rollouts', '8']
        with patch('sys.argv', required + ['--tree_branch', '1x2']):
            args = trainer.parse_args()
            self.assertEqual((args.tree_branch, args.shadow_reflections, args.teacher_ce_coef, args.shadow_coef,
                              args.teacher_ce_filter), ([[1, 2]], 1, .1, 1.0, 'all'))
        with patch('sys.argv', required + ['--teacher_ce_filter', 'agree']):
            self.assertEqual(trainer.parse_args().teacher_ce_filter, 'agree')
        for extra in (['--tree_branch', '2'], ['--verdict_target', 'soft_am'], ['--max_refinements', '0'],
                      ['--text_policy_gradient', 'transitions'], ['--shadow_reflections', '-1'],
                      ['--teacher_ce_coef', 'nan'], ['--length_normalization', 'constant'],
                      ['--teacher_ce_filter', 'some']):
            with self.subTest(extra=extra), patch('sys.argv', required + extra), \
                    patch('sys.stderr'), self.assertRaises(SystemExit):
                trainer.parse_args()

    def test_method_metadata_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            args = policy_args(output_dir=directory, save_total_limit=3, max_seq_len=64, objective='teacher_tree',
                               tree_branch=[[1, 2]], reset_context_per_turn=True, verdict_rule='correction',
                               verdict_target='all_pass', verdict_pass_threshold=.5, shadow_reflections=1,
                               shadow_coef=1.0, teacher_ce_coef=.1, teacher_max_tokens=96, teacher_ce_filter='agree',
                               reward_model_name_or_path='/x/Qwen3-VL-8B-Instruct/')
            method = trainer.rl_method_settings(args)
            self.assertEqual({k: method[k] for k in ('objective', 'teacher_model', 'teacher_prompt_version',
                                                     'shadow_reflections', 'verdict_target')},
                             dict(objective='teacher_tree', teacher_model='Qwen3-VL-8B-Instruct',
                                  teacher_prompt_version='teacher_reflect_v1', shadow_reflections=1,
                                  verdict_target='all_pass'))
            model = get_peft_model(tiny_base(), LoraConfig(r=2, target_modules=['q_proj'], task_type='CAUSAL_LM'))
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            trainer.save_checkpoint(model, opt, scheduler, 1, args, torch.device('cpu'))
            ck = Path(directory) / 'checkpoint-1'
            self.assertEqual(trainer.validate_resume_checkpoint(ck, args)['rl_method'], method)
            for key, value in (('teacher_ce_coef', .2), ('shadow_reflections', 2), ('objective', 'final_episode'),
                               ('tree_branch', [[1, 3]]), ('teacher_ce_filter', 'all')):
                changed = copy.copy(args)
                setattr(changed, key, value)
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                    trainer.validate_resume_checkpoint(ck, changed)
            # A teacher_tree checkpoint written before the filter existed distilled everything: 'all' only.
            legacy = dict(method)
            legacy.pop('teacher_ce_filter')
            (ck / 'state.json').write_text(json.dumps({'step': 1, 'objective': trainer.RL_OBJECTIVE,
                                                       'rl_method': legacy}))
            with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                trainer.validate_resume_checkpoint(ck, args)
            legacy_args = SimpleNamespace(**dict(vars(args), teacher_ce_filter='all'))
            self.assertEqual(trainer.validate_resume_checkpoint(ck, legacy_args)['step'], 1)
            # A Janus checkpoint written before the teacher keys still resumes as Janus, never as teacher_tree.
            janus = dict(method, objective='janus_one_iteration', tree_branch=[], reset_context_per_turn=False)
            for key in ('shadow_reflections', 'shadow_coef', 'teacher_ce_coef', 'teacher_max_tokens',
                        'teacher_prompt_version', 'teacher_model', 'teacher_ce_filter'):
                janus.pop(key)
            (ck / 'state.json').write_text(json.dumps({'step': 1, 'objective': trainer.RL_OBJECTIVE,
                                                       'rl_method': janus}))
            janus_args = policy_args(max_seq_len=64, objective='janus_one_iteration', verdict_rule='correction',
                                     verdict_target='all_pass', verdict_pass_threshold=.5)
            self.assertEqual(trainer.validate_resume_checkpoint(ck, janus_args)['step'], 1)
            with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                trainer.validate_resume_checkpoint(ck, args)


if __name__ == '__main__':
    unittest.main()
