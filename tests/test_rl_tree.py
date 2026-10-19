"""Adversarial checks for tree rollouts with final-reward credit (--tree_branch)."""
import copy
import itertools
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_NONE, KIND_TXT, grpo_loss
from llava.train.rl.rollout import (RolloutBatch, TreeNode, Trajectory, assign_tree_masses,
                                    parse_tree_branch, validate_tree_branch)
from test_rl_episodes import ScriptedModel, episode_rollout, PROMPTS
from test_rl_policy import small_rollout, policy_args, tiny_base, get_peft_model, LoraConfig
from test_rl_reset_context import new_model, update

EPISODE_CALLS = [("image", [[8]*3, [9]*3]), ("text", [[2], [16, 3]]), ("image", [[10]*3]),
                 ("text", [[7, 3]]), ("image", [[11]*3]), ("text", [[2]])]
# Two drafts, two reflections each (R=2 after the draft), unbranched afterwards.
R2_CALLS = [("image", [[8]*3, [9]*3]), ("text", [[2], [3], [3], [2]]),
            ("image", [[10]*3, [11]*3]), ("text", [[2], [2]])]


def tree(calls, branch, **overrides):
    rollout = episode_rollout(**overrides)
    model = ScriptedModel(calls)
    batch = rollout.run_tree(model, PROMPTS, branch)
    assert model.calls == [], model.calls
    return rollout, batch


def set_rewards(batch, rewards):
    for t, r in zip(batch.trajectories, rewards):
        t.reward = t.am = r


def row_map(rows):
    """Row token tuple -> (kinds, adv, weight[, pg_mask]) for order-free comparison."""
    out = {}
    for i in range(len(rows['input_ids'])):
        n = int(rows['attention_mask'][i].sum())
        key = tuple(rows['input_ids'][i, :n].tolist())
        assert key not in out
        out[key] = [rows[k][i, :n] for k in ('pos_kind', 'adv', 'weight', 'pg_mask') if k in rows]
    return out


class TreeRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_spec_parsing_and_validation(self):
        self.assertEqual(parse_tree_branch(""), [])
        self.assertEqual(parse_tree_branch("1x1,1"), [])
        self.assertEqual(parse_tree_branch(" 2 , 2x2 ,1"), [[2, 1], [2, 2]])
        self.assertEqual(parse_tree_branch([[2, 1], [1, 1]]), [[2, 1]])
        self.assertEqual(parse_tree_branch(json.loads(json.dumps([[4, 1]]))), [[4, 1]])
        for bad in ("0", "2x", "a", "2,,2", "2x0", "-1", [[2]], [[2, True]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_tree_branch(bad)
        validate_tree_branch([[2, 1]] * 3, 3, 4, 32)
        with self.assertRaisesRegex(ValueError, 'max_refinements'):
            validate_tree_branch([[2, 1]] * 4, 3, 4, 64)
        with self.assertRaisesRegex(ValueError, 'leaves'):
            validate_tree_branch([[2, 2], [2, 1]], 3, 4, 31)

    def test_unbranched_tree_equals_independent_episodes(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                legacy_rollout = episode_rollout(reset_context_per_turn=reset)
                legacy = legacy_rollout.run(ScriptedModel(EPISODE_CALLS), PROMPTS)
                rollout, batch = tree(EPISODE_CALLS, "1x1", reset_context_per_turn=reset)
                self.assertEqual(len(batch.trajectories), len(legacy.trajectories))
                for t, old in zip(batch.trajectories, legacy.trajectories):
                    for key in ('seq', 'kinds', 'images', 'reflections', 'stop_reason', 'prompt_idx'):
                        self.assertEqual(getattr(t, key), getattr(old, key), key)
                    self.assertEqual([(s.seq, s.kinds) for s in t.segments],
                                     [(s.seq, s.kinds) for s in old.segments])
                    self.assertEqual(t.mass, 1.0)
                for rewards in ([.1, .9], [.4, .4]):
                    for norm in ('mean', 'std'):
                        set_rewards(legacy, rewards)
                        set_rewards(batch, rewards)
                        cfg = GRPOConfig(adv_norm=norm)
                        self.assertEqual(trainer.assign_tree_advantages(batch, cfg),
                                         trainer.assign_advantages(legacy, cfg))
                        for t, old in zip(batch.trajectories, legacy.trajectories):
                            self.assertAlmostEqual(t.adv, old.adv, places=12)
                set_rewards(legacy, [.1, .9])
                set_rewards(batch, [.1, .9])
                trainer.assign_advantages(legacy, GRPOConfig())
                trainer.assign_tree_advantages(batch, GRPOConfig())
                for mode in ('episode', 'constant'):
                    for text_pg in ('all', 'transitions'):
                        expected = row_map(legacy_rollout.build_training_rows(
                            legacy.trajectories, .7, mode, text_pg))
                        actual = row_map(rollout.build_tree_training_rows(batch, .7, mode, text_pg))
                        self.assertEqual(set(actual), set(expected))
                        for key in expected:
                            for a, b in zip(actual[key], expected[key]):
                                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
                    models = [new_model(), new_model()]
                    metrics = [update(models[0], legacy_rollout, legacy, 1, mode),
                               update(models[1], rollout, batch, 1, mode)]
                    for p, q in zip(*(m.parameters() for m in models)):
                        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
                    for key in ('loss', 'pg_loss', 'kl', 'clip_frac', 'approx_kl_old'):
                        self.assertAlmostEqual(metrics[0][key], metrics[1][key], places=6)
                stats = trainer.rollout_stats(batch, "x/")
                self.assertEqual(stats, trainer.rollout_stats(legacy, "x/"))

    def test_reflection_branching_structure_contexts_and_rows(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                rollout, batch = tree(R2_CALLS, "2", reset_context_per_turn=reset)
                t0, c0, t1, c1 = batch.trajectories
                self.assertEqual(t0.seq, [1, 3, 5, 8, 8, 8, 4, 7, 2])
                self.assertEqual(c0.seq, [1, 3, 5, 8, 8, 8, 4, 7, 3, 5, 10, 10, 10, 4, 7, 2])
                self.assertEqual(t1.seq, [1, 3, 5, 9, 9, 9, 4, 7, 3, 5, 11, 11, 11, 4, 7, 2])
                self.assertEqual(c1.seq, [1, 3, 5, 9, 9, 9, 4, 7, 2])
                self.assertEqual([t.stop_reason for t in batch.trajectories], ['eos'] * 4)
                self.assertEqual([t.images for t in batch.trajectories],
                                 [[[0]*3], [[0]*3, [2]*3], [[1]*3, [3]*3], [[1]*3]])
                # No list aliasing between clones.
                for a, b in itertools.combinations(batch.trajectories, 2):
                    self.assertIsNot(a.seq, b.seq)
                    self.assertIsNot(a.kinds, b.kinds)
                    self.assertIsNot(a.images, b.images)
                    for s, r in zip(a.segments, b.segments):
                        self.assertIsNot(s.seq, r.seq)
                self.assertEqual(len(batch.nodes), 10)
                self.assertEqual([t.mass for t in batch.trajectories], [.5] * 4)
                self.assertIs(t0.nodes[0], c0.nodes[0])
                self.assertIsNot(t0.nodes[1], c0.nodes[1])
                self.assertEqual(t0.nodes[1].context, [1, 3, 5, 8, 8, 8, 4, 7])
                self.assertEqual(c0.nodes[2].context, [1, 3, 5, 8, 8, 8, 4, 7, 3, 5])
                self.assertEqual(c0.nodes[3].context, [1, 3, 5, 10, 10, 10, 4, 7] if reset
                                 else [1, 3, 5, 8, 8, 8, 4, 7, 3, 5, 10, 10, 10, 4, 7])
                self.assertEqual(c0.nodes[1].actions, [3])
                self.assertEqual(c0.nodes[1].action_kinds, [KIND_TXT])
                self.assertIs(t0.final_image_node, t0.nodes[0])
                self.assertIs(c0.final_image_node, c0.nodes[2])
                # Logical episodes keep exact sampled actions; segments match contexts.
                if reset:
                    self.assertEqual(c0.segments[-1].seq, [1, 3, 5, 10, 10, 10, 4, 7, 2])
                set_rewards(batch, [.2, .6, .9, .5])
                trainer.assign_tree_advantages(batch, GRPOConfig())
                d0, d1 = t0.nodes[0], t1.nodes[0]
                self.assertAlmostEqual(d0.value, .4)
                self.assertAlmostEqual(d1.value, .7)
                expected = {d0: -.15, d1: .15, t0.nodes[1]: -.2, c0.nodes[1]: .2, c0.nodes[2]: .2,
                            c0.nodes[3]: .2, t1.nodes[1]: .2, t1.nodes[2]: .2, t1.nodes[3]: .2,
                            c1.nodes[1]: -.2}
                for node, adv in expected.items():
                    self.assertAlmostEqual(node.adv, adv, places=12)
                rows = rollout.build_tree_training_rows(batch, .7, 'episode')
                self.assertEqual(len(rows['input_ids']), 8 if reset else 6)
                # Every sampled action is trained exactly once, with its node's advantage.
                self.assertEqual(int(rows['train_mask'].sum()), sum(len(n.actions) for n in batch.nodes))
                self.assertTrue(torch.all(rows['weight'][~rows['train_mask']] == 0))
                self.assertAlmostEqual(float(rows['weight'].sum()), 2.0, places=5)  # one per draft
                ids, mask = rows['input_ids'], rows['train_mask']
                for token, adv in [(8, -.15), (9, .15), (10, .2), (11, .2)]:
                    sel = mask & (ids == token)
                    self.assertEqual(int(sel.sum()), 3)
                    self.assertTrue(torch.allclose(rows['adv'][sel], torch.tensor(adv)))
                constant = rollout.build_tree_training_rows(batch, .7, 'constant')
                total = sum(n.mass * sum(.7 if k == KIND_TXT else 1 for k in n.action_kinds)
                            for n in batch.nodes)
                self.assertAlmostEqual(float(constant['weight'].sum()), total / 100, places=6)
                torch.testing.assert_close(rows['adv'], constant['adv'], rtol=0, atol=0)
                # std: divide by the leaf-reward std (0.25 here).
                trainer.assign_tree_advantages(batch, GRPOConfig(adv_norm='std'))
                self.assertAlmostEqual(d0.adv, -.15 / (.25 + 1e-4), places=9)
                stats = trainer.rollout_stats(batch, "")
                self.assertAlmostEqual(stats['n_final'], 2.0)
                self.assertAlmostEqual(stats['mean_refinements'], 1.0)
                tree_stats = trainer.finalize_tree_stats(trainer.tree_stats(batch, 2, 3, ""), "")
                self.assertEqual(tree_stats['tree_leaves_per_prompt'], 4)
                self.assertAlmostEqual(tree_stats['tree_mean_adv_stop'], -.2 / (.25 + 1e-4))
                self.assertAlmostEqual(tree_stats['tree_mean_adv_continue'], .2 / (.25 + 1e-4))

    def test_image_branching_masses_do_not_depend_on_descendant_count(self):
        calls = [("image", [[8]*3, [9]*3]), ("text", [[3], [2], [3], [3]]),
                 ("image", [[10]*3, [11]*3, [12]*3, [13]*3, [14]*3, [15]*3]),
                 ("text", [[2]] * 6)]
        rollout, batch = tree(calls, "2x2")
        self.assertEqual(len(batch.trajectories), 7)
        self.assertEqual([t.mass for t in batch.trajectories], [.25, .25, .5, .25, .25, .25, .25])
        reflections = [n for n in batch.nodes if n.kind == 'reflection' and n.parent.parent is None]
        self.assertEqual([n.mass for n in reflections], [.5] * 4)  # stop and refine alike
        self.assertEqual([len(n.children) for n in reflections], [2, 0, 2, 2])
        for draft in (n for n in batch.nodes if n.parent is None):
            leaves = [t for t in batch.trajectories if t.nodes[0] is draft]
            self.assertAlmostEqual(sum(t.mass for t in leaves), 1.0)
        set_rewards(batch, [.9, .1, .3, .4, .6, .2, .8])
        trainer.assign_tree_advantages(batch, GRPOConfig())
        constant = rollout.build_tree_training_rows(batch, 1, 'constant')
        ids = constant['input_ids']
        refine, stop = reflections[0], reflections[1]
        self.assertAlmostEqual(refine.value, .5)
        self.assertAlmostEqual(stop.value, .3)
        # A reflection's token weight is its mass, not the number of leaves under it.
        for node in (refine, stop):
            row = next(i for i in range(len(ids)) if ids[i, :len(node.context)+1].tolist()
                       == node.context + node.actions)
            self.assertAlmostEqual(float(constant['weight'][row, len(node.context)]), .5 / 100)
        self.assertAlmostEqual(float(rollout.build_tree_training_rows(batch, 1)['weight'].sum()), 2, places=5)

    def test_room_and_cap_truncations_keep_one_leaf_per_terminal_node(self):
        # No room for another reflection: the branch request after image 1 is ignored.
        rollout, batch = tree([("image", [[8]*3, [9]*3]), ("text", [[3], [3]]),
                               ("image", [[10]*3, [11]*3])], "1,2", max_seq_len=15)
        self.assertEqual([t.stop_reason for t in batch.trajectories], ['max_seq_len'] * 2)
        for t in batch.trajectories:
            self.assertIs(t.nodes[-1], t.final_image_node)
            self.assertEqual([n.kind for n in t.nodes], ['image', 'reflection', 'image'])
        # A capped <im_start> leaves its reflection as the terminal node; no image clones.
        rollout, batch = tree([("image", [[8]*3, [9]*3]), ("text", [[3], [3], [3], [3]])],
                              "2x2", max_refinements=0)
        self.assertEqual([t.stop_reason for t in batch.trajectories], ['max_refinements'] * 4)
        self.assertEqual([n.kind for n in batch.trajectories[0].nodes], ['image', 'reflection'])
        self.assertEqual(len(batch.nodes), 6)
        # Reflection budget truncation.
        rollout, batch = tree([("image", [[8]*3, [9]*3]), ("text", [[7]*4]*4)], "2")
        self.assertEqual([t.stop_reason for t in batch.trajectories], ['reflection_limit'] * 4)
        set_rewards(batch, [.1, .2, .3, .4])
        trainer.assign_tree_advantages(batch, GRPOConfig())
        rows = rollout.build_tree_training_rows(batch, 1, 'episode', 'transitions')
        prose = rows['pos_kind'] == KIND_TXT  # every sampled text token is prose here
        self.assertTrue(prose.any())
        self.assertTrue(torch.all(rows['pg_mask'][prose] == 0))
        self.assertTrue(torch.all(rows['pg_mask'][~prose] == 1))

    def test_scoring_renders_and_scores_each_final_node_once(self):
        calls = [("image", [[8]*3, [9]*3]), ("text", [[2], [2], [3], [2]]),
                 ("image", [[10]*3]), ("text", [[2]])]
        rollout, batch = tree(calls, "2")
        decoded, scored = [], []
        def decode(codes):
            decoded.append(codes)
            return [object()]
        def score(images, questions):
            scored.append(images)
            return [.1, .7, .3], [.0, .0, .0], [[.1], [.7], [.3]]
        trainer.score_rollouts(batch, SimpleNamespace(score_images=score),
                               SimpleNamespace(decode_codes=decode))
        self.assertEqual(decoded, [[[0]*3], [[2]*3], [[1]*3]])  # leaf order, one image per call
        t0, c0, t1, c1 = batch.trajectories
        self.assertIs(t0.final_image, c0.final_image)  # both stopped on draft 0
        self.assertIsNot(t1.final_image, c1.final_image)
        self.assertEqual([t.reward for t in batch.trajectories], [.1, .1, .7, .3])
        self.assertIsNot(t0.question_scores, c0.question_scores)
        self.assertEqual(scored[0], [t0.final_image, t1.final_image, c1.final_image])

    def test_tied_drafts_with_differing_subtrees_keep_credit(self):
        rollout, batch = tree(R2_CALLS, "2")
        set_rewards(batch, [.2, .6, .5, .3])
        zero_var, groups = trainer.assign_tree_advantages(batch, GRPOConfig(adv_norm='std'))
        self.assertEqual((zero_var, groups), (0, 1))
        self.assertEqual(batch.trajectories[0].nodes[0].adv, 0.0)
        self.assertGreater(batch.trajectories[1].nodes[1].adv, 0)
        set_rewards(batch, [.5] * 4)
        self.assertEqual(trainer.assign_tree_advantages(batch, GRPOConfig(adv_norm='std')), (1, 1))
        self.assertTrue(all(n.adv == 0 for n in batch.nodes))

    def test_constant_mode_tree_gradient_is_unbiased(self):
        """Exact expectation over every outcome of a 2 drafts x 2 reflections x 2 images tree.

        Self-inclusive sibling means scale each level by (K-1)/K = 1/2, so the
        expected update must be exactly half the true gradient / max_seq_len.
        """
        rollout = small_rollout()
        rollout.cfg.max_seq_len = 32
        theta = torch.tensor([.3, -.4, .6], dtype=torch.float64, requires_grad=True)
        lp_d = (theta[0] * torch.tensor([1., -1.], dtype=torch.float64)).log_softmax(-1)
        lp_r = (theta[1] * torch.tensor([1., -1.], dtype=torch.float64)).log_softmax(-1)
        lp_i = (theta[2] * torch.tensor([1., -1.], dtype=torch.float64)).log_softmax(-1)
        r_stop = {8: .3, 9: .7}
        r_ref = {(8, 10): .9, (8, 11): .1, (9, 10): .4, (9, 11): .6}
        p_d, p_r, p_i = lp_d.exp(), lp_r.exp(), lp_i.exp()
        J = sum(p_d[a] * (p_r[0] * r_stop[d] + p_r[1] * sum(p_i[b] * r_ref[d, i]
                for b, i in enumerate((10, 11)))) for a, d in enumerate((8, 9)))
        target = torch.autograd.grad(J, theta, retain_graph=True)[0]
        lookup = {8: lp_d[0], 9: lp_d[1], 2: lp_r[0], 3: lp_r[1], 10: lp_i[0], 11: lp_i[1]}
        reflection_outcomes = [None] + list(itertools.product((10, 11), repeat=2))
        objective = theta.sum() * 0
        for drafts in itertools.product((8, 9), repeat=2):
            for outcomes in itertools.product(reflection_outcomes, repeat=4):
                nodes, leaves = [], []
                def node(kind, parent, context, action):
                    n = TreeNode(len(nodes), 0, kind, parent, context, [action],
                                 [KIND_IMG if kind == 'image' else KIND_TXT])
                    if parent is not None:
                        parent.children.append(n)
                    nodes.append(n)
                    return n
                def leaf(path, reward):
                    t = Trajectory(0, [], [], reward=reward)
                    t.nodes = path
                    leaves.append(t)
                for j, d in enumerate(drafts):
                    dn = node('image', None, [1], d)
                    for outcome in outcomes[2*j:2*j+2]:
                        rn = node('reflection', dn, [1, d, 4], 2 if outcome is None else 3)
                        if outcome is None:
                            leaf([dn, rn], r_stop[d])
                        for i in outcome or ():
                            leaf([dn, rn, node('image', rn, [1, d, 4, 3, 5], i)], r_ref[d, i])
                assign_tree_masses(nodes, leaves)
                batch = RolloutBatch(PROMPTS, leaves, nodes)
                trainer.assign_tree_advantages(batch, GRPOConfig())
                rows = rollout.build_tree_training_rows(batch, 1, 'constant')
                sel = rows['train_mask']
                token_lp = torch.stack([lookup[t] for t in rows['input_ids'][sel].tolist()])
                loss, _ = grpo_loss(token_lp, token_lp.detach(), token_lp.detach(), rows['adv'][sel],
                                    rows['weight'][sel], rows['pos_kind'][sel], 2,
                                    GRPOConfig(kl_coef=0, length_normalization='constant'))
                prob = torch.stack([lookup[n.actions[0]] for n in nodes]).sum().exp().detach()
                objective = objective + prob * loss
        ascent = -torch.autograd.grad(objective, theta)[0]
        torch.testing.assert_close(ascent, target * .5 / 32, rtol=1e-5, atol=1e-9)

    def test_cli_and_checkpoint_metadata(self):
        required = ['train', '--model_name_or_path', 'unused', '--data_path', 'unused',
                    '--output_dir', 'unused']
        for extra, expected in [([], ([], 'all')), (['--tree_branch', '2,2x2'], ([[2, 1], [2, 2]], 'all')),
                                (['--tree_branch', '1x1', '--text_policy_gradient', 'transitions'],
                                 ([], 'transitions'))]:
            with self.subTest(extra=extra), patch('sys.argv', required + extra):
                args = trainer.parse_args()
                self.assertEqual((args.tree_branch, args.text_policy_gradient), expected)
        for extra in (['--tree_branch', '0'], ['--tree_branch', '2,2,2,2'],
                      ['--tree_branch', '4x4', '--max_tree_leaves_per_prompt', '32'],
                      ['--text_policy_gradient', 'none']):
            with self.subTest(extra=extra), patch('sys.argv', required + extra), \
                    patch('sys.stderr'), self.assertRaises(SystemExit):
                trainer.parse_args()
        with tempfile.TemporaryDirectory() as directory:
            args = policy_args(output_dir=directory, save_total_limit=3, max_seq_len=32,
                               length_normalization='episode', tree_branch=[[2, 1]],
                               text_policy_gradient='transitions')
            model = get_peft_model(tiny_base(), LoraConfig(r=2, target_modules=['q_proj'],
                                                          task_type='CAUSAL_LM'))
            opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
            trainer.save_checkpoint(model, opt, scheduler, 1, args, torch.device('cpu'))
            ck = Path(directory) / 'checkpoint-1'
            saved = json.loads((ck / 'state.json').read_text())['rl_method']
            self.assertEqual((saved['tree_branch'], saved['text_policy_gradient']),
                             ([[2, 1]], 'transitions'))
            trainer.validate_resume_checkpoint(ck, args)
            for key, value in [('tree_branch', []), ('tree_branch', [[2, 2]]),
                               ('text_policy_gradient', 'all')]:
                changed = copy.copy(args)
                setattr(changed, key, value)
                with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                    trainer.validate_resume_checkpoint(ck, changed)
            # Checkpoints without the new keys mean independent episodes and all-text PG.
            state = json.loads((ck / 'state.json').read_text())
            for key in ('tree_branch', 'text_policy_gradient'):
                del state['rl_method'][key]
            (ck / 'state.json').write_text(json.dumps(state))
            legacy = policy_args(max_seq_len=32, length_normalization='episode')
            trainer.validate_resume_checkpoint(ck, legacy)
            with self.assertRaisesRegex(ValueError, 'RL method settings differ'):
                trainer.validate_resume_checkpoint(ck, args)

    def test_tree_updates_are_microbatch_invariant(self):
        for reset, mode in itertools.product((False, True), ('episode', 'constant')):
            with self.subTest(reset=reset, mode=mode):
                models, metrics = [], []
                for micro in (1, 3, 16):
                    rollout, batch = tree(R2_CALLS, "2", reset_context_per_turn=reset)
                    set_rewards(batch, [.2, .6, .9, .5])
                    trainer.assign_tree_advantages(batch, GRPOConfig())
                    model = new_model()
                    metrics.append(update(model, rollout, batch, micro, mode))
                    models.append(model)
                for model in models[1:]:
                    for p, q in zip(models[0].parameters(), model.parameters()):
                        torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
                for m in metrics:
                    self.assertAlmostEqual(m['loss'], m['pg_loss'] + .01 * m['kl'], places=6)

    def test_actual_sampling_probabilities_match_tree_training_rows(self):
        from llava.train.rl.grpo import masked_token_logprobs
        for reset in (False, True):
            with self.subTest(reset=reset):
                model = new_model().eval()
                rollout = episode_rollout(reset_context_per_turn=reset, reflect_tokens=32)
                generate = model.generate
                captured = {}
                def capture(**kwargs):
                    result = generate(**kwargs, return_dict_in_generate=True, output_scores=True)
                    n = kwargs['input_ids'].shape[1]
                    for row in range(len(result.sequences)):
                        prefix = kwargs['input_ids'][row][kwargs['attention_mask'][row].bool()].tolist()
                        for j, token in enumerate(result.sequences[row, n:].tolist()):
                            prefix.append(token)
                            captured[tuple(prefix)] = result.scores[j][row].float().log_softmax(-1)[token]
                            if kwargs['eos_token_id'] is not None and token in kwargs['eos_token_id']:
                                break
                    return result.sequences
                for seed in range(20):
                    torch.manual_seed(seed)
                    captured.clear()
                    with patch.object(model, 'generate', side_effect=capture):
                        batch = rollout.run_tree(model, PROMPTS, "2x2,2")
                    if max(len(t.images) for t in batch.trajectories) >= 3:
                        break
                self.assertGreaterEqual(max(len(t.images) for t in batch.trajectories), 3)
                self.assertGreater(len(batch.trajectories), 4)
                rows = rollout.build_tree_training_rows(batch, 1)
                expected = []
                for ids, mask in zip(rows['input_ids'], rows['train_mask']):
                    for i in mask.nonzero().flatten().tolist():
                        expected.append(captured[tuple(ids[:i+1].tolist())])
                actual = masked_token_logprobs(model, rows['input_ids'], rows['attention_mask'],
                                               rows['train_mask'], rows['pos_kind'], 8, 16)
                torch.testing.assert_close(actual, torch.stack(expected), rtol=1e-5, atol=1e-6)

    def test_distributed_tree_update_matches_combined_batch(self):
        with tempfile.TemporaryDirectory(prefix='tar-tree-dist-') as directory:
            mp.spawn(distributed_tree_worker, args=(directory,), nprocs=2, join=True)


RANK_CALLS = [R2_CALLS, [("image", [[9]*3, [8]*3]), ("text", [[3], [3], [2], [3]]),
                         ("image", [[12]*3, [13]*3, [14]*3]), ("text", [[2], [3], [2]]),
                         ("image", [[15]*3]), ("text", [[2]])]]


def rank_batch(rank):
    rollout, batch = tree(RANK_CALLS[rank], "2", reset_context_per_turn=True)
    set_rewards(batch, [.2, .6, .9, .5] if rank == 0 else [.3, .8, .1, .4])
    trainer.assign_tree_advantages(batch, GRPOConfig())
    return rollout, batch


def distributed_tree_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + directory + '/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=90))
    try:
        for mode in ('episode', 'constant'):
            model = new_model()
            central = copy.deepcopy(model)
            rollout, batch = rank_batch(rank)
            metrics = update(model, rollout, batch, 1, mode)
            reduced = {}
            for key in ('loss', 'pg_loss', 'kl'):
                value = torch.tensor(metrics[key], dtype=torch.float64)
                dist.all_reduce(value)
                reduced[key] = float(value) / 2
            if rank == 0:
                other = rank_batch(1)[1]
                combined = RolloutBatch(PROMPTS, batch.trajectories + other.trajectories,
                                        batch.nodes + other.nodes)
                with patch.object(trainer.dist, 'is_initialized', return_value=False):
                    expected = update(central, rollout, combined, 2, mode)
                for p, q in zip(model.parameters(), central.parameters()):
                    torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
                for key in reduced:
                    torch.testing.assert_close(torch.tensor(reduced[key]), torch.tensor(expected[key]),
                                               rtol=1e-5, atol=1e-7)
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
