"""Adversarial checks for EOS episodes and final-image GRPO credit."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from llava.train.rl import train_grpo as trainer
from llava.train.rl.grpo import GRPOConfig, KIND_IMG, KIND_NONE, KIND_TXT, grpo_loss, masked_token_logprobs
from llava.train.rl.rollout import ImageVocabOnly, RolloutBatch, Trajectory
from test_rl_policy import small_rollout, tiny_base, policy_args, training_batch, get_peft_model, LoraConfig


class ScriptedModel(torch.nn.Module):
    """Script generation only; exercise the real episode loop and padding."""

    def __init__(self, calls):
        super().__init__()
        self.calls = list(calls)
        self.config = SimpleNamespace(max_position_embeddings=1000)

    def generate(self, **kwargs):
        phase, rows = self.calls.pop(0)
        actual_phase = "image" if isinstance(kwargs["logits_processor"][0], ImageVocabOnly) else "text"
        assert phase == actual_phase, (phase, actual_phase)
        assert len(rows) == len(kwargs["input_ids"])
        assert all(len(row) <= kwargs["max_new_tokens"] for row in rows)
        output = torch.full((len(rows), max(map(len, rows))), kwargs["pad_token_id"], dtype=torch.long)
        for i, row in enumerate(rows):
            output[i, :len(row)] = torch.tensor(row)
        return torch.cat((kwargs["input_ids"], output), dim=1)


def episode_rollout(**overrides):
    rollout = small_rollout()
    rollout.cfg.num_rollouts = 2
    rollout.cfg.max_seq_len = 100
    rollout.reflect_splice = [4, 7]
    rollout.image_splice = [5]
    rollout.make_prefix = lambda prompt: [1, 3, 5]
    rollout.tok = SimpleNamespace(decode=lambda ids, **kwargs: "looks good" if 16 in ids else "correct it")
    for key, value in overrides.items():
        setattr(rollout.cfg, key, value)
    return rollout


PROMPTS = [{"prompt": "two red dogs", "vqa_list": [("How many dogs?", "two")]}]


class EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_episodes_continue_until_eos_not_looks_good(self):
        rollout = episode_rollout()
        model = ScriptedModel([
            ("image", [[8] * 3, [9] * 3]),
            ("text", [[2], [16, 3]]),  # "looks good" without EOS must continue
            ("image", [[10] * 3]),
            ("text", [[7, 3]]),
            ("image", [[11] * 3]),
            ("text", [[2]]),
        ])
        batch = rollout.run(model, PROMPTS)
        a, b = batch.trajectories
        self.assertEqual([len(t.images) for t in batch.trajectories], [1, 3])
        self.assertEqual([t.stop_reason for t in batch.trajectories], ["eos", "eos"])
        self.assertEqual(a.images, [[0] * 3])
        self.assertEqual(b.images, [[1] * 3, [2] * 3, [3] * 3])
        self.assertIsNot(a.seq, b.seq)
        self.assertEqual(b.seq, [1, 3, 5, 9, 9, 9, 4, 7, 16, 3, 5,
                                 10, 10, 10, 4, 7, 7, 3, 5, 11, 11, 11, 4, 7, 2])
        self.assertEqual(b.kinds[9], KIND_TXT)  # sampled image-transition action
        self.assertEqual(b.kinds[10], KIND_NONE)  # forced scale
        self.assertEqual(b.kinds[-1], KIND_TXT)  # sampled terminal EOS
        self.assertEqual(model.calls, [])
        self.assertTrue(model.training)  # original mode restored

    def test_limits_do_not_invent_eos_or_generate_partial_images(self):
        cases = [
            ({"max_refinements": 0}, [[2], [3]], ["eos", "max_refinements"]),
            ({}, [[7] * 4, [7] * 4], ["reflection_limit"] * 2),
            ({"max_seq_len": 10}, [[7, 3], [7, 7]], ["max_seq_len"] * 2),
        ]
        for options, reflections, reasons in cases:
            with self.subTest(options=options):
                rollout = episode_rollout(**options)
                model = ScriptedModel([("image", [[8] * 3, [9] * 3]), ("text", reflections)])
                batch = rollout.run(model, PROMPTS)
                self.assertEqual([t.stop_reason for t in batch.trajectories], reasons)
                for trajectory, tokens in zip(batch.trajectories, reflections):
                    self.assertEqual(len(trajectory.images), 1)
                    self.assertEqual(trajectory.seq[-len(tokens):], tokens)
                    self.assertLessEqual(len(trajectory.seq), rollout.cfg.max_seq_len)
                self.assertEqual(model.calls, [])

    def test_context_budget_checked_before_any_sampling(self):
        rollout = episode_rollout(max_seq_len=8)
        model = ScriptedModel([])
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            rollout.run(model, PROMPTS)
        self.assertTrue(model.training)

    def test_architectural_limit_and_generation_failure_restore_mode(self):
        rollout = episode_rollout()
        model = ScriptedModel([])
        model.config.max_position_embeddings = 8
        model.eval()
        with self.assertRaises(ValueError):
            rollout.run(model, PROMPTS)
        self.assertFalse(model.training)

    def test_eos_equal_to_padding_is_still_a_trainable_action(self):
        rollout = episode_rollout()
        rollout.eos_id = rollout.pad_id
        rollout.text_stop_ids = [rollout.eos_id, rollout.im_start_tok]
        model = ScriptedModel([("image", [[8] * 3, [9] * 3]), ("text", [[0], [7, 0]])])
        batch = rollout.run(model, PROMPTS)
        rows = rollout.build_training_rows(batch.trajectories, 1.0)
        for i, trajectory in enumerate(batch.trajectories):
            end = len(trajectory.seq) - 1
            self.assertEqual(trajectory.stop_reason, "eos")
            self.assertTrue(rows["train_mask"][i, end])
            self.assertEqual(rows["attention_mask"][i, end], 1)
            self.assertGreater(rows["weight"][i, end], 0)
        self.assertFalse(rows["train_mask"][0, -1])  # actual batch padding

    def test_final_reward_reverses_draft_preference_and_credits_entire_episode(self):
        rollout = episode_rollout()
        model = ScriptedModel([
            ("image", [[8] * 3, [9] * 3]), ("text", [[3], [3]]),
            ("image", [[10] * 3, [11] * 3]), ("text", [[2], [2]]),
        ])
        batch = rollout.run(model, PROMPTS)
        # Draft scores would be .8 and .2, but must never be requested. Finals
        # .85 and .95 reverse the draft ordering used by the old objective.
        from PIL import Image
        rendered = []
        def decode_codes(codes):
            self.assertIn(codes, [[[2] * 3], [[3] * 3]])
            image = Image.new("RGB", (4, 3), (codes[0][0], 0, 0))
            rendered.append(image)
            return [image]
        def score_images(images, questions):
            self.assertEqual(images, rendered)
            self.assertEqual([im.getpixel((0, 0))[0] for im in images], [2, 3])
            self.assertEqual(questions, [PROMPTS[0]["vqa_list"]] * 2)
            return [0.85, 0.95], [0.8, 0.9], [[0.85], [0.95]]
        reward = SimpleNamespace(score_images=score_images, combine=lambda am, gm: am)
        trainer.score_rollouts(batch, reward, SimpleNamespace(decode_codes=decode_codes))
        zero, groups = trainer.assign_advantages(batch, GRPOConfig(adv_norm="mean"))
        self.assertEqual((zero, groups), (0, 1))
        self.assertAlmostEqual(batch.trajectories[0].adv, -0.05)
        self.assertAlmostEqual(batch.trajectories[1].adv, 0.05)
        rows = rollout.build_training_rows(batch.trajectories, 1.0)
        for i, expected in enumerate((-0.05, 0.05)):
            selected = rows["train_mask"][i]
            torch.testing.assert_close(rows["adv"][i, selected], torch.full_like(rows["adv"][i, selected], expected))
            self.assertTrue(rows["train_mask"][i, 3])  # initial image
            self.assertTrue(rows["train_mask"][i, -1])  # terminal EOS
            self.assertTrue(torch.all(rows["weight"][i, ~selected] == 0))

    def test_prompt_groups_do_not_mix_and_equal_rewards_are_finite(self):
        trajectories = [Trajectory(i, [1, 2], [KIND_NONE, KIND_TXT], reward=r)
                        for i, r in [(0, 0.8), (1, 0.2), (0, 0.6), (1, 0.2)]]
        batch = RolloutBatch(PROMPTS * 2, trajectories)
        for mode in ("mean", "std"):
            zero, groups = trainer.assign_advantages(batch, GRPOConfig(adv_norm=mode))
            self.assertEqual((zero, groups), (1, 2))
            self.assertGreater(trajectories[0].adv, 0)
            self.assertLess(trajectories[2].adv, 0)
            self.assertEqual([trajectories[1].adv, trajectories[3].adv], [0, 0])

    def test_loss_averages_episodes_instead_of_weighting_by_length(self):
        rollout = episode_rollout()
        trajectories = [Trajectory(0, [1, 8, 2], [KIND_NONE, KIND_IMG, KIND_TXT], adv=1.0),
                        Trajectory(0, [1, 8, 9, 10, 7, 2],
                                   [KIND_NONE, KIND_IMG, KIND_IMG, KIND_IMG, KIND_TXT, KIND_TXT], adv=-1.0)]
        rows = rollout.build_training_rows(trajectories, 0.7)
        torch.testing.assert_close(rows["weight"].sum(1), torch.ones(2))
        selected = rows["train_mask"]
        logp = torch.zeros(int(selected.sum()), requires_grad=True)
        loss, _ = grpo_loss(logp, logp.detach(), logp.detach(), rows["adv"][selected],
                            rows["weight"][selected], rows["pos_kind"][selected], 2, GRPOConfig())
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logp.grad).all())

    def test_terminal_eos_alone_receives_policy_gradient(self):
        torch.manual_seed(41)
        model = get_peft_model(tiny_base(), LoraConfig(r=2, lora_alpha=4,
                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        rollout = episode_rollout()
        # Only EOS is a sampled action in this fixture, so any policy gradient
        # must have come from training the terminal decision.
        trajectory = Trajectory(0, [1, 8, 7, 2], [KIND_NONE] * 3 + [KIND_TXT], adv=1.0)
        rows = rollout.build_training_rows([trajectory], 1.0)
        logp = masked_token_logprobs(model, rows["input_ids"], rows["attention_mask"],
                                     rows["train_mask"], rows["pos_kind"], 8, 16)
        loss, _ = grpo_loss(logp, logp.detach(), logp.detach(), torch.ones(1),
                            torch.ones(1), torch.tensor([KIND_TXT]), 1, GRPOConfig())
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.parameters() if p.requires_grad))

    def test_microbatch_partition_does_not_change_episode_update(self):
        torch.manual_seed(42)
        model = get_peft_model(tiny_base(), LoraConfig(r=2, lora_alpha=4,
                              target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        models = [copy.deepcopy(model), copy.deepcopy(model)]
        for micro, candidate in enumerate(models, start=1):
            args = policy_args(train_micro_batch=micro)
            optimizer = torch.optim.SGD([p for p in candidate.parameters() if p.requires_grad], lr=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            trainer.train_on_rollouts(candidate, episode_rollout(), training_batch(0), args,
                                      GRPOConfig(), optimizer, scheduler, torch.device("cpu"))
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)

    def test_truncations_are_reported_separately_from_eos(self):
        rollout = episode_rollout(max_refinements=0)
        batch = rollout.run(ScriptedModel([("image", [[8] * 3, [9] * 3]),
                                           ("text", [[2], [3]])]), PROMPTS)
        stats = trainer.finalize_stats(trainer.rollout_stats(batch, "train/"), "train/")
        self.assertEqual(stats["train/eos_rate"], 0.5)
        self.assertEqual(stats["train/truncated_rate"], 0.5)
        self.assertEqual(stats["train/max_refinements_rate"], 0.5)

    def test_validation_uses_same_episode_policy_without_changing_config(self):
        rollout = episode_rollout()
        model = ScriptedModel([("image", [[8] * 3]), ("text", [[2]])])
        reward = SimpleNamespace(score_images=lambda *a: ([0.8], [0.7], [[0.8]]), combine=lambda a, g: a)
        args = SimpleNamespace(seed=42, gen_batch_size=2, log_images=0)
        metrics = trainer.run_validation(model, rollout, reward, PROMPTS, args, torch.device("cpu"), 0,
                                       SimpleNamespace(decode_codes=lambda codes: [object()]))
        self.assertEqual(metrics["val/am_final"], 0.8)
        self.assertEqual(metrics["val/eos_rate"], 1.0)
        self.assertTrue(rollout.cfg.reflect_sample)
        self.assertEqual(rollout.cfg.num_rollouts, 2)

    def test_old_objective_checkpoints_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state.json"
            state.write_text(json.dumps({"step": 12}))
            with self.assertRaisesRegex(ValueError, "old local-reward"):
                trainer.validate_resume_checkpoint(directory)
            state.write_text(json.dumps({"step": 12, "objective": trainer.RL_OBJECTIVE}))
            self.assertEqual(trainer.validate_resume_checkpoint(directory)["step"], 12)

    def test_visual_artifacts_preserve_history_and_renderer_seed(self):
        from PIL import Image
        random_draws = []
        def decode(trajectory):
            random_draws.append(torch.rand(1).item())
            return [Image.new("RGB", (4, 3), (i * 100, 0, 0))
                    for i in range(len(trajectory.images))]
        decoder = SimpleNamespace(device=torch.device("cpu"), decode_images=decode)
        trajectory = Trajectory(0, [], [], images=[[1] * 3, [2] * 3],
                                reflections=["correct the count", "done"],
                                stop_reason="eos", reward=0.9, am=0.9, gm=0.8)
        with tempfile.TemporaryDirectory() as directory:
            rng = torch.get_rng_state().clone()
            for step in (0, 25):
                logged = trainer.save_visual_evaluation(
                    decoder, [("<two> red dogs / unsafe-path", trajectory)], directory,
                    step, 42, {"val/reward_final": 0.9})
                self.assertEqual(len(logged), 1)
                self.assertIn("Final reward=0.9000", logged[0][1])
                with Image.open(logged[0][0]) as strip:
                    self.assertEqual(strip.size, (8, 3))
                    self.assertEqual(strip.getpixel((7, 0)), (100, 0, 0))
                folder = Path(directory) / f"step-{step}"
                record = json.loads((folder / "evaluation.json").read_text())
                self.assertEqual(record["step"], step)
                example = record["examples"][0]
                self.assertEqual(example["semantic_codes"], trajectory.images)
                self.assertEqual(example["reflections"], trajectory.reflections)
                self.assertEqual(example["stop_reason"], "eos")
                self.assertTrue(all((folder / name).is_file() for name in example["images"]))
            self.assertEqual(random_draws[0], random_draws[1])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_validation_saves_visuals_with_and_without_wandb(self):
        from PIL import Image
        decoder = SimpleNamespace(device=torch.device("cpu"),
                                  decode_images=lambda t: [t.final_image],
                                  decode_codes=lambda codes: [Image.new("RGB", (4, 3))])
        reward = SimpleNamespace(score_images=lambda *a: ([0.8], [0.7], [[0.8]]), combine=lambda a, g: a)
        with tempfile.TemporaryDirectory() as directory:
            for step, report in enumerate(("none", "wandb")):
                args = SimpleNamespace(seed=42, gen_batch_size=2, log_images=1,
                                       eval_output_dir=directory, output_dir="unused", report_to=report)
                model = ScriptedModel([("image", [[8] * 3]), ("text", [[2]])])
                fake_wandb = SimpleNamespace(Image=lambda path, caption: (path, caption))
                with patch.dict("sys.modules", {"wandb": fake_wandb}):
                    metrics = trainer.run_validation(model, episode_rollout(), reward,
                                                      PROMPTS, args, torch.device("cpu"), step, decoder)
                self.assertEqual(metrics["val/reward_final"], 0.8)
                self.assertEqual("val/images" in metrics, report == "wandb")
                record = json.loads((Path(directory) / f"step-{step}" / "evaluation.json").read_text())
                self.assertEqual(record["examples"][0]["prompt"], PROMPTS[0]["prompt"])
                if report == "wandb":
                    self.assertTrue(Path(metrics["val/images"][0][0]).is_file())

    def test_obsolete_cli_and_invalid_episode_limits_fail(self):
        required = ["train_grpo", "--model_name_or_path", "missing", "--data_path", "missing", "--output_dir", "missing"]
        for extra in (["--branch", "4,2"], ["--group", "parent"], ["--num_rollouts", "1"],
                      ["--max_refinements", "-1"], ["--max_seq_len", "0"], ["--reflect_tokens", "0"]):
            with self.subTest(extra=extra), patch("sys.argv", required + extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    trainer.parse_args()
        with patch("sys.argv", required):
            args = trainer.parse_args()
        self.assertEqual((args.num_rollouts, args.max_refinements, args.adv_norm), (4, 3, "mean"))


if __name__ == "__main__":
    unittest.main()
