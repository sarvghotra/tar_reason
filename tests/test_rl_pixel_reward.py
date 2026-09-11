"""Pixel reward boundary and official GenEval2 protocol regressions (CPU)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from llava.train.rl.pixel_reward import QwenPixelReward
from llava.train.rl.qwen_reward_worker import accepted_variants, score_question
from llava.train.rl.rollout import Trajectory
from llava.train.rl.train_grpo import ImageDecoder, save_visual_evaluation


class PixelRewardTests(unittest.TestCase):
    def test_official_variants_include_number_and_space_forms(self):
        self.assertEqual(accepted_variants("How many dogs?", "two"),
                         ["two", "Two", " two", " Two", "2", " 2"])
        self.assertEqual(accepted_variants("Are they red?", "Yes"), ["Yes", "yes", " yes", " Yes"])

    def test_official_prompt_and_first_token_sum_without_deduplication(self):
        class Inputs(dict):
            def to(self, device):
                return self
        seen = {}
        def template(messages, **kwargs):
            seen["messages"] = messages
            self.assertTrue(kwargs["add_generation_prompt"])
            return Inputs(input_ids=torch.tensor([[1, 2]]))
        def generate(**kwargs):
            self.assertFalse(kwargs["do_sample"])
            self.assertEqual(kwargs["max_new_tokens"], 1)
            self.assertFalse(torch.is_grad_enabled())
            return SimpleNamespace(scores=[torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))])
        # Deliberate token collisions and a leading-space token. Reference code
        # sums each first token, not unique IDs or whole-answer likelihoods.
        mapping = {"two": [1], "Two": [2], " two": [1], " Two": [2], "2": [3], " 2": [0, 3]}
        processor = SimpleNamespace(apply_chat_template=template,
                                    tokenizer=SimpleNamespace(encode=lambda text: mapping[text]))
        model = SimpleNamespace(device="cpu", generate=generate)
        result = score_question(model, processor, "/tmp/exact.png", "How many dogs?", "two")
        self.assertAlmostEqual(result, 1.5, places=6)
        content = seen["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "image", "image": "/tmp/exact.png"})
        self.assertEqual(content[1]["text"], "How many dogs? Answer in one word.")

    def start_fake_worker(self, response):
        real_popen = subprocess.Popen
        code = '''import sys,json
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
 r=json.loads(line)
 assert open(r["images"][0],"rb").read(8)==b"\\x89PNG\\r\\n\\x1a\\n"
 print(RESPONSE,flush=True)
'''.replace("RESPONSE", repr(json.dumps(response)))
        with patch("llava.train.rl.pixel_reward.subprocess.Popen",
                   side_effect=lambda *a, **k: real_popen([sys.executable, "-u", "-c", code], **k)):
            return QwenPixelReward(sys.executable, "fake", "cpu", timeout=30)

    def test_worker_receives_png_and_returns_am_not_gm(self):
        judge = self.start_fake_worker({"per_question": [[0.25, 1.0]]})
        try:
            am, gm, scores = judge.score_images([Image.new("RGB", (2, 2))],
                                                [[("a", "Yes"), ("b", "Yes")]])
            self.assertEqual(am, [0.625])
            self.assertEqual(gm, [0.5])
            self.assertEqual(scores, [[0.25, 1.0]])
        finally:
            judge.close()
        self.assertIsNotNone(judge.process.poll())

    def test_worker_failures_and_malformed_rewards_never_become_zero_reward(self):
        for response in ({"error": "judge OOM"}, {"per_question": []},
                         {"per_question": [[float("nan")]]}):
            with self.subTest(response=response):
                judge = self.start_fake_worker(response)
                try:
                    with self.assertRaises((RuntimeError, ValueError)):
                        judge.score_images([Image.new("RGB", (2, 2))], [[("a", "Yes")]])
                finally:
                    judge.close()

    def test_logged_final_pixels_are_identical_to_reward_input(self):
        decoder = ImageDecoder.__new__(ImageDecoder)
        decoder.device = torch.device("cpu")
        calls = []
        def decode(codes):
            calls.append(codes)
            return [Image.new("RGB", (4, 3), "blue") for _ in codes]
        decoder.decode_codes = decode
        scored = Image.new("RGB", (4, 3), "red")
        trajectory = Trajectory(0, [], [], images=[[1], [2]], final_image=scored,
                                reward=0.7, question_scores=[0.7], stop_reason="eos")
        with tempfile.TemporaryDirectory() as directory:
            save_visual_evaluation(decoder, [("red", trajectory)], directory, 0, 42, {})
            folder = Path(directory) / "step-0"
            record = json.loads((folder / "evaluation.json").read_text())["examples"][0]
            with Image.open(folder / record["images"][-1]) as saved:
                self.assertEqual(saved.tobytes(), scored.tobytes())
            self.assertEqual(record["question_scores"], [0.7])
        self.assertEqual(calls, [[[1]]])  # final semantic tokens never re-rendered


if __name__ == "__main__":
    unittest.main()
