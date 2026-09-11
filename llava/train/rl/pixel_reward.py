"""Pixel-space GenEval2 reward client; the Qwen worker may use a separate venv."""

import atexit
import json
import math
import os
import select
import subprocess
import tempfile
from pathlib import Path


class QwenPixelReward:
    def __init__(self, python, model_path, device, timeout=1800):
        worker = Path(__file__).with_name("qwen_reward_worker.py")
        self.timeout = timeout
        self.process = subprocess.Popen(
            [python, "-u", str(worker), "--model", model_path, "--device", str(device)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        atexit.register(self.close)
        try:
            if self._read() != {"ready": True}:
                raise RuntimeError("Unexpected Qwen reward worker startup response.")
        except BaseException:
            self.close()
            raise

    def _read(self):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            self.close()
            raise TimeoutError("Qwen reward worker timed out; no substitute reward was used.")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Qwen reward worker exited. Check its stderr and REWARD_PYTHON/model path.")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(f"Qwen reward worker failed: {result['error']}")
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()

    def score_images(self, images, vqa_lists):
        if len(images) != len(vqa_lists) or any(not q for q in vqa_lists):
            raise ValueError("Each rendered image must have a nonempty VQA list.")
        with tempfile.TemporaryDirectory(prefix="tar-qwen-reward-") as directory:
            paths = []
            for index, image in enumerate(images):
                path = os.path.join(directory, f"{index}.png")
                image.save(path)
                paths.append(path)
            self.process.stdin.write(json.dumps(dict(images=paths, vqa_lists=vqa_lists)) + "\n")
            self.process.stdin.flush()
            per_question = self._read()["per_question"]
        if (len(per_question) != len(images) or
                any(len(p) != len(q) for p, q in zip(per_question, vqa_lists)) or
                any(not math.isfinite(v) or v < 0 for row in per_question for v in row)):
            raise ValueError("Invalid Qwen per-question rewards.")
        am = [sum(row) / len(row) for row in per_question]
        gm = [0.0 if any(v == 0 for v in row) else
              math.exp(sum(math.log(v) for v in row) / len(row)) for row in per_question]
        return am, gm, per_question
