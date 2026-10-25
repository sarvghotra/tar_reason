"""Pixel-space GenEval2 reward client; the Qwen worker may use a separate venv."""

import atexit
import json
import math
import os
import select
import subprocess
import tempfile
import time
from pathlib import Path


class QwenPixelReward:
    def __init__(self, python, model_path, device, timeout=1800, question_batch_size=1, wait=True):
        """Start the worker; with `wait=False` it loads in the background until `wait_ready()`."""
        worker = Path(__file__).with_name("qwen_reward_worker.py")
        self.timeout = timeout
        if question_batch_size < 1:
            raise ValueError("question_batch_size must be positive.")
        command = [python, "-u", str(worker), "--model", model_path, "--device", str(device)]
        if question_batch_size > 1:  # batch 1 keeps the exact legacy worker command
            command += ["--question_batch_size", str(question_batch_size)]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._response_buffer = bytearray()
        self._ready = False
        atexit.register(self.close)
        try:
            os.set_blocking(self.process.stdout.fileno(), False)
        except BaseException:
            self.close()
            raise
        if wait:
            self.wait_ready()

    def wait_ready(self):
        if self._ready:
            return
        try:
            if self._read() != {"ready": True}:
                raise RuntimeError("Unexpected Qwen reward worker startup response.")
        except BaseException:
            self.close()
            raise
        self._ready = True

    def _read(self):
        deadline = time.monotonic() + self.timeout
        # Never mix TextIOWrapper reads with these raw reads: its read-ahead
        # buffer is invisible to select(). Keep trailing bytes for the next line.
        while b"\n" not in self._response_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                self.close()
                raise TimeoutError("Qwen reward worker timed out; no substitute reward was used.")
            try:
                chunk = os.read(self.process.stdout.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise RuntimeError("Qwen reward worker exited before a complete response. "
                                   "Check its stderr and REWARD_PYTHON/model path.")
            self._response_buffer.extend(chunk)
        end = self._response_buffer.index(b"\n") + 1
        line = bytes(self._response_buffer[:end])
        del self._response_buffer[:end]
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

    def generate_texts(self, images, texts, max_new_tokens, batch_size=8):
        """Greedy judge-model replies to one (image, text) user turn each."""
        if len(images) != len(texts) or not all(isinstance(t, str) and t for t in texts):
            raise ValueError("Each image needs one nonempty text.")
        if not images:
            return []
        with tempfile.TemporaryDirectory(prefix="tar-qwen-teacher-") as directory:
            paths = []
            for index, image in enumerate(images):
                path = os.path.join(directory, f"{index}.png")
                image.save(path)
                paths.append(path)
            self.process.stdin.write(json.dumps(dict(op="generate", images=paths, texts=list(texts),
                                                     max_new_tokens=int(max_new_tokens),
                                                     batch_size=int(batch_size))) + "\n")
            self.process.stdin.flush()
            replies = self._read()["texts"]
        if len(replies) != len(images) or not all(isinstance(r, str) for r in replies):
            raise ValueError("Invalid Qwen generate response.")
        return replies
