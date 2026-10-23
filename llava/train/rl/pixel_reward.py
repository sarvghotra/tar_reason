"""Pixel-space GenEval2 soft-TIFA reward: de-tokenize the image, then ask a
frozen VLM judge the benchmark's VQA questions on the decoded pixels.

Same interface as ``TarLatentVQAReward`` (``score_images`` / ``combine``), so
``train_grpo.py::score_tree`` is agnostic to which one it holds. The difference
is what the judge sees: the latent reward scores the emitted image *tokens*
with Tar itself, this one scores the PNG a user would actually get, with the
external judge the benchmark uses. With GenEval2's own Qwen3-VL-8B judge in
``--answer_id_mode geneval2`` the reward *is* the benchmark score (the oracle);
a different judge (e.g. Gemma 4 26B) is a stricter, cheaper-to-trust proxy.

The judge lives in a separate process (``pixel_reward_server.py``) because it
needs a newer transformers than the Tar policy stack (pinned to 4.50), and
because a 26B judge (~50G in bf16) does not fit next to the policy on every
rank. Here we only decode codes -> pixels and POST them.

Note that the AR de-tokenizer samples, so identical image tokens decode to
slightly different pixels on each call: unlike the latent reward, a node's
score is not exactly reproducible.
"""

import base64
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Sequence, Tuple

from llava.train.rl.reward import RewardConfig, geometric_mean, combine_scores


def _png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class PixelVQAReward:
    """Scores (image_codes, vqa_list) pairs by decoding to pixels first.

    ``decode_fn(list_of_code_lists) -> list[PIL.Image]`` is supplied by the
    caller (``train_grpo.py::ImageDecoder.decode_pils``) so the visual
    de-tokenizer is loaded once per rank and shared with image logging.

    Images are decoded ``decode_chunk`` at a time (None: all at once; the AR
    de-tokenizer is launch-bound, so 8 -> 32 images per call is ~3x cheaper
    per image) and each chunk's requests go out on background threads while
    the next chunk decodes, so the judge works during the decode. Up to
    ``max_inflight`` requests are open at once; the server serialises them.
    """

    def __init__(self, decode_fn: Callable, server_url: str, cfg: RewardConfig,
                 images_per_request: int = 8, timeout: float = 1800.0,
                 retries: int = 3, decode_chunk: Optional[int] = None,
                 max_inflight: int = 4):
        self.decode_fn = decode_fn
        self.url = server_url.rstrip("/")
        self.cfg = cfg
        self.images_per_request = max(1, images_per_request)
        self.decode_chunk = decode_chunk
        self.max_inflight = max(1, max_inflight)
        self.timeout = timeout
        self.retries = max(1, retries)

    # -- server ---------------------------------------------------------------

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.url}/health", timeout=60) as r:
            return json.loads(r.read())

    def _post_score(self, items: List[dict]) -> List[List[float]]:
        body = json.dumps({"images": items}).encode()
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(
                f"{self.url}/score", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())["per_question"]
            except urllib.error.HTTPError as e:      # judge-side failure: don't retry
                raise RuntimeError(f"reward server {e.code}: {e.read().decode()[:500]}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                if attempt + 1 < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"reward server {self.url} unreachable: {last}")

    # -- scoring --------------------------------------------------------------

    def score_images(self, codes: Sequence[Sequence[int]],
                     vqa_lists: Sequence[Sequence[Tuple[str, str]]]
                     ) -> Tuple[List[float], List[float], List[List[float]]]:
        """Returns (AM, GM, per_question_probs) per image."""
        assert len(codes) == len(vqa_lists)

        def request(images, chunk_vqa):
            items = [{"png_b64": _png_b64(im), "vqa": [list(qa) for qa in vqa]}
                     for im, vqa in zip(images, chunk_vqa)]
            scores = self._post_score(items)
            assert len(scores) == len(items) and \
                all(len(s) == len(v) for s, v in zip(scores, chunk_vqa)), \
                "reward server returned a mismatched score shape"
            return scores

        step = self.decode_chunk or max(len(codes), 1)
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_inflight) as pool:
            for d0 in range(0, len(codes), step):
                images = self.decode_fn(codes[d0:d0 + step])
                for r0 in range(0, len(images), self.images_per_request):
                    r1 = min(r0 + self.images_per_request, len(images))
                    futures.append(pool.submit(request, images[r0:r1],
                                               vqa_lists[d0 + r0:d0 + r1]))
            per_question: List[List[float]] = [s for f in futures for s in f.result()]
        am = [sum(q) / len(q) if q else 0.0 for q in per_question]
        gm = [geometric_mean(q) if q else 0.0 for q in per_question]
        return am, gm, per_question

    def combine(self, am: float, gm: float) -> float:
        return combine_scores(self.cfg, am, gm)
