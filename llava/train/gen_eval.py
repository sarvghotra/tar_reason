"""In-loop generation eval for the iterative draft -> critique -> refine SFT.

Runs inside the SFT trainer's ``evaluate()`` on the training GPUs, so no
separate job is needed. Everything is reused from the RL stack: the rollout
(``llava/train/rl/rollout.py``), the GenEval2 soft-TIFA pixel judge served by
``pixel_reward_server.py``, the oracle critique and the tree statistics of
``train_grpo.py``. What is measured, per val set in the gen-eval YAML:

``gen_<set>/*``         the loop as deployed: draft, greedy critique, refine
                        (am_0, am_1, am_final, delta_am = am_final - am_0 per
                        rollout, looks_good_pass/fail_1, fix_1, brk_1 per
                        skill, improved/degraded, delta_am_flagged_*).
``bank_<set>/critic_*`` the critic on the frozen draft bank (recall,
                        specificity, grounding, length) and its decision
                        log-prob at the "Self-reflect:" branch (AUROC, acc).
``bank_<set>/oracle_*`` the refiner on the bank's failing drafts under oracle
                        critiques (fix / brk per skill, delta_am).

The bank (``--gen_eval_bank``) is a JSONL written once by
``llava/train/build_draft_bank.py``: one row per (set, idx) with the draft's
image codes and the judge's per-question probabilities. It never changes
between checkpoints, so the critic and refiner numbers are comparable across
them, unlike numbers taken on each checkpoint's own drafts.

The judge is started on demand for each eval by the first local rank of every
node (one server per node, on the node's last GPU, next to a training rank)
and stopped afterwards, so its memory is only held while the eval runs.

The decision metric (``decision_*``) needs no generation or judge and runs at
every CE eval; the rest runs when ``step % gen_eval_steps == 0`` or at the
final step.
"""

import json
import os
import re
import subprocess
import sys
import sysconfig
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import Qwen2ForCausalLM

from llava.train.rl.dataset import GenEval2PromptDataset
from llava.train.rl.grpo import KIND_IMG, KIND_NONE
from llava.train.rl.oracle_critique import (BE_RE, COUNT_RE, OBJECT_RE, oracle_critique,
                                            prompt_nouns, singularize, split_subject)
from llava.train.rl.pixel_reward import PixelVQAReward
from llava.train.rl.reward import RewardConfig
from llava.train.rl.rollout import (LOOKS_GOOD_RE, Critique, Node, RolloutBatch, RolloutConfig,
                                    TreeRollout)
from llava.train.rl.train_grpo import (ImageDecoder, finalize_stats, reduce_sum, run_validation,
                                       score_nodes, tree_stats)
from llava.utils import rank0_print


@dataclass
class GenEvalConfig:
    data_path: str                      # YAML of val .jsonl files (GenEval2 format, lognames)
    steps: int = 4000                   # generation eval cadence (decision metric: every eval)
    max_prompts: int = 400              # per set, first rows of the file
    bank: Optional[str] = None          # draft bank JSONL (build_draft_bank.py)
    # Judge server (started per eval by the first rank of every node).
    judge_python: str = ""
    judge_model: str = ""
    judge_port: int = 8765
    judge_batch_size: int = 32
    judge_load_timeout: int = 1800
    answer_suffix: str = "geneval2"
    # De-tokenizer for the pixel judge / image logging.
    ar_path: str = ""
    encoder_path: str = ""
    decoder_path: str = ""
    siglip2_path: Optional[str] = None
    cfg_scale: float = 4.0
    decode_batch: int = 24
    images_per_request: int = 8
    # Sampling, as the benchmark eval script.
    scale: int = 0
    gen_seq_len: int = 729
    img_temperature: float = 1.0
    img_top_k: int = 1200
    img_top_p: float = 0.95
    reflect_tokens: int = 128
    gen_batch_size: int = 16
    fail_threshold: float = 0.5
    log_images: int = 4
    seed: int = 42
    decision_batch_size: int = 8


class PlainGenerate:
    """The SFT model (LlavaQwenForCausalLM) through the plain Qwen2 generate.

    Its own ``generate`` override turns the ids into ``inputs_embeds``, after
    which HF returns the new tokens only, while the rollout slices the prompt
    off itself (it was written for the plain Qwen2 policy). Routing around the
    override keeps the rollout's slicing right; with no images every forward
    is the plain text path anyway.
    """

    def __init__(self, model):
        self.m = model

    @property
    def training(self):
        return self.m.training

    def eval(self):
        self.m.eval()
        return self

    def train(self, mode=True):
        self.m.train(mode)
        return self

    def generate(self, **kwargs):
        return Qwen2ForCausalLM.generate(self.m, **kwargs)


class JudgeServer:
    """One pixel judge per node, started on demand on the node's last GPU."""

    def __init__(self, cfg: GenEvalConfig, out_dir: str):
        self.cfg = cfg
        self.out_dir = out_dir
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self.node_id = os.environ.get("SLURM_NODEID", "0")
        self.job_id = os.environ.get("SLURM_JOB_ID", "local")
        self.proc: Optional[subprocess.Popen] = None
        self.url = f"http://127.0.0.1:{cfg.judge_port}"
        self.ready_file = os.path.join(out_dir, f".judge_ready_{self.job_id}_n{self.node_id}")
        self.log_file = os.path.join(out_dir, f"judge_{self.job_id}_n{self.node_id}.log")

    @property
    def launcher(self) -> bool:
        return self.local_rank == 0

    def _ld_library_path(self) -> str:
        # Same fix as the RL launch script: the judge env's own libnvJitLink
        # must come before the cuda module's (see judge_ld_path there).
        try:
            purelib = subprocess.check_output(
                [self.cfg.judge_python, "-c",
                 "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                text=True, timeout=120).strip()
            nvjit = os.path.join(purelib, "nvidia", "nvjitlink", "lib")
        except Exception:
            nvjit = ""
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        if nvjit and os.path.isdir(nvjit):
            return f"{nvjit}:{cur}" if cur else nvjit
        return cur

    def start(self):
        if not self.launcher:
            self._barrier()
            return
        if os.path.exists(self.ready_file):
            os.remove(self.ready_file)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(self.local_world - 1)
        env["LD_LIBRARY_PATH"] = self._ld_library_path()
        # Not a torchrun child: drop the rendezvous vars so nothing in the
        # judge env tries to join the training process group.
        for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR",
                  "MASTER_PORT", "GROUP_RANK", "TORCHELASTIC_RUN_ID"):
            env.pop(k, None)
        cmd = [self.cfg.judge_python, "llava/train/rl/pixel_reward_server.py",
               "--model", self.cfg.judge_model, "--host", "127.0.0.1",
               "--port", str(self.cfg.judge_port), "--batch_size", str(self.cfg.judge_batch_size),
               "--answer_suffix", self.cfg.answer_suffix, "--answer_id_mode", "geneval2",
               "--attn_implementation", "sdpa", "--ready_file", self.ready_file]
        rank0_print(f"[gen_eval] starting judge on node {self.node_id} GPU {self.local_world - 1}: "
                    f"{self.cfg.judge_model} -> {self.url} (log {self.log_file})")
        log = open(self.log_file, "a")
        self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=os.getcwd())
        waited = 0
        while not os.path.exists(self.ready_file):
            if self.proc.poll() is not None:
                raise RuntimeError(f"judge died while loading; see {self.log_file}")
            if waited >= self.cfg.judge_load_timeout:
                self.stop(barrier=False)
                raise RuntimeError(f"judge not ready after {waited}s; see {self.log_file}")
            time.sleep(5)
            waited += 5
        rank0_print(f"[gen_eval] judge ready after {waited}s")
        self._barrier()

    def stop(self, barrier: bool = True):
        if barrier:
            self._barrier()      # nobody is still posting to it
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        if self.launcher and os.path.exists(self.ready_file):
            os.remove(self.ready_file)

    @staticmethod
    def _barrier():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


# ---------------------------------------------------------------------------
# Critique grounding
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z]+")


def failed_question_terms(vqa_list, skills, per_question, threshold) -> List[str]:
    """Content words that a critique of the failed questions should mention."""
    nouns = prompt_nouns(vqa_list)
    terms = set()
    for (q, a), p in zip(vqa_list, per_question):
        if p >= threshold:
            continue
        m = COUNT_RE.match(q) or OBJECT_RE.match(q)
        if m:
            plural = m.group(1)
            terms.update({plural, singularize(plural)})
            continue
        m = BE_RE.match(q)
        if m:
            subject, predicate = split_subject(m.group(2), nouns)
            terms.update({subject, singularize(subject)})
            # the attribute / relation words too ("blue", "left", "behind")
            terms.update(w for w in _WORD_RE.findall(predicate.lower())
                         if w not in _STOP)
            continue
        terms.update(w for w in _WORD_RE.findall(q.lower()) if w not in _STOP)
    return [t.lower() for t in terms if t]


_STOP = {"the", "a", "an", "is", "are", "in", "image", "there", "any", "of", "to", "and",
         "it", "its", "do", "does", "how", "many", "with", "on", "at", "or", "than", "this",
         "that", "than", "have", "has"}


def critique_grounded(text: str, terms: List[str]) -> bool:
    words = set(_WORD_RE.findall(text.lower()))
    return any(t in words or (" " in t and t in text.lower()) for t in terms)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class GenEvaluator:
    def __init__(self, cfg: GenEvalConfig, tokenizer, out_dir: str, device):
        self.cfg = cfg
        self.out_dir = out_dir
        self.device = device
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        image_start_id = tokenizer.convert_tokens_to_ids("<I0>")
        num_image_tokens = sum(1 for t in tokenizer.get_vocab()
                               if t.startswith("<I") and t[2:-1].isdigit())
        assert image_start_id is not None and image_start_id != tokenizer.unk_token_id
        rcfg = RolloutConfig(
            branch=[1, 1], scale=cfg.scale, gen_seq_len=cfg.gen_seq_len,
            img_temperature=cfg.img_temperature, img_top_k=cfg.img_top_k, img_top_p=cfg.img_top_p,
            reflect_tokens=cfg.reflect_tokens, reflect_sample=False,
            gen_batch_size=cfg.gen_batch_size, min_refines=0, images_per_critique=1)
        self.rollout = TreeRollout(tokenizer, rcfg, image_start_id, num_image_tokens, device)
        self.tok = tokenizer
        self.img_start = image_start_id
        # First token of the SFT no-issue reflection after "Self-reflect:".
        self.no_issue_first_id = self.rollout._encode(" no issues")[0]

        ds = GenEval2PromptDataset(cfg.data_path, seed=0, rank=self.rank, world_size=self.world)
        groups = ds.rank_groups()
        per_rank = max(1, cfg.max_prompts // self.world)
        self.val_sets: Dict[str, List[Dict]] = {k: v[:per_rank] for k, v in groups.items()}
        rank0_print(f"[gen_eval] val prompts per rank: {({k: len(v) for k, v in self.val_sets.items()})}")

        self.bank: Dict[str, Dict[int, Dict]] = {}
        if cfg.bank:
            self.load_bank(cfg.bank)

        self.judge = JudgeServer(cfg, out_dir)
        # Mirrors the argparse namespace run_validation / score_tree read.
        self.val_args = SimpleNamespace(
            seed=cfg.seed, eval_branch_list=[1, 1], gen_batch_size=cfg.gen_batch_size,
            draft_reward="own", stop_penalty=0.0, stop_threshold=1.0,
            oracle_fail_threshold=cfg.fail_threshold, break_weight=1.0, stop_reward="penalty",
            critique_break_weight=None, log_images=cfg.log_images)
        from tok.ta_tok import DEFAULT_SIGLIP2_PATH
        self.decoder_args = SimpleNamespace(
            ar_path=cfg.ar_path, encoder_path=cfg.encoder_path, decoder_path=cfg.decoder_path,
            siglip2_path=cfg.siglip2_path or DEFAULT_SIGLIP2_PATH, gen_seq_len=cfg.gen_seq_len, scale=cfg.scale,
            cfg_scale=cfg.cfg_scale, reward_decode_batch=cfg.decode_batch)

    # -- bank ------------------------------------------------------------------

    def load_bank(self, path: str):
        if not os.path.exists(path):
            rank0_print(f"[gen_eval] WARNING: bank {path} not found; bank metrics are skipped")
            return
        wanted = {(name, r["idx"]) for name, rows in self.val_sets.items() for r in rows}
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["set"], int(row["idx"]))
                if key not in wanted:
                    continue
                self.bank.setdefault(row["set"], {})[int(row["idx"])] = row
                n += 1
        missing = len(wanted) - n
        rank0_print(f"[gen_eval] bank {path}: {n} of this rank's {len(wanted)} val prompts "
                    f"({missing} missing)")

    def bank_nodes(self, name: str) -> Tuple[List[Dict], List[Node]]:
        """Round-0 nodes for this rank's banked drafts of set ``name``."""
        rows = self.val_sets[name]
        bank = self.bank.get(name, {})
        prompts, nodes = [], []
        for r in rows:
            b = bank.get(r["idx"])
            if b is None:
                continue
            prefix = self.rollout.make_prefix(r["prompt"])
            img = [c + self.img_start for c in b["codes"]]
            n = Node(prompt_idx=len(prompts), round=0, parent=None, first_born=True,
                     seq=prefix + img, kinds=[KIND_NONE] * len(prefix) + [KIND_IMG] * len(img),
                     seg_start=len(prefix), codes=list(b["codes"]))
            pq = [float(x) for x in b["per_question"]]
            n.per_question = pq
            n.am = sum(pq) / len(pq) if pq else 0.0
            n.gm = float(b.get("gm", 0.0))
            prompts.append(r)
            nodes.append(n)
        return prompts, nodes

    # -- decision metric (no generation, no judge) ------------------------------

    @torch.no_grad()
    def decision_metrics(self, model) -> Dict[str, float]:
        """P(first reflection token == ' no') at the 'Self-reflect:' branch of
        every banked draft, against the judge's pass/fail label."""
        out: Dict[str, float] = {}
        if not self.bank:
            return out
        was_training = model.training
        model.eval()
        thr = self.cfg.fail_threshold
        for name in self.val_sets:
            _, nodes = self.bank_nodes(name)
            scores, labels = [], []
            rows = [n.seq + self.rollout.reflect_splice for n in nodes]
            bs = self.cfg.decision_batch_size
            for s in range(0, len(rows), bs):
                ids, mask = self.rollout._left_pad(rows[s:s + bs])
                # Left padding: positions must start at the first real token
                # (generate derives them from the mask the same way).
                pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)
                h = model.get_model()(input_ids=ids, attention_mask=mask, position_ids=pos_ids,
                                      use_cache=False, return_dict=True).last_hidden_state[:, -1]
                logp = torch.log_softmax(model.lm_head(h).float(), dim=-1)
                scores.extend(logp[:, self.no_issue_first_id].exp().tolist())
            labels.extend(float(all(q >= thr for q in n.per_question)) for n in nodes)
            pairs = [(x, y) for x, y in zip(scores, labels)]
            if self.world > 1:
                gathered: List[List] = [None] * self.world
                dist.all_gather_object(gathered, pairs)
                pairs = [p for g in gathered for p in g]
            if not pairs:
                continue
            pos = [x for x, y in pairs if y > 0.5]
            neg = [x for x, y in pairs if y <= 0.5]
            pre = f"bank_{name}/decision_"
            out[pre + "n"] = float(len(pairs))
            out[pre + "p_no_pass"] = sum(pos) / len(pos) if pos else 0.0
            out[pre + "p_no_fail"] = sum(neg) / len(neg) if neg else 0.0
            out[pre + "acc"] = sum((x > 0.5) == (y > 0.5) for x, y in pairs) / len(pairs)
            if pos and neg:
                wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
                out[pre + "auroc"] = wins / (len(pos) * len(neg))
        if was_training:
            model.train()
        return out

    # -- generation evals ----------------------------------------------------------

    def critic_on_bank(self, gen, name: str) -> Dict[str, float]:
        prompts, nodes = self.bank_nodes(name)
        thr = self.cfg.fail_threshold
        sums: Dict[str, float] = {k: 0.0 for k in (
            "n_pass", "n_fail", "lg_pass", "flag_fail", "grounded", "grounded_n", "words", "words_n")}
        if nodes:
            texts = self.rollout.sample_critiques(gen, nodes)
            for p, n, t in zip(prompts, nodes, texts):
                ok = all(q >= thr for q in n.per_question)
                lg = LOOKS_GOOD_RE.search(t) is not None
                sums["n_pass" if ok else "n_fail"] += 1
                if ok:
                    sums["lg_pass"] += lg
                else:
                    sums["flag_fail"] += not lg
                    if not lg:
                        terms = failed_question_terms(p["vqa_list"], p["skills"], n.per_question, thr)
                        sums["grounded"] += critique_grounded(t, terms)
                        sums["grounded_n"] += 1
                sums["words"] += len(t.split())
                sums["words_n"] += 1
        sums = reduce_sum(sums, self.device)
        pre = f"bank_{name}/critic_"
        return {
            pre + "n": sums["n_pass"] + sums["n_fail"],
            pre + "specificity": sums["lg_pass"] / max(sums["n_pass"], 1.0),
            pre + "recall": sums["flag_fail"] / max(sums["n_fail"], 1.0),
            pre + "grounded": sums["grounded"] / max(sums["grounded_n"], 1.0),
            pre + "words": sums["words"] / max(sums["words_n"], 1.0),
            pre + "looks_good": (sums["lg_pass"] + sums["n_fail"] - sums["flag_fail"])
            / max(sums["n_pass"] + sums["n_fail"], 1.0),
        }

    def oracle_refine_on_bank(self, gen, reward, name: str) -> Dict[str, float]:
        """Refine every failing banked draft under the oracle critique."""
        prompts, roots = self.bank_nodes(name)
        thr = self.cfg.fail_threshold
        children: List[Node] = []
        leaves: List[Node] = []
        for n in roots:
            if all(q >= thr for q in n.per_question):
                leaves.append(n)
                continue
            p = prompts[n.prompt_idx]
            text = oracle_critique(p["vqa_list"], p["skills"], n.per_question, thr)
            splice = (self.rollout.reflect_splice + self.rollout._encode(" " + text)
                      + self.rollout.image_splice)
            c = Node(prompt_idx=n.prompt_idx, round=1, parent=n, first_born=True,
                     seq=n.seq + splice, kinds=n.kinds + [KIND_NONE] * len(splice),
                     seg_start=len(n.seq) + len(splice), teacher_forced=True, reflection=text,
                     critique_source="oracle")
            n.children.append(c)
            children.append(c)
        if children:
            images = self.rollout._sample_images(gen, [c.seq for c in children])
            for c, img in zip(children, images):
                c.seq.extend(img)
                c.kinds.extend([KIND_IMG] * len(img))
                c.codes = [t - self.img_start for t in img]
            score_nodes(children, prompts, reward)
        leaves.extend(children)
        batch = RolloutBatch(prompts=prompts, roots=roots, leaves=leaves,
                             nodes_by_round=[roots, children])
        prefix = f"bank_{name}/oracle_"
        sums = reduce_sum(tree_stats(batch, prefix, 1, thr), self.device)
        out = finalize_stats(sums, prefix)
        keep = ("am_0", "am_1", "am_final", "delta_am", "delta_gm", "delta_am_1",
                "improved_1", "degraded_1", "n_1")
        return {k: v for k, v in out.items()
                if k[len(prefix):] in keep or k[len(prefix):].startswith(("fix_", "brk_"))}

    def run(self, model, step: int) -> Dict[str, float]:
        """Generation evals; the judge and the de-tokenizer live only inside."""
        gen = PlainGenerate(model)
        metrics: Dict[str, float] = {}
        was_training = model.training
        model.eval()
        t0 = time.time()
        # run_validation reseeds the global RNG for reproducible sampling;
        # hand the training run its RNG streams back afterwards.
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None
        self.judge.start()
        decoder = None
        try:
            decoder = ImageDecoder(self.decoder_args, self.device)
            reward = PixelVQAReward(
                decoder.decode_pils, self.judge.url,
                RewardConfig(batch_size=16, answer_suffix=self.cfg.answer_suffix,
                             scale=self.cfg.scale, alpha=1.0),
                images_per_request=self.cfg.images_per_request,
                decode_chunk=self.cfg.decode_batch)
            info = reward.health()
            rank0_print(f"[gen_eval] step {step}: judge {info.get('model')} at {self.judge.url}")
            images: Dict[str, object] = {}
            for name, rows in self.val_sets.items():
                out = run_validation(
                    gen, self.rollout, reward, rows, self.val_args, self.device, step,
                    decoder if (self.rank == 0 and self.cfg.log_images > 0) else None,
                    prefix=f"gen_{name}/")
                # wandb.Image objects must not reach Trainer.log (they would land
                # in the JSON trainer state); log them straight to wandb instead.
                for k in [k for k in out if k.endswith("images")]:
                    images[k] = out.pop(k)
                metrics.update(out)
                if self.bank:
                    metrics.update(self.critic_on_bank(gen, name))
                    metrics.update(self.oracle_refine_on_bank(gen, reward, name))
            if images and self.rank == 0:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(images, commit=False)   # joins the trainer's next log
                except ImportError:
                    pass
        finally:
            self.judge.stop()
            del decoder
            torch.cuda.empty_cache()
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state(rng_cuda, self.device)
            if was_training:
                model.train()
        metrics["gen_eval/seconds"] = time.time() - t0
        return metrics

    def maybe_run(self, model, step: int, final: bool) -> Dict[str, float]:
        out = self.decision_metrics(model)
        if self.cfg.steps > 0 and (step % self.cfg.steps == 0 or final):
            out.update(self.run(model, step))
        return out
