"""Judge TIIF-Bench generations with a local Qwen2.5-VL, via transformers.

This is a drop-in replacement for TIIF-Bench's ``eval/eval_with_vlm.py`` that
talks to a local HF checkpoint instead of an OpenAI-compatible endpoint. Its
output files are byte-compatible with that script's, so TIIF-Bench's own
``eval/summary_results.py`` and ``eval/summary_dimension_results.py`` compute
the metrics unchanged -- the scores are the benchmark's own numbers.

Everything that defines the judgement is copied verbatim from eval_with_vlm.py:
the three prompt templates, the question formatting, the yes/no extraction, the
task collection (directory layout, per-prompt skip-if-exists) and the result
schema. Only the transport changes.

Why not vLLM's OpenAI server: in the ``vllm`` env's build,
``FusedInputNorm.forward`` (vllm/model_executor/models/vision.py) calls
``F.batch_norm(..., eps=0.0)``, which this torch rejects, so Qwen2.5-VL dies
during startup memory profiling on both the image and the video path.

Two deliberate improvements over eval_with_vlm.py:

* Batched generation, since every task is an independent single-image prompt.
* Bounded retries. eval_with_vlm.py loops forever when the judge returns the
  wrong number of answers; here the first pass is greedy (reproducible) and
  each retry re-rolls with sampling and a different template, up to
  ``--max_retries``. Anything still unparsed is reported, not retried forever.

Usage (one GPU per iteration, from any directory):

    python eval/tiif_bench_vlm_judge.py \\
        --jsonl_dir <TIIF-Bench>/data/testmini_eval_prompts \\
        --image_dir <results>/images --eval_model 7B_iter1 \\
        --output_dir <results>/eval_results --model <Qwen2.5-VL path>
"""

import argparse
import glob
import json
import os
import random
import re
import sys

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Verbatim from TIIF-Bench eval/eval_with_vlm.py -- do not reword. The judge
# picks uniformly among the three phrasings for every question list.
# ---------------------------------------------------------------------------
raw_prompt = '''
You are tasked with conducting a careful examination of the provided image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
1. Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
2. The order of answers must correspond exactly to the order of the questions.
3. Each question must have only one answer.
4. Directly return the answers to each question, without any additional content.
5. Each answer must be on its own line!
6. Make sure the number of output answers equal to the number of questions!
'''

raw_prompt_1 = '''
You are tasked with conducting a careful examination of the image. Based on the content of the image, please answer the following yes or no questions:

Questions:
##YNQuestions##

Note that:
Each answer should be on a separate line, starting with "yes" or "no", followed by the reason.
The order of answers must correspond exactly to the order of the questions.
Each question must have only one answer. Output one answer if there is only one question.
Directly return the answers to each question, without any additional content.
Each answer must be on its own line!
Make sure the number of output answers equal to the number of questions!
'''

raw_prompt_2 = '''
You are tasked with carefully examining the provided image and answering the following yes or no questions:

Questions:
##YNQuestions##

Instructions:

1. Answer each question on a separate line, starting with "yes" or "no", followed by a brief reason.
2. Maintain the exact order of the questions in your answers.
3. Provide only one answer per question.
4. Return only the answers—no additional commentary.
5. Each answer must be on its own line.
6. Ensure the number of answers matches the number of questions.
'''

PROMPT_TEMPLATES = (raw_prompt, raw_prompt_1, raw_prompt_2)
SYSTEM_PROMPT = "You are a professional image critic."


class OutputFormatError(Exception):
    pass


def first_yes_no(model_output):
    """The leading yes/no of a single-question completion."""
    match = re.search(r"^\s*(yes|no)\b", model_output.strip(),
                      flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        raise OutputFormatError(f"No yes/no line in: {model_output[:120]!r}")
    return match.group(1).lower()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_dir", required=True,
                        help="TIIF-Bench eval prompt directory (the *_eval_prompts.jsonl).")
    parser.add_argument("--image_dir", required=True,
                        help="Root of the <dimension>/<eval_model>/<desc>/<idx>.png tree.")
    parser.add_argument("--eval_model", required=True,
                        help="Model-name directory level inside --image_dir.")
    parser.add_argument("--output_dir", required=True,
                        help="Per-prompt result JSONs land in <output_dir>/<eval_model>/...")
    parser.add_argument("--model", required=True, help="Qwen2.5-VL checkpoint path.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Prompts per generate() call.")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Floor on the per-batch generation budget. The "
                             "actual budget is the larger of this and "
                             "--tokens_per_question x the batch's longest "
                             "question list, since every answer carries a reason.")
    parser.add_argument("--tokens_per_question", type=int, default=48,
                        help="Per-question share of the generation budget. A "
                             "20-question prompt needs ~450 tokens of "
                             "'yes/no + reason' lines; truncating it makes the "
                             "answer count unparsable and burns every retry.")
    parser.add_argument("--max_retries", type=int, default=4,
                        help="Re-rolls for a task whose answer count does not "
                             "match its question count.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for retries. The first pass "
                             "is always greedy, for reproducibility.")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="Cap the processor's visual token budget per image.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--postfix", default="",
                        help="Appended to the dimension directory name, as in "
                             "eval_with_vlm.py.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_jsonl_lines(jsonl_file):
    lines = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    return lines


def find_image_by_idx(img_dir, idx):
    """Same glob as eval_with_vlm.py: '<idx>.<ext>' for a known image type."""
    files = [f for f in glob.glob(os.path.join(img_dir, f"{idx}.*"))
             if f.lower().endswith((".png", ".jpg", ".jpeg", "webp"))]
    if not files:
        raise FileNotFoundError(f"No image found for index {idx} in {img_dir}")
    return files[0]


def collect_tasks(args):
    """Mirror eval_with_vlm.py's collect_tasks, including the skip-if-done."""
    tasks = []
    missing = 0
    for jsonl_file in sorted(glob.glob(os.path.join(args.jsonl_dir, "*.jsonl"))):
        lines = load_jsonl_lines(jsonl_file)
        if not lines:
            continue
        attr_type = lines[0]["type"]
        for desc in ("long_description", "short_description"):
            img_dir = os.path.join(args.image_dir, attr_type + args.postfix,
                                   args.eval_model, desc)
            out_dir = os.path.join(args.output_dir, args.eval_model, attr_type,
                                   "long" if desc.startswith("long") else "short")
            os.makedirs(out_dir, exist_ok=True)
            for idx, line in enumerate(lines):
                out_path = os.path.join(out_dir, f"{idx}.json")
                if os.path.exists(out_path):
                    continue
                try:
                    img_path = find_image_by_idx(img_dir, idx)
                except FileNotFoundError as error:
                    print(f"[Warning] {error}")
                    missing += 1
                    continue
                tasks.append({
                    "attribute": attr_type,
                    "desc": desc,
                    "jsonl_file": os.path.basename(jsonl_file),
                    "line_idx": idx,
                    "jsonl_line": line,
                    "img_path": img_path,
                    "out_path": out_path,
                })
    return tasks, missing


def format_questions_prompt(questions, template):
    """Verbatim from eval_with_vlm.py, with the template chosen by the caller."""
    formatted_questions = "\n".join(item.strip() for item in questions)
    return template.replace("##YNQuestions##", formatted_questions)


def extract_yes_no(model_output, questions):
    """Verbatim from eval_with_vlm.py."""
    lines = [line.strip() for line in model_output.strip().split("\n") if line.strip()]
    preds = []
    for line in lines:
        match = re.match(r"^(yes|no)\b", line.strip(), flags=re.IGNORECASE)
        if match:
            preds.append(match.group(1).lower())
    if len(preds) != len(questions):
        raise OutputFormatError(
            f"Preds count {len(preds)} != questions count {len(questions)}")
    return preds


def questions_of(task):
    return task["jsonl_line"].get("yn_question_list", [])


def batch_budget(args, batch):
    """Every answer is a 'yes/no + reason' line, so the budget has to scale with
    the question count: at --max_new_tokens=512 a 20-question prompt truncates
    mid-list and no number of re-rolls can make the answer count match."""
    longest = max(len(questions_of(task)) for task in batch)
    return max(args.max_new_tokens, args.tokens_per_question * longest)


def write_result(task, model_pred, model_output):
    result = {
        "attribute": task["attribute"],
        "desc": task["desc"],
        "jsonl_file": task["jsonl_file"],
        "line_idx": task["line_idx"],
        "questions": questions_of(task),
        "gt_answers": task["jsonl_line"].get("yn_answer_list", []),
        "model_pred": model_pred,
        "model_output": model_output,
    }
    with open(task["out_path"], "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)


class Judge:
    def __init__(self, args):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        proc_kwargs = {}
        if args.max_pixels is not None:
            proc_kwargs["max_pixels"] = args.max_pixels
        self.processor = AutoProcessor.from_pretrained(args.model, **proc_kwargs)
        # Batched generation needs the shorter rows padded on the left so every
        # row's last position is a real token.
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(args.device).eval()
        self.args = args

    @torch.inference_mode()
    def answer(self, prompts, image_paths, sample, max_new_tokens=None):
        """Return one decoded completion per (prompt, image) pair."""
        images, messages = [], []
        for prompt, image_path in zip(prompts, image_paths):
            image = Image.open(image_path).convert("RGB")
            images.append([image])
            messages.append([
                {"role": "system",
                 "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]},
            ])
        texts = [self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True)
            for message in messages]
        inputs = self.processor(text=texts, images=images, padding=True,
                                return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.args.max_new_tokens,
            do_sample=sample,
            temperature=self.args.temperature if sample else None,
            top_p=0.95 if sample else None,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        completions = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.batch_decode(
            completions, skip_special_tokens=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks, missing = collect_tasks(args)
    print(f"Total tasks to process: {len(tasks)}"
          + (f" ({missing} skipped, image not found)" if missing else ""))
    if not tasks:
        print("Nothing to do.")
        return

    judge = Judge(args)

    # Pass 0 is greedy; every later pass re-rolls the unparsed tasks with
    # sampling and a freshly drawn template.
    pending = tasks
    for attempt in range(args.max_retries + 1):
        sample = attempt > 0
        template_index = attempt % len(PROMPT_TEMPLATES)
        retry = []
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            prompts = [
                format_questions_prompt(
                    task["jsonl_line"].get("yn_question_list", []),
                    PROMPT_TEMPLATES[template_index]
                    if sample else PROMPT_TEMPLATES[0])
                for task in batch]
            try:
                outputs = judge.answer(prompts, [t["img_path"] for t in batch],
                                       sample, batch_budget(args, batch))
            except Exception as error:  # OOM or a corrupt image: retry smaller
                print(f"[Error] batch of {len(batch)} failed: {error}")
                retry.extend(batch)
                continue
            for task, model_output in zip(batch, outputs):
                questions = task["jsonl_line"].get("yn_question_list", [])
                try:
                    model_pred = extract_yes_no(model_output, questions)
                except OutputFormatError as error:
                    if args.verbose:
                        print(f"[Retry] {task['img_path']}: {error}")
                    retry.append(task)
                    continue
                write_result(task, model_pred, model_output)
            print(f"pass {attempt}: {start + len(batch)}/{len(pending)}",
                  flush=True)
        if not retry:
            pending = []
            break
        print(f"pass {attempt}: {len(retry)} task(s) returned an unusable "
              f"answer count; re-rolling with sampling")
        pending = retry

    # Last resort: ask one question at a time. A single-question prompt cannot
    # produce the wrong answer count as long as the judge answers at all, so the
    # long question lists that defeat the batched prompt still land in the
    # denominator instead of dropping out of it.
    if pending:
        print(f"{len(pending)} task(s) still unparsed; falling back to "
              f"one question per prompt", flush=True)
        still_pending = []
        for task in pending:
            questions = questions_of(task)
            preds, outputs = [], []
            for question in questions:
                prompt = format_questions_prompt([question], PROMPT_TEMPLATES[0])
                answer = None
                for attempt in range(args.max_retries + 1):
                    try:
                        completion = judge.answer(
                            [prompt], [task["img_path"]], attempt > 0,
                            args.tokens_per_question * 4)[0]
                        # One question, so the first yes/no line is the
                        # answer -- an extra trailing line is not a mismatch.
                        answer = first_yes_no(completion)
                    except Exception as error:
                        if args.verbose:
                            print(f"[Retry-single] {task['img_path']}: {error}")
                        continue
                    break
                if answer is None:
                    break
                preds.append(answer)
                outputs.append(completion.strip())
            if len(preds) != len(questions):
                still_pending.append(task)
                continue
            write_result(task, preds, "\n".join(outputs))
            print(f"  recovered {task['img_path']}", flush=True)
        pending = still_pending

    if pending:
        print(f"ERROR: {len(pending)} task(s) never produced a parsable answer "
              f"after {args.max_retries} retries:")
        for task in pending[:20]:
            print(f"  {task['img_path']}")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
