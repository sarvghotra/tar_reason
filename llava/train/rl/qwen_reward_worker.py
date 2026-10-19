"""Persistent frozen Qwen3-VL judge. Run with a Qwen3-compatible Python env.

Requests are JSON lines. ``{"images", "vqa_lists"}`` scores questions (below).
``{"op": "generate", "images", "texts", "max_new_tokens", "batch_size"}`` returns
greedy free-form replies (teacher reflections for --objective teacher_tree).

Scoring protocol: https://github.com/facebookresearch/GenEval2/blob/main/evaluation.py
Match its first-token answer-variant sum literally, including duplicate token
IDs and space-leading numeric variants. This is benchmark compatibility, not
a normalized probability over unique answer tokens. Do not silently substitute
the latent scorer's deduplication/space filtering here.
"""

import argparse
import json
import sys
import traceback
from contextlib import redirect_stdout


NUMBER_WORDS = dict(zip(
    "one two three four five six seven eight nine ten".split(), map(str, range(1, 11))))


def accepted_variants(question, answer):
    if not question.startswith("How many"):
        return ["Yes", "yes", " yes", " Yes"]
    digit = NUMBER_WORDS.get(answer, "other")
    return [answer, answer.capitalize(), " " + answer, " " + answer.capitalize(), digit, " " + digit]


def score_question(model, processor, image, question, answer):
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question + " Answer in one word."}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                 output_scores=True, return_dict_in_generate=True)
        probs = torch.softmax(outputs.scores[0], dim=-1)
    return sum(probs[0, processor.tokenizer.encode(variant)[0]].item()
               for variant in accepted_variants(question, answer))


def score_batched(model, processor, images, vqa_lists, batch_size):
    """Score all (image, question) rows, ``batch_size`` per forward (left-padded).

    Same prompt and first-token variant summation as ``score_question``; reads
    the fp32 softmax at the last position, which is what ``generate`` scores.
    """
    import torch
    rows = [(i, slot, image, question, answer)
            for i, (image, vqa) in enumerate(zip(images, vqa_lists))
            for slot, (question, answer) in enumerate(vqa)]
    out = [[0.0] * len(vqa) for vqa in vqa_lists]
    tokenizer = processor.tokenizer
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # the last position must be every row's last real token
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question + " Answer in one word."}]}]
                for _, _, image, question, _ in chunk]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors="pt",
                                                  padding=True).to(model.device)
            with torch.inference_mode():
                logits = model(**inputs, logits_to_keep=1).logits
                probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
            for j, (i, slot, _, question, answer) in enumerate(chunk):
                out[i][slot] = sum(probs[j, tokenizer.encode(variant)[0]].item()
                                   for variant in accepted_variants(question, answer))
    finally:
        tokenizer.padding_side = previous_side
    return out


def score_request(model, processor, images, vqa_lists, batch_size=1):
    if batch_size <= 1:
        return [[score_question(model, processor, image, question, answer)
                 for question, answer in vqa]
                for image, vqa in zip(images, vqa_lists)]
    return score_batched(model, processor, images, vqa_lists, batch_size)


def generate_request(model, processor, images, texts, max_new_tokens, batch_size):
    """Greedy free-form replies to (image, text) user turns (teacher reflections).

    Not a benchmark path: rows are left-padded in batches of ``batch_size`` and
    only the newly generated tokens are decoded.
    """
    import torch
    tokenizer = processor.tokenizer
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    replies = []
    try:
        for start in range(0, len(images), batch_size):
            messages = [[{"role": "user", "content": [{"type": "image", "image": image},
                                                      {"type": "text", "text": text}]}]
                        for image, text in zip(images[start:start + batch_size], texts[start:start + batch_size])]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors="pt",
                                                  padding=True).to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            replies.extend(processor.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                                  skip_special_tokens=True))
    finally:
        tokenizer.padding_side = previous_side
    return replies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--question_batch_size", type=int, default=1,
                        help="(image, question) rows per forward; 1 = per-question generate.")
    args = parser.parse_args()
    # Keep stdout exclusively JSON; HF/accelerator startup messages go to stderr.
    with redirect_stdout(sys.stderr):
        import torch
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("REWARD_PYTHON needs a Qwen3-VL-compatible Transformers installation "
                               "(4.57+); keep it separate from the tar training environment.") from exc
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype="auto", device_map={"": args.device}, local_files_only=True).eval()
        model.requires_grad_(False)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "generate":
                if len(request["images"]) != len(request["texts"]):
                    raise ValueError("Image/text count mismatch.")
                max_new_tokens, batch_size = int(request["max_new_tokens"]), int(request["batch_size"])
                if max_new_tokens <= 0 or batch_size <= 0:
                    raise ValueError("max_new_tokens and batch_size must be positive.")
                with redirect_stdout(sys.stderr):
                    replies = generate_request(model, processor, request["images"], request["texts"],
                                               max_new_tokens, batch_size)
                response = json.dumps({"texts": replies})
            elif "op" in request:
                raise ValueError(f"Unknown request op {request['op']!r}.")
            else:  # scoring request: the unchanged benchmark protocol
                if len(request["images"]) != len(request["vqa_lists"]):
                    raise ValueError("Image/question count mismatch.")
                with redirect_stdout(sys.stderr):
                    scores = score_request(model, processor, request["images"], request["vqa_lists"],
                                           args.question_batch_size)
                response = json.dumps({"per_question": scores}, allow_nan=False)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = json.dumps({"error": str(exc)})
        finally:
            # Return free inference buffers to the GPU for the training process.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(response, flush=True)


if __name__ == "__main__":
    main()
