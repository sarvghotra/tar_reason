"""Persistent frozen Qwen3-VL judge. Run with a Qwen3-compatible Python env.

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
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
            if len(request["images"]) != len(request["vqa_lists"]):
                raise ValueError("Image/question count mismatch.")
            with redirect_stdout(sys.stderr):
                scores = [[score_question(model, processor, image, question, answer)
                           for question, answer in vqa]
                          for image, vqa in zip(request["images"], request["vqa_lists"])]
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
