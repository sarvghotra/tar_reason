"""GPU check for --objective teacher_tree: teacher reflections on saved SFT drafts.

Renders the first ``--n_images`` saved draft codes (timing render batch sizes),
judges them (batch-1 benchmark path), asks the judge model for a teacher
reflection on each image (image + prompt only), and parses the replies into the
policy reflection format. Reports format compliance, policy-token lengths,
teacher-vs-judge (all_pass) verdict agreement and timings, and saves examples.
Writes ``report.json`` / ``examples.json``; exits non-zero when the malformed
rate exceeds ``--max_malformed_rate``.
"""
import argparse
import contextlib
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', required=True, help='Directory of saved prompt-*.json eval records')
    p.add_argument('--policy', required=True, help='SFT checkpoint (policy tokenizer)')
    p.add_argument('--n_images', type=int, default=64)
    p.add_argument('--render_batch_sizes', default='24,48')
    p.add_argument('--teacher_max_tokens', type=int, default=96)
    p.add_argument('--teacher_batch_size', type=int, default=16)
    p.add_argument('--reflect_tokens', type=int, default=128)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--max_malformed_rate', type=float, default=0.05)
    p.add_argument('--save_images', type=int, default=16)
    p.add_argument('--ar_path', required=True)
    p.add_argument('--encoder_path', required=True)
    p.add_argument('--decoder_path', required=True)
    p.add_argument('--reward_model', required=True)
    p.add_argument('--reward_python', required=True)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()
    from easydict import EasyDict
    from transformers import AutoTokenizer
    from llava.train.rl.pixel_reward import QwenPixelReward
    from llava.train.rl.teacher import TEACHER_PROMPT_VERSION, parse_teacher_text, teacher_instruction
    from llava.train.rl.train_grpo import ImageDecoder, render_codes, verdict_target

    os.makedirs(args.output_dir, exist_ok=True)
    records = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.records, 'prompt-*.json')))]
    records = records[:args.n_images]
    if len(records) < args.n_images:
        raise SystemExit(f'Only {len(records)} records found')
    codes = [r['semantic_codes'][0] for r in records]
    vqa = [[tuple(q) for q in r['vqa_list']] for r in records]
    prompts = [r['prompt'] for r in records]
    tok = AutoTokenizer.from_pretrained(args.policy)
    eos, im_start = tok.eos_token_id, tok.convert_tokens_to_ids('<im_start>')
    image_start = tok.convert_tokens_to_ids('<I0>')
    image_end = image_start + sum(1 for t in tok.get_vocab() if t.startswith('<I') and t[2:-1].isdigit())
    device = torch.device('cuda:0')
    loader = (torch.serialization.safe_globals([EasyDict]) if hasattr(torch.serialization, 'safe_globals')
              else contextlib.nullcontext())
    with loader:
        decoder = ImageDecoder(SimpleNamespace(ar_path=args.ar_path, encoder_path=args.encoder_path,
                                               decoder_path=args.decoder_path, gen_seq_len=729, scale=0,
                                               cfg_scale=4.0), device)
    report = dict(n_images=len(codes), teacher_prompt_version=TEACHER_PROMPT_VERSION,
                  teacher_max_tokens=args.teacher_max_tokens, teacher_batch_size=args.teacher_batch_size)
    torch.manual_seed(0)
    render_codes(decoder, codes[:2], 2)  # warm-up
    images = None
    for size in [int(x) for x in args.render_batch_sizes.split(',')]:
        torch.cuda.synchronize()
        start = time.time()
        try:
            rendered = render_codes(decoder, codes, size)
            torch.cuda.synchronize()
            report[f'render_seconds_per_image_batch{size}'] = (time.time() - start) / len(codes)
            report[f'render_peak_gib_batch{size}'] = torch.cuda.max_memory_allocated() / 2 ** 30
            images = images or rendered
        except torch.cuda.OutOfMemoryError:
            report[f'render_seconds_per_image_batch{size}'] = 'OOM'
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    del decoder
    torch.cuda.empty_cache()

    reward = QwenPixelReward(args.reward_python, args.reward_model, device, 3600)
    try:
        start = time.time()
        am, _, per_question = reward.score_images(images, vqa)
        report['judge_seconds_per_image'] = (time.time() - start) / len(images)
        reward.generate_texts(images[:2], [teacher_instruction(x) for x in prompts[:2]],
                              args.teacher_max_tokens, args.teacher_batch_size)  # warm-up
        start = time.time()
        replies = reward.generate_texts(images, [teacher_instruction(x) for x in prompts],
                                        args.teacher_max_tokens, args.teacher_batch_size)
        report['teacher_seconds_per_image'] = (time.time() - start) / len(images)
    finally:
        reward.close()

    examples, counts = [], dict(malformed=0, too_long=0, special=0, looks_good=0, passed=0,
                                correct=0, looks_good_on_pass=0, looks_good_on_fail=0)
    lengths = []
    for i, (prompt, reply, scores) in enumerate(zip(prompts, replies, per_question)):
        passed = verdict_target(scores, args.threshold)
        parsed = parse_teacher_text(reply)
        entry = dict(index=records[i]['index'], prompt=prompt, vqa_list=vqa[i], question_scores=scores,
                     am=am[i], all_pass=passed, reply=reply, policy_text=None, verdict=None, policy_tokens=None)
        if parsed is None:
            counts['malformed'] += 1
        else:
            text, verdict = parsed
            tokens = tok(text, add_special_tokens=False).input_ids + [eos if verdict else im_start]
            entry.update(policy_text=text, verdict=verdict, policy_tokens=len(tokens))
            if any(image_start <= t < image_end or t in (eos, im_start) for t in tokens[:-1]):
                counts['special'] += 1
            elif len(tokens) > args.reflect_tokens:
                counts['too_long'] += 1
            else:
                lengths.append(len(tokens))
                counts['looks_good'] += verdict
                counts['passed'] += passed
                counts['correct'] += verdict == passed
                counts['looks_good_on_pass'] += verdict and passed
                counts['looks_good_on_fail'] += verdict and not passed
        if i < args.save_images:
            name = f'image-{i:03d}.png'
            images[i].save(os.path.join(args.output_dir, name))
            entry['image'] = name
        examples.append(entry)
    usable = len(lengths)
    report.update(counts)
    report.update(usable=usable,
                  malformed_rate=(len(records) - usable) / len(records),
                  judge_pass_rate=sum(verdict_target(s, args.threshold) for s in per_question) / len(records),
                  teacher_looks_good_rate=counts['looks_good'] / max(usable, 1),
                  teacher_verdict_accuracy=counts['correct'] / max(usable, 1),
                  teacher_verdict_precision=counts['looks_good_on_pass'] / max(counts['looks_good'], 1),
                  teacher_verdict_recall=counts['looks_good_on_pass'] / max(counts['passed'], 1),
                  teacher_looks_good_on_fail=counts['looks_good_on_fail'] / max(usable - counts['passed'], 1),
                  policy_tokens_mean=sum(lengths) / max(usable, 1), policy_tokens_max=max(lengths, default=0))
    report['passed'] = report['malformed_rate'] <= args.max_malformed_rate
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.output_dir, 'examples.json'), 'w') as f:
        json.dump(examples, f, indent=2)
    print('TEACHER_REFLECTION_CHECK ' + json.dumps(report), flush=True)
    if not report['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
