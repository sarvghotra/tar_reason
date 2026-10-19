"""GPU check: batched renderer timing and batched-judge equivalence.

Uses saved SFT GenEval2 draft codes. Renders the first ``--render_compare`` images
one at a time and in one batch (timing only: the renderer samples, so pixels differ
by RNG), then judges the same batched-render PNGs with the per-question
``generate`` path (batch 1) and the batched forward path. Writes ``report.json`` and
exits non-zero if the judge paths differ beyond the tolerances.
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
    p.add_argument('--n_images', type=int, default=64)
    p.add_argument('--render_compare', type=int, default=24)
    p.add_argument('--render_batch_size', type=int, default=24)
    p.add_argument('--question_batch_size', type=int, default=32)
    p.add_argument('--ar_path', required=True)
    p.add_argument('--encoder_path', required=True)
    p.add_argument('--decoder_path', required=True)
    p.add_argument('--reward_model', required=True)
    p.add_argument('--reward_python', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--question_tol', type=float, default=2e-3)
    p.add_argument('--am_tol', type=float, default=1e-3)
    args = p.parse_args()
    from easydict import EasyDict
    from llava.train.rl.pixel_reward import QwenPixelReward
    from llava.train.rl.train_grpo import ImageDecoder, render_codes

    os.makedirs(args.output_dir, exist_ok=True)
    records = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.records, 'prompt-*.json')))]
    records = records[:args.n_images]
    if len(records) < args.n_images:
        raise SystemExit(f'Only {len(records)} records found')
    codes = [r['semantic_codes'][0] for r in records]
    vqa = [[tuple(q) for q in r['vqa_list']] for r in records]
    device = torch.device('cuda:0')
    loader = (torch.serialization.safe_globals([EasyDict]) if hasattr(torch.serialization, 'safe_globals')
              else contextlib.nullcontext())
    with loader:
        decoder = ImageDecoder(SimpleNamespace(ar_path=args.ar_path, encoder_path=args.encoder_path,
                                               decoder_path=args.decoder_path, gen_seq_len=729, scale=0,
                                               cfg_scale=4.0), device)
    report = dict(n_images=len(codes), render_batch_size=args.render_batch_size,
                  question_batch_size=args.question_batch_size)
    torch.manual_seed(0)
    render_codes(decoder, codes[:2], 1)  # warm-up
    torch.cuda.synchronize()
    start = time.time()
    render_codes(decoder, codes[:args.render_compare], 1)
    torch.cuda.synchronize()
    report['render_seconds_per_image_batch1'] = (time.time() - start) / args.render_compare
    start = time.time()
    images = render_codes(decoder, codes, args.render_batch_size)
    torch.cuda.synchronize()
    report['render_seconds_per_image_batched'] = (time.time() - start) / len(codes)
    del decoder
    torch.cuda.empty_cache()

    scores = {}
    for size in (1, args.question_batch_size):
        reward = QwenPixelReward(args.reward_python, args.reward_model, device, 3600, question_batch_size=size)
        try:
            reward.score_images(images[:2], vqa[:2])  # warm-up
            start = time.time()
            am, gm, per_question = reward.score_images(images, vqa)
            report[f'judge_seconds_per_image_batch{size}'] = (time.time() - start) / len(images)
            scores[size] = dict(am=am, per_question=per_question)
        finally:
            reward.close()
    a, b = scores[1], scores[args.question_batch_size]
    q_diff = [abs(x - y) for ra, rb in zip(a['per_question'], b['per_question']) for x, y in zip(ra, rb)]
    am_diff = [abs(x - y) for x, y in zip(a['am'], b['am'])]
    report.update(n_questions=len(q_diff), max_question_abs_diff=max(q_diff),
                  mean_question_abs_diff=sum(q_diff) / len(q_diff), max_am_abs_diff=max(am_diff),
                  mean_am_batch1=sum(a['am']) / len(a['am']),
                  mean_am_batched=sum(b['am']) / len(b['am']),
                  question_tol=args.question_tol, am_tol=args.am_tol)
    report['passed'] = report['max_question_abs_diff'] <= args.question_tol and report['max_am_abs_diff'] <= args.am_tol
    with open(os.path.join(args.output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.output_dir, 'scores.json'), 'w') as f:
        json.dump(dict(batch1=a, batched=b), f)
    print('BATCHED_REWARD_CHECK ' + json.dumps(report), flush=True)
    if not report['passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()
