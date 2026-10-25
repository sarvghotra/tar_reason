"""Dump numerics of one Python environment for an old-vs-new environment comparison.

Run once per environment on the same GPU node (GPU type changes numerics too),
then compare the dumps with scripts/env_parity/compare.py. Measured, from saved
evaluation records (exact sampled sequences and the final images the judge saw):

  tokenizer  policy token ids of prompts, reflections and special tokens (must be identical)
  policy     teacher-forced log-probs of the sampled tokens, full and phase-restricted
             (image vocab for image tokens, non-image vocab for text), SFT base and +adapter
  renderer   TA-Tok features of semantic codes and VQ-decoder pixels of fixed codes
             (deterministic), plus seeded AR samples (reported only: the RNG may differ)
  judge      Qwen3-VL GenEval2 per-question scores of the saved final images
"""
import argparse
import glob
import json
import os
from pathlib import Path
import platform
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def records(directory, n):
    paths = sorted(glob.glob(os.path.join(directory, 'prompt-*.json')))[:n]
    if len(paths) < n:
        raise SystemExit(f'Need {n} records in {directory}, found {len(paths)}')
    return [(Path(p), json.loads(Path(p).read_text())) for p in paths]


def policy_logprobs(model, segments, img_start, n_img):
    import torch
    out = []
    for ids, kinds in segments:
        x = torch.tensor([ids], device=model.device)
        with torch.inference_mode():
            logits = model(input_ids=x, use_cache=False).logits[0, :-1].float()
        target = x[0, 1:]
        kind = torch.tensor(kinds[1:], device=model.device)
        keep = kind != 0
        full = logits.log_softmax(-1).gather(1, target[:, None])[:, 0]
        image = torch.full_like(logits, float('-inf'))
        image[:, img_start:img_start + n_img] = logits[:, img_start:img_start + n_img]
        text = logits.clone()
        text[:, img_start:img_start + n_img] = float('-inf')
        restricted = torch.where(kind == 1, image.log_softmax(-1).gather(1, target[:, None])[:, 0],
                                 text.log_softmax(-1).gather(1, target[:, None])[:, 0])
        out.append(dict(full=full[keep].cpu(), restricted=restricted[keep].cpu(), kinds=kind[keep].cpu()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--records', required=True, help='geneval2_final directory of a saved evaluation')
    p.add_argument('--n_policy', type=int, default=16)
    p.add_argument('--n_judge', type=int, default=200)
    p.add_argument('--base', required=True)
    p.add_argument('--adapter', default=None)
    p.add_argument('--attn', default='sdpa')
    p.add_argument('--ar_path', required=True)
    p.add_argument('--encoder_path', required=True)
    p.add_argument('--decoder_path', required=True)
    p.add_argument('--judge', default=None, help='Qwen3-VL-8B-Instruct; omit to skip the judge')
    p.add_argument('--skip', nargs='*', default=[], choices=['policy', 'renderer', 'judge'])
    p.add_argument('--out', required=True)
    args = p.parse_args()

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    from llava.train.rl.hf_compat import dtype_kwargs
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    dump = dict(meta=dict(python=platform.python_version(), torch=torch.__version__,
                          transformers=transformers.__version__, cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(0), attn=args.attn, base=args.base,
                          adapter=args.adapter, records=args.records))
    try:
        import peft
        dump['meta']['peft'] = peft.__version__
    except ImportError:
        pass
    timing = {}
    rows = records(args.records, max(args.n_policy, args.n_judge))
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    img_start = tokenizer.convert_tokens_to_ids('<I0>')
    n_img = sum(1 for t in tokenizer.get_vocab() if t.startswith('<I') and t[2:-1].isdigit())
    texts = ['<im_start>', '<I0>', '<I65535>', tokenizer.eos_token, ' looks good.',
             'no issues, it matches the prompt.\nCorrection: looks good.']
    for _, r in rows:
        texts.append(r['prompt'])
        texts.extend(r.get('reflections', []))
    dump['tokenizer'] = dict(vocab_size=len(tokenizer), img_start=img_start, n_img=n_img,
                             eos=tokenizer.eos_token_id, ids=[tokenizer.encode(t, add_special_tokens=False) for t in texts],
                             texts=texts)

    if 'policy' not in args.skip:
        started = time.monotonic()
        segments = [[(s['input_ids'], s['position_kinds']) for s in r['context_segments']]
                    for _, r in rows[:args.n_policy]]
        model = Qwen2ForCausalLM.from_pretrained(args.base, **dtype_kwargs(torch.bfloat16),
                                                 attn_implementation=args.attn).to(device).eval()
        dump['policy_base'] = [policy_logprobs(model, s, img_start, n_img) for s in segments]
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
            dump['adapter_lora_tensors'] = sum('lora_' in n for n, _ in model.named_parameters())
            dump['policy_adapter'] = [policy_logprobs(model, s, img_start, n_img) for s in segments]
        del model
        torch.cuda.empty_cache()
        timing['policy'] = time.monotonic() - started

    if 'renderer' not in args.skip:
        started = time.monotonic()
        from types import SimpleNamespace
        from llava.train.rl.train_grpo import ImageDecoder
        decoder = ImageDecoder(SimpleNamespace(ar_path=args.ar_path, encoder_path=args.encoder_path,
                                               decoder_path=args.decoder_path, gen_seq_len=729, scale=0,
                                               cfg_scale=4.0), device)
        codes = torch.tensor([r['semantic_codes'][-1] for _, r in rows[:8]], device=device)
        with torch.inference_mode():
            dump['ta_tok_features'] = decoder.tok.encoder.decode_from_bottleneck(codes).float().cpu()
            torch.manual_seed(1234)
            features = decoder.tok.encoder.decode_from_bottleneck(codes[:4])
            dump['ar_samples'] = decoder.tok.ar_sample(features, {'cfg_scale': 4.0}).cpu()
            g = torch.Generator().manual_seed(4321)
            fixed = torch.randint(0, decoder.tok.decoder.codebook_size, tuple(dump['ar_samples'].shape),
                                  generator=g).to(device)
            dump['vq_pixels'] = decoder.tok.decode_from_vqvae_indices(fixed).cpu()
        del decoder
        torch.cuda.empty_cache()
        timing['renderer'] = time.monotonic() - started

    if args.judge and 'judge' not in args.skip:
        started = time.monotonic()
        from PIL import Image
        from llava.train.rl.pixel_reward import QwenPixelReward
        judge = QwenPixelReward(sys.executable, args.judge, device)
        judged = rows[:args.n_judge]
        images = [Image.open(path.parent/r['image']).convert('RGB') for path, r in judged]
        _, _, scores = judge.score_images(images, [r['vqa_list'] for _, r in judged])
        judge.close()
        dump['judge'] = dict(index=[r['index'] for _, r in judged], scores=scores,
                             recorded=[r['question_scores'] for _, r in judged])
        timing['judge'] = time.monotonic() - started
    dump['meta']['seconds'] = timing
    torch.save(dump, args.out)
    print(json.dumps(dump['meta'], indent=1))


if __name__ == '__main__':
    main()
