"""Compare two scripts/env_parity/dump.py outputs (reference environment first).

Gates (exit 1 if any fails):
  tokenizer ids, vocabulary and special token ids identical
  policy restricted log-probs: mean |diff| <= --logprob_mean_tol and p99 <= --logprob_p99_tol
  TA-Tok features / VQ pixels from fixed codes: max |diff| <= --feature_tol / --pixel_tol
  judge: mean |per-question diff| <= --judge_mean_tol; reports AM drift
AR samples (seeded) and the share of tokens whose greedy argmax changed are reported only.
"""
import argparse
import json
import sys


def stats(diffs):
    import torch
    d = torch.cat([x.flatten() for x in diffs]) if diffs else torch.zeros(0)
    if not len(d):
        return dict(n=0)
    a = d.abs()
    return dict(n=len(d), mean_abs=a.mean().item(), max_abs=a.max().item(),
                p99_abs=a.quantile(0.99).item() if len(a) > 1 else a.max().item(), mean=d.mean().item())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('reference')
    p.add_argument('candidate')
    p.add_argument('--logprob_mean_tol', type=float, default=0.02)
    p.add_argument('--logprob_p99_tol', type=float, default=0.25)
    p.add_argument('--feature_tol', type=float, default=0.05)
    p.add_argument('--pixel_tol', type=float, default=8.0)
    p.add_argument('--judge_mean_tol', type=float, default=0.02)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    import torch
    ref, new = torch.load(args.reference, weights_only=False), torch.load(args.candidate, weights_only=False)
    report = dict(reference=ref['meta'], candidate=new['meta'], gates={})
    gates = report['gates']

    rt, nt = ref['tokenizer'], new['tokenizer']
    same_ids = rt['texts'] == nt['texts'] and rt['ids'] == nt['ids']
    gates['tokenizer'] = same_ids and all(rt[k] == nt[k] for k in ('vocab_size', 'img_start', 'n_img', 'eos'))
    report['tokenizer'] = dict(texts=len(rt['texts']), identical_ids=same_ids,
                               **{k: (rt[k], nt[k]) for k in ('vocab_size', 'img_start', 'n_img', 'eos')})

    for key in ('policy_base', 'policy_adapter'):
        if key not in ref or key not in new:
            continue
        for kind in ('restricted', 'full'):
            diffs = [n[kind] - r[kind] for rr, nn in zip(ref[key], new[key]) for r, n in zip(rr, nn)]
            report[f'{key}_{kind}'] = stats(diffs)
        s = report[f'{key}_restricted']
        gates[key] = s['mean_abs'] <= args.logprob_mean_tol and s['p99_abs'] <= args.logprob_p99_tol
    if 'adapter_lora_tensors' in new:
        report['adapter_lora_tensors'] = (ref.get('adapter_lora_tensors'), new['adapter_lora_tensors'])
        gates['adapter_loaded'] = ref.get('adapter_lora_tensors') == new['adapter_lora_tensors'] > 0

    if 'ta_tok_features' in ref and 'ta_tok_features' in new:
        f = stats([new['ta_tok_features'] - ref['ta_tok_features']])
        v = stats([new['vq_pixels'].float() - ref['vq_pixels'].float()])
        report['ta_tok_features'], report['vq_pixels'] = f, v
        gates['ta_tok_features'] = f['max_abs'] <= args.feature_tol
        gates['vq_pixels'] = v['max_abs'] <= args.pixel_tol
        report['ar_samples_identical_fraction'] = (ref['ar_samples'] == new['ar_samples']).float().mean().item()

    if 'judge' in ref and 'judge' in new:
        assert ref['judge']['index'] == new['judge']['index']
        rq = [v for row in ref['judge']['scores'] for v in row]
        nq = [v for row in new['judge']['scores'] for v in row]
        q = stats([torch.tensor(nq) - torch.tensor(rq)])
        am = lambda rows: [sum(r) / len(r) for r in rows]
        a = stats([torch.tensor(am(new['judge']['scores'])) - torch.tensor(am(ref['judge']['scores']))])
        report['judge_question'], report['judge_am'] = q, a
        report['judge_am_mean'] = (sum(am(ref['judge']['scores'])) / len(ref['judge']['scores']),
                                   sum(am(new['judge']['scores'])) / len(new['judge']['scores']))
        recorded = [v for row in ref['judge']['recorded'] for v in row]
        report['reference_vs_recorded_question'] = stats([torch.tensor(rq) - torch.tensor(recorded)])
        gates['judge'] = q['mean_abs'] <= args.judge_mean_tol

    report['passed'] = all(gates.values())
    text = json.dumps(report, indent=1, default=str)
    if args.out:
        open(args.out, 'w').write(text)
    print(text)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
