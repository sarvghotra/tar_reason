"""Re-grade existing GenEval2 images with a different VQA judge (default Qwen3.5-27B).

No images are generated. For every ``geneval2_results*.json`` ({prompt: image
path}) found under each source run dir, the same soft-TIFA scoring as
GenEval2's ``evaluation_optz.py`` is run with the new judge and written into a
mirrored run dir ``<src>_judge-<tag>`` (the source dir is never touched):

    geneval2_results.json        -> results_geneval2.jsonl       + eval_log.txt
    geneval2_results_bon<n>.json -> results_geneval2_bon<n>.jsonl + eval_log_bon<n>.txt

The {prompt: path} file is copied next to them, so the mirror dir looks like a
normal run dir to viz_geneval2_iter_results.py and eval/stats/geneval2.py.

Scoring is unchanged from evaluation_optz.py (same questions, same
"<question> Answer in one word." prompt, P(answer tokens) at the first answer
position) except for the judge-specific input preparation:
  * model loaded with AutoModelForImageTextToText (Qwen3_5ForConditionalGeneration);
  * chat template rendered with enable_thinking=False, so the generation prompt
    ends in "<think>\\n\\n</think>\\n\\n" and the next token is the answer
    (Qwen3.5 thinks by default; without this the next token is "<think>").

With --in_place the scores are written next to each geneval2_results*.json
instead (no mirror dir, no copy): the scoring step of the e2e eval launchers,
e.g. scripts/eval/iter_geneval2_adhoc_e2e_MDP_vllm_qwn27b.sh.

One process per GPU; jobs are split round-robin with --rank/--world_size.
Finished jobs (output file present) are skipped, so a re-submit resumes;
--overwrite re-scores them (needed when the images were regenerated).
Run in the ``t2v_metric`` env (transformers >= 5, the geneval2 env's 4.57 has
no qwen3_5).
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from scipy.stats import gmean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src_dirs', nargs='+', required=True)
    p.add_argument('--benchmark_data', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--tag', default='Qwen3.5-27B', help='Mirror dir suffix: <src>_judge-<tag>')
    p.add_argument('--in_place', action='store_true',
                   help='Write scores next to each geneval2_results*.json (no mirror dir).')
    p.add_argument('--overwrite', action='store_true',
                   help='Re-score jobs whose output file already exists.')
    p.add_argument('--geneval2_dir', default='/network/scratch/s/sarvjeet-singh.ghotra/git/GenEval2')
    p.add_argument('--method', default='soft_tifa', choices=['soft_tifa', 'vqascore'])
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world_size', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--prefetch', type=int, default=3)
    p.add_argument('--attn_implementation', default='sdpa')
    p.add_argument('--no_bucket', action='store_true',
                   help='Plain evaluation_optz batching (left-padded) instead of length buckets.')
    p.add_argument('--list_jobs', action='store_true', help='Print the job list and exit (no GPU).')
    p.add_argument('--show_prompt', action='store_true',
                   help='Print one rendered judge prompt (processor only, no model) and exit.')
    p.add_argument('--limit', type=int, default=None,
                   help='Debug: score only the first N work items of the first job, print, no writes.')
    return p.parse_args()


def mirror_root(src, tag):
    src = Path(src).resolve()
    return src.parent / f'{src.name}_judge-{tag}'


def collect_jobs(src_dirs, tag, in_place=False):
    """(image_filepath_data, output_file, log_file) for every graded file in the source runs."""
    jobs = []
    for src in src_dirs:
        src = Path(src).resolve()
        dst_root = src if in_place else mirror_root(src, tag)
        for f in sorted(src.rglob('geneval2_results*.json')):
            rel = f.relative_to(src)
            if 'gen_logs' in rel.parts:
                continue
            if f.name == 'geneval2_results.json':
                out_name, log_name = 'results_geneval2.jsonl', 'eval_log.txt'
            elif f.name.startswith('geneval2_results_bon'):
                n = f.name[len('geneval2_results_bon'):-len('.json')]
                out_name, log_name = f'results_geneval2_bon{n}.jsonl', f'eval_log_bon{n}.txt'
            else:
                continue
            dst_dir = dst_root / rel.parent
            jobs.append(dict(src=str(f), copy_to=str(dst_dir / f.name),
                             output_file=str(dst_dir / out_name), log_file=str(dst_dir / log_name)))
    return jobs


def setup_judge(ge2, model_path, device, attn_implementation, load_weights=True):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    # Batched scoring reads the logits at the last position: left padding.
    processor.tokenizer.padding_side = 'left'
    ge2.qwen_processor = processor

    # transformers 5.4 apply_chat_template: a kwarg that is not a template variable
    # *replaces* processor_kwargs (processing_utils.py:1732-1737), which would drop
    # padding. So pass enable_thinking only to templates that read it (Qwen3.5 yes,
    # Qwen3-VL-8B-Instruct no).
    template = processor.chat_template if isinstance(processor.chat_template, str) else ''
    template_kwargs = {'enable_thinking': False} if 'enable_thinking' in template else {}
    print(f'chat template kwargs: {template_kwargs}', flush=True)

    def prepare_batch(batch, image_cache):
        messages = [ge2.construct_message_with_image(item['text'], image_cache.get(item['image_filepath']))
                    for item in batch]
        return processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt',
            processor_kwargs={'padding': True},
            **template_kwargs,
        )
    # ge2.run looks prepare_batch up as a module global at call time.
    ge2.prepare_batch = prepare_batch

    if load_weights:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation=attn_implementation)
        model.to(device)
        model.eval()
        ge2.qwen_model = model
    return processor


def run_bucketed(ge2, processor, items, batch_size, num_workers, prefetch):
    """ge2.run, but every batch holds prompts of one token length: no padding.

    Left padding shifts Qwen3.5's scores by up to ~0.02 per question (bf16
    numerics through the padded path), enough to move AM by ~0.3. Prompt length
    = fixed template + question tokens + image tokens, and the question sits
    between special tokens, so the key (question token count, image size) fixes
    it. Each batch's attention_mask is checked to be all ones.
    """
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from tqdm import tqdm

    sizes = {}
    def image_size(path):
        if path not in sizes:
            with Image.open(path) as im:  # header only
                sizes[path] = im.size
        return sizes[path]

    groups = {}
    for item in items:
        n_text = len(processor.tokenizer(item['text'], add_special_tokens=False)['input_ids'])
        groups.setdefault((n_text, image_size(item['image_filepath'])), []).append(item)
    batches = [g[i:i + batch_size] for _, g in sorted(groups.items()) for i in range(0, len(g), batch_size)]
    print(f'{len(items)} items -> {len(groups)} length buckets, {len(batches)} batches '
          f'(vs {-(-len(items) // batch_size)} unbucketed)', flush=True)

    # Buckets cut across images, so keep every image decoded (512px RGB ~0.8 MB each).
    image_cache = ge2.ImageCache(maxsize=len(sizes) + 1)
    padded = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        pending, nxt = [], 0
        for _ in range(min(prefetch, len(batches))):
            pending.append(pool.submit(ge2.prepare_batch, batches[nxt], image_cache)); nxt += 1
        for batch in tqdm(batches, unit='batch'):
            inputs = pending.pop(0).result()
            if nxt < len(batches):
                pending.append(pool.submit(ge2.prepare_batch, batches[nxt], image_cache)); nxt += 1
            if 'attention_mask' in inputs and not bool(inputs['attention_mask'].all()):
                padded += 1
            lm_probs, _ = ge2.score_batch(batch, inputs, False)
            for item, p in zip(batch, lm_probs):
                item['lm_prob'] = p
    if padded:
        print(f'WARNING: {padded}/{len(batches)} batches still had padding', flush=True)
    return items


def score_job(ge2, benchmark_data, job, args, processor=None):
    image_data = json.load(open(job['src']))
    items, lens = ge2.build_work_items(benchmark_data, image_data, args.method)
    if args.limit:
        items = items[:args.limit]
    if args.no_bucket:
        items = ge2.run(items, args.batch_size, args.num_workers, args.prefetch, want_pred=False)
    else:
        items = run_bucketed(ge2, processor, items, args.batch_size, args.num_workers, args.prefetch)
    if args.limit:
        for it in items:
            print(f"{it['lm_prob']:.6f}  {it['text']}  <- {os.path.basename(os.path.dirname(os.path.dirname(it['image_filepath'])))}")
        return None
    scores = [[None] * n for n in lens]
    for it in items:
        scores[it['prompt_idx']][it['slot']] = float(it['lm_prob'])
    return scores


def write_job(job, scores, args):
    dst_dir = os.path.dirname(job['output_file'])
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.abspath(job['src']) != os.path.abspath(job['copy_to']):
        shutil.copyfile(job['src'], job['copy_to'])
    am = [sum(s) / len(s) for s in scores]
    lines = [f"Judge: {args.model_path}", "AM Score: {}".format(100 * sum(am) / len(am))]
    if 'soft_tifa' in args.method:
        gm = [gmean(s) for s in scores]
        lines.append("GM Score: {}".format(100 * sum(gm) / len(gm)))
    with open(job['log_file'], 'w') as f:
        f.write('\n'.join(lines) + '\n')
    # Output last: its presence marks the job done.
    tmp = job['output_file'] + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(scores, f)
    os.replace(tmp, job['output_file'])
    print('\n'.join(lines), flush=True)


def main():
    args = parse_args()
    sys.path.insert(0, args.geneval2_dir)
    import evaluation_optz as ge2

    jobs = collect_jobs(args.src_dirs, args.tag, args.in_place)
    if args.list_jobs:
        for j in jobs:
            print(('[done] ' if os.path.exists(j['output_file']) else '[todo] ') + j['src'])
        print(f'{len(jobs)} jobs')
        return

    benchmark_data = [json.loads(l) for l in open(args.benchmark_data)]

    if args.show_prompt:
        processor = setup_judge(ge2, args.model_path, 'cpu', args.attn_implementation, load_weights=False)
        image_data = json.load(open(jobs[0]['src']))
        items, _ = ge2.build_work_items(benchmark_data, image_data, args.method)
        inputs = ge2.prepare_batch(items[:2], ge2.ImageCache())
        for row in inputs['input_ids']:
            text = processor.tokenizer.decode(row)
            print(repr(text.replace('<|image_pad|>' * 4, '')[-300:]))
        print('input_ids', tuple(inputs['input_ids'].shape), 'pixel_values', tuple(inputs['pixel_values'].shape))
        print('answer ids', items[0]['answer_list'], items[0]['answer_ids'],
              [processor.tokenizer.decode([i]) for i in items[0]['answer_ids']])
        return

    mine = jobs[args.rank::args.world_size]
    todo = [j for j in mine if args.overwrite or not os.path.exists(j['output_file'])]
    print(f'[rank {args.rank}] {len(mine)} jobs, {len(todo)} to do', flush=True)
    if not todo:
        return

    processor = setup_judge(ge2, args.model_path, 'cuda', args.attn_implementation)
    # answer_token_ids in build_work_items now uses the new judge's tokenizer.
    for i, job in enumerate(todo):
        t0 = time.time()
        print(f'[rank {args.rank}] ({i+1}/{len(todo)}) {job["src"]}', flush=True)
        scores = score_job(ge2, benchmark_data, job, args, processor)
        if scores is None:  # --limit debug run
            return
        write_job(job, scores, args)
        print(f'[rank {args.rank}] done in {time.time()-t0:.0f}s -> {job["output_file"]}', flush=True)


if __name__ == '__main__':
    main()
