"""Resumable sharded TIIF Qwen2.5-VL and GenAI Qwen3.5-27B evaluation."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from types import SimpleNamespace

from llava.train.rl.geneval2_eval_io import atomic_json, ensure_manifest, protocol_id, sha256_file
from llava.train.rl.instruction_eval_io import check_score, read_record, summarize


def tiif_answer(judge, row, image_path, seed, retries=4):
    import torch
    from eval.tiif_bench_vlm_judge import (
        PROMPT_TEMPLATES, OutputFormatError, extract_yes_no, first_yes_no, format_questions_prompt)
    torch.manual_seed(seed)
    questions = row['questions']
    attempts = []
    for attempt in range(retries + 1):
        prompt = format_questions_prompt(questions, PROMPT_TEMPLATES[attempt % len(PROMPT_TEMPLATES)])
        output = judge.answer([prompt], [str(image_path)], attempt > 0,
                              max(1024, 48 * len(questions)))[0]
        attempts.append(output)
        try:
            return extract_yes_no(output, questions), output, attempts, False
        except OutputFormatError:
            pass
    # Same single-question recovery as the upstream local TIIF judge.
    predictions, outputs = [], []
    for question in questions:
        for attempt in range(retries + 1):
            prompt = format_questions_prompt([question], PROMPT_TEMPLATES[0])
            output = judge.answer([prompt], [str(image_path)], attempt > 0, 192)[0]
            attempts.append(output)
            try:
                pred = first_yes_no(output)
                break
            except OutputFormatError:
                if attempt == retries:
                    raise
        predictions.append(pred)
        outputs.append(output)
    return predictions, '\n'.join(outputs), attempts, True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, default=421)
    p.add_argument('--summary_only', action='store_true')
    p.add_argument('--report_to', choices=['wandb', 'none'], default='wandb')
    args = p.parse_args()
    directory = Path(args.output_dir)
    if args.summary_only:
        metrics = summarize(directory)
        print(json.dumps(metrics, indent=2), flush=True)
        if args.report_to == 'wandb':
            import wandb
            with wandb.init(project=os.environ.get('WANDB_PROJECT', 'tar_reasoning'),
                            entity=os.environ.get('WANDB_ENTITY'), job_type='instruction_benchmark_scoring',
                            name=directory.name+'-scores', id=directory.name+'-scores', resume='allow',
                            config=metrics['judge']) as run:
                run.log({metrics['benchmark']: metrics})
        return
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    # t2v_metrics uses device_map='auto'. Give each worker only its own GPU.
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[local_rank] if devices != [''] else str(local_rank)
    import torch
    torch.set_num_threads(1)
    manifest = json.loads((directory/'manifest.json').read_text())
    data = json.loads(Path(manifest['benchmark']).read_text())
    if sha256_file(manifest['benchmark']) != manifest['benchmark_sha256']:
        raise ValueError('Benchmark changed after generation')
    rows, benchmark = data['rows'], data['benchmark']
    model = Path(args.model).resolve(strict=True)
    provenance = json.loads((model/'download_provenance.json').read_text())
    expected_repo = 'Qwen/Qwen2.5-VL-7B-Instruct' if benchmark == 'tiif' else 'Qwen/Qwen3.5-27B'
    if provenance['repo'] != expected_repo:
        raise ValueError(f'This scoring protocol requires {expected_repo}')
    score_dir = directory / 'scores'
    score_dir.mkdir(exist_ok=True)
    code_root = Path(__file__).resolve().parents[3]
    score_manifest = dict(protocol=benchmark+'_upstream_judge_v1',
                          generation_protocol_id=protocol_id(manifest), model=str(model),
                          model_provenance=provenance, model_config_sha256=sha256_file(model/'config.json'),
                          seed=args.seed, environment={k: version(k) for k in ('torch', 'transformers')},
                          evaluator_sha256=sha256_file(__file__),
                          io_sha256=sha256_file(code_root/'llava/train/rl/instruction_eval_io.py'),
                          upstream_judge_sha256=sha256_file(code_root/('eval/tiif_bench_vlm_judge.py' if benchmark == 'tiif'
                            else 'eval/vendor/qwen_vqascore/models/vqascore_models/qwen3vl_model.py')),
                          batch_size=1, max_retries=4 if benchmark == 'tiif' else 0)
    identity = ensure_manifest(score_dir, score_manifest)
    pending = []
    for index in range(rank, len(rows), world):
        row = rows[index]
        generated = read_record(directory, index, row, protocol_id(manifest))
        if generated is None:
            raise ValueError(f'Missing generation {index}; finish generation first')
        score_path = score_dir / f'prompt-{index:04d}.json'
        if score_path.exists():
            check_score(json.loads(score_path.read_text()), row, generated, identity)
        else:
            pending.append((index, row, generated, score_path))
    if not pending:
        print(f'Rank {rank}: all scores verified', flush=True)
        return
    if benchmark == 'tiif':
        from eval.tiif_bench_vlm_judge import Judge
        judge = Judge(SimpleNamespace(model=str(model), device='cuda', max_pixels=None,
                                      max_new_tokens=1024, temperature=1.0))
    else:
        from eval.vendor.qwen_vqascore.models.vqascore_models.qwen3vl_model import QWEN3_VL_MODELS, Qwen3VLModel
        QWEN3_VL_MODELS['qwen3.5-27b']['tokenizer']['path'] = str(model)
        judge = Qwen3VLModel(model_name='qwen3.5-27b', checkpoint=str(model), device='cuda')
    for index, row, generated, score_path in pending:
        image_path = directory/generated['image']
        record = dict(id=row['id'], index=index, protocol_id=identity, image_sha256=generated['image_sha256'])
        if benchmark == 'tiif':
            predictions, output, attempts, fallback = tiif_answer(judge, row, image_path, args.seed * 1000003 + index)
            record.update(attribute=row['dimension'], desc=row['register'], line_idx=row['line_idx'],
                          jsonl_file=row['jsonl_file'], questions=row['questions'], gt_answers=row['answers'],
                          model_pred=predictions, model_output=output, attempts=attempts, single_question_fallback=fallback)
        else:
            score = judge.forward([str(image_path)], [row['prompt']]).item()
            record.update(score=score, question='Does this figure show "{}"? Please answer Yes or No.'.format(row['prompt']),
                          answer='Yes')
        check_score(record, row, generated, identity)
        atomic_json(score_path, record)
        if benchmark == 'tiif':
            # Same schema/tree accepted by TIIF's official Excel summary scripts.
            atomic_json(directory/'eval_results'/'final'/row['dimension']/row['register'].split('_')[0]/f"{row['line_idx']}.json", record)
        print(json.dumps(dict(rank=rank, index=index, benchmark=benchmark, score=record.get('score'),
                              n_completed=index//world+1)), flush=True)


if __name__ == '__main__':
    main()
