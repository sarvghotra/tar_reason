"""Render saved episode images 1–MAX_ORDINAL without resampling or extending the policy.

Image N means up to N images: early termination retains the last complete
image. Episode image counts identify actual ordinal coverage separately.
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

from llava.train.rl import geneval2_eval_io as ge
from llava.train.rl import instruction_eval_io as ins

MAX_ORDINAL = 8  # looks_good episodes may hold up to max_refinements + 1 images


def selection(record, ordinal):
    images = record['semantic_codes']
    if type(ordinal) is not int or not 1 <= ordinal <= MAX_ORDINAL or not images:
        raise ValueError(f'Require ordinal 1–{MAX_ORDINAL} and at least one complete image')
    for codes in images:
        if len(codes) != 729 or any(type(c) is not int or not 0 <= c < 65536 for c in codes):
            raise ValueError('Invalid complete semantic image')
    chosen = min(ordinal, len(images)) - 1
    return dict(requested_ordinal=ordinal, selected_ordinal=chosen + 1,
                n_episode_images=len(images), actual_second_available=len(images) >= 2,
                reused_final_image=chosen == len(images) - 1)


def source_data(source):
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text())
    if ge.sha256_file(manifest['benchmark']) != manifest['benchmark_sha256']:
        raise ValueError('Source benchmark changed')
    kind = ('geneval2' if manifest['protocol'] == 'geneval2_full_final_pixel_v1'
            else json.loads(Path(manifest['benchmark']).read_text())['benchmark'])
    if kind not in ('geneval2', 'tiif', 'genai'):
        raise ValueError('Unsupported benchmark')
    rows = (ge.load_benchmark(manifest['benchmark']) if kind == 'geneval2'
            else ins.load_benchmark(manifest['benchmark']))
    if len(rows) != manifest['n_prompts']:
        raise ValueError('Source prompt count mismatch')
    hashes = [ge.sha256_file(source/f'prompt-{i:04d}.json') for i in range(len(rows))]
    return kind, rows, manifest, hashes


def checked_source(source, index, row, manifest, digest, kind):
    path = Path(source)/f'prompt-{index:04d}.json'
    if ge.sha256_file(path) != digest:
        raise ValueError('Source episode changed')
    reader = ge.read_record if kind == 'geneval2' else ins.read_record
    record = reader(source, index, row, ge.protocol_id(manifest))
    if record is None or ge.sha256_file(path) != digest:
        raise ValueError('Source episode missing or changed during read')
    if 'max_refinements' in manifest and len(record['semantic_codes']) > manifest['max_refinements'] + 1:
        raise ValueError('Source episode exceeds its declared image cap')
    return record


def check_selection(record, source_record, digest, ordinal):
    if record.get('selection') != selection(source_record, ordinal) or record.get('source_record_sha256') != digest:
        raise ValueError('Stale intermediate image selection')
    if record['selection']['reused_final_image'] and record['image_sha256'] != source_record['image_sha256']:
        raise ValueError('Final pixels were not reused exactly')


def validate_outputs(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    kind, rows, original, hashes = source_data(manifest['source_dir'])
    if hashes != manifest['source_record_sha256'] or ge.protocol_id(original) != manifest['source_protocol_id']:
        raise ValueError('Source changed after replay')
    reader = ge.read_record if kind == 'geneval2' else ins.read_record
    records = []
    for i, row in enumerate(rows):
        source = checked_source(manifest['source_dir'], i, row, original, hashes[i], kind)
        record = reader(directory, i, row, ge.protocol_id(manifest))
        if record is None:
            raise ValueError('Intermediate generation incomplete')
        check_selection(record, source, hashes[i], manifest['image_ordinal'])
        records.append(record)
    return kind, rows, manifest, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_dir')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--image_ordinal', type=int, choices=range(1, MAX_ORDINAL + 1))
    p.add_argument('--reward_python')
    p.add_argument('--summary_only', action='store_true')
    args = p.parse_args()
    directory = Path(args.output_dir)
    if args.summary_only:
        kind, rows, manifest, records = validate_outputs(directory)
        if kind == 'geneval2':
            metrics = ge.summarize(directory, rows, manifest)
        else:
            metrics = dict(n_prompts=len(rows), protocol_id=ge.protocol_id(manifest))
            ge.atomic_json(directory/'generation_complete.json', metrics)
        print('INTERMEDIATE_COMPLETE ' + json.dumps(metrics), flush=True)
        return
    if args.source_dir is None or args.image_ordinal is None:
        p.error('Generation requires --source_dir and --image_ordinal')
    import torch
    from importlib.metadata import version
    from easydict import EasyDict
    from llava.train.rl.train_grpo import ImageDecoder
    from llava.train.rl.pixel_reward import QwenPixelReward

    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
    torch.cuda.set_device(device)
    kind, rows, original, hashes = source_data(args.source_dir)
    renderer = original['renderer']
    if any(renderer[k] != v for k, v in dict(temperature=1, top_k=0, top_p=1).items()):
        raise ValueError('Unsupported source renderer sampling')
    environment = {k: version(k) for k in original['environment']}
    if environment != original['environment']:
        raise ValueError('Replay renderer environment differs from source')
    manifest = dict(original, protocol='saved_episode_intermediate_pixel_v1',
                    benchmark_kind=kind, image_ordinal=args.image_ordinal,
                    source_dir=str(Path(args.source_dir).resolve()),
                    source_protocol_id=ge.protocol_id(original), source_record_sha256=hashes,
                    replay_code_sha256=ge.sha256_file(__file__),
                    replay_snapshot_sha256=ge.sha256_file(Path(__file__).resolve().parents[3]/'manifest.json'),
                    replay_world_size=world,
                    renderer_seed='source_seed * 1000003 + prompt_index * 7919 + 17',
                    early_stop_policy='retain_last_complete_image; separately report actual ordinal coverage')
    directory.mkdir(parents=True, exist_ok=True)
    identity = ge.ensure_manifest(directory, manifest)
    reader = ge.read_record if kind == 'geneval2' else ins.read_record
    decoder, reward = None, None
    try:
        for i in range(rank, len(rows), world):
            source = checked_source(args.source_dir, i, rows[i], original, hashes[i], kind)
            choice = selection(source, args.image_ordinal)
            existing = reader(directory, i, rows[i], identity)
            if existing is not None:
                check_selection(existing, source, hashes[i], args.image_ordinal)
                continue
            filename = f'prompt-{i:04d}.png'
            temporary = directory/(filename+'.tmp')
            if choice['reused_final_image']:
                shutil.copyfile(Path(args.source_dir)/filename, temporary)
                if ge.sha256_file(temporary) != source['image_sha256']:
                    raise ValueError('Source final image changed while copying')
            else:
                if decoder is None:
                    with torch.serialization.safe_globals([EasyDict]):
                        decoder = ImageDecoder(SimpleNamespace(ar_path=renderer['ar'],
                            encoder_path=renderer['encoder'], decoder_path=renderer['decoder'],
                            gen_seq_len=729, scale=0, cfg_scale=renderer['cfg']), device)
                    decoder.tok.requires_grad_(False)
                with torch.random.fork_rng(devices=[device]):
                    torch.manual_seed(original['seed'] * 1000003 + i * 7919 + 17)
                    image = decoder.decode_codes([source['semantic_codes'][choice['selected_ordinal']-1]])[0]
                image.save(temporary, format='PNG')
                torch.cuda.empty_cache()
            os.replace(temporary, directory/filename)
            record = dict(source, protocol_id=identity, selection=choice,
                          source_record_sha256=hashes[i], image_sha256=ge.sha256_file(directory/filename))
            if kind == 'geneval2' and not choice['reused_final_image']:
                if reward is None:
                    if not args.reward_python:
                        raise ValueError('GenEval2 requires reward_python')
                    reward = QwenPixelReward(args.reward_python, original['reward_model'], device, 1800)
                am, gm, scores = reward.score_images([image], [rows[i]['vqa_list']])
                record.update(am=am[0], gm=gm[0], question_scores=scores[0])
            ge.atomic_json(directory/f'prompt-{i:04d}.json', record)
            print(json.dumps(dict(index=i, **choice)), flush=True)
    finally:
        if reward is not None:
            reward.close()


if __name__ == '__main__':
    main()
