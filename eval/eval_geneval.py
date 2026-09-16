import argparse
import json
import os
import re
from functools import partial

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from llava.model.builder import load_pretrained_model
from eval.eval_dpg_bench import get_prompt_template, load_visual_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts', type=str, default="../geneval/prompts/evaluation_metadata.jsonl")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument("--ar_path", type=str, default=None)
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--decoder_path", type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=729)
    parser.add_argument('--seq_scale', type=int, default=1)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--repeat', type=int, default=4)
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Number of prompts generated per model call (effective batch is batch_size * repeat).')
    parser.add_argument(
        '--decode_batch_size', type=int, default=4,
        help='Maximum number of generated images decoded at once.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attn', type=str)
    parser.add_argument('--geneval2', action='store_true', default=False)
    return parser.parse_args()


class GenEvalDataset(Dataset):
    def __init__(self, args, tokenizer):
        with open(args.prompts) as prompt_file:
            self.prompts = [line.strip() for line in prompt_file]
        prompt_temp = get_prompt_template(args)
        self.all_input_ids = []
        for prompt in self.prompts:
            question = prompt_temp.format(json.loads(prompt)['prompt'])
            input_ids = tokenizer_image_token(
                question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            self.all_input_ids.append(input_ids)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.prompts[idx], idx


def collate_prompts(batch, pad_token_id):
    """Left-pad variable-length prompts for decoder-only batched generation."""
    input_ids, metadata, indices = zip(*batch)
    max_length = max(ids.numel() for ids in input_ids)
    padded_ids = torch.full(
        (len(input_ids), max_length), pad_token_id, dtype=input_ids[0].dtype)
    attention_mask = torch.zeros(
        (len(input_ids), max_length), dtype=torch.bool)

    for row, ids in enumerate(input_ids):
        length = ids.numel()
        padded_ids[row, -length:] = ids
        attention_mask[row, -length:] = True

    return padded_ids, attention_mask, metadata, torch.tensor(indices)


def get_image_token_range(tokenizer, model):
    """Return the contiguous image-token range, or None for legacy tokenizers."""
    start = tokenizer.convert_tokens_to_ids('<I0>')
    num_tokens = getattr(model.config, 'num_image_tokens', None)
    if not isinstance(start, int) or start == tokenizer.unk_token_id:
        return None

    if not isinstance(num_tokens, int) or num_tokens <= 0:
        end = getattr(model.config, 'image_end_token_id', None)
        if isinstance(end, int) and end >= start:
            num_tokens = end - start + 1
        else:
            return None

    last = tokenizer.convert_tokens_to_ids(f'<I{num_tokens - 1}>')
    if last != start + num_tokens - 1:
        return None
    return start, num_tokens


def extract_image_codes(generated_ids, seq_len, image_token_range, tokenizer):
    """Extract visual codes without decoding the whole generated batch to text."""
    if image_token_range is None:
        text_outputs = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False)
        codes = []
        for text_output in text_outputs:
            code = [int(x) for x in re.findall(r'<I(\d+)>', text_output)]
            codes.append((code + [0] * seq_len)[:seq_len])
        return torch.tensor(codes, dtype=torch.long)

    image_start_id, num_image_tokens = image_token_range
    generated_ids = generated_ids.detach().cpu()
    codes = torch.zeros(
        (generated_ids.shape[0], seq_len), dtype=torch.long)
    for row, token_ids in enumerate(generated_ids):
        mask = ((token_ids >= image_start_id)
                & (token_ids < image_start_id + num_image_tokens))
        row_codes = token_ids[mask][:seq_len] - image_start_id
        codes[row, :row_codes.numel()] = row_codes
    return codes


if __name__ == '__main__':
    # distribtued init
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    args = parse_args()
    if args.batch_size < 1 or args.decode_batch_size < 1:
        raise ValueError('batch_size and decode_batch_size must both be positive')
    torch.manual_seed(args.seed)
    if args.geneval2:
        assert args.repeat == 1, "For geneval2, not sure how to eval for more than one image per prompt"
    # load visual tokenizer
    args.ar_path = args.ar_path or hf_hub_download(
        "csuhan/TA-Tok", "ar_dtok_lp_256px.pth"
    )
    args.encoder_path = args.encoder_path or hf_hub_download(
        "csuhan/TA-Tok", "ta_tok.pth"
    )
    args.decoder_path = args.decoder_path or hf_hub_download(
        "peizesun/llamagen_t2i", "vq_ds16_t2i.pt"
    )
    visual_tokenizer = load_visual_tokenizer(args).to(device)

    tokenizer, model, _, _ = load_pretrained_model(args.model, None, 'llava_qwen', device_map=device, multimodal=True, attn_implementation=args.attn)
    model.eval().to(device=device, dtype=dtype)

    dataset = GenEvalDataset(args, tokenizer)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=partial(collate_prompts, pad_token_id=pad_token_id),
    )

    seq_len = args.seq_len
    image_token_range = get_image_token_range(tokenizer, model)
    args.save_dir = os.path.abspath(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    geneval2_results = {}
    progress = tqdm(dataloader, disable=rank != 0)
    for input_ids, attention_mask, meta_data, indices in progress:
        input_ids = input_ids.repeat_interleave(args.repeat, dim=0)
        attention_mask = attention_mask.repeat_interleave(args.repeat, dim=0)

        with autocast(dtype=model.dtype):
            cont = model.generate(
                input_ids.to(device, non_blocking=True),
                attention_mask=attention_mask.to(device, non_blocking=True),
                images=None,
                do_sample=True, temperature=1.0,
                max_new_tokens=seq_len,
                pad_token_id=pad_token_id)

        codes = extract_image_codes(
            cont, seq_len, image_token_range, tokenizer)

        for start in range(0, codes.shape[0], args.decode_batch_size):
            end = min(start + args.decode_batch_size, codes.shape[0])
            recs = visual_tokenizer.decode_from_encoder_indices(
                codes[start:end].to(device, non_blocking=True),
                {'cfg_scale': 4.0})

            for offset, rec in enumerate(recs, start=start):
                prompt_offset, sample_idx = divmod(offset, args.repeat)
                idx = indices[prompt_offset].item()
                save_path = os.path.join(
                    args.save_dir, f'{idx:05d}', 'samples',
                    f'{sample_idx:05d}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(rec.numpy()).save(save_path)

        for metadata, idx_tensor in zip(meta_data, indices):
            idx = idx_tensor.item()
            meta_save_path = os.path.join(
                args.save_dir, f'{idx:05d}', 'metadata.jsonl')
            with open(meta_save_path, 'w') as f:
                f.write(metadata)

            if args.geneval2:
                prompt_text = json.loads(metadata)['prompt']
                geneval2_results[prompt_text] = os.path.join(
                    args.save_dir, f'{idx:05d}', 'samples', '00000.png')

    dist.barrier()
    if args.geneval2:
        all_results = [None] * world_size
        dist.all_gather_object(all_results, geneval2_results)
        if rank == 0:
            merged = {}
            for r in all_results:
                merged.update(r)
            jsonl_path = os.path.join(args.save_dir, "geneval2_results.json")
            with open(jsonl_path, 'w') as f:
                json.dump(merged, f, indent=2)
    dist.destroy_process_group()
