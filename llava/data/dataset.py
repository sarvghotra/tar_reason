import copy
import glob
import hashlib
import io
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from collections.abc import Iterable, Mapping

import pyarrow.parquet as pq
import torch
import transformers
import yaml
from PIL import Image, ImageFile
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.utils import rank0_print

ImageFile.LOAD_TRUNCATED_IMAGES = True


def derive_seed(*parts) -> int:
    """Derive an independent 64-bit seed from arbitrary identifying parts.

    Seeds here have to separate several dimensions at once — run seed, mixture
    entry, epoch, worker id. Packing those into one integer by addition needs a
    stride per dimension and a documented bound on each, and silently aliases
    once a bound is crossed. Hashing needs neither.
    """
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def preprocess_multimodal(sources: Sequence[str], data_args) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_TOKEN
            # NOTE: only add im_start_end when image generation
            if data_args.mm_use_im_start_end and sentence['from'] == 'gpt':
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    #tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if 'image_token_index' not in globals():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        global image_token_index
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # if has_image:
    #     tokenizer.add_tokens(["<image>"], special_tokens=True)

    # image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)

            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = copy.deepcopy(tokenizer)
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            try:
                with open(data_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                    self.list_data_dict.extend(cur_data_dict)
            except:
                with open(data_path, "r") as file:
                    cur_data_dict = [json.loads(line) for line in file.readlines()]
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                    self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size

        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = "image" in self.list_data_dict[i]
        data_dict = preprocess_qwen(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]), (crop_size["width"], crop_size["height"]), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        return data_dict


class LazyCustomDataset(Dataset):
    """Dataset for JSON files where each record has the format:
    {
        "metadata": {"dataset_name": ..., "sub_dataset_name": ..., "sample_id": ...},
        "data": [
            {"role": "human", "content": [
                {"type": "image", "image": "path/to/img.jpg"},
                {"type": "text",  "text": "<image>\nQuestion text"}
            ]},
            {"role": "gpt", "content": [
                {"type": "image", "image": "path/to/img2.jpg"},
                {"type": "text",  "text": "<image>\nAnswer text"}
            ]}
        ]
    }
    Images are loaded from data_args.image_folder / <image path>.
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazyCustomDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict = []

        if data_path.endswith(".jsonl"):
            with open(data_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.list_data_dict.append(json.loads(line))
        else:
            with open(data_path, "r") as f:
                data = json.load(f)
            self.list_data_dict = data if isinstance(data, list) else [data]

        self.modality = torch.tensor(0) # 0 is for und task, 1 is for gen task

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         has_image = any(
    #             item["type"] == "image"
    #             for turn in sample["data"]
    #             for item in turn["content"]
    #         )
    #         img_tokens = 128 if has_image else 0
    #         text_len = sum(
    #             len(item["text"].split())
    #             for turn in sample["data"]
    #             for item in turn["content"]
    #             if item["type"] == "text"
    #         )
    #         length_list.append(text_len + img_tokens)
    #     return length_list

    # @property
    # def modality_lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         has_image = any(
    #             item["type"] == "image"
    #             for turn in sample["data"]
    #             for item in turn["content"]
    #         )
    #         text_len = sum(
    #             len(item["text"].split())
    #             for turn in sample["data"]
    #             for item in turn["content"]
    #             if item["type"] == "text"
    #         )
    #         assert text_len > 0, f"Conversation length is 0 for sample {sample}"
    #         length_list.append(text_len if has_image else -text_len)
    #     return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn
        image_size = image.size
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        # return image, image_size, "image"
        return image, image_size, self.modality

    def parse_item(self, item):
        """Convert new-format item into (conversations, image_files).

        Each turn's content list may contain interleaved image/text entries.
        Image entries define which files to load (in order); text entries carry
        the conversation text, which already embeds <image> tokens at the right
        positions.  We concatenate all text pieces per turn into a single value
        string so that preprocess_qwen sees the standard {"from", "value"} format.
        """
        conversations = []
        image_files = []

        for turn in item["data"]:
            role = turn["role"]  # "human" or "gpt"
            turn_text = ""
            for c in turn["content"]:
                if c["type"] == "image":
                    image_files.append(c["image"])
                elif c["type"] == "text":
                    turn_text += c["text"]

            conversations.append({"from": role, "value": turn_text})

        return conversations, image_files

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(i)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                return self._get_item(next_index)
            except Exception as e:
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)

        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        raw = copy.deepcopy(self.list_data_dict[i])
        conversations, image_files = self.parse_item(raw)

        has_image = len(image_files) > 0

        if has_image:
            if len(image_files) > 1:
                # image = [self.process_image(f, "pad") for f in image_files]
                # image = [[im[0], im[1], "image"] for im in image]
                image = [self.process_image(f) for f in image_files]
                image = [[im[0], im[1], self.modality] for im in image]
            else:
                image = [self.process_image(image_files[0])]
            sources = preprocess_multimodal([conversations], self.data_args)
        else:
            sources = [conversations]

        data_dict = preprocess_qwen(sources, self.tokenizer, has_image=has_image)

        prompt = data_dict.get("prompt", None)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if has_image:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            # data_dict["image"] = [
            #     (torch.zeros(1, 3, crop_size["height"], crop_size["width"]),
            #      (crop_size["width"], crop_size["height"]),
            #      "text"),
            # ]
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]),
                 (crop_size["width"], crop_size["height"]),
                 self.modality),
            ]

        if prompt is not None:
            data_dict["prompt"] = prompt

        metadata = raw.get("metadata", {})
        data_dict["id"] = metadata.get("sample_id", i)

        return data_dict


class LazyParquetDataset(IterableDataset):
    ITERATIVE_IMG_GEN_PROMPT_PREFIX = "Generate an image iteratively by self-reflecting and correcting.\n"
    NO_CORRECTION_SUFFIX_PROBABILITY = 0.35
    NO_CORRECTION_SUFFIX = (
        "Self-reflect: no issues, it matches the prompt.\n\n"
        "Correction: looks good."
    )
    DEFAULT_INTERLEAVE_SHARDS = 3
    DEFAULT_PARQUET_BATCH_SIZE = 16
    # Shards that ship several conversation variants per image name them
    # 'conversations_1', 'conversations_2', ... instead of 'conversations'.
    CONVERSATION_VARIANT_RE = re.compile(r"^conversations_(\d+)$")

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_paths,
        data_args,
        data_type=None,
        add_prompt_prefix=False,
        no_eos=False,
        add_suffix_no_correction=False,
    ):
        super().__init__()
        # Optional tag identifying the kind of data (e.g. 'understanding', 'text',
        # 't2i'); propagated to each sample so the trainer can log per-type losses.
        self.data_type = data_type
        self.add_prompt_prefix = add_prompt_prefix
        self.add_suffix_no_correction = add_suffix_no_correction
        self.tokenizer = copy.deepcopy(tokenizer)
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.urls_all = []
        for data_path in data_paths:
            if data_path.endswith('.txt'):
                urls = [x.strip() for x in open(data_path).readlines()]
            else:
                urls = glob.glob(os.path.join(data_path, "*.parquet"))
                urls.sort()
            self.urls_all.extend(urls)
        self.urls_all.sort()

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.dataset_seed = int(getattr(data_args, "dataset_seed", 0) or 0)
        # Bumped by the Trainer/Accelerate via set_epoch() in the *parent*
        # process, before workers are forked, so a second epoch does not replay
        # the first one's random decisions.
        self._epoch = 0

        self.modality = torch.tensor(0) # 0 is for und task, 1 is for gen task
        self.no_eos = no_eos
        self.interleave_shards = int(
            getattr(data_args, "parquet_interleave_shards", self.DEFAULT_INTERLEAVE_SHARDS)
        )
        self.parquet_batch_size = int(
            getattr(data_args, "parquet_batch_size", self.DEFAULT_PARQUET_BATCH_SIZE)
        )
        if self.interleave_shards < 1:
            raise ValueError("parquet_interleave_shards must be at least 1")
        if self.parquet_batch_size < 1:
            raise ValueError("parquet_batch_size must be at least 1")

        self._SELF_CORRECTION_RE = re.compile(r"Correction\s*:", re.IGNORECASE)
        # self._NO_ISSUE_RE = re.compile(
        #     r"\b(?:"
        #     r"looks?\s+good(?:\s+as[- ]is)?|"
        #     r"all\s+good|"
        #     r"no\s+(?:issues?|problems?)|"
        #     r"no\s+(?:changes?|corrections?|adjustments?|fix(?:es)?)"
        #     r"(?:\s+(?:are|is))?\s+(?:needed|required|necessary)|"
        #     r"no\s+need\s+(?:for|to)\s+(?:correct|change|adjust|fix)|"
        #     r"nothing\s+(?:to|needs?\s+to\s+be)\s+"
        #     r"(?:corrected|changed|adjusted|fixed)|"
        #     r"keep\s+(?:it|the\s+image)\s+as[- ]is"
        #     r")\b",
        #     re.IGNORECASE,
        # )
        self._NO_ISSUE_RE = re.compile(r"\b(?:looks\s+good|no\s+issues)\b", re.IGNORECASE)

    def set_epoch(self, epoch):
        """Called by Accelerate's dataloader wrapper before each epoch."""
        self._epoch = int(epoch)

    def make_worker_rng(self, worker_id):
        """Build the RNG a DataLoader worker should use for this epoch."""
        return random.Random(derive_seed(self.dataset_seed, self._epoch, worker_id))

    def _conversation_variant_keys(self, sources):
        """Return 'conversations_<n>' keys, ordered by n, for a multi-variant row.

        Empty when the row already carries a plain 'conversations' column, so
        single-conversation shards keep their current behaviour untouched.
        """
        if sources.get("conversations") is not None:
            return []
        numbered = []
        for key in sources:
            match = self.CONVERSATION_VARIANT_RE.match(key)
            if match is not None:
                numbered.append((int(match.group(1)), key))
        numbered.sort()
        return [key for _, key in numbered]

    def _pick_conversation_variant(self, variant_keys, rng):
        return variant_keys[rng.randrange(len(variant_keys))]

    def _select_conversation_variant(self, sources, rng):
        """Collapse a 'conversations_1..n' row down to a single 'conversations'.

        Rows in the multi-caption T2I shards hold several independent
        conversations for the same image; one is drawn per sample so the rest
        of the pipeline sees the single-conversation layout it expects.
        `sources` is mutated in place because subclasses re-inspect the same
        dict after `super()._get_item()` returns.
        """
        variant_keys = self._conversation_variant_keys(sources)
        if not variant_keys:
            return sources

        chosen = self._pick_conversation_variant(variant_keys, rng)
        sources["conversations"] = sources[chosen]
        # Drop the unused variants so nothing downstream re-reads them.
        for key in variant_keys:
            sources.pop(key, None)
        return sources

    def _add_prompt_prefix(self, sources):
        sources = sources.copy()

        for key in ("conversations", "conversations_short"):
            conversations = sources.get(key)
            if conversations is None:
                continue
            if isinstance(conversations, str):
                conversations = json.loads(conversations)
            elif not isinstance(conversations, list):
                conversations = conversations.tolist()
            else:
                conversations = conversations.copy()

            prompt_prefix_added = False
            for index, conversation in enumerate(conversations):
                role = conversation.get("from", conversation.get("role"))
                if role in ("human", "user") and not prompt_prefix_added:
                    conversation = conversation.copy()
                    content_key = "value" if "value" in conversation else "content"
                    conversation[content_key] = (
                        self.ITERATIVE_IMG_GEN_PROMPT_PREFIX + conversation[content_key]
                    )
                    conversations[index] = conversation
                    prompt_prefix_added = True
                elif role in ("gpt", "assistant") and isinstance(conversation.get("value"), str):
                    conversation = conversation.copy()
                    conversation["value"] = conversation["value"].replace(
                        "Self-correction", "Correction"
                    )
                    conversations[index] = conversation

            sources[key] = conversations

        return sources

    def _add_no_correction_suffix(self, sources):
        """Append the no_correction response to the last assistant turn."""
        sources = sources.copy()

        for key in ("conversations", "conversations_short"):
            conversations = sources.get(key)
            if conversations is None:
                continue
            if isinstance(conversations, str):
                conversations = json.loads(conversations)
            elif not isinstance(conversations, list):
                conversations = conversations.tolist()
            else:
                conversations = conversations.copy()

            for index in range(len(conversations) - 1, -1, -1):
                conversation = conversations[index]
                if not isinstance(conversation, dict):
                    continue
                role = conversation.get("from", conversation.get("role"))
                if role not in ("gpt", "assistant"):
                    continue
                content_key = "value" if "value" in conversation else "content"
                content = conversation.get(content_key)
                if not isinstance(content, str):
                    continue

                conversation = conversation.copy()
                separator = "" if content.endswith("\n") else "\n"
                conversation[content_key] = (
                    content + separator + self.NO_CORRECTION_SUFFIX
                )
                conversations[index] = conversation
                break

            sources[key] = conversations

        return sources

    def _correction_needed(self, conversations: object) -> bool:
        """Whether a GPT turn's self-correction contains a correction request. "Correction: looks good." is not present"""
        if isinstance(conversations, dict):
            conversations = conversations.get("conversations")
        if isinstance(conversations, str):
            try:
                conversations = json.loads(conversations)
            except (TypeError, ValueError):
                return False
        elif hasattr(conversations, "tolist"):
            conversations = conversations.tolist()

        if not isinstance(conversations, list):
            return False

        stack = conversations.copy()
        while stack:
            turn = stack.pop()
            if isinstance(turn, list):
                stack.extend(turn)
                continue
            if not isinstance(turn, dict):
                continue

            role = turn.get("from", turn.get("role"))
            if role not in ("gpt", "assistant"):
                continue
            text = turn.get("value", turn.get("content"))
            if not isinstance(text, str):
                continue

            marker = self._SELF_CORRECTION_RE.search(text)
            if marker is None:
                continue
            correction = text[marker.end():].strip()
            if correction and \
                self._NO_ISSUE_RE.search(correction) is not None:   # Found "Correction: looks good"
                return False

        return True

    def process_image(self, image):
        processor = self.data_args.image_processor
        image_size = image.size
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, self.modality

    def _iter_parquet_rows(self, file_path):
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=self.parquet_batch_size):
            table = batch.to_pandas()
            for i in range(len(table)):
                yield table.iloc[i].to_dict()

    def _interleave_parquet_rows(self, file_paths, rng):
        """Interleave rows from a bounded window of parquet shards.

        Each active iterator retains at most one Arrow/Pandas batch. With the
        defaults this keeps at most 3 * 16 raw rows resident per worker, while
        ensuring that rows from every active shard are consumed each round.

        `rng` is required: silently falling back to the `random` module would
        reintroduce the shared-across-workers state this seeding exists to
        avoid, and would do so without any visible symptom.
        """
        pending_files = iter(file_paths)
        active = []
        for _ in range(min(self.interleave_shards, len(file_paths))):
            active.append(self._iter_parquet_rows(next(pending_files)))

        while active:
            # Randomize shard order each round, but consume exactly one row from
            # each active shard to avoid long same-shard streaks.
            rng.shuffle(active)
            next_active = []
            for shard_rows in active:
                try:
                    sample = next(shard_rows)
                except StopIteration:
                    try:
                        file_path = next(pending_files)
                    except StopIteration:
                        continue
                    next_active.append(self._iter_parquet_rows(file_path))
                    continue

                next_active.append(shard_rows)
                yield sample
            active = next_active

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Hugging Face Trainer prepares IterableDatasets with Accelerate, which
        # already partitions the sample stream across distributed processes.
        # Partitioning by RANK here as well would shard the data twice. Only
        # split files among this process's DataLoader workers; Accelerate owns
        # the process-level split.
        if not self.urls_all:
            raise RuntimeError("No parquet files found for training")
        # Per-worker seed, so each worker walks its own stream instead of
        # replaying the state it inherited from the forked parent.
        rng = self.make_worker_rng(worker_id)
        urls = list(self.urls_all)
        if len(urls) < num_workers:
            urls.extend(urls[i % len(urls)] for i in range(num_workers - len(urls)))

        files_iter = urls[worker_id::num_workers]

        while True:
            rng.shuffle(files_iter)

            for sample in self._interleave_parquet_rows(files_iter, rng=rng):
                try:
                    yield self._get_item(sample, rng=rng)
                except Exception as e:
                    print(e)
                    # print(sample)
                    continue

    def parse_item(self, sources, rng):
        # parse conversations
        if isinstance(sources['conversations'], str):
            sources['conversations'] = json.loads(sources['conversations'])
        elif not isinstance(sources['conversations'], list):
            sources['conversations'] = sources['conversations'].tolist()

        # random short-long prompt
        if 'conversations_short' in sources and rng.random() < 0.5:
            sources['conversations'] = sources['conversations_short']

        # parse image
        if 'image' in sources and sources['image'] is not None:
            if isinstance(sources['image'], bytes):
                image = sources['image']
                sources['image'] = Image.open(io.BytesIO(image)).convert('RGB')
            # single image
            elif 'bytes' in sources['image']:
                image = sources['image']['bytes']
                sources['image'] = Image.open(io.BytesIO(image)).convert('RGB')
            # multiple images
            else:
                images = [s['bytes'] for s in sources['image']]
                sources['image'] = [Image.open(io.BytesIO(image)).convert('RGB') for image in images]
        return sources

    def _get_item(self, sources, rng, remove_eos=True):
        sources = self._select_conversation_variant(sources, rng)
        if self.add_prompt_prefix:
            sources = self._add_prompt_prefix(sources)
        if self.add_suffix_no_correction:
            add_suffix = (
                self.NO_CORRECTION_SUFFIX_PROBABILITY >= 1.0
                or rng.random() < self.NO_CORRECTION_SUFFIX_PROBABILITY
            )
            if add_suffix:
                sources = self._add_no_correction_suffix(sources)

        sources = self.parse_item(copy.deepcopy(sources), rng=rng)
        has_image = "image" in sources and sources["image"] is not None
        # id = sources.get('id', '0')
        if has_image:
            image_file = sources["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                if len(image_file) > 1:
                    image = [self.process_image(f) for f in image_file]
                    image = [[im[0], im[1], self.modality] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal([sources["conversations"]], self.data_args)
        else:
            sources = copy.deepcopy([sources["conversations"]])

        data_dict = preprocess_qwen(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if has_image:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (torch.zeros(1, 3, crop_size["height"], crop_size["width"]),
                torch.tensor([crop_size["width"], crop_size["height"]]),
                self.modality),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = "0"

        # Remove EOS token
        if self.no_eos and \
            self.tokenizer.eos_token_id is not None and \
            remove_eos and \
            self._correction_needed(sources):
            eos_positions = torch.where(
                data_dict["input_ids"] == self.tokenizer.eos_token_id
            )[0]
            if eos_positions.numel():
                sequence_end = eos_positions[-1].item()
                if len(data_dict["input_ids"]) > sequence_end:
                    data_dict["input_ids"] = torch.cat((data_dict["input_ids"][:sequence_end], data_dict["input_ids"][sequence_end + 1:]))
                    data_dict["labels"] = torch.cat((data_dict["labels"][:sequence_end], data_dict["labels"][sequence_end + 1:]))
                else:
                    data_dict["input_ids"] = data_dict["input_ids"][:sequence_end]
                    data_dict["labels"] = data_dict["labels"][:sequence_end]

        if self.data_type is not None:
            data_dict["data_type"] = self.data_type

        return data_dict


class LazySelfReflectParquetDataset(LazyParquetDataset):
    """SFT dataset where the assistant response starts with '<image>\\nSelf-reflect: <answer>'.
    Loss is computed on the text response, masking only the leading image tokens."""

    # TODO: change "Self-correction" to "Correction" because the tokenizer breaks "Self-correction" into Self -cor rection"
    # Text following <image> in the assistant response that should be masked from loss
    MASK_PREFIX = "\nSelf-reflect:"

    def __init__(
        self,
        tokenizer,
        data_paths,
        data_args,
        data_type=None,
        add_prompt_prefix=False,
        no_eos=False,
        add_suffix_no_correction=False,
    ):
        super().__init__(
            tokenizer,
            data_paths,
            data_args,
            data_type=data_type,
            add_prompt_prefix=add_prompt_prefix,
            no_eos=no_eos,
            add_suffix_no_correction=add_suffix_no_correction,
        )
        self.no_eos = no_eos

    def _get_item(self, sources, rng, remove_eos=True):
        data_dict = super()._get_item(sources, rng, remove_eos=False)
        data_dict["labels"] = self._mask_assistant_prefix(
            data_dict["input_ids"], data_dict["labels"]
        )

        if self.no_eos and self.tokenizer.eos_token_id is not None and self._correction_needed(sources):
            eos_positions = torch.where(
                data_dict["input_ids"] == self.tokenizer.eos_token_id
            )[0]
            if eos_positions.numel():
                sequence_end = eos_positions[-1].item()
                if len(data_dict["input_ids"]) > sequence_end:
                    data_dict["input_ids"] = torch.cat((data_dict["input_ids"][:sequence_end], data_dict["input_ids"][sequence_end + 1:]))
                    data_dict["labels"] = torch.cat((data_dict["labels"][:sequence_end], data_dict["labels"][sequence_end + 1:]))
                else:
                    data_dict["input_ids"] = data_dict["input_ids"][:sequence_end]
                    data_dict["labels"] = data_dict["labels"][:sequence_end]

        if self.data_type is not None:
            data_dict["data_type"] = self.data_type

        return data_dict

    def _mask_assistant_prefix(self, input_ids, labels):
        """Set IGNORE_INDEX on '<im_start><image><im_end>' tokens in the assistant turn.

        preprocess_multimodal replaces <image> with <im_start><image><im_end> for gpt turns
        when mm_use_im_start_end is set, so both forms are handled here.
        """
        im_start_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        im_end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)

        inp = input_ids.tolist()
        lbl = labels.tolist()

        for i, tok in enumerate(inp):
            if tok != IMAGE_TOKEN_INDEX:
                continue
            if lbl[i] == IGNORE_INDEX:
                # <image> is in a human/system turn — not what we want
                continue
            # IMAGE_TOKEN_INDEX in the assistant turn
            # Include preceding <im_start> in the mask if present
            mask_start = i - 1 if (i > 0 and inp[i - 1] == im_start_id) else i
            # Skip trailing <im_end> before checking for the text prefix
            after_img = i + 1
            if after_img < len(inp) and inp[after_img] == im_end_id:
                after_img += 1
            for j in range(mask_start, after_img):
                lbl[j] = IGNORE_INDEX
            break  # Only process the first match in the assistant turn

        return torch.tensor(lbl, dtype=labels.dtype)


class LazyOnlySelfReflectParquetDataset(LazySelfReflectParquetDataset):
    """Train only on the reflection/correction text in image-edit responses.

    These samples contain both the image being critiqued and the corrected
    image, for example ``<image>\nSelf-reflect: ...\nCorrection: ...\n<image>``.
    Mask every assistant-side image span so neither image contributes to the
    language-model loss while the text between them remains supervised.
    """

    def _mask_assistant_prefix(self, input_ids, labels):
        """Set ``IGNORE_INDEX`` on every assistant-side image span."""
        im_start_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        im_end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)

        inp = input_ids.tolist()
        masked_labels = labels.clone()

        for i, token_id in enumerate(inp):
            if token_id != IMAGE_TOKEN_INDEX:
                continue
            if masked_labels[i].item() == IGNORE_INDEX:
                # The image belongs to a human/system turn, which is already
                # excluded from the loss by preprocess_qwen.
                continue

            mask_start = i - 1 if i > 0 and inp[i - 1] == im_start_id else i
            mask_end = i + 1
            if mask_end < len(inp) and inp[mask_end] == im_end_id:
                mask_end += 1
            masked_labels[mask_start:mask_end] = IGNORE_INDEX

        return masked_labels


class LazyCorrectionParquetDataset(LazyParquetDataset):
    """Re-use iterative image-generation data as image-editing data.

    The source rows hold two images and a single assistant turn of the form
    ``<image>\nSelf-reflect: ...\nCorrection: <instruction>\n<image>``, i.e. a
    first draft, a critique of it and the corrected image. Rewritten here as a
    plain edit pair::

        human: <image> Correction: <instruction>
        gpt:   <image>

    The first image stays on the human side as the image to edit and the second
    becomes the supervised target, so the loss covers only the edited image.

    With probability ``NO_CHANGE_PROB`` the row is instead turned into a
    no-change sample: the already-corrected image (``images[1]``) is used on
    both sides, so the model learns to copy the input unchanged when the
    requested edit is already present in it.
    """

    # Both spellings occur in the source shards ('Self-correction:' in
    # HumanEdit, 'Correction:' in the gpt-edit shards).
    CORRECTION_MARKER_RE = re.compile(r"(?:Self-)?Correction\s*:", re.IGNORECASE)
    HUMAN_PROMPT_TEMPLATE = DEFAULT_IMAGE_TOKEN + " Correction: {correction}"
    # Fraction of samples rewritten as 'edit already applied -> keep as is'.
    NO_CHANGE_PROB = 0.1

    def _extract_correction(self, conversations):
        """Return the correction instruction from the assistant turn."""
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            if turn.get("from", turn.get("role")) not in ("gpt", "assistant"):
                continue
            text = turn.get("value", turn.get("content"))
            if not isinstance(text, str):
                continue
            marker = self.CORRECTION_MARKER_RE.search(text)
            if marker is None:
                continue
            correction = text[marker.end():]
            # Drop the trailing target image token (and anything after it).
            correction = correction.split(DEFAULT_IMAGE_TOKEN)[0].strip()
            if correction:
                return correction
        return None

    def parse_item(self, sources, rng):
        sources = super().parse_item(sources, rng)

        images = sources.get("image")
        if not isinstance(images, list) or len(images) != 2:
            raise ValueError(
                f"correction_parquet expects exactly 2 images, got "
                f"{0 if images is None else (len(images) if isinstance(images, list) else 1)}"
            )

        correction = self._extract_correction(sources["conversations"])
        if correction is None:
            raise ValueError("correction_parquet sample has no correction instruction")
        if self._NO_ISSUE_RE.search(correction) is not None:
            # 'Correction: looks good' — nothing to edit, so there is no
            # image-editing supervision in this row.
            raise ValueError("correction_parquet sample requires no correction")

        if self.NO_CHANGE_PROB > 0 and rng.random() < self.NO_CHANGE_PROB:
            # The correction is already applied in images[1]; ask for the same
            # edit on it so the target is identical to the input image.
            sources["image"] = [images[1], images[1]]

        sources["conversations"] = [
            {
                "from": "human",
                "value": self.HUMAN_PROMPT_TEMPLATE.format(correction=correction),
            },
            {"from": "gpt", "value": DEFAULT_IMAGE_TOKEN},
        ]
        return sources


class FiniteParquetDatasetMixin:
    """Iterate over validation parquet rows exactly once across all workers."""

    NO_CORRECTION_SUFFIX_PROBABILITY = 1.0

    def set_epoch(self, epoch):
        # Validation must stay byte-identical between evals to be comparable,
        # so it stays pinned to epoch 0 while training re-randomises.
        return

    def _pick_conversation_variant(self, variant_keys, rng):
        # Same reason as the short-prompt pop below: validation must stay
        # byte-identical between evals, so it always takes the first variant.
        return variant_keys[0]

    def parse_item(self, sources, rng):
        # Training randomly alternates between long and short prompts. Keep
        # validation deterministic by always using the primary conversation.
        sources = sources.copy()
        sources.pop("conversations_short", None)
        return super().parse_item(sources, rng)

    def _iter_parquet_rows(self, file_path):
        parquet_file = pq.ParquetFile(file_path)
        row_group_id = getattr(self, "_val_row_group_id", 0)
        row_group_count = getattr(self, "_val_row_group_count", 1)
        row_groups = range(
            row_group_id, parquet_file.num_row_groups, row_group_count
        )
        for batch in parquet_file.iter_batches(
            batch_size=self.parquet_batch_size, row_groups=row_groups
        ):
            table = batch.to_pandas()
            for i in range(len(table)):
                yield table.iloc[i].to_dict()

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if not self.urls_all:
            raise RuntimeError("No parquet files found for validation")

        # Accelerate shards the resulting iterable across distributed
        # processes. Divide work only among local DataLoader workers here to
        # avoid dropping validation rows through a second rank-level shard.
        rng = self.make_worker_rng(worker_id)
        self._val_row_group_id = 0
        self._val_row_group_count = 1
        if len(self.urls_all) < num_workers:
            file_count = len(self.urls_all)
            file_id = worker_id % file_count
            files_iter = [self.urls_all[file_id]]
            self._val_row_group_id = worker_id // file_count
            self._val_row_group_count = (
                (num_workers - 1 - file_id) // file_count
            ) + 1
        else:
            files_iter = self.urls_all[worker_id::num_workers]

        for sample in self._interleave_parquet_rows(files_iter, rng=rng):
            try:
                yield self._get_item(sample, rng=rng)
            except Exception as e:
                print(e)


class LazyParquetValDataset(FiniteParquetDatasetMixin, LazyParquetDataset):
    pass


class LazySelfReflectParquetValDataset(
    FiniteParquetDatasetMixin, LazySelfReflectParquetDataset
):
    pass


class LazyOnlySelfReflectParquetValDataset(
    FiniteParquetDatasetMixin, LazyOnlySelfReflectParquetDataset
):
    pass


class LazyCorrectionParquetValDataset(
    FiniteParquetDatasetMixin, LazyCorrectionParquetDataset
):
    # Validation stays edit-only so it is comparable across evals.
    NO_CHANGE_PROB = 0.0


def _assert_finite_eval_cls(name, dataset_cls, data_path):
    """Reject streaming dataset classes on the validation side.

    `LazyParquetDataset.__iter__` re-cycles its files forever, so a validation
    entry naming a non-`_val` class makes the eval dataloader infinite: the
    trainer drains it until the walltime runs out, never reaching step 1 and
    never logging. The `_val` classes mix in `FiniteParquetDatasetMixin`, which
    drops that outer loop. Non-parquet classes are map-style and always finite.
    """
    if issubclass(dataset_cls, LazyParquetDataset) and not issubclass(
        dataset_cls, FiniteParquetDatasetMixin
    ):
        raise ValueError(
            f"Validation dataset '{name}' in {data_path} uses the streaming "
            f"class {dataset_cls.__name__}, which never raises StopIteration. "
            f"Use '{name}_val' instead."
        )


class WeightedDataset(IterableDataset):
    def __init__(self, tokenizer, data_path, data_args, is_eval=False):
        super().__init__()
        # Validation must stay byte-identical across runs to be comparable, so
        # it ignores the run-level seed and keeps the historical seeding.
        self.is_eval = is_eval
        self._epoch = 0
        with open(data_path, "r") as file:
            yaml_data = yaml.safe_load(file)
            datasets = yaml_data.get("datasets")
            self.datasets = []
            self.ratios = []
            run_seed = 0 if is_eval else int(getattr(data_args, "dataset_seed", 0) or 0)
            self.dataset_seed = run_seed
            for idx, dataset in enumerate(datasets):
                dataset_name = dataset.get('name', 'parquet')
                dataset_cls = get_dataset_cls(dataset_name)
                if is_eval:
                    _assert_finite_eval_cls(dataset_name, dataset_cls, data_path)
                ratio = dataset.get('ratio', 1)
                data_type = dataset.get('data_type', None)
                extra_kwargs = {}
                if issubclass(dataset_cls, LazyParquetDataset):
                    extra_kwargs['data_type'] = data_type
                    extra_kwargs['add_prompt_prefix'] = dataset.get('add_prompt_prefix', False)
                    extra_kwargs['no_eos'] = dataset.get('no_eos', False)
                    extra_kwargs['add_suffix_no_correction'] = dataset.get(
                        'add_suffix_no_correction', False
                    )
                # if issubclass(dataset_cls, LazySelfReflectParquetDataset):

                dataset = dataset_cls(tokenizer, dataset.get('json_path'), data_args, **extra_kwargs)
                # Sub-datasets read `data_args.dataset_seed` themselves, so
                # overwrite it with a per-entry value: without the `idx` offset
                # every entry walks the same stream, so two entries on the same
                # json_path would emit identical rows in lockstep and `ratio`
                # would only change how fast each is consumed, never what it
                # yields. For eval the run seed is dropped so validation content
                # stays comparable across runs that vary --dataset_seed.
                dataset.dataset_seed = derive_seed(self.dataset_seed, idx)
                rank0_print(
                    f"Loading dataset: {dataset} "
                    f"(seed {getattr(dataset, 'dataset_seed', None)})"
                )
                self.datasets.append(dataset)
                self.ratios.append(ratio)

    def set_epoch(self, epoch):
        self._epoch = int(epoch)
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # Sub-datasets use derive_seed(self.dataset_seed, idx), so the raw
        # run seed used here cannot collide with any of them.
        rng = random.Random(derive_seed(self.dataset_seed, self._epoch, worker_id))
        iterators = [iter(dataset) for dataset in self.datasets]
        ratios = {it: r for r, it in zip(self.ratios, iterators)}

        while True:
            it = rng.choices(iterators, weights=ratios.values())[0]
            try:
                yield next(it)
            except StopIteration:
                iterators.remove(it)
                del ratios[it]
                if not iterators:
                    return


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        if "data_type" in instances[0]:
            batch["data_types"] = [instance.get("data_type") for instance in instances]
        return batch

def get_dataset_cls(name):
    if name == 'llava':
        dataset_cls = LazySupervisedDataset
    elif name == 'custom':
        dataset_cls = LazyCustomDataset
    elif name == 'parquet':
        dataset_cls = LazyParquetDataset
    elif name == 'parquet_val':
        dataset_cls = LazyParquetValDataset
    elif name == 'self_reflect_parquet':
        dataset_cls = LazySelfReflectParquetDataset
    elif name == 'self_reflect_parquet_val':
        dataset_cls = LazySelfReflectParquetValDataset
    elif name == 'only_self_reflect_parquet':
        dataset_cls = LazyOnlySelfReflectParquetDataset
    elif name == 'only_self_reflect_parquet_val':
        dataset_cls = LazyOnlySelfReflectParquetValDataset
    elif name == 'correction_parquet':
        dataset_cls = LazyCorrectionParquetDataset
    elif name == 'correction_parquet_val':
        dataset_cls = LazyCorrectionParquetValDataset
    elif name == 'weighted_parquet':
        dataset_cls = WeightedDataset
    else:
        raise ValueError(f'Unknown dataset class {name}')
    return dataset_cls

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = get_dataset_cls(data_args.dataset_cls)
    train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    eval_dataset = None
    eval_data_path = getattr(data_args, "eval_data_path", None)
    if eval_data_path:
        # Only the weighted mixture takes `is_eval`; it uses the flag to pin the
        # validation seed so eval loss stays comparable across runs that vary
        # `--dataset_seed`.
        if dataset_cls is WeightedDataset:
            eval_kwargs = {"is_eval": True}
        else:
            eval_kwargs = {}
            _assert_finite_eval_cls(
                data_args.dataset_cls, dataset_cls, eval_data_path
            )
        eval_dataset = dataset_cls(
            tokenizer=tokenizer,
            data_path=eval_data_path,
            data_args=data_args,
            **eval_kwargs,
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
