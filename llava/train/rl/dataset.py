import glob
import json
import os
import random
from typing import Iterator, List

import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import IterableDataset, get_worker_info


class T2IPromptDataset(IterableDataset):
    """Iterable parquet reader that yields only the user prompt text from
    standard Tar T2I conversation rows. The target image (gpt turn) is
    discarded — online on-policy RL samples fresh completions each step.
    """

    def __init__(self, data_paths, shuffle_files: bool = True, infinite: bool = True):
        super().__init__()
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        urls: List[str] = []
        for p in data_paths:
            if p.endswith(".txt"):
                urls.extend(x.strip() for x in open(p) if x.strip())
            elif os.path.isdir(p):
                urls.extend(sorted(glob.glob(os.path.join(p, "*.parquet"))))
            elif p.endswith(".parquet"):
                urls.append(p)
            else:
                raise ValueError(f"Unsupported data path: {p}")
        urls.sort()
        if not urls:
            raise ValueError(f"No parquet files found under: {data_paths}")

        self.urls_all = urls
        self.shuffle_files = shuffle_files
        self.infinite = infinite

    def __iter__(self) -> Iterator[dict]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        total_workers = world_size * num_workers
        global_worker_id = rank * num_workers + worker_id

        urls = self.urls_all[global_worker_id::total_workers]
        if not urls:
            urls = [self.urls_all[global_worker_id % len(self.urls_all)]]

        while True:
            if self.shuffle_files:
                random.shuffle(urls)
            for file_path in urls:
                try:
                    pf = pq.ParquetFile(file_path)
                except Exception as e:
                    print(f"[T2IPromptDataset] failed to open {file_path}: {e}")
                    continue
                for rg in range(pf.num_row_groups):
                    table = pf.read_row_group(rg).to_pandas()
                    for i in range(len(table)):
                        try:
                            yield self._parse_row(table.iloc[i].to_dict())
                        except Exception as e:
                            print(f"[T2IPromptDataset] parse error: {e}")
                            continue
            if not self.infinite:
                return

    @staticmethod
    def _coerce_convs(value):
        if isinstance(value, str):
            return json.loads(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    def _parse_row(self, row: dict) -> dict:
        if "conversations_short" in row and row["conversations_short"] is not None and random.random() < 0.5:
            convs = self._coerce_convs(row["conversations_short"])
        else:
            convs = self._coerce_convs(row["conversations"])

        prompt = None
        for turn in convs:
            from_key = turn.get("from") or turn.get("role")
            value = turn.get("value") or turn.get("content")
            if from_key in ("human", "user"):
                prompt = value
                break
        if not prompt:
            raise ValueError("No user prompt found in row")
        return {"prompt": str(prompt).strip()}


def build_dataset_from_yaml(data_yaml_path: str) -> T2IPromptDataset:
    """Resolve a data YAML in the same format used by the SFT pipeline,
    flattening all listed parquet directories into a single prompt stream.
    """
    with open(data_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    paths: List[str] = []
    for ds in cfg.get("datasets", []):
        jp = ds.get("json_path")
        if jp is None:
            continue
        if isinstance(jp, list):
            paths.extend(jp)
        else:
            paths.append(jp)
    if not paths:
        raise ValueError(f"No data paths in {data_yaml_path}")
    return T2IPromptDataset(paths)


def collate_prompts(batch):
    return {"prompts": [b["prompt"] for b in batch]}
