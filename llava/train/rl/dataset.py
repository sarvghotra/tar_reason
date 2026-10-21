"""Prompt + VQA-question dataset for GRPO on iterative image generation.

Reads the GenEval2-style JSONL files referenced by a data YAML
(``datasets[*].json_path``), e.g. ``output_dir/rl_ft/data.yaml``::

    datasets:
        - json_path:
          - /path/to/evaluation_metadata_shuf_train.jsonl
          ratio: 1

Each JSONL row is ``{"prompt": str, "vqa_list": [[question, answer], ...],
"skills": [...], "atom_count": int}``. Only ``prompt``, ``vqa_list`` and
``atom_count`` are used. The per-dataset ``ratio`` is ignored: every row of
every listed file is pooled into one shuffled stream. Each row is tagged with
its dataset's ``logname`` (default: the first file's stem) so validation can
report every listed set separately (``rank_groups``).
"""

import json
import os
import random
from typing import Dict, Iterator, List, Optional

import yaml


class GenEval2PromptDataset:
    """Deterministically shuffled, rank-sharded prompt stream with resume support.

    Sample ``i`` of epoch ``e`` on rank ``r`` is fixed by ``(seed, e, r, i)`` so
    a resumed job (``skip(n)``) replays exactly the prompts it would have seen.
    """

    def __init__(self, yaml_path: str, seed: int = 0, rank: int = 0,
                 world_size: int = 1, max_atoms: Optional[int] = None):
        self.rows = self._load_rows(yaml_path)
        if max_atoms is not None:
            self.rows = [r for r in self.rows if r["atom_count"] <= max_atoms]
        if not self.rows:
            raise ValueError(f"No prompts loaded from {yaml_path}")
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    @staticmethod
    def _load_rows(yaml_path: str) -> List[Dict]:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        paths: List[tuple] = []   # (path, set name)
        for ds in cfg.get("datasets", []):
            jp = ds.get("json_path")
            if jp is None:
                continue
            jp = jp if isinstance(jp, list) else [jp]
            name = ds.get("logname") or os.path.splitext(os.path.basename(jp[0]))[0]
            paths.extend((path, str(name)) for path in jp)
        if not paths:
            raise ValueError(f"No json_path entries in {yaml_path}")

        rows = []
        per_set_idx: Dict[str, int] = {}   # row position within its set (bank lookups)
        for path, name in paths:
            path = os.path.expanduser(path)
            if not path.endswith(".jsonl"):
                raise ValueError(f"Expected a .jsonl prompt file, got {path}")
            with open(path) as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "prompt" not in obj or "vqa_list" not in obj:
                        raise ValueError(
                            f"{path}:{line_number}: row needs 'prompt' and 'vqa_list'")
                    vqa_list = [tuple(qa) for qa in obj["vqa_list"]]
                    if not vqa_list:
                        continue
                    # Per-question skill tags; the oracle critique phrases a
                    # fix by skill. Older files lack them.
                    skills = list(obj.get("skills") or ["unknown"] * len(vqa_list))
                    if len(skills) != len(vqa_list):
                        raise ValueError(f"{path}:{line_number}: {len(skills)} skills for "
                                         f"{len(vqa_list)} questions")
                    rows.append({
                        "prompt": str(obj["prompt"]).strip(),
                        "vqa_list": vqa_list,
                        "skills": skills,
                        "atom_count": int(obj.get("atom_count", len(vqa_list))),
                        "set": name,
                        "idx": per_set_idx.setdefault(name, 0),
                    })
                    per_set_idx[name] += 1
        return rows

    def __len__(self):
        return len(self.rows)

    def epoch_order(self, epoch: int) -> List[int]:
        order = list(range(len(self.rows)))
        random.Random(self.seed * 1_000_003 + epoch).shuffle(order)
        return order[self.rank::self.world_size]

    def per_epoch(self) -> int:
        return len(self.epoch_order(0))

    def iterate(self, batch_size: int, skip_batches: int = 0) -> Iterator[List[Dict]]:
        """Infinite stream of ``batch_size``-row batches for this rank.

        ``skip_batches`` fast-forwards past batches already consumed (resume).
        Incomplete trailing batches of an epoch are dropped so every rank
        yields the same number of batches per epoch.
        """
        per_epoch = self.per_epoch() // batch_size
        if per_epoch == 0:
            raise ValueError("batch_size larger than this rank's share of the data")
        epoch, offset = divmod(skip_batches, per_epoch)
        while True:
            order = self.epoch_order(epoch)
            for b in range(offset, per_epoch):
                idx = order[b * batch_size:(b + 1) * batch_size]
                yield [self.rows[i] for i in idx]
            epoch += 1
            offset = 0

    def all_rows(self) -> List[Dict]:
        """This rank's rows in file order (used for validation)."""
        return self.rows[self.rank::self.world_size]

    def rank_groups(self) -> Dict[str, List[Dict]]:
        """This rank's rows per set, in file order (used for validation).

        Each set is sharded on its own so every rank gets a share of every set
        (the per-set stats are all-reduced, so no rank may be left out)."""
        groups: Dict[str, List[Dict]] = {}
        for row in self.rows:
            groups.setdefault(row["set"], []).append(row)
        for name, rows in groups.items():
            if len(rows) < self.world_size:
                raise ValueError(f"val set '{name}' has {len(rows)} prompts, fewer than "
                                 f"world_size={self.world_size}")
        return {name: rows[self.rank::self.world_size] for name, rows in groups.items()}
