"""Check launcher inputs without loading model tensors or starting GPU work."""
import json
import os
import sys
from pathlib import Path

import yaml


def check_path(path):
    path = Path(path)
    if not path.exists() or not os.access(path, os.R_OK):
        raise ValueError(f"Missing or unreadable input: {path}")
    if path.name == "config.json":
        indexes = list(path.parent.glob("*.index.json"))
        weights = set()
        for index in indexes:
            with index.open() as stream:
                weights.update(json.load(stream).get("weight_map", {}).values())
        if not weights:
            weights.update(p.name for pattern in ("*.safetensors", "pytorch_model*.bin")
                           for p in path.parent.glob(pattern))
        if not weights:
            raise ValueError(f"No model weights found beside {path}")
        for name in sorted(weights):
            check_path(path.parent / name)
    elif path.suffix in (".yaml", ".yml"):
        with path.open() as stream:
            config = yaml.safe_load(stream)
        for dataset in config.get("datasets", []):
            entries = dataset.get("json_path", [])
            for entry in [entries] if isinstance(entries, str) else entries:
                check_path(entry)


if __name__ == "__main__":
    errors = []
    for argument in sys.argv[1:]:
        try:
            check_path(argument)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        sys.exit("\n".join(errors))
    print("Input paths and checkpoint shard permissions checked.")
