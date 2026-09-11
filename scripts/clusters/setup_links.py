"""Configure checkout-local links without copying data or changing their targets."""
import argparse
import os
import re
import uuid
from pathlib import Path


def configure(root, cluster, targets, replace=False):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", cluster):
        raise ValueError("Invalid cluster name")
    if not (root / "scripts/clusters" / cluster / "profile.sh").is_file():
        raise ValueError(f"No profile for {cluster}")
    changes = []
    for name, target in targets.items():
        target = Path(target).expanduser().resolve()
        link = root / name
        if not target.is_dir():
            raise ValueError(f"Target directory missing: {target}")
        if target == link or link in target.parents:
            raise ValueError(f"Self-referencing link: {name}")
        if link.is_symlink():
            if link.resolve() == target:
                continue
            if not replace:
                raise ValueError(f"{link} points elsewhere; use --replace-links to change it")
        elif link.exists():
            raise ValueError(f"Refusing to replace a real file/directory: {link}")
        changes.append((link, target))
    # Preflight every input before changing any link.
    for link, target in changes:
        temporary = root / (".link-" + uuid.uuid4().hex)
        try:
            temporary.symlink_to(target, target_is_directory=True)
            os.replace(temporary, link)
        finally:
            if temporary.is_symlink():
                temporary.unlink()
    (root / ".cluster").write_text(cluster + "\n")
    for name in targets:
        print(f"{name}/ -> {(root / name).resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True)
    for flag in ("data-root", "models-root", "sft-model", "reward-model", "results-root"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--replace-links", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    configure(root, args.cluster, dict(data=args.data_root, models=args.models_root,
              sft_model=args.sft_model, reward_model=args.reward_model, results=args.results_root),
              args.replace_links)


if __name__ == "__main__":
    main()
