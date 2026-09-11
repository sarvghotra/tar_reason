"""Download and pin the official GenEval2 judge to scratch-backed results/."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo = "Qwen/Qwen3-VL-8B-Instruct"
target = Path(__file__).resolve().parents[1] / "results" / "pretrained" / "Qwen3-VL-8B-Instruct"
info = HfApi().model_info(repo)
print(f"Downloading {repo} revision {info.sha} to {target}", flush=True)
snapshot_download(repo, revision=info.sha, local_dir=str(target), max_workers=4,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "README.md", "LICENSE*"])
(target / "download_provenance.json").write_text(json.dumps(
    dict(repo_id=repo, revision=info.sha), indent=2) + "\n")
print("Download complete.", flush=True)
