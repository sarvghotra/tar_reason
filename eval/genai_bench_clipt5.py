"""GenAI-Bench with the *official* VQAScore judge, CLIP-FlanT5-XXL (Lin et al. 2024), 2026-10-30.

jeet's pipeline scores GenAI-Bench with Qwen3.5-27B VQAScore (his t2v_metrics fork, which dropped the
CLIP-FlanT5 models). Published GenAI-Bench numbers use ``clip-flant5-xxl``, so this runs his
``genai_bench/evaluate.py`` unchanged (same prompts, image layout, per-skill summary printout) but with the
CLIP-FlanT5 code of the last upstream t2v_metrics commit that still shipped it (275bde5, 2025-11-02).

That old package's ``__init__`` imports its whole model zoo (LLaVA, InternVL, ...), so instead of installing it we
register stub packages for ``t2v_metrics``, ``t2v_metrics.models`` and ``t2v_metrics.models.vqascore_models`` whose
``__path__`` points at the old checkout: only the modules the CLIP-FlanT5 scorer needs are imported, verbatim
(``score.py``, ``constants.py``, ``models/model.py``, ``vqascore_models/{vqa_model,mm_utils,clip_t5_model}.py``,
``clip_t5/``). The env needs only torch, transformers (4.36.1 = the official pin), sentencepiece, opencv, pandas,
scipy, scikit-learn (his dataset.py).

Weights: ``zhiqiulin/clip-flant5-xxl`` (T5 + projector, loaded with ``cache_dir`` = ``--cache_dir``) and
``openai/clip-vit-large-patch14-336`` (vision tower + tokenizer, loaded from the default HF cache, i.e. $HF_HOME/hub).
Pre-download both into $HF_HOME/hub and pass ``--cache_dir $HF_HOME/hub`` (compute nodes have no internet).

    T2V_CLIPT5_DIR=<old t2v_metrics checkout> T2V_FORK_DIR=<jeet's fork> \\
    python eval/genai_bench_clipt5.py --model clip-flant5-xxl --root_dir ... --output_dir ... --gen_model ... \\
        --num_prompts 1600 --cache_dir $HF_HOME/hub --result_dir ...      # = genai_bench/evaluate.py's arguments
"""

import importlib
import os
import runpy
import sys
import types

T2V_OLD = os.environ.get("T2V_CLIPT5_DIR")
FORK = os.environ.get("T2V_FORK_DIR")
if not (T2V_OLD and FORK):
    sys.exit("set T2V_CLIPT5_DIR (old t2v_metrics checkout with clip_t5) and T2V_FORK_DIR (jeet's fork)")
if "t2v_metrics" in sys.modules:
    sys.exit("t2v_metrics is already imported; run this script from a directory without a t2v_metrics package")


def stub(name, path):
    """A package whose __init__ is never executed: submodules resolve under ``path``."""
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


pkg = stub("t2v_metrics", os.path.join(T2V_OLD, "t2v_metrics"))
stub("t2v_metrics.models", os.path.join(T2V_OLD, "t2v_metrics", "models"))
stub("t2v_metrics.models.vqascore_models", os.path.join(T2V_OLD, "t2v_metrics", "models", "vqascore_models"))
constants = importlib.import_module("t2v_metrics.constants")
pkg.constants = constants
Score = importlib.import_module("t2v_metrics.score").Score
clip_t5_model = importlib.import_module("t2v_metrics.models.vqascore_models.clip_t5_model")


class CLIPT5VQAScore(Score):
    """t2v_metrics.VQAScore restricted to the CLIP-T5 family (same forward / batch_forward)."""

    def prepare_scoremodel(self, model="clip-flant5-xxl", device="cuda", cache_dir=constants.HF_CACHE_DIR, **kw):
        return clip_t5_model.CLIPT5Model(model, device=device, cache_dir=cache_dir)

    def list_all_models(self):
        return list(clip_t5_model.CLIP_T5_MODELS)


def get_score_model(model="clip-flant5-xxl", device="cuda", cache_dir=constants.HF_CACHE_DIR, **kwargs):
    return CLIPT5VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)


pkg.get_score_model = get_score_model
pkg.VQAScore = CLIPT5VQAScore
pkg.list_all_models = lambda: list(clip_t5_model.CLIP_T5_MODELS)

sys.path.insert(0, FORK)                      # his dataset.py (GenAIBench_Image)
print(f"[clipt5] t2v_metrics code: {T2V_OLD}; evaluate.py + dataset.py: {FORK}", flush=True)
runpy.run_path(os.path.join(FORK, "genai_bench", "evaluate.py"), run_name="__main__")
