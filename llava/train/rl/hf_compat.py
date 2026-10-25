"""Keyword compatibility across the Transformers versions this code runs under.

HEAD runs under the legacy tar environment (4.50), the qwen_reward
environment (4.57) and the uv project (5.x).
"""
from contextlib import contextmanager

import transformers
from packaging.version import Version

DTYPE_KEYWORD = 'dtype' if Version(transformers.__version__) >= Version('4.56.0') else 'torch_dtype'


def dtype_kwargs(dtype):
    """`from_pretrained` dtype keyword for the installed version.

    4.56 renamed `torch_dtype` to `dtype` (5.x keeps the old name only as a
    deprecated alias). Earlier versions forward unknown keywords to the model
    config, so passing `dtype` there would silently load float32 weights.
    """
    return {DTYPE_KEYWORD: dtype}


TRANSFORMERS_5 = Version(transformers.__version__) >= Version('5.0.0')
# Before 5.0 this keyword makes `generate` ignore the checkpoint's generation_config.json.
NO_MODEL_DEFAULTS = {} if TRANSFORMERS_5 else {'use_model_defaults': False}


@contextmanager
def without_model_generation_defaults(model):
    """Keep the checkpoint's generation_config.json out of `generate`.

    5.x removed `use_model_defaults` and always fills every unset field of the
    passed config from `model.generation_config` (the SFT checkpoint sets
    temperature 0.7, top-k 20, top-p 0.8 and repetition penalty 1.05). Blanking
    it for the call leaves the global defaults plus explicit arguments, as
    `use_model_defaults=False` did. PEFT copies the wrapper's config onto the
    base model inside `generate`, so both levels are blanked, then restored
    exactly. A no-op before 5.0.
    """
    if not TRANSFORMERS_5:
        yield
        return
    from transformers import GenerationConfig
    targets = [model] + ([model.get_base_model()] if hasattr(model, 'get_base_model') else [])
    saved = []
    for i, target in enumerate(targets):
        if any(target is earlier for earlier in targets[:i]):  # by identity: models may be unhashable
            continue
        if hasattr(target, 'generation_config'):
            own = 'generation_config' in vars(target)
            saved.append((target, own, vars(target).get('generation_config')))
            target.generation_config = GenerationConfig()
    try:
        yield
    finally:
        for target, own, value in reversed(saved):
            if own:
                target.generation_config = value
            else:
                del target.generation_config
