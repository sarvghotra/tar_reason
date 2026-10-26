"""Model-family backends for eval/baselines/unified_gen.py.

Each module defines DEFAULT_MODEL (HF id), DEFAULT_BATCH (prompts per call), DEFAULT_SIZE (px, for dry runs),
``add_args(parser)`` for its own flags, and ``Backend(args, device)`` with ``run(prompts, seeds)`` returning,
per prompt, ``(images, rounds)`` (see unified_gen.py).
"""
