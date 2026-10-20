"""vLLM model class for Tar checkpoints, used by generate_imgs_parquet_vllm.py.

Tar checkpoints are saved as LlavaQwenForCausalLM: a plain Qwen2 LM plus the
TA-Tok vision tower under `model.vision_tower.*`. Text-to-image generation only
needs the LM, so this is vLLM's Qwen2ForCausalLM with those weights skipped
(vLLM's loader otherwise rejects them as unexpected).
"""

from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM


class TarQwen2ForCausalLM(Qwen2ForCausalLM):
    def load_weights(self, weights):
        return super().load_weights(
            (name, tensor) for name, tensor in weights
            if not name.startswith("model.vision_tower.")
        )
