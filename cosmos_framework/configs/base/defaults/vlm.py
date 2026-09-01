"""Compatibility shim: behavior branch renamed defaults/vlm -> defaults/reasoner.
Hydra/export may still locate class paths under defaults.vlm.* for checkpoints
serialized before the rename; forward them to the reasoner definitions.
"""
from cosmos_framework.configs.base.defaults.reasoner import (
    PretrainedWeightsConfig,
    VLMConfig,
    create_qwen2_tokenizer_with_download,
    create_vlm_config,
)

__all__ = [
    "PretrainedWeightsConfig",
    "VLMConfig",
    "create_qwen2_tokenizer_with_download",
    "create_vlm_config",
]
