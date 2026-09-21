"""Torch-only Gemma architecture configs used by PI0/PI0.5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict = field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_2b"]


def get_config(variant: Variant) -> Config:
    if variant == "dummy":
        return Config(64, 4, 128, 8, 1, 16)
    if variant == "gemma_300m":
        return Config(1024, 18, 4096, 8, 1, 256)
    if variant == "gemma_2b":
        return Config(2048, 18, 16_384, 8, 1, 256)
    raise ValueError(f"Torch PI0.5 does not support Gemma variant {variant!r}.")
