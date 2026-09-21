"""Causal three-stream backbone used by ZeVA CTE."""

from __future__ import annotations

import torch
from torch import nn

try:  # pragma: no cover - the CUDA extension is optional for import-only tooling.
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover
    Mamba = None


class TriStreamBlock(nn.Module):
    """Temporal vision/action/interaction streams with causal cross-attention."""

    def __init__(self, config, block_idx: int):
        super().__init__()
        if Mamba is None:
            raise ImportError("ZeVA CTE training requires mamba_ssm.")
        dim = config.d_model
        layer_ids = (block_idx * 3, block_idx * 3 + 1, block_idx * 3 + 2)
        self.temporal = nn.ModuleList(
            Mamba(d_model=dim, d_state=16, d_conv=4, expand=2, layer_idx=layer_id)
            for layer_id in layer_ids
        )
        self.temporal_norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.vision_from_action = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=config.dropout)
        self.action_from_vision = nn.MultiheadAttention(dim, 4, batch_first=True, dropout=config.dropout)
        self.interaction_from_transition = nn.MultiheadAttention(
            dim, 4, batch_first=True, dropout=config.dropout
        )
        self.attention_norm = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))

    def forward(self, vision, action, interaction, inference_params=None):
        streams = [vision, action, interaction]
        streams = [
            value
            + temporal(norm(value), inference_params=inference_params)
            for value, temporal, norm in zip(
                streams, self.temporal, self.temporal_norm, strict=True
            )
        ]
        vision, action, interaction = streams
        batch, steps, dim = vision.shape
        vision = vision.reshape(batch * steps, 1, dim)
        action = action.reshape(batch * steps, 1, dim)
        interaction = interaction.reshape(batch * steps, 1, dim)

        update, _ = self.vision_from_action(self.attention_norm[0](vision), action, action)
        vision = vision + update
        update, _ = self.action_from_vision(self.attention_norm[1](action), vision, vision)
        action = action + update
        transition = torch.cat((vision, action), dim=1)
        update, _ = self.interaction_from_transition(
            self.attention_norm[2](interaction), transition, transition
        )
        interaction = interaction + update
        return (
            vision.reshape(batch, steps, dim),
            action.reshape(batch, steps, dim),
            interaction.reshape(batch, steps, dim),
        )
