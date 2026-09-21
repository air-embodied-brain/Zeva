"""Frozen PI-feature-to-ZTE task retrieval used by RoboTwin Zeva."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class CausalRetrievalHead(nn.Module):
    """Project a PI0.5 language or VLM feature into normalized ZTE task space."""

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 128,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(source_features.float()), dim=-1)
