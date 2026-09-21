from __future__ import annotations

import torch
from torch import nn


class MemoryContextEncoder(nn.Module):
    """Fuse task, phase, BIT and phase-retrieved PIM into a causal prompt."""

    def __init__(
        self,
        *,
        task_dim: int = 256,
        phase_dim: int = 128,
        signal_dim: int = 256,
        context_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.task_projector = nn.Linear(task_dim, context_dim)
        self.phase_projector = nn.Linear(phase_dim, context_dim)
        self.signal_projector = nn.Linear(signal_dim, context_dim)
        self.memory_attention = nn.MultiheadAttention(context_dim, num_heads, dropout=dropout, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(context_dim * 3, context_dim),
            nn.LayerNorm(context_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(context_dim, context_dim),
            nn.LayerNorm(context_dim),
        )

    def forward(
        self,
        task_token: torch.Tensor,
        phase_token: torch.Tensor,
        brief_signals: torch.Tensor | None = None,
        retrieved_signals: torch.Tensor | None = None,
        brief_mask: torch.Tensor | None = None,
        retrieved_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        task = self.task_projector(task_token)
        phase = self.phase_projector(phase_token)
        streams = []
        masks = []
        for stream, mask in (
            (brief_signals, brief_mask),
            (retrieved_signals, retrieved_mask),
        ):
            if stream is None or not stream.numel():
                continue
            streams.append(stream)
            masks.append(
                torch.ones(stream.shape[:2], dtype=torch.bool, device=stream.device)
                if mask is None
                else mask.to(device=stream.device, dtype=torch.bool)
            )
        if streams:
            memory = self.signal_projector(torch.cat(streams, dim=1))
            valid = torch.cat(masks, dim=1)
            memory_context, _ = self.memory_attention(
                phase.unsqueeze(1), memory, memory, key_padding_mask=~valid, need_weights=False
            )
            memory_context = memory_context.squeeze(1)
        else:
            memory_context = torch.zeros_like(phase)
        return self.output(torch.cat([task, phase, memory_context], dim=-1))
