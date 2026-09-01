# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Static task-context retrieval, distinct from phase-conditioned PIM retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class StaticTaskContextRetrievalConfig:
    input_dim: int = 4096
    hidden_dim: int = 1024
    output_dim: int = 128
    dropout: float = 0.1

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class StaticTaskContextRetrievalHead(nn.Module):
    """Map a frozen policy readout into the static task-context key space."""

    def __init__(self, config: StaticTaskContextRetrievalConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, readout: Tensor) -> Tensor:
        if readout.ndim != 2 or readout.shape[-1] != self.config.input_dim:
            raise ValueError(f"Expected readout [B,{self.config.input_dim}], got {tuple(readout.shape)}")
        return F.normalize(self.projection(readout), dim=-1)


def bidirectional_supervised_contrastive_loss(
    queries: Tensor,
    keys: Tensor,
    semantic_ids: Tensor,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """Symmetric multi-positive InfoNCE for aligned query/key mini-batches."""
    if queries.ndim != 2 or keys.shape != queries.shape:
        raise ValueError(f"Expected aligned [B,D] queries/keys, got {tuple(queries.shape)} and {tuple(keys.shape)}")
    if semantic_ids.shape != (queries.shape[0],):
        raise ValueError(f"Expected semantic_ids [{queries.shape[0]}], got {tuple(semantic_ids.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    queries = F.normalize(queries.float(), dim=-1)
    keys = F.normalize(keys.float(), dim=-1)
    logits = queries @ keys.T / temperature
    positive = semantic_ids[:, None].eq(semantic_ids[None, :])
    if not positive.diagonal().all():
        raise ValueError("Every aligned query/key pair must be positive")

    def direction_loss(scores: Tensor, mask: Tensor) -> Tensor:
        positive_logsumexp = torch.logsumexp(scores.masked_fill(~mask, -torch.inf), dim=1)
        all_logsumexp = torch.logsumexp(scores, dim=1)
        return (all_logsumexp - positive_logsumexp).mean()

    return 0.5 * (direction_loss(logits, positive) + direction_loss(logits.T, positive.T))


@torch.no_grad()
def retrieve_static_task_context(
    queries: Tensor,
    memory_keys: Tensor,
    memory_values: Tensor,
    *,
    top_k: int = 5,
    temperature: float = 0.07,
) -> tuple[Tensor, Tensor, Tensor]:
    """Retrieve and softmax-average frozen task-context values for each query."""
    if queries.ndim != 2 or memory_keys.ndim != 2 or memory_values.ndim != 2:
        raise ValueError("queries, memory_keys, and memory_values must be rank-2 tensors")
    if queries.shape[-1] != memory_keys.shape[-1]:
        raise ValueError("query and memory-key dimensions must match")
    if memory_keys.shape[0] != memory_values.shape[0]:
        raise ValueError("memory key/value counts must match")
    if not 1 <= top_k <= memory_keys.shape[0]:
        raise ValueError(f"top_k must be in [1,{memory_keys.shape[0]}]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scores = F.normalize(queries.float(), dim=-1) @ F.normalize(memory_keys.float(), dim=-1).T
    top_scores, top_indices = scores.topk(top_k, dim=-1)
    weights = F.softmax(top_scores / temperature, dim=-1)
    selected = memory_values[top_indices]
    values = torch.einsum("bk,bkd->bd", weights.to(selected.dtype), selected)
    return values, top_indices, top_scores
