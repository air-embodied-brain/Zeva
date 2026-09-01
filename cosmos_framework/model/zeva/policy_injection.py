# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Zeva policy-injection modules for the frozen diffusion policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .brief_interaction_trace import BriefInteractionTrace


@dataclass
class PolicyInjectionConfig:
    global_dim: int = 256
    phase_dim: int = 128
    effect_dim: int = 128
    effect_history_length: int = 4
    action_dim: int = 8
    horizon: int = 32
    num_anchors: int = 8
    hidden_dim: int = 256
    num_heads: int = 4
    min_std: float = 1e-3


class PolicyInjectionPrior(nn.Module):
    """Fuse task context, phase, and BIT into an action prior."""

    def __init__(self, cfg: PolicyInjectionConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or PolicyInjectionConfig()
        self.global_to_anchors = nn.Linear(self.cfg.global_dim, self.cfg.num_anchors * self.cfg.hidden_dim)
        self.anchor_position = nn.Parameter(torch.empty(1, self.cfg.num_anchors, self.cfg.hidden_dim))
        self.phase_query = nn.Sequential(nn.LayerNorm(self.cfg.phase_dim), nn.Linear(self.cfg.phase_dim, self.cfg.hidden_dim))
        self.effect_project = nn.Sequential(nn.LayerNorm(self.cfg.effect_dim), nn.Linear(self.cfg.effect_dim, self.cfg.hidden_dim))
        self.effect_position = nn.Parameter(torch.empty(1, self.cfg.effect_history_length, self.cfg.hidden_dim))
        self.bos_effect = nn.Parameter(torch.empty(1, 1, self.cfg.effect_dim))
        self.effect_attention = nn.MultiheadAttention(self.cfg.hidden_dim, self.cfg.num_heads, batch_first=True)
        self.effect_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.effect_gate = nn.Sequential(
            nn.LayerNorm(2 * self.cfg.hidden_dim), nn.Linear(2 * self.cfg.hidden_dim, self.cfg.hidden_dim), nn.Sigmoid()
        )
        self.progress_attention = nn.MultiheadAttention(self.cfg.hidden_dim, self.cfg.num_heads, batch_first=True)
        self.context_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.distribution_head = nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(self.cfg.hidden_dim, 2 * self.cfg.horizon * self.cfg.action_dim),
        )
        nn.init.normal_(self.anchor_position, std=0.02)
        nn.init.normal_(self.effect_position, std=0.02)
        nn.init.normal_(self.bos_effect, std=0.02)

    def init_weights(self) -> None:
        """Initialize parameters after the Cosmos meta-device construction path."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.progress_attention._reset_parameters()
        self.effect_attention._reset_parameters()
        nn.init.normal_(self.anchor_position, std=0.02)
        nn.init.normal_(self.effect_position, std=0.02)
        nn.init.normal_(self.bos_effect, std=0.02)

    def forward(
        self,
        z_global: Tensor,
        z_phase: Tensor,
        z_effect: Tensor | BriefInteractionTrace,
        effect_valid: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return Gaussian mean and std, both ``[B,horizon,action_dim]``."""
        if isinstance(z_effect, BriefInteractionTrace):
            if effect_valid is not None:
                raise ValueError("Pass either BriefInteractionTrace or effect_valid, not both")
            effect_valid = z_effect.valid
            z_effect = z_effect.effects
        if z_global.ndim != 2 or z_global.shape[-1] != self.cfg.global_dim:
            raise ValueError(f"Expected z_global [B,{self.cfg.global_dim}], got {tuple(z_global.shape)}")
        if z_phase.ndim != 2 or z_phase.shape != (z_global.shape[0], self.cfg.phase_dim):
            raise ValueError(f"Expected z_phase [B,{self.cfg.phase_dim}], got {tuple(z_phase.shape)}")
        expected_effect_shape = (z_global.shape[0], self.cfg.effect_history_length, self.cfg.effect_dim)
        if z_effect.ndim != 3 or z_effect.shape != expected_effect_shape:
            raise ValueError(
                f"Expected z_effect [B,{self.cfg.effect_history_length},{self.cfg.effect_dim}], "
                f"got {tuple(z_effect.shape)}"
            )
        if effect_valid is None:
            effect_valid = torch.ones(z_effect.shape[:2], dtype=torch.bool, device=z_effect.device)
        if effect_valid.shape != z_effect.shape[:2]:
            raise ValueError("effect_valid must be [B,effect_history_length]")
        anchors = self.global_to_anchors(z_global).view(
            -1, self.cfg.num_anchors, self.cfg.hidden_dim
        ) + self.anchor_position
        phase_query = self.phase_query(z_phase)
        bos = self.bos_effect.expand(z_effect.shape[0], self.cfg.effect_history_length, -1)
        effect_input = torch.where(effect_valid.unsqueeze(-1), z_effect, bos)
        effect_tokens = self.effect_project(effect_input) + self.effect_position
        effect_context, _ = self.effect_attention(
            phase_query.unsqueeze(1), effect_tokens, effect_tokens, need_weights=False
        )
        effect_context = self.effect_norm(effect_context.squeeze(1))
        gate = self.effect_gate(torch.cat((phase_query, effect_context), dim=-1))
        local_query = gate * phase_query + (1.0 - gate) * effect_context
        context, _ = self.progress_attention(local_query.unsqueeze(1), anchors, anchors, need_weights=False)
        params = self.distribution_head(self.context_norm(context.squeeze(1)))
        mean, raw_scale = params.chunk(2, dim=-1)
        mean = mean.view(-1, self.cfg.horizon, self.cfg.action_dim)
        std = (F.softplus(raw_scale) + self.cfg.min_std).view(-1, self.cfg.horizon, self.cfg.action_dim)
        return mean, std


def gaussian_prior_nll(target: Tensor, mean: Tensor, std: Tensor, valid: Tensor | None = None) -> Tensor:
    """Mean diagonal-Gaussian NLL in raw action coordinates."""
    if target.shape != mean.shape or mean.shape != std.shape:
        raise ValueError("target, mean, and std must have the same [B,H,D] shape")
    nll = 0.5 * (((target - mean) / std).square() + 2 * std.log()).mean(dim=-1)
    if valid is None:
        return nll.mean()
    if valid.shape != nll.shape:
        raise ValueError("valid must have shape [B,H]")
    return (nll * valid.to(nll.dtype)).sum() / valid.sum().clamp_min(1)


class CausalPromptPolicyAdapter(nn.Module):
    """Project the Zeva action prior into frozen-policy action-token space."""

    def __init__(self, *, action_dim: int = 8, hidden_dim: int = 2048) -> None:
        super().__init__()
        self.prior_to_action_embedding = nn.Linear(action_dim, hidden_dim, bias=True)

    def init_weights(self) -> None:
        nn.init.zeros_(self.prior_to_action_embedding.weight)
        nn.init.zeros_(self.prior_to_action_embedding.bias)

    def forward(self, prior_mean: Tensor, *, action_length: int, leading_condition_steps: int = 1) -> Tensor:
        if prior_mean.ndim != 3:
            raise ValueError("prior_mean must be [B,H,D]")
        if action_length < leading_condition_steps:
            raise ValueError("action_length must include leading conditioning steps")
        if action_length - leading_condition_steps != prior_mean.shape[1]:
            raise ValueError("Policy-injection horizon must equal the noisy action-step count")
        embedded = self.prior_to_action_embedding(prior_mean)
        if leading_condition_steps:
            embedded = torch.cat(
                (
                    embedded.new_zeros((embedded.shape[0], leading_condition_steps, embedded.shape[-1])),
                    embedded,
                ),
                dim=1,
            )
        return embedded
