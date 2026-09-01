"""Learned action residual projector for the Cosmos transition-memory prefix.

This module is intentionally tiny and dependency-light so the same class can
be copied into the remote Cosmos framework checkout.  It is not a selector:
it receives only the server-produced action chunk, the current phase/visual
features and the learned transition-memory context, and emits a bounded residual.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


ACTION_PROJECTOR_FORMAT = "robocasa_atomic5_transition_memory_action_projector_v1"


class TransitionMemoryPolicyAdapter(nn.Module):
    """Predict a bounded 32x7 residual from the learned memory context."""

    def __init__(
        self,
        *,
        context_dim: int = 256,
        phase_dim: int = 128,
        visual_dim: int = 128,
        action_horizon: int = 32,
        action_dim: int = 7,
        hidden_dim: int = 256,
        residual_scale: float = 0.5,
        effective_horizon: int = 16,
        residual_scale_max: float | None = None,
    ) -> None:
        super().__init__()
        if action_horizon != 32 or action_dim != 7:
            raise ValueError("Atomic-5 action projector requires horizon=32 and action_dim=7")
        if effective_horizon < 1 or effective_horizon > action_horizon:
            raise ValueError("effective_horizon must be in [1, action_horizon]")
        if residual_scale_max is None:
            residual_scale_max = float(residual_scale)
        if not torch.isfinite(torch.tensor(float(residual_scale_max))):
            raise ValueError("residual_scale_max must be finite")
        if float(residual_scale_max) <= 0.0:
            raise ValueError("residual_scale_max must be positive")
        if not torch.isfinite(torch.tensor(float(residual_scale))):
            raise ValueError("residual_scale must be finite")
        if float(residual_scale) < 0.0 or float(residual_scale) > float(residual_scale_max):
            raise ValueError("residual_scale must lie in [0, residual_scale_max]")
        self.context_dim = int(context_dim)
        self.phase_dim = int(phase_dim)
        self.visual_dim = int(visual_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.residual_scale_max = float(residual_scale_max)
        self.effective_horizon = int(effective_horizon)
        input_dim = context_dim + phase_dim + visual_dim + action_horizon * action_dim
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.residual_head = nn.Linear(hidden_dim, action_horizon * action_dim)

    def forward(
        self,
        context: Tensor,
        phase: Tensor,
        visual_key: Tensor,
        baseline_actions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if phase.ndim == 1:
            phase = phase.unsqueeze(0)
        if visual_key.ndim == 1:
            visual_key = visual_key.unsqueeze(0)
        if baseline_actions.ndim != 3:
            raise ValueError("baseline_actions must be [B,32,7]")
        expected = (self.action_horizon, self.action_dim)
        if tuple(baseline_actions.shape[1:]) != expected:
            raise ValueError(f"baseline_actions must be [B,{expected[0]},{expected[1]}]")
        if context.shape[0] != baseline_actions.shape[0] or phase.shape[0] != baseline_actions.shape[0] or visual_key.shape[0] != baseline_actions.shape[0]:
            raise ValueError("projector batch dimensions do not match")
        # The evaluator executes only the first ``effective_horizon`` controls
        # of a 32-action prediction.  Do not let the unexecuted suffix affect
        # the learned correction: it is often stochastic padding/noise.
        baseline_for_features = baseline_actions.float()
        if self.effective_horizon < self.action_horizon:
            mask = torch.zeros_like(baseline_for_features)
            mask[:, : self.effective_horizon] = 1.0
            baseline_for_features = baseline_for_features * mask
        features = torch.cat(
            (context.float(), phase.float(), visual_key.float(), baseline_for_features.reshape(baseline_actions.shape[0], -1)),
            dim=-1,
        )
        residual = torch.tanh(self.residual_head(self.body(features))).reshape(-1, *expected)
        if self.effective_horizon < self.action_horizon:
            mask = torch.zeros_like(residual)
            mask[:, : self.effective_horizon] = 1.0
            residual = residual * mask
        residual = residual * self.residual_scale
        return baseline_actions + residual, residual

    def set_residual_scale(self, value: float) -> None:
        """Set a calibrated bounded residual scale without changing weights."""
        value = float(value)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("residual scale must be finite")
        if value < 0.0 or value > self.residual_scale_max:
            raise ValueError(
                f"residual scale {value} is outside [0, {self.residual_scale_max}]"
            )
        self.residual_scale = value
