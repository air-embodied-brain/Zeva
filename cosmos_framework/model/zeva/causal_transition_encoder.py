# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Causal Transition Encoder (CTE) for robot trajectories.

This is intentionally independent from the Cosmos action denoiser.  It learns a
per-time-step causal interaction state from the same multi-view observation and
8-D joint-position command format consumed by Cosmos Action.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class CausalTransitionEncoderConfig:
    action_dim: int = 8
    hidden_dim: int = 256
    retrieval_dim: int = 128
    phase_dim: int = 128
    effect_dim: int = 128
    transition_steps: int = 4
    # Phase stays at the VAE's 4x cadence.  Effect uses four such transitions
    # (16 raw controls) so it is supervised by an observable visual outcome.
    effect_window_transitions: int = 4
    effect_target_grid: tuple[int, int] = (4, 6)
    num_layers: int = 4
    num_heads: int = 8
    image_channels: int = 3
    ema_decay: float = 0.996
    # GRU is the stable/default RoboCasa lineage. Mamba remains opt-in so that
    # installing mamba_ssm cannot silently change the checkpoint architecture.
    use_mamba: bool = False
    vision_chunk_size: int = 256
    vision_gradient_checkpointing: bool = True
    vision_checkpoint_threshold: int = 1024

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class _VisionStem(nn.Module):
    """Small trainable visual stem; inputs are DROID's composite multi-view frames."""

    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        chunk_size: int,
        gradient_checkpointing: bool,
        checkpoint_threshold: int,
    ) -> None:
        super().__init__()
        width = max(hidden_dim // 4, 32)
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.chunk_size = chunk_size
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_threshold = checkpoint_threshold

    def _encode_flat(self, flat: Tensor) -> Tensor:
        return self.proj(self.net(flat).flatten(1))

    def forward(self, frames: Tensor) -> Tensor:
        # [B,T,C,H,W] -> [B,T,D]
        batch, steps = frames.shape[:2]
        flat = frames.flatten(0, 1)
        # A complete DROID segment can exceed 2,000 frames.  Checkpoint each
        # visual micro-batch so the CTE remains full-trajectory causal while
        # its convolutional activations do not scale with segment length.
        use_checkpoint = (
            self.training
            and self.gradient_checkpointing
            and torch.is_grad_enabled()
            and flat.shape[0] > self.checkpoint_threshold
        )
        if not use_checkpoint:
            return self._encode_flat(flat).unflatten(0, (batch, steps))
        chunks = []
        for frame_chunk in flat.split(self.chunk_size, dim=0):
            if use_checkpoint:
                chunks.append(checkpoint(self._encode_flat, frame_chunk, use_reentrant=False))
            else:
                chunks.append(self._encode_flat(frame_chunk))
        return torch.cat(chunks, dim=0).unflatten(0, (batch, steps))


class _FrozenVAEDeltaTarget(nn.Module):
    """Fixed spatial projection of cached Cosmos-VAE latents.

    Unlike the EMA vision stem, this target cannot drift to make an auxiliary
    loss easy.  Retaining a 4x6 pooled grid makes camera/object displacement
    visible instead of reducing every frame to one spatial mean.
    """

    def __init__(self, channels: int, hidden_dim: int, grid: tuple[int, int]) -> None:
        super().__init__()
        self.grid = grid
        input_dim = channels * grid[0] * grid[1]
        generator = torch.Generator().manual_seed(20260815)
        projection = torch.randn(hidden_dim, input_dim, generator=generator) / input_dim**0.5
        self.register_buffer("projection", projection, persistent=True)

    @torch.no_grad()
    def forward(self, frames: Tensor) -> Tensor:
        batch, steps = frames.shape[:2]
        flat = frames.flatten(0, 1).float()
        pooled = F.adaptive_avg_pool2d(flat, self.grid).flatten(1)
        pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        return F.linear(pooled, self.projection).unflatten(0, (batch, steps))


class _CausalMixer(nn.Module):
    """Causal temporal mixer with a portable GRU fallback when mamba_ssm is absent."""

    def __init__(self, hidden_dim: int, use_mamba: bool) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.kind = "gru"
        if use_mamba:
            try:
                from mamba_ssm import Mamba  # type: ignore[import-not-found]

                self.mixer: nn.Module = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
                self.kind = "mamba"
            except ImportError:
                self.mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            self.mixer = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        if self.kind == "mamba":
            y = self.mixer(y)
        else:
            y, _ = self.mixer(y)
        return x + y


class _CausalInteractionBlock(nn.Module):
    """Causal visual/action/interaction streams with same-step cross attention."""

    def __init__(self, cfg: CausalTransitionEncoderConfig) -> None:
        super().__init__()
        self.visual = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.action = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.interaction_state = _CausalMixer(cfg.hidden_dim, cfg.use_mamba)
        self.cross = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(cfg.hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, 4 * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.hidden_dim, cfg.hidden_dim),
        )

    def forward(self, visual: Tensor, action: Tensor, interaction_state: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        visual, action, interaction_state = (
            self.visual(visual),
            self.action(action),
            self.interaction_state(interaction_state),
        )
        # The three tokens for each timestep attend only within that timestep.
        tokens = torch.stack((visual, action, interaction_state), dim=2)
        batch, steps, streams, dim = tokens.shape
        tokens = tokens.reshape(batch * steps, streams, dim)
        attended, _ = self.cross(tokens, tokens, tokens, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(tokens)
        tokens = tokens.reshape(batch, steps, streams, dim)
        visual, action, interaction_state = tokens.unbind(dim=2)
        mask = valid.unsqueeze(-1).to(visual.dtype)
        return visual * mask, action * mask, interaction_state * mask


class CausalTransitionEncoder(nn.Module):
    """Causal transition encoder with phase and causal-effect projections."""

    def __init__(self, cfg: CausalTransitionEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CausalTransitionEncoderConfig()
        self.visual_encoder = _VisionStem(
            self.cfg.image_channels,
            self.cfg.hidden_dim,
            self.cfg.vision_chunk_size,
            self.cfg.vision_gradient_checkpointing,
            self.cfg.vision_checkpoint_threshold,
        )
        self.target_visual_encoder = deepcopy(self.visual_encoder).requires_grad_(False)
        # The interaction/action stream is intentionally right shifted. At state
        # v_b it sees only the completed transition u_(b-1), never u_b.  The
        # latter is supplied exclusively to the auxiliary effect heads.
        transition_dim = self.cfg.transition_steps * self.cfg.action_dim
        self.action_encoder = nn.Sequential(nn.LayerNorm(transition_dim), nn.Linear(transition_dim, self.cfg.hidden_dim))
        self.bos_action = nn.Parameter(torch.zeros(1, 1, self.cfg.hidden_dim))
        nn.init.normal_(self.bos_action, std=0.02)
        self.interaction_state_token = nn.Parameter(torch.zeros(1, 1, self.cfg.hidden_dim))
        nn.init.normal_(self.interaction_state_token, std=0.02)
        self.blocks = nn.ModuleList([_CausalInteractionBlock(self.cfg) for _ in range(self.cfg.num_layers)])
        self.final_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.retrieval_head = nn.Linear(self.cfg.hidden_dim, self.cfg.retrieval_dim)
        self.phase_head = nn.Linear(self.cfg.hidden_dim, self.cfg.phase_dim)
        self.action_head = nn.Sequential(nn.LayerNorm(self.cfg.hidden_dim), nn.Linear(self.cfg.hidden_dim, transition_dim))
        self.visual_head = nn.Sequential(nn.LayerNorm(self.cfg.hidden_dim), nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim))
        effect_window_dim = self.cfg.effect_window_transitions * transition_dim
        self.effect_action_encoder = nn.Sequential(
            nn.LayerNorm(effect_window_dim), nn.Linear(effect_window_dim, self.cfg.hidden_dim)
        )
        self.frozen_effect_target = _FrozenVAEDeltaTarget(
            self.cfg.image_channels, self.cfg.hidden_dim, self.cfg.effect_target_grid
        )
        self.effect_pre_head = nn.Sequential(
            nn.LayerNorm(2 * self.cfg.hidden_dim), nn.Linear(2 * self.cfg.hidden_dim, self.cfg.effect_dim)
        )
        # Observed effect is visual only.  Passing executed actions here lets
        # post simply echo the action-conditioned pre code and caused v2's
        # rank-one collapse.
        self.effect_post_head = nn.Sequential(
            nn.LayerNorm(2 * self.cfg.hidden_dim), nn.Linear(2 * self.cfg.hidden_dim, self.cfg.effect_dim)
        )
        self.effect_action_head = nn.Sequential(
            nn.LayerNorm(self.cfg.effect_dim), nn.Linear(self.cfg.effect_dim, 4 * self.cfg.action_dim)
        )
        self.effect_outcome_head = nn.Sequential(
            nn.LayerNorm(self.cfg.effect_dim), nn.Linear(self.cfg.effect_dim, self.cfg.hidden_dim)
        )

    @torch.no_grad()
    def update_ema_target(self) -> None:
        """Call once after every optimizer step, before building the next target."""
        for target, online in zip(self.target_visual_encoder.parameters(), self.visual_encoder.parameters(), strict=True):
            target.lerp_(online, 1.0 - self.cfg.ema_decay)

    @torch.no_grad()
    def encode_target_vision(self, frames: Tensor) -> Tensor:
        return self.target_visual_encoder(frames)

    def forward(
        self,
        frames: Tensor,
        transition_actions: Tensor,
        valid_mask: Tensor | None = None,
        transition_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode padded full trajectories.

        Args:
            frames: DROID composite observations ``[B,T,3,H,W]`` in [0, 1].
            transition_actions: completed/raw four-action transitions
                ``[B,T-1,4,A]``.  At state ``b`` the main action stream only
                receives transition ``b-1``; transition ``b`` is used solely
                to supervise the effect heads.
            valid_mask: ``[B,T]`` true for real timesteps, false for padding.
        """
        if frames.ndim != 5 or transition_actions.ndim != 4:
            raise ValueError("Expected frames [B,T,C,H,W] and transition_actions [B,T-1,4,A].")
        if (
            transition_actions.shape[0] != frames.shape[0]
            or transition_actions.shape[1] != frames.shape[1] - 1
            or transition_actions.shape[2:] != (self.cfg.transition_steps, self.cfg.action_dim)
        ):
            raise ValueError("Frame/transition time axes or action shape do not match CTE config.")
        if valid_mask is None:
            valid_mask = torch.ones(frames.shape[:2], dtype=torch.bool, device=frames.device)
        if transition_valid is None:
            transition_valid = torch.ones(transition_actions.shape[:-1], dtype=torch.bool, device=frames.device)
        if transition_valid.shape != transition_actions.shape[:-1]:
            raise ValueError("transition_valid must have shape [B,T-1,4].")
        visual = self.visual_encoder(frames)
        transition_flat = transition_actions.flatten(-2)
        transition_embed = self.action_encoder(transition_flat)
        action = self.bos_action.expand(frames.shape[0], frames.shape[1], -1).clone()
        action[:, 1:] = transition_embed
        action[:, 1:] *= transition_valid.all(dim=-1, keepdim=True).to(action.dtype)
        interaction_state = self.interaction_state_token.expand_as(visual)
        for block in self.blocks:
            visual, action, interaction_state = block(visual, action, interaction_state, valid_mask)
        z = self.final_norm(interaction_state) * valid_mask.unsqueeze(-1)
        with torch.no_grad():
            target_visual = self.target_visual_encoder(frames)
        transition_complete = valid_mask[:, :-1] & valid_mask[:, 1:] & transition_valid.all(dim=-1)
        effect_windows = (frames.shape[1] - 1) // self.cfg.effect_window_transitions
        effect_steps = self.cfg.effect_window_transitions
        if effect_windows:
            effect_start = torch.arange(effect_windows, device=frames.device) * effect_steps
            effect_end = effect_start + effect_steps
            effect_actions = transition_actions[:, : effect_windows * effect_steps].reshape(
                frames.shape[0], effect_windows, effect_steps, self.cfg.transition_steps, self.cfg.action_dim
            )
            effect_action_valid = transition_valid[:, : effect_windows * effect_steps].reshape(
                frames.shape[0], effect_windows, effect_steps, self.cfg.transition_steps
            )
            effect_complete = (
                valid_mask[:, effect_start]
                & valid_mask[:, effect_end]
                & effect_action_valid.all(dim=(-1, -2))
            )
            effect_embed = self.effect_action_encoder(effect_actions.flatten(-3))
            frozen_visual = self.frozen_effect_target(frames)
            effect_delta = frozen_visual[:, effect_end] - frozen_visual[:, effect_start]
            effect_pre_raw = self.effect_pre_head(torch.cat((z[:, effect_start], effect_embed), dim=-1))
            effect_post_raw = self.effect_post_head(torch.cat((frozen_visual[:, effect_start], effect_delta), dim=-1))
        else:
            effect_actions = transition_actions.new_zeros((frames.shape[0], 0, effect_steps, self.cfg.transition_steps, self.cfg.action_dim))
            effect_complete = torch.zeros((frames.shape[0], 0), dtype=torch.bool, device=frames.device)
            effect_delta = z[:, :0]
            effect_pre_raw = z[:, :0]
            effect_post_raw = z[:, :0]
        # At reset there is a single observed frame and no completed
        # transition/effect window.  LayerNorm cannot be applied to the empty
        # ``[..., 0, hidden_dim]`` fallback above because the effect heads
        # expect ``effect_dim``.  Preserve the causal no-effect state with
        # correctly shaped empty outputs; callers then provide only BOS/pad
        # effect tokens to the policy.
        if effect_pre_raw.shape[1] == 0:
            effect_pre = effect_pre_raw.new_zeros(
                (frames.shape[0], 0, self.cfg.effect_dim)
            )
            effect_post = effect_pre_raw.new_zeros(
                (frames.shape[0], 0, self.cfg.effect_dim)
            )
            effect_outcome_pre = effect_pre_raw.new_zeros(
                (frames.shape[0], 0, self.cfg.hidden_dim)
            )
            effect_outcome_post = effect_pre_raw.new_zeros(
                (frames.shape[0], 0, self.cfg.hidden_dim)
            )
            effect_action = effect_pre_raw.new_zeros(
                (frames.shape[0], 0, 4 * self.cfg.action_dim)
            )
        else:
            effect_pre = F.normalize(effect_pre_raw, dim=-1)
            effect_post = F.normalize(effect_post_raw, dim=-1)
            effect_outcome_pre = self.effect_outcome_head(effect_pre_raw)
            effect_outcome_post = self.effect_outcome_head(effect_post_raw)
            effect_action = self.effect_action_head(effect_pre_raw)
        return {
            "causal_interaction_state": z,
            "retrieval": F.normalize(self.retrieval_head(z), dim=-1),
            "phase": F.normalize(self.phase_head(z), dim=-1),
            "next_action": self.action_head(z[:, :-1]).view(
                frames.shape[0], frames.shape[1] - 1, self.cfg.transition_steps, self.cfg.action_dim
            ),
            "next_vision": self.visual_head(z),
            # Frozen target-visual representation used as the PIM retrieval key.
            "visual_key": F.normalize(target_visual[..., : self.cfg.phase_dim], dim=-1),
            "effect_pre": effect_pre,
            "effect_post": effect_post,
            "effect_pre_raw": effect_pre_raw,
            "effect_post_raw": effect_post_raw,
            "effect_outcome_pre": effect_outcome_pre,
            "effect_outcome_post": effect_outcome_post,
            "effect_action": effect_action,
            "effect_actions": effect_actions,
            "effect_delta_target": effect_delta,
            "effect_complete": effect_complete,
            "target_visual": target_visual,
            "transition_complete": transition_complete,
        }
