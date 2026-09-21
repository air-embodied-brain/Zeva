"""Causal ZeVA transition encoder (v2).

This module deliberately does not modify :mod:`transition_encoder`.  The v2
encoder is a small, self-contained research implementation with an explicit
information-flow contract:

* ``pre_context`` and the JEPA/forward-effect head see the observation before
  a transition and its executed H15 action chunk, but never ``images_after``;
  the next-action head defaults to that same pre-transition context.  An
  opt-in ``action_prediction_context="phase"`` variant instead consumes only
  the exported, normalized post-transition ``phase_token``; this is for the
  next replanning boundary and does not change the forward-effect path;
* ``causal_signal`` is produced after a post-transition effect stream has
  consumed the EMA visual effect ``phi(s_{t+1}) - phi(s_t)``;
* the H15 action chunk is encoded as an ordered token sequence.  There is no
  mean pooling over action steps;
* the initial state ``B0`` contains the first visual state and the task-only
  PI0.5 language embedding.  Local task/phase heads are trained from visual,
  action and effect streams, while language consistency regularisation belongs
  to the trainer (see ``train_robotwin_zte_v2.py``).

The public output has separate global, phase and causal prompts so downstream
PI0.5 adapters do not need to infer which representation they are receiving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F  # noqa: N812

from openpi.zeva.config import ZevaConfig
from openpi.zeva.vision import LightweightVisionEncoder

try:  # pragma: no cover - availability depends on the training image.
    from mamba_ssm import Mamba
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:  # pragma: no cover - the fallback is used by CPU unit tests.
    Mamba = None
    InferenceParams = None


@dataclass(frozen=True)
class TransitionEncoderV2Config:
    """Configuration for the v2 encoder.

    The defaults are the RoboTwin contract: three RGB cameras, absolute
    Joint14 observations supplied by the dataset, H15 executed EEF16 chunks,
    and H50 PI0.5 predictions.
    """

    action_dim: int = 16
    action_horizon: int = 50
    executed_action_steps: int = 15
    model_dim: int = 256
    phase_dim: int = 128
    signal_dim: int = 256
    task_dim: int = 128
    goal_dim: int = 2048
    num_views: int = 3
    task_count: int = 0
    num_mamba_layers: int = 4
    mamba_state_dim: int = 16
    mamba_conv_width: int = 4
    mamba_expand: int = 2
    cross_attention_heads: int = 4
    dropout: float = 0.1
    image_size: int = 224
    vision_pretrained: bool = True
    ema_decay: float = 0.99
    use_effect_stream: bool = True
    # ``pre`` preserves the original checkpoint tensor shapes and semantics.
    # ``phase`` is an explicitly post-H15 replanning objective: its action
    # head receives only the normalized exported phase token.
    action_prediction_context: Literal["pre", "phase"] = "pre"
    use_mamba: bool = True
    goal_dropout: float = 0.25
    vision_microbatch_size: int = 16

    @classmethod
    def from_zeva_config(
        cls,
        config: ZevaConfig | TransitionEncoderV2Config,
        **overrides: Any,
    ) -> TransitionEncoderV2Config:
        """Convert the existing ZeVA config without changing its file/API."""

        if isinstance(config, cls):
            values = {name: getattr(config, name) for name in cls.__dataclass_fields__}
        else:
            values = {
                name: getattr(config, name)
                for name in cls.__dataclass_fields__
                if hasattr(config, name)
            }
        values.update(overrides)
        # ZevaConfig predates the explicit H15 field and uses 50 as the PI
        # horizon.  Keep the v2 contract explicit rather than guessing from a
        # checkpoint at runtime.
        values.setdefault("executed_action_steps", 15)
        return cls(**values)


@dataclass
class TransitionEncodingV2:
    """Outputs consumed by global, phase and causal PI0.5 prompt adapters."""

    global_prompt: torch.Tensor
    global_context: torch.Tensor
    phase_token: torch.Tensor
    causal_signal: torch.Tensor
    phase_progress: torch.Tensor
    task_embedding: torch.Tensor
    predicted_effect: torch.Tensor
    target_effect: torch.Tensor
    predicted_action: torch.Tensor
    target_action: torch.Tensor
    causal_target_signal: torch.Tensor
    pre_context: torch.Tensor
    post_context: torch.Tensor
    initial_phase_token: torch.Tensor
    initial_phase_progress: torch.Tensor
    task_prototypes: torch.Tensor | None = None

    @property
    def global_task_embedding(self) -> torch.Tensor:
        """Compatibility alias for code that used the v1 output name."""

        return self.global_prompt


@dataclass
class TransitionEncoderV2State:
    """Deployment state for recursive H15 transitions.

    Production retains three Mamba caches and a running task-token sum; its
    memory does not grow with episode length. Raw history fields are used only
    by the CPU-only convolution reference. No episode/task oracle is stored.
    """

    goal_embedding: torch.Tensor | None
    before_images: torch.Tensor | None
    after_images: torch.Tensor | None
    actions: torch.Tensor | None
    transition_count: int = 0
    initial_token: torch.Tensor | None = None
    task_sum: torch.Tensor | None = None
    visual_cache: Any = None
    action_cache: Any = None
    effect_cache: Any = None


class _CausalMambaLayer(nn.Module):
    """Mamba layer with a small CPU-safe fallback for unit tests."""

    def __init__(self, config: TransitionEncoderV2Config, layer_idx: int):
        super().__init__()
        self.norm = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.mamba = None
        if config.use_mamba and Mamba is None:
            raise ImportError("Production ZTE v2 requires mamba_ssm; use_mamba=False is test-only.")
        if config.use_mamba and Mamba is not None:
            self.mamba = Mamba(
                d_model=config.model_dim,
                d_state=config.mamba_state_dim,
                d_conv=config.mamba_conv_width,
                expand=config.mamba_expand,
                layer_idx=layer_idx,
            )
        else:
            # This branch is only for environments without the optional CUDA
            # extension.  It remains causal (left padding) and keeps shape
            # tests usable on a laptop; production configs use Mamba.
            self.conv = nn.Conv1d(
                config.model_dim,
                config.model_dim,
                kernel_size=config.mamba_conv_width,
                groups=config.model_dim,
            )
            self.in_proj = nn.Linear(config.model_dim, config.model_dim * 2)
            self.out_proj = nn.Linear(config.model_dim, config.model_dim)

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        residual = x
        normalized = self.norm(x)
        if self.mamba is not None:
            update = self.mamba(normalized, inference_params=inference_params)
        else:
            gate, value = self.in_proj(normalized).chunk(2, dim=-1)
            value = self.conv(
                F.pad(value.transpose(1, 2), (self.conv.kernel_size[0] - 1, 0))
            ).transpose(1, 2)
            update = self.out_proj(torch.tanh(value) * torch.sigmoid(gate))
        return residual + self.dropout(update)


class _CausalMambaStack(nn.Module):
    def __init__(self, config: TransitionEncoderV2Config, layer_offset: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _CausalMambaLayer(config, layer_offset + index)
                for index in range(config.num_mamba_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.model_dim)

    def forward(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        return self.final_norm(x)


class CausalTransitionEncoderV2(nn.Module):
    """Three-stream causal Mamba transition encoder.

    ``images_before`` and ``images_after`` can be a width-concatenated
    RoboTwin canvas ``[B,T,3,H,3W]`` or explicit views ``[B,T,3,3,H,W]``.
    Actions must be ``[B,T,15,16]`` (or a single-transition equivalent).
    ``forward`` predicts the next H15 chunk as an auxiliary representation
    target. The separate PI policy retains its H50 prediction contract.
    """

    def __init__(
        self,
        config: TransitionEncoderV2Config | ZevaConfig | None = None,
        **config_overrides: Any,
    ):
        super().__init__()
        if config is None:
            config = TransitionEncoderV2Config(**config_overrides)
        elif config_overrides:
            config = TransitionEncoderV2Config.from_zeva_config(config, **config_overrides)
        else:
            config = TransitionEncoderV2Config.from_zeva_config(config)
        self.config = config
        if config.action_horizon < config.executed_action_steps:
            raise ValueError("action_horizon must be >= executed_action_steps.")
        if config.action_prediction_context not in ("pre", "phase"):
            raise ValueError(
                "action_prediction_context must be 'pre' or 'phase', "
                f"got {config.action_prediction_context!r}."
            )
        if config.num_views != 3:
            raise ValueError("RoboTwin v2 requires the head/left-wrist/right-wrist three-view contract.")
        if config.model_dim % config.cross_attention_heads:
            raise ValueError("model_dim must be divisible by cross_attention_heads.")

        self.vision_encoder = LightweightVisionEncoder(
            output_dim=config.model_dim,
            pretrained=config.vision_pretrained,
            num_views=config.num_views,
        )
        self.target_vision_encoder = LightweightVisionEncoder(
            output_dim=config.model_dim,
            pretrained=False,
            num_views=config.num_views,
        )
        self.target_vision_encoder.load_state_dict(self.vision_encoder.state_dict())
        for parameter in self.target_vision_encoder.parameters():
            parameter.requires_grad = False

        self.action_step_projector = nn.Linear(config.action_dim, config.model_dim)
        self.action_step_norm = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.action_position = nn.Parameter(
            torch.randn(1, 1, config.executed_action_steps, config.model_dim) * 0.02
        )

        self.goal_projector = nn.Sequential(
            nn.Linear(config.goal_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.initial_projector = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.initial_token = nn.Parameter(torch.randn(1, 1, config.model_dim) * 0.02)

        # The three recurrent streams use disjoint layer indices.  This is
        # important for Mamba's inference cache and makes the architecture
        # visibly different from the old shared/mean-pooled CTE.
        layers = config.num_mamba_layers
        self.visual_stream = _CausalMambaStack(config, layer_offset=0)
        self.action_stream = _CausalMambaStack(config, layer_offset=layers)
        self.effect_stream = _CausalMambaStack(config, layer_offset=2 * layers)

        heads = config.cross_attention_heads
        self.visual_to_action = nn.MultiheadAttention(
            config.model_dim, heads, dropout=config.dropout, batch_first=True
        )
        self.action_to_visual = nn.MultiheadAttention(
            config.model_dim, heads, dropout=config.dropout, batch_first=True
        )
        self.action_readout = nn.MultiheadAttention(
            config.model_dim, heads, dropout=config.dropout, batch_first=True
        )
        self.effect_to_pre = nn.MultiheadAttention(
            config.model_dim, heads, dropout=config.dropout, batch_first=True
        )
        self.pre_fusion = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.post_fusion = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )

        self.effect_encoder = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.effect_target_projector = nn.Sequential(
            nn.Linear(config.model_dim, config.signal_dim),
            nn.LayerNorm(config.signal_dim),
        )
        # This is a fixed target coordinate, not an accidentally unused online
        # head. The predictor is trained against its detached outputs.
        self.effect_target_projector.requires_grad_(False)
        self.predicted_effect_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        # Always initialize the legacy-shaped head first.  This keeps all
        # subsequent shared parameters bitwise identical between ``pre`` and
        # ``phase`` models created with the same seed.  The optional phase
        # input projection is initialized in a forked RNG context so adding
        # this experiment does not shift global initialization state.
        self.predicted_action_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.executed_action_steps * config.action_dim),
        )
        if config.action_prediction_context == "phase":
            with torch.random.fork_rng(devices=[]):
                self.predicted_action_head[0] = nn.Linear(config.phase_dim, config.model_dim)
        self.phase_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.phase_dim),
        )
        self.causal_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.signal_dim),
        )
        self.task_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.task_dim),
        )
        self.progress_head = nn.Linear(config.model_dim, 1)
        self.task_prototypes = (
            nn.Parameter(torch.randn(config.task_count, config.task_dim) * 0.02)
            if config.task_count > 0
            else None
        )

    def update_ema(self) -> None:
        """Update the target visual encoder after each optimizer step."""

        with torch.no_grad():
            decay = self.config.ema_decay
            for online, target in zip(
                self.vision_encoder.parameters(), self.target_vision_encoder.parameters(), strict=True
            ):
                target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)
            for online, target in zip(
                self.vision_encoder.buffers(), self.target_vision_encoder.buffers(), strict=True
            ):
                target.copy_(online)

    def train(self, mode: bool = True):
        super().train(mode)
        # BatchNorm across B*T would let a prediction see future frames and
        # make valid outputs depend on padded frames. Keep pretrained running
        # statistics, while retaining gradients for visual weights/affines.
        for module in self.vision_encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.target_vision_encoder.eval()
        return self

    @staticmethod
    def _to_bcthw(images: torch.Tensor) -> torch.Tensor:
        """Canonicalise one image tensor to ``[B,T,3,H,W]``."""

        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim == 4:
            if images.shape[1] == 3:
                images = images.unsqueeze(1)
            elif images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2).unsqueeze(1)
            else:
                raise ValueError(f"Expected BCHW/BHWC image, got {tuple(images.shape)}.")
        elif images.ndim == 5:
            if images.shape[2] == 3:  # BTCHW
                pass
            elif images.shape[-1] == 3:  # BTHWC
                images = images.permute(0, 1, 4, 2, 3)
            elif images.shape[1] == 3:  # BVCHW is a single-time multi-view tensor.
                images = images.unsqueeze(1)
            else:
                raise ValueError(f"Expected BTCHW/BTHWC or BVCHW image, got {tuple(images.shape)}.")
        else:
            raise ValueError(f"Expected image rank 3, 4 or 5, got {tuple(images.shape)}.")
        if images.shape[2] != 3:
            raise ValueError(f"Image channel dimension must be 3, got {tuple(images.shape)}.")
        return images.contiguous().to(dtype=torch.float32)

    def _canonical_images(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``[B,T,V,3,H,W]`` while preserving camera order."""

        if images.ndim == 6:
            if images.shape[2] != self.config.num_views or images.shape[3] != 3:
                raise ValueError(
                    f"Explicit views must be [B,T,{self.config.num_views},3,H,W], got {tuple(images.shape)}."
                )
            views = images
        else:
            canvas = self._to_bcthw(images)
            if canvas.shape[-1] % self.config.num_views:
                raise ValueError(
                    f"Width {canvas.shape[-1]} is not divisible by {self.config.num_views} cameras."
                )
            width = canvas.shape[-1] // self.config.num_views
            views = canvas.reshape(
                canvas.shape[0], canvas.shape[1], 3, canvas.shape[-2], self.config.num_views, width
            ).permute(0, 1, 4, 2, 3, 5)
        views = views / 255.0 if views.detach().amax() > 2.0 else views
        if views.detach().amin() < 0.0:
            views = (views + 1.0) / 2.0
        views = views.clamp(0.0, 1.0)
        mean = views.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 1, 3, 1, 1)
        std = views.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 1, 3, 1, 1)
        views = (views - mean) / std
        if views.shape[-2:] != (self.config.image_size, self.config.image_size):
            views = F.interpolate(
                views.flatten(0, 2),
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            ).view(views.shape[0], views.shape[1], views.shape[2], 3, self.config.image_size, self.config.image_size)
        return views.contiguous()

    def _encode_visuals(
        self, images_before: torch.Tensor, images_after: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        before = self._canonical_images(images_before)
        after = self._canonical_images(images_after)
        if before.shape[:2] != after.shape[:2] or before.shape[2:] != after.shape[2:]:
            raise ValueError("images_before and images_after must have the same [B,T,V,3,H,W] shape.")
        batch_size, sequence_length = before.shape[:2]
        before_flat = before.flatten(0, 1)
        after_flat = after.flatten(0, 1)
        microbatch = max(1, self.config.vision_microbatch_size)
        online_chunks = []
        for chunk in before_flat.split(microbatch):
            if self.training and torch.is_grad_enabled():
                encoded = checkpoint(self.vision_encoder, chunk, use_reentrant=False)
            else:
                encoded = self.vision_encoder(chunk)
            online_chunks.append(encoded)
        online_before = torch.cat(online_chunks).view(batch_size, sequence_length, -1)
        self.target_vision_encoder.eval()
        with torch.no_grad():
            target_before = torch.cat([
                self.target_vision_encoder(chunk) for chunk in before_flat.split(microbatch)
            ]).view(batch_size, sequence_length, -1)
            target_after = torch.cat([
                self.target_vision_encoder(chunk) for chunk in after_flat.split(microbatch)
            ]).view(batch_size, sequence_length, -1)
        return online_before, target_before.detach(), target_after.detach()

    def _canonical_actions(self, actions: torch.Tensor, sequence_length: int) -> torch.Tensor:
        actions = actions[..., : self.config.action_dim].to(dtype=torch.float32)
        if actions.ndim == 2:
            actions = actions[:, None, None, :]
        elif actions.ndim == 3:
            if sequence_length == 1:
                actions = actions[:, None, :, :]
            elif actions.shape[1] == sequence_length:
                actions = actions[:, :, None, :]
            else:
                raise ValueError(f"Action sequence length {actions.shape[1]} != {sequence_length}.")
        elif actions.ndim != 4:
            raise ValueError(f"Expected BA, BTA or BTKA actions, got {tuple(actions.shape)}.")
        if actions.shape[1] != sequence_length:
            raise ValueError(f"Action sequence length {actions.shape[1]} != {sequence_length}.")
        if actions.shape[2] == 1:
            actions = actions.expand(-1, -1, self.config.executed_action_steps, -1)
        if actions.shape[2] != self.config.executed_action_steps:
            raise ValueError(
                f"v2 requires exactly H{self.config.executed_action_steps} executed actions, "
                f"got H{actions.shape[2]}."
            )
        if actions.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected EEF{self.config.action_dim}, got {actions.shape[-1]}.")
        return actions

    def _encode_action_steps(self, actions: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, chunk_length, _ = actions.shape
        steps = self.action_step_projector(actions)
        steps = steps + self.action_position[:, :, :chunk_length].to(dtype=steps.dtype)
        steps = self.action_step_norm(steps)
        encoded = self.action_stream(steps.flatten(1, 2))
        return encoded.view(batch_size, sequence_length, chunk_length, -1)

    def _initial_state(
        self, initial_visual: torch.Tensor, goal_embedding: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if goal_embedding is None:
            goal_embedding = initial_visual.new_zeros((initial_visual.shape[0], self.config.goal_dim))
        if goal_embedding.ndim != 2 or goal_embedding.shape != (
            initial_visual.shape[0], self.config.goal_dim
        ):
            raise ValueError(
                f"goal_embedding must be [B,{self.config.goal_dim}], got {tuple(goal_embedding.shape)}."
            )
        goal = self.goal_projector(goal_embedding.to(dtype=initial_visual.dtype))
        b0 = self.initial_projector(torch.cat([initial_visual, goal], dim=-1))
        return b0.unsqueeze(1) + self.initial_token.to(dtype=b0.dtype), goal_embedding

    @staticmethod
    def _readout_action(
        visual_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        visual_to_action: nn.MultiheadAttention,
        action_to_visual: nn.MultiheadAttention,
        action_readout: nn.MultiheadAttention,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-attend one transition while retaining H15 token order."""

        # Querying the ordered H15 action tokens with the visual state keeps a
        # per-step representation; no action-step mean is ever taken.
        visual_query = visual_tokens.unsqueeze(1)
        visual_context, _ = visual_to_action(visual_query, action_tokens, action_tokens)
        action_context, _ = action_to_visual(action_tokens, visual_query, visual_query)
        # Attention to one visual key alone is identical for every action
        # query. Retain the ordered action residual before the readout.
        action_context = action_tokens + action_context
        readout_query = visual_tokens.new_zeros((visual_tokens.shape[0], 1, visual_tokens.shape[-1]))
        action_context, _ = action_readout(
            readout_query,
            torch.cat([action_context, visual_context], dim=1),
            torch.cat([action_context, visual_context], dim=1),
        )
        return visual_context[:, 0], action_context[:, 0]

    def _monotonic_progress(self, hidden: torch.Tensor) -> torch.Tensor:
        # Ordering must be learned and measured, never guaranteed by cumsum.
        return torch.sigmoid(self.progress_head(hidden)).squeeze(-1)

    def forward(
        self,
        images_before: torch.Tensor,
        executed_actions: torch.Tensor,
        images_after: torch.Tensor,
        goal_embedding: torch.Tensor | None = None,
        *,
        goal_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> TransitionEncodingV2:
        before_features, target_before, target_after = self._encode_visuals(images_before, images_after)
        batch_size, sequence_length = before_features.shape[:2]
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, sequence_length, device=before_features.device, dtype=torch.bool)
        if valid_mask.shape != (batch_size, sequence_length) or not valid_mask[:, 0].all():
            raise ValueError("valid_mask must be [B,T] with at least one valid transition per episode.")
        if (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any():
            raise ValueError("Only right-padded episode masks are supported.")
        actions = self._canonical_actions(executed_actions, sequence_length)
        if goal_mask is not None:
            goal_mask = goal_mask.to(device=before_features.device, dtype=torch.bool).reshape(batch_size)
            if goal_embedding is None:
                raise ValueError("goal_mask requires goal_embedding.")
            goal_embedding = goal_embedding.masked_fill(goal_mask[:, None], 0.0)

        target_effect = (target_after - target_before).detach()
        effect_features = self.effect_encoder(target_effect)
        if not self.config.use_effect_stream:
            effect_features = torch.zeros_like(effect_features)

        current_action_steps = self._encode_action_steps(actions)
        b0, goal_value = self._initial_state(before_features[:, 0], goal_embedding)
        visual_sequence = self.visual_stream(torch.cat([b0, before_features], dim=1))
        visual_transitions = visual_sequence[:, 1:]
        effect_sequence = self.effect_stream(
            torch.cat([torch.zeros_like(b0), effect_features], dim=1)
        )
        effect_transitions = effect_sequence[:, 1:]

        # Attention is local to each transition: combine B*T for parallel
        # execution without permitting attention across future transitions.
        visual_context, action_context = self._readout_action(
            visual_transitions.flatten(0, 1),
            current_action_steps.flatten(0, 1),
            self.visual_to_action,
            self.action_to_visual,
            self.action_readout,
        )
        visual_context = visual_context.view(batch_size, sequence_length, -1)
        action_context = action_context.view(batch_size, sequence_length, -1)

        pre_context = self.pre_fusion(torch.cat([visual_context, action_context], dim=-1))
        # The effect query is post-transition by construction.  It never feeds
        # ``pre_context`` or the forward-effect head.  The optional phase
        # action head receives only the normalized exported phase token below;
        # it never receives a post-context bypass.
        effect_query = effect_transitions.flatten(0, 1).unsqueeze(1)
        keys = torch.stack([visual_context, action_context], dim=2).flatten(0, 1)
        effect_context, _ = self.effect_to_pre(effect_query, keys, keys)
        effect_context = (effect_context + effect_query).view(batch_size, sequence_length, -1)
        post_context = self.post_fusion(torch.cat([effect_context, pre_context], dim=-1))

        predicted_effect = self.predicted_effect_head(pre_context)
        phase_token = F.normalize(self.phase_head(post_context), dim=-1)
        action_prediction_input = (
            pre_context
            if self.config.action_prediction_context == "pre"
            else phase_token
        )
        predicted_action = self.predicted_action_head(action_prediction_input).view(
            batch_size,
            sequence_length,
            self.config.executed_action_steps,
            self.config.action_dim,
        )
        causal_signal = F.normalize(self.causal_head(post_context), dim=-1)
        causal_target_signal = F.normalize(self.effect_target_projector(target_effect), dim=-1)
        task_embedding = F.normalize(self.task_head(post_context), dim=-1)
        phase_progress = self._monotonic_progress(post_context)

        weights = valid_mask.to(task_embedding.dtype).unsqueeze(-1)
        task_pool = F.normalize((task_embedding * weights).sum(dim=1) / weights.sum(dim=1), dim=-1)
        # The exported global token is exactly the supervised pooled feature;
        # an extra unsupervised projector here would export a random coordinate.
        global_prompt = task_pool
        # Any learned language/trajectory fusion belongs to Stage2 and must
        # receive its loss. Export the unprojected coordinates from Stage1.
        global_context = torch.cat([task_pool, goal_value.to(dtype=task_pool.dtype)], dim=-1)
        initial_phase = F.normalize(self.phase_head(b0[:, 0]), dim=-1)

        return TransitionEncodingV2(
            global_prompt=global_prompt,
            global_context=global_context,
            phase_token=phase_token,
            causal_signal=causal_signal,
            phase_progress=phase_progress,
            task_embedding=task_embedding,
            predicted_effect=predicted_effect,
            target_effect=target_effect,
            predicted_action=predicted_action,
            target_action=actions,
            causal_target_signal=causal_target_signal,
            pre_context=pre_context,
            post_context=post_context,
            initial_phase_token=initial_phase,
            initial_phase_progress=before_features.new_zeros(batch_size),
            task_prototypes=(
                F.normalize(self.task_prototypes, dim=-1)
                if self.task_prototypes is not None
                else None
            ),
        )

    @torch.no_grad()
    def initialize_phase_state(
        self,
        image: torch.Tensor,
        goal_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, TransitionEncoderV2State]:
        canonical = self._canonical_images(image)
        visual = self.vision_encoder(canonical[:, 0])
        b0, goal = self._initial_state(visual, goal_embedding)
        phase = F.normalize(self.phase_head(b0[:, 0]), dim=-1)
        return phase, TransitionEncoderV2State(
            goal_embedding=goal,
            before_images=None,
            after_images=None,
            actions=None,
            transition_count=0,
            initial_token=b0,
        )

    @torch.no_grad()
    def initialize_phase(
        self,
        image: torch.Tensor,
        goal_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        phase, _ = self.initialize_phase_state(image, goal_embedding)
        return phase

    @torch.no_grad()
    def forward_step(
        self,
        image_before: torch.Tensor,
        executed_actions: torch.Tensor,
        image_after: torch.Tensor,
        state: TransitionEncoderV2State,
    ) -> tuple[TransitionEncodingV2, TransitionEncoderV2State]:
        if self.training:
            raise RuntimeError("Stateful deployment requires eval() with dropout disabled.")
        if self.config.use_mamba:
            return self._cached_forward_step(image_before, executed_actions, image_after, state)
        # Store raw images. Caching normalized images here and sending them
        # through forward would normalize a second time on every replan.
        before = image_before.unsqueeze(1) if image_before.ndim == 4 else image_before
        after = image_after.unsqueeze(1) if image_after.ndim == 4 else image_after
        actions = self._canonical_actions(executed_actions, sequence_length=before.shape[1])
        if state.before_images is None:
            state.before_images = before
            state.after_images = after
            state.actions = actions
        else:
            state.before_images = torch.cat([state.before_images, before], dim=1)
            state.after_images = torch.cat([state.after_images, after], dim=1)
            state.actions = torch.cat([state.actions, actions], dim=1)
        state.transition_count += int(before.shape[1])
        outputs = self(
            state.before_images,
            state.actions,
            state.after_images,
            state.goal_embedding,
        )
        return outputs, state

    @staticmethod
    def _advance_cache(stream, tokens, cache):
        if cache is None:
            cache = InferenceParams(max_seqlen=1_000_000, max_batch_size=tokens.shape[0])
        if cache.seqlen_offset == 0:
            result = stream(tokens, inference_params=cache)
            cache.seqlen_offset += tokens.shape[1]
        else:
            # Released Mamba's recurrent kernel accepts one token per call.
            # H15 is a fixed-size loop, independent of prior episode length.
            parts = []
            for token in tokens.split(1, dim=1):
                parts.append(stream(token, inference_params=cache))
                cache.seqlen_offset += 1
            result = torch.cat(parts, dim=1)
        return result, cache

    def _cached_forward_step(self, image_before, executed_actions, image_after, state):
        before, target_before, target_after = self._encode_visuals(image_before, image_after)
        if before.shape[1] != 1 or state.initial_token is None:
            raise ValueError("A cache update requires one H15 transition and initialize_phase_state().")
        batch = before.shape[0]
        actions = self._canonical_actions(executed_actions, sequence_length=1)
        target_effect = (target_after - target_before).detach()
        effect = self.effect_encoder(target_effect)
        if not self.config.use_effect_stream:
            effect = torch.zeros_like(effect)
        b0 = state.initial_token
        visual_input = torch.cat([b0, before], dim=1) if state.transition_count == 0 else before
        effect_input = torch.cat([torch.zeros_like(b0), effect], dim=1) if state.transition_count == 0 else effect
        visual, state.visual_cache = self._advance_cache(self.visual_stream, visual_input, state.visual_cache)
        effect, state.effect_cache = self._advance_cache(self.effect_stream, effect_input, state.effect_cache)
        action_steps = self.action_step_norm(
            self.action_step_projector(actions) + self.action_position.to(dtype=actions.dtype)
        ).flatten(1, 2)
        ordered, state.action_cache = self._advance_cache(self.action_stream, action_steps, state.action_cache)
        visual_context, action_context = self._readout_action(
            visual[:, -1], ordered, self.visual_to_action, self.action_to_visual, self.action_readout,
        )
        visual_context, action_context = visual_context[:, None], action_context[:, None]
        pre = self.pre_fusion(torch.cat([visual_context, action_context], dim=-1))
        effect_query = effect[:, -1:]
        keys = torch.cat([visual_context, action_context], dim=1)
        effect_context, _ = self.effect_to_pre(effect_query, keys, keys)
        post = self.post_fusion(torch.cat([effect_context + effect_query, pre], dim=-1))
        phase_token = F.normalize(self.phase_head(post), dim=-1)
        action_prediction_input = (
            pre if self.config.action_prediction_context == "pre" else phase_token
        )
        task = F.normalize(self.task_head(post), dim=-1)
        state.task_sum = task[:, 0] if state.task_sum is None else state.task_sum + task[:, 0]
        state.transition_count += 1
        pooled = F.normalize(state.task_sum / state.transition_count, dim=-1)
        output = TransitionEncodingV2(
            global_prompt=pooled,
            global_context=torch.cat([pooled, state.goal_embedding.to(dtype=pooled.dtype)], dim=-1),
            phase_token=phase_token,
            causal_signal=F.normalize(self.causal_head(post), dim=-1),
            phase_progress=self._monotonic_progress(post),
            task_embedding=task,
            predicted_effect=self.predicted_effect_head(pre),
            target_effect=target_effect,
            predicted_action=self.predicted_action_head(action_prediction_input).view(
                batch, 1, self.config.executed_action_steps, self.config.action_dim
            ),
            target_action=actions,
            causal_target_signal=F.normalize(self.effect_target_projector(target_effect), dim=-1),
            pre_context=pre,
            post_context=post,
            initial_phase_token=F.normalize(self.phase_head(b0[:, 0]), dim=-1),
            initial_phase_progress=before.new_zeros(batch),
            task_prototypes=F.normalize(self.task_prototypes, dim=-1) if self.task_prototypes is not None else None,
        )
        return output, state

    @torch.no_grad()
    def step(
        self,
        image_before: torch.Tensor,
        executed_actions: torch.Tensor,
        image_after: torch.Tensor,
        state: TransitionEncoderV2State | None = None,
        goal_embedding: torch.Tensor | None = None,
    ) -> tuple[TransitionEncodingV2, TransitionEncoderV2State]:
        if state is None:
            _, state = self.initialize_phase_state(image_before, goal_embedding)
        return self.forward_step(image_before, executed_actions, image_after, state)


# Explicit aliases make the v2 interface discoverable to downstream code while
# leaving the v1 class untouched.
CausalTransitionEncoderV2 = CausalTransitionEncoderV2
CausalEncodingV2 = TransitionEncodingV2
