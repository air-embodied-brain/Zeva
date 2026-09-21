from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.zeva.config import ZevaConfig
from openpi.zeva.vision import LightweightVisionEncoder

try:
    from mamba_ssm import Mamba
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:  # pragma: no cover - exercised only in incomplete environments.
    Mamba = None
    InferenceParams = Any


@dataclass
class CausalEncoding:
    """One Zeva action-effect transition encoded in causal latent space."""

    phase_token: torch.Tensor
    causal_signal: torch.Tensor
    phase_progress: torch.Tensor
    task_embedding: torch.Tensor
    predicted_effect: torch.Tensor
    target_effect: torch.Tensor
    predicted_action: torch.Tensor
    target_action: torch.Tensor
    global_task_embedding: torch.Tensor | None = None
    task_prototypes: torch.Tensor | None = None
    initial_phase_token: torch.Tensor | None = None
    initial_phase_progress: torch.Tensor | None = None


class ResidualMambaBlock(nn.Module):
    def __init__(self, config: ZevaConfig, layer_idx: int):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required by Zeva. Install the bundled extension with "
                "`uv pip install --no-build-isolation ./mamba`."
            )
        self.norm = nn.LayerNorm(config.model_dim)
        self.mamba = Mamba(
            d_model=config.model_dim,
            d_state=config.mamba_state_dim,
            d_conv=config.mamba_conv_width,
            expand=config.mamba_expand,
            layer_idx=layer_idx,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, inference_params: InferenceParams | None = None) -> torch.Tensor:
        residual = self.mamba(self.norm(x), inference_params=inference_params)
        return x + self.dropout(residual)


class CausalTransitionEncoder(nn.Module):
    """Mamba CTE for Zeva.

    The encoder consumes the visual state before an executed action chunk, the
    action chunk itself, and the visual effect observed afterwards. Mamba is
    used for both offline trajectory encoding and stateful deployment-time
    recurrence; no GRU state is used.
    """

    def __init__(self, config: ZevaConfig):
        super().__init__()
        self.config = config
        self.vision_encoder = LightweightVisionEncoder(
            output_dim=config.model_dim,
            pretrained=config.vision_pretrained,
            num_views=config.num_views,
        )
        self.target_vision_encoder = copy.deepcopy(self.vision_encoder)
        for parameter in self.target_vision_encoder.parameters():
            parameter.requires_grad = False

        self.action_step_projector = nn.Linear(config.action_dim, config.model_dim)
        self.action_position = nn.Parameter(
            torch.randn(1, 1, config.action_horizon, config.model_dim) * 0.02
        )
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.effect_encoder = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.transition_projector = nn.Sequential(
            nn.Linear(config.model_dim * 3, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.goal_projector = nn.Sequential(
            nn.Linear(config.goal_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.initial_state_projector = nn.Sequential(
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.LayerNorm(config.model_dim),
            nn.SiLU(),
        )
        self.initial_token = nn.Parameter(torch.randn(1, 1, config.model_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [ResidualMambaBlock(config, layer_idx=index) for index in range(config.num_mamba_layers)]
        )
        self.final_norm = nn.LayerNorm(config.model_dim)

        self.phase_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.phase_dim),
        )
        self.signal_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.signal_dim),
        )
        self.progress_head = nn.Linear(config.model_dim, 1)
        # Progress is represented as accumulated positive hazard.  Initializing
        # the per-transition hazard near 0.1 gives a useful trajectory-scale
        # prior while preserving an exact p_0 = 0 and strict monotonicity.
        nn.init.zeros_(self.progress_head.weight)
        nn.init.constant_(self.progress_head.bias, -2.2)
        self.task_head = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.task_dim),
        )
        self.goal_task_head = nn.Sequential(
            nn.Linear(config.goal_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.task_dim),
        )
        self.task_prototypes = (
            nn.Parameter(torch.randn(config.task_count, config.task_dim) * 0.02)
            if config.task_count > 0
            else None
        )
        self.effect_decoder = nn.Sequential(
            nn.Linear(config.signal_dim + config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.action_horizon * config.action_dim),
        )

    def update_ema(self) -> None:
        with torch.no_grad():
            decay = self.config.ema_decay
            for online, target in zip(
                self.vision_encoder.parameters(), self.target_vision_encoder.parameters(), strict=True
            ):
                target.data.mul_(decay).add_(online.data, alpha=1.0 - decay)
            # BatchNorm running statistics are buffers rather than parameters.
            for online, target in zip(
                self.vision_encoder.buffers(), self.target_vision_encoder.buffers(), strict=True
            ):
                target.copy_(online)

    @staticmethod
    def _ensure_bchw(images: torch.Tensor) -> torch.Tensor:
        if images.ndim not in (4, 5):
            raise ValueError(f"Expected BCHW/BHWC or BTCHW/BTHWC images, got {tuple(images.shape)}.")
        if images.shape[-1] == 3:
            order = (0, 3, 1, 2) if images.ndim == 4 else (0, 1, 4, 2, 3)
            images = images.permute(*order)
        channel_axis = 1 if images.ndim == 4 else 2
        if images.shape[channel_axis] != 3:
            raise ValueError(f"Image channel dimension must be 3, got {tuple(images.shape)}.")
        return images.contiguous().to(dtype=torch.float32)

    @staticmethod
    def _normalize_images(images: torch.Tensor) -> torch.Tensor:
        """Map raw uint8, [0, 1], or OpenPI [-1, 1] images to ImageNet space."""
        if images.detach().amax() > 2.0:
            images = images / 255.0
        elif images.detach().amin() < 0.0:
            images = (images + 1.0) / 2.0
        mean_shape = (1,) * (images.ndim - 3) + (3, 1, 1)
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(mean_shape)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(mean_shape)
        return (images.clamp(0.0, 1.0) - mean) / std

    def _resize_images(self, images: torch.Tensor) -> torch.Tensor:
        target_size = (self.config.image_size, self.config.image_size)
        if images.shape[-2:] == target_size:
            return images
        if images.ndim == 5:
            batch_size, sequence_length = images.shape[:2]
            resized = F.interpolate(
                images.flatten(0, 1),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            return resized.view(batch_size, sequence_length, *resized.shape[1:])
        return F.interpolate(images, size=target_size, mode="bilinear", align_corners=False)

    def _split_and_resize_views(self, images: torch.Tensor) -> torch.Tensor:
        """Split a width-concatenated camera canvas before resizing each view."""
        if self.config.num_views == 1:
            return self._resize_images(images)
        if images.shape[-1] % self.config.num_views:
            raise ValueError(
                f"Image width {images.shape[-1]} is not divisible by {self.config.num_views} views."
            )
        views = torch.stack(torch.chunk(images, self.config.num_views, dim=-1), dim=-4)
        prefix = views.shape[:-3]
        resized = F.interpolate(
            views.reshape(-1, *views.shape[-3:]),
            size=(self.config.image_size, self.config.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.view(*prefix, 3, self.config.image_size, self.config.image_size)

    def _encode_images(
        self, images_before: torch.Tensor, images_after: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images_before = self._ensure_bchw(images_before)
        images_after = self._ensure_bchw(images_after)
        images_before = self._normalize_images(images_before)
        images_after = self._normalize_images(images_after)
        images_before = self._split_and_resize_views(images_before)
        images_after = self._split_and_resize_views(images_after)
        sequence = images_before.ndim == 5
        if self.config.num_views > 1:
            sequence = images_before.ndim == 6
        if sequence:
            batch_size, sequence_length = images_before.shape[:2]
            before_flat = images_before.flatten(0, 1)
            after_flat = images_after.flatten(0, 1)
        else:
            batch_size, sequence_length = images_before.shape[0], 1
            before_flat = images_before
            after_flat = images_after

        before_features = self.vision_encoder(before_flat)
        # Both sides of the effect target must come from the same frozen EMA
        # representation. Mixing online-before with EMA-after adds encoder and
        # BatchNorm drift to the physical state change.
        self.target_vision_encoder.eval()
        with torch.no_grad():
            target_before = self.target_vision_encoder(before_flat)
            target_after = self.target_vision_encoder(after_flat)
        if sequence:
            before_features = before_features.view(batch_size, sequence_length, -1)
            target_before = target_before.view(batch_size, sequence_length, -1)
            target_after = target_after.view(batch_size, sequence_length, -1)
        else:
            before_features = before_features.unsqueeze(1)
            target_before = target_before.unsqueeze(1)
            target_after = target_after.unsqueeze(1)
        return before_features, target_before, target_after

    def _canonical_actions(self, executed_actions: torch.Tensor, sequence_length: int) -> torch.Tensor:
        actions = executed_actions[..., : self.config.action_dim].to(dtype=torch.float32)
        if actions.ndim == 2:
            actions = actions[:, None, None, :]
        elif actions.ndim == 3:
            if sequence_length == 1:
                # Online chunks arrive as [B, chunk, A] for one transition.
                actions = actions[:, None, :, :]
            elif actions.shape[1] == sequence_length:
                actions = actions[:, :, None, :]
            else:
                raise ValueError(
                    f"Action sequence length {actions.shape[1]} does not match image sequence length "
                    f"{sequence_length}."
                )
        elif actions.ndim != 4:
            raise ValueError(f"Expected BA, BTA or BTKA actions, got {tuple(actions.shape)}.")
        if actions.shape[1] != sequence_length:
            raise ValueError(
                f"Action sequence length {actions.shape[1]} does not match image sequence length {sequence_length}."
            )
        if actions.shape[2] == 1:
            actions = actions.expand(-1, -1, self.config.action_horizon, -1)
        # PI predicts H50, while Stage 1 and deployment commit only the H15
        # prefix that was actually executed. The action encoder pools over time
        # and therefore supports any non-empty prefix up to the PI horizon.
        if not 1 <= actions.shape[2] <= self.config.action_horizon:
            raise ValueError(
                f"Action chunks must have horizon in [1,{self.config.action_horizon}], "
                f"got {actions.shape[2]}."
            )
        return actions

    def _encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        chunk_length = actions.shape[2]
        steps = self.action_step_projector(actions)
        steps = steps + self.action_position[:, :, :chunk_length].to(dtype=steps.dtype)
        return self.action_encoder(steps).mean(dim=2)

    def _run_mamba(
        self,
        transition_tokens: torch.Tensor,
        inference_params: InferenceParams | None,
    ) -> torch.Tensor:
        hidden = transition_tokens
        for block in self.blocks:
            hidden = block(hidden, inference_params=inference_params)
        if inference_params is not None:
            inference_params.seqlen_offset += transition_tokens.shape[1]
        return self.final_norm(hidden)

    def _initial_state(
        self,
        initial_visual: torch.Tensor,
        goal_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        if goal_embedding is None:
            goal_embedding = initial_visual.new_zeros((initial_visual.shape[0], self.config.goal_dim))
        if goal_embedding.ndim != 2 or goal_embedding.shape != (
            initial_visual.shape[0],
            self.config.goal_dim,
        ):
            raise ValueError(
                f"Goal embedding must be [B,{self.config.goal_dim}], got {tuple(goal_embedding.shape)}."
            )
        goal = self.goal_projector(goal_embedding.to(dtype=initial_visual.dtype))
        state = self.initial_state_projector(torch.cat([initial_visual, goal], dim=-1))
        return state.unsqueeze(1) + self.initial_token.to(dtype=state.dtype)

    def _make_outputs(
        self,
        hidden: torch.Tensor,
        action_features: torch.Tensor,
        target_effect: torch.Tensor,
        target_action: torch.Tensor,
        phase_progress: torch.Tensor,
        predicted_action: torch.Tensor | None = None,
    ) -> CausalEncoding:
        phase_token = F.normalize(self.phase_head(hidden), dim=-1)
        causal_signal = F.normalize(self.signal_head(hidden), dim=-1)
        task_embedding = F.normalize(self.task_head(hidden), dim=-1)
        predicted_effect = self.effect_decoder(torch.cat([causal_signal, action_features], dim=-1))
        if predicted_action is None:
            predicted_action = self.action_decoder(hidden).view(
                *hidden.shape[:2], self.config.action_horizon, self.config.action_dim
            )
        return CausalEncoding(
            phase_token=phase_token,
            causal_signal=causal_signal,
            phase_progress=phase_progress,
            task_embedding=task_embedding,
            predicted_effect=predicted_effect,
            target_effect=target_effect,
            predicted_action=predicted_action,
            target_action=target_action,
            task_prototypes=(
                F.normalize(self.task_prototypes, dim=-1) if self.task_prototypes is not None else None
            ),
        )

    def pool_global_task_embedding(
        self,
        task_embeddings: torch.Tensor,
        goal_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        """Episode-level task coordinate, separated from local phase state."""
        pooled = task_embeddings.mean(dim=1)
        if goal_embedding is not None:
            pooled = pooled + self.goal_task_head(goal_embedding.to(dtype=pooled.dtype))
        return F.normalize(pooled, dim=-1)

    def _monotonic_progress(
        self,
        hidden: torch.Tensor,
        previous_mass: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map positive accumulated transition mass to progress in [0, 1)."""
        increments = F.softplus(self.progress_head(hidden)).squeeze(-1)
        mass = increments.cumsum(dim=1)
        if previous_mass is not None:
            mass = mass + previous_mass[:, None].to(dtype=mass.dtype)
        return 1.0 - torch.exp(-mass), mass[:, -1]

    def forward(
        self,
        images_before: torch.Tensor,
        executed_actions: torch.Tensor,
        images_after: torch.Tensor,
        goal_embedding: torch.Tensor | None = None,
    ) -> CausalEncoding:
        before_features, target_before, target_after = self._encode_images(images_before, images_after)
        target_effect = (target_after - target_before).detach()
        target_action = self._canonical_actions(executed_actions, before_features.shape[1])
        action_features = self._encode_actions(target_action)
        effect_features = self.effect_encoder(target_effect)
        if not self.config.use_effect_stream:
            effect_features = torch.zeros_like(effect_features)
        transition_tokens = self.transition_projector(
            torch.cat([before_features, action_features, effect_features], dim=-1)
        )
        # Match deployment exactly: first seed the recurrent state with
        # F_init(g, s0), then consume action-effect transitions. The
        # transition outputs describe the post-action observations.
        initial_visual = self._initial_state(before_features[:, 0], goal_embedding)
        hidden = self._run_mamba(torch.cat([initial_visual, transition_tokens], dim=1), inference_params=None)
        # Predict action t from B_{t-1}; the current action is never visible to
        # its own reconstruction head.
        predicted_action = self.action_decoder(hidden[:, :-1]).view(
            *target_action.shape[:2], self.config.action_horizon, self.config.action_dim
        )
        phase_progress, _ = self._monotonic_progress(hidden[:, 1:])
        outputs = self._make_outputs(
            hidden[:, 1:],
            action_features,
            target_effect,
            target_action,
            phase_progress,
            predicted_action,
        )
        return CausalEncoding(
            phase_token=outputs.phase_token,
            causal_signal=outputs.causal_signal,
            phase_progress=outputs.phase_progress,
            task_embedding=outputs.task_embedding,
            predicted_effect=outputs.predicted_effect,
            target_effect=outputs.target_effect,
            predicted_action=outputs.predicted_action,
            target_action=outputs.target_action,
            global_task_embedding=self.pool_global_task_embedding(outputs.task_embedding, goal_embedding),
            task_prototypes=outputs.task_prototypes,
            initial_phase_token=F.normalize(self.phase_head(hidden[:, 0]), dim=-1),
            initial_phase_progress=hidden.new_zeros(hidden.shape[0]),
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
    def initialize_phase_state(
        self,
        image: torch.Tensor,
        goal_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, InferenceParams]:
        """Initialize B0 = F_init(g, s0) and the deployment Mamba cache."""
        image = self._normalize_images(self._ensure_bchw(image))
        image = self._split_and_resize_views(image)
        visual = self.vision_encoder(image)
        inference_params = InferenceParams(max_seqlen=4096, max_batch_size=image.shape[0])
        token = self._initial_state(visual, goal_embedding)
        hidden = self._run_mamba(token, inference_params=inference_params)[:, 0]
        inference_params.zeva_pending_action_prediction = self.action_decoder(hidden).view(
            image.shape[0], self.config.action_horizon, self.config.action_dim
        )
        inference_params.zeva_progress_mass = hidden.new_zeros(image.shape[0])
        return F.normalize(self.phase_head(hidden), dim=-1), inference_params

    @torch.no_grad()
    def step(
        self,
        image_before: torch.Tensor,
        executed_actions: torch.Tensor,
        image_after: torch.Tensor,
        inference_params: InferenceParams | None = None,
    ) -> tuple[CausalEncoding, InferenceParams]:
        image_before = self._ensure_bchw(image_before)
        if inference_params is None:
            _, inference_params = self.initialize_phase_state(image_before, goal_embedding=None)
        if inference_params.seqlen_offset >= inference_params.max_seqlen:
            raise ValueError("Zeva causal history exceeded its configured Mamba context length.")
        outputs = self.forward_step(image_before, executed_actions, image_after, inference_params)
        return outputs, inference_params

    @torch.no_grad()
    def forward_step(
        self,
        image_before: torch.Tensor,
        executed_actions: torch.Tensor,
        image_after: torch.Tensor,
        inference_params: InferenceParams,
    ) -> CausalEncoding:
        before_features, target_before, target_after = self._encode_images(image_before, image_after)
        target_effect = target_after - target_before
        target_action = self._canonical_actions(executed_actions, sequence_length=1)
        action_features = self._encode_actions(target_action)
        effect_features = self.effect_encoder(target_effect)
        if not self.config.use_effect_stream:
            effect_features = torch.zeros_like(effect_features)
        transition_tokens = self.transition_projector(
            torch.cat([before_features, action_features, effect_features], dim=-1)
        )
        hidden = self._run_mamba(transition_tokens, inference_params=inference_params)
        predicted_action = getattr(inference_params, "zeva_pending_action_prediction", None)
        if predicted_action is None:
            raise RuntimeError("Mamba state was not initialized with B0 before a ZeVA transition.")
        predicted_action = predicted_action[:, None]
        previous_mass = getattr(inference_params, "zeva_progress_mass", None)
        if previous_mass is None:
            raise RuntimeError("Mamba state was not initialized with ZeVA progress state.")
        phase_progress, progress_mass = self._monotonic_progress(hidden, previous_mass)
        inference_params.zeva_progress_mass = progress_mass
        outputs = self._make_outputs(
            hidden,
            action_features,
            target_effect,
            target_action,
            phase_progress,
            predicted_action,
        )
        inference_params.zeva_pending_action_prediction = self.action_decoder(hidden[:, -1]).view(
            image_before.shape[0], self.config.action_horizon, self.config.action_dim
        )
        return CausalEncoding(
            phase_token=outputs.phase_token[:, -1],
            causal_signal=outputs.causal_signal[:, -1],
            phase_progress=outputs.phase_progress[:, -1],
            task_embedding=outputs.task_embedding[:, -1],
            predicted_effect=outputs.predicted_effect[:, -1],
            target_effect=outputs.target_effect[:, -1],
            predicted_action=outputs.predicted_action[:, -1],
            target_action=outputs.target_action[:, -1],
            global_task_embedding=None,
            task_prototypes=outputs.task_prototypes,
        )
