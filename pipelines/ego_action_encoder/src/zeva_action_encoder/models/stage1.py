"""Stage 1: action-conditioned future-feature reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from zeva_action_encoder.models.univla_blocks import SpatialTransformer, SpatioTemporalTransformer, VectorQuantizer


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: Tensor) -> Tensor:
        return value.view_as(value)

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor]:
        return (-gradient,)


def gradient_reverse(value: Tensor) -> Tensor:
    """Keep the forward value and reverse only its backward gradient."""

    return _GradientReversal.apply(value)


@dataclass(frozen=True)
class Stage1ModelConfig:
    """Minimal Stage 1 architecture configuration.

    Defaults follow the scale of UniVLA's released Stage 1 configuration. Tests
    and early experiments can instantiate a much smaller model.
    """

    visual_dim: int
    action_dim: int
    model_dim: int = 768
    latent_dim: int = 128
    num_environment_tokens: int = 4
    codebook_size: int | None = 16
    encoder_blocks: int = 12
    decoder_blocks: int = 12
    num_heads: int = 12
    dropout: float = 0.0
    code_restart: bool = True
    normalize_bottleneck: bool = False
    camera_motion_max_steps: int = 0
    action_endpoint_adversary: bool = False

    def __post_init__(self) -> None:
        positive = {
            "visual_dim": self.visual_dim,
            "action_dim": self.action_dim,
            "model_dim": self.model_dim,
            "latent_dim": self.latent_dim,
            "num_environment_tokens": self.num_environment_tokens,
            "encoder_blocks": self.encoder_blocks,
            "decoder_blocks": self.decoder_blocks,
            "num_heads": self.num_heads,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.codebook_size is not None and self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive when specified, got {self.codebook_size}")
        if self.codebook_size is None and self.code_restart:
            raise ValueError("continuous bottleneck requires code_restart=False")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.camera_motion_max_steps < 0:
            raise ValueError("camera_motion_max_steps must be non-negative")


@dataclass
class Stage1Output:
    reconstructed_future: Tensor
    environment_tokens: Tensor
    bottleneck_tokens: Tensor
    codebook_embeddings: Tensor | None
    codebook_indices: Tensor | None
    codebook_size: int | None
    camera_motion_prediction: Tensor | None = None
    action_endpoint_prediction: Tensor | None = None


class Stage1Model(nn.Module):
    """A target-aware posterior paired with a target-free decoder.

    Input features are expected to come from a frozen visual encoder. Dataset
    adapters and the visual encoder itself are intentionally outside Stage 1.
    """

    ENVIRONMENT = 0
    VISUAL = 1
    ACTION = 2

    def __init__(self, config: Stage1ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.visual_projection = nn.Sequential(
            nn.LayerNorm(config.visual_dim),
            nn.Linear(config.visual_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )
        self.action_projection = nn.Linear(config.action_dim, config.model_dim)
        self.modality_embedding = nn.Embedding(3, config.model_dim)

        self.environment_queries = nn.Parameter(
            torch.empty(1, 1, config.num_environment_tokens, config.model_dim)
        )
        nn.init.uniform_(self.environment_queries, -1.0, 1.0)

        self.posterior = SpatioTemporalTransformer(
            model_dim=config.model_dim,
            num_blocks=config.encoder_blocks,
            num_heads=config.num_heads,
            dropout=config.dropout,
            causal_temporal=True,
        )
        self.to_codebook = nn.Linear(config.model_dim, config.latent_dim)
        self.vector_quantizer = (
            None
            if config.codebook_size is None
            else VectorQuantizer(
                num_latents=config.codebook_size,
                latent_dim=config.latent_dim,
                code_restart=config.code_restart,
            )
        )

        self.environment_up = nn.Linear(config.latent_dim, config.model_dim)
        # Keep every shared parameter's initialization independent of optional
        # supervision heads so shaped runs can be controlled against the
        # reconstruction-only baseline with the same random seed.
        self.decoder = SpatialTransformer(
            model_dim=config.model_dim,
            out_dim=config.visual_dim,
            num_blocks=config.decoder_blocks,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        flattened_latent_dim = config.num_environment_tokens * config.latent_dim
        self.camera_motion_head = (
            nn.Linear(flattened_latent_dim, config.camera_motion_max_steps * 6)
            if config.camera_motion_max_steps > 0
            else None
        )
        self.action_endpoint_head = (
            nn.Linear(flattened_latent_dim, 14) if config.action_endpoint_adversary else None
        )

    def configure_execution(
        self,
        *,
        attention_backend: str,
        activation_checkpointing: bool,
    ) -> None:
        """Set training execution policy without changing checkpoint weights."""

        self.posterior.set_attention_backend(attention_backend)
        self.decoder.set_attention_backend(attention_backend)
        self.posterior.activation_checkpointing = activation_checkpointing
        self.decoder.activation_checkpointing = activation_checkpointing

    def _type(self, index: int, reference: Tensor) -> Tensor:
        index_tensor = torch.tensor(index, device=reference.device)
        return self.modality_embedding(index_tensor).to(reference.dtype)

    def _action_tokens(
        self,
        action_chunk: Tensor,
        action_dimension_mask: Tensor | None,
    ) -> Tensor:
        if action_dimension_mask is not None:
            action_chunk = action_chunk.masked_fill(~action_dimension_mask, 0.0)
        action_tokens = self.action_projection(action_chunk)
        action_tokens = action_tokens + self._type(self.ACTION, action_tokens)
        return action_tokens

    def encode_environment(
        self,
        start_features: Tensor,
        future_features: Tensor,
        action_chunk: Tensor,
        action_mask: Tensor | None = None,
        action_dimension_mask: Tensor | None = None,
    ) -> Tensor:
        """Infer continuous environment tokens from both observations.

        ``action_chunk`` must already be expressed in the first observation's
        camera coordinate frame. Coordinate conversion belongs to the dataset
        adapter and is intentionally absent from this model.
        """

        action_mask = self._validate_inputs(
            start_features,
            action_chunk,
            action_mask,
            future_features=future_features,
        )
        action_dimension_mask = self._validate_action_dimension_mask(
            action_chunk,
            action_dimension_mask,
            action_mask,
        )
        batch_size, visual_length, _ = start_features.shape

        visual = torch.stack([start_features, future_features], dim=1)
        visual_tokens = self.visual_projection(visual)
        visual_tokens = visual_tokens + self._type(self.VISUAL, visual_tokens)
        action_tokens = self._action_tokens(action_chunk, action_dimension_mask)
        action_tokens = action_tokens.unsqueeze(1).expand(-1, 2, -1, -1)

        environment_queries = self.environment_queries.expand(batch_size, 2, -1, -1)
        environment_queries = environment_queries + self._type(self.ENVIRONMENT, environment_queries)
        posterior_input = torch.cat(
            [environment_queries, visual_tokens, action_tokens],
            dim=2,
        )

        fixed_valid = torch.ones(
            batch_size,
            self.config.num_environment_tokens + visual_length,
            dtype=torch.bool,
            device=start_features.device,
        )
        spatial_valid_mask = torch.cat([fixed_valid, action_mask], dim=1)
        posterior_output = self.posterior(
            posterior_input,
            spatial_valid_mask=spatial_valid_mask,
        )
        future_environment = posterior_output[:, 1, : self.config.num_environment_tokens]
        return self.to_codebook(future_environment)

    def decode_future(
        self,
        start_features: Tensor,
        action_chunk: Tensor,
        bottleneck_tokens: Tensor,
        action_mask: Tensor | None = None,
        action_dimension_mask: Tensor | None = None,
    ) -> Tensor:
        """Reconstruct the future without any target-feature argument."""

        action_mask = self._validate_inputs(start_features, action_chunk, action_mask)
        action_dimension_mask = self._validate_action_dimension_mask(
            action_chunk,
            action_dimension_mask,
            action_mask,
        )
        batch_size, visual_length, _ = start_features.shape
        expected_environment_shape = (
            batch_size,
            self.config.num_environment_tokens,
            self.config.latent_dim,
        )
        if tuple(bottleneck_tokens.shape) != expected_environment_shape:
            raise ValueError(
                "bottleneck_tokens must have shape "
                f"{expected_environment_shape}, got {tuple(bottleneck_tokens.shape)}"
            )

        environment_tokens = self.environment_up(bottleneck_tokens)
        environment_tokens = environment_tokens + self._type(self.ENVIRONMENT, environment_tokens)
        visual_tokens = self.visual_projection(start_features)
        visual_tokens = visual_tokens + self._type(self.VISUAL, visual_tokens)
        action_tokens = self._action_tokens(action_chunk, action_dimension_mask)

        decoder_input = torch.cat(
            [environment_tokens, visual_tokens, action_tokens],
            dim=1,
        )
        fixed_valid = torch.ones(
            batch_size,
            self.config.num_environment_tokens + visual_length,
            dtype=torch.bool,
            device=start_features.device,
        )
        decoder_valid_mask = torch.cat([fixed_valid, action_mask], dim=1)
        decoded = self.decoder(decoder_input, valid_mask=decoder_valid_mask)
        visual_start = self.config.num_environment_tokens
        return decoded[:, visual_start : visual_start + visual_length]

    def forward(
        self,
        start_features: Tensor,
        future_features: Tensor,
        action_chunk: Tensor,
        action_mask: Tensor | None = None,
        action_dimension_mask: Tensor | None = None,
    ) -> Stage1Output:
        environment_tokens = self.encode_environment(
            start_features=start_features,
            future_features=future_features,
            action_chunk=action_chunk,
            action_mask=action_mask,
            action_dimension_mask=action_dimension_mask,
        )
        if self.config.normalize_bottleneck:
            environment_tokens = self._rms_normalize_tokens(environment_tokens)
        if self.vector_quantizer is None:
            bottleneck_tokens = environment_tokens
            codebook_embeddings = None
            codebook_indices = None
        else:
            bottleneck_tokens, codebook_embeddings, _, codebook_indices = self.vector_quantizer(
                environment_tokens
            )
        reconstructed_future = self.decode_future(
            start_features=start_features,
            action_chunk=action_chunk,
            bottleneck_tokens=bottleneck_tokens,
            action_mask=action_mask,
            action_dimension_mask=action_dimension_mask,
        )
        flattened_bottleneck = bottleneck_tokens.flatten(start_dim=1)
        camera_motion_prediction = None
        if self.camera_motion_head is not None:
            camera_motion_prediction = self.camera_motion_head(flattened_bottleneck).reshape(
                len(flattened_bottleneck), self.config.camera_motion_max_steps, 6
            )
        action_endpoint_prediction = None
        if self.action_endpoint_head is not None:
            action_endpoint_prediction = self.action_endpoint_head(
                gradient_reverse(flattened_bottleneck)
            )
        return Stage1Output(
            reconstructed_future=reconstructed_future,
            environment_tokens=environment_tokens,
            bottleneck_tokens=bottleneck_tokens,
            codebook_embeddings=codebook_embeddings,
            codebook_indices=codebook_indices,
            codebook_size=self.config.codebook_size,
            camera_motion_prediction=camera_motion_prediction,
            action_endpoint_prediction=action_endpoint_prediction,
        )

    @staticmethod
    def _rms_normalize_tokens(tokens: Tensor, epsilon: float = 1e-6) -> Tensor:
        """Remove per-token scale without adding learned parameters."""

        inverse_rms = torch.rsqrt(tokens.float().square().mean(dim=-1, keepdim=True) + epsilon)
        return tokens * inverse_rms.to(dtype=tokens.dtype)

    def _validate_inputs(
        self,
        start_features: Tensor,
        action_chunk: Tensor,
        action_mask: Tensor | None,
        *,
        future_features: Tensor | None = None,
    ) -> Tensor:
        if start_features.ndim != 3 or start_features.shape[-1] != self.config.visual_dim:
            raise ValueError(
                f"start_features must have shape [B, N, {self.config.visual_dim}], "
                f"got {tuple(start_features.shape)}"
            )
        if future_features is not None and future_features.shape != start_features.shape:
            raise ValueError(
                "future_features must have the same shape as start_features, got "
                f"{tuple(future_features.shape)} and {tuple(start_features.shape)}"
            )
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"action_chunk must have shape [B, L, {self.config.action_dim}], "
                f"got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[1] == 0:
            raise ValueError("action_chunk must contain at least one token")

        batch_size = start_features.shape[0]
        if action_chunk.shape[0] != batch_size:
            raise ValueError("all inputs must share the batch dimension")
        if action_mask is None:
            action_mask = torch.ones(
                batch_size,
                action_chunk.shape[1],
                dtype=torch.bool,
                device=action_chunk.device,
            )
        elif action_mask.shape != action_chunk.shape[:2] or action_mask.dtype != torch.bool:
            raise ValueError(
                "action_mask must be boolean with shape [B, L], got "
                f"shape={tuple(action_mask.shape)}, dtype={action_mask.dtype}"
            )
        if not action_mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid action token")
        return action_mask

    def _validate_action_dimension_mask(
        self,
        action_chunk: Tensor,
        action_dimension_mask: Tensor | None,
        action_mask: Tensor,
    ) -> Tensor | None:
        if action_dimension_mask is None:
            return None
        if action_dimension_mask.dtype != torch.bool:
            raise ValueError("action_dimension_mask must be boolean")
        if action_dimension_mask.shape == (
            action_chunk.shape[0],
            action_chunk.shape[2],
        ):
            action_dimension_mask = (
                action_dimension_mask[:, None, :].expand_as(action_chunk)
                & action_mask[..., None]
            )
        elif action_dimension_mask.shape != action_chunk.shape:
            raise ValueError(
                "action_dimension_mask must have shape [B, D] or [B, L, D], got "
                f"{tuple(action_dimension_mask.shape)}"
            )
        measured = action_dimension_mask.any(dim=-1)
        if not measured[action_mask].all():
            raise ValueError("every valid action token must contain at least one measured dimension")
        if measured[~action_mask].any():
            raise ValueError("padded action tokens must not expose measured dimensions")
        return action_dimension_mask
