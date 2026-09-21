"""Small, dependency-free ports of the Stage 1 blocks used by UniVLA.

The module follows the encoder/decoder/VQ structure in
``OpenDriveLab/UniVLA/latent_action_model/genie/modules/blocks.py``. It has
been rewritten to use only PyTorch, avoid device-specific calls, and support
padding masks for variable-length action chunks. UniVLA is Apache-2.0
licensed; see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


SUPPORTED_ATTENTION_BACKENDS = ("manual", "sdpa")


def sinusoidal_position_encoding(x: Tensor) -> Tensor:
    """Return an index-based sinusoidal encoding for ``x.shape[-2]`` tokens."""

    length, dimension = x.shape[-2:]
    position = torch.arange(length, device=x.device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=x.device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    angles = position * frequency.unsqueeze(0)
    encoding = torch.zeros(length, dimension, device=x.device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=x.dtype)


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional key mask and causal masking."""

    sdpa_batch_chunk_size = 60_000

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_backend: str = "manual",
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attention_backend = _validate_attention_backend(attention_backend)
        self.to_qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Dropout(dropout))
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        *,
        valid_mask: Tensor | None = None,
        causal: bool = False,
    ) -> Tensor:
        batch_size, length, model_dim = x.shape
        qkv = self.to_qkv(x).view(batch_size, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if valid_mask is not None:
            if valid_mask.shape != (batch_size, length) or valid_mask.dtype != torch.bool:
                raise ValueError("valid_mask must be boolean with shape [B, S]")
        if self.attention_backend == "sdpa":
            attention_mask = None if valid_mask is None else valid_mask[:, None, None, :]
            is_causal = causal
            if causal and attention_mask is not None:
                causal_mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
                attention_mask = attention_mask & causal_mask.view(1, 1, length, length)
                is_causal = False
            if query.is_cuda:
                # PyTorch 2.7.1's automatic Flash path is unstable for the
                # large batched shapes used here. The efficient fused backend
                # remains linear-memory and is stable on modern accelerators.
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    attended = self._scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attention_mask=attention_mask,
                        is_causal=is_causal,
                    )
            else:
                attended = self._scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attention_mask=attention_mask,
                    is_causal=is_causal,
                )
            attended = attended.transpose(1, 2).reshape(batch_size, length, model_dim)
            return self.to_out(attended)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if causal:
            causal_mask = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], torch.finfo(scores.dtype).min)

        weights = self.attention_dropout(scores.softmax(dim=-1))
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch_size, length, model_dim)
        return self.to_out(attended)

    def set_attention_backend(self, attention_backend: str) -> None:
        self.attention_backend = _validate_attention_backend(attention_backend)

    def _scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attention_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        if query.shape[0] > self.sdpa_batch_chunk_size:
            outputs = []
            for start in range(0, query.shape[0], self.sdpa_batch_chunk_size):
                stop = min(start + self.sdpa_batch_chunk_size, query.shape[0])
                chunk_mask = attention_mask
                if attention_mask is not None and attention_mask.shape[0] == query.shape[0]:
                    chunk_mask = attention_mask[start:stop]
                outputs.append(
                    self._scaled_dot_product_attention_once(
                        query[start:stop],
                        key[start:stop],
                        value[start:stop],
                        attention_mask=chunk_mask,
                        is_causal=is_causal,
                    )
                )
            return torch.cat(outputs, dim=0)
        return self._scaled_dot_product_attention_once(
            query,
            key,
            value,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )

    def _scaled_dot_product_attention_once(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        attention_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale,
        )


class SpatioTemporalBlock(nn.Module):
    """UniVLA-style spatial attention followed by temporal attention."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.spatial_attention = SelfAttention(model_dim, num_heads, dropout)
        self.temporal_attention = SelfAttention(model_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
        )
        self.spatial_norm = nn.LayerNorm(model_dim)
        self.temporal_norm = nn.LayerNorm(model_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        x: Tensor,
        *,
        spatial_valid_mask: Tensor | None = None,
        causal_temporal: bool = True,
    ) -> Tensor:
        batch_size, time_length, spatial_length, model_dim = x.shape

        spatial = self.spatial_norm(x).reshape(batch_size * time_length, spatial_length, model_dim)
        expanded_mask = None
        if spatial_valid_mask is not None:
            expanded_mask = (
                spatial_valid_mask[:, None, :]
                .expand(batch_size, time_length, spatial_length)
                .reshape(batch_size * time_length, spatial_length)
            )
        spatial = self.spatial_attention(spatial, valid_mask=expanded_mask)
        x = x + spatial.view(batch_size, time_length, spatial_length, model_dim)

        temporal = self.temporal_norm(x).permute(0, 2, 1, 3)
        temporal = temporal.reshape(batch_size * spatial_length, time_length, model_dim)
        temporal = self.temporal_attention(temporal, causal=causal_temporal)
        temporal = temporal.view(batch_size, spatial_length, time_length, model_dim).permute(0, 2, 1, 3)
        x = x + temporal

        return x + self.feed_forward(self.feed_forward_norm(x))


class SpatioTemporalTransformer(nn.Module):
    """Two-frame posterior used to infer transition tokens."""

    def __init__(
        self,
        model_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        causal_temporal: bool = True,
        attention_backend: str = "manual",
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [SpatioTemporalBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.causal_temporal = causal_temporal
        self.activation_checkpointing = activation_checkpointing
        self.set_attention_backend(attention_backend)
        self.final_norm = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, *, spatial_valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 4:
            raise ValueError("SpatioTemporalTransformer input must have shape [B, T, S, D]")
        x = x + sinusoidal_position_encoding(x).view(1, 1, x.shape[2], x.shape[3])
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    block,
                    x,
                    spatial_valid_mask=spatial_valid_mask,
                    causal_temporal=self.causal_temporal,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    spatial_valid_mask=spatial_valid_mask,
                    causal_temporal=self.causal_temporal,
                )
        return self.final_norm(x)

    def set_attention_backend(self, attention_backend: str) -> None:
        for block in self.blocks:
            block.spatial_attention.set_attention_backend(attention_backend)
            block.temporal_attention.set_attention_backend(attention_backend)


class SpatialBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attention = SelfAttention(model_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 4 * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * model_dim, model_dim),
        )
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feed_forward_norm = nn.LayerNorm(model_dim)

    def forward(self, x: Tensor, *, valid_mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), valid_mask=valid_mask)
        return x + self.feed_forward(self.feed_forward_norm(x))


class SpatialTransformer(nn.Module):
    """Target-free spatial decoder used to reconstruct future DINO features."""

    def __init__(
        self,
        model_dim: int,
        out_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float = 0.0,
        attention_backend: str = "manual",
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SpatialBlock(model_dim, num_heads, dropout) for _ in range(num_blocks)])
        self.activation_checkpointing = activation_checkpointing
        self.set_attention_backend(attention_backend)
        self.final_norm = nn.LayerNorm(model_dim)
        self.to_output = nn.Linear(model_dim, out_dim)

    def forward(self, x: Tensor, *, valid_mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("SpatialTransformer input must have shape [B, S, D]")
        x = x + sinusoidal_position_encoding(x).unsqueeze(0)
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    block,
                    x,
                    valid_mask=valid_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, valid_mask=valid_mask)
        return self.to_output(self.final_norm(x))

    def set_attention_backend(self, attention_backend: str) -> None:
        for block in self.blocks:
            block.attention.set_attention_backend(attention_backend)


class VectorQuantizer(nn.Module):
    """UniVLA-style straight-through VQ with code-usage accounting."""

    def __init__(self, num_latents: int, latent_dim: int, code_restart: bool = True) -> None:
        super().__init__()
        self.num_latents = num_latents
        self.code_restart = code_restart
        self.codebook = nn.Embedding(num_latents, latent_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_latents, 1.0 / num_latents)
        self.register_buffer(
            "usage",
            torch.zeros(num_latents, dtype=torch.int64),
            persistent=True,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError("VectorQuantizer input must have shape [B, K, D]")
        flat_x = x.reshape(-1, x.shape[-1])
        codebook = self.codebook.weight
        distance = (
            flat_x.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * flat_x @ codebook.t()
        )
        indices = distance.argmin(dim=1).view(x.shape[0], x.shape[1])
        embeddings = self.codebook(indices)
        quantized = x + (embeddings - x).detach()
        return quantized, embeddings, x, indices

    @torch.no_grad()
    def add_global_usage(self, counts: Tensor) -> None:
        """Accumulate one already-synchronized assignment histogram."""

        expected_shape = (self.num_latents,)
        if tuple(counts.shape) != expected_shape or counts.dtype != torch.int64:
            raise ValueError(
                f"global code counts must be int64 with shape {expected_shape}, "
                f"got shape={tuple(counts.shape)}, dtype={counts.dtype}"
            )
        if torch.any(counts < 0):
            raise ValueError("global code counts must be non-negative")
        self.usage.add_(counts.to(device=self.usage.device))

    @torch.no_grad()
    def restart_codes(self, code_indices: Tensor, replacements: Tensor) -> None:
        """Replace selected code rows with encoder-latent representatives."""

        if not self.code_restart:
            raise RuntimeError("code restart is disabled in the model config")
        indices = code_indices.to(device=self.codebook.weight.device, dtype=torch.long)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("code_indices must be a nonempty one-dimensional tensor")
        if torch.any(indices < 0) or torch.any(indices >= self.num_latents):
            raise ValueError("code_indices contain an out-of-range code")
        if torch.unique(indices).numel() != len(indices):
            raise ValueError("code_indices must not contain duplicates")
        expected_shape = (len(indices), self.codebook.embedding_dim)
        if tuple(replacements.shape) != expected_shape:
            raise ValueError(
                f"replacement latents must have shape {expected_shape}, "
                f"got {tuple(replacements.shape)}"
            )
        if not torch.isfinite(replacements).all():
            raise ValueError("replacement latents must be finite")
        self.codebook.weight[indices] = replacements.to(self.codebook.weight)

    @torch.no_grad()
    def reset_usage(self) -> None:
        self.usage.zero_()


def _validate_attention_backend(attention_backend: str) -> str:
    if attention_backend not in SUPPORTED_ATTENTION_BACKENDS:
        raise ValueError(
            f"attention_backend must be one of {SUPPORTED_ATTENTION_BACKENDS}, "
            f"got {attention_backend!r}"
        )
    return attention_backend
