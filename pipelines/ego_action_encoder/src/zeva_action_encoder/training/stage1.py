"""Minimal Stage 1 objective, matching UniVLA's VQ-VAE loss structure."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from zeva_action_encoder.config import Stage1LossConfig
from zeva_action_encoder.models.stage1 import Stage1Output
from zeva_action_encoder.training.hsic import normalized_rbf_hsic


@dataclass
class Stage1Loss:
    total: Tensor
    reconstruction: Tensor
    codebook: Tensor
    commitment: Tensor
    code_usage: Tensor | None
    camera_motion: Tensor
    action_endpoint_adversary: Tensor
    action_endpoint_hsic: Tensor
    bottleneck_variance: Tensor


def compute_stage1_loss(
    output: Stage1Output,
    target_future: Tensor,
    *,
    config: Stage1LossConfig,
    camera_motion_target: Tensor | None = None,
    camera_motion_mask: Tensor | None = None,
    action_endpoint_target: Tensor | None = None,
    action_endpoint_hsic_indices: Tensor | None = None,
) -> Stage1Loss:
    """Compute reconstruction, optional VQ, and linear auxiliary objectives."""

    if output.reconstructed_future.shape != target_future.shape:
        raise ValueError(
            "target_future must match reconstructed_future, got "
            f"{tuple(target_future.shape)} and {tuple(output.reconstructed_future.shape)}"
        )
    reconstruction = F.mse_loss(output.reconstructed_future, target_future)
    if output.codebook_embeddings is None:
        if output.codebook_indices is not None or output.codebook_size is not None:
            raise ValueError("continuous Stage 1 output contains partial codebook metadata")
        codebook = reconstruction.new_zeros(())
        commitment = reconstruction.new_zeros(())
        code_usage = None
    else:
        if output.codebook_indices is None or output.codebook_size is None:
            raise ValueError("quantized Stage 1 output is missing codebook metadata")
        codebook = F.mse_loss(output.codebook_embeddings, output.environment_tokens.detach())
        commitment = F.mse_loss(output.environment_tokens, output.codebook_embeddings.detach())
        unique_codes = torch.unique(output.codebook_indices).numel()
        code_usage = target_future.new_tensor(unique_codes / output.codebook_size)
    camera_motion = _camera_motion_loss(
        output.camera_motion_prediction,
        camera_motion_target,
        camera_motion_mask,
        reference=reconstruction,
    )
    action_endpoint_adversary = _action_endpoint_loss(
        output.action_endpoint_prediction,
        action_endpoint_target,
        enabled=config.action_endpoint_adversary_weight > 0.0,
        reference=reconstruction,
    )
    action_endpoint_hsic = _action_endpoint_hsic_loss(
        output.bottleneck_tokens,
        action_endpoint_target,
        action_endpoint_hsic_indices,
        config=config,
        reference=reconstruction,
    )
    bottleneck_variance = _bottleneck_variance_floor_loss(
        output.bottleneck_tokens,
        floor=config.bottleneck_variance_floor,
        enabled=config.bottleneck_variance_weight > 0.0,
        reference=reconstruction,
    )
    total = (
        reconstruction
        + codebook
        + config.vq_beta * commitment
        + config.camera_motion_weight * camera_motion
        + config.action_endpoint_adversary_weight * action_endpoint_adversary
        + config.action_endpoint_hsic_weight * action_endpoint_hsic
        + config.bottleneck_variance_weight * bottleneck_variance
    )
    return Stage1Loss(
        total=total,
        reconstruction=reconstruction,
        codebook=codebook,
        commitment=commitment,
        code_usage=code_usage,
        camera_motion=camera_motion,
        action_endpoint_adversary=action_endpoint_adversary,
        action_endpoint_hsic=action_endpoint_hsic,
        bottleneck_variance=bottleneck_variance,
    )


def _bottleneck_variance_floor_loss(
    bottleneck_tokens: Tensor,
    *,
    floor: float,
    enabled: bool,
    reference: Tensor,
) -> Tensor:
    """Penalize per-slot, per-coordinate collapse across local samples."""

    if not enabled:
        return reference.new_zeros(())
    if bottleneck_tokens.ndim != 3:
        raise ValueError("bottleneck tokens must have shape [B, N, D]")
    # Each local batch is already large (1024 in the formal run). Keeping token
    # identity separate prevents fixed differences between slots from faking
    # cross-sample variation. FP32 avoids BF16 cancellation near the floor.
    values = bottleneck_tokens.float()
    std = torch.sqrt(values.var(dim=0, correction=0) + 1e-8)
    return F.relu(std.new_tensor(floor) - std).mean()


def _camera_motion_loss(
    prediction: Tensor | None,
    target: Tensor | None,
    mask: Tensor | None,
    *,
    reference: Tensor,
) -> Tensor:
    if prediction is None:
        if target is not None or mask is not None:
            raise ValueError("camera target was supplied without a camera prediction head")
        return reference.new_zeros(())
    if target is None:
        if mask is not None:
            raise ValueError("camera mask requires a camera target")
        # Keep the head in the DDP graph on fixed-camera/source-unsupervised batches.
        return prediction.sum() * 0.0
    if mask is None or mask.dtype != torch.bool:
        raise ValueError("camera supervision requires a boolean camera_motion_mask")
    if target.ndim != 3 or target.shape[-1] != 6 or target.shape[:2] != mask.shape:
        raise ValueError("camera target must have shape [B, L, 6] aligned with its mask")
    if prediction.shape[0] != target.shape[0] or prediction.shape[2] != 6:
        raise ValueError("camera prediction must have shape [B, max_steps, 6]")
    if prediction.shape[1] < target.shape[1]:
        raise ValueError("camera prediction head is shorter than the target chunk")
    if not mask.any(dim=1).all():
        raise ValueError("each camera target must contain at least one valid step")
    per_step = (prediction[:, : target.shape[1]] - target).float().square().mean(dim=-1)
    # First average within each sample, then across samples: long chunks do not
    # receive more objective weight merely because they contain more steps.
    per_sample = (per_step * mask).sum(dim=1) / mask.sum(dim=1)
    return per_sample.mean()


def _action_endpoint_loss(
    prediction: Tensor | None,
    target: Tensor | None,
    *,
    enabled: bool,
    reference: Tensor,
) -> Tensor:
    if not enabled:
        if prediction is not None:
            raise ValueError("action endpoint prediction was produced while its adversary loss is disabled")
        return reference.new_zeros(())
    if prediction is None:
        raise ValueError("action endpoint adversary loss requires a prediction head")
    if target is None:
        raise ValueError("action endpoint adversary head requires a target on every source")
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 14:
        raise ValueError("action endpoint prediction and target must both have shape [B, 14]")
    return F.mse_loss(prediction.float(), target.float())


def _action_endpoint_hsic_loss(
    bottleneck_tokens: Tensor,
    target: Tensor | None,
    indices: Tensor | None,
    *,
    config: Stage1LossConfig,
    reference: Tensor,
) -> Tensor:
    if config.action_endpoint_hsic_weight == 0.0:
        if indices is not None:
            raise ValueError("HSIC indices were supplied while the HSIC objective is disabled")
        return reference.new_zeros(())
    if target is None:
        raise ValueError("action endpoint HSIC requires a target")
    if target.ndim != 2 or target.shape != (bottleneck_tokens.shape[0], 14):
        raise ValueError("action endpoint HSIC target must have shape [B, 14]")
    if indices is None:
        sample_count = min(len(target), config.action_endpoint_hsic_max_samples)
        indices = torch.arange(sample_count, device=target.device)
    if indices.ndim != 1 or indices.dtype != torch.long or len(indices) < 2:
        raise ValueError("action endpoint HSIC indices must be a 1D long tensor with at least two entries")
    if int(indices.min()) < 0 or int(indices.max()) >= len(target):
        raise IndexError("action endpoint HSIC indices are outside the batch")
    if len(indices) > config.action_endpoint_hsic_max_samples:
        raise ValueError("action endpoint HSIC indices exceed the configured sample limit")
    return normalized_rbf_hsic(
        bottleneck_tokens.index_select(0, indices),
        target.index_select(0, indices),
        kernel_scales=config.action_endpoint_hsic_kernel_scales,
    )
