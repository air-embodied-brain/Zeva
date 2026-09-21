"""Kernel independence objectives for representation shaping."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


DEFAULT_RBF_SCALES = (0.5, 1.0, 2.0)


def normalized_rbf_hsic(
    first: Tensor,
    second: Tensor,
    *,
    kernel_scales: Sequence[float] = DEFAULT_RBF_SCALES,
    epsilon: float = 1e-8,
) -> Tensor:
    """Return biased normalized HSIC with stop-gradient median RBF bandwidths.

    Inputs share only their sample axis; their remaining shapes may differ. A
    separate multi-scale RBF Gram matrix is built for each input, so this does
    not compare representation coordinates with target coordinates.
    """

    if first.ndim < 2 or second.ndim < 2:
        raise ValueError("HSIC inputs must have a sample axis and at least one feature axis")
    if first.shape[0] != second.shape[0]:
        raise ValueError("HSIC inputs must share the sample count")
    if first.shape[0] < 2:
        raise ValueError("HSIC requires at least two samples")
    scales = tuple(float(value) for value in kernel_scales)
    if not scales or any(value <= 0.0 for value in scales):
        raise ValueError("kernel_scales must contain positive values")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    first_flat = first.float().flatten(start_dim=1)
    second_flat = second.float().flatten(start_dim=1)
    if not torch.isfinite(first_flat).all() or not torch.isfinite(second_flat).all():
        raise ValueError("HSIC inputs must be finite")

    first_kernel = _median_multiscale_rbf(first_flat, scales=scales, epsilon=epsilon)
    second_kernel = _median_multiscale_rbf(second_flat, scales=scales, epsilon=epsilon)
    first_centered = _center_gram(first_kernel)
    second_centered = _center_gram(second_kernel)
    numerator = (first_centered * second_centered).sum()
    denominator = torch.sqrt(
        first_centered.square().sum() * second_centered.square().sum()
    ).clamp_min(epsilon)
    return numerator / denominator


def _median_multiscale_rbf(
    features: Tensor,
    *,
    scales: tuple[float, ...],
    epsilon: float,
) -> Tensor:
    distances = _pairwise_squared_distance(features)
    sample_count = len(features)
    off_diagonal = distances[
        torch.triu_indices(sample_count, sample_count, offset=1, device=features.device).unbind()
    ]
    bandwidth = off_diagonal.median().detach().clamp_min(epsilon)
    kernels = [torch.exp(-distances / (bandwidth * scale)) for scale in scales]
    return torch.stack(kernels, dim=0).mean(dim=0)


def _pairwise_squared_distance(features: Tensor) -> Tensor:
    # Pairwise distances are exactly translation invariant.  Centering first
    # avoids catastrophic cancellation when RMS-normalized latents share a
    # large common component but have very small across-sample variation.
    features = features - features.mean(dim=0, keepdim=True)
    squared_norm = features.square().sum(dim=1, keepdim=True)
    distances = squared_norm + squared_norm.transpose(0, 1) - 2.0 * (features @ features.transpose(0, 1))
    return distances.clamp_min(0.0)


def _center_gram(kernel: Tensor) -> Tensor:
    return kernel - kernel.mean(dim=0, keepdim=True) - kernel.mean(dim=1, keepdim=True) + kernel.mean()
