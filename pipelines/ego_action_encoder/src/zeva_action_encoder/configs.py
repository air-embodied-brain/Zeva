"""Small public configuration surface for the encoder objectives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage1LossConfig:
    vq_beta: float = 0.25
    camera_motion_weight: float = 0.0
    action_endpoint_adversary_weight: float = 0.0
    action_endpoint_hsic_weight: float = 0.0
    action_endpoint_hsic_max_samples: int = 256
    action_endpoint_hsic_kernel_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    bottleneck_variance_weight: float = 0.0
    bottleneck_variance_floor: float = 0.05

    def __post_init__(self) -> None:
        weights = (
            self.vq_beta,
            self.camera_motion_weight,
            self.action_endpoint_adversary_weight,
            self.action_endpoint_hsic_weight,
            self.bottleneck_variance_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("Stage 1 loss weights must be non-negative")
        if self.action_endpoint_hsic_max_samples < 2:
            raise ValueError("action_endpoint_hsic_max_samples must be at least two")
        if self.action_endpoint_adversary_weight > 0.0 and self.action_endpoint_hsic_weight > 0.0:
            raise ValueError("action endpoint adversary and HSIC losses are mutually exclusive")


@dataclass(frozen=True)
class Stage1AuxiliaryTargetStats:
    source_id: str
    action_endpoint_mean: tuple[float, ...]
    action_endpoint_std: tuple[float, ...]
    camera_motion_mean: tuple[float, ...] | None = None
    camera_motion_std: tuple[float, ...] | None = None


EnvironmentEncodingLossConfig = Stage1LossConfig
EnvironmentEncodingAuxiliaryTargetStats = Stage1AuxiliaryTargetStats
