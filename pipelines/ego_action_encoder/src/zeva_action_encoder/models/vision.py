"""Frozen DINOv2 feature extraction used before Stage 1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from zeva_action_encoder.data.image import (
    STAGE1_IMAGE_HEIGHT,
    STAGE1_IMAGE_PREPROCESSING_ID,
    STAGE1_IMAGE_WIDTH,
    full_frame_resize_preprocessing_id,
)

DINOV2_IMAGE_PREPROCESSING_ID = STAGE1_IMAGE_PREPROCESSING_ID


@dataclass(frozen=True)
class DinoV2Config:
    """The DINOv2 variant and preprocessing used by UniVLA Stage 1."""

    repository: str = "facebookresearch/dinov2"
    repository_ref: str | None = None
    model_name: str = "dinov2_vitb14_reg"
    preprocessing_id: str | None = None
    image_height: int = STAGE1_IMAGE_HEIGHT
    image_width: int = STAGE1_IMAGE_WIDTH
    patch_size: int = 14
    feature_dim: int = 768
    patch_token_key: str = "x_norm_patchtokens"
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.image_height <= 0 or self.image_width <= 0 or self.patch_size <= 0 or self.feature_dim <= 0:
            raise ValueError("image_height, image_width, patch_size, and feature_dim must be positive")
        if self.image_height % self.patch_size != 0 or self.image_width % self.patch_size != 0:
            raise ValueError("image_height and image_width must be divisible by patch_size")
        expected_preprocessing = full_frame_resize_preprocessing_id(
            self.image_height,
            self.image_width,
        )
        if self.preprocessing_id is None:
            object.__setattr__(self, "preprocessing_id", expected_preprocessing)
        elif self.preprocessing_id != expected_preprocessing:
            raise ValueError(
                "preprocessing_id differs from image geometry: "
                f"actual={self.preprocessing_id!r}, expected={expected_preprocessing!r}"
            )
        for name in ("repository", "model_name", "preprocessing_id", "patch_token_key"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("normalization_mean and normalization_std must contain three values")
        if any(value <= 0.0 for value in self.normalization_std):
            raise ValueError("normalization_std values must be positive")

    @property
    def hub_repository(self) -> str:
        if self.repository_ref is None:
            return self.repository
        return f"{self.repository}:{self.repository_ref}"


class FrozenDinoV2(nn.Module):
    """Extract only normalized patch tokens from a frozen DINOv2 backbone.

    Inputs must be RGB float tensors in ``[0, 1]`` with the configured spatial
    size. Resizing and augmentation remain outside this module. A backbone may
    be injected for unit tests; otherwise it is loaded through ``torch.hub``.
    """

    def __init__(
        self,
        config: DinoV2Config | None = None,
        *,
        backbone: nn.Module | None = None,
        allowed_image_sizes: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DinoV2Config()
        configured_sizes = (
            ((self.config.image_height, self.config.image_width),)
            if allowed_image_sizes is None
            else allowed_image_sizes
        )
        self.allowed_image_sizes = tuple((int(height), int(width)) for height, width in configured_sizes)
        if not self.allowed_image_sizes or len(set(self.allowed_image_sizes)) != len(self.allowed_image_sizes):
            raise ValueError("allowed_image_sizes must be nonempty and unique")
        primary_size = (self.config.image_height, self.config.image_width)
        if primary_size not in self.allowed_image_sizes:
            raise ValueError(f"allowed_image_sizes must include the configured primary size {primary_size}")
        for height, width in self.allowed_image_sizes:
            if height <= 0 or width <= 0 or height % self.config.patch_size or width % self.config.patch_size:
                raise ValueError("every allowed image size must be positive and divisible by DINO patch_size")
        if backbone is None:
            cached_repository = _cached_hub_repository(self.config)
            if cached_repository is not None:
                backbone = torch.hub.load(
                    str(cached_repository),
                    self.config.model_name,
                    source="local",
                )
            else:
                backbone = torch.hub.load(
                    self.config.hub_repository,
                    self.config.model_name,
                    trust_repo=True,
                    skip_validation=True,
                )
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.register_buffer(
            "image_mean",
            torch.tensor(self.config.normalization_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(self.config.normalization_std).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True) -> FrozenDinoV2:
        """Allow the wrapper to follow its parent while keeping DINO in eval mode."""

        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        """Return ``x_norm_patchtokens`` with shape ``[B, N, D]``."""

        self._validate_images(images)
        normalized = (images - self.image_mean.to(images)) / self.image_std.to(images)
        with torch.no_grad():
            features = self.backbone.forward_features(normalized)
        if not isinstance(features, Mapping) or self.config.patch_token_key not in features:
            raise RuntimeError(
                f"DINO backbone must return a mapping containing {self.config.patch_token_key!r}"
            )
        patch_tokens = features[self.config.patch_token_key]
        expected_patches = (
            (images.shape[-2] // self.config.patch_size)
            * (images.shape[-1] // self.config.patch_size)
        )
        expected_shape = (images.shape[0], expected_patches, self.config.feature_dim)
        if tuple(patch_tokens.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected DINO patch-token shape: expected {expected_shape}, "
                f"got {tuple(patch_tokens.shape)}"
            )
        return patch_tokens

    def forward_pair(self, image_t0: Tensor, image_t1: Tensor) -> tuple[Tensor, Tensor]:
        """Extract both frames with one concatenated DINO forward pass."""

        if image_t0.shape != image_t1.shape:
            raise ValueError(
                f"image_t0 and image_t1 must have the same shape, got "
                f"{tuple(image_t0.shape)} and {tuple(image_t1.shape)}"
            )
        batch_size = image_t0.shape[0]
        patch_tokens = self.forward(torch.cat([image_t0, image_t1], dim=0))
        return patch_tokens[:batch_size], patch_tokens[batch_size:]

    def _validate_images(self, images: Tensor) -> None:
        actual_size = tuple(images.shape[-2:]) if images.ndim == 4 else None
        if images.ndim != 4 or images.shape[1] != 3 or actual_size not in self.allowed_image_sizes:
            raise ValueError(
                "images must have shape [B, 3, H, W] with an allowed image size; "
                f"allowed={self.allowed_image_sizes}, got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise TypeError(f"images must be floating point in [0, 1], got {images.dtype}")
        if not torch.isfinite(images).all():
            raise ValueError("images contain non-finite values")


def _cached_hub_repository(config: DinoV2Config) -> Path | None:
    """Resolve a pinned torch.hub checkout without any network operation."""

    if config.repository_ref is None:
        return None
    try:
        owner, repository = config.repository.split("/", maxsplit=1)
    except ValueError:
        return None
    normalized_ref = config.repository_ref.replace("/", "_")
    candidate = Path(torch.hub.get_dir()) / f"{owner}_{repository}_{normalized_ref}"
    return candidate if candidate.is_dir() else None
