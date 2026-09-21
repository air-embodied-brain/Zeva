"""Single full-frame RGB resize contract shared by online and cached data."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


STAGE1_IMAGE_HEIGHT = 196
STAGE1_IMAGE_WIDTH = 350


def full_frame_resize_preprocessing_id(image_height: int, image_width: int) -> str:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    return f"full-frame-resize-{image_height}x{image_width}-bilinear-antialias-uint8-v1"


STAGE1_IMAGE_PREPROCESSING_ID = full_frame_resize_preprocessing_id(
    STAGE1_IMAGE_HEIGHT,
    STAGE1_IMAGE_WIDTH,
)


def resize_rgb_frames_to_uint8(
    frames: Sequence[np.ndarray],
    *,
    image_height: int = STAGE1_IMAGE_HEIGHT,
    image_width: int = STAGE1_IMAGE_WIDTH,
) -> Tensor:
    """Resize complete RGB frames without cropping or padding.

    The source and target aspect ratios are deliberately close for EgoDex. The
    complete source field of view is retained; the small ratio mismatch is an
    anisotropic resize rather than hidden cropping or black padding. Quantizing
    once to uint8 makes online and cache paths bit-identical after `/ 255`.
    """

    if not frames:
        raise ValueError("frames must not be empty")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be positive")
    height, width = frames[0].shape[:2]
    if any(frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 for frame in frames):
        raise ValueError("every frame must be an HxWx3 uint8 RGB array")
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all frames in a preprocessing batch must share one resolution")

    images = torch.from_numpy(np.ascontiguousarray(np.stack(frames, axis=0))).permute(0, 3, 1, 2).float()
    if (height, width) != (image_height, image_width):
        images = F.interpolate(
            images,
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return images.round_().clamp_(0.0, 255.0).to(torch.uint8)
