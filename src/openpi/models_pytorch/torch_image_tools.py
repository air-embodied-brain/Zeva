"""Pure-Torch image helpers for H100 training runtimes without JAX."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def resize_with_pad_torch(images: torch.Tensor, height: int, width: int, mode: str = "bilinear"):
    original_ndim = images.ndim
    channels_last = images.shape[-1] <= 4
    if original_ndim == 3:
        images = images.unsqueeze(0)
    if channels_last:
        images = images.permute(0, 3, 1, 2)
    _, _, current_height, current_width = images.shape
    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    values = F.interpolate(images, size=(resized_height, resized_width), mode=mode,
                           align_corners=False if mode == "bilinear" else None)
    if images.dtype == torch.uint8:
        values = values.round().clamp(0, 255).to(torch.uint8)
        fill = 0
    elif images.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        values = values.clamp(-1.0, 1.0)
        fill = -1.0
    else:
        raise ValueError(f"Unsupported image dtype {images.dtype}.")
    pad_h0, extra_h = divmod(height - resized_height, 2)
    pad_w0, extra_w = divmod(width - resized_width, 2)
    values = F.pad(values, (pad_w0, pad_w0 + extra_w, pad_h0, pad_h0 + extra_h), value=fill)
    if channels_last:
        values = values.permute(0, 2, 3, 1)
    return values.squeeze(0) if original_ndim == 3 else values
