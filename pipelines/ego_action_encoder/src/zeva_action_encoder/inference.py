"""Checkpoint loading and image-pair encoding for the public action encoder."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from zeva_action_encoder.models import TaskEncoder
from zeva_action_encoder.models import TaskEncoderConfig
from zeva_action_encoder.models.vision import DinoV2Config
from zeva_action_encoder.models.vision import FrozenDinoV2

DEFAULT_IMAGE_SIZES = (
    (196, 350),
    (196, 294),
    (210, 280),
    (252, 252),
    (280, 210),
    (294, 196),
    (350, 196),
)


@dataclass(frozen=True)
class EncodedTransitions:
    task_tokens: Tensor
    environment_tokens: Tensor


def load_encoder(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    dino_config: DinoV2Config | None = None,
    dino_backbone: torch.nn.Module | None = None,
    allowed_image_sizes: tuple[tuple[int, int], ...] = DEFAULT_IMAGE_SIZES,
) -> tuple[TaskEncoder, FrozenDinoV2]:
    """Load a task-encoding checkpoint and its frozen visual backbone.

    The versioned checkpoint contains ``model`` and ``model_config``. The
    caller controls where the checkpoint and optional DINO checkout live.
    """

    target = torch.device(device)
    payload = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise TypeError("encoder checkpoint must be a dictionary")
    if payload.get("schema") == "zeva-ego-action-encoder-v1":
        if payload.get("stage") != "task_encoding":
            raise ValueError("inference requires a task-encoding checkpoint")
        raw_config = payload.get("model_config")
    else:
        # Compatibility with the inference-only checkpoint released before the
        # versioned public training contract.
        raw_config = payload.get("stage2_model_config")
    state = payload.get("model")
    if not isinstance(raw_config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint must contain model and task-encoding model configuration")
    config = TaskEncoderConfig(**raw_config)
    model = TaskEncoder(config)
    model.load_state_dict(state, strict=True)
    model.configure_execution(attention_backend="sdpa", activation_checkpointing=False)
    model.to(target).eval()
    vision = (
        FrozenDinoV2(
            dino_config or DinoV2Config(),
            backbone=dino_backbone,
            allowed_image_sizes=allowed_image_sizes,
        )
        .to(target)
        .eval()
    )
    return model, vision


def encode_tensor_pairs(
    pairs: Tensor,
    *,
    model: TaskEncoder,
    vision: FrozenDinoV2,
) -> EncodedTransitions:
    """Encode float RGB pairs shaped ``[B,2,3,H,W]`` in ``[0,1]``."""

    if pairs.ndim != 5 or pairs.shape[1:3] != (2, 3):
        raise ValueError("pairs must have shape [B,2,3,H,W]")
    if not pairs.is_floating_point():
        raise TypeError("pairs must be floating point")
    if not torch.isfinite(pairs).all():
        raise ValueError("pairs contain non-finite values")
    with torch.inference_mode():
        start, future = vision.forward_pair(pairs[:, 0], pairs[:, 1])
        environment, task = model.encode_transition(start, future)
    return EncodedTransitions(task_tokens=task, environment_tokens=environment)


def iter_numpy_pair_batches(
    pairs: NDArray[np.uint8],
    *,
    batch_size: int,
) -> Iterator[NDArray[np.uint8]]:
    """Normalize either supported uint8 pair layout to channels-first batches."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if pairs.dtype != np.uint8 or pairs.ndim != 5 or pairs.shape[1] != 2:
        raise ValueError("pairs must be uint8 [N,2,H,W,3] or [N,2,3,H,W]")
    channels_last = pairs.shape[-1] == 3
    if not channels_last and pairs.shape[2] != 3:
        raise ValueError("pairs do not contain a three-channel RGB axis")
    for first in range(0, len(pairs), batch_size):
        batch = np.array(pairs[first : first + batch_size], copy=True)
        if channels_last:
            batch = batch.transpose(0, 1, 4, 2, 3)
        yield np.ascontiguousarray(batch)
