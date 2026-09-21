"""Tensor contracts consumed by the public training steps."""

from __future__ import annotations

from typing import TypedDict

import torch
from torch import Tensor


class Stage1Batch(TypedDict, total=False):
    images: Tensor
    action_chunk: Tensor
    action_mask: Tensor
    action_dimension_mask: Tensor
    camera_motion_target: Tensor
    camera_motion_mask: Tensor
    action_endpoint_target: Tensor


class Stage2Batch(TypedDict, total=False):
    images: Tensor
    action_chunk: Tensor
    action_chunk_physical: Tensor
    action_mask: Tensor
    action_dimension_mask: Tensor
    gripper_aperture_range: Tensor
    duration_seconds: Tensor
    horizon_steps: Tensor
    camera_endpoint_motion: Tensor
    episode_id: Tensor


# Public names describe the method stages. The original aliases remain so
# checkpoints and downstream imports from the research implementation continue
# to load without conversion.
EnvironmentEncodingBatch = Stage1Batch
TaskEncodingBatch = Stage2Batch


def validate_stage1_batch(batch: Stage1Batch) -> None:
    images = batch["images"]
    action = batch["action_chunk"]
    mask = batch["action_mask"]
    dimensions = batch["action_dimension_mask"]
    if images.ndim != 5 or images.shape[1] != 2:
        raise ValueError("environment-encoding images must have shape [B,2,C,H,W]")
    if action.ndim != 3 or action.shape[0] != images.shape[0]:
        raise ValueError("environment-encoding action_chunk must have shape [B,L,D]")
    if mask.shape != action.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("environment-encoding action_mask must be boolean [B,L]")
    if dimensions.shape != action.shape or dimensions.dtype != torch.bool:
        raise ValueError("environment-encoding action_dimension_mask must be boolean [B,L,D]")


def validate_stage2_batch(batch: Stage2Batch) -> None:
    images = batch["images"]
    action = batch["action_chunk"]
    if images.ndim != 5 or images.shape[1] != 3:
        raise ValueError("task-encoding images must have shape [B,3,C,H,W]")
    if action.ndim != 4 or action.shape[:2] != (images.shape[0], 3):
        raise ValueError("task-encoding action_chunk must have shape [B,3,L,D]")
    mask = batch["action_mask"]
    dimensions = batch["action_dimension_mask"]
    if mask.shape != action.shape[:3] or mask.dtype != torch.bool:
        raise ValueError("task-encoding action_mask must be boolean [B,3,L]")
    if dimensions.shape != action.shape or dimensions.dtype != torch.bool:
        raise ValueError("task-encoding action_dimension_mask must be boolean [B,3,L,D]")


validate_environment_encoding_batch = validate_stage1_batch
validate_task_encoding_batch = validate_stage2_batch
