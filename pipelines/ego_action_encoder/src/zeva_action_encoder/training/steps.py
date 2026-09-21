"""Dataset-agnostic training steps for the two encoder stages."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from zeva_action_encoder.configs import Stage1AuxiliaryTargetStats, Stage1LossConfig
from zeva_action_encoder.contracts import Stage1Batch, Stage2Batch, validate_stage1_batch, validate_stage2_batch
from zeva_action_encoder.training.camera_residual import FrozenCameraResidualPredictor
from zeva_action_encoder.training.stage1 import Stage1Loss, compute_stage1_loss
from zeva_action_encoder.training.stage2_runner import Stage2BatchLoss, Stage2LossConfig, compute_stage2_batch_loss


def stage1_train_step(
    *,
    model: nn.Module,
    visual_encoder: nn.Module,
    batch: Stage1Batch,
    loss_config: Stage1LossConfig,
) -> Stage1Loss:
    validate_stage1_batch(batch)
    images = batch["images"]
    with torch.no_grad():
        start, future = visual_encoder.forward_pair(images[:, 0], images[:, 1])
    output = model(
        start,
        future,
        batch["action_chunk"],
        batch["action_mask"],
        batch["action_dimension_mask"],
    )
    return compute_stage1_loss(
        output,
        future,
        config=loss_config,
        camera_motion_target=batch.get("camera_motion_target"),
        camera_motion_mask=batch.get("camera_motion_mask"),
        action_endpoint_target=batch.get("action_endpoint_target"),
    )


def stage2_train_step(
    *,
    model: nn.Module,
    teacher: nn.Module | None,
    visual_encoder: nn.Module,
    batch: Stage2Batch,
    loss_config: Stage2LossConfig,
    source_index: int = 0,
    optimizer_step: int = 0,
    camera_residual_predictor: FrozenCameraResidualPredictor | None = None,
    action_endpoint_stats: Stage1AuxiliaryTargetStats | None = None,
) -> Stage2BatchLoss:
    validate_stage2_batch(batch)
    return compute_stage2_batch_loss(
        distributed_model=model,
        teacher=teacher,
        visual_encoder=visual_encoder,
        batch=dict(batch),
        source_index=source_index,
        config=loss_config,
        optimizer_step=optimizer_step,
        camera_residual_predictor=camera_residual_predictor,
        action_endpoint_stats=action_endpoint_stats,
    )


def optimize_one_step(
    *,
    loss: Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float | None = None,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return float(loss.detach())
