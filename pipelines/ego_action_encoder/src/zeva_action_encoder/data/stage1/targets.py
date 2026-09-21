"""Physical auxiliary targets used to shape Stage 1 environment tokens."""

from __future__ import annotations

import torch
from torch import Tensor

from zeva_action_encoder.data.stage1.schema import STAGE1_ACTION_DIM, hand_action_slice


ACTION_ENDPOINT_DIM = 14
CAMERA_MOTION_DIM = 6


def action_endpoint_from_chunk(
    action_chunk_physical: Tensor,
    action_mask: Tensor,
    action_dimension_mask: Tensor,
) -> Tensor:
    """Integrate wrist/gripper increments into a 14D physical endpoint target.

    Each hand contributes translation (3), an SO(3)-composed rotation vector
    (3), and gripper aperture change (1). Fingertips are intentionally absent.
    """

    if action_chunk_physical.ndim != 3 or action_chunk_physical.shape[-1] != STAGE1_ACTION_DIM:
        raise ValueError(
            f"action_chunk_physical must have shape [B, L, {STAGE1_ACTION_DIM}]"
        )
    if action_mask.shape != action_chunk_physical.shape[:2] or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be boolean [B, L]")
    if action_dimension_mask.shape != action_chunk_physical.shape:
        raise ValueError("action_dimension_mask must match action_chunk_physical")
    if action_dimension_mask.dtype != torch.bool:
        raise ValueError("action_dimension_mask must be boolean")
    if not torch.isfinite(action_chunk_physical).all():
        raise ValueError("physical action chunks must be finite")

    hands: list[Tensor] = []
    valid_values = action_mask[..., None]
    batch_size, steps, _ = action_chunk_physical.shape
    for side in ("left", "right"):
        slots = hand_action_slice(side)
        required = torch.cat(
            [
                action_dimension_mask[..., slots.wrist_translation],
                action_dimension_mask[..., slots.wrist_rotation],
                action_dimension_mask[..., slots.gripper],
            ],
            dim=-1,
        )
        if not bool(required[action_mask].all()):
            raise ValueError(f"{side} wrist/gripper endpoint target contains an unmeasured slot")
        translation = (
            action_chunk_physical[..., slots.wrist_translation] * valid_values
        ).sum(dim=1)
        gripper = (action_chunk_physical[..., slots.gripper] * valid_values).sum(dim=1)
        increments = _rotvec_to_rotation_matrix(
            action_chunk_physical[..., slots.wrist_rotation]
        )
        rotation = torch.eye(
            3,
            dtype=action_chunk_physical.dtype,
            device=action_chunk_physical.device,
        ).expand(batch_size, -1, -1)
        for step in range(steps):
            composed = increments[:, step] @ rotation
            rotation = torch.where(action_mask[:, step, None, None], composed, rotation)
        hands.append(torch.cat([translation, _rotation_matrix_to_rotvec(rotation), gripper], dim=-1))
    return torch.cat(hands, dim=-1)


def normalize_auxiliary_target(target: Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> Tensor:
    """Apply checkpoint-owned, per-dimension z-score statistics."""

    if target.shape[-1] != len(mean) or len(mean) != len(std):
        raise ValueError("auxiliary target and normalization statistics have incompatible dimensions")
    mean_tensor = target.new_tensor(mean)
    std_tensor = target.new_tensor(std)
    if not torch.isfinite(mean_tensor).all() or not torch.isfinite(std_tensor).all():
        raise ValueError("auxiliary normalization statistics must be finite")
    if torch.any(std_tensor <= 0):
        raise ValueError("auxiliary normalization std must be positive")
    return (target - mean_tensor) / std_tensor


def _skew(vector: Tensor) -> Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def _rotvec_to_rotation_matrix(rotvec: Tensor) -> Tensor:
    theta_squared = rotvec.square().sum(dim=-1)
    theta = theta_squared.sqrt()
    ordinary = theta_squared > 1e-8
    safe_theta = theta.clamp_min(1e-8)
    sine_scale = torch.where(
        ordinary,
        torch.sin(theta) / safe_theta,
        1.0 - theta_squared / 6.0 + theta_squared.square() / 120.0,
    )
    cosine_scale = torch.where(
        ordinary,
        (1.0 - torch.cos(theta)) / theta_squared.clamp_min(1e-8),
        0.5 - theta_squared / 24.0 + theta_squared.square() / 720.0,
    )
    skew = _skew(rotvec)
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return (
        identity
        + sine_scale[..., None, None] * skew
        + cosine_scale[..., None, None] * (skew @ skew)
    )


def _rotation_matrix_to_rotvec(rotation: Tensor) -> Tensor:
    skew_vector = 0.5 * torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    sine = torch.linalg.vector_norm(skew_vector, dim=-1)
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.atan2(sine, cosine)
    scale = torch.where(sine > 1e-7, angle / sine, torch.ones_like(sine))
    result = skew_vector * scale[..., None]
    near_pi = (sine <= 1e-7) & (cosine < 0.0)
    if bool(near_pi.any()):
        symmetric = (rotation[near_pi] + torch.eye(3, dtype=rotation.dtype, device=rotation.device)) * 0.5
        _, eigenvectors = torch.linalg.eigh(symmetric)
        axis = eigenvectors[..., -1]
        largest = axis.abs().argmax(dim=-1, keepdim=True)
        sign = axis.gather(-1, largest).sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        result[near_pi] = axis * sign * angle[near_pi][..., None]
    return result
