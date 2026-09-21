"""Common camera-frame action trajectories for cross-embodiment matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from zeva_action_encoder.data.stage1.schema import STAGE1_ACTION_DIM, hand_action_slice


@dataclass(frozen=True)
class ActionDescriptorConfig:
    """Fixed-length sampling contract for relative physical trajectories.

    The descriptor contains, in order, left/right wrist translation and
    rotation trajectories and left/right relative gripper trajectories. Each
    trajectory has ``num_trajectory_points`` samples and is relative to the
    first action state. Duration is deliberately not part of the descriptor;
    it gates which pairs may be compared. The initial all-zero state is not
    stored; a one-point descriptor therefore represents the chunk endpoint.
    """

    num_trajectory_points: int

    def __post_init__(self) -> None:
        if self.num_trajectory_points <= 0:
            raise ValueError("num_trajectory_points must be positive")

    @property
    def descriptor_dim(self) -> int:
        return 14 * self.num_trajectory_points

    @property
    def group_slices(self) -> tuple[tuple[slice, slice], tuple[slice, slice], tuple[slice, slice]]:
        """Return left/right slices for translation, rotation, and gripper."""

        points = self.num_trajectory_points
        per_side = 7 * points
        return (
            (slice(0, 3 * points), slice(per_side, per_side + 3 * points)),
            (slice(3 * points, 6 * points), slice(per_side + 3 * points, per_side + 6 * points)),
            (slice(6 * points, 7 * points), slice(per_side + 6 * points, per_side + 7 * points)),
        )


@dataclass(frozen=True)
class ActionDescriptor:
    values: Tensor
    dimension_mask: Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError("descriptor values must have shape [B, D]")
        if self.dimension_mask.shape != self.values.shape or self.dimension_mask.dtype != torch.bool:
            raise ValueError("descriptor dimension_mask must be boolean [B, D]")
        if not torch.isfinite(self.values).all():
            raise ValueError("descriptor values must be finite")
        if torch.any(self.values.masked_select(~self.dimension_mask) != 0.0):
            raise ValueError("invalid descriptor dimensions must be exactly zero")


def build_action_descriptor(
    action_chunk_physical: Tensor,
    action_mask: Tensor,
    action_dimension_mask: Tensor,
    gripper_aperture_range: Tensor,
    *,
    config: ActionDescriptorConfig,
) -> ActionDescriptor:
    """Integrate physical deltas and resample a common trajectory descriptor.

    Wrist translations and rotations are the adjacent deltas already expressed
    in the chunk's fixed first-camera frame. Translations and gripper aperture
    are cumulatively summed. Rotations are composed on SO(3) through unit
    quaternions rather than by adding rotation vectors.
    """

    _validate_inputs(
        action_chunk_physical,
        action_mask,
        action_dimension_mask,
        gripper_aperture_range,
    )
    batch_size, _, _ = action_chunk_physical.shape
    output_values: list[Tensor] = []
    output_masks: list[Tensor] = []

    for side_index, side in enumerate(("left", "right")):
        slots = hand_action_slice(side)
        translation_valid = _group_is_valid(
            action_dimension_mask[..., slots.wrist_translation],
            action_mask,
        )
        rotation_valid = _group_is_valid(
            action_dimension_mask[..., slots.wrist_rotation],
            action_mask,
        )
        gripper_valid = _group_is_valid(
            action_dimension_mask[..., slots.gripper],
            action_mask,
        )

        translation_delta = action_chunk_physical[..., slots.wrist_translation]
        translation_states = _integrate_additive(translation_delta, action_mask)
        translation_trajectory = _resample_states(
            translation_states,
            action_mask,
            config.num_trajectory_points,
        )
        translation_mask = translation_valid[:, None, None].expand_as(translation_trajectory)
        output_values.append(translation_trajectory.masked_fill(~translation_mask, 0.0).flatten(1))
        output_masks.append(translation_mask.flatten(1))

        rotation_delta = action_chunk_physical[..., slots.wrist_rotation]
        rotation_states = _integrate_rotvec(rotation_delta, action_mask)
        rotation_trajectory = _resample_quaternions_as_rotvec(
            rotation_states,
            action_mask,
            config.num_trajectory_points,
        )
        rotation_mask = rotation_valid[:, None, None].expand_as(rotation_trajectory)
        output_values.append(rotation_trajectory.masked_fill(~rotation_mask, 0.0).flatten(1))
        output_masks.append(rotation_mask.flatten(1))

        # EgoDex aperture is measured in metres while AgiBot Alpha exposes a
        # source-specific actuator coordinate. Divide by an explicit per-side
        # full-aperture range before cross-source comparison so raw units can
        # never be compared silently.
        gripper_delta = action_chunk_physical[..., slots.gripper] / gripper_aperture_range[:, side_index, None, None]
        gripper_states = _integrate_additive(gripper_delta, action_mask)
        gripper_trajectory = _resample_states(
            gripper_states,
            action_mask,
            config.num_trajectory_points,
        )
        gripper_mask = gripper_valid[:, None, None].expand_as(gripper_trajectory)
        output_values.append(gripper_trajectory.masked_fill(~gripper_mask, 0.0).flatten(1))
        output_masks.append(gripper_mask.flatten(1))

    values = torch.cat(output_values, dim=1)
    dimension_mask = torch.cat(output_masks, dim=1)
    if values.shape != (batch_size, config.descriptor_dim):
        raise RuntimeError(
            f"unexpected action descriptor shape {tuple(values.shape)}; expected {(batch_size, config.descriptor_dim)}"
        )
    return ActionDescriptor(values=values, dimension_mask=dimension_mask)


def _validate_inputs(
    action_chunk: Tensor,
    action_mask: Tensor,
    action_dimension_mask: Tensor,
    gripper_aperture_range: Tensor,
) -> None:
    if action_chunk.ndim != 3 or action_chunk.shape[-1] != STAGE1_ACTION_DIM:
        raise ValueError(f"action_chunk_physical must have shape [B, L, {STAGE1_ACTION_DIM}]")
    if not action_chunk.is_floating_point() or not torch.isfinite(action_chunk).all():
        raise ValueError("action_chunk_physical must be finite and floating")
    if action_mask.shape != action_chunk.shape[:2] or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be boolean [B, L]")
    if action_dimension_mask.shape != action_chunk.shape or action_dimension_mask.dtype != torch.bool:
        raise ValueError("action_dimension_mask must be boolean [B, L, 44]")
    if not action_mask.any(dim=1).all():
        raise ValueError("every action chunk must contain a valid time step")
    if torch.any(action_mask[:, 1:] & ~action_mask[:, :-1]):
        raise ValueError("valid action time steps must form a contiguous prefix")
    if torch.any(action_dimension_mask & ~action_mask[..., None]):
        raise ValueError("padded action steps cannot contain valid physical dimensions")
    if torch.any(action_chunk.masked_select(~action_dimension_mask) != 0.0):
        raise ValueError("unmeasured and padded physical action values must be zero")
    if gripper_aperture_range.shape != (action_chunk.shape[0], 2):
        raise ValueError("gripper_aperture_range must have shape [B, 2]")
    if not gripper_aperture_range.is_floating_point() or not torch.isfinite(gripper_aperture_range).all():
        raise ValueError("gripper_aperture_range must be finite and floating")
    if torch.any(gripper_aperture_range <= 0.0):
        raise ValueError("gripper_aperture_range must be positive")


def _group_is_valid(group_mask: Tensor, action_mask: Tensor) -> Tensor:
    valid_or_padded = group_mask | ~action_mask[..., None]
    return valid_or_padded.all(dim=(1, 2)) & action_mask.any(dim=1)


def _integrate_additive(delta: Tensor, action_mask: Tensor) -> Tensor:
    masked_delta = delta.masked_fill(~action_mask[..., None], 0.0)
    cumulative = torch.cumsum(masked_delta, dim=1)
    initial = torch.zeros(
        delta.shape[0],
        1,
        delta.shape[2],
        dtype=delta.dtype,
        device=delta.device,
    )
    return torch.cat([initial, cumulative], dim=1)


def _integrate_rotvec(delta: Tensor, action_mask: Tensor) -> Tensor:
    delta_quaternion = _rotvec_to_quaternion(delta.masked_fill(~action_mask[..., None], 0.0))
    identity = delta.new_zeros(delta.shape[0], 4)
    identity[:, 0] = 1.0
    states = [identity]
    current = identity
    for index in range(delta.shape[1]):
        candidate = _quaternion_multiply(delta_quaternion[:, index], current)
        current = torch.where(action_mask[:, index, None], candidate, current)
        current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True).clamp_min(1e-12)
        states.append(current)
    return torch.stack(states, dim=1)


def _resample_states(states: Tensor, action_mask: Tensor, num_points: int) -> Tensor:
    lower, upper, alpha = _resample_indices(action_mask, num_points)
    batch = torch.arange(states.shape[0], device=states.device)[:, None]
    left = states[batch, lower]
    right = states[batch, upper]
    return torch.lerp(left, right, alpha[..., None].to(dtype=states.dtype))


def _resample_quaternions_as_rotvec(states: Tensor, action_mask: Tensor, num_points: int) -> Tensor:
    lower, upper, alpha = _resample_indices(action_mask, num_points)
    batch = torch.arange(states.shape[0], device=states.device)[:, None]
    left = states[batch, lower]
    right = states[batch, upper]
    right = torch.where((left * right).sum(dim=-1, keepdim=True) < 0.0, -right, right)
    interpolated = torch.lerp(left, right, alpha[..., None].to(dtype=states.dtype))
    interpolated = interpolated / torch.linalg.vector_norm(
        interpolated,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-12)
    return _quaternion_to_rotvec(interpolated)


def _resample_indices(action_mask: Tensor, num_points: int) -> tuple[Tensor, Tensor, Tensor]:
    lengths = action_mask.sum(dim=1)
    fractions = torch.linspace(
        1.0 / num_points,
        1.0,
        num_points,
        dtype=torch.float32,
        device=action_mask.device,
    )
    positions = lengths[:, None].to(torch.float32) * fractions[None]
    lower = torch.floor(positions).to(torch.long)
    upper = torch.ceil(positions).to(torch.long)
    alpha = positions - lower
    return lower, upper, alpha


def _rotvec_to_quaternion(rotvec: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1e-6,
        torch.sin(half) / angle.clamp_min(1e-12),
        0.5 - angle.square() / 48.0,
    )
    return torch.cat([torch.cos(half), rotvec * scale], dim=-1)


def _quaternion_to_rotvec(quaternion: Tensor) -> Tensor:
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sine, quaternion[..., :1].clamp_min(0.0))
    scale = torch.where(
        sine > 1e-6,
        angle / sine.clamp_min(1e-12),
        2.0 + sine.square() / 3.0,
    )
    return vector * scale


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dim=-1,
    )
