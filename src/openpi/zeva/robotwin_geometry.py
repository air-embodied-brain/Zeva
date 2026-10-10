"""Chunk-start-relative EEF16 transforms for RoboTwin cached poses."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def canonicalize_quaternion_xyzw(
    quaternion: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Normalize xyzw quaternions and choose one deterministic hemisphere."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4 or not np.isfinite(value).all():
        raise ValueError("quaternion must be finite and end in dimension 4")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("quaternion norm must be positive")
    result = value / norm
    flat = result.reshape(-1, 4)
    for row in flat:
        flip = row[3] < 0.0
        if abs(row[3]) <= 1e-12:
            nonzero = np.flatnonzero(np.abs(row[:3]) > 1e-12)
            flip = bool(len(nonzero) and row[int(nonzero[0])] < 0.0)
        if flip:
            row[:] *= -1.0
    return np.asarray(result, dtype=np.float32)


def rotation_matrix_to_quaternion_xyzw(
    rotation: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Convert proper rotation matrices to canonical unit xyzw quaternions."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrices must be finite and end in [3,3]")
    r00 = matrix[..., 0, 0]
    r11 = matrix[..., 1, 1]
    r22 = matrix[..., 2, 2]
    quaternion = np.stack(
        (
            np.copysign(
                np.sqrt(np.maximum(0.0, 1.0 + r00 - r11 - r22)) * 0.5,
                matrix[..., 2, 1] - matrix[..., 1, 2],
            ),
            np.copysign(
                np.sqrt(np.maximum(0.0, 1.0 - r00 + r11 - r22)) * 0.5,
                matrix[..., 0, 2] - matrix[..., 2, 0],
            ),
            np.copysign(
                np.sqrt(np.maximum(0.0, 1.0 - r00 - r11 + r22)) * 0.5,
                matrix[..., 1, 0] - matrix[..., 0, 1],
            ),
            np.sqrt(np.maximum(0.0, 1.0 + r00 + r11 + r22)) * 0.5,
        ),
        axis=-1,
    )
    return canonicalize_quaternion_xyzw(quaternion)


def quaternion_xyzw_to_rotation_matrix(
    quaternion: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Convert arbitrary nonzero xyzw quaternions to rotation matrices."""

    value = canonicalize_quaternion_xyzw(quaternion).astype(np.float64)
    x, y, z, w = np.moveaxis(value, -1, 0)
    result = np.empty((*value.shape[:-1], 3, 3), dtype=np.float32)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def robotwin_absolute_to_relative_eef16(
    state: NDArray[np.floating],
    future: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Convert the legacy absolute cache to the shared relative EEF16 action.

    The cache is arm-major (each gripper follows its arm).  The model contract
    keeps both grippers in the final two slots.  Translation and rotation are
    relative to the chunk-start EEF pose and remain expressed on the fixed main
    camera axes; gripper commands remain absolute.
    """

    current = np.asarray(state, dtype=np.float64)
    targets = np.asarray(future, dtype=np.float64)
    if (
        current.shape[-1:] != (16,)
        or targets.ndim < 2
        or targets.shape[-1] != 16
        or targets.shape[:-2] != current.shape[:-1]
    ):
        raise ValueError("RoboTwin absolute EEF values must have shape [...,16] and [...,T,16]")
    if len(targets) == 0 or not np.isfinite(current).all() or not np.isfinite(targets).all():
        raise ValueError("RoboTwin absolute EEF values must be nonempty and finite")

    result = np.empty(targets.shape, dtype=np.float32)
    legacy = ((slice(0, 3), slice(3, 7), 7), (slice(8, 11), slice(11, 15), 15))
    canonical = (
        (slice(0, 3), slice(3, 7)),
        (slice(7, 10), slice(10, 14)),
    )
    for (source_position, source_quaternion, source_gripper), (
        target_position,
        target_quaternion,
    ) in zip(legacy, canonical, strict=True):
        result[..., target_position] = targets[..., source_position] - current[..., source_position][..., None, :]
        start_rotation = quaternion_xyzw_to_rotation_matrix(current[..., source_quaternion]).astype(np.float64)
        future_rotation = quaternion_xyzw_to_rotation_matrix(targets[..., source_quaternion]).astype(np.float64)
        spatial_delta = future_rotation @ np.swapaxes(start_rotation, -1, -2)[..., None, :, :]
        result[..., target_quaternion] = rotation_matrix_to_quaternion_xyzw(spatial_delta)
        result[..., 14 + (0 if source_gripper == 7 else 1)] = targets[..., source_gripper]
    return result
