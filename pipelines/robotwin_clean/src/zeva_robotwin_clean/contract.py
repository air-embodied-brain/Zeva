"""RoboTwin Joint14 state and chunk-start-relative EEF16 action contract."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


ACTION_HORIZON = 50
STATE_DIMENSION = 14
ACTION_DIMENSION = 16
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass(frozen=True)
class EEF16Slots:
    left_position: slice = field(default_factory=lambda: slice(0, 3))
    left_quaternion: slice = field(default_factory=lambda: slice(3, 7))
    right_position: slice = field(default_factory=lambda: slice(7, 10))
    right_quaternion: slice = field(default_factory=lambda: slice(10, 14))
    grippers: slice = field(default_factory=lambda: slice(14, 16))


EEF16 = EEF16Slots()


def absolute_to_chunk_start_eef16(
    state: NDArray[np.floating],
    future: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Convert arm-major absolute EEF poses to chunk-start-relative EEF16.

    ``state`` uses ``left xyz/xyzw/gripper, right xyz/xyzw/gripper``.
    Translation and spatial rotation are relative to the current state and stay
    expressed in the fixed main-camera axes. Gripper targets remain absolute.
    """

    current = np.asarray(state, dtype=np.float64)
    targets = np.asarray(future, dtype=np.float64)
    if (
        current.shape[-1:] != (ACTION_DIMENSION,)
        or targets.ndim < 2
        or targets.shape[-1] != ACTION_DIMENSION
        or targets.shape[:-2] != current.shape[:-1]
        or targets.shape[-2] == 0
    ):
        raise ValueError("state and future must have shapes [...,16] and [...,T,16]")
    if not np.isfinite(current).all() or not np.isfinite(targets).all():
        raise ValueError("EEF values must be finite")

    result = np.empty(targets.shape, dtype=np.float32)
    source_slots = (
        (slice(0, 3), slice(3, 7), 7),
        (slice(8, 11), slice(11, 15), 15),
    )
    output_slots = (
        (EEF16.left_position, EEF16.left_quaternion, 14),
        (EEF16.right_position, EEF16.right_quaternion, 15),
    )
    for (source_position, source_quaternion, source_gripper), (
        output_position,
        output_quaternion,
        output_gripper,
    ) in zip(source_slots, output_slots, strict=True):
        result[..., output_position] = (
            targets[..., source_position] - current[..., source_position][..., None, :]
        )
        start_rotation = quaternion_xyzw_to_rotation_matrix(
            current[..., source_quaternion]
        ).astype(np.float64)
        future_rotation = quaternion_xyzw_to_rotation_matrix(
            targets[..., source_quaternion]
        ).astype(np.float64)
        spatial_delta = future_rotation @ np.swapaxes(start_rotation, -1, -2)[..., None, :, :]
        result[..., output_quaternion] = rotation_matrix_to_quaternion_xyzw(spatial_delta)
        result[..., output_gripper] = targets[..., source_gripper]
    return result


def align_grippers_to_model_convention(action: NDArray[np.floating]) -> NDArray[np.float32]:
    """Convert cache-side ``1=closed`` to model-side ``1=open``."""

    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape[-1] != ACTION_DIMENSION or not np.isfinite(result).all():
        raise ValueError("action must be finite and end in dimension 16")
    grippers = result[..., EEF16.grippers]
    if np.any(grippers < -1e-4) or np.any(grippers > 1.0 + 1e-4):
        raise ValueError("gripper targets must lie in [0,1]")
    result[..., EEF16.grippers] = 1.0 - np.clip(grippers, 0.0, 1.0)
    return result


def canonicalize_quaternion_xyzw(
    quaternion: NDArray[np.floating],
) -> NDArray[np.float32]:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4 or not np.isfinite(value).all():
        raise ValueError("quaternion must be finite and end in dimension four")
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
            row *= -1.0
    return np.asarray(result, dtype=np.float32)


def quaternion_xyzw_to_rotation_matrix(
    quaternion: NDArray[np.floating],
) -> NDArray[np.float32]:
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


def rotation_matrix_to_quaternion_xyzw(
    rotation: NDArray[np.floating],
) -> NDArray[np.float32]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrices must be finite and end in [3,3]")
    r00 = matrix[..., 0, 0]
    r11 = matrix[..., 1, 1]
    r22 = matrix[..., 2, 2]
    quaternion = np.stack(
        (
            np.copysign(np.sqrt(np.maximum(0.0, 1.0 + r00 - r11 - r22)) * 0.5, matrix[..., 2, 1] - matrix[..., 1, 2]),
            np.copysign(np.sqrt(np.maximum(0.0, 1.0 - r00 + r11 - r22)) * 0.5, matrix[..., 0, 2] - matrix[..., 2, 0]),
            np.copysign(np.sqrt(np.maximum(0.0, 1.0 - r00 - r11 + r22)) * 0.5, matrix[..., 1, 0] - matrix[..., 0, 1]),
            np.sqrt(np.maximum(0.0, 1.0 + r00 + r11 + r22)) * 0.5,
        ),
        axis=-1,
    )
    return canonicalize_quaternion_xyzw(quaternion)
