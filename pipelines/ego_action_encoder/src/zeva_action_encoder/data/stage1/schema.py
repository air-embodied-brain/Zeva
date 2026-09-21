"""Universal Stage 1 physical-action slots shared across embodiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


STAGE1_ACTION_SCHEMA = "dual-hand-camera-t0-wrist-gripper-fingertips-44d-v1"
STAGE1_ACTION_REPRESENTATION = "wrist_translation_rotvec_gripper_fingertip_translation_delta"
STAGE1_ACTION_DIM = 44


@dataclass(frozen=True)
class Stage1HandActionSlice:
    """Contiguous slots belonging to one hand or robot arm."""

    all: slice
    wrist_translation: slice
    wrist_rotation: slice
    gripper: slice
    fingertips: slice


def hand_action_slice(side: str) -> Stage1HandActionSlice:
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    start = 0 if side == "left" else 22
    return Stage1HandActionSlice(
        all=slice(start, start + 22),
        wrist_translation=slice(start, start + 3),
        wrist_rotation=slice(start + 3, start + 6),
        gripper=slice(start + 6, start + 7),
        fingertips=slice(start + 7, start + 22),
    )


def _dimension_names() -> tuple[str, ...]:
    axes = ("x", "y", "z")
    names: list[str] = []
    for side in ("left", "right"):
        names.extend(f"{side}_wrist_translation_delta_{axis}" for axis in axes)
        names.extend(f"{side}_wrist_rotvec_delta_{axis}" for axis in axes)
        names.append(f"{side}_gripper_aperture_delta")
        for finger in ("thumb", "index", "middle", "ring", "little"):
            names.extend(f"{side}_{finger}_tip_translation_delta_{axis}" for axis in axes)
    return tuple(names)


def _dimension_mask(*, fingertips: bool) -> NDArray[np.bool_]:
    mask = np.zeros(STAGE1_ACTION_DIM, dtype=np.bool_)
    for side in ("left", "right"):
        slots = hand_action_slice(side)
        mask[slots.wrist_translation] = True
        mask[slots.wrist_rotation] = True
        mask[slots.gripper] = True
        mask[slots.fingertips] = fingertips
    mask.flags.writeable = False
    return mask


STAGE1_ACTION_DIMENSION_NAMES = _dimension_names()
EGODEX_ACTION_DIMENSION_MASK = _dimension_mask(fingertips=True)
AGIBOT_ACTION_DIMENSION_MASK = _dimension_mask(fingertips=False)

if len(STAGE1_ACTION_DIMENSION_NAMES) != STAGE1_ACTION_DIM:
    raise RuntimeError("Stage 1 action names must match the universal action dimension")
if int(EGODEX_ACTION_DIMENSION_MASK.sum()) != 44:
    raise RuntimeError("EgoDex must populate all universal Stage 1 action dimensions")
if int(AGIBOT_ACTION_DIMENSION_MASK.sum()) != 14:
    raise RuntimeError("AgiBot gripper episodes must populate exactly fourteen action dimensions")
