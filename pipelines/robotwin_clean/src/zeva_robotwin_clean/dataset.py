"""Validation wrapper for already-prepared RoboTwin clean datasets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset

from zeva_robotwin_clean.contract import ACTION_DIMENSION, ACTION_HORIZON, CAMERA_KEYS, STATE_DIMENSION


class PreparedRoboTwinDatasetAdapter(Dataset[dict[str, Any]]):
    """Expose a prepared dataset through the exact LeRobot sample contract.

    The wrapped dataset owns all storage and decoding. This adapter performs no
    download, trajectory conversion, sampling, or augmentation.
    """

    def __init__(self, dataset: Dataset[Mapping[str, Any]], *, validate_every_sample: bool = True) -> None:
        self.dataset = dataset
        self.validate_every_sample = validate_every_sample
        if hasattr(dataset, "meta"):
            self.meta = dataset.meta
        if hasattr(dataset, "absolute_to_relative_idx"):
            self.absolute_to_relative_idx = dataset.absolute_to_relative_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        if self.validate_every_sample:
            validate_prepared_sample(sample)
        return sample


def validate_prepared_sample(sample: Mapping[str, Any]) -> None:
    required = {"observation.state", "action", "task", *CAMERA_KEYS}
    missing = sorted(required - sample.keys())
    if missing:
        raise KeyError(f"RoboTwin sample is missing keys: {missing}")
    state = sample["observation.state"]
    action = sample["action"]
    if not isinstance(state, torch.Tensor) or state.shape != (STATE_DIMENSION,):
        raise ValueError("observation.state must be a tensor shaped [14]")
    if not isinstance(action, torch.Tensor) or action.shape != (ACTION_HORIZON, ACTION_DIMENSION):
        raise ValueError("action must be a tensor shaped [50,16]")
    if not state.is_floating_point() or not action.is_floating_point():
        raise TypeError("state and action must be floating point")
    if not torch.isfinite(state).all() or not torch.isfinite(action).all():
        raise ValueError("state and action must be finite")
    if not isinstance(sample["task"], str) or not sample["task"].strip():
        raise ValueError("task must be a non-empty string")
    for key in CAMERA_KEYS:
        image = sample[key]
        if not isinstance(image, torch.Tensor) or image.ndim != 3 or 3 not in (image.shape[0], image.shape[-1]):
            raise ValueError(f"{key} must be a three-channel image tensor")
