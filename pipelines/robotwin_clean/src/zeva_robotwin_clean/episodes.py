"""Lazy episode-to-window adapter for RoboTwin clean post-training."""

from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset

from zeva_robotwin_clean.contract import ACTION_HORIZON
from zeva_robotwin_clean.contract import CAMERA_KEYS
from zeva_robotwin_clean.contract import absolute_to_chunk_start_eef16
from zeva_robotwin_clean.contract import align_grippers_to_model_convention


@dataclass(frozen=True)
class EpisodeDescriptor:
    """Metadata needed to index an episode without loading its arrays."""

    reference: str
    split: str
    task: str
    instruction: str
    length: int

    def __post_init__(self) -> None:
        if self.split != "Clean":
            raise ValueError("the public post-training adapter accepts only RoboTwin Clean episodes")
        if not self.reference or not self.task or not self.instruction.strip() or self.length < 1:
            raise ValueError("episode descriptor fields must be non-empty and length must be positive")


@dataclass(frozen=True)
class EpisodeArrays:
    """One decoded episode in the public array contract."""

    joint14: NDArray[np.floating]
    absolute_eef16: NDArray[np.floating]
    images: Mapping[str, NDArray[np.generic]]


EpisodeLoader = Callable[[EpisodeDescriptor], EpisodeArrays]


class RoboTwinCleanWindowDataset(Dataset[dict[str, Any]]):
    """Expose every Clean episode frame using natural window frequency.

    The caller owns storage and supplies a lazy ``episode_loader``. No task is
    balanced or oversampled: an episode of length ``T`` contributes ``T``
    windows. Future targets are clipped at the episode boundary, matching the
    released 50-step action-chunk contract.
    """

    def __init__(
        self,
        descriptors: Sequence[EpisodeDescriptor],
        episode_loader: EpisodeLoader,
        *,
        meta: object | None = None,
    ) -> None:
        if not descriptors:
            raise ValueError("at least one Clean episode is required")
        self.descriptors = tuple(descriptors)
        self.episode_loader = episode_loader
        self._ends: list[int] = []
        total = 0
        for descriptor in self.descriptors:
            total += descriptor.length
            self._ends.append(total)
        self.episodes = list(range(len(self.descriptors)))
        self.absolute_to_relative_idx = None
        if meta is not None:
            self.meta = meta

    def __len__(self) -> int:
        return self._ends[-1]

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len(self.descriptors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._ends, index)
        start = 0 if episode_index == 0 else self._ends[episode_index - 1]
        frame_index = index - start
        descriptor = self.descriptors[episode_index]
        arrays = self.episode_loader(descriptor)
        _validate_episode_arrays(descriptor, arrays)
        future_indices = np.minimum(frame_index + np.arange(1, ACTION_HORIZON + 1), descriptor.length - 1)
        anchor = arrays.absolute_eef16[frame_index]
        future = arrays.absolute_eef16[future_indices]
        action = align_grippers_to_model_convention(absolute_to_chunk_start_eef16(anchor, future))
        sample: dict[str, Any] = {
            "observation.state": torch.as_tensor(np.array(arrays.joint14[frame_index], copy=True), dtype=torch.float32),
            "action": torch.from_numpy(action),
            "task": descriptor.instruction,
            "index": int(index),
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
        }
        for key in CAMERA_KEYS:
            sample[key] = _image_tensor(arrays.images[key][frame_index])
        return sample


def _validate_episode_arrays(descriptor: EpisodeDescriptor, arrays: EpisodeArrays) -> None:
    if arrays.joint14.shape != (descriptor.length, 14):
        raise ValueError("joint14 must have shape [T,14]")
    if arrays.absolute_eef16.shape != (descriptor.length, 16):
        raise ValueError("absolute_eef16 must have shape [T,16]")
    if not np.isfinite(arrays.joint14).all() or not np.isfinite(arrays.absolute_eef16).all():
        raise ValueError("episode state/action arrays must be finite")
    missing = sorted(set(CAMERA_KEYS) - arrays.images.keys())
    if missing:
        raise KeyError(f"episode is missing camera arrays: {missing}")
    for key in CAMERA_KEYS:
        if len(arrays.images[key]) != descriptor.length:
            raise ValueError(f"{key} length differs from the episode")


def _image_tensor(image: NDArray[np.generic]) -> torch.Tensor:
    value = np.asarray(image)
    if value.ndim != 3:
        raise ValueError("camera frames must be HWC or CHW")
    if value.shape[-1] == 3:
        value = value.transpose(2, 0, 1)
    elif value.shape[0] != 3:
        raise ValueError("camera frames must contain three RGB channels")
    tensor = torch.as_tensor(np.ascontiguousarray(value))
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255.0)
    tensor = tensor.float()
    if not torch.isfinite(tensor).all():
        raise ValueError("camera frames must be finite")
    return tensor
