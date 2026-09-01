# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Read-only RoboCasa365 LeRobot-v2.1 adapter for Cosmos3 Action.

The target set is a directory of 50 independent LeRobot roots.  Current
LeRobot rejects v2.1 metadata, so this reader indexes the immutable JSONL and
per-episode parquet files directly while using LeRobot's torchcodec decoder.
"""

from __future__ import annotations

import bisect
import json
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.data.generator.action.action_processing import ActionNormalizer, resolve_action_normalization
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import split_episode_ids
from cosmos_framework.data.generator.action.domain_utils import get_domain_id

_CAMERAS = {
    "left": "observation.images.robot0_agentview_left",
    "right": "observation.images.robot0_agentview_right",
    "wrist": "observation.images.robot0_eye_in_hand",
}

_CAMERA_LAYOUTS = {
    "three_view_grid": ("left", "right", "wrist"),
    "left_wrist_horizontal": ("left", "wrist"),
}

_ACTION_INDICES = {
    "full12": tuple(range(12)),
    # RoboCasa365 layout: base xy-yaw-height (0:4), control mode (4),
    # end-effector delta pose + gripper (5:12).
    "arm7": tuple(range(5, 12)),
}

_STATE_INDICES = {
    "full16": tuple(range(16)),
    # RoboCasa365 state: base pose (0:7), relative end-effector pose (7:14),
    # gripper qpos (14:16). Fixed-base policies keep only the latter 9 values.
    "arm9": tuple(range(7, 16)),
}


def _humanize_task(task: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", task).strip()
    return (words[:1].upper() + words[1:].lower() + ".") if words else task


@dataclass(frozen=True)
class RoboCasa365Episode:
    root: Path
    category: str
    task_name: str
    episode_id: int
    length: int
    command: str
    parquet_path: Path
    video_paths: dict[str, Path]


class RoboCasa365LeRobotDataset(Dataset):
    """Cosmos policy windows over the 50-task RoboCasa365 target dataset."""

    EMBODIMENT_TYPE = "robocasa-panda-omron"

    def __init__(
        self,
        root: str,
        *,
        fps: float = 20.0,
        chunk_length: int = 32,
        split: str = "train",
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        mode: str = "wam",
        viewpoint: str = "concat_view",
        use_state: bool = False,
        use_image_augmentation: bool = False,
        emit_behavior_metadata: bool = False,
        task_set: str = "target50",
        task_names: Sequence[str] | None = None,
        camera_layout: str = "three_view_grid",
        action_mode: str = "full12",
        state_mode: str = "full16",
        action_normalization: str | None = "minmax",
        action_stats_path: str | None = None,
        parquet_cache_size: int = 8,
        max_roots: int | None = None,
    ) -> None:
        if viewpoint not in {"concat_view", "third_person_view"}:
            raise ValueError(f"Unsupported RoboCasa365 viewpoint={viewpoint!r}")
        if split not in {"train", "val", "full"}:
            raise ValueError("split must be train, val or full")
        if task_set not in {"target50", "atomic", "composite", "composite_seen", "composite_unseen"}:
            raise ValueError(f"Unsupported RoboCasa365 task_set={task_set!r}")
        if camera_layout not in _CAMERA_LAYOUTS:
            raise ValueError(f"Unsupported RoboCasa365 camera_layout={camera_layout!r}")
        if action_mode not in _ACTION_INDICES:
            raise ValueError(f"Unsupported RoboCasa365 action_mode={action_mode!r}")
        if state_mode not in _STATE_INDICES:
            raise ValueError(f"Unsupported RoboCasa365 state_mode={state_mode!r}")
        self.root = Path(root)
        self.fps = float(fps)
        self.chunk_length = int(chunk_length)
        self.split = split
        self.split_seed = int(split_seed)
        self.split_val_ratio = float(split_val_ratio)
        self.mode = mode
        self.viewpoint = viewpoint
        self.use_state = bool(use_state)
        self.use_image_augmentation = bool(use_image_augmentation)
        self.emit_behavior_metadata = bool(emit_behavior_metadata)
        self.task_set = task_set
        self.task_names = frozenset(task_names) if task_names is not None else None
        self.camera_layout = camera_layout
        self.action_mode = action_mode
        self.action_indices = _ACTION_INDICES[action_mode]
        self.state_mode = state_mode
        self.state_indices = _STATE_INDICES[state_mode]
        self.camera_names = _CAMERA_LAYOUTS[camera_layout]
        self.domain_id = get_domain_id(self.EMBODIMENT_TYPE)
        self.action_dim = len(self.action_indices)
        self._parquet_cache_size = int(parquet_cache_size)
        self._parquet_cache: OrderedDict[Path, dict[str, torch.Tensor]] = OrderedDict()
        self._normalizer: ActionNormalizer | None = None
        if action_normalization is not None:
            stats_path = (
                Path(action_stats_path)
                if action_stats_path
                else Path(__file__).parents[1] / "normalizer_stats" / "robocasa365_target_action_stats.json"
            )
            if not stats_path.is_file():
                raise FileNotFoundError(
                    "RoboCasa365 action normalization requires the target50 global action statistics; "
                    f"missing {stats_path}"
                )
            raw = json.loads(stats_path.read_text())
            raw = raw.get("action", raw)
            index = torch.tensor(self.action_indices, dtype=torch.long)
            stats = {
                key: torch.tensor(raw[key], dtype=torch.float32).index_select(0, index)
                for key in ("min", "max", "mean", "std", "q01", "q99")
            }
            self._normalizer = resolve_action_normalization(action_normalization, stats)

        roots = sorted(path.parent.parent for path in self.root.glob("*/*/*/lerobot/meta/info.json"))
        roots = [path for path in roots if self._include_root(path)]
        if max_roots is not None:
            roots = roots[:max_roots]
        if not roots:
            raise FileNotFoundError(f"No RoboCasa365 LeRobot-v2.1 roots under {self.root}")
        self.episodes: list[RoboCasa365Episode] = []
        for root_path in roots:
            self._index_root(root_path)
        # A length-H command needs H action rows and H+1 observation frames.
        self._window_lengths = [max(0, episode.length - self.chunk_length) for episode in self.episodes]
        self._cum_ends: list[int] = []
        total = 0
        for length in self._window_lengths:
            total += length
            self._cum_ends.append(total)
        self._num_windows = total

    def _include_root(self, path: Path) -> bool:
        category, task_name = path.relative_to(self.root).parts[:2]
        if self.task_names is not None and task_name not in self.task_names:
            return False
        if self.task_set == "atomic":
            return category == "atomic"
        if self.task_set == "composite":
            return category == "composite"
        if self.task_set == "composite_seen":
            return category == "composite" and task_name not in _COMPOSITE_UNSEEN
        if self.task_set == "composite_unseen":
            return category == "composite" and task_name in _COMPOSITE_UNSEEN
        return True

    def _index_root(self, path: Path) -> None:
        info = json.loads((path / "meta" / "info.json").read_text())
        if info["codebase_version"] != "v2.1" or info["fps"] != self.fps:
            raise ValueError(f"Unexpected RoboCasa365 metadata in {path}")
        category, task_name = path.relative_to(self.root).parts[:2]
        records = [json.loads(line) for line in (path / "meta" / "episodes.jsonl").read_text().splitlines()]
        selected = split_episode_ids(len(records), self.split_seed, self.split_val_ratio, self.split)
        selected.sort()
        for episode_id in selected:
            record = records[episode_id]
            length = int(record["length"])
            if length <= self.chunk_length:
                continue
            command = str(record.get("tasks", [""])[0]).strip() or _humanize_task(task_name)
            video_paths = {
                name: path / f"videos/chunk-{episode_id // 1000:03d}/{key}/episode_{episode_id:06d}.mp4"
                for name, key in _CAMERAS.items()
                if name in self.camera_names
            }
            parquet_path = path / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
            if not parquet_path.is_file() or any(not video.is_file() for video in video_paths.values()):
                raise FileNotFoundError(f"Incomplete RoboCasa365 episode {path}#{episode_id}")
            self.episodes.append(
                RoboCasa365Episode(path, category, task_name, episode_id, length, command, parquet_path, video_paths)
            )

    def __len__(self) -> int:
        return self._num_windows

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        start = 0
        for length in self._window_lengths:
            blocks.append((start, length))
            start += length
        return blocks

    def _resolve_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self._cum_ends, index)
        start = 0 if episode_index == 0 else self._cum_ends[episode_index - 1]
        return episode_index, index - start

    def _load_low_dim(self, episode: RoboCasa365Episode) -> dict[str, torch.Tensor]:
        cached = self._parquet_cache.pop(episode.parquet_path, None)
        if cached is None:
            table = pq.read_table(episode.parquet_path, columns=["action", "observation.state"])
            payload = table.to_pydict()
            cached = {
                "action": torch.tensor(payload["action"], dtype=torch.float32),
                "state": torch.tensor(payload["observation.state"], dtype=torch.float32),
            }
        self._parquet_cache[episode.parquet_path] = cached
        while len(self._parquet_cache) > self._parquet_cache_size:
            self._parquet_cache.popitem(last=False)
        return cached

    def _decode(self, path: Path, start: int) -> torch.Tensor:
        timestamps = [(start + offset) / self.fps for offset in range(self.chunk_length + 1)]
        return decode_video_frames(path, timestamps, tolerance_s=2e-4, backend="torchcodec")

    def _compose_video(self, episode: RoboCasa365Episode, start: int) -> torch.Tensor:
        left = self._decode(episode.video_paths["left"], start)
        if self.viewpoint == "third_person_view":
            return left
        wrist = self._decode(episode.video_paths["wrist"], start)
        if self.camera_layout == "left_wrist_horizontal":
            if self.use_image_augmentation:
                brightness = random.uniform(0.85, 1.15)
                contrast = random.uniform(0.85, 1.15)
                combined = torch.cat((left, wrist), dim=0)
                mean = combined.mean(dim=(-2, -1), keepdim=True)
                combined = ((combined - mean) * contrast + mean).mul(brightness).clamp(0, 1)
                left, wrist = combined.chunk(2, dim=0)
            # Preserve both native 256x256 views: agent-left | wrist.
            return torch.cat((left, wrist), dim=-1)
        right = self._decode(episode.video_paths["right"], start)
        if self.use_image_augmentation:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            combined = torch.cat((left, right, wrist), dim=0)
            mean = combined.mean(dim=(-2, -1), keepdim=True)
            combined = ((combined - mean) * contrast + mean).mul(brightness).clamp(0, 1)
            n = left.shape[0]
            left, right, wrist = combined[:n], combined[n : 2 * n], combined[2 * n :]
        right = F.interpolate(right, size=(128, 128), mode="bilinear", align_corners=False)
        wrist = F.interpolate(wrist, size=(128, 128), mode="bilinear", align_corners=False)
        # Left agent view on top, right agent + wrist views in the bottom row.
        return torch.cat((left, torch.cat((right, wrist), dim=-1)), dim=-2)

    def get_action_normalizer(self, _sample: dict[str, Any] | None = None) -> ActionNormalizer | None:
        return self._normalizer

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_offset = self._resolve_index(index)
        episode = self.episodes[episode_index]
        low_dim = self._load_low_dim(episode)
        action = low_dim["action"][frame_offset : frame_offset + self.chunk_length, self.action_indices]
        if self.use_state:
            # State and action have different physical layouts (16-D vs 12-D),
            # so expose state separately instead of pretending it is an action.
            initial_state = low_dim["state"][frame_offset, self.state_indices]
        else:
            initial_state = None
        video = self._compose_video(episode, frame_offset)
        result: dict[str, Any] = {
            "ai_caption": episode.command,
            "video": (video * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3),
            "action": action,
            "conditioning_fps": torch.tensor(int(self.fps), dtype=torch.long),
            "mode": self.mode,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "viewpoint": self.viewpoint,
            "additional_view_description": (
                "The left panel is the left agent view and the right panel is the wrist view."
                if self.camera_layout == "left_wrist_horizontal"
                else "The top panel is the left agent view. The bottom-left panel is the right agent view, "
                "and the bottom-right panel is the wrist view."
            ),
            "task_cluster": episode.task_name,
            "task_category": episode.category,
        }
        if initial_state is not None:
            result["proprio"] = initial_state
        if self.emit_behavior_metadata:
            # Keep ordinal source_index for diagnostics; resolve the canonical
            # Stage-1 key from task + episode_id in the behavior store.
            result["behavior_source_index"] = torch.tensor(episode_index, dtype=torch.long)
            result["behavior_episode_id"] = torch.tensor(episode.episode_id, dtype=torch.long)
            result["behavior_task_cluster"] = episode.task_name
            result["behavior_frame_offset"] = torch.tensor(frame_offset, dtype=torch.long)
        return result


_COMPOSITE_UNSEEN = {
    "ArrangeBreadBasket", "ArrangeTea", "BreadSelection", "CategorizeCondiments",
    "CuttingToolSelection", "GarnishPancake", "GatherTableware", "HeatKebabSandwich",
    "MakeIceLemonade", "PanTransfer", "PortionHotDogs", "RecycleBottlesByType",
    "SeparateFreezerRack", "WaffleReheat", "WashFruitColander", "WeighIngredients",
}
