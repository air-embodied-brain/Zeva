"""RoboTwin Joint14/EEF16 dataset backed by aligned episode caches."""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_NAMES
from openpi.zeva.robotwin_contract import ROBOTWIN_ADAPTER_SCHEMA
from openpi.zeva.robotwin_geometry import robotwin_absolute_to_relative_eef16

ROBOTWIN_LEROBOT_ADAPTER_SCHEMA = ROBOTWIN_ADAPTER_SCHEMA
ROBOTWIN_LEROBOT_ACTION_HORIZON = ROBOTWIN_ACTION_HORIZON

_CAMERA_KEYS = {
    "base_0_rgb": "observation.images.cam_high",
    "left_wrist_0_rgb": "observation.images.cam_left_wrist",
    "right_wrist_0_rgb": "observation.images.cam_right_wrist",
}
_JOINT_NAMES = [
    *(f"left_joint_{index}" for index in range(6)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(6)),
    "right_gripper",
]
_ACTION_NAMES = list(ROBOTWIN_ACTION_NAMES)


def align_relative_eef16_grippers_with_stage1(
    action: np.ndarray,
) -> np.ndarray:
    """Convert RoboTwin's 1=closed cache values to Stage-1's 1=open direction."""

    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape[-1] != 16 or not np.isfinite(result).all():
        raise ValueError("relative EEF action must be finite and end in dimension 16")
    grippers = result[..., 14:16]
    if np.any(grippers < -1e-4) or np.any(grippers > 1.0 + 1e-4):
        raise ValueError("RoboTwin EEF grippers must lie in [0,1]")
    result[..., 14:16] = 1.0 - np.clip(grippers, 0.0, 1.0)
    return result


class RoboTwinLeRobotMeta:
    """The small metadata surface consumed by the LeRobot trainer."""

    def __init__(
        self,
        *,
        stats: dict[str, dict[str, torch.Tensor]],
        episodes: dict[str, Any],
        image_shape: list[int],
    ) -> None:
        self.features = {
            "observation.state": {"dtype": "float32", "shape": [14], "names": _JOINT_NAMES},
            **{
                key: {
                    "dtype": "video",
                    "shape": image_shape,
                    "names": ["height", "width", "channel"],
                }
                for key in _CAMERA_KEYS.values()
            },
            "action": {"dtype": "float32", "shape": [16], "names": _ACTION_NAMES},
            "task": {"dtype": "string", "shape": [1], "names": None},
        }
        self.stats = stats
        self.episodes = episodes
        self.fps = 15

    @property
    def camera_keys(self) -> list[str]:
        return list(_CAMERA_KEYS.values())

    @property
    def depth_keys(self) -> list[str]:
        return []

    @property
    def has_language_columns(self) -> bool:
        return True


class RoboTwinLeRobotEEF16Dataset(Dataset[dict[str, Any]]):
    """Joint-state / relative-EEF-action dataset backed by existing local caches."""

    def __init__(self, manifest: str | Path, *, subset: str = "train") -> None:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        if payload.get("schema") != ROBOTWIN_LEROBOT_ADAPTER_SCHEMA:
            raise ValueError("unsupported RoboTwin LeRobot adapter manifest")
        if int(payload.get("action_horizon", -1)) != ROBOTWIN_LEROBOT_ACTION_HORIZON:
            raise ValueError("RoboTwin LeRobot action horizon must be exactly 50")
        if subset not in {"train", "validation", "all"}:
            raise ValueError("subset must be train, validation, or all")

        self.dataset_root = Path(payload["dataset_root"])
        self.eef_cache_root = Path(payload["eef_cache_root"])
        self.joint_cache_root = Path(payload["joint_cache_root"])
        self._opened: dict[int, dict[str, Any]] = {}
        self.video_backend = str(payload.get("video_backend", "torchcodec"))
        self.source_fps = int(payload.get("source_fps", 15))
        self.video_tolerance_s = float(payload.get("video_tolerance_s", 1e-4))
        self.return_uint8 = bool(payload.get("return_uint8", False))

        eef_partitions = json.loads((self.eef_cache_root / "partitions.json").read_text())
        joint_partitions = json.loads((self.joint_cache_root / "partitions.json").read_text())
        eef_by_key = {(row["split"], row["task"]): row for row in eef_partitions}
        joint_by_key = {(row["split"], row["task"]): row for row in joint_partitions}
        selected_splits = set(payload.get("splits", ["Clean"]))
        selected_tasks = payload.get("tasks")
        selected_tasks = None if selected_tasks is None else set(selected_tasks)
        keys = sorted(
            key
            for key in eef_by_key.keys() & joint_by_key.keys()
            if key[0] in selected_splits and (selected_tasks is None or key[1] in selected_tasks)
        )
        if not keys:
            raise ValueError("RoboTwin adapter selected no aligned EEF/joint partitions")

        records: list[dict[str, Any]] = []
        for partition_id, key in enumerate(keys):
            eef_root = self.eef_cache_root / eef_by_key[key]["cache_relpath"]
            joint_root = self.joint_cache_root / joint_by_key[key]["cache_relpath"]
            eef_indices = np.load(eef_root / "episode-indices.npy", mmap_mode="r")
            eef_lengths = np.load(eef_root / "source-lengths.npy", mmap_mode="r")
            joint_indices = np.load(joint_root / "episode-indices.npy", mmap_mode="r")
            joint_lengths = np.load(joint_root / "source-lengths.npy", mmap_mode="r")
            if not np.array_equal(eef_indices, joint_indices) or not np.array_equal(eef_lengths, joint_lengths):
                raise ValueError(f"EEF and Joint14 episode indices differ for {key}")
            for episode_position, length in enumerate(eef_lengths):
                if int(length) > 0:
                    records.append(
                        {
                            "partition_id": partition_id,
                            "key": key,
                            "eef_root": eef_root,
                            "joint_root": joint_root,
                            "episode_position": episode_position,
                            "episode_index": int(eef_indices[episode_position]),
                            "length": int(length),
                            "source_root": self.dataset_root / eef_by_key[key]["source_relpath"],
                        }
                    )

        partition_ids = np.asarray([row["partition_id"] for row in records], dtype=np.int32)
        episode_positions = np.asarray([row["episode_position"] for row in records], dtype=np.int32)
        fraction = float(payload.get("validation_fraction", 0.0))
        if not 0.0 <= fraction < 1.0:
            raise ValueError("validation_fraction must lie in [0,1)")
        if fraction > 0.0 and subset != "all":
            validation = robotwin_validation_episode_mask(
                partition_ids,
                episode_positions,
                fraction=fraction,
                seed=int(payload.get("split_seed", 42)),
            )
            records = [
                row
                for row, held_out in zip(records, validation, strict=True)
                if bool(held_out) == (subset == "validation")
            ]
        elif subset == "validation":
            records = []
        if not records:
            raise ValueError(f"RoboTwin {subset} subset is empty")

        self._records = records
        # Native LeRobot trains from every recorded frame.  Future actions past
        # an episode boundary repeat the terminal action; mirror that sampling
        # rather than dropping the last 50 anchors of every episode.
        counts = np.asarray([row["length"] for row in records], dtype=np.int64)
        self._cumulative = np.concatenate(([0], np.cumsum(counts)))
        self.episodes = list(range(len(records)))
        self.absolute_to_relative_idx = None
        stats = _load_stats(Path(payload["stats_path"]))
        tasks = [[row["key"][1]] for row in records]
        self.meta = RoboTwinLeRobotMeta(
            stats=stats,
            episodes={
                "dataset_from_index": self._cumulative[:-1].tolist(),
                "dataset_to_index": self._cumulative[1:].tolist(),
                "tasks": tasks,
            },
            image_shape=list(payload.get("image_shape", [480, 640, 3])),
        )

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    @property
    def num_frames(self) -> int:
        return len(self)

    @property
    def num_episodes(self) -> int:
        return len(self._records)

    def __getitem__(self, item: int) -> dict[str, Any]:
        if item < 0:
            item += len(self)
        if not 0 <= item < len(self):
            raise IndexError(item)
        episode = bisect.bisect_right(self._cumulative, item) - 1
        frame = item - int(self._cumulative[episode])
        record = self._records[episode]
        opened = self._open_episode_group(episode)
        source_position = int(record["episode_position"])
        eef_start = int(opened["eef_offsets"][source_position])
        native_indices = np.minimum(
            frame + np.arange(1, ROBOTWIN_LEROBOT_ACTION_HORIZON + 1),
            int(record["length"]) - 1,
        )
        current_eef = np.asarray(opened["eef"][eef_start + 2 * frame], dtype=np.float32)
        future_eef = np.asarray(opened["eef"][eef_start + 2 * native_indices], dtype=np.float32)
        action = align_relative_eef16_grippers_with_stage1(robotwin_absolute_to_relative_eef16(current_eef, future_eef))
        joint_start = int(opened["joint_offsets"][source_position])
        state = np.asarray(opened["joint"][joint_start + frame], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError("native RoboTwin state is not Joint14")
        sample = {
            "observation.state": torch.from_numpy(state.copy()),
            "task": opened["prompts"][source_position],
            **self._read_source_images(record, frame),
        }
        sample.update(
            {
                "action": torch.from_numpy(action),
                "index": int(item),
                "episode_index": int(episode),
                "frame_index": int(frame),
            }
        )
        return sample

    def _open_episode_group(self, episode: int) -> dict[str, Any]:
        partition_id = int(self._records[episode]["partition_id"])
        if partition_id in self._opened:
            return self._opened[partition_id]
        record = next(row for row in self._records if int(row["partition_id"]) == partition_id)
        opened = {
            "eef": np.load(record["eef_root"] / "absolute-eef-30hz.npy", mmap_mode="r"),
            "eef_offsets": np.load(record["eef_root"] / "episode-offsets.npy", mmap_mode="r"),
            "joint": np.load(record["joint_root"] / "absolute-joint15.npy", mmap_mode="r"),
            "joint_offsets": np.load(record["joint_root"] / "episode-offsets.npy", mmap_mode="r"),
            "prompts": json.loads((record["eef_root"] / "prompts.json").read_text()),
        }
        self._opened[partition_id] = opened
        return opened

    def _read_source_images(self, record: dict[str, Any], frame: int) -> dict[str, Any]:
        """Decode one v2.1 RoboTwin frame without converting the dataset to v3."""

        from lerobot.datasets.video_utils import decode_video_frames  # noqa: PLC0415

        episode_index = int(record["episode_index"])
        timestamp = frame / self.source_fps
        chunk = episode_index // 1000
        result: dict[str, torch.Tensor] = {}
        # The released v2.1 RoboTwin video directories use the LeRobot
        # observation feature names, not the simulator stream aliases.
        for output_key in _CAMERA_KEYS.values():
            path = (
                record["source_root"]
                / "videos"
                / f"chunk-{chunk:03d}"
                / output_key
                / f"episode_{episode_index:06d}.mp4"
            )
            frames = decode_video_frames(
                path,
                [timestamp],
                tolerance_s=self.video_tolerance_s,
                backend=self.video_backend,
                return_uint8=self.return_uint8,
            )
            if frames.ndim != 4 or len(frames) != 1:
                raise ValueError(f"RoboTwin video returned an invalid frame: {path}")
            result[output_key] = frames[0]
        return result


def _load_stats(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, torch.Tensor]] = {}
    for key in ("observation.state", "action"):
        values = payload.get(key)
        if not isinstance(values, dict):
            raise TypeError(f"normalization artifact lacks mapping {key}")
        result[key] = {
            name: torch.tensor(values[name], dtype=torch.float32)
            for name in ("mean", "std", "min", "max", "q01", "q99")
            if name in values
        }
        if not {"mean", "std"} <= set(result[key]):
            raise ValueError(f"normalization artifact lacks mean/std for {key}")
        dimension = 14 if key == "observation.state" else 16
        if any(value.shape != (dimension,) or not torch.isfinite(value).all() for value in result[key].values()):
            raise ValueError(f"normalization artifact has invalid statistics for {key}")
        if torch.any(result[key]["std"] <= 0):
            raise ValueError(f"normalization artifact has non-positive std for {key}")
    return result


def robotwin_validation_episode_mask(
    partition_ids: np.ndarray,
    episode_positions: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Select a deterministic, partition-stratified episode holdout."""

    partitions = np.asarray(partition_ids, dtype=np.int64)
    episodes = np.asarray(episode_positions, dtype=np.int64)
    if partitions.shape != episodes.shape or partitions.ndim != 1:
        raise ValueError("RoboTwin episode identifiers must be parallel 1D arrays")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    result = np.zeros(len(partitions), dtype=np.bool_)
    for partition in np.unique(partitions):
        candidates = np.flatnonzero(partitions == partition)
        if len(candidates) < 2:
            continue
        count = min(len(candidates) - 1, max(1, round(len(candidates) * fraction)))
        scores = []
        for index in candidates:
            value = f"{seed}:{int(partition)}:{int(episodes[index])}".encode()
            scores.append(int.from_bytes(hashlib.sha256(value).digest()[:8], "little"))
        selected = candidates[np.argsort(np.asarray(scores, dtype=np.uint64))[:count]]
        result[selected] = True
    return result
