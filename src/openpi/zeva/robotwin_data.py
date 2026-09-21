"""RoboTwin dataset helpers shared by ZeVA training stages."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch

from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS


class TorchCodecRoboTwinDataset:
    """Exact handoff adapter with cached TorchCodec video decoders."""

    def __init__(self, manifest: str | Path, subset: str):
        from egoscale.data.robotwin.lerobot import RoboTwinLeRobotEEF16Dataset  # noqa: PLC0415

        self.dataset = RoboTwinLeRobotEEF16Dataset(manifest, subset=subset)
        self.dataset.video_backend = "torchcodec"
        self.dataset._read_source_images = lambda _record, _frame: {}  # noqa: SLF001

    def read_images(
        self, record: dict[str, Any], frames: list[int]
    ) -> dict[str, torch.Tensor]:
        import typing  # noqa: PLC0415
        import typing_extensions  # noqa: PLC0415

        for name in ("Self", "Unpack", "NotRequired"):
            if not hasattr(typing, name):
                setattr(typing, name, getattr(typing_extensions, name))

        import lerobot  # noqa: PLC0415

        if "lerobot.datasets" not in sys.modules:
            package = ModuleType("lerobot.datasets")
            package.__path__ = [str(Path(next(iter(lerobot.__path__))) / "datasets")]
            package.__package__ = "lerobot.datasets"
            sys.modules["lerobot.datasets"] = package
        from lerobot.datasets.video_utils import decode_video_frames  # noqa: PLC0415

        episode_index = int(record["episode_index"])
        chunk = episode_index // 1000
        unique_frames = sorted(set(frames))
        timestamps = [frame / self.dataset.source_fps for frame in unique_frames]
        result = {}
        for camera in ROBOTWIN_CAMERA_KEYS:
            path = (
                record["source_root"]
                / "videos"
                / f"chunk-{chunk:03d}"
                / camera
                / f"episode_{episode_index:06d}.mp4"
            )
            decoded = decode_video_frames(
                path,
                timestamps,
                tolerance_s=self.dataset.video_tolerance_s,
                backend="torchcodec",
                return_uint8=True,
            )
            if decoded.ndim != 4 or len(decoded) != len(unique_frames):
                raise RuntimeError(f"TorchCodec returned an invalid frame batch for {path}.")
            by_frame = {frame: decoded[index] for index, frame in enumerate(unique_frames)}
            result[camera] = torch.stack([by_frame[frame] for frame in frames])
        return result
