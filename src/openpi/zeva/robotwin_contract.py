"""Frozen RoboTwin interface used by the released LeRobot PI0.5 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

ROBOTWIN_HANDOFF_SCHEMA = "egoscale-robotwin-memory-baseline-handoff-v1"
ROBOTWIN_STATS_SCHEMA = "egoscale-robotwin-lerobot-mean-std-v1"
ROBOTWIN_ADAPTER_SCHEMA = "egoscale-robotwin-lerobot-relative-eef16-v1"
ROBOTWIN_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ROBOTWIN_ACTION_NAMES = (
    "left_dx",
    "left_dy",
    "left_dz",
    "left_dqx",
    "left_dqy",
    "left_dqz",
    "left_dqw",
    "right_dx",
    "right_dy",
    "right_dz",
    "right_dqx",
    "right_dqy",
    "right_dqz",
    "right_dqw",
    "left_gripper",
    "right_gripper",
)
ROBOTWIN_STATE_DIM = 14
ROBOTWIN_ACTION_DIM = 16
ROBOTWIN_ACTION_HORIZON = 50
ROBOTWIN_IMAGE_SHAPE = (480, 640, 3)
ROBOTWIN_MODEL_GRIPPER_OPEN = 1.0


def prepare_robotwin_pi_image(image: torch.Tensor | Any, *, name: str) -> torch.Tensor:
    """Convert one RoboTwin view to PI0.5's exact CHW/BCHW float32 [0, 1] domain.

    The released checkpoint declares visual normalization as IDENTITY.  Its
    model later maps ``x`` to ``2 * x - 1`` but does not divide byte images by
    255, so that conversion must happen before the saved preprocessor.  Float
    images are required to already be in [0, 1]; accepting float [0, 255]
    would hide a caller contract error.
    """
    value = torch.as_tensor(image)
    height, width, channels = ROBOTWIN_IMAGE_SHAPE
    if value.ndim == 3:
        if tuple(value.shape) == (height, width, channels):
            value = value.permute(2, 0, 1)
        elif tuple(value.shape) != (channels, height, width):
            raise ValueError(
                f"{name} must be HWC {ROBOTWIN_IMAGE_SHAPE} or CHW "
                f"{(channels, height, width)}, got {tuple(value.shape)}."
            )
    elif value.ndim == 4:
        if tuple(value.shape[1:]) == (height, width, channels):
            value = value.permute(0, 3, 1, 2)
        elif tuple(value.shape[1:]) != (channels, height, width):
            raise ValueError(
                f"{name} must be BHWC (*, {height}, {width}, {channels}) or "
                f"BCHW (*, {channels}, {height}, {width}), got {tuple(value.shape)}."
            )
    else:
        raise ValueError(f"{name} must have 3 or 4 dimensions, got {tuple(value.shape)}.")

    if value.dtype == torch.uint8:
        value = value.to(torch.float32).div_(255.0)
    elif value.is_floating_point():
        value = value.to(torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite pixels.")
        minimum, maximum = float(value.min()), float(value.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"{name} float pixels must already be in [0,1], got [{minimum:.6g},{maximum:.6g}]."
            )
    else:
        raise TypeError(f"{name} must be uint8 or floating point, got {value.dtype}.")
    return value.contiguous()


@dataclass(frozen=True)
class MeanStdActionNormalizer:
    """Exact model-domain transform used by the selected RoboTwin checkpoint."""

    mean: torch.Tensor
    std: torch.Tensor
    source: str

    @classmethod
    def from_stats_file(cls, path: str | Path) -> MeanStdActionNormalizer:
        path = Path(path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != ROBOTWIN_STATS_SCHEMA:
            raise ValueError(f"Unexpected RoboTwin statistics schema in {path}.")
        entry = payload.get("action", {})
        mean = torch.as_tensor(entry.get("mean"), dtype=torch.float32)
        std = torch.as_tensor(entry.get("std"), dtype=torch.float32)
        if mean.shape != (ROBOTWIN_ACTION_DIM,) or std.shape != (ROBOTWIN_ACTION_DIM,):
            raise ValueError("RoboTwin action statistics must describe EEF16.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("RoboTwin action mean/std must be finite with positive standard deviations.")
        return cls(mean=mean, std=std, source=str(path))

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=actions.device, dtype=actions.dtype)
        std = self.std.to(device=actions.device, dtype=actions.dtype)
        return (actions[..., :ROBOTWIN_ACTION_DIM] - mean) / (std + 1e-8)

    def unnormalize(self, actions: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=actions.device, dtype=actions.dtype)
        std = self.std.to(device=actions.device, dtype=actions.dtype)
        return actions[..., :ROBOTWIN_ACTION_DIM] * std + mean

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "mean_std",
            "mean": self.mean.cpu().clone(),
            "std": self.std.cpu().clone(),
            "source": self.source,
        }


@dataclass(frozen=True)
class RobotWinHandoff:
    root: Path
    checkpoint: Path
    contract: Path
    statistics: Path

    @classmethod
    def from_root(cls, root: str | Path) -> RobotWinHandoff:
        root = Path(root).resolve()
        handoff = cls(
            root=root,
            checkpoint=root / "checkpoint" / "pretrained_model",
            contract=root / "runtime" / "baseline" / "contract.json",
            statistics=root / "reference" / "mean-std-eef16-h50-stage1grip-train95-v2.json",
        )
        handoff.validate()
        return handoff

    def validate(self) -> None:
        for path in (self.contract, self.statistics):
            if not path.is_file():
                raise FileNotFoundError(path)
        contract = json.loads(self.contract.read_text(encoding="utf-8"))
        if contract.get("schema") != ROBOTWIN_HANDOFF_SCHEMA:
            raise ValueError("Unsupported RoboTwin handoff schema.")
        if contract.get("memory_code_included") is not False:
            raise ValueError("The foundation handoff must not contain an earlier memory implementation.")
        physical = contract.get("physical_contract", {})
        expected_physical = {
            "state": "absolute-joint14",
            "action": "chunk-start-relative-eef16",
            "action_horizon": ROBOTWIN_ACTION_HORIZON,
            "gripper_model_convention": "1=open",
            "gripper_action_slots": [14, 15],
            "image_shape": list(ROBOTWIN_IMAGE_SHAPE),
            "camera_order": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        }
        if physical != expected_physical:
            raise ValueError(f"RoboTwin physical contract drifted: {physical!r}.")
        normalization = contract.get("normalization", {})
        if normalization.get("type") != "mean-std" or normalization.get("recompute") is not False:
            raise ValueError("The selected PI0.5 checkpoint requires its frozen mean/std statistics.")

        config_path = self.checkpoint / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self._validate_checkpoint_config(config)
        required = (
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_3_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
        )
        for relative in required:
            path = self.checkpoint / relative
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(path)
        MeanStdActionNormalizer.from_stats_file(self.statistics)

    @staticmethod
    def _validate_checkpoint_config(config: dict[str, Any]) -> None:
        if config.get("type") != "pi05":
            raise ValueError("RoboTwin foundation checkpoint must be LeRobot PI0.5.")
        if config.get("chunk_size") != ROBOTWIN_ACTION_HORIZON:
            raise ValueError("RoboTwin PI0.5 chunk size must remain 50.")
        if config.get("n_action_steps") != ROBOTWIN_ACTION_HORIZON:
            raise ValueError("RoboTwin PI0.5 execution horizon must remain 50.")
        if config.get("input_features", {}).get("observation.state", {}).get("shape") != [ROBOTWIN_STATE_DIM]:
            raise ValueError("RoboTwin PI0.5 state must be absolute Joint14.")
        if config.get("output_features", {}).get("action", {}).get("shape") != [ROBOTWIN_ACTION_DIM]:
            raise ValueError("RoboTwin PI0.5 output must be EEF16.")
        for key in ROBOTWIN_CAMERA_KEYS:
            if config.get("input_features", {}).get(key, {}).get("shape") != list(ROBOTWIN_IMAGE_SHAPE):
                raise ValueError(f"RoboTwin PI0.5 camera contract drifted for {key}.")
        if config.get("normalization_mapping") != {
            "ACTION": "MEAN_STD",
            "STATE": "MEAN_STD",
            "VISUAL": "IDENTITY",
        }:
            raise ValueError("RoboTwin PI0.5 processor normalization must remain mean/std + visual identity.")
        action_names = tuple(config.get("action_feature_names") or ())
        if action_names and action_names != ROBOTWIN_ACTION_NAMES:
            raise ValueError("RoboTwin EEF16 slot order does not match the selected checkpoint.")
