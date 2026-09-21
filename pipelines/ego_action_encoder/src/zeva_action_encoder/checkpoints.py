"""Versioned checkpoint I/O for the Zeva-Ego action encoder."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import is_dataclass
import os
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

CHECKPOINT_SCHEMA = "zeva-ego-action-encoder-v1"
EncoderStage = Literal["environment_encoding", "task_encoding"]


def save_training_checkpoint(
    path: str | Path,
    *,
    stage: EncoderStage,
    step: int,
    model: nn.Module,
    model_config: object,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Atomically save model state and optional resume state.

    The public format intentionally records no host, dataset path, checksum, or
    experiment identifier. Applications may keep those details beside the
    checkpoint without changing the model contract.
    """

    target = Path(path)
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    if not is_dataclass(model_config):
        raise TypeError("model_config must be a dataclass instance")
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": stage,
        "step": int(step),
        "model": model.state_dict(),
        "model_config": asdict(model_config),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint_payload(
    path: str | Path,
    *,
    expected_stage: EncoderStage | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a public checkpoint payload."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("encoder checkpoint must be a dictionary")
    required = {"schema", "stage", "step", "model", "model_config"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"encoder checkpoint is missing fields: {missing}")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported encoder checkpoint schema: {payload['schema']!r}")
    if payload["stage"] not in {"environment_encoding", "task_encoding"}:
        raise ValueError(f"unsupported encoder stage: {payload['stage']!r}")
    if expected_stage is not None and payload["stage"] != expected_stage:
        raise ValueError(f"expected {expected_stage!r} checkpoint, found {payload['stage']!r}")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    if not isinstance(payload["model"], dict) or not isinstance(payload["model_config"], dict):
        raise TypeError("checkpoint model and model_config must be dictionaries")
    return payload


def restore_resume_state(
    payload: dict[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> int:
    """Restore model and optional optimizer/scheduler state and return the step."""

    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        state = payload.get("optimizer")
        if not isinstance(state, dict):
            raise ValueError("resume checkpoint does not contain optimizer state")
        optimizer.load_state_dict(state)
    if scheduler is not None:
        state = payload.get("scheduler")
        if not isinstance(state, dict):
            raise ValueError("resume checkpoint does not contain scheduler state")
        scheduler.load_state_dict(state)
    return int(payload["step"])
