"""Dataset-agnostic runtime used by the public encoder training CLIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler

DatasetFactory = Callable[[Mapping[str, Any], str], Dataset[dict[str, Any]]]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("training configuration must be a JSON object")
    return payload


def import_dataset_factory(specification: str) -> DatasetFactory:
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("dataset factory must use package.module:function syntax")
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise TypeError(f"dataset factory is not callable: {specification}")
    return factory


def initialize_distributed(device_argument: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        initialized_here = True
    if device_argument == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_argument)
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def finish_distributed(context: DistributedContext) -> None:
    if context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int, *, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def make_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    workers: int,
    context: DistributedContext,
    seed: int,
) -> tuple[DataLoader[dict[str, Any]], DistributedSampler]:
    # Use DistributedSampler even for one process. Its epoch-based permutation
    # makes a resumed run reconstruct the same batch position deterministically.
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )
    generator = torch.Generator().manual_seed(seed + context.rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        drop_last=True,
        pin_memory=context.device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )
    return loader, sampler


def cycle_loader(
    loader: DataLoader[dict[str, Any]],
    sampler: DistributedSampler,
    *,
    consumed_batches: int = 0,
):
    if consumed_batches < 0 or len(loader) == 0:
        raise ValueError("consumed_batches must be non-negative and the loader must be non-empty")
    epoch, skip = divmod(consumed_batches, len(loader))
    while True:
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(skip):
            next(iterator)
        yield from iterator
        epoch += 1
        skip = 0


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()
    }


def wrap_ddp(model: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    if context.world_size == 1:
        return model
    kwargs: dict[str, Any] = {"find_unused_parameters": False}
    if context.device.type == "cuda":
        kwargs["device_ids"] = [context.local_rank]
        kwargs["output_device"] = context.local_rank
    return DistributedDataParallel(model, **kwargs)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def build_optimizer(model: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
        betas=tuple(config.get("betas", (0.9, 0.95))),
        eps=float(config.get("epsilon", 1e-8)),
    )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum learning-rate ratio must be in [0,1]")

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
