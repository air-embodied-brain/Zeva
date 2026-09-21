"""Minimal hook that keeps optimization inside the upstream LeRobot trainer."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from torch.utils.data import Dataset


DatasetPair = tuple[Dataset[dict[str, Any]], Dataset[dict[str, Any]] | None]
DatasetFactory = Callable[[Any], DatasetPair]


def import_dataset_factory(specification: str) -> DatasetFactory:
    """Resolve ``package.module:function`` supplied by the application."""

    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("dataset factory must use package.module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name, None)
    if not callable(factory):
        raise TypeError(f"dataset factory is not callable: {specification}")
    return factory


def run_lerobot_training(dataset_factory: DatasetFactory) -> None:
    """Patch only LeRobot's dataset boundary and run its native trainer."""

    from lerobot.scripts import lerobot_train

    lerobot_train.make_train_eval_datasets = dataset_factory
    lerobot_train.train()
