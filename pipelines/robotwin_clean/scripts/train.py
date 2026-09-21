#!/usr/bin/env python3
"""Run upstream LeRobot training with an application-provided RoboTwin dataset."""

from __future__ import annotations

import os

from zeva_robotwin_clean.training import import_dataset_factory, run_lerobot_training


def main() -> None:
    specification = os.environ.get("ZEVA_ROBOTWIN_DATASET_FACTORY", "")
    if not specification:
        raise RuntimeError("set ZEVA_ROBOTWIN_DATASET_FACTORY=package.module:function")
    run_lerobot_training(import_dataset_factory(specification))


if __name__ == "__main__":
    main()
