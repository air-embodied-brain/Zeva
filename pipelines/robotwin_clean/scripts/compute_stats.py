#!/usr/bin/env python3
"""Compute RoboTwin clean train-split normalization statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from zeva_robotwin_clean.normalization import compute_train_statistics
from zeva_robotwin_clean.training import import_dataset_factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-factory", default=os.environ.get("ZEVA_ROBOTWIN_DATASET_FACTORY", ""))
    args = parser.parse_args()
    if not args.dataset_factory:
        raise RuntimeError("set --dataset-factory or ZEVA_ROBOTWIN_DATASET_FACTORY")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train, _ = import_dataset_factory(args.dataset_factory)(config)
    result = compute_train_statistics(train)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: result[key]["count"] for key in ("observation.state", "action")}))


if __name__ == "__main__":
    main()
