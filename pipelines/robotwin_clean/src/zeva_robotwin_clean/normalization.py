"""Train-split normalization statistics for RoboTwin clean post-training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

FEATURES = ("observation.state", "action")


def compute_train_statistics(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the statistics consumed by LeRobot's QUANTILES normalizer.

    The iterable must contain only training samples. Quantiles are accumulated
    with LeRobot's own streaming estimator so this function does not require
    materializing the full RoboTwin dataset in memory.
    """

    try:
        from lerobot.datasets.compute_stats import RunningQuantileStats  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - exercised in the train environment
        raise RuntimeError("install zeva-robotwin-clean[train] to compute statistics") from error

    accumulators = {key: RunningQuantileStats() for key in FEATURES}
    totals: dict[str, np.ndarray] = {}
    squares: dict[str, np.ndarray] = {}
    minima: dict[str, np.ndarray] = {}
    maxima: dict[str, np.ndarray] = {}
    counts = dict.fromkeys(FEATURES, 0)
    for sample in samples:
        for key in FEATURES:
            tensor = sample[key]
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise TypeError(f"{key} must be a floating tensor")
            values = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
            if not np.isfinite(flat).all():
                raise ValueError(f"{key} contains non-finite values")
            if key not in totals:
                totals[key] = np.zeros(flat.shape[-1], dtype=np.float64)
                squares[key] = np.zeros(flat.shape[-1], dtype=np.float64)
                minima[key] = np.full(flat.shape[-1], np.inf)
                maxima[key] = np.full(flat.shape[-1], -np.inf)
            counts[key] += len(flat)
            totals[key] += flat.sum(axis=0)
            squares[key] += np.square(flat).sum(axis=0)
            minima[key] = np.minimum(minima[key], flat.min(axis=0))
            maxima[key] = np.maximum(maxima[key], flat.max(axis=0))
            accumulators[key].update(flat.astype(np.float32, copy=False))
    result: dict[str, Any] = {"schema": "zeva-ego-robotwin-normalization-v1", "subset": "train"}
    for key in FEATURES:
        if counts[key] == 0:
            raise ValueError("the training sample iterable is empty")
        mean = totals[key] / counts[key]
        variance = np.maximum(squares[key] / counts[key] - np.square(mean), 0.0)
        quantiles = accumulators[key].get_statistics()
        result[key] = {
            "count": counts[key],
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": minima[key].tolist(),
            "max": maxima[key].tolist(),
            **{name: np.asarray(quantiles[name]).tolist() for name in ("q01", "q10", "q50", "q90", "q99")},
        }
    return result


def quantile_normalize(values: torch.Tensor, statistics: Mapping[str, Any]) -> torch.Tensor:
    """Map the 1st--99th percentile interval to [-1,1]."""

    lower = values.new_tensor(statistics["q01"])
    upper = values.new_tensor(statistics["q99"])
    scale = torch.clamp(upper - lower, min=1e-6)
    return 2.0 * (values - lower) / scale - 1.0


def quantile_unnormalize(values: torch.Tensor, statistics: Mapping[str, Any]) -> torch.Tensor:
    lower = values.new_tensor(statistics["q01"])
    upper = values.new_tensor(statistics["q99"])
    return (values + 1.0) * 0.5 * (upper - lower) + lower
