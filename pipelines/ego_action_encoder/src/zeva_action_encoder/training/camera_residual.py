"""Optional frozen camera-motion regressor used by one Stage 2 objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FrozenCameraResidualPredictor:
    input_mean: Tensor
    input_scale: Tensor
    weights: Tensor
    output_mean: Tensor
    trajectory_points: int

    def predict(self, action_descriptor: Tensor, duration_seconds: Tensor) -> Tensor:
        expected_dim = 14 * self.trajectory_points
        if action_descriptor.ndim != 2 or action_descriptor.shape[1] != expected_dim:
            raise ValueError(f"camera predictor expects action descriptor [N,{expected_dim}]")
        if duration_seconds.shape != (len(action_descriptor),):
            raise ValueError("duration_seconds must have shape [N]")
        features = torch.cat([action_descriptor.float(), duration_seconds.float()[:, None]], dim=1)
        return ((features - self.input_mean) / self.input_scale) @ self.weights + self.output_mean
