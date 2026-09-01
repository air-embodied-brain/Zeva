# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.model.zeva import (
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    causal_transition_encoder_loss,
)


def test_cte_forward_loss_and_ema() -> None:
    model = CausalTransitionEncoder(CausalTransitionEncoderConfig(hidden_dim=32, retrieval_dim=16, phase_dim=16, num_layers=1, num_heads=4, use_mamba=False))
    frames = torch.rand(3, 5, 3, 48, 48)
    actions = torch.rand(3, 4, 4, 8)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    output = model(frames, actions, valid)
    assert output["causal_interaction_state"].shape == (3, 5, 32)
    assert output["retrieval"].shape == (3, 5, 16)
    losses = causal_transition_encoder_loss(output, actions, valid, torch.tensor([0, 0, 1]))
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    before = next(model.target_visual_encoder.parameters()).clone()
    with torch.no_grad():
        next(model.visual_encoder.parameters()).add_(0.01)
    model.update_ema_target()
    assert not torch.equal(before, next(model.target_visual_encoder.parameters()))
