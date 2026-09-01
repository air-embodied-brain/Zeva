from __future__ import annotations

import torch

from cosmos_framework.model.zeva import (
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    normalize_cte_state_dict,
)


def test_cte_key_normalization_preserves_every_tensor() -> None:
    config = CausalTransitionEncoderConfig(
        hidden_dim=16,
        retrieval_dim=8,
        phase_dim=8,
        effect_dim=8,
        num_layers=1,
        num_heads=4,
        use_mamba=False,
    )
    source = CausalTransitionEncoder(config)
    serialized = {}
    for key, value in source.state_dict().items():
        serialized_key = key.replace("interaction_state_token", "behavior_token")
        serialized_key = serialized_key.replace(".interaction_state.", ".behavior.")
        serialized[serialized_key] = value.clone()

    restored = CausalTransitionEncoder(config)
    restored.load_state_dict(normalize_cte_state_dict(serialized), strict=True)
    for key, value in source.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
