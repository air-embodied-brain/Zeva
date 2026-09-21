import numpy as np
import pytest
import torch

from zeva_action_encoder.contracts import validate_stage1_batch, validate_stage2_batch
from zeva_action_encoder.inference import iter_numpy_pair_batches


def test_stage1_batch_contract() -> None:
    validate_stage1_batch(
        {
            "images": torch.zeros(2, 2, 3, 14, 14),
            "action_chunk": torch.zeros(2, 5, 16),
            "action_mask": torch.ones(2, 5, dtype=torch.bool),
            "action_dimension_mask": torch.ones(2, 5, 16, dtype=torch.bool),
        }
    )


def test_stage2_batch_contract() -> None:
    validate_stage2_batch(
        {
            "images": torch.zeros(2, 3, 3, 14, 14),
            "action_chunk": torch.zeros(2, 3, 5, 16),
            "action_mask": torch.ones(2, 3, 5, dtype=torch.bool),
            "action_dimension_mask": torch.ones(2, 3, 5, 16, dtype=torch.bool),
        }
    )


def test_numpy_pair_layout_conversion() -> None:
    source = np.zeros((3, 2, 14, 28, 3), dtype=np.uint8)
    batches = list(iter_numpy_pair_batches(source, batch_size=2))
    assert [batch.shape for batch in batches] == [(2, 2, 3, 14, 28), (1, 2, 3, 14, 28)]


def test_pair_input_rejects_non_rgb() -> None:
    source = np.zeros((1, 2, 14, 28, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        list(iter_numpy_pair_batches(source, batch_size=1))
