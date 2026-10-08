from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openpi.zeva.pim_policy import AttemptPersistentMemory  # noqa: E402


def token(x, y=0):
    return torch.tensor([[float(x), float(y)]])


def test_uniform_mode_preserves_historical_sampling():
    memory = AttemptPersistentMemory(compression="uniform", max_entries_per_attempt=4)
    for number in range(10):
        memory.append_bit(token(number, 1), token(number + 20, 1))
    assert memory.entries() == (None, None)
    memory.reset_attempt()
    phase, bit = memory.entries()
    assert phase[0, :, 0].tolist() == [0, 3, 6, 9]
    assert bit[0, :, 0].tolist() == [20, 23, 26, 29]


def test_default_mode_merges_similar_entries():
    memory = AttemptPersistentMemory()
    assert memory.compression == "similarity_merge"
    assert memory.merge_threshold == 0.95
    assert memory.phase_weight == 0.5
    memory.append_bit(token(1), token(2))
    memory.append_bit(token(3), token(4))
    memory.reset_attempt()
    phase, bit = memory.entries()
    assert phase.shape == bit.shape == (1, 1, 2)
    torch.testing.assert_close(phase[0, 0], token(2)[0])
    torch.testing.assert_close(bit[0, 0], token(3)[0])


def test_similarity_mode_merges_with_count_weights_across_attempts():
    memory = AttemptPersistentMemory(compression="similarity_merge")
    memory.append_bit(token(1), token(2))
    memory.append_bit(token(3), token(4))
    memory.reset_attempt()
    memory.append_bit(token(5), token(6))
    memory.append_bit(token(0, 1), token(0, 1))
    memory.reset_attempt()
    phase, bit = memory.entries()
    assert phase.shape == bit.shape == (1, 2, 2)
    torch.testing.assert_close(phase[0, 0], token(3)[0])
    torch.testing.assert_close(bit[0, 0], token(4)[0])
    torch.testing.assert_close(phase[0, 1], token(0, 1)[0])


def test_similarity_mode_uses_effect_and_evicts_whole_attempts():
    memory = AttemptPersistentMemory(compression="similarity_merge", max_attempts=1)
    memory.append_bit(token(1), token(1))
    memory.append_bit(token(1), token(-1))
    memory.reset_attempt()
    assert memory.snapshot()["pim_entries"] == 2
    memory.append_bit(token(0, 1), token(0, 3))
    memory.reset_attempt()
    phase, bit = memory.entries()
    assert phase.shape == (1, 1, 2)
    torch.testing.assert_close(bit[0, 0], token(0, 3)[0])
    memory.reset_episode()
    assert memory.entries() == (None, None)
    assert memory.snapshot()["pim_attempts"] == 0


def test_similarity_mode_uniformly_bounds_unmerged_entries():
    memory = AttemptPersistentMemory(compression="similarity_merge", max_entries_per_attempt=2)
    for number in range(4):
        value = torch.eye(4)[number : number + 1]
        memory.append_bit(value, value)
    memory.reset_attempt()
    phase, _ = memory.entries()
    torch.testing.assert_close(phase[0], torch.eye(4)[[0, 3]])


@pytest.mark.parametrize(
    "options",
    [
        {"compression": "unknown"},
        {"merge_threshold": float("nan")},
        {"merge_threshold": 1.1},
        {"phase_weight": -0.1},
    ],
)
def test_memory_rejects_invalid_compression_options(options):
    with pytest.raises(ValueError, match="PIM"):
        AttemptPersistentMemory(**options)
