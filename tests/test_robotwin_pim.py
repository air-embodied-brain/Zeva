from __future__ import annotations

import json
from pathlib import Path

import torch

from openpi.zeva.pim_policy import AttemptPersistentMemory
from openpi.zeva.pim_policy import EpisodePersistentMemory
from scripts.robotwin.train_within_episode_pim import causal_episode_history


def _token(value: float) -> torch.Tensor:
    return torch.full((1, 256), value)


def test_cross_attempt_memory_commits_only_at_attempt_reset() -> None:
    memory = AttemptPersistentMemory(max_attempts=2, max_entries_per_attempt=4)
    memory.append_bit(_token(1), _token(2))
    assert memory.entries() == (None, None)

    memory.reset_attempt()
    phase, bit = memory.entries()
    assert phase is not None and bit is not None
    assert phase.shape == bit.shape == (1, 1, 256)
    assert torch.equal(phase[:, 0], _token(1))
    assert torch.equal(bit[:, 0], _token(2))

    memory.reset_episode()
    assert memory.entries() == (None, None)


def test_within_episode_memory_is_causal_and_bounded() -> None:
    memory = EpisodePersistentMemory(max_entries=2)
    memory.append_bit(_token(1), _token(11))
    memory.append_bit(_token(2), _token(12))
    memory.append_bit(_token(3), _token(13))

    phase, bit = memory.entries()
    assert phase is not None and bit is not None
    assert phase.shape == bit.shape == (1, 2, 256)
    assert torch.equal(phase[:, 0], _token(2))
    assert torch.equal(phase[:, 1], _token(3))

    memory.reset_episode()
    assert memory.entries() == (None, None)


def test_training_prefix_excludes_current_boundary() -> None:
    row = {
        "phase": torch.stack([_token(1)[0], _token(2)[0], _token(3)[0]]),
        "effect": torch.stack([_token(11)[0], _token(12)[0], _token(13)[0]]),
    }
    phase, bit, mask = causal_episode_history(row, timestep=2, capacity=4)
    assert mask.tolist() == [True, True, False, False]
    assert torch.equal(phase[0], row["phase"][0])
    assert torch.equal(phase[1], row["phase"][1])
    assert torch.equal(bit[1], row["effect"][1])


def test_public_training_settings_point_to_existing_entrypoints() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs/robotwin_pim_training_settings.json").read_text())
    paths = [
        settings["parent"]["entrypoint"],
        settings["pim"]["cross_attempt"]["artifact_builder"],
        settings["pim"]["cross_attempt"]["entrypoint"],
        settings["pim"]["within_episode"]["entrypoint"],
    ]
    assert all((root / path).is_file() for path in paths)

