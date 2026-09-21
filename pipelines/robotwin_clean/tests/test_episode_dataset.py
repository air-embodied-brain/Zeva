import numpy as np
import torch
from zeva_robotwin_clean.contract import CAMERA_KEYS
from zeva_robotwin_clean.episodes import EpisodeArrays
from zeva_robotwin_clean.episodes import EpisodeDescriptor
from zeva_robotwin_clean.episodes import RoboTwinCleanWindowDataset


def _episode() -> EpisodeArrays:
    joint = np.arange(42, dtype=np.float32).reshape(3, 14)
    eef = np.zeros((3, 16), dtype=np.float32)
    eef[:, 6] = 1.0
    eef[:, 14] = 1.0
    eef[:, 0] = (0.0, 0.1, 0.3)
    eef[:, 7] = (0.0, 1.0, 1.0)
    eef[:, 15] = (1.0, 0.0, 0.0)
    images = {key: np.zeros((3, 8, 10, 3), dtype=np.uint8) for key in CAMERA_KEYS}
    return EpisodeArrays(joint14=joint, absolute_eef16=eef, images=images)


def test_clean_episode_to_natural_windows() -> None:
    descriptor = EpisodeDescriptor("episode-0", "Clean", "turn_switch", "turn the switch", 3)
    dataset = RoboTwinCleanWindowDataset([descriptor], lambda _: _episode())
    assert len(dataset) == 3
    first = dataset[0]
    assert first["index"] == 0
    assert first["episode_index"] == 0
    assert first["frame_index"] == 0
    assert dataset.num_frames == 3
    assert dataset.num_episodes == 1
    assert first["observation.state"].shape == (14,)
    assert first["action"].shape == (50, 16)
    torch.testing.assert_close(first["action"][0, 0], torch.tensor(0.1))
    torch.testing.assert_close(first["action"][1, 0], torch.tensor(0.3))
    torch.testing.assert_close(first["action"][-1, 0], torch.tensor(0.3))
    torch.testing.assert_close(first["action"][0, 14:], torch.tensor([0.0, 1.0]))
    assert first[CAMERA_KEYS[0]].shape == (3, 8, 10)


def test_last_window_repeats_terminal_target() -> None:
    descriptor = EpisodeDescriptor("episode-0", "Clean", "turn_switch", "turn the switch", 3)
    sample = RoboTwinCleanWindowDataset([descriptor], lambda _: _episode())[-1]
    torch.testing.assert_close(
        sample["action"][:, :14], torch.tensor([0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]).expand(50, -1)
    )
