import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from zeva_robotwin_clean.contract import absolute_to_chunk_start_eef16
from zeva_robotwin_clean.contract import align_grippers_to_model_convention
from zeva_robotwin_clean.dataset import PreparedRoboTwinDatasetAdapter


def _identity_pose_state() -> np.ndarray:
    state = np.zeros(16, dtype=np.float32)
    state[6] = 1.0
    state[14] = 1.0
    return state


def test_chunk_start_relative_action() -> None:
    state = _identity_pose_state()
    future = np.repeat(state[None], 2, axis=0)
    future[0, :3] = (0.1, -0.2, 0.3)
    future[1, 8:11] = (-0.4, 0.5, 0.6)
    result = absolute_to_chunk_start_eef16(state, future)
    np.testing.assert_allclose(result[0, :3], (0.1, -0.2, 0.3), atol=1e-6)
    np.testing.assert_allclose(result[1, 7:10], (-0.4, 0.5, 0.6), atol=1e-6)
    np.testing.assert_allclose(
        result[:, 3:7],
        np.tile((0.0, 0.0, 0.0, 1.0), (2, 1)),
        atol=1e-6,
    )


def test_gripper_direction() -> None:
    action = np.zeros((2, 16), dtype=np.float32)
    action[:, 14:] = ((0.0, 1.0), (0.25, 0.75))
    converted = align_grippers_to_model_convention(action)
    np.testing.assert_allclose(converted[:, 14:], ((1.0, 0.0), (0.75, 0.25)))


class _PreparedDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        if index:
            raise IndexError(index)
        images = torch.zeros(3, 8, 8)
        return {
            "observation.state": torch.zeros(14),
            "observation.images.cam_high": images,
            "observation.images.cam_left_wrist": images,
            "observation.images.cam_right_wrist": images,
            "action": torch.zeros(50, 16),
            "task": "put the object in the container",
        }


def test_prepared_dataset_adapter() -> None:
    sample = PreparedRoboTwinDatasetAdapter(_PreparedDataset())[0]
    assert sample["action"].shape == (50, 16)


def test_adapter_rejects_wrong_horizon() -> None:
    source = _PreparedDataset()
    sample = source[0]
    sample["action"] = torch.zeros(15, 16)

    class _Invalid(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            return sample

    with pytest.raises(ValueError):
        PreparedRoboTwinDatasetAdapter(_Invalid())[0]
