"""RoboTwin cache loading and physical contracts, without private dependencies."""

import json
import sys
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openpi.zeva.robotwin_contract import ROBOTWIN_ADAPTER_SCHEMA  # noqa: E402
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS  # noqa: E402
from openpi.zeva.robotwin_data import TorchCodecRoboTwinDataset  # noqa: E402
from openpi.zeva.robotwin_dataset import RoboTwinLeRobotEEF16Dataset  # noqa: E402
from openpi.zeva.robotwin_dataset import align_relative_eef16_grippers_with_stage1  # noqa: E402
from openpi.zeva.robotwin_dataset import robotwin_validation_episode_mask  # noqa: E402
from openpi.zeva.robotwin_geometry import canonicalize_quaternion_xyzw  # noqa: E402
from openpi.zeva.robotwin_geometry import quaternion_xyzw_to_rotation_matrix  # noqa: E402
from openpi.zeva.robotwin_geometry import robotwin_absolute_to_relative_eef16  # noqa: E402


@pytest.fixture
def cache_manifest(tmp_path):
    lengths = np.array([32, 33, 34, 35], dtype=np.int64)
    indices = np.array([1001, 1004, 1007, 1010], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    eef = np.zeros((2 * offsets[-1], 16), dtype=np.float32)
    joint = np.zeros((offsets[-1], 14), dtype=np.float32)
    for episode, length in enumerate(lengths):
        start = 2 * offsets[episode]
        poses = eef[start : start + 2 * length]
        poses[:, 0] = episode * 100 + np.arange(2 * length) / 2
        poses[:, 8] = episode * 200 + np.arange(2 * length)
        poses[:, [6, 14]] = 1
        poses[:, 7] = 0.2
        poses[:, 15] = 0.8
        joint[offsets[episode] : offsets[episode + 1]] = episode + 0.25
    for kind in ("eef", "joint"):
        root = tmp_path / kind
        root.mkdir()
        (root / "partitions.json").write_text(
            json.dumps([{"split": "Clean", "task": "demo", "cache_relpath": ".", "source_relpath": "Clean/demo"}])
        )
        np.save(root / "episode-indices.npy", indices)
        np.save(root / "source-lengths.npy", lengths)
        np.save(root / "episode-offsets.npy", offsets * (2 if kind == "eef" else 1))
    np.save(tmp_path / "eef/absolute-eef-30hz.npy", eef)
    np.save(tmp_path / "joint/absolute-joint15.npy", joint)
    (tmp_path / "eef/prompts.json").write_text(json.dumps([f"instruction {i}" for i in range(4)]))
    stats = {key: {"mean": [0] * dim, "std": [1] * dim} for key, dim in (("action", 16), ("observation.state", 14))}
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    manifest = tmp_path / "adapter.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": ROBOTWIN_ADAPTER_SCHEMA,
                "dataset_root": str(tmp_path / "source"),
                "eef_cache_root": str(tmp_path / "eef"),
                "joint_cache_root": str(tmp_path / "joint"),
                "stats_path": str(tmp_path / "stats.json"),
                "action_horizon": 50,
                "validation_fraction": 0.25,
                "split_seed": 1000,
            }
        )
    )
    return manifest


def test_wrapper_loads_without_egoscale_and_preserves_h50(cache_manifest):
    wrapper = TorchCodecRoboTwinDataset(cache_manifest, "all")
    dataset = wrapper.dataset
    assert "egoscale" not in sys.modules
    assert dataset.num_episodes == 4
    assert dataset.num_frames == 134
    sample = dataset[0]
    assert sample["action"].shape == (50, 16)
    assert sample["observation.state"].shape == (14,)
    assert sample["task"] == "instruction 0"
    assert not any(key in sample for key in ROBOTWIN_CAMERA_KEYS)
    np.testing.assert_array_equal(sample["action"][:3, 0].numpy(), [1, 2, 3])
    np.testing.assert_array_equal(sample["action"][:3, 7].numpy(), [2, 4, 6])
    torch.testing.assert_close(sample["action"][:, 14:], torch.tensor([0.8, 0.2]).expand(50, 2))
    # Terminal padding stays within this episode, never the next episode.
    terminal = dataset[31]["action"]
    assert terminal[:, [0, 7]].count_nonzero() == 0
    assert dataset[32]["task"] == "instruction 1"
    assert dataset[-1]["episode_index"] == 3
    assert dataset[-1]["frame_index"] == 34
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_episode_split_is_disjoint_and_repeatable(cache_manifest):
    train = RoboTwinLeRobotEEF16Dataset(cache_manifest, subset="train")
    validation = RoboTwinLeRobotEEF16Dataset(cache_manifest, subset="validation")
    repeated = RoboTwinLeRobotEEF16Dataset(cache_manifest, subset="validation")
    train_ids = {row["episode_index"] for row in train._records}  # noqa: SLF001
    val_ids = {row["episode_index"] for row in validation._records}  # noqa: SLF001
    assert not train_ids & val_ids
    assert train_ids | val_ids == {1001, 1004, 1007, 1010}
    assert len(train_ids) == 3
    assert len(val_ids) == 1
    assert val_ids == {row["episode_index"] for row in repeated._records}  # noqa: SLF001
    assert train.meta.stats["action"]["mean"].shape == (16,)
    mask = robotwin_validation_episode_mask(np.array([0, 0, 1]), np.array([0, 1, 0]), fraction=0.5, seed=1000)
    assert mask[:2].sum() == 1
    assert not mask[2]


def test_misaligned_cache_is_rejected(cache_manifest):
    indices = cache_manifest.parent / "joint/episode-indices.npy"
    np.save(indices, np.array([0, 1, 2, 3]))
    with pytest.raises(ValueError, match="indices differ"):
        RoboTwinLeRobotEEF16Dataset(cache_manifest)


@pytest.mark.parametrize(("key", "value"), [("schema", "wrong"), ("action_horizon", 15), ("validation_fraction", -0.1)])
def test_invalid_manifest_is_rejected(cache_manifest, key, value):
    payload = json.loads(cache_manifest.read_text())
    payload[key] = value
    cache_manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=r"manifest|horizon|validation_fraction"):
        RoboTwinLeRobotEEF16Dataset(cache_manifest)


def test_invalid_stats_are_rejected(cache_manifest):
    path = cache_manifest.parent / "stats.json"
    stats = json.loads(path.read_text())
    stats["action"].pop("mean")
    stats["action"]["min"] = [0] * 16
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="mean/std"):
        RoboTwinLeRobotEEF16Dataset(cache_manifest)


def test_spatial_rotation_order_and_quaternion_sign():
    state = np.zeros(16, dtype=np.float32)
    state[3:7] = canonicalize_quaternion_xyzw(np.array([1, 0, 0, 1]))
    state[11:15] = [0, 0, 0, 1]
    future = np.tile(state, (2, 1))
    future[:, 3:7] = canonicalize_quaternion_xyzw(np.array([0, 1, 0, 1]))
    future[:, 0] = 3
    result = robotwin_absolute_to_relative_eef16(state, future)
    expected = quaternion_xyzw_to_rotation_matrix(future[:, 3:7]) @ quaternion_xyzw_to_rotation_matrix(state[3:7]).T
    np.testing.assert_allclose(quaternion_xyzw_to_rotation_matrix(result[:, 3:7]), expected, atol=1e-6)
    np.testing.assert_array_equal(result[:, 0], [3, 3])
    negated = future.copy()
    negated[:, 3:7] *= -1
    np.testing.assert_array_equal(result, robotwin_absolute_to_relative_eef16(state, negated))
    with pytest.raises(ValueError, match="norm"):
        quaternion_xyzw_to_rotation_matrix(np.zeros(4))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        align_relative_eef16_grippers_with_stage1(np.full((1, 16), 2))


def test_video_frame_paths_order_and_backend(cache_manifest, monkeypatch):
    calls = []
    package = ModuleType("lerobot")
    package.__path__ = ["/unused"]
    datasets = ModuleType("lerobot.datasets")
    videos = ModuleType("lerobot.datasets.video_utils")

    def decode(path, timestamps, **kwargs):
        calls.append((path, timestamps, kwargs))
        return torch.stack([torch.full((3, 2, 2), round(t * 15), dtype=torch.uint8) for t in timestamps])

    videos.decode_video_frames = decode
    for name, module in (
        ("lerobot", package),
        ("lerobot.datasets", datasets),
        ("lerobot.datasets.video_utils", videos),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    wrapper = TorchCodecRoboTwinDataset(cache_manifest, "all")
    images = wrapper.read_images(wrapper.dataset._records[0], [2, 0, 2])  # noqa: SLF001
    assert tuple(images) == ROBOTWIN_CAMERA_KEYS
    for key, image in images.items():
        assert image[:, 0, 0, 0].tolist() == [2, 0, 2]
        assert any(str(path).endswith(f"chunk-001/{key}/episode_001001.mp4") for path, _, _ in calls)
    assert len(calls) == 3
    assert all(
        timestamps == [0.0, 2 / 15] and kwargs["backend"] == "torchcodec" and kwargs["return_uint8"]
        for _, timestamps, kwargs in calls
    )


def test_cte_episode_batch_uses_public_adapter(cache_manifest, monkeypatch):
    pytest.importorskip("tyro")
    from scripts.robotwin.train_cte import Episodes  # noqa: PLC0415
    from scripts.robotwin.train_cte import collate  # noqa: PLC0415

    def images(_self, _record, frames):
        return {key: torch.zeros(len(frames), 3, 2, 2, dtype=torch.uint8) for key in ROBOTWIN_CAMERA_KEYS}

    monkeypatch.setattr(TorchCodecRoboTwinDataset, "read_images", images)
    dataset = Episodes(cache_manifest, "all", ["demo"])
    batch = collate([dataset[0], dataset[1]])
    assert batch["images"].shape == (2, 3, 3, 3, 224, 224)
    assert batch["actions"].shape == (2, 2, 15, 16)
    assert batch["valid"].all()
    assert batch["images"].eq(-1).all()
    torch.testing.assert_close(batch["actions"][0, 0, :, 0], torch.arange(1, 16, dtype=torch.float32))
    assert torch.isfinite(batch["actions"]).all()
