# RoboTwin CTE dataset adapter

The dataset adapter is included in
[`openpi.zeva.robotwin_dataset`](../../src/openpi/zeva/robotwin_dataset.py).
It does not require the internal `egoscale` package.

Create `<dataset-root>/adapter.json` with paths to your prepared caches and
matching normalization statistics:

```json
{
  "schema": "egoscale-robotwin-lerobot-relative-eef16-v1",
  "dataset_root": "/path/to/robotwin-lerobot",
  "eef_cache_root": "/path/to/eef-cache",
  "joint_cache_root": "/path/to/joint-cache",
  "stats_path": "/path/to/mean-std.json",
  "splits": ["Clean", "Randomized"],
  "action_horizon": 50,
  "source_fps": 15,
  "video_backend": "torchcodec",
  "image_shape": [480, 640, 3],
  "validation_fraction": 0.05,
  "split_seed": 1000
}
```

The schema string is retained for compatibility with existing manifests;
it is not a Python dependency. Use absolute paths. `tasks` may optionally
restrict the task names.

Each cache root contains `partitions.json`, a list of entries such as:

```json
[
  {
    "split": "Clean",
    "task": "your_task",
    "cache_relpath": "Clean/your_task",
    "source_relpath": "Clean/your_task"
  }
]
```

Each `cache_relpath` directory contains:

| File | Contents |
| --- | --- |
| `episode-indices.npy` | Original episode IDs, one per episode |
| `source-lengths.npy` | Number of native 15 Hz frames per episode |
| `episode-offsets.npy` | Start row of each episode in the concatenated cache |
| EEF: `absolute-eef-30hz.npy` | Float32 `[N,16]`: left xyz/xyzw/gripper, then right xyz/xyzw/gripper |
| EEF: `prompts.json` | Instruction strings in episode order |
| Joint: `absolute-joint15.npy` | Float32 `[M,14]`: left six joints/gripper, then right six joints/gripper |

EEF and joint episode IDs and source lengths must match. In the EEF cache,
each native frame is at row `episode_offset + 2 * frame`; in the joint cache
it is at `episode_offset + frame`. EEF poses must use the fixed main-camera
reference frame and xyzw quaternions; cached grippers use `1=closed`.

Videos are read from
`<dataset_root>/<source_relpath>/videos/chunk-XXX/<camera>/episode_XXXXXX.mp4`,
where cameras are `observation.images.cam_high`,
`observation.images.cam_left_wrist`, and `observation.images.cam_right_wrist`.
Decoding requires LeRobot, TorchCodec, and a compatible FFmpeg installation.

Statistics contain `observation.state` and `action` mappings, each with
`mean` and positive `std` arrays of length 14 and 16, respectively. Action
statistics must match chunk-start-relative EEF16 with grippers in the final
two slots and `1=open`. The loader uses existing statistics; it does not
generate caches, compute statistics, or convert a foundation checkpoint.

Check the manifest before training:

```bash
PYTHONPATH=src python -c 'from openpi.zeva.robotwin_data import TorchCodecRoboTwinDataset; d = TorchCodecRoboTwinDataset("/path/to/dataset-root/adapter.json", "train"); print(d.dataset.num_episodes, d.dataset[0]["action"].shape)'
```
