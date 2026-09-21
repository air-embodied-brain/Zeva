# Zeva-Ego action encoder

This package contains the two-stage visual action encoder used for Zeva-Ego
egocentric mid-training. It is independent of the ICCL implementation in
`src/openpi/zeva`.

The public boundary is code-only. It excludes video datasets, data acquisition
and cleaning tools, checkpoints, optimizer state, and machine-specific launch
files.

## Method contract

The **environment encoding stage** consumes an RGB transition
`[B,2,3,H,W]`, an action chunk `[B,L,D]`, and boolean step/dimension masks. It
learns four environment tokens while reconstructing the future visual
features.

The **task encoding stage** consumes aligned frame triplets
`[B,3,3,H,W]` and transition actions `[B,3,L,D]`. It learns 16 task tokens
while preserving the environment representation learned by the first stage.

Inference consumes already-selected RGB frame pairs and returns task and
environment tokens. Frame selection, temporal resampling, and resizing remain
the caller's responsibility. A one-second interval after temporal resampling
is the recommended default. The tokens represent visual transitions; they are
not executable robot commands.

## Installation

```bash
python -m pip install -e 'pipelines/ego_action_encoder[vision,test]'
```

The verified release environment uses Python 3.11, PyTorch 2.7.1, and DINOv2
ViT-B/14 with registers. The configuration files pin the DINOv2 source
revision used by the release recipe.

## Dataset hook

Training deliberately does not assume a storage format. Provide a callable
using `package.module:function`:

```python
def make_dataset(data_config: dict, stage: str):
    # Return a torch Dataset. Stage is either environment_encoding or
    # task_encoding. See zeva_action_encoder.contracts for required tensors.
    return dataset
```

The CLI adds `distributed_rank` and `distributed_world_size` to `data_config`
before calling the factory. Multi-source task training should assign one source
to each rank group so every local minibatch remains source-homogeneous.

Environment-encoding samples follow `EnvironmentEncodingBatch`.
Task-encoding samples follow `TaskEncodingBatch`; they must additionally
contain an integer `source_index`. Every task-encoding minibatch must be
homogeneous in `source_index`.

## Train

The release recipes contain model, loss, optimizer, scheduler, batch, and
checkpoint settings without data paths:

```bash
export ZEVA_EGO_DATASET_FACTORY=my_project.ego_data:make_dataset

torchrun --standalone --nproc-per-node=8 \
  pipelines/ego_action_encoder/scripts/train_environment_encoder.py \
  --config pipelines/ego_action_encoder/configs/environment_encoding.release.json

# Run this command on both 8-GPU nodes, with node rank 0 and 1 respectively.
torchrun --nnodes=2 --nproc-per-node=8 \
  --node-rank=<0-or-1> --master-addr=<node-0-address> --master-port=29500 \
  pipelines/ego_action_encoder/scripts/train_task_encoder.py \
  --config pipelines/ego_action_encoder/configs/task_encoding.release.json \
  --environment-checkpoint /path/to/environment-encoding-step.pt
```

Pass `--resume /path/to/checkpoint.pt` to restore model, optimizer, scheduler,
and optimizer-step state. Checkpoints use the versioned
`zeva-ego-action-encoder-v1` schema and contain no host or dataset identity.

## Encode RGB pairs

Prepare a uint8 NPY array shaped `[N,2,H,W,3]` or `[N,2,3,H,W]`, then run:

```bash
python pipelines/ego_action_encoder/scripts/encode_pairs.py \
  --input /path/to/rgb_pairs.npy \
  --checkpoint /path/to/task-encoding-checkpoint.pt \
  --output /path/to/new_output
```

The entry point does not crop, resize, temporally resample, or select frames.
DINOv2 is loaded by the vision wrapper; offline applications can populate
PyTorch's hub cache or inject a backbone through the Python API.
