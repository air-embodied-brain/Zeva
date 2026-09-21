# Zeva-Ego RoboTwin clean post-training

This package isolates the RoboTwin training and evaluation boundary used by
Zeva-Ego. It does not modify or depend on the ICCL implementation.

The release is code-only: users provide RoboTwin episodes, model checkpoints,
and simulator assets. No data downloader, private storage layout, generated
trajectory, or cluster launcher is included.

## Training contract

- training split: RoboTwin `Clean` only;
- observation state: Joint14;
- images: high, left-wrist, and right-wrist RGB cameras;
- action: 50-step, chunk-start-relative EEF16;
- EEF order: left xyz/xyzw, right xyz/xyzw, left/right gripper;
- pose deltas: spatial deltas in fixed main-camera axes;
- gripper: absolute command with model-side `1=open`;
- sampling: natural window sampling, with no task augmentation or balancing;
- normalization: train-split QUANTILES for state/action and IDENTITY for RGB.

`RoboTwinCleanWindowDataset` converts lazy Clean episodes into this contract.
Applications retain control of storage by supplying an `episode_loader`.
`PreparedRoboTwinDatasetAdapter` validates datasets that are already converted.

## Installation

```bash
python -m pip install -e 'pipelines/robotwin_clean[train,test]'
```

The `train` extra pins the exact public LeRobot revision used by the release
configuration.

## Dataset factory

Expose a callable with LeRobot's dataset-factory signature:

```python
def make_datasets(train_config):
    return prepared_train_dataset, prepared_validation_dataset
```

The factory should construct the dataset with only Clean episodes and attach
the train-split statistics through the normal LeRobot metadata interface.

## Normalization statistics

```bash
export ZEVA_ROBOTWIN_DATASET_FACTORY=my_project.robotwin:make_datasets
python pipelines/robotwin_clean/scripts/compute_stats.py \
  --config pipelines/robotwin_clean/configs/robotwin_clean.release.json \
  --output outputs/robotwin-clean-stats.json
```

The statistics builder uses LeRobot's streaming quantile estimator and never
uses validation or randomized evaluation episodes.

## Post-train

Copy the release JSON and replace only the prepared dataset, initializer, and
output paths. It records the complete 60k-step training recipe.

```bash
export ZEVA_ROBOTWIN_DATASET_FACTORY=my_project.robotwin:make_datasets
python -m accelerate.commands.launch \
  --num_machines 1 --num_processes 8 --mixed_precision no \
  pipelines/robotwin_clean/scripts/train.py \
  --config_path=pipelines/robotwin_clean/configs/robotwin_clean.release.json
```

Resume uses the upstream LeRobot checkpoint and `--resume=true`; no custom
optimizer or checkpoint implementation is inserted.

The initializer's policy preprocessor also owns its tokenizer reference. On
offline clusters, cache that tokenizer in advance or replace its
`tokenizer_name` with a local Transformers tokenizer directory. For faster
distributed startup, stage the initializer and resume checkpoint on node-local
storage before launching all ranks.

## Evaluate

The published protocol uses all 50 tasks, `demo_randomized`, seen
instructions, 100 episodes per task, seeds 1000--1099, and H15 replanning.
Provide a callable that executes one job and returns `bool` or a mapping with a
`success` field:

```python
def evaluate_episode(job: dict, application_config: dict):
    return {"success": run_robotwin(job)}
```

Then run:

```bash
export ZEVA_ROBOTWIN_EVALUATOR=my_project.robotwin_eval:evaluate_episode
python pipelines/robotwin_clean/scripts/evaluate.py \
  --config pipelines/robotwin_clean/configs/robotwin_randomized_eval.release.json \
  --output outputs/robotwin-randomized-episodes.jsonl \
  --summary outputs/robotwin-randomized-summary.json
```

The JSONL output is append-only and resumable. The summary requires all 5,000
episodes and reports per-task and macro success rates.
