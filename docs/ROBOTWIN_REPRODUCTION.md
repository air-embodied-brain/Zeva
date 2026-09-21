# RoboTwin training reproduction

This document contains the public data contract and training commands. Replace every placeholder path with a local path.

## Data contract

The RoboTwin adapter is expected at:

```text
<dataset-root>/adapter.json
```

Each record must expose:

- three RGB streams defined by `ROBOTWIN_CAMERA_KEYS`;
- joint state used by the foundation-policy preprocessor;
- EEF16 actions;
- task text;
- episode identity and length.

ZeVA predicts H50 and executes H15 before replanning. Action normalization is read from the supplied RoboTwin handoff and is not recomputed during the method stages.

A task subset file is a JSON object:

```json
{
  "task_names": ["task_a", "task_b"]
}
```

## 1. Train CTE and BIT

```bash
uv run python scripts/robotwin/train_cte.py \
  --dataset-root <dataset-root> \
  --handoff-root <robotwin-handoff> \
  --task-subset <task-subset.json> \
  --save-dir <runs>/cte
```

Default recipe:

- 80 epochs
- batch size 8
- ImageNet-initialized ResNet-18 visual encoder
- four tri-stream causal blocks
- vision learning rate `1e-5`
- remaining CTE learning rate `1e-4`
- effect-loss weight `0.2`
- validation every five epochs

The epoch-80 checkpoint must be finite and pass the action, vision, and effect validation checks before Stage 2 loads it.

## 2. Export CTE artifacts

```bash
uv run python scripts/robotwin/export_cte_artifacts.py \
  --dataset-root <dataset-root> \
  --handoff-root <robotwin-handoff> \
  --task-subset <task-subset.json> \
  --checkpoint <runs>/cte/cte_epoch_080.pth \
  --output <runs>/cte_artifacts.pth
```

The artifact contains train-only task memory plus recurrent H15 traces for the training and validation splits. It records source hashes and rejects overlapping episodes.

## 3. Train CTE + BIT + EAP

Launch with `accelerate` or `torchrun`. Choose per-rank batch size and accumulation so that:

```text
world_size × batch_size × accumulation = 256
```

```bash
uv run accelerate launch scripts/robotwin/train_cte_eap.py \
  --dataset-root <dataset-root> \
  --handoff-root <robotwin-handoff> \
  --foundation-checkpoint <foundation-checkpoint> \
  --cte-checkpoint <runs>/cte/cte_epoch_080.pth \
  --artifacts <runs>/cte_artifacts.pth \
  --retrieval-checkpoint <task-language-retrieval.pth> \
  --save-dir <runs>/cte_bit_eap
```

The fixed recipe uses 5000 optimizer steps, foundation-policy learning rate `5e-6`, EAP learning rate `5e-5`, BF16, and gradient norm clipping at 1.0.

## 4A. Cross-attempt PIM

Build label-free train-only pairings:

```bash
uv run python scripts/robotwin/build_cross_attempt_pim.py \
  --cte-artifacts <runs>/cte_artifacts.pth \
  --output <runs>/cross_attempt_pairs.pth
```

Train:

```bash
uv run accelerate launch scripts/robotwin/train_cross_attempt_pim.py \
  --dataset-root <dataset-root> \
  --handoff-root <robotwin-handoff> \
  --foundation-checkpoint <foundation-checkpoint> \
  --cte-checkpoint <runs>/cte/cte_epoch_080.pth \
  --cte-artifacts <runs>/cte_artifacts.pth \
  --pim-artifacts <runs>/cross_attempt_pairs.pth \
  --retrieval-checkpoint <task-language-retrieval.pth> \
  --parent-stage2-checkpoint <runs>/cte_bit_eap/005000 \
  --save-dir <runs>/cross_attempt_pim
```

The first 500 steps train PIM only. The remaining steps train the foundation policy, EAP, and PIM with learning rates defined in the settings file. PIM-off batches preserve the empty-memory path.

At deployment:

- `reset(scope="attempt")` commits the current BIT trace;
- `reset(scope="episode")` clears BIT recurrence and all PIM state.

## 4B. Within-episode PIM

```bash
uv run accelerate launch scripts/robotwin/train_within_episode_pim.py \
  --dataset-root <dataset-root> \
  --handoff-root <robotwin-handoff> \
  --foundation-checkpoint <foundation-checkpoint> \
  --cte-checkpoint <runs>/cte/cte_epoch_080.pth \
  --cte-artifacts <runs>/cte_artifacts.pth \
  --retrieval-checkpoint <task-language-retrieval.pth> \
  --parent-stage2-checkpoint <runs>/cte_bit_eap/005000 \
  --save-dir <runs>/within_episode_pim
```

Only PIM parameters are trainable. The recipe uses global batch 256, 2000 steps, PIM learning rate `5e-5`, and capacity 64.

For boundary `t`, the memory contains only boundaries `[0, t)`. The current BIT is appended after action prediction. Episode reset clears both recurrent BIT state and PIM.

## Checkpoints

Every 500-step Stage-2/PIM checkpoint contains:

```text
model.safetensors
zeva_adapter.pth
training_state.pth
rng_rank0.pth ... rng_rankN.pth
COMPLETE
```

Training refuses to overwrite a non-empty output directory.
