# Two-stage training and completed-trial memory

The pipeline trains a causal transition encoder (CTE), then a Cosmos action policy and two small memory-gain predictors. Replay keeps the model weights frozen. The memory bank and evidence used to select a retrieval mode can change after each completed trial.

## Run

Use the Cosmos overlay described in the [installation instructions](../README.md#installation) and run from its root. Copy [the example config](../examples/training/two_stage.example.json) to a local config. Supply your dataset adapter, action-policy TOML, and frozen-prediction replay adapter. The example contains placeholder commands and paths.

```bash
cp examples/training/two_stage.example.json examples/training/two_stage.local.json
python -m examples.training.zeva_two_stage --config examples/training/two_stage.local.json
```

The phases are `prepare`, `stage1`, `stage2`, and `memory`. Use `--phase` to run one phase. The full run follows this order and reuses completed outputs. Use a fresh output directory when changing model, data, action coordinates, or sampling configuration. Config paths are resolved relative to the config file. Adapter commands are argument arrays executed from the current directory.

| Adapter | Input and responsibility | Output environment variable |
| --- | --- | --- |
| `prepare_command` | Align observations and executed controls. Assign episode splits before fitting any statistics. | `ZEVA_EPISODE_MANIFEST` |
| `policy_command` | Start the existing Cosmos trainer with a compatible recipe and process layout. Read `ZEVA_CTE_CHECKPOINT` and `ZEVA_CTE_FEATURES` in the loader. | `ZEVA_POLICY_CHECKPOINT` |
| `replay_command` | Load the frozen policy, use identical inputs and sampling noise for paired predictions, and score completed trials. | `ZEVA_PAIRED_EVIDENCE` |

`ZEVA_RUN_DIR` is available to all adapters. The policy checkpoint must be a nonempty file or a native distributed-checkpoint model directory containing `.metadata`. Write file outputs atomically before a command completes. A prepared manifest or paired-evidence file can also be supplied directly. The existing Cosmos recipe carries the action representation, normalization, horizon, model initialization, optimizer and hardware configuration.

## Stage 1 · causal transition encoder

The episode manifest contains an `episodes` array of records with a unique `id`, a `split` of `train`, `validation` or `holdout`, and a tensor `file` relative to the manifest. An optional integer `semantic_id` groups trajectories for the CTE's contrastive objective. Each tensor file contains the following.

| Key | Shape | Meaning |
| --- | --- | --- |
| `frames` | `[T, C, H, W]` | Aligned observations as uint8 or floats in [0, 1] |
| `actions` | `[T-1, transition_steps, action_dim]` | Executed controls between adjacent observation boundaries, in consistent configured coordinates |

The runner pads batches, passes causal validity masks to the repository CTE, and selects a checkpoint using the validation split. Export freezes the CTE and writes phase, retrieval, visual keys, next-action and next-vision predictions, and completed effect records. Use effects with `effect_end <= decision_index` to build the Brief Interaction Trace (BIT). The exported `executed_action_effect_forecast` is conditioned on recorded controls. At inference, a forecast for an action proposal must be computed from the candidate action and observations available at the decision.

## Stage 2 · action policy and memory gain

Train the action policy with the dataset's existing Cosmos recipe. Its loader can build BIT from completed effects and use other completed episodes as persistent memory. The paired replay freezes this policy. For one decision, compare its empty-memory forecast with each candidate forecast and a random-stage control from the same candidate episode. Compare each forecast to the same observed result. The gain is the baseline error minus proposal error.

The gain module provides `blend_action` to apply a bounded correction to explicitly selected action coordinates, `candidate_evidence` to score matched and random proposals, and `fit_gain_head` to fit separate action and effect predictors. Put forecasts and targets in the same physical or normalized coordinate system before scoring. Keep deployment features restricted to current observations, completed transitions, and the candidate. Keep targets in the evidence file for fitting and scoring.

The evidence file has a `fit` array of source episodes, an `initial_donors` array of completed donor IDs, and a chronological `stream` of later episodes. Fit queries and donors must be separate from the stream and its initial donors. An optional `warmup_trials` count excludes the first stream episodes from reported scores while allowing their completed evidence to calibrate selection. Each episode contains the following records.

```json
{
  "episode": "query_01",
  "decisions": [
    {
      "candidates": [
        {
          "donor": "support_00",
          "features": [0.4, 0.2],
          "distance": 0.5,
          "gain": {"action": 0.12, "effect": 0.01},
          "random_gain": {"action": -0.03, "effect": 0.0}
        }
      ]
    }
  ]
}
```

Record candidates from the previously completed episodes eligible for each memory policy being compared. Use a consistent feature layout in fit and replay. Distances rank phase matches, and the gain heads learn which candidate transferred best to an independent query. Episode weighting makes the fit insensitive to how many decisions were recorded per episode.

At the start of each trial, the evidence gate chooses empty memory, nearest-stage retrieval, or learned ranking independently for action and effect. A mode has to improve on empty memory and its matched random-stage control on recent completed trials. The gate can return to empty memory as new evidence arrives. After scoring the completed trial, rolling memory can insert it and evict the oldest record. The runner compares a fixed bank with rolling banks of one, two and four completed episodes. The two-entry bank keeps two candidates available and applies at most one per prediction channel.

## Outputs and validation

| Artifact | Contents |
| --- | --- |
| `cte.pt` and `cte_features/` | Selected CTE and frozen per-episode features |
| Configured policy checkpoint | Cosmos action-policy weights |
| `selector.json` | Action and effect gain predictors |
| `memory_replay.json` | Per-episode modes and gains, matched random controls, paired intervals, coverage and harm |

Check gains by held-out condition, episode-level uncertainty, coverage, and harm among activated trials. Keep the holdout split separate when selecting training settings. The offline protocol measures prediction error against recorded demonstrations. Evaluating the effects of the robot's own executed policy requires a separate closed-loop experiment.
