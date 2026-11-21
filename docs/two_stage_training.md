# Two-stage training and memory-conditioned actions

This guide covers training the causal transition encoder (CTE) and Cosmos action
policy, conditioning the policy on memory, and evaluating retrieval choices in
chronological offline replay. The offline gain selector is an extension for
comparing memory policies on recorded demonstrations.

## How memory changes actions

### The Zeva policy generates actions with memory as context

1. **Encode a completed interaction.** Once a control chunk has been executed
   and its outcome observed, the CTE represents the interaction as a phase
   token `p` and an effect token `e`. The phase describes where the robot is
   in the interaction. The effect describes how the observation changed after
   the executed controls.
2. **Read the relevant history.** The Brief Interaction Trace (BIT) holds
   recent completed effects from the current attempt. Persistent Interaction
   Memory (PIM) retains phase–effect entries across repeated attempts in the
   same episode and consolidates entries that are similar in both phase and
   effect. The current phase retrieves the most relevant stored effects. BIT
   resets with each attempt. PIM resets when the episode changes.
3. **Condition action generation.** The memory encoder combines the task
   context, current phase, BIT and retrieved PIM entries into a Causal Prompt.
   The action model attends to that context as it generates a new action chunk.
   With the same observation, initial noise and model weights, changing the
   memory can change the denoising predictions and the final action.
   After the resulting controls are executed, their observed effects can
   update the memory for the next decision.

The published algorithm retrieves latent effects and lets the policy generate
the controls. Its memory encoder and policy weights stay frozen during
deployment. [Zeva paper, Sections III-D–F](https://arxiv.org/html/2608.30880v1)

### Where the online code applies the prompt

The online path supplies the retrieved phases, effects and validity mask as
`behavior_pim_phase`, `behavior_pim_effect` and `behavior_pim_valid`.
`CausalPromptEncoder` combines them with the current phase, BIT and global
behavior context. In Cosmos, the projected prompt is added to the behavior
prefix before action sampling. With `has_pim` equal to 1 when any persistent
slot is valid, the injection is

```text
conditioned_prefix = base_prefix + tanh(gate) * projected_prompt * has_pim
```

When every PIM slot is invalid or the gate is zero, this residual is zero.
The base conditioning remains available. The gate is a trained model
parameter. It stays fixed during inference, while the prompt can change as the
retrieved memory changes. The server also has a separate, disabled-by-default
option to replay an earlier action chunk. That option is independent of prompt
injection and of the offline gain selector.

[Memory encoder and residual](../cosmos_framework/model/zeva/persistent_interaction_memory.py#L270)
· [Cosmos behavior prefix](../cosmos_framework/model/generator/mot/cosmos3_vfm_network.py#L718)

### What the offline gain selector does

The replay adapter generates an empty-PIM baseline `a_0` and memory-conditioned
proposals for the same decision. It holds the observation, other conditioning
such as BIT, and sampling noise fixed. A proposal can come from the online
prompt path or an explicit action-space correction. Both must be computed before
using the demonstrated continuation for scoring.

For an action-space correction, `blend_action` is an optional helper. The
adapter supplies a candidate chunk `a_m` in the same coordinates, units and
time grid as `a_0`. On the selected action coordinates it computes

```text
a_out[t] = a_0[t] + weight[t] * (a_m[t] - a_0[t])
```

The weight is constant over the configured initial horizon and tapers to zero
over the remaining steps, if any. All other coordinates keep the baseline
values. For a synthetic scalar example, a baseline of 2 and a candidate of 4
give 2.5 at weight 0.25. Select coordinates where linear interpolation is
valid. Representations such as quaternions require an appropriate interpolation
rule in the adapter.

The selector learns to predict a **gain**, defined as baseline error minus
proposal error against the same recorded continuation. It fits separate action
and effect predictors. At the start of each trial, completed-trial evidence
chooses one mode per prediction channel. The choices are empty PIM,
nearest-stage retrieval, or learned ranking. Within a trial, the selector
uses each decision's observable features to choose a candidate under that
mode. An effect prediction is scored separately and does not directly
change a command.

The runner consumes the adapter's precomputed evidence to replay those choices.
The selector and blend helper are not wired into the online policy server by
this workflow. A proposal that predicts the recorded continuation more closely
has demonstrated an offline prediction gain. Its physical outcome would need
to be observed in a closed-loop execution.

[Gain fit, blend and replay](../cosmos_framework/model/zeva/gain_memory.py)

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

Train the action policy with the dataset's Cosmos recipe. Its loader can build
BIT from completed effects and use completed episodes as persistent memory.
Then freeze the policy and produce paired evidence with the replay adapter.
For each candidate, also record a random-stage proposal from the same donor
episode. `candidate_evidence` scores both against the same baseline and
observed result. `fit_gain_head` fits the two gain predictors from those
records. Keep the feature layout consistent across fit and replay, and reserve
targets for fitting and scoring.

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

Record candidates from previously completed episodes eligible for each memory
policy under comparison. Distances rank phase matches, and the gain heads
predict transfer to an independent query. The fit weights episodes equally
even when they contain different numbers of decisions.

For a mode to be eligible, recent completed trials must support improvement
over empty PIM and the matched random-stage control. The observed fraction of
harmful decisions must also stay within the configured limit. After scoring
the completed trial, the rolling bank can insert it and evict its oldest
record. The runner compares a fixed bank with rolling banks of one, two and
four completed episodes. Each prediction channel uses up to one candidate
per decision regardless of bank size.

## Outputs and validation

| Artifact | Contents |
| --- | --- |
| `cte.pt` and `cte_features/` | Selected CTE and frozen per-episode features |
| Configured policy checkpoint | Cosmos action-policy weights |
| `selector.json` | Action and effect gain predictors |
| `memory_replay.json` | Per-episode modes and gains, matched random controls, paired intervals, coverage and harm |

Check gains by held-out condition, episode-level uncertainty, coverage, and harm among activated trials. Keep the holdout split separate when selecting training settings. The offline protocol measures prediction error against recorded demonstrations. Evaluating the effects of the robot's own executed policy requires a separate closed-loop experiment.
