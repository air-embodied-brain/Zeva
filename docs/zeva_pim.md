# Persistent Interaction Memory

Zeva uses two interaction memories:

- **Brief Interaction Trace (BIT)** retains recent causal effects inside the
  current attempt.
- **Persistent Interaction Memory (PIM)** retains completed interaction evidence
  across attempts and retrieves it with the current phase representation.

PIM entries contain a phase, a causal effect, an attempt ID, and an observation
count. Similar entries are merged; retrieval returns the top phase-matched
evidence for Causal Prompt construction.

## Model components

| Paper component | Implementation |
| --- | --- |
| Causal Transition Encoder | `CausalTransitionEncoder` |
| Phase representation | `phase_head` / `phase` |
| Causal effect | `effect_pre` / `effect_post` |
| Brief Interaction Trace | `BriefInteractionTrace` |
| Persistent Interaction Memory | `PersistentInteractionMemory` |
| Phase-Conditioned Retrieval | `PhaseConditionedPIMRetrieval` |
| Causal Prompt | `CausalPromptEncoder` |
| Policy Injection | `CausalPromptPolicyAdapter` |

## Start a PIM policy server

Set the release and dependency paths:

```bash
export ZEVA_RELEASE=/absolute/path/to/zeva
export ZEVA_PIM_CHECKPOINT=/absolute/path/to/iter_000000500
export WAN_VAE_PATH=/absolute/path/to/Wan2.2_VAE.pth
export PYTHONPATH=.
```

Start the server with a Zeva-PIM checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m \
  cosmos_framework.scripts.action_policy_server_robocasa365_zeva_pim \
  --checkpoint-path "$ZEVA_PIM_CHECKPOINT" \
  --allow-dcp-checkpoint \
  --experiment action_policy_robocasa365_atomic5_zeva_pim_inference \
  --experiment-overrides model.config.tokenizer.vae_path="$WAN_VAE_PATH" \
  --task-context-bank \
    "$ZEVA_RELEASE/weights/stage1/train_memory_effect_v3.pt" \
  --cte-checkpoint \
    "$ZEVA_RELEASE/weights/stage1/cte_step_000500.pt" \
  --static-task-context-checkpoint \
    "$ZEVA_RELEASE/weights/stage3_iter_000005000/best.pt" \
  --pim-enabled --pim-top-k 4 \
  --pim-success-replay-enabled \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --action-dim 7 --action-chunk-size 32 --conditioning-fps 20 \
  --image-height 256 --image-width 512 \
  --no-use-state --history-length 0 \
  --host 0.0.0.0 --port 8300
```

Use `--pim-shadow` instead of `--pim-enabled` to execute PIM lifecycle and
retrieval without injecting the retrieved evidence into policy actions.

## Fixed-seed cross-attempt evaluation

```bash
python -m cosmos_framework.scripts.eval_zeva_robocasa365_pim \
  --task TurnOnElectricKettle \
  --host 127.0.0.1 --port 8300 \
  --episodes 6 --seed 197 --seed-mode fixed \
  --expect-pim-conditioning on \
  --output-dir results/kettle_seed197 --no-video
```

The fixed seed preserves the task instance while the attempt ID increases.
Success replay is activated only after PIM has observed a successful attempt.

## Reproducible case studies

Each sequence was reproduced twice from empty PIM. `F` denotes failure and `S`
denotes success.

| Task | Seed | Attempts |
| --- | ---: | --- |
| OpenStandMixerHead | 199 | `FSSSS` |
| TurnOnElectricKettle | 197 | `FSSSSS` |
| CloseToasterOvenDoor | 199 | `FFFSSSSSSSS` |
| TurnOnMicrowave | 200 | `FFFFSSSS` |
| CoffeeSetupMug | 204 | `FFFFFFFFFSSSS` |

Validate an exported case-study pair with:

```bash
python experiment_manifests/verify_pim_evolve_case.py \
  --first /path/to/first_run \
  --repeat /path/to/repeat_run \
  --task TurnOnElectricKettle --seed 197
```
