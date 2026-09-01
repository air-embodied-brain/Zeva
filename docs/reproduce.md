# Reproduce the RoboCasa365 result

## Setup

Download the release from
[chen123fu/zeva](https://huggingface.co/chen123fu/zeva) and set:

```bash
export ZEVA_RELEASE=/absolute/path/to/zeva
export QWEN_VLM_PATH=/absolute/path/to/Qwen3-VL-8B-Instruct
export WAN_VAE_PATH=/absolute/path/to/Wan2.2_VAE.pth
export ROBOCASA365_ROOT=/absolute/path/to/robocasa365/target
export PYTHONPATH=.
```

RoboCasa evaluation also requires `openpi_client`. For headless rendering use:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Start the policy server

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m \
  cosmos_framework.scripts.action_policy_server_robocasa365_zeva \
  --checkpoint-path "$ZEVA_RELEASE/weights/stage2_iter_000005000" \
  --allow-dcp-checkpoint \
  --experiment action_policy_robocasa365_atomic5_zeva \
  --experiment-overrides model.config.tokenizer.vae_path="$WAN_VAE_PATH" \
  --task-context-bank \
    "$ZEVA_RELEASE/weights/stage1/train_memory_effect_v3.pt" \
  --cte-checkpoint \
    "$ZEVA_RELEASE/weights/stage1/cte_step_000500.pt" \
  --static-task-context-checkpoint \
    "$ZEVA_RELEASE/weights/stage3_iter_000005000/best.pt" \
  --static-task-context-top-k 5 \
  --domain-name robocasa-panda-omron \
  --host 0.0.0.0 --port 8300 \
  --num-steps 30 --guidance 3.0 --shift 5.0 \
  --action-dim 7 --action-chunk-size 32 --conditioning-fps 20 \
  --image-height 256 --image-width 512 \
  --format-prompt-as-json \
  --no-use-state --history-length 0 \
  --output-dir /tmp/zeva_server
```

Wait until the server reports the task-context bank, domain ID 22, action
horizon 32, and `ready` status.

## Evaluate

Use the following fixed protocol:

- target split;
- five Atomic-5 tasks;
- seeds 195–244 for every task;
- left-agent and wrist cameras concatenated to 256×512;
- 32 executed actions per query at 20 Hz;
- 30 UniPC steps, guidance 3.0, shift 5.0;
- maximum 300 environment steps;
- no retries.

Run each task with the Zeva evaluator:

```bash
python -m cosmos_framework.scripts.eval_zeva_robocasa365 \
  --task OpenStandMixerHead \
  --host 127.0.0.1 --port 8300 \
  --episodes 50 --seed 195 --open-loop-steps 32 \
  --split target --render-gpu-device-id 0 --no-video \
  --output-dir results/OpenStandMixerHead
```

Repeat the command for:

```text
TurnOnElectricKettle
CloseToasterOvenDoor
TurnOnMicrowave
CoffeeSetupMug
```

## Expected result

| Task | Success |
| --- | ---: |
| OpenStandMixerHead | 48/50 (96%) |
| TurnOnElectricKettle | 45/50 (90%) |
| CloseToasterOvenDoor | 38/50 (76%) |
| TurnOnMicrowave | 47/50 (94%) |
| CoffeeSetupMug | 17/50 (34%) |
| **Total** | **195/250 (78.0%)** |
