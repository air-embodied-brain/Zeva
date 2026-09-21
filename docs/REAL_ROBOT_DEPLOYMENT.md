# Real-robot deployment

ZeVA uses the OpenPI websocket boundary so that the robot computer and GPU host can use separate environments.

## GPU host

For a standard OpenPI checkpoint:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config <training-config-name> \
  --policy.dir <checkpoint-directory> \
  --port 8000
```

For ZeVA RoboTwin/real-robot integration, construct `ZevaCTEEAPPolicy`, `ZevaCrossAttemptPIMPolicy`, or `ZevaEpisodePIMPolicy` in the server process and expose its inference method through the same websocket protocol.

## Robot computer

Install the lightweight client only:

```bash
pip install -e packages/openpi-client
```

Minimal query loop:

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="GPU_HOST", port=8000)

observation = {
    "observation/image": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(exterior_image, 224, 224)
    ),
    "observation/wrist_image": image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, 224, 224)
    ),
    "observation/state": robot_state,
    "prompt": instruction,
}
action_chunk = client.infer(observation)["actions"]
```

## State lifecycle

Call the policy reset endpoint before the first observation of every episode.

- CTE/BIT/EAP and within-episode PIM: episode reset only.
- Cross-attempt PIM: use attempt reset to commit a failed attempt, and episode reset when the scene or instruction changes.

Never reuse PIM across tasks, scenes, or episodes.

## Control contract

- RGB inputs are uint8 images.
- The server applies the checkpoint preprocessor and normalizer.
- The policy returns a full action chunk.
- Execute only the configured H15 prefix before replanning.
- Feed the actually executed H15 chunk back to CTE at the next boundary.
- Stop execution when robot safety checks reject a command.

ALOHA utilities are under `third_party/aloha`; generic DROID/ALOHA transforms remain in `src/openpi/policies`.
