# RoboTwin cross-attempt PIM checkpoint

The inference package is available at
[chen123fu/Zeva-CrossAttempt-PIM](https://huggingface.co/chen123fu/Zeva-CrossAttempt-PIM).
Keep all files together:

```text
cross-attempt-pim/
  release.json                         relative paths, file hashes, memory mode
  COMPLETE
  README.md / NOTICE / LICENSE*         loader guide and upstream weight terms
  THIRD_PARTY_NOTICES.md
  handoff/
    checkpoint/pretrained_model/       final policy, processors, tokenizer
    runtime/baseline/contract.json      EEF16 / Joint14 / three-camera contract
    reference/                         frozen action mean and standard deviation
  cte/                                 Mamba config and encoder weights
  eap.safetensors                       EAP and PIM weights
  memory/
    task_memory.json
    task_memory.safetensors             train-only retrieval keys and values
    retrieval.json
    retrieval.safetensors
    goal/                              frozen language embedding and tokenizer
```

## Load

In a RoboTwin adapter with the matching LeRobot PI0.5 and Mamba runtime:

```python
from openpi.zeva.pim_policy import ZevaCrossAttemptPIMPolicy
from huggingface_hub import snapshot_download

root = snapshot_download("chen123fu/Zeva-CrossAttempt-PIM")
policy = ZevaCrossAttemptPIMPolicy.from_release(root)
```
See the [evaluation guide](README_EVALUATION.md).
