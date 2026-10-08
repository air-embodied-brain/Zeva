# Documentation

## Runtime

Python 3.11 and a CUDA 12 environment are recommended.

```bash
git clone --branch feature/zeva_ego https://github.com/air-embodied-brain/Zeva.git
cd Zeva
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install -e causal-conv1d
uv pip install -e mamba
```

The checkpoint loader needs the LeRobot PI0.5 runtime, Torch, Transformers, Safetensors, and the compiled Mamba/causal-convolution extensions. RoboTwin and its simulator adapter must be installed separately.

## Guides

- [PIM checkpoint loading and memory options](../scripts/robotwin/README_CHECKPOINT.md)
- [Multi-attempt evaluation and cumulative rates](../scripts/robotwin/README_EVALUATION.md)
- [Real-robot integration](REAL_ROBOT_DEPLOYMENT.md)
- [Remote inference](remote_inference.md)
- [Docker environment](docker.md)

For ALOHA integration, also install `uv pip install -e third_party/aloha`.

## Independent pipelines

These packages keep their own installation, training, and inference guides:

- [Ego action encoder](../pipelines/ego_action_encoder/README.md)
- [RoboTwin Clean](../pipelines/robotwin_clean/README.md)
