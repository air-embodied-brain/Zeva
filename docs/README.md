# Documentation

This directory contains the shared Zeva-Ego environment, reproduction, and deployment documentation. Package-specific installation and commands remain in each package README.

## Environment setup

Python 3.11 and a CUDA 12 environment are recommended.

```bash
git clone https://github.com/dsdfasfaac/zeva-vla.git
cd zeva-vla
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install -e third_party/aloha
```

Install the local Mamba and causal-convolution packages when they are not already available:

```bash
uv pip install -e causal-conv1d
uv pip install -e mamba
```

## Reproduction guides

- [RoboTwin ICCL reproduction](ROBOTWIN_REPRODUCTION.md)
- [Real-robot deployment](REAL_ROBOT_DEPLOYMENT.md)
- [Normalization statistics](norm_stats.md)
- [Remote inference](remote_inference.md)
- [Docker environment](docker.md)

## Independent pipelines

- [Ego action encoder](../pipelines/ego_action_encoder/README.md): encoder installation, dataset hook, training, resume, and RGB-pair inference.
- [RoboTwin Clean](../pipelines/robotwin_clean/README.md): clean post-training, normalization, and randomized evaluation.
