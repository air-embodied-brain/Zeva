from dataclasses import dataclass


@dataclass(frozen=True)
class ZevaConfig:
    """Architecture shared by Zeva causal extraction and memory injection."""

    action_dim: int = 16
    action_horizon: int = 50
    model_dim: int = 256
    phase_dim: int = 128
    signal_dim: int = 256
    task_dim: int = 128
    goal_dim: int = 2048
    num_views: int = 1
    task_count: int = 0
    num_mamba_layers: int = 4
    mamba_state_dim: int = 16
    mamba_conv_width: int = 4
    mamba_expand: int = 2
    dropout: float = 0.1
    image_size: int = 224
    vision_pretrained: bool = True
    ema_decay: float = 0.99
    use_effect_stream: bool = True

    brief_memory_size: int = 8
    persistent_memory_size: int = 256
    retrieval_top_k: int = 5
    merge_phase_weight: float = 0.6
    merge_signal_weight: float = 0.4
    merge_threshold: float = 0.85
    use_brief_memory: bool = True
    use_persistent_memory: bool = True
