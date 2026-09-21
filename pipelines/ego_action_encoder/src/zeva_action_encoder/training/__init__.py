from zeva_action_encoder.training.stage2_runner import Stage2LossConfig
from zeva_action_encoder.training.steps import optimize_one_step
from zeva_action_encoder.training.steps import stage1_train_step
from zeva_action_encoder.training.steps import stage2_train_step

environment_encoding_step = stage1_train_step
task_encoding_step = stage2_train_step
TaskEncodingLossConfig = Stage2LossConfig

__all__ = [
    "TaskEncodingLossConfig",
    "environment_encoding_step",
    "optimize_one_step",
    "task_encoding_step",
]
