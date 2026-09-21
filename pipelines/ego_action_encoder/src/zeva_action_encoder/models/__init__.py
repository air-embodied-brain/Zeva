"""Encoder model definitions."""

from zeva_action_encoder.models.stage1 import Stage1Model
from zeva_action_encoder.models.stage1 import Stage1ModelConfig
from zeva_action_encoder.models.stage2 import FrozenStage1EnvironmentTeacher
from zeva_action_encoder.models.stage2 import Stage2Model
from zeva_action_encoder.models.stage2 import Stage2ModelConfig
from zeva_action_encoder.models.vision import DinoV2Config
from zeva_action_encoder.models.vision import FrozenDinoV2

EnvironmentEncoder = Stage1Model
EnvironmentEncoderConfig = Stage1ModelConfig
FrozenEnvironmentTeacher = FrozenStage1EnvironmentTeacher
TaskEncoder = Stage2Model
TaskEncoderConfig = Stage2ModelConfig

__all__ = [
    "DinoV2Config",
    "EnvironmentEncoder",
    "EnvironmentEncoderConfig",
    "FrozenDinoV2",
    "FrozenEnvironmentTeacher",
    "TaskEncoder",
    "TaskEncoderConfig",
]
