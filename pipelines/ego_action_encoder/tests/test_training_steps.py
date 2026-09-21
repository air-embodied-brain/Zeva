import torch
from torch import nn
from zeva_action_encoder.configs import Stage1LossConfig
from zeva_action_encoder.models import Stage1Model
from zeva_action_encoder.models import Stage1ModelConfig
from zeva_action_encoder.models import Stage2Model
from zeva_action_encoder.models import Stage2ModelConfig
from zeva_action_encoder.models.stage2 import FrozenStage1EnvironmentTeacher
from zeva_action_encoder.training import stage1_train_step
from zeva_action_encoder.training import stage2_train_step
from zeva_action_encoder.training.stage2_runner import Stage2LossConfig


class _TinyVision(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-2, -1))
        return torch.cat([pooled, pooled[..., :1]], dim=-1).unsqueeze(1).expand(-1, 2, -1)

    def forward_pair(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        both = self.forward(torch.cat([first, second]))
        return both[: len(first)], both[len(first) :]


def _environment_model() -> Stage1Model:
    return Stage1Model(
        Stage1ModelConfig(
            visual_dim=4,
            action_dim=4,
            model_dim=8,
            latent_dim=4,
            num_environment_tokens=2,
            codebook_size=None,
            encoder_blocks=1,
            decoder_blocks=1,
            num_heads=2,
            code_restart=False,
        )
    )


def test_environment_encoding_step_backward() -> None:
    model = _environment_model()
    batch = {
        "images": torch.rand(2, 2, 3, 14, 14),
        "action_chunk": torch.rand(2, 3, 4),
        "action_mask": torch.ones(2, 3, dtype=torch.bool),
        "action_dimension_mask": torch.ones(2, 3, 4, dtype=torch.bool),
    }
    loss = stage1_train_step(
        model=model,
        visual_encoder=_TinyVision(),
        batch=batch,
        loss_config=Stage1LossConfig(),
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_task_encoding_step_backward() -> None:
    environment = _environment_model()
    config = Stage2ModelConfig(
        visual_dim=4,
        model_dim=8,
        latent_dim=4,
        num_environment_tokens=2,
        num_task_tokens=2,
        encoder_blocks=1,
        decoder_blocks=1,
        num_heads=2,
        environment_mode="joint_query_v1",
    )
    model = Stage2Model(config)
    model.initialize_encoder_from_stage1(environment)
    model.initialize_decoder_from_stage1(environment)
    batch = {
        "images": torch.rand(2, 3, 3, 14, 14),
        "action_chunk": torch.rand(2, 3, 3, 4),
        "action_mask": torch.ones(2, 3, 3, dtype=torch.bool),
        "action_dimension_mask": torch.ones(2, 3, 3, 4, dtype=torch.bool),
    }
    loss = stage2_train_step(
        model=model,
        teacher=FrozenStage1EnvironmentTeacher(environment),
        visual_encoder=_TinyVision(),
        batch=batch,
        loss_config=Stage2LossConfig(
            reconstruction_weight=1.0,
            environment_consistency_weight=1.0,
            additive_weight=0.0,
            reversal_weight=0.0,
            contrastive_weight=0.0,
            ranking_weight=0.0,
        ),
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
