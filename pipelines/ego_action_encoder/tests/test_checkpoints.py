from pathlib import Path

import pytest
import torch
from zeva_action_encoder.checkpoints import load_checkpoint_payload
from zeva_action_encoder.checkpoints import restore_resume_state
from zeva_action_encoder.checkpoints import save_training_checkpoint
from zeva_action_encoder.models import Stage1Model
from zeva_action_encoder.models import Stage1ModelConfig


def _tiny_model() -> tuple[Stage1Model, Stage1ModelConfig]:
    config = Stage1ModelConfig(
        visual_dim=8,
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
    return Stage1Model(config), config


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model, config = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    path = tmp_path / "environment.pt"
    save_training_checkpoint(
        path,
        stage="environment_encoding",
        step=7,
        model=model,
        model_config=config,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    payload = load_checkpoint_payload(path, expected_stage="environment_encoding")
    assert "training_config" not in payload
    restored, _ = _tiny_model()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1.0)
    assert (
        restore_resume_state(
            payload,
            model=restored,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
        )
        == 7
    )
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_rejects_wrong_stage(tmp_path: Path) -> None:
    model, config = _tiny_model()
    path = tmp_path / "environment.pt"
    save_training_checkpoint(
        path,
        stage="environment_encoding",
        step=0,
        model=model,
        model_config=config,
    )
    with pytest.raises(ValueError, match="expected"):
        load_checkpoint_payload(path, expected_stage="task_encoding")
