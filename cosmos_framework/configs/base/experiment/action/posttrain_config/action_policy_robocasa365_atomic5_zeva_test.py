from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_atomic5_zeva import (
    action_policy_robocasa365_atomic5_zeva,
)


def test_released_stage2_loads_trained_action_heads() -> None:
    assert action_policy_robocasa365_atomic5_zeva["checkpoint"][
        "keys_to_skip_loading"
    ] == ["net_ema."]


def test_inference_recipe_does_not_construct_training_loaders() -> None:
    assert action_policy_robocasa365_atomic5_zeva["dataloader_train"] is None
    assert action_policy_robocasa365_atomic5_zeva["dataloader_val"] is None
