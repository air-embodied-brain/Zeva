# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference recipe for the fixed-base RoboCasa Atomic-5 Zeva policy."""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)

cs = ConfigStore.instance()

action_policy_robocasa365_atomic5_zeva = copy.deepcopy(action_policy_droid_nano)
action_policy_robocasa365_atomic5_zeva["job"].update(
    project="zeva",
    group="zeva_robocasa_atomic5",
    name="action_policy_robocasa365_atomic5_zeva",
)
action_policy_robocasa365_atomic5_zeva["model"]["config"]["behavior_stage2"] = dict(
    enabled=True,
    global_dim=256,
    phase_dim=128,
    effect_dim=128,
    effect_history_length=4,
    action_dim=7,
    horizon=32,
    num_anchors=8,
    hidden_dim=256,
    num_heads=4,
    prior_loss_weight=0.01,
    prior_dropout_rate=0.4,
    prior_inference_guidance_scale=0.5,
    global_prefix_tokens=1,
    leading_condition_steps=0,
)
action_policy_robocasa365_atomic5_zeva["model"]["config"]["proprio_condition"] = dict(
    enabled=True,
    input_dim=9,
    prefix_tokens=1,
)
# This is a post-trained Zeva checkpoint, not a fresh DROID initialization.
# Load every trained policy/action tensor; only the unused EMA namespace may be
# absent. Inheriting DROID's action-head skip list silently changes inference.
action_policy_robocasa365_atomic5_zeva["checkpoint"]["keys_to_skip_loading"] = ["net_ema."]
# The server supplies Qwen explicitly; retain the same environment contract as
# the training recipe so checkpoint metadata and manual startup agree.
action_policy_robocasa365_atomic5_zeva["model"]["config"]["vlm_config"]["tokenizer"][
    "pretrained_model_name"
] = "${oc.env:QWEN_VLM_PATH}"

# Serving does not require training dataloaders.
action_policy_robocasa365_atomic5_zeva["dataloader_train"] = None
action_policy_robocasa365_atomic5_zeva["dataloader_val"] = None

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_atomic5_zeva",
    node=action_policy_robocasa365_atomic5_zeva,
)
