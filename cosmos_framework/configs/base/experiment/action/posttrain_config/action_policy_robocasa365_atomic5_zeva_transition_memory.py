# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Experimental transition-memory adaptation for the RoboCasa Atomic-5 Zeva policy.

The CTE path and frozen diffusion policy are loaded from the verified
Atomic-5 Stage-2 checkpoint and frozen.  The separately trained
history-only ``TransitionMemoryEncoder`` is consumed through the cached
``behavior_online_context`` token; this recipe trains the zero-init Cosmos
transition-memory projector so that the new context affects only the action branch.
"""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_atomic5_zeva import (
    action_policy_robocasa365_atomic5_zeva,
)

cs = ConfigStore.instance()

action_policy_robocasa365_atomic5_zeva_transition_memory = copy.deepcopy(
    action_policy_robocasa365_atomic5_zeva
)
action_policy_robocasa365_atomic5_zeva_transition_memory["job"].update(
    project="zeva",
    group="transition_memory_robocasa_atomic5",
    name="action_policy_robocasa365_atomic5_zeva_transition_memory",
)
zeva_policy = action_policy_robocasa365_atomic5_zeva_transition_memory["model"]["config"]["behavior_stage2"]
zeva_policy.update(
    online_memory_enabled=True,
    online_context_dim=256,
    online_prefix_tokens=1,
)

# The already trained policy-injection prior and task-context projector are deliberately not added
# again.  Only the new zero-init Cosmos action-branch projector is optimized;
# the query/key/value/history encoder is trained by the preceding
# history-only recipe and loaded into the context cache.
keys = action_policy_robocasa365_atomic5_zeva_transition_memory["optimizer"]["keys_to_select"]
# Stage-2 checkpoint parameters and the Cosmos backbone must remain frozen in
# this adaptation.  Replacing (rather than subtracting from) the inherited
# selector also prevents the base RoboCasa recipe's action/proprio modules from
# being silently optimized.
keys[:] = ["behavior_online_projector"]
action_policy_robocasa365_atomic5_zeva_transition_memory["optimizer"]["lr_multipliers"].update(
    behavior_online_projector=5.0,
)

train_loader = action_policy_robocasa365_atomic5_zeva_transition_memory.get("dataloader_train")
if train_loader is not None:
    dataset = train_loader["dataloader"]["datasets"]["robocasa365"]["dataset"]
    dataset["behavior_memory_bank"] = "${oc.env:ZEVA_TASK_CONTEXT_BANK}"
    dataset["behavior_phase_cache"] = "${oc.env:ZEVA_CTE_FEATURE_CACHE}"
    dataset["behavior_online_context_cache"] = "${oc.env:ZEVA_TRANSITION_CONTEXT_CACHE}"

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_atomic5_zeva_transition_memory",
    node=action_policy_robocasa365_atomic5_zeva_transition_memory,
)
