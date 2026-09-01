# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference config that loads trained PIM weights instead of warm-start skipping them."""

from __future__ import annotations

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robocasa365_atomic5_zeva_pim import (
    action_policy_robocasa365_atomic5_zeva_pim,
)

cs = ConfigStore.instance()

action_policy_robocasa365_atomic5_zeva_pim_inference = copy.deepcopy(
    action_policy_robocasa365_atomic5_zeva_pim
)
action_policy_robocasa365_atomic5_zeva_pim_inference["job"]["name"] = (
    "action_policy_robocasa365_atomic5_zeva_pim_inference"
)
# The training recipe skips these keys only because its input checkpoint is the
# old Stage-2 model.  A trained PIM DCP contains them and inference must load
# them.  Keep the established net_ema exclusion used by the Stage-2 release.
action_policy_robocasa365_atomic5_zeva_pim_inference["checkpoint"][
    "keys_to_skip_loading"
] = ["net_ema."]

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_robocasa365_atomic5_zeva_pim_inference",
    node=action_policy_robocasa365_atomic5_zeva_pim_inference,
)
