#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

# Short frozen-policy adaptation for the paper-aligned PIM residual.  The base
# checkpoint must be the verified Atomic-5 GRU Stage-2 iter-5k DCP.

set -u

TOML_FILE="examples/toml/sft_config/action_policy_robocasa365_atomic5_zeva_pim.toml"

: "${ROBOCASA365_ROOT:?Set ROBOCASA365_ROOT to the Atomic-5 RoboCasa365 dataset}"
: "${BASE_CHECKPOINT_PATH:?Set BASE_CHECKPOINT_PATH to the verified GRU Stage-2 iter-5k DCP}"
: "${ZEVA_TASK_CONTEXT_BANK:?Set ZEVA_TASK_CONTEXT_BANK to the static task-context bank}"
: "${ZEVA_CTE_FEATURE_CACHE:?Set ZEVA_CTE_FEATURE_CACHE to the CTE feature-cache directory}"
: "${QWEN_VLM_PATH:?Set QWEN_VLM_PATH to Qwen3-VL-8B-Instruct}"
: "${WAN_VAE_PATH:?Set WAN_VAE_PATH to the Wan2.2 VAE checkpoint}"

# Accept the earlier artifact name as a compatibility alias.
ZEVA_PIM_TRAINING_BANK="${ZEVA_PIM_TRAINING_BANK:-}"
: "${ZEVA_PIM_TRAINING_BANK:?Build the PIM training bank first}"

export ROBOCASA365_ROOT BASE_CHECKPOINT_PATH ZEVA_TASK_CONTEXT_BANK ZEVA_CTE_FEATURE_CACHE
export ZEVA_PIM_TRAINING_BANK QWEN_VLM_PATH WAN_VAE_PATH
DATASET_PATH="$ROBOCASA365_ROOT"
EXTRA_DATASET_CHECK='
[[ -f "$ZEVA_TASK_CONTEXT_BANK" ]] || { echo "ERROR: missing $ZEVA_TASK_CONTEXT_BANK" >&2; exit 1; }
[[ -d "$ZEVA_CTE_FEATURE_CACHE" ]] || { echo "ERROR: missing $ZEVA_CTE_FEATURE_CACHE" >&2; exit 1; }
[[ -f "$ZEVA_PIM_TRAINING_BANK" ]] || { echo "ERROR: missing $ZEVA_PIM_TRAINING_BANK" >&2; exit 1; }
[[ -d "$QWEN_VLM_PATH" ]] || { echo "ERROR: missing $QWEN_VLM_PATH" >&2; exit 1; }
'

TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
