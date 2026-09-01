# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION.
# SPDX-License-Identifier: OpenMDW-1.1

"""Checkpoint loading utilities for Zeva modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_cte_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize serialized CTE parameter names without changing tensors."""

    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key.replace("behavior_token", "interaction_state_token")
        new_key = new_key.replace(".behavior.", ".interaction_state.")
        if new_key in normalized:
            raise ValueError(f"Duplicate CTE checkpoint key after normalization: {new_key}")
        normalized[new_key] = value
    return normalized
