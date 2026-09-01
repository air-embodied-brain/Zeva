# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION.
# SPDX-License-Identifier: OpenMDW-1.1

"""Brief Interaction Trace (BIT) for the current attempt."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class BriefInteractionTrace:
    """Ordered causal effects completed during the current attempt."""

    effects: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        if self.effects.ndim != 3:
            raise ValueError("BIT effects must be [B,L,D]")
        if self.valid.shape != self.effects.shape[:2]:
            raise ValueError("BIT valid mask must be [B,L]")
