"""Core Stage 2 objectives that do not depend on a concrete data loader."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from zeva_action_encoder.models.stage2 import Stage2TaskRelations


@dataclass(frozen=True)
class Stage2AlgebraLoss:
    total: Tensor
    additive: Tensor
    reversal: Tensor


def compute_stage2_algebra_loss(
    relations: Stage2TaskRelations,
    *,
    additive_weight: float = 1.0,
    reversal_weight: float = 1.0,
    normalization_epsilon: float = 1e-12,
) -> Stage2AlgebraLoss:
    """Enforce scale-relative additive and reversal residuals.

    All four terms must originate from one aligned triplet batch. Keeping the
    token axes intact makes the relation slot-wise and prevents accidental
    broadcasting between different task-query layouts. Dividing each squared
    residual by the energy of its operands removes the incentive to reduce the
    loss by shrinking every task token.
    """

    if additive_weight < 0.0 or reversal_weight < 0.0:
        raise ValueError("algebra loss weights must be non-negative")
    if normalization_epsilon <= 0.0:
        raise ValueError("normalization_epsilon must be positive")
    shapes = {tuple(value.shape) for value in (relations.ab, relations.bc, relations.ac, relations.ba)}
    if len(shapes) != 1:
        raise ValueError(f"all task relations must have one shape, got {sorted(shapes)}")
    if relations.ab.ndim != 3:
        raise ValueError("task relations must have shape [B, N_task, D]")
    if not all(
        value.is_floating_point() and torch.isfinite(value).all()
        for value in (relations.ab, relations.bc, relations.ac, relations.ba)
    ):
        raise ValueError("task relations must be finite floating tensors")

    additive_residual = relations.ab.float() + relations.bc.float() - relations.ac.float()
    reversal_residual = relations.ab.float() + relations.ba.float()
    additive_energy = (
        relations.ab.float().square()
        + relations.bc.float().square()
        + relations.ac.float().square()
    ).mean()
    reversal_energy = (
        relations.ab.float().square()
        + relations.ba.float().square()
    ).mean()
    additive = additive_residual.square().mean() / (additive_energy + normalization_epsilon)
    reversal = reversal_residual.square().mean() / (reversal_energy + normalization_epsilon)
    total = additive_weight * additive + reversal_weight * reversal
    return Stage2AlgebraLoss(total=total, additive=additive, reversal=reversal)
