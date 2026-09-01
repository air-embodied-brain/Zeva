# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Learning objectives for the Causal Transition Encoder (CTE)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class CTELossConfig:
    action_weight: float = 1.0
    vision_weight: float = 1.0
    task_weight: float = 0.2
    phase_weight: float = 0.1
    # Effect-v3: observed outcomes are contrastive and explicitly protected
    # against rank collapse. Action reconstruction remains a weak auxiliary.
    effect_weight: float = 0.25
    effect_contrastive_weight: float = 1.0
    effect_action_weight: float = 0.05
    effect_align_weight: float = 0.1
    effect_variance_weight: float = 1.0
    effect_covariance_weight: float = 0.04
    effect_temperature: float = 0.07
    temperature: float = 0.1


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    weight = mask.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def _task_identity_clustering_loss(
    retrieval: Tensor, valid: Tensor, semantic_ids: Tensor, temperature: float
) -> Tensor:
    """Multi-positive supervised contrastive loss; samples without a positive are ignored."""
    z = F.normalize(_masked_mean(retrieval, valid), dim=-1)
    logits = z @ z.T / temperature
    logits.fill_diagonal_(-torch.inf)
    positives = semantic_ids[:, None].eq(semantic_ids[None, :])
    positives.fill_diagonal_(False)
    usable = positives.any(dim=1)
    if not bool(usable.any()):
        return z.new_zeros(())
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1).clamp_min(1))[usable].mean()


def _phase_progression_loss(phase: Tensor, valid: Tensor, margin: float = 0.01) -> Tensor:
    """Encourage consecutive phase tokens to advance along one trajectory direction."""
    losses: list[Tensor] = []
    for sequence, mask in zip(phase, valid, strict=True):
        z = sequence[mask]
        if z.shape[0] < 3:
            continue
        direction = F.normalize(z[-1] - z[0], dim=-1)
        increments = (z[1:] - z[:-1]) @ direction
        losses.append(F.relu(margin - increments).mean())
    return torch.stack(losses).mean() if losses else phase.new_zeros(())


def summarize_effect_window(actions: Tensor) -> Tensor:
    """Summarize an executed low-rate effect window ``[B,W,K,4,A]``.

    Keeping first, net displacement, mean and standard deviation makes this a
    light action regularizer, not a loss that can be solved by copying the
    current four-control input into the effect code.
    """
    if actions.ndim != 5:
        raise ValueError("Effect actions must have shape [B,W,K,4,A].")
    sequence = actions.flatten(-3, -2)
    return torch.cat(
        (
            sequence[..., 0, :],
            sequence[..., -1, :] - sequence[..., 0, :],
            sequence.mean(dim=-2),
            sequence.std(dim=-2, correction=0),
        ),
        dim=-1,
    )


def _effect_nce(query: Tensor, target: Tensor, mask: Tensor, temperature: float) -> Tensor:
    query, target = query[mask], target[mask]
    if query.shape[0] < 2:
        return query.new_zeros(())
    logits = F.normalize(query, dim=-1) @ F.normalize(target, dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))


def _vicreg(code: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    code = code[mask]
    if code.shape[0] < 2:
        zero = code.new_zeros(())
        return zero, zero
    centered = code - code.mean(dim=0, keepdim=True)
    std = (centered.var(dim=0, correction=0) + 1e-4).sqrt()
    variance = F.relu(1.0 - std).mean()
    covariance = centered.T @ centered / max(code.shape[0] - 1, 1)
    covariance.fill_diagonal_(0)
    covariance_loss = covariance.square().sum() / code.shape[-1]
    return variance, covariance_loss


def causal_transition_encoder_loss(
    outputs: dict[str, Tensor],
    transition_actions: Tensor,
    valid_mask: Tensor,
    semantic_ids: Tensor,
    cfg: CTELossConfig | None = None,
) -> dict[str, Tensor]:
    """CTE objective: causal effect, task identity, and phase progression."""
    cfg = cfg or CTELossConfig()
    transition_complete = outputs["transition_complete"]
    if transition_complete.shape != transition_actions.shape[:2]:
        raise ValueError("Effect outputs and transition-action cache are not temporally aligned.")
    action_error = F.smooth_l1_loss(
        outputs["next_action"], transition_actions, reduction="none"
    ).mean(dim=(-1, -2))
    action_loss = (action_error * transition_complete).sum() / transition_complete.sum().clamp_min(1)
    target_vision = outputs["target_visual"]
    vision_error = F.mse_loss(
        outputs["next_vision"][:, :-1], target_vision[:, 1:], reduction="none"
    ).mean(dim=-1)
    vision_loss = (vision_error * transition_complete).sum() / transition_complete.sum().clamp_min(1)
    task_loss = _task_identity_clustering_loss(outputs["retrieval"], valid_mask, semantic_ids, cfg.temperature)
    phase_loss = _phase_progression_loss(outputs["phase"], valid_mask)
    effect_mask = outputs["effect_complete"]
    effect_target = outputs["effect_delta_target"]
    effect_nce_pre = _effect_nce(outputs["effect_outcome_pre"], effect_target, effect_mask, cfg.effect_temperature)
    effect_nce_post = _effect_nce(outputs["effect_outcome_post"], effect_target, effect_mask, cfg.effect_temperature)
    effect_contrastive_loss = 0.5 * (effect_nce_pre + effect_nce_post)
    # Diagnostic only: absolute deltas have a strong zero baseline, so this
    # MSE is intentionally not optimized directly.
    effect_visual_error = F.mse_loss(outputs["effect_outcome_pre"], effect_target, reduction="none").mean(dim=-1)
    effect_visual_loss = (effect_visual_error * effect_mask).sum() / effect_mask.sum().clamp_min(1)
    action_target = summarize_effect_window(outputs["effect_actions"])
    effect_action_error = F.smooth_l1_loss(
        outputs["effect_action"], action_target, reduction="none"
    ).mean(dim=-1)
    effect_action_loss = (effect_action_error * effect_mask).sum() / effect_mask.sum().clamp_min(1)
    effect_align_error = 1.0 - (outputs["effect_pre"] * outputs["effect_post"]).sum(dim=-1)
    effect_align_loss = (effect_align_error * effect_mask).sum() / effect_mask.sum().clamp_min(1)
    effect_variance_loss, effect_covariance_loss = _vicreg(outputs["effect_post_raw"], effect_mask)
    effect_loss = (
        cfg.effect_contrastive_weight * effect_contrastive_loss
        + cfg.effect_action_weight * effect_action_loss
        + cfg.effect_align_weight * effect_align_loss
        + cfg.effect_variance_weight * effect_variance_loss
        + cfg.effect_covariance_weight * effect_covariance_loss
    )
    total = (
        cfg.action_weight * action_loss
        + cfg.vision_weight * vision_loss
        + cfg.task_weight * task_loss
        + cfg.phase_weight * phase_loss
        + cfg.effect_weight * effect_loss
    )
    return {
        "total": total,
        "action": action_loss,
        "vision": vision_loss,
        "loss_task": task_loss,
        "loss_phase": phase_loss,
        "loss_effect": effect_loss,
        "effect_contrastive": effect_contrastive_loss,
        "effect_visual": effect_visual_loss,
        "effect_action": effect_action_loss,
        "effect_align": effect_align_loss,
        "effect_variance": effect_variance_loss,
        "effect_covariance": effect_covariance_loss,
    }
