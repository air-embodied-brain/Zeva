"""Action-neighbour contrastive objective for Stage 2 task latents."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from zeva_action_encoder.data.stage2 import ActionDescriptor, ActionDescriptorConfig


@dataclass(frozen=True)
class PairwiseActionDistance:
    mean_squared: Tensor
    valid_pairs: Tensor
    shared_dimensions: Tensor


@dataclass(frozen=True)
class ContrastivePairMasks:
    positive: Tensor
    negative: Tensor


@dataclass(frozen=True)
class Stage2ContrastiveLoss:
    loss: Tensor
    active_anchors: Tensor
    positive_pairs: Tensor
    negative_pairs: Tensor


@dataclass(frozen=True)
class Stage2RelativeRankingLoss:
    """Cross-source ordering supervision for anchors without a hard positive."""

    loss: Tensor
    active_anchors: Tensor
    mean_nearest_action_distance: Tensor
    mean_near_latent_distance: Tensor
    mean_hard_negative_latent_distance: Tensor


@dataclass(frozen=True)
class Stage2ActionNeighborhoodKLLoss:
    """Match equal-duration action and task-token neighbourhood distributions."""

    loss: Tensor
    active_anchors: Tensor
    target_effective_neighbours: Tensor
    predicted_effective_neighbours: Tensor
    target_top1_probability: Tensor
    predicted_top1_probability: Tensor


def pairwise_masked_action_distance(
    descriptor: ActionDescriptor,
    *,
    config: ActionDescriptorConfig,
    min_shared_dimensions: int,
    group_mse_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PairwiseActionDistance:
    """Compute equal-weight translation/rotation/gripper MSE in one batch."""

    result = cross_masked_action_distance(
        descriptor,
        descriptor,
        config=config,
        min_shared_dimensions=min_shared_dimensions,
        group_mse_scales=group_mse_scales,
    )
    valid_pairs = result.valid_pairs.clone()
    identity = torch.eye(
        descriptor.values.shape[0],
        dtype=torch.bool,
        device=descriptor.values.device,
    )
    valid_pairs = valid_pairs & ~identity
    return PairwiseActionDistance(
        mean_squared=result.mean_squared.masked_fill(~valid_pairs, torch.inf),
        valid_pairs=valid_pairs,
        shared_dimensions=result.shared_dimensions,
    )


def cross_masked_action_distance(
    anchors: ActionDescriptor,
    candidates: ActionDescriptor,
    *,
    config: ActionDescriptorConfig,
    min_shared_dimensions: int,
    group_mse_scales: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> PairwiseActionDistance:
    """Rectangular equal-weight group MSE from local to global actions.

    Translation (metres), rotation (radians), and relative gripper aperture
    are each averaged internally, then the available group distances are
    divided by frozen group MSE scales, then the available dimensionless
    group distances are averaged with equal weight. The matrix identity avoids a ``[B, B, D]`` difference tensor.
    """

    if anchors.values.shape[1] != candidates.values.shape[1]:
        raise ValueError("anchor and candidate descriptor dimensions differ")
    if anchors.values.shape[1] != config.descriptor_dim:
        raise ValueError("descriptor values differ from ActionDescriptorConfig")
    if min_shared_dimensions <= 0:
        raise ValueError("min_shared_dimensions must be positive")
    if len(group_mse_scales) != len(config.group_slices) or any(
        not math.isfinite(scale) or scale <= 0.0 for scale in group_mse_scales
    ):
        raise ValueError("action-distance group MSE scales must contain three positive finite values")
    group_distances: list[Tensor] = []
    group_validity: list[Tensor] = []
    group_shared: list[Tensor] = []
    for slices, group_scale in zip(config.group_slices, group_mse_scales, strict=True):
        left = torch.cat([anchors.values[:, part] for part in slices], dim=1).float()
        right = torch.cat([candidates.values[:, part] for part in slices], dim=1).to(
            device=left.device,
            dtype=left.dtype,
        )
        left_mask = torch.cat([anchors.dimension_mask[:, part] for part in slices], dim=1)
        right_mask = torch.cat([candidates.dimension_mask[:, part] for part in slices], dim=1).to(left.device)
        left_mask_float = left_mask.to(dtype=left.dtype)
        right_mask_float = right_mask.to(dtype=left.dtype)
        shared = left_mask_float @ right_mask_float.transpose(0, 1)
        left_masked = left * left_mask_float
        right_masked = right * right_mask_float
        numerator = (
            (left.square() * left_mask_float) @ right_mask_float.transpose(0, 1)
            + left_mask_float @ (right.square() * right_mask_float).transpose(0, 1)
            - 2.0 * (left_masked @ right_masked.transpose(0, 1))
        ).clamp_min(0.0)
        # Normalize raw group MSE by a frozen mixed-train median before the equal group mean.
        group_distances.append(numerator / shared.clamp_min(1.0) / group_scale)
        group_validity.append(shared > 0.0)
        group_shared.append(shared)

    valid_groups = torch.stack(group_validity, dim=0)
    distance = (torch.stack(group_distances, dim=0) * valid_groups).sum(dim=0) / valid_groups.sum(
        dim=0
    ).clamp_min(1)
    shared = torch.stack(group_shared, dim=0).sum(dim=0)
    valid = (shared >= float(min_shared_dimensions)) & valid_groups.any(dim=0)
    return PairwiseActionDistance(
        mean_squared=distance.masked_fill(~valid, torch.inf),
        valid_pairs=valid,
        shared_dimensions=shared.to(torch.int64),
    )


def mask_action_distance_by_effective_duration(
    distance: PairwiseActionDistance,
    anchor_duration_seconds: Tensor,
    candidate_duration_seconds: Tensor,
    *,
    tolerance_seconds: float,
) -> PairwiseActionDistance:
    """Make unequal post-slowdown durations invisible to every matching loss."""

    anchors, candidates = distance.mean_squared.shape
    if anchor_duration_seconds.shape != (anchors,) or candidate_duration_seconds.shape != (candidates,):
        raise ValueError("duration tensors must align with the action-distance matrix")
    if tolerance_seconds < 0.0:
        raise ValueError("duration tolerance must be non-negative")
    if not anchor_duration_seconds.is_floating_point() or not candidate_duration_seconds.is_floating_point():
        raise ValueError("effective durations must be floating point")
    if not torch.isfinite(anchor_duration_seconds).all() or not torch.isfinite(candidate_duration_seconds).all():
        raise ValueError("effective durations must be finite")
    if torch.any(anchor_duration_seconds <= 0.0) or torch.any(candidate_duration_seconds <= 0.0):
        raise ValueError("effective durations must be positive")
    candidate_duration_seconds = candidate_duration_seconds.to(anchor_duration_seconds.device)
    same_duration = (
        anchor_duration_seconds[:, None] - candidate_duration_seconds[None, :]
    ).abs() <= tolerance_seconds
    valid = distance.valid_pairs & same_duration
    return PairwiseActionDistance(
        mean_squared=distance.mean_squared.masked_fill(~valid, torch.inf),
        valid_pairs=valid,
        shared_dimensions=distance.shared_dimensions,
    )


def select_contrastive_pairs(
    distance: PairwiseActionDistance,
    source_ids: Tensor,
    *,
    positive_max_distance: float,
    negative_min_distance: float | None = None,
) -> ContrastivePairMasks:
    """Select cross-source positives and negatives from action distance.

    With no explicit negative threshold, one threshold partitions every valid
    pair: distances at or below ``positive_max_distance`` are positive and
    larger distances are negative. Passing ``negative_min_distance`` retains
    the older conservative-gap behavior for reproducibility.
    """

    batch_size = distance.mean_squared.shape[0]
    if distance.mean_squared.shape != (batch_size, batch_size):
        raise ValueError("pairwise action distance must be square")
    if source_ids.shape != (batch_size,):
        raise ValueError("source_ids must have shape [B]")
    if positive_max_distance < 0.0:
        raise ValueError("positive_max_distance must be non-negative")
    if negative_min_distance is not None and negative_min_distance <= positive_max_distance:
        raise ValueError("negative_min_distance must exceed positive_max_distance")
    return select_cross_contrastive_pairs(
        distance,
        source_ids,
        source_ids,
        positive_max_distance=positive_max_distance,
        negative_min_distance=negative_min_distance,
    )


def select_cross_contrastive_pairs(
    distance: PairwiseActionDistance,
    anchor_source_ids: Tensor,
    candidate_source_ids: Tensor,
    *,
    positive_max_distance: float,
    negative_min_distance: float | None = None,
) -> ContrastivePairMasks:
    """Select cross-source positives and valid negatives."""

    anchors, candidates = distance.mean_squared.shape
    if anchor_source_ids.shape != (anchors,) or candidate_source_ids.shape != (candidates,):
        raise ValueError("source IDs must align with rectangular action distance")
    if positive_max_distance < 0.0:
        raise ValueError("positive_max_distance must be non-negative")
    if negative_min_distance is not None and negative_min_distance <= positive_max_distance:
        raise ValueError("negative_min_distance must exceed positive_max_distance")
    cross_source = anchor_source_ids[:, None] != candidate_source_ids[None, :]
    positive = distance.valid_pairs & cross_source & (distance.mean_squared <= positive_max_distance)
    if negative_min_distance is None:
        negative = distance.valid_pairs & (distance.mean_squared > positive_max_distance)
    else:
        negative = distance.valid_pairs & (distance.mean_squared >= negative_min_distance)
    return ContrastivePairMasks(positive=positive, negative=negative)


def compute_stage2_contrastive_loss(
    task_tokens: Tensor,
    pairs: ContrastivePairMasks,
    *,
    temperature: float,
) -> Stage2ContrastiveLoss:
    """Apply symmetric multi-positive InfoNCE directly to task-token geometry."""

    return compute_stage2_cross_contrastive_loss(
        task_tokens,
        task_tokens,
        pairs,
        temperature=temperature,
    )


def compute_stage2_cross_contrastive_loss(
    anchor_task_tokens: Tensor,
    candidate_task_tokens: Tensor,
    pairs: ContrastivePairMasks,
    *,
    temperature: float,
    latent_distance: Tensor | None = None,
) -> Stage2ContrastiveLoss:
    """Apply multi-positive InfoNCE from local anchors to global candidates."""

    if anchor_task_tokens.ndim != 3 or not anchor_task_tokens.is_floating_point():
        raise ValueError("task_tokens must be floating [B, N, D]")
    if candidate_task_tokens.ndim != 3 or not candidate_task_tokens.is_floating_point():
        raise ValueError("candidate task_tokens must be floating [B, N, D]")
    if anchor_task_tokens.shape[1:] != candidate_task_tokens.shape[1:]:
        raise ValueError("anchor and candidate task-token layouts differ")
    if not torch.isfinite(anchor_task_tokens).all() or not torch.isfinite(candidate_task_tokens).all():
        raise ValueError("task_tokens must be finite")
    expected = (anchor_task_tokens.shape[0], candidate_task_tokens.shape[0])
    if pairs.positive.shape != expected or pairs.negative.shape != expected:
        raise ValueError("positive and negative masks must have shape [B, B]")
    if pairs.positive.dtype != torch.bool or pairs.negative.dtype != torch.bool:
        raise ValueError("positive and negative masks must be boolean")
    if torch.any(pairs.positive & pairs.negative):
        raise ValueError("a pair cannot be both positive and negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    distance = _latent_distance(
        anchor_task_tokens,
        candidate_task_tokens,
        latent_distance,
    )
    logits = -distance / temperature
    active = pairs.positive.any(dim=1) & pairs.negative.any(dim=1)
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        loss = anchor_task_tokens.float().sum() * 0.0
    else:
        active_logits = logits[active_indices]
        positive = pairs.positive[active_indices]
        candidates = positive | pairs.negative[active_indices]
        positive_logsumexp = torch.logsumexp(
            active_logits.masked_fill(~positive, -torch.inf),
            dim=1,
        )
        candidate_logsumexp = torch.logsumexp(
            active_logits.masked_fill(~candidates, -torch.inf),
            dim=1,
        )
        loss = (candidate_logsumexp - positive_logsumexp).mean()
    return Stage2ContrastiveLoss(
        loss=loss,
        active_anchors=active.sum(),
        positive_pairs=pairs.positive.sum(),
        negative_pairs=pairs.negative.sum(),
    )


def compute_stage2_relative_ranking_loss(
    anchor_task_tokens: Tensor,
    candidate_task_tokens: Tensor,
    action_distance: PairwiseActionDistance,
    anchor_source_ids: Tensor,
    candidate_source_ids: Tensor,
    *,
    positive_max_distance: float,
    negative_min_distance: float | None,
    margin: float,
    latent_distance: Tensor | None = None,
) -> Stage2RelativeRankingLoss:
    """Order cross-source candidates when no reliable hard positive exists.

    The action-nearest candidate is not declared equivalent to the anchor. The
    loss only requires it to be closer in task-latent mean-squared Euclidean distance than the
    currently hardest candidate whose physical-action distance is safely far.
    Anchors already covered by the stricter hard-positive InfoNCE are excluded.
    """

    if anchor_task_tokens.ndim != 3 or candidate_task_tokens.ndim != 3:
        raise ValueError("task tokens must have shape [B, N, D]")
    if anchor_task_tokens.shape[1:] != candidate_task_tokens.shape[1:]:
        raise ValueError("anchor and candidate task-token layouts differ")
    if not anchor_task_tokens.is_floating_point() or not candidate_task_tokens.is_floating_point():
        raise ValueError("task tokens must be floating")
    if not torch.isfinite(anchor_task_tokens).all() or not torch.isfinite(candidate_task_tokens).all():
        raise ValueError("task tokens must be finite")
    anchor_count = anchor_task_tokens.shape[0]
    candidate_count = candidate_task_tokens.shape[0]
    expected = (anchor_count, candidate_count)
    if action_distance.mean_squared.shape != expected or action_distance.valid_pairs.shape != expected:
        raise ValueError("action distance must have shape [anchors, candidates]")
    if anchor_source_ids.shape != (anchor_count,) or candidate_source_ids.shape != (candidate_count,):
        raise ValueError("source IDs must align with anchors and candidates")
    if positive_max_distance < 0.0 or (
        negative_min_distance is not None and negative_min_distance <= positive_max_distance
    ):
        raise ValueError("ranking thresholds must define a positive-to-negative gap")
    if margin <= 0.0:
        raise ValueError("ranking margin must be positive")

    cross_source = anchor_source_ids[:, None] != candidate_source_ids[None, :]
    comparable = action_distance.valid_pairs & cross_source
    hard_positive = comparable & (action_distance.mean_squared <= positive_max_distance)
    if negative_min_distance is None:
        far = comparable & (action_distance.mean_squared > positive_max_distance)
    else:
        far = comparable & (action_distance.mean_squared >= negative_min_distance)
    uncovered = ~hard_positive.any(dim=1)
    has_comparable = comparable.any(dim=1)
    has_far = far.any(dim=1)
    active = uncovered & has_comparable & has_far

    action_for_nearest = action_distance.mean_squared.masked_fill(~comparable, torch.inf)
    nearest_action_distance, nearest_index = action_for_nearest.min(dim=1)
    pairwise_latent_distance = _latent_distance(
        anchor_task_tokens,
        candidate_task_tokens,
        latent_distance,
    )
    row_index = torch.arange(anchor_count, device=pairwise_latent_distance.device)
    near_latent_distance = pairwise_latent_distance[row_index, nearest_index]
    # Exclude the selected near candidate even if its absolute action distance
    # is already large; the relation remains an ordering statement, not an
    # assertion that the nearest available cross-source action is equivalent.
    far = far.clone()
    far[row_index, nearest_index] = False
    has_far_after_exclusion = far.any(dim=1)
    active = active & has_far_after_exclusion
    hard_negative_latent_distance = pairwise_latent_distance.masked_fill(~far, torch.inf).min(dim=1).values

    if bool(active.any()):
        loss = F.relu(margin + near_latent_distance[active] - hard_negative_latent_distance[active]).mean()
        mean_action = nearest_action_distance[active].mean()
        mean_near = near_latent_distance[active].mean()
        mean_far = hard_negative_latent_distance[active].mean()
    else:
        zero = anchor_task_tokens.float().sum() * 0.0
        loss = zero
        mean_action = zero.detach()
        mean_near = zero.detach()
        mean_far = zero.detach()
    return Stage2RelativeRankingLoss(
        loss=loss,
        active_anchors=active.sum(),
        mean_nearest_action_distance=mean_action,
        mean_near_latent_distance=mean_near,
        mean_hard_negative_latent_distance=mean_far,
    )



def compute_stage2_listwise_ranking_loss(
    anchor_task_tokens: Tensor,
    candidate_task_tokens: Tensor,
    action_distance: PairwiseActionDistance,
    *,
    normalization_epsilon: float,
    latent_distance: Tensor | None = None,
) -> Stage2RelativeRankingLoss:
    """Align latent-distance ordering with every valid physical-action rank.

    ``action_distance.valid_pairs`` is expected to contain only equal-duration,
    non-self pairs. No source restriction or action-distance threshold is
    applied. Ground-truth action distances are converted to per-anchor ordinal
    ranks, then correlated with centered latent distances. This is listwise,
    uses every valid candidate, and is invariant to a positive global rescale
    of latent distances.
    """

    if anchor_task_tokens.ndim != 3 or candidate_task_tokens.ndim != 3:
        raise ValueError("task tokens must have shape [B, N, D]")
    if anchor_task_tokens.shape[1:] != candidate_task_tokens.shape[1:]:
        raise ValueError("anchor and candidate task-token layouts differ")
    if normalization_epsilon <= 0.0:
        raise ValueError("normalization_epsilon must be positive")
    anchors, candidates = action_distance.mean_squared.shape
    if anchors != len(anchor_task_tokens) or candidates != len(candidate_task_tokens):
        raise ValueError("action distance must align with anchor and candidate tokens")
    valid = action_distance.valid_pairs
    counts = valid.sum(dim=1)
    active = counts >= 2

    action_for_sort = action_distance.mean_squared.masked_fill(~valid, torch.inf)
    order = torch.argsort(action_for_sort, dim=1, stable=True)
    ordinal_rank = torch.empty_like(action_for_sort)
    ordinal_rank.scatter_(
        1,
        order,
        torch.arange(candidates, device=order.device, dtype=action_for_sort.dtype)
        .unsqueeze(0)
        .expand(anchors, -1),
    )
    latent = _latent_distance(anchor_task_tokens, candidate_task_tokens, latent_distance)
    valid_float = valid.to(dtype=latent.dtype)
    denominator = counts.clamp_min(1).to(dtype=latent.dtype)
    rank_mean = (ordinal_rank * valid_float).sum(dim=1) / denominator
    latent_mean = (latent * valid_float).sum(dim=1) / denominator
    centered_rank = (ordinal_rank - rank_mean[:, None]) * valid_float
    centered_latent = (latent - latent_mean[:, None]) * valid_float
    numerator = (centered_rank * centered_latent).sum(dim=1)
    norm = (
        centered_rank.square().sum(dim=1)
        * centered_latent.square().sum(dim=1)
    ).clamp_min(0.0).sqrt()
    correlation = numerator / (norm + normalization_epsilon)

    nearest_index = action_for_sort.argmin(dim=1)
    farthest_index = action_for_sort.masked_fill(~valid, -torch.inf).argmax(dim=1)
    row = torch.arange(anchors, device=latent.device)
    nearest_action = action_for_sort[row, nearest_index]
    nearest_latent = latent[row, nearest_index]
    farthest_latent = latent[row, farthest_index]
    if bool(active.any()):
        loss = (1.0 - correlation[active]).mean()
        mean_action = nearest_action[active].mean()
        mean_near = nearest_latent[active].mean()
        mean_far = farthest_latent[active].mean()
    else:
        zero = anchor_task_tokens.float().sum() * 0.0
        loss = zero
        mean_action = zero.detach()
        mean_near = zero.detach()
        mean_far = zero.detach()
    return Stage2RelativeRankingLoss(
        loss=loss,
        active_anchors=active.sum(),
        mean_nearest_action_distance=mean_action,
        mean_near_latent_distance=mean_near,
        mean_hard_negative_latent_distance=mean_far,
    )


def compute_stage2_local_triplet_ranking_loss(
    anchor_task_tokens: Tensor,
    candidate_task_tokens: Tensor,
    action_distance: PairwiseActionDistance,
    *,
    latent_distance: Tensor | None = None,
) -> Stage2RelativeRankingLoss:
    """Rank two randomly sampled equal-duration candidates per anchor.

    The caller supplies the valid-pair mask, including duration matching and
    self-pair exclusion. Two valid candidates are sampled without replacement
    and ordered only by their physical-action distance. No source restriction,
    action-distance threshold, or listwise constraint is applied.
    """

    if anchor_task_tokens.ndim != 3 or candidate_task_tokens.ndim != 3:
        raise ValueError("task tokens must have shape [B, N, D]")
    if anchor_task_tokens.shape[1:] != candidate_task_tokens.shape[1:]:
        raise ValueError("anchor and candidate task-token layouts differ")
    anchors, candidates = action_distance.mean_squared.shape
    if anchors != len(anchor_task_tokens) or candidates != len(candidate_task_tokens):
        raise ValueError("action distance must align with anchor and candidate tokens")

    valid = action_distance.valid_pairs
    has_two_candidates = valid.sum(dim=1) >= 2
    random_priority = torch.rand(
        valid.shape,
        dtype=torch.float32,
        device=valid.device,
    ).masked_fill(~valid, -torch.inf)
    sampled = random_priority.topk(k=2, dim=1).indices
    sampled_action = action_distance.mean_squared.gather(1, sampled)
    sampled_latent = _latent_distance(
        anchor_task_tokens,
        candidate_task_tokens,
        latent_distance,
    ).gather(1, sampled)

    action_order = sampled_action.argsort(dim=1, stable=True)
    ordered_action = sampled_action.gather(1, action_order)
    ordered_latent = sampled_latent.gather(1, action_order)
    near_action, far_action = ordered_action.unbind(dim=1)
    near_latent, far_latent = ordered_latent.unbind(dim=1)
    # Exact ties do not define a ranking relation. This is not an action-size
    # threshold: every strictly ordered equal-duration triplet remains active.
    eligible = has_two_candidates & (near_action < far_action)
    violated = eligible & (near_latent > far_latent)

    if bool(eligible.any()):
        loss = F.relu(near_latent[eligible] - far_latent[eligible]).mean()
        mean_action = near_action[eligible].mean()
        mean_near = near_latent[eligible].mean()
        mean_far = far_latent[eligible].mean()
    else:
        zero = anchor_task_tokens.float().sum() * 0.0
        loss = zero
        mean_action = zero.detach()
        mean_near = zero.detach()
        mean_far = zero.detach()
    return Stage2RelativeRankingLoss(
        loss=loss,
        active_anchors=violated.sum(),
        mean_nearest_action_distance=mean_action,
        mean_near_latent_distance=mean_near,
        mean_hard_negative_latent_distance=mean_far,
    )


def compute_stage2_action_neighborhood_kl_loss(
    anchor_task_tokens: Tensor,
    candidate_task_tokens: Tensor,
    action_distance: PairwiseActionDistance,
    anchor_horizon_steps: Tensor,
    *,
    action_temperature_by_horizon: tuple[float, ...] | list[float],
    token_temperature_by_horizon: tuple[float, ...] | list[float],
    latent_distance: Tensor | None = None,
    row_chunk_size: int = 64,
) -> Stage2ActionNeighborhoodKLLoss:
    """Minimize ``KL(q_action || p_token)`` over every valid candidate.

    The caller is responsible for masking unequal effective durations and
    exact self pairs. Both distributions use all remaining candidates. Token
    distance is raw mean-squared Euclidean distance: no per-sample L2
    normalization removes radial information from the learned geometry.
    """

    if anchor_task_tokens.ndim != 3 or candidate_task_tokens.ndim != 3:
        raise ValueError("task tokens must have shape [B, N, D]")
    if anchor_task_tokens.shape[1:] != candidate_task_tokens.shape[1:]:
        raise ValueError("anchor and candidate task-token layouts differ")
    anchors, candidates = action_distance.mean_squared.shape
    if anchors != len(anchor_task_tokens) or candidates != len(candidate_task_tokens):
        raise ValueError("action distance must align with anchor and candidate tokens")
    if anchor_horizon_steps.shape != (anchors,) or anchor_horizon_steps.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise ValueError("anchor_horizon_steps must be an integer tensor with shape [anchors]")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    action_temperatures = _temperature_table(
        action_temperature_by_horizon,
        name="action",
        device=anchor_task_tokens.device,
    )
    token_temperatures = _temperature_table(
        token_temperature_by_horizon,
        name="token",
        device=anchor_task_tokens.device,
    )
    maximum_horizon = int(anchor_horizon_steps.max()) if anchors else 0
    if maximum_horizon >= len(action_temperatures) or maximum_horizon >= len(token_temperatures):
        raise ValueError("temperature table does not cover every anchor horizon")

    token_distance = _latent_distance(
        anchor_task_tokens,
        candidate_task_tokens,
        latent_distance,
    )
    valid = action_distance.valid_pairs
    active = valid.any(dim=1)
    differentiable_zero = anchor_task_tokens.float().sum() * 0.0
    if not bool(active.any()):
        zero = differentiable_zero.detach()
        return Stage2ActionNeighborhoodKLLoss(
            loss=differentiable_zero,
            active_anchors=active.sum(),
            target_effective_neighbours=zero,
            predicted_effective_neighbours=zero,
            target_top1_probability=zero,
            predicted_top1_probability=zero,
        )

    loss_sum = differentiable_zero
    target_neff_sum = torch.zeros((), device=valid.device, dtype=torch.float32)
    predicted_neff_sum = torch.zeros_like(target_neff_sum)
    target_top1_sum = torch.zeros_like(target_neff_sum)
    predicted_top1_sum = torch.zeros_like(target_neff_sum)
    for start in range(0, anchors, row_chunk_size):
        stop = min(start + row_chunk_size, anchors)
        chunk_active = active[start:stop]
        if not bool(chunk_active.any()):
            continue
        chunk_valid = valid[start:stop][chunk_active]
        horizons = anchor_horizon_steps[start:stop][chunk_active].long()
        action_temperature = action_temperatures[horizons, None]
        token_temperature = token_temperatures[horizons, None]
        action_logits = (
            -action_distance.mean_squared[start:stop][chunk_active].detach() / action_temperature
        ).masked_fill(~chunk_valid, -torch.inf)
        token_logits = (
            -token_distance[start:stop][chunk_active] / token_temperature
        ).masked_fill(~chunk_valid, -torch.inf)
        target_log_probability = F.log_softmax(action_logits, dim=1)
        target_probability = target_log_probability.exp()
        predicted_log_probability = F.log_softmax(token_logits, dim=1)
        row_kl = (
            target_probability
            * (target_log_probability - predicted_log_probability)
        ).masked_fill(~chunk_valid, 0.0).sum(dim=1)
        loss_sum = loss_sum + row_kl.sum()

        with torch.no_grad():
            target_entropy = -(
                target_probability * target_log_probability
            ).masked_fill(~chunk_valid, 0.0).sum(dim=1)
            predicted_probability = predicted_log_probability.exp()
            predicted_entropy = -(
                predicted_probability * predicted_log_probability
            ).masked_fill(~chunk_valid, 0.0).sum(dim=1)
            target_neff_sum += target_entropy.exp().sum()
            predicted_neff_sum += predicted_entropy.exp().sum()
            target_top1_sum += target_probability.max(dim=1).values.sum()
            predicted_top1_sum += predicted_probability.max(dim=1).values.sum()

    active_count = active.sum()
    denominator = active_count.to(dtype=torch.float32)
    return Stage2ActionNeighborhoodKLLoss(
        loss=loss_sum / denominator,
        active_anchors=active_count,
        target_effective_neighbours=target_neff_sum / denominator,
        predicted_effective_neighbours=predicted_neff_sum / denominator,
        target_top1_probability=target_top1_sum / denominator,
        predicted_top1_probability=predicted_top1_sum / denominator,
    )


def _temperature_table(
    values: tuple[float, ...] | list[float],
    *,
    name: str,
    device: torch.device,
) -> Tensor:
    if not values:
        raise ValueError(f"{name} neighbourhood temperature table is empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{name} neighbourhood temperatures must be positive and finite")
    return torch.tensor(values, device=device, dtype=torch.float32)


@torch.no_grad()
def calibrate_token_neighborhood_temperatures(
    latent_distance: Tensor,
    valid_pairs: Tensor,
    anchor_horizon_steps: Tensor,
    *,
    target_effective_neighbours: float,
    maximum_horizon: int = 30,
    iterations: int = 32,
) -> Tensor:
    """Calibrate one fixed token temperature per horizon on an initial batch."""

    if latent_distance.shape != valid_pairs.shape or valid_pairs.dtype != torch.bool:
        raise ValueError("latent distance and valid-pair mask must have the same shape")
    if anchor_horizon_steps.shape != (len(latent_distance),):
        raise ValueError("anchor horizons must align with latent-distance rows")
    if target_effective_neighbours <= 1.0 or iterations <= 0:
        raise ValueError("temperature calibration target and iterations must be positive")
    result = torch.zeros(maximum_horizon + 1, device=latent_distance.device, dtype=torch.float64)
    distance = latent_distance.detach().double()
    for horizon in range(maximum_horizon + 1):
        rows = (anchor_horizon_steps == horizon) & (
            valid_pairs.sum(dim=1) >= math.ceil(target_effective_neighbours)
        )
        if not bool(rows.any()):
            continue
        row_distance = distance[rows]
        row_valid = valid_pairs[rows]
        finite = row_distance.masked_select(row_valid)
        characteristic = finite.median().clamp_min(torch.finfo(torch.float64).eps)
        low = characteristic * 1e-6
        high = characteristic * 1e3
        for _ in range(iterations):
            temperature = (low + high) * 0.5
            logits = (-row_distance / temperature).masked_fill(~row_valid, -torch.inf)
            log_probability = F.log_softmax(logits, dim=1)
            probability = log_probability.exp()
            entropy = -(probability * log_probability).masked_fill(~row_valid, 0.0).sum(dim=1)
            median_neighbours = entropy.exp().median()
            if median_neighbours < target_effective_neighbours:
                low = temperature
            else:
                high = temperature
        result[horizon] = (low + high) * 0.5
    return result.float()


def compute_stage2_latent_distance(
    anchor_task_tokens: Tensor,
    candidate_task_tokens: Tensor,
) -> Tensor:
    """Compute mean-squared Euclidean task-token distance for matching."""

    if anchor_task_tokens.ndim != 3 or candidate_task_tokens.ndim != 3:
        raise ValueError("task tokens must have shape [B, N, D]")
    if anchor_task_tokens.shape[1:] != candidate_task_tokens.shape[1:]:
        raise ValueError("anchor and candidate task-token layouts differ")
    anchor_embedding = anchor_task_tokens.flatten(1).float()
    candidate_embedding = candidate_task_tokens.flatten(1).float()
    squared = (
        anchor_embedding.square().sum(dim=1, keepdim=True)
        + candidate_embedding.square().sum(dim=1).unsqueeze(0)
        - 2.0 * (anchor_embedding @ candidate_embedding.transpose(0, 1))
    ).clamp_min(0.0)
    return squared / anchor_embedding.shape[1]


def _latent_distance(
    anchor_task_tokens: Tensor,
    candidate_task_tokens: Tensor,
    latent_distance: Tensor | None,
) -> Tensor:
    expected = (anchor_task_tokens.shape[0], candidate_task_tokens.shape[0])
    if latent_distance is None:
        return compute_stage2_latent_distance(anchor_task_tokens, candidate_task_tokens)
    if latent_distance.shape != expected or not latent_distance.is_floating_point():
        raise ValueError("latent_distance must be floating with shape [anchors, candidates]")
    if not torch.isfinite(latent_distance).all() or torch.any(latent_distance < 0.0):
        raise ValueError("latent_distance must be finite and non-negative")
    return latent_distance
