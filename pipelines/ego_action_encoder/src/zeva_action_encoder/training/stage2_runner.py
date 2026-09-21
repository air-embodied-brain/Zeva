"""Stage 2 triplet forward and distributed action-matching objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

from zeva_action_encoder.config import Stage1AuxiliaryTargetStats
from zeva_action_encoder.data.stage1.targets import action_endpoint_from_chunk, normalize_auxiliary_target
from zeva_action_encoder.data.stage2 import (
    ActionDescriptor,
    ActionDescriptorConfig,
    build_action_descriptor,
)
from zeva_action_encoder.models.stage2 import (
    EMAStage2EnvironmentTeacher,
    FrozenStage1EnvironmentTeacher,
    Stage2TaskRelations,
    Stage2TripletOutput,
)
from zeva_action_encoder.training.hsic import DEFAULT_RBF_SCALES, normalized_rbf_hsic
from zeva_action_encoder.training.camera_residual import FrozenCameraResidualPredictor
from zeva_action_encoder.training.stage2 import compute_stage2_algebra_loss
from zeva_action_encoder.training.stage2_contrastive import (
    PairwiseActionDistance,
    Stage2ActionNeighborhoodKLLoss,
    Stage2ContrastiveLoss,
    Stage2RelativeRankingLoss,
    compute_stage2_action_neighborhood_kl_loss,
    calibrate_token_neighborhood_temperatures,
    compute_stage2_cross_contrastive_loss,
    compute_stage2_latent_distance,
    compute_stage2_listwise_ranking_loss,
    compute_stage2_local_triplet_ranking_loss,
    compute_stage2_relative_ranking_loss,
    cross_masked_action_distance,
    mask_action_distance_by_effective_duration,
    select_cross_contrastive_pairs,
)


@dataclass(frozen=True)
class Stage2LossConfig:
    reconstruction_weight: float = 1.0
    environment_consistency_weight: float = 0.0
    additive_weight: float = 0.1
    reversal_weight: float = 0.1
    contrastive_weight: float = 0.05
    ranking_weight: float = 0.05
    camera_hsic_weight: float = 0.0
    camera_hsic_target: str = "raw_camera_v1"
    camera_hsic_max_samples_per_bin: int = 1024
    camera_hsic_kernel_scales: tuple[float, ...] = DEFAULT_RBF_SCALES
    camera_hsic_horizon_bin_edges: tuple[int, ...] = (5, 10, 15, 20, 25, 31)
    environment_action_hsic_weight: float = 0.0
    environment_action_hsic_max_samples: int = 1024
    environment_action_hsic_kernel_scales: tuple[float, ...] = DEFAULT_RBF_SCALES
    environment_variance_weight: float = 0.0
    environment_variance_floor: float = 0.05
    contrastive_temperature: float = 0.1
    positive_max_distance: float = 0.003
    negative_min_distance: float | None = None
    ranking_margin: float = 0.1
    ranking_mode: str = "legacy_threshold_v1"
    ranking_normalization_epsilon: float = 1e-12
    trajectory_points: int = 8
    effective_duration_tolerance_seconds: float = 1e-6
    min_shared_dimensions: int = 1
    action_translation_distance_scale: float = 1.0
    action_rotation_distance_scale: float = 1.0
    action_gripper_distance_scale: float = 1.0
    action_neighborhood_temperature_by_horizon: tuple[float, ...] = ()
    token_neighborhood_temperature_by_horizon: tuple[float, ...] = ()
    token_temperature_initial_multiplier: float = 1.0
    token_temperature_final_multiplier: float = 1.0
    token_temperature_anneal_start_step: int = 0
    token_temperature_anneal_end_step: int = 0
    neighborhood_row_chunk_size: int = 64
    neighborhood_target_effective_neighbours: float = 32.0

    def __post_init__(self) -> None:
        weights = (
            self.reconstruction_weight,
            self.environment_consistency_weight,
            self.additive_weight,
            self.reversal_weight,
            self.contrastive_weight,
            self.ranking_weight,
            self.camera_hsic_weight,
            self.environment_action_hsic_weight,
            self.environment_variance_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("Stage 2 loss weights must be non-negative")
        if self.contrastive_temperature <= 0.0:
            raise ValueError("Stage 2 temperature must be positive")
        if self.ranking_mode not in {
            "legacy_threshold_v1",
            "listwise_same_duration_v1",
            "triplet_same_duration_v1",
            "neighborhood_kl_same_duration_v1",
        }:
            raise ValueError("unsupported Stage 2 ranking mode")
        if self.ranking_margin < 0.0 or (
            self.ranking_mode == "legacy_threshold_v1" and self.ranking_margin <= 0.0
        ):
            raise ValueError("legacy Stage 2 ranking requires a positive margin")
        if self.ranking_normalization_epsilon <= 0.0:
            raise ValueError("ranking_normalization_epsilon must be positive")
        if self.positive_max_distance < 0.0:
            raise ValueError("Stage 2 positive action threshold must be non-negative")
        if (
            self.negative_min_distance is not None
            and self.negative_min_distance <= self.positive_max_distance
        ):
            raise ValueError("Stage 2 negative threshold must exceed the positive threshold")
        descriptor_dim = ActionDescriptorConfig(self.trajectory_points).descriptor_dim
        if not 0 < self.min_shared_dimensions <= descriptor_dim:
            raise ValueError("min_shared_dimensions is outside the action descriptor")
        if self.effective_duration_tolerance_seconds < 0.0:
            raise ValueError("effective duration tolerance must be non-negative")
        if self.camera_hsic_max_samples_per_bin < 2:
            raise ValueError("camera_hsic_max_samples_per_bin must be at least two")
        if self.environment_action_hsic_max_samples < 2:
            raise ValueError("environment_action_hsic_max_samples must be at least two")
        if self.camera_hsic_target not in {"raw_camera_v1", "action_horizon_residual_v1"}:
            raise ValueError("unsupported camera HSIC target")
        if not self.camera_hsic_kernel_scales or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.camera_hsic_kernel_scales
        ):
            raise ValueError("camera HSIC kernel scales must be positive and finite")
        if not self.environment_action_hsic_kernel_scales or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.environment_action_hsic_kernel_scales
        ):
            raise ValueError("environment action HSIC kernel scales must be positive and finite")
        if not math.isfinite(self.environment_variance_floor) or self.environment_variance_floor < 0.0:
            raise ValueError("environment variance floor must be finite and non-negative")
        edges = tuple(int(value) for value in self.camera_hsic_horizon_bin_edges)
        if (
            len(edges) < 2
            or edges[0] != 5
            or edges[-1] != 31
            or any(left >= right for left, right in zip(edges[:-1], edges[1:], strict=True))
        ):
            raise ValueError("camera HSIC horizon bins must be increasing edges covering 5--30")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in self.action_distance_group_scales
        ):
            raise ValueError("action-distance group scales must be positive and finite")
        if self.ranking_mode == "neighborhood_kl_same_duration_v1":
            for name, values in (
                ("action", self.action_neighborhood_temperature_by_horizon),
                ("token", self.token_neighborhood_temperature_by_horizon),
            ):
                if len(values) <= 30 or any(
                    not math.isfinite(value) or value <= 0.0 for value in values[5:31]
                ):
                    raise ValueError(
                        f"{name} neighbourhood temperature table must cover horizons 5--30"
                    )
            if self.neighborhood_row_chunk_size <= 0:
                raise ValueError("neighborhood_row_chunk_size must be positive")
            if self.neighborhood_target_effective_neighbours <= 1.0:
                raise ValueError("neighborhood target effective neighbours must exceed one")
            multipliers = (
                self.token_temperature_initial_multiplier,
                self.token_temperature_final_multiplier,
            )
            if any(not math.isfinite(value) or value <= 0.0 for value in multipliers):
                raise ValueError("token-temperature multipliers must be positive and finite")
            if self.token_temperature_anneal_start_step < 0:
                raise ValueError("token-temperature anneal start step must be non-negative")
            if self.token_temperature_anneal_end_step < self.token_temperature_anneal_start_step:
                raise ValueError("token-temperature anneal end step must not precede its start")
            if (
                self.token_temperature_initial_multiplier
                != self.token_temperature_final_multiplier
                and self.token_temperature_anneal_end_step
                == self.token_temperature_anneal_start_step
            ):
                raise ValueError("a changing token temperature requires a non-empty anneal interval")

    @property
    def action_distance_group_scales(self) -> tuple[float, float, float]:
        return (
            self.action_translation_distance_scale,
            self.action_rotation_distance_scale,
            self.action_gripper_distance_scale,
        )


@dataclass
class Stage2BatchLoss:
    total: Tensor
    reconstruction: Tensor
    environment_consistency: Tensor
    additive: Tensor
    reversal: Tensor
    contrastive: Tensor
    ranking: Tensor
    camera_hsic: Tensor
    camera_hsic_translation: Tensor
    camera_hsic_rotation: Tensor
    camera_hsic_samples: Tensor
    camera_hsic_active_bins: Tensor
    environment_action_hsic: Tensor
    environment_action_hsic_samples: Tensor
    environment_variance: Tensor
    task_token_rms: Tensor
    contrastive_active_anchors: Tensor
    ranking_active_anchors: Tensor
    positive_pairs: Tensor
    negative_pairs: Tensor
    nearest_action_distance: Tensor
    near_latent_distance: Tensor
    hard_negative_latent_distance: Tensor
    target_effective_neighbours: Tensor
    predicted_effective_neighbours: Tensor
    target_top1_probability: Tensor
    predicted_top1_probability: Tensor
    calibrated_token_temperature_by_horizon: Tensor


def compute_stage2_batch_loss(
    *,
    distributed_model: nn.Module,
    teacher: FrozenStage1EnvironmentTeacher | None,
    visual_encoder: nn.Module,
    batch: dict[str, Tensor | str],
    source_index: int,
    config: Stage2LossConfig,
    optimizer_step: int = 0,
    calibrate_token_temperature: bool = False,
    camera_residual_predictor: FrozenCameraResidualPredictor | None = None,
    ema_teacher: EMAStage2EnvironmentTeacher | None = None,
    action_endpoint_stats: Stage1AuxiliaryTargetStats | None = None,
) -> Stage2BatchLoss:
    """Run one homogeneous local triplet batch and global cross-source mining."""

    images = _tensor(batch, "images")
    action = _tensor(batch, "action_chunk")
    action_mask = _tensor(batch, "action_mask")
    dimension_mask = _tensor(batch, "action_dimension_mask")
    batch_size = images.shape[0]
    if images.ndim != 5 or images.shape[1] != 3:
        raise ValueError("Stage 2 images must have shape [B, 3, C, H, W]")
    if action.shape[:2] != (batch_size, 3):
        raise ValueError("Stage 2 action relations must align with three frame pairs")

    flat_images = images.flatten(0, 1)
    with torch.no_grad():
        flat_features = visual_encoder(flat_images)
    features = flat_features.unflatten(0, (batch_size, 3))
    frame_a, frame_b, frame_c = features.unbind(dim=1)
    transition_starts = torch.cat([frame_a, frame_b, frame_a], dim=0)
    transition_futures = torch.cat([frame_b, frame_c, frame_c], dim=0)
    flat_action = action.transpose(0, 1).flatten(0, 1)
    flat_action_mask = action_mask.transpose(0, 1).flatten(0, 1)
    flat_dimension_mask = dimension_mask.transpose(0, 1).flatten(0, 1)
    stage2_model = getattr(distributed_model, "module", distributed_model)
    environment_mode = stage2_model.config.environment_mode
    teacher_environment: Tensor | None = None
    if teacher is not None:
        teacher_environment = teacher(
            transition_starts,
            transition_futures,
            flat_action,
            flat_action_mask,
            flat_dimension_mask,
        )
    ema_environment: Tensor | None = None
    if ema_teacher is not None:
        ema_environment = ema_teacher(transition_starts, transition_futures)
    if environment_mode == "joint_query_v1":
        if config.environment_consistency_weight > 0.0 and teacher_environment is None:
            raise ValueError("joint-query environment consistency requires a frozen Stage 1 teacher")
        if ema_environment is not None:
            raise ValueError("legacy joint-query Stage 2 must not receive an EMA teacher")
        output = distributed_model(frame_a, frame_b, frame_c=frame_c)
    elif environment_mode == "joint_query_ema_v1":
        if teacher_environment is not None:
            raise ValueError("EMA joint-query Stage 2 must not receive a frozen Stage 1 teacher")
        if config.environment_consistency_weight > 0.0 and ema_environment is None:
            raise ValueError("EMA joint-query environment consistency requires an EMA teacher")
        output = distributed_model(frame_a, frame_b, frame_c=frame_c)
    else:
        if ema_environment is not None:
            raise ValueError("external-teacher Stage 2 must not receive an EMA teacher")
        if teacher_environment is None:
            raise ValueError("external-teacher Stage 2 requires a frozen Stage 1 teacher")
        environment_ab, environment_bc, environment_ac = teacher_environment.split(
            batch_size,
            dim=0,
        )
        output = distributed_model(
            frame_a,
            frame_b,
            environment_ab,
            frame_c=frame_c,
            environment_bc=environment_bc,
            environment_ac=environment_ac,
        )
    if not isinstance(output, Stage2TripletOutput):
        raise TypeError("Stage 2 triplet forward returned a single-transition output")
    reconstruction = torch.stack(
        [
            F.mse_loss(output.reconstructed_ab.float(), frame_b.float()),
            F.mse_loss(output.reconstructed_bc.float(), frame_c.float()),
            F.mse_loss(output.reconstructed_ac.float(), frame_c.float()),
        ]
    ).mean()
    student_environment = torch.cat(
        [
            output.environments.ab,
            output.environments.bc,
            output.environments.ac,
        ],
        dim=0,
    )
    if environment_mode == "joint_query_v1" and teacher_environment is not None:
        environment_consistency = F.mse_loss(
            student_environment.float(),
            teacher_environment.detach().float(),
        )
    elif environment_mode == "joint_query_ema_v1" and ema_environment is not None:
        environment_consistency = F.mse_loss(
            student_environment.float(),
            ema_environment.detach().float(),
        )
    else:
        environment_consistency = student_environment.float().sum() * 0.0
    environment_action_hsic = _zero_environment_action_hsic_loss(student_environment)
    if config.environment_action_hsic_weight > 0.0:
        if action_endpoint_stats is None:
            raise ValueError("environment action HSIC requires source-specific endpoint statistics")
        action_endpoint = normalize_auxiliary_target(
            action_endpoint_from_chunk(
                _tensor(batch, "action_chunk_physical").transpose(0, 1).flatten(0, 1),
                flat_action_mask,
                flat_dimension_mask,
            ),
            action_endpoint_stats.action_endpoint_mean,
            action_endpoint_stats.action_endpoint_std,
        )
        environment_action_hsic = compute_stage2_environment_action_hsic_loss(
            student_environment,
            action_endpoint,
            episode_ids=batch.get("episode_id"),
            max_samples=config.environment_action_hsic_max_samples,
            kernel_scales=config.environment_action_hsic_kernel_scales,
        )
    elif action_endpoint_stats is not None:
        raise ValueError("action endpoint statistics were supplied while environment HSIC is disabled")
    environment_variance = compute_stage2_environment_variance_floor_loss(
        student_environment,
        floor=config.environment_variance_floor,
        enabled=config.environment_variance_weight > 0.0,
    )
    algebra = compute_stage2_algebra_loss(output.relations)
    local_task_tokens = _matching_task_tokens(output.relations)
    contrastive, ranking = _zero_matching_losses(local_task_tokens)
    neighborhood = _zero_neighborhood_loss(local_task_tokens)
    calibrated_token_temperature = torch.zeros(31, device=images.device, dtype=torch.float32)
    local_descriptor: ActionDescriptor | None = None
    local_duration: Tensor | None = None
    needs_descriptor = (
        config.contrastive_weight > 0.0
        or config.ranking_weight > 0.0
        or (
            config.camera_hsic_weight > 0.0
            and config.camera_hsic_target == "action_horizon_residual_v1"
        )
    )
    if needs_descriptor:
        physical = _tensor(batch, "action_chunk_physical")
        duration = _tensor(batch, "duration_seconds")
        aperture_range = _tensor(batch, "gripper_aperture_range")
        descriptor_config = ActionDescriptorConfig(config.trajectory_points)
        local_descriptor = build_action_descriptor(
            physical.transpose(0, 1).flatten(0, 1),
            flat_action_mask,
            flat_dimension_mask,
            aperture_range[None, :, :].expand(3, -1, -1).flatten(0, 1),
            config=descriptor_config,
        )
        local_duration = duration.transpose(0, 1).flatten()
    camera_hsic = _zero_camera_hsic_loss(local_task_tokens)
    if config.camera_hsic_weight > 0.0:
        if source_index == 0:
            camera_endpoint = _tensor(batch, "camera_endpoint_motion")
            if camera_endpoint.shape != (batch_size, 3, 6):
                raise ValueError("EgoDex camera endpoint motion must have shape [B, 3, 6]")
            camera_target = camera_endpoint.transpose(0, 1).flatten(0, 1).float()
            if config.camera_hsic_target == "action_horizon_residual_v1":
                if camera_residual_predictor is None or local_descriptor is None or local_duration is None:
                    raise ValueError("camera residual HSIC requires a frozen predictor and action descriptor")
                if not bool(local_descriptor.dimension_mask.all()):
                    raise ValueError("camera residual predictor requires a complete EgoDex action descriptor")
                camera_target = camera_target - camera_residual_predictor.predict(
                    local_descriptor.values,
                    local_duration,
                )
            elif camera_residual_predictor is not None:
                raise ValueError("raw camera HSIC must not receive a residual predictor")
            camera_hsic = compute_stage2_camera_hsic_loss(
                local_task_tokens,
                camera_target,
                _tensor(batch, "horizon_steps").transpose(0, 1).flatten().to(torch.int64),
                episode_ids=batch.get("episode_id"),
                config=config,
            )
        elif "camera_endpoint_motion" in batch:
            raise ValueError("camera endpoint motion must only be present for EgoDex")
    if config.contrastive_weight > 0.0 or config.ranking_weight > 0.0:
        if local_descriptor is None or local_duration is None:
            raise RuntimeError("action-matching losses require a local action descriptor")
        if config.ranking_mode == "neighborhood_kl_same_duration_v1" and config.ranking_weight > 0.0:
            candidate_task_tokens = _all_gather_with_grad(local_task_tokens)
        else:
            candidate_task_tokens = _all_gather_detached(local_task_tokens)
        candidate_descriptor = ActionDescriptor(
            values=_all_gather_detached(local_descriptor.values),
            dimension_mask=_all_gather_detached(local_descriptor.dimension_mask.to(torch.uint8)).bool(),
        )
        local_source_ids = torch.full(
            (len(local_task_tokens),), source_index, dtype=torch.int64, device=images.device
        )
        candidate_source_ids = _all_gather_detached(local_source_ids)
        candidate_duration = _all_gather_detached(local_duration)
        action_distance = cross_masked_action_distance(
            local_descriptor,
            candidate_descriptor,
            config=descriptor_config,
            min_shared_dimensions=config.min_shared_dimensions,
            group_mse_scales=config.action_distance_group_scales,
        )
        action_distance = mask_action_distance_by_effective_duration(
            action_distance,
            local_duration,
            candidate_duration,
            tolerance_seconds=config.effective_duration_tolerance_seconds,
        )
        latent_distance = compute_stage2_latent_distance(
            local_task_tokens,
            candidate_task_tokens,
        )
        if calibrate_token_temperature:
            horizon_steps = _tensor(batch, "horizon_steps").transpose(0, 1).flatten().to(torch.int64)
            calibration_distance = _exclude_local_self_pairs(
                action_distance,
                len(local_task_tokens),
            )
            calibrated_token_temperature = calibrate_token_neighborhood_temperatures(
                latent_distance,
                calibration_distance.valid_pairs,
                horizon_steps,
                target_effective_neighbours=config.neighborhood_target_effective_neighbours,
            )
    if config.contrastive_weight > 0.0:
        pairs = select_cross_contrastive_pairs(
            action_distance,
            local_source_ids,
            candidate_source_ids,
            positive_max_distance=config.positive_max_distance,
            negative_min_distance=config.negative_min_distance,
        )
        contrastive = compute_stage2_cross_contrastive_loss(
            local_task_tokens,
            candidate_task_tokens,
            pairs,
            temperature=config.contrastive_temperature,
            latent_distance=latent_distance,
        )
    if config.ranking_weight > 0.0:
        if config.ranking_mode in {
            "listwise_same_duration_v1",
            "triplet_same_duration_v1",
            "neighborhood_kl_same_duration_v1",
        }:
            action_distance = _exclude_local_self_pairs(action_distance, len(local_task_tokens))
        if config.ranking_mode == "listwise_same_duration_v1":
            ranking = compute_stage2_listwise_ranking_loss(
                local_task_tokens,
                candidate_task_tokens,
                action_distance,
                normalization_epsilon=config.ranking_normalization_epsilon,
                latent_distance=latent_distance,
            )
        elif config.ranking_mode == "triplet_same_duration_v1":
            ranking = compute_stage2_local_triplet_ranking_loss(
                local_task_tokens,
                candidate_task_tokens,
                action_distance,
                latent_distance=latent_distance,
            )
        elif config.ranking_mode == "neighborhood_kl_same_duration_v1":
            horizon_steps = _tensor(batch, "horizon_steps").transpose(0, 1).flatten().to(torch.int64)
            token_temperatures = scheduled_token_neighborhood_temperatures(
                config,
                optimizer_step=optimizer_step,
            )
            neighborhood = compute_stage2_action_neighborhood_kl_loss(
                local_task_tokens,
                candidate_task_tokens,
                action_distance,
                horizon_steps,
                action_temperature_by_horizon=config.action_neighborhood_temperature_by_horizon,
                token_temperature_by_horizon=token_temperatures,
                latent_distance=latent_distance,
                row_chunk_size=config.neighborhood_row_chunk_size,
            )
            ranking = Stage2RelativeRankingLoss(
                loss=neighborhood.loss,
                active_anchors=neighborhood.active_anchors,
                mean_nearest_action_distance=ranking.mean_nearest_action_distance,
                mean_near_latent_distance=ranking.mean_near_latent_distance,
                mean_hard_negative_latent_distance=ranking.mean_hard_negative_latent_distance,
            )
        else:
            ranking = compute_stage2_relative_ranking_loss(
                local_task_tokens,
                candidate_task_tokens,
                action_distance,
                local_source_ids,
                candidate_source_ids,
                positive_max_distance=config.positive_max_distance,
                negative_min_distance=config.negative_min_distance,
                margin=config.ranking_margin,
                latent_distance=latent_distance,
            )
    total = (
        config.reconstruction_weight * reconstruction
        + config.environment_consistency_weight * environment_consistency
        + config.additive_weight * algebra.additive
        + config.reversal_weight * algebra.reversal
        + config.contrastive_weight * contrastive.loss
        + config.ranking_weight * ranking.loss
        + config.camera_hsic_weight * camera_hsic.loss
        + config.environment_action_hsic_weight * environment_action_hsic.loss
        + config.environment_variance_weight * environment_variance
    )
    return Stage2BatchLoss(
        total=total,
        reconstruction=reconstruction,
        environment_consistency=environment_consistency,
        additive=algebra.additive,
        reversal=algebra.reversal,
        contrastive=contrastive.loss,
        ranking=ranking.loss,
        camera_hsic=camera_hsic.loss,
        camera_hsic_translation=camera_hsic.translation,
        camera_hsic_rotation=camera_hsic.rotation,
        camera_hsic_samples=camera_hsic.samples,
        camera_hsic_active_bins=camera_hsic.active_bins,
        environment_action_hsic=environment_action_hsic.loss,
        environment_action_hsic_samples=environment_action_hsic.samples,
        environment_variance=environment_variance,
        task_token_rms=local_task_tokens.float().square().mean().sqrt(),
        contrastive_active_anchors=contrastive.active_anchors,
        ranking_active_anchors=ranking.active_anchors,
        positive_pairs=contrastive.positive_pairs,
        negative_pairs=contrastive.negative_pairs,
        nearest_action_distance=ranking.mean_nearest_action_distance,
        near_latent_distance=ranking.mean_near_latent_distance,
        hard_negative_latent_distance=ranking.mean_hard_negative_latent_distance,
        target_effective_neighbours=neighborhood.target_effective_neighbours,
        predicted_effective_neighbours=neighborhood.predicted_effective_neighbours,
        target_top1_probability=neighborhood.target_top1_probability,
        predicted_top1_probability=neighborhood.predicted_top1_probability,
        calibrated_token_temperature_by_horizon=calibrated_token_temperature,
    )


@dataclass(frozen=True)
class Stage2EnvironmentActionHSICLoss:
    loss: Tensor
    samples: Tensor


def compute_stage2_environment_action_hsic_loss(
    environment_tokens: Tensor,
    action_endpoint: Tensor,
    *,
    episode_ids: object,
    max_samples: int,
    kernel_scales: tuple[float, ...],
) -> Stage2EnvironmentActionHSICLoss:
    """Apply Stage 1-style env/action HSIC without horizon partitioning."""

    sample_count = len(environment_tokens)
    if environment_tokens.ndim != 3:
        raise ValueError("environment HSIC tokens must have shape [N, T, D]")
    if action_endpoint.shape != (sample_count, 14):
        raise ValueError("environment HSIC action endpoint must have shape [N, 14]")
    if max_samples < 2:
        raise ValueError("environment HSIC requires a sample limit of at least two")
    if not torch.isfinite(action_endpoint).all():
        raise ValueError("environment HSIC action endpoints must be finite")

    groups = _flatten_relation_episode_ids(episode_ids, sample_count=sample_count)
    candidates = _interleave_relation_candidates(
        torch.arange(sample_count, dtype=torch.int64, device=environment_tokens.device),
        relation_size=sample_count // 3,
    )
    indices = _episode_diverse_subset(candidates, groups, limit=max_samples)
    if len(indices) < 2:
        raise ValueError("environment HSIC requires at least two selected samples")
    loss = normalized_rbf_hsic(
        environment_tokens.index_select(0, indices),
        action_endpoint.index_select(0, indices),
        kernel_scales=kernel_scales,
    )
    return Stage2EnvironmentActionHSICLoss(
        loss=loss,
        samples=torch.tensor(len(indices), dtype=torch.int64, device=environment_tokens.device),
    )


def _zero_environment_action_hsic_loss(
    environment_tokens: Tensor,
) -> Stage2EnvironmentActionHSICLoss:
    differentiable_zero = environment_tokens.float().sum() * 0.0
    return Stage2EnvironmentActionHSICLoss(
        loss=differentiable_zero,
        samples=torch.zeros((), dtype=torch.int64, device=environment_tokens.device),
    )


def compute_stage2_environment_variance_floor_loss(
    environment_tokens: Tensor,
    *,
    floor: float,
    enabled: bool,
) -> Tensor:
    """Prevent HSIC from winning by collapsing every env coordinate."""

    if environment_tokens.ndim != 3:
        raise ValueError("environment tokens must have shape [N, T, D]")
    if not enabled:
        return environment_tokens.float().sum() * 0.0
    if floor < 0.0:
        raise ValueError("environment variance floor must be non-negative")
    values = environment_tokens.float()
    standard_deviation = torch.sqrt(values.var(dim=0, correction=0) + 1e-8)
    return F.relu(standard_deviation.new_tensor(floor) - standard_deviation).mean()


@dataclass(frozen=True)
class Stage2CameraHSICLoss:
    loss: Tensor
    translation: Tensor
    rotation: Tensor
    samples: Tensor
    active_bins: Tensor


def compute_stage2_camera_hsic_loss(
    task_tokens: Tensor,
    camera_endpoint_motion: Tensor,
    horizon_steps: Tensor,
    *,
    episode_ids: object,
    config: Stage2LossConfig,
) -> Stage2CameraHSICLoss:
    """Penalize task/camera dependence separately inside fixed horizon bins."""

    sample_count = len(task_tokens)
    if task_tokens.ndim != 3:
        raise ValueError("camera HSIC task tokens must have shape [N, T, D]")
    if camera_endpoint_motion.shape != (sample_count, 6):
        raise ValueError("camera endpoint motion must have shape [N, 6]")
    if horizon_steps.shape != (sample_count,) or horizon_steps.dtype != torch.int64:
        raise ValueError("camera HSIC horizon steps must be an int64 [N] tensor")
    if not torch.isfinite(camera_endpoint_motion).all():
        raise ValueError("camera endpoint motion must be finite")

    flattened_groups = _flatten_relation_episode_ids(episode_ids, sample_count=sample_count)
    weighted_translation = task_tokens.float().sum() * 0.0
    weighted_rotation = task_tokens.float().sum() * 0.0
    selected_total = 0
    active_bins = 0
    edges = tuple(int(value) for value in config.camera_hsic_horizon_bin_edges)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        candidates = torch.nonzero(
            (horizon_steps >= lower) & (horizon_steps < upper),
            as_tuple=False,
        ).flatten()
        candidates = _interleave_relation_candidates(
            candidates,
            relation_size=sample_count // 3,
        )
        indices = _episode_diverse_subset(
            candidates,
            flattened_groups,
            limit=config.camera_hsic_max_samples_per_bin,
        )
        count = len(indices)
        if count < 2:
            continue
        selected_tokens = task_tokens.index_select(0, indices)
        selected_camera = camera_endpoint_motion.index_select(0, indices)
        translation = normalized_rbf_hsic(
            selected_tokens,
            selected_camera[:, :3],
            kernel_scales=config.camera_hsic_kernel_scales,
        )
        rotation = normalized_rbf_hsic(
            selected_tokens,
            selected_camera[:, 3:],
            kernel_scales=config.camera_hsic_kernel_scales,
        )
        weighted_translation = weighted_translation + count * translation
        weighted_rotation = weighted_rotation + count * rotation
        selected_total += count
        active_bins += 1

    if selected_total == 0:
        return _zero_camera_hsic_loss(task_tokens)
    translation_loss = weighted_translation / selected_total
    rotation_loss = weighted_rotation / selected_total
    return Stage2CameraHSICLoss(
        loss=0.5 * (translation_loss + rotation_loss),
        translation=translation_loss,
        rotation=rotation_loss,
        samples=torch.tensor(selected_total, dtype=torch.int64, device=task_tokens.device),
        active_bins=torch.tensor(active_bins, dtype=torch.int64, device=task_tokens.device),
    )


def _zero_camera_hsic_loss(task_tokens: Tensor) -> Stage2CameraHSICLoss:
    differentiable_zero = task_tokens.float().sum() * 0.0
    scalar_zero = differentiable_zero.detach()
    count_zero = torch.zeros((), dtype=torch.int64, device=task_tokens.device)
    return Stage2CameraHSICLoss(
        loss=differentiable_zero,
        translation=scalar_zero,
        rotation=scalar_zero,
        samples=count_zero,
        active_bins=count_zero,
    )


def _flatten_relation_episode_ids(episode_ids: object, *, sample_count: int) -> list[object] | None:
    if episode_ids is None:
        return None
    if not isinstance(episode_ids, (list, tuple)):
        raise TypeError("episode_id must be a sequence for camera HSIC")
    if sample_count != 3 * len(episode_ids):
        raise ValueError("episode_id must align with three flattened Stage 2 relations")
    return list(episode_ids) * 3


def _episode_diverse_subset(
    candidates: Tensor,
    groups: list[object] | None,
    *,
    limit: int,
) -> Tensor:
    if candidates.ndim != 1 or candidates.dtype != torch.int64:
        raise ValueError("camera HSIC candidate indices must be an int64 vector")
    if len(candidates) <= limit or groups is None:
        return candidates[:limit]
    candidate_values = candidates.detach().cpu().tolist()
    selected: list[int] = []
    selected_set: set[int] = set()
    seen_groups: set[object] = set()
    for index in candidate_values:
        group = groups[index]
        if group in seen_groups:
            continue
        seen_groups.add(group)
        selected.append(index)
        selected_set.add(index)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for index in candidate_values:
            if index in selected_set:
                continue
            selected.append(index)
            if len(selected) == limit:
                break
    return torch.tensor(selected, dtype=torch.int64, device=candidates.device)


def _interleave_relation_candidates(candidates: Tensor, *, relation_size: int) -> Tensor:
    """Prevent a capped HSIC bin from being filled by AB before BC/AC."""

    if relation_size <= 0 or 3 * relation_size <= int(candidates.max().item() if len(candidates) else -1):
        raise ValueError("camera HSIC relations must contain three equal-size blocks")
    relation_lists = [
        candidates[(candidates // relation_size) == relation].detach().cpu().tolist()
        for relation in range(3)
    ]
    interleaved = [
        relation_lists[relation][offset]
        for offset in range(max((len(values) for values in relation_lists), default=0))
        for relation in range(3)
        if offset < len(relation_lists[relation])
    ]
    return torch.tensor(interleaved, dtype=torch.int64, device=candidates.device)


def token_temperature_multiplier(config: Stage2LossConfig, *, optimizer_step: int) -> float:
    """Return the deterministic log-cosine token-temperature multiplier."""

    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    start = config.token_temperature_anneal_start_step
    end = config.token_temperature_anneal_end_step
    initial = config.token_temperature_initial_multiplier
    final = config.token_temperature_final_multiplier
    if optimizer_step <= start or start == end:
        return initial
    if optimizer_step >= end:
        return final
    fraction = (optimizer_step - start) / (end - start)
    cosine_progress = 0.5 * (1.0 - math.cos(math.pi * fraction))
    return math.exp(
        (1.0 - cosine_progress) * math.log(initial)
        + cosine_progress * math.log(final)
    )


def scheduled_token_neighborhood_temperatures(
    config: Stage2LossConfig,
    *,
    optimizer_step: int,
) -> tuple[float, ...]:
    """Scale the frozen per-horizon table without mutating loss configuration."""

    multiplier = token_temperature_multiplier(config, optimizer_step=optimizer_step)
    return tuple(
        temperature * multiplier
        for temperature in config.token_neighborhood_temperature_by_horizon
    )



def _exclude_local_self_pairs(
    distance: PairwiseActionDistance,
    local_count: int,
) -> PairwiseActionDistance:
    """Remove each local anchor's exact entry from rank-ordered global candidates."""

    if local_count <= 0 or distance.mean_squared.shape[0] != local_count:
        raise ValueError("local_count must align with action-distance anchors")
    if not dist.is_initialized():
        rank = 0
    else:
        rank = dist.get_rank()
    candidate_offset = rank * local_count
    columns = candidate_offset + torch.arange(local_count, device=distance.mean_squared.device)
    if int(columns[-1]) >= distance.mean_squared.shape[1]:
        raise ValueError("global candidate layout does not contain the local rank block")
    rows = torch.arange(local_count, device=distance.mean_squared.device)
    valid = distance.valid_pairs.clone()
    valid[rows, columns] = False
    return PairwiseActionDistance(
        mean_squared=distance.mean_squared.masked_fill(~valid, torch.inf),
        valid_pairs=valid,
        shared_dimensions=distance.shared_dimensions,
    )

def _matching_task_tokens(relations: Stage2TaskRelations) -> Tensor:
    return torch.cat([relations.ab, relations.bc, relations.ac], dim=0)


def _zero_matching_losses(task_tokens: Tensor) -> tuple[Stage2ContrastiveLoss, Stage2RelativeRankingLoss]:
    differentiable_zero = task_tokens.float().sum() * 0.0
    scalar_zero = differentiable_zero.detach()
    count_zero = torch.zeros((), dtype=torch.int64, device=task_tokens.device)
    return (
        Stage2ContrastiveLoss(
            loss=differentiable_zero,
            active_anchors=count_zero,
            positive_pairs=count_zero,
            negative_pairs=count_zero,
        ),
        Stage2RelativeRankingLoss(
            loss=differentiable_zero,
            active_anchors=count_zero,
            mean_nearest_action_distance=scalar_zero,
            mean_near_latent_distance=scalar_zero,
            mean_hard_negative_latent_distance=scalar_zero,
        ),
    )


def _zero_neighborhood_loss(task_tokens: Tensor) -> Stage2ActionNeighborhoodKLLoss:
    differentiable_zero = task_tokens.float().sum() * 0.0
    scalar_zero = differentiable_zero.detach()
    count_zero = torch.zeros((), dtype=torch.int64, device=task_tokens.device)
    return Stage2ActionNeighborhoodKLLoss(
        loss=differentiable_zero,
        active_anchors=count_zero,
        target_effective_neighbours=scalar_zero,
        predicted_effective_neighbours=scalar_zero,
        target_top1_probability=scalar_zero,
        predicted_top1_probability=scalar_zero,
    )


def _all_gather_detached(value: Tensor) -> Tensor:
    detached = value.detach().contiguous()
    if not dist.is_available() or not dist.is_initialized():
        return detached
    gathered = [torch.empty_like(detached) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, detached)
    return torch.cat(gathered, dim=0)


def _all_gather_with_grad(value: Tensor) -> Tensor:
    """Gather candidates while retaining the exact distributed KL gradient."""

    if not dist.is_available() or not dist.is_initialized():
        return value
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(value.contiguous()), dim=0)


def _tensor(batch: dict[str, Tensor | str], name: str) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"Stage 2 batch {name!r} must be a tensor")
    return value
