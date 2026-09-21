"""Stage 2 visual transition encoders and environment teachers."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn

from zeva_action_encoder.models.stage1 import Stage1Model
from zeva_action_encoder.models.univla_blocks import SpatialTransformer
from zeva_action_encoder.models.univla_blocks import SpatioTemporalTransformer


@dataclass(frozen=True)
class Stage2ModelConfig:
    """Architecture owned by a Stage 2 checkpoint.

    Task-token count is configurable between experiments but fixed for one
    checkpoint. Stage 2 deliberately has no VQ/codebook. The legacy mode uses
    an external frozen teacher; the joint-query mode copies Stage 1's frozen
    environment queries and trainable posterior into the Stage 2 encoder.
    ``joint_query_ema_v1`` keeps the same checkpoint-owned architecture while
    allowing reconstruction gradients through the student environment branch.
    """

    visual_dim: int
    model_dim: int = 768
    latent_dim: int = 128
    num_environment_tokens: int = 4
    num_task_tokens: int = 4
    encoder_blocks: int = 12
    decoder_blocks: int = 12
    num_heads: int = 12
    dropout: float = 0.0
    environment_mode: str = "external_teacher_v1"
    normalize_environment_tokens: bool = False

    def __post_init__(self) -> None:
        positive = {
            "visual_dim": self.visual_dim,
            "model_dim": self.model_dim,
            "latent_dim": self.latent_dim,
            "num_environment_tokens": self.num_environment_tokens,
            "num_task_tokens": self.num_task_tokens,
            "encoder_blocks": self.encoder_blocks,
            "decoder_blocks": self.decoder_blocks,
            "num_heads": self.num_heads,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.environment_mode not in {
            "external_teacher_v1",
            "joint_query_v1",
            "joint_query_ema_v1",
        }:
            raise ValueError("unsupported Stage 2 environment mode")


@dataclass
class Stage2Output:
    reconstructed_future: Tensor
    environment_tokens: Tensor
    task_tokens: Tensor


@dataclass
class Stage2TaskRelations:
    """Task latents needed by additive and reversal constraints."""

    ab: Tensor
    bc: Tensor
    ac: Tensor
    ba: Tensor


@dataclass
class Stage2EnvironmentRelations:
    """Student environment latents aligned with the three reconstructed transitions."""

    ab: Tensor
    bc: Tensor
    ac: Tensor


@dataclass
class Stage2TripletOutput:
    """Three reconstructions plus all task relations from one DDP forward."""

    reconstructed_ab: Tensor
    reconstructed_bc: Tensor
    reconstructed_ac: Tensor
    environments: Stage2EnvironmentRelations
    relations: Stage2TaskRelations


class FrozenStage1EnvironmentTeacher(nn.Module):
    """Expose only Stage 1's bottleneck while keeping every teacher weight fixed."""

    def __init__(self, model: Stage1Model) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> FrozenStage1EnvironmentTeacher:
        # A parent training module may call ``train()`` recursively. The
        # teacher must remain deterministic and frozen regardless.
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        start_features: Tensor,
        future_features: Tensor,
        action_chunk: Tensor,
        action_mask: Tensor | None = None,
        action_dimension_mask: Tensor | None = None,
    ) -> Tensor:
        environment = self.model.encode_environment(
            start_features=start_features,
            future_features=future_features,
            action_chunk=action_chunk,
            action_mask=action_mask,
            action_dimension_mask=action_dimension_mask,
        )
        if self.model.config.normalize_bottleneck:
            environment = self.model._rms_normalize_tokens(environment)
        if self.model.vector_quantizer is not None:
            environment, _, _, _ = self.model.vector_quantizer(environment)
        return environment.detach()


class EMAStage2EnvironmentTeacher(nn.Module):
    """Momentum copy of only the Stage 2 joint encoder, exposing env tokens.

    Environment tokens are sample-dependent and therefore cannot themselves be
    averaged across unrelated minibatches.  Instead, this module maintains an
    exponential moving average of the parameters that produce those tokens.
    Decoder modules are removed from the copy because the teacher never uses
    them.  The joint task queries remain: their self-attention context affects
    the environment output even though the teacher task output is discarded.
    """

    _DECODER_MODULES = (
        "decoder_visual_projection",
        "environment_up",
        "task_up",
        "decoder_modality_embedding",
        "decoder",
    )

    def __init__(self, student: Stage2Model) -> None:
        super().__init__()
        if student.config.environment_mode != "joint_query_ema_v1":
            raise ValueError("EMA environment teacher requires joint_query_ema_v1")
        self.model = copy.deepcopy(student)
        for name in self._DECODER_MODULES:
            delattr(self.model, name)
        self.model.task_encoder.activation_checkpointing = False
        self.model.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool = True) -> EMAStage2EnvironmentTeacher:
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, start_features: Tensor, future_features: Tensor) -> Tensor:
        environment, _ = self.model._encode_latents(start_features, future_features)
        if environment is None:
            raise RuntimeError("EMA joint encoder did not produce environment tokens")
        return environment.detach()

    @torch.no_grad()
    def update(self, student: Stage2Model, *, momentum: float) -> None:
        """Move teacher encoder parameters and buffers toward the student."""

        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must be in [0, 1)")
        if student.config != self.model.config:
            raise ValueError("EMA teacher and student model configs differ")

        student_parameters = dict(student.named_parameters())
        for name, teacher_parameter in self.model.named_parameters():
            source = student_parameters.get(name)
            if source is None or source.shape != teacher_parameter.shape:
                raise ValueError(f"EMA teacher parameter {name!r} is not aligned with the student")
            teacher_parameter.mul_(momentum).add_(source.detach(), alpha=1.0 - momentum)

        student_buffers = dict(student.named_buffers())
        for name, teacher_buffer in self.model.named_buffers():
            source = student_buffers.get(name)
            if source is None or source.shape != teacher_buffer.shape:
                raise ValueError(f"EMA teacher buffer {name!r} is not aligned with the student")
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(momentum).add_(source.detach(), alpha=1.0 - momentum)
            else:
                teacher_buffer.copy_(source)


class Stage2Model(nn.Module):
    """A visual task encoder plus a trainable Stage 1 decoder.

    Legacy checkpoints receive environment tokens from an action-conditioned
    Stage 1 teacher. ``joint_query_v1`` instead copies Stage 1's posterior and
    frozen environment queries, then infers environment and task latents from
    the visual pair alone.
    """

    TASK_QUERY = 0
    TASK_VISUAL = 1

    ENVIRONMENT = 0
    DECODER_VISUAL = 1
    TASK = 2

    def __init__(self, config: Stage2ModelConfig) -> None:
        super().__init__()
        self.config = config

        joint_environment = config.environment_mode in {"joint_query_v1", "joint_query_ema_v1"}
        self.task_visual_projection = nn.Sequential(
            nn.LayerNorm(config.visual_dim),
            nn.Linear(config.visual_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )
        self.task_encoder_modality_embedding = nn.Embedding(
            3 if joint_environment else 2,
            config.model_dim,
        )
        if joint_environment:
            self.environment_queries = nn.Parameter(
                torch.empty(1, 1, config.num_environment_tokens, config.model_dim),
                requires_grad=False,
            )
            nn.init.uniform_(self.environment_queries, -1.0, 1.0)
        self.task_queries = nn.Parameter(torch.empty(1, 1, config.num_task_tokens, config.model_dim))
        nn.init.uniform_(self.task_queries, -1.0, 1.0)
        self.task_encoder = SpatioTemporalTransformer(
            model_dim=config.model_dim,
            num_blocks=config.encoder_blocks,
            num_heads=config.num_heads,
            dropout=config.dropout,
            causal_temporal=True,
        )
        if joint_environment:
            self.to_environment_latent = nn.Linear(config.model_dim, config.latent_dim)
        self.to_task_latent = nn.Linear(config.model_dim, config.latent_dim)

        # These modules form the decoder side. Their compatible weights are
        # copied from Stage 1 before training; task_up is necessarily new.
        self.decoder_visual_projection = nn.Sequential(
            nn.LayerNorm(config.visual_dim),
            nn.Linear(config.visual_dim, config.model_dim),
            nn.LayerNorm(config.model_dim),
        )
        self.environment_up = nn.Linear(config.latent_dim, config.model_dim)
        self.task_up = nn.Linear(config.latent_dim, config.model_dim)
        self.decoder_modality_embedding = nn.Embedding(3, config.model_dim)
        self.decoder = SpatialTransformer(
            model_dim=config.model_dim,
            out_dim=config.visual_dim,
            num_blocks=config.decoder_blocks,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

    def configure_execution(
        self,
        *,
        attention_backend: str,
        activation_checkpointing: bool,
    ) -> None:
        self.task_encoder.set_attention_backend(attention_backend)
        self.decoder.set_attention_backend(attention_backend)
        self.task_encoder.activation_checkpointing = activation_checkpointing
        self.decoder.activation_checkpointing = activation_checkpointing

    @staticmethod
    def _embedding(embedding: nn.Embedding, index: int, reference: Tensor) -> Tensor:
        index_tensor = torch.tensor(index, device=reference.device)
        return embedding(index_tensor).to(reference.dtype)

    def encode_task(self, start_features: Tensor, future_features: Tensor) -> Tensor:
        """Encode one visual transition and return only its task branch."""

        _, task = self._encode_latents(start_features, future_features)
        return task

    def encode_transition(
        self,
        start_features: Tensor,
        future_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return sample-dependent environment and task tokens from two frames."""

        environment, task = self._encode_latents(start_features, future_features)
        if environment is None:
            raise RuntimeError("encode_transition requires a joint-query environment mode")
        return environment, task

    def _encode_latents(
        self,
        start_features: Tensor,
        future_features: Tensor,
    ) -> tuple[Tensor | None, Tensor]:
        """Encode one or more paired transitions in the configured architecture."""

        self._validate_visual_pair(start_features, future_features)
        batch_size, _, _ = start_features.shape
        visual = torch.stack([start_features, future_features], dim=1)
        visual = self.task_visual_projection(visual)
        if self.config.environment_mode == "external_teacher_v1":
            visual = visual + self._embedding(
                self.task_encoder_modality_embedding,
                self.TASK_VISUAL,
                visual,
            )
            queries = self.task_queries.expand(batch_size, 2, -1, -1)
            queries = queries + self._embedding(
                self.task_encoder_modality_embedding,
                self.TASK_QUERY,
                queries,
            )
            encoded = self.task_encoder(torch.cat([queries, visual], dim=2))
            future_queries = encoded[:, 1, : self.config.num_task_tokens]
            return None, self.to_task_latent(future_queries)

        visual = visual + self._embedding(
            self.task_encoder_modality_embedding,
            self.DECODER_VISUAL,
            visual,
        )
        environment_queries = self.environment_queries.expand(batch_size, 2, -1, -1)
        environment_queries = environment_queries + self._embedding(
            self.task_encoder_modality_embedding,
            self.ENVIRONMENT,
            environment_queries,
        )
        task_queries = self.task_queries.expand(batch_size, 2, -1, -1)
        task_queries = task_queries + self._embedding(
            self.task_encoder_modality_embedding,
            self.TASK,
            task_queries,
        )
        encoded = self.task_encoder(torch.cat([environment_queries, visual, task_queries], dim=2))
        visual_length = visual.shape[2]
        future_environment = encoded[:, 1, : self.config.num_environment_tokens]
        task_start = self.config.num_environment_tokens + visual_length
        future_task = encoded[:, 1, task_start : task_start + self.config.num_task_tokens]
        environment = self.to_environment_latent(future_environment)
        if self.config.normalize_environment_tokens:
            environment = self._rms_normalize_tokens(environment)
        return environment, self.to_task_latent(future_task)

    def encode_task_relations(
        self,
        frame_a: Tensor,
        frame_b: Tensor,
        frame_c: Tensor,
    ) -> Stage2TaskRelations:
        """Encode AB, BC, AC, and BA together so one batch owns each relation."""

        self._validate_visual_pair(frame_a, frame_b)
        self._validate_visual_pair(frame_a, frame_c)
        batch_size = frame_a.shape[0]
        starts = torch.cat([frame_a, frame_b, frame_a, frame_b], dim=0)
        futures = torch.cat([frame_b, frame_c, frame_c, frame_a], dim=0)
        _, tasks = self._encode_latents(starts, futures)
        ab, bc, ac, ba = tasks.split(batch_size, dim=0)
        return Stage2TaskRelations(ab=ab, bc=bc, ac=ac, ba=ba)

    def decode_future(
        self,
        start_features: Tensor,
        environment_tokens: Tensor,
        task_tokens: Tensor,
    ) -> Tensor:
        """Reconstruct the future from the first frame and the two latent branches."""

        self._validate_visual_pair(start_features, None)
        batch_size, visual_length, _ = start_features.shape
        expected_environment = (
            batch_size,
            self.config.num_environment_tokens,
            self.config.latent_dim,
        )
        expected_task = (
            batch_size,
            self.config.num_task_tokens,
            self.config.latent_dim,
        )
        if tuple(environment_tokens.shape) != expected_environment:
            raise ValueError(
                f"environment_tokens must have shape {expected_environment}, got {tuple(environment_tokens.shape)}"
            )
        if tuple(task_tokens.shape) != expected_task:
            raise ValueError(f"task_tokens must have shape {expected_task}, got {tuple(task_tokens.shape)}")

        environment = self.environment_up(environment_tokens)
        environment = environment + self._embedding(
            self.decoder_modality_embedding,
            self.ENVIRONMENT,
            environment,
        )
        visual = self.decoder_visual_projection(start_features)
        visual = visual + self._embedding(
            self.decoder_modality_embedding,
            self.DECODER_VISUAL,
            visual,
        )
        task = self.task_up(task_tokens)
        task = task + self._embedding(
            self.decoder_modality_embedding,
            self.TASK,
            task,
        )
        # Preserve Stage 1's [environment, visual, action-like] token order so
        # copied decoder position-dependent behavior starts from a valid state.
        decoded = self.decoder(torch.cat([environment, visual, task], dim=1))
        visual_start = self.config.num_environment_tokens
        return decoded[:, visual_start : visual_start + visual_length]

    def forward(
        self,
        start_features: Tensor,
        future_features: Tensor,
        environment_tokens: Tensor | None = None,
        *,
        frame_c: Tensor | None = None,
        environment_bc: Tensor | None = None,
        environment_ac: Tensor | None = None,
    ) -> Stage2Output | Stage2TripletOutput:
        if frame_c is not None:
            return self.forward_triplet(
                start_features,
                future_features,
                frame_c,
                environment_tokens,
                environment_bc,
                environment_ac,
            )
        if environment_bc is not None or environment_ac is not None:
            raise ValueError("environment_bc/environment_ac require frame_c")
        inferred_environment, task_tokens = self._encode_latents(start_features, future_features)
        if self.config.environment_mode in {"joint_query_v1", "joint_query_ema_v1"}:
            if environment_tokens is not None:
                raise ValueError("joint-query Stage 2 must not receive external environment tokens")
            assert inferred_environment is not None
            environment_tokens = inferred_environment
        elif environment_tokens is None:
            raise ValueError("external-teacher Stage 2 requires environment_tokens")
        decoder_environment = (
            environment_tokens.detach() if self.config.environment_mode == "joint_query_v1" else environment_tokens
        )
        reconstructed = self.decode_future(start_features, decoder_environment, task_tokens)
        return Stage2Output(
            reconstructed_future=reconstructed,
            environment_tokens=environment_tokens,
            task_tokens=task_tokens,
        )

    def forward_triplet(
        self,
        frame_a: Tensor,
        frame_b: Tensor,
        frame_c: Tensor,
        environment_ab: Tensor | None,
        environment_bc: Tensor | None,
        environment_ac: Tensor | None,
    ) -> Stage2TripletOutput:
        """Run the complete Stage 2 triplet graph through one DDP forward."""

        batch_size = frame_a.shape[0]
        relation_starts = torch.cat([frame_a, frame_b, frame_a, frame_b], dim=0)
        relation_futures = torch.cat([frame_b, frame_c, frame_c, frame_a], dim=0)
        inferred_environments, relation_tasks = self._encode_latents(
            relation_starts,
            relation_futures,
        )
        ab, bc, ac, ba = relation_tasks.split(batch_size, dim=0)
        relations = Stage2TaskRelations(ab=ab, bc=bc, ac=ac, ba=ba)
        starts = torch.cat([frame_a, frame_b, frame_a], dim=0)
        tasks = torch.cat([ab, bc, ac], dim=0)
        if self.config.environment_mode in {"joint_query_v1", "joint_query_ema_v1"}:
            if any(value is not None for value in (environment_ab, environment_bc, environment_ac)):
                raise ValueError("joint-query Stage 2 must not receive external environments")
            assert inferred_environments is not None
            environment_ab, environment_bc, environment_ac, _ = inferred_environments.split(
                batch_size,
                dim=0,
            )
        elif any(value is None for value in (environment_ab, environment_bc, environment_ac)):
            raise ValueError("external-teacher triplet forward requires three environments")
        assert environment_ab is not None and environment_bc is not None and environment_ac is not None
        environment_relations = Stage2EnvironmentRelations(
            ab=environment_ab,
            bc=environment_bc,
            ac=environment_ac,
        )
        environments = torch.cat([environment_ab, environment_bc, environment_ac], dim=0)
        decoder_environments = (
            environments.detach() if self.config.environment_mode == "joint_query_v1" else environments
        )
        reconstructed = self.decode_future(starts, decoder_environments, tasks)
        reconstructed_ab, reconstructed_bc, reconstructed_ac = reconstructed.split(batch_size, dim=0)
        return Stage2TripletOutput(
            reconstructed_ab=reconstructed_ab,
            reconstructed_bc=reconstructed_bc,
            reconstructed_ac=reconstructed_ac,
            environments=environment_relations,
            relations=relations,
        )

    @torch.no_grad()
    def initialize_encoder_from_stage1(self, stage1: Stage1Model) -> None:
        """Copy Stage 1's environment-reading posterior into the joint encoder."""

        if self.config.environment_mode not in {"joint_query_v1", "joint_query_ema_v1"}:
            raise ValueError("encoder initialization requires a joint-query environment mode")
        expected = {
            "visual_dim": (self.config.visual_dim, stage1.config.visual_dim),
            "model_dim": (self.config.model_dim, stage1.config.model_dim),
            "latent_dim": (self.config.latent_dim, stage1.config.latent_dim),
            "num_environment_tokens": (
                self.config.num_environment_tokens,
                stage1.config.num_environment_tokens,
            ),
            "encoder_blocks": (self.config.encoder_blocks, stage1.config.encoder_blocks),
            "num_heads": (self.config.num_heads, stage1.config.num_heads),
            "normalize_environment_tokens": (
                self.config.normalize_environment_tokens,
                stage1.config.normalize_bottleneck,
            ),
        }
        mismatches = [name for name, (actual, source) in expected.items() if actual != source]
        if mismatches:
            details = ", ".join(
                f"{name}: stage2={expected[name][0]}, stage1={expected[name][1]}" for name in mismatches
            )
            raise ValueError(f"Stage 1 encoder is incompatible with Stage 2: {details}")
        if stage1.vector_quantizer is not None:
            raise ValueError("joint-query Stage 2 requires a continuous Stage 1 bottleneck")
        self.task_visual_projection.load_state_dict(stage1.visual_projection.state_dict())
        self.task_encoder_modality_embedding.load_state_dict(stage1.modality_embedding.state_dict())
        self.task_encoder.load_state_dict(stage1.posterior.state_dict())
        self.environment_queries.copy_(stage1.environment_queries)
        self.environment_queries.requires_grad_(False)
        self.to_environment_latent.load_state_dict(stage1.to_codebook.state_dict())

    @torch.no_grad()
    def initialize_decoder_from_stage1(self, stage1: Stage1Model) -> None:
        """Copy every compatible decoder-side Stage 1 weight.

        Stage 1's action projection is intentionally not copied: Stage 2 task
        latents are continuous 128D tokens rather than 44D action steps.
        """

        expected = {
            "visual_dim": (self.config.visual_dim, stage1.config.visual_dim),
            "model_dim": (self.config.model_dim, stage1.config.model_dim),
            "latent_dim": (self.config.latent_dim, stage1.config.latent_dim),
            "num_environment_tokens": (
                self.config.num_environment_tokens,
                stage1.config.num_environment_tokens,
            ),
            "decoder_blocks": (self.config.decoder_blocks, stage1.config.decoder_blocks),
            "num_heads": (self.config.num_heads, stage1.config.num_heads),
        }
        mismatches = [name for name, (actual, teacher) in expected.items() if actual != teacher]
        if mismatches:
            details = ", ".join(
                f"{name}: stage2={expected[name][0]}, stage1={expected[name][1]}" for name in mismatches
            )
            raise ValueError(f"Stage 1 decoder is incompatible with Stage 2: {details}")

        self.decoder_visual_projection.load_state_dict(stage1.visual_projection.state_dict())
        self.environment_up.load_state_dict(stage1.environment_up.state_dict())
        self.decoder.load_state_dict(stage1.decoder.state_dict())
        self.decoder_modality_embedding.weight.copy_(stage1.modality_embedding.weight)

    def initialize_encoder_from_environment_encoder(self, encoder: Stage1Model) -> None:
        """Initialize the task encoder from the released environment encoder."""

        self.initialize_encoder_from_stage1(encoder)

    def initialize_decoder_from_environment_encoder(self, encoder: Stage1Model) -> None:
        """Initialize the task decoder from the released environment encoder."""

        self.initialize_decoder_from_stage1(encoder)

    def _validate_visual_pair(
        self,
        start_features: Tensor,
        future_features: Tensor | None,
    ) -> None:
        if start_features.ndim != 3 or start_features.shape[-1] != self.config.visual_dim:
            raise ValueError(
                f"visual features must have shape [B, N, {self.config.visual_dim}], got {tuple(start_features.shape)}"
            )
        if future_features is not None and future_features.shape != start_features.shape:
            raise ValueError(
                "paired visual features must have the same shape, got "
                f"{tuple(start_features.shape)} and {tuple(future_features.shape)}"
            )

    @staticmethod
    def _rms_normalize_tokens(tokens: Tensor, epsilon: float = 1e-6) -> Tensor:
        inverse_rms = torch.rsqrt(tokens.float().square().mean(dim=-1, keepdim=True) + epsilon)
        return tokens * inverse_rms.to(dtype=tokens.dtype)
