# SPDX-FileCopyrightText: Copyright (c) 2026 Z-Trans CORPORATION.
# SPDX-License-Identifier: OpenMDW-1.1

"""Persistent Interaction Memory (PIM) for causal repeated-attempt adaptation.

The runtime store is deliberately non-parametric: it accepts only completed
interaction effects, merges similar phase/effect prototypes, and retrieves by
the *current* phase.  ``CausalPromptEncoder`` (the paper's ``F_mem``) is the
small trainable module that turns task, phase, BIT, and retrieved PIM tensors
into a single residual context for the frozen policy prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PersistentInteractionMemoryConfig:
    phase_dim: int = 128
    effect_dim: int = 128
    capacity: int = 64
    top_k: int = 4
    merge_threshold: float = 0.85
    phase_merge_weight: float = 0.5
    effect_merge_weight: float = 0.5

    def __post_init__(self) -> None:
        if min(self.phase_dim, self.effect_dim, self.capacity, self.top_k) < 1:
            raise ValueError("PIM dimensions, capacity, and top_k must be positive")
        if self.top_k > self.capacity:
            raise ValueError("PIM top_k cannot exceed capacity")
        if not -1.0 <= self.merge_threshold <= 1.0:
            raise ValueError("PIM merge_threshold must lie in [-1, 1]")
        if self.phase_merge_weight < 0.0 or self.effect_merge_weight < 0.0:
            raise ValueError("PIM merge weights must be non-negative")
        if self.phase_merge_weight + self.effect_merge_weight <= 0.0:
            raise ValueError("At least one PIM merge weight must be positive")


@dataclass
class PIMMemoryEntry:
    """One merged phase/effect prototype in an episode-scoped PIM."""

    task_cluster: str
    phase: Tensor
    effect: Tensor
    merge_count: int = 1
    first_attempt_id: int = 0
    last_attempt_id: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def cpu(self) -> "PIMMemoryEntry":
        return PIMMemoryEntry(
            task_cluster=self.task_cluster,
            phase=self.phase.detach().float().cpu().contiguous(),
            effect=self.effect.detach().float().cpu().contiguous(),
            merge_count=int(self.merge_count),
            first_attempt_id=int(self.first_attempt_id),
            last_attempt_id=int(self.last_attempt_id),
            metadata=dict(self.metadata),
        )


class PersistentInteractionMemory:
    """Episode-scoped PIM retained across attempts and cleared across episodes."""

    def __init__(self, config: PersistentInteractionMemoryConfig | None = None) -> None:
        self.config = config or PersistentInteractionMemoryConfig()
        self._task_cluster: str | None = None
        self._attempt_id: int | None = None
        self._entries: list[PIMMemoryEntry] = []

    @property
    def task_cluster(self) -> str | None:
        return self._task_cluster

    @property
    def attempt_id(self) -> int | None:
        return self._attempt_id

    @property
    def entries(self) -> tuple[PIMMemoryEntry, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def reset_episode(self, task_cluster: str, *, attempt_id: int = 0) -> None:
        if not task_cluster:
            raise ValueError("PIM task_cluster cannot be empty")
        if attempt_id != 0:
            raise ValueError("A new PIM episode must begin at attempt_id=0")
        self._task_cluster = str(task_cluster)
        self._attempt_id = 0
        self._entries.clear()

    def begin_attempt(self, attempt_id: int) -> None:
        """Advance the attempt boundary without clearing persistent entries."""
        if self._task_cluster is None or self._attempt_id is None:
            raise RuntimeError("reset_episode() is required before begin_attempt()")
        attempt_id = int(attempt_id)
        if attempt_id not in {self._attempt_id, self._attempt_id + 1}:
            raise ValueError(
                f"PIM attempt_id must stay constant or increment by one: "
                f"active={self._attempt_id}, requested={attempt_id}"
            )
        self._attempt_id = attempt_id

    def _validate(self, phase: Tensor, effect: Tensor, task_cluster: str) -> tuple[Tensor, Tensor]:
        if self._task_cluster is None or self._attempt_id is None:
            raise RuntimeError("reset_episode() is required before PIM writes")
        if str(task_cluster) != self._task_cluster:
            raise ValueError(
                f"PIM task mismatch: active={self._task_cluster!r}, write={task_cluster!r}"
            )
        if phase.shape != (self.config.phase_dim,) or effect.shape != (self.config.effect_dim,):
            raise ValueError("PIM phase/effect shape mismatch")
        if not torch.isfinite(phase).all() or not torch.isfinite(effect).all():
            raise ValueError("PIM refuses non-finite phase/effect values")
        return (
            F.normalize(phase.detach().float().cpu(), dim=0),
            F.normalize(effect.detach().float().cpu(), dim=0),
        )

    def _merge_scores(self, phase: Tensor, effect: Tensor) -> Tensor:
        if not self._entries:
            return torch.empty(0, dtype=torch.float32)
        phases = torch.stack([entry.phase for entry in self._entries])
        effects = torch.stack([entry.effect for entry in self._entries])
        phase_scores = F.normalize(phases, dim=-1) @ phase
        effect_scores = F.normalize(effects, dim=-1) @ effect
        total = self.config.phase_merge_weight + self.config.effect_merge_weight
        return (
            self.config.phase_merge_weight * phase_scores
            + self.config.effect_merge_weight * effect_scores
        ) / total

    def append_completed(
        self,
        *,
        task_cluster: str,
        phase: Tensor,
        effect: Tensor,
        attempt_id: int,
        metadata: dict[str, object] | None = None,
    ) -> tuple[int, bool]:
        """Write one completed interaction, returning ``(index, merged)``."""
        phase, effect = self._validate(phase, effect, task_cluster)
        self.begin_attempt(int(attempt_id))
        scores = self._merge_scores(phase, effect)
        if scores.numel():
            best_score, best_index = scores.max(dim=0)
            if float(best_score) >= self.config.merge_threshold:
                index = int(best_index)
                entry = self._entries[index]
                old_count = int(entry.merge_count)
                new_count = old_count + 1
                entry.phase = F.normalize((entry.phase * old_count + phase) / new_count, dim=0)
                entry.effect = F.normalize((entry.effect * old_count + effect) / new_count, dim=0)
                entry.merge_count = new_count
                entry.last_attempt_id = int(attempt_id)
                entry.metadata.update(dict(metadata or {}))
                entry.metadata["last_merge_score"] = float(best_score)
                return index, True

        if len(self._entries) >= self.config.capacity:
            # Deterministic bounded storage: evict the least consolidated, then
            # oldest prototype.  This preserves frequently reused interaction
            # schemas without making the result dependent on object identity.
            evict = min(
                range(len(self._entries)),
                key=lambda i: (self._entries[i].merge_count, self._entries[i].last_attempt_id, i),
            )
            self._entries.pop(evict)
        self._entries.append(
            PIMMemoryEntry(
                task_cluster=str(task_cluster),
                phase=phase,
                effect=effect,
                merge_count=1,
                first_attempt_id=int(attempt_id),
                last_attempt_id=int(attempt_id),
                metadata=dict(metadata or {}),
            )
        )
        return len(self._entries) - 1, False

    def query_phase(
        self, phase: Tensor, *, top_k: int | None = None
    ) -> tuple[list[PIMMemoryEntry], Tensor]:
        """Retrieve PIM evidence using only the current phase, as in the paper."""
        if phase.shape != (self.config.phase_dim,):
            raise ValueError("PIM query phase shape mismatch")
        if not torch.isfinite(phase).all():
            raise ValueError("PIM refuses a non-finite query phase")
        requested_k = self.config.top_k if top_k is None else int(top_k)
        if requested_k < 1 or requested_k > self.config.capacity:
            raise ValueError("PIM query top_k is outside the configured capacity")
        if not self._entries:
            return [], torch.empty(0, dtype=torch.float32)
        query = F.normalize(phase.detach().float().cpu(), dim=0)
        keys = torch.stack([F.normalize(entry.phase, dim=0) for entry in self._entries])
        k = min(requested_k, len(self._entries))
        scores, indices = torch.topk(keys @ query, k=k, largest=True, sorted=True)
        return [self._entries[int(index)] for index in indices], scores

    def query_tensors(
        self, phase: Tensor, *, top_k: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return padded ``phase/effect/valid/scores`` tensors for model input."""
        k = self.config.top_k if top_k is None else int(top_k)
        if k < 1 or k > self.config.capacity:
            raise ValueError("PIM query top_k is outside the configured capacity")
        entries, scores = self.query_phase(phase, top_k=k)
        phases = torch.zeros((k, self.config.phase_dim), dtype=torch.float32)
        effects = torch.zeros((k, self.config.effect_dim), dtype=torch.float32)
        valid = torch.zeros(k, dtype=torch.bool)
        padded_scores = torch.full((k,), -torch.inf, dtype=torch.float32)
        for index, entry in enumerate(entries):
            phases[index] = entry.phase
            effects[index] = entry.effect
            valid[index] = True
            padded_scores[index] = scores[index]
        return phases, effects, valid, padded_scores


class PhaseConditionedPIMRetrieval:
    """Paper-aligned phase-only retrieval interface over one episode PIM."""

    def __init__(self, memory: PersistentInteractionMemory) -> None:
        self.memory = memory

    def __call__(
        self, phase: Tensor, *, top_k: int | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.memory.query_tensors(phase, top_k=top_k)


@dataclass(frozen=True)
class CausalPromptConfig:
    global_dim: int = 256
    phase_dim: int = 128
    effect_dim: int = 128
    brief_length: int = 4
    persistent_length: int = 4
    hidden_dim: int = 256
    num_heads: int = 4

    def __post_init__(self) -> None:
        dimensions = (
            self.global_dim,
            self.phase_dim,
            self.effect_dim,
            self.brief_length,
            self.persistent_length,
            self.hidden_dim,
            self.num_heads,
        )
        if min(dimensions) < 1:
            raise ValueError("PIM prompt dimensions and lengths must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("PIM prompt hidden_dim must be divisible by num_heads")


class CausalPromptEncoder(nn.Module):
    """Fuse task/global, current phase, BIT, and retrieved PIM evidence."""

    def __init__(self, config: CausalPromptConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalPromptConfig()
        cfg = self.config
        self.global_project = nn.Sequential(nn.LayerNorm(cfg.global_dim), nn.Linear(cfg.global_dim, cfg.hidden_dim))
        self.phase_project = nn.Sequential(nn.LayerNorm(cfg.phase_dim), nn.Linear(cfg.phase_dim, cfg.hidden_dim))
        self.effect_project = nn.Sequential(nn.LayerNorm(cfg.effect_dim), nn.Linear(cfg.effect_dim, cfg.hidden_dim))
        self.brief_position = nn.Parameter(torch.empty(1, cfg.brief_length, cfg.hidden_dim))
        self.persistent_position = nn.Parameter(torch.empty(1, cfg.persistent_length, cfg.hidden_dim))
        self.bos_brief = nn.Parameter(torch.empty(1, 1, cfg.hidden_dim))
        self.bos_persistent = nn.Parameter(torch.empty(1, 1, cfg.hidden_dim))
        self.brief_attention = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.persistent_attention = nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True)
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * cfg.hidden_dim),
            nn.Linear(3 * cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.brief_attention._reset_parameters()
        self.persistent_attention._reset_parameters()
        nn.init.normal_(self.brief_position, std=0.02)
        nn.init.normal_(self.persistent_position, std=0.02)
        nn.init.normal_(self.bos_brief, std=0.02)
        nn.init.normal_(self.bos_persistent, std=0.02)

    def forward(
        self,
        z_global: Tensor,
        z_phase: Tensor,
        brief_effect: Tensor,
        brief_valid: Tensor,
        persistent_phase: Tensor,
        persistent_effect: Tensor,
        persistent_valid: Tensor,
    ) -> Tensor:
        cfg = self.config
        batch = z_global.shape[0]
        expected = {
            "z_global": (batch, cfg.global_dim),
            "z_phase": (batch, cfg.phase_dim),
            "brief_effect": (batch, cfg.brief_length, cfg.effect_dim),
            "brief_valid": (batch, cfg.brief_length),
            "persistent_phase": (batch, cfg.persistent_length, cfg.phase_dim),
            "persistent_effect": (batch, cfg.persistent_length, cfg.effect_dim),
            "persistent_valid": (batch, cfg.persistent_length),
        }
        values = {
            "z_global": z_global,
            "z_phase": z_phase,
            "brief_effect": brief_effect,
            "brief_valid": brief_valid,
            "persistent_phase": persistent_phase,
            "persistent_effect": persistent_effect,
            "persistent_valid": persistent_valid,
        }
        for name, shape in expected.items():
            if tuple(values[name].shape) != shape:
                raise ValueError(f"Expected {name} {shape}, got {tuple(values[name].shape)}")

        query = self.global_project(z_global) + self.phase_project(z_phase)
        brief_valid = brief_valid.to(torch.bool)
        persistent_valid = persistent_valid.to(torch.bool)
        brief = self.effect_project(brief_effect) + self.brief_position
        brief = torch.where(
            brief_valid.unsqueeze(-1), brief, self.bos_brief.expand(batch, cfg.brief_length, -1)
        )
        brief_padding = ~brief_valid
        brief_empty = ~brief_valid.any(dim=-1)
        brief_padding = brief_padding.clone()
        brief_padding[brief_empty, 0] = False
        brief_context, _ = self.brief_attention(
            query.unsqueeze(1), brief, brief, key_padding_mask=brief_padding, need_weights=False
        )
        persistent = (
            self.phase_project(persistent_phase)
            + self.effect_project(persistent_effect)
            + self.persistent_position
        )
        persistent = torch.where(
            persistent_valid.unsqueeze(-1),
            persistent,
            self.bos_persistent.expand(batch, cfg.persistent_length, -1),
        )
        persistent_padding = ~persistent_valid
        persistent_empty = ~persistent_valid.any(dim=-1)
        persistent_padding = persistent_padding.clone()
        persistent_padding[persistent_empty, 0] = False
        persistent_context, _ = self.persistent_attention(
            query.unsqueeze(1),
            persistent,
            persistent,
            key_padding_mask=persistent_padding,
            need_weights=False,
        )
        return self.fusion(
            torch.cat((query, brief_context.squeeze(1), persistent_context.squeeze(1)), dim=-1)
        )


def inject_causal_prompt(
    base_prefix: Tensor,
    causal_prompt: Tensor,
    gate: Tensor,
    persistent_valid: Tensor,
) -> Tensor:
    """Inject the paper's causal prompt with an exact ``gate == 0`` bypass."""
    if base_prefix.shape != causal_prompt.shape:
        raise ValueError("Causal prompt must match the existing task-context prefix shape")
    if persistent_valid.ndim != 2 or persistent_valid.shape[0] != base_prefix.shape[0]:
        raise ValueError("persistent_valid must be [B,K]")
    has_pim = persistent_valid.any(dim=-1, keepdim=True).to(causal_prompt.dtype)
    return base_prefix + torch.tanh(gate).to(causal_prompt.dtype) * causal_prompt * has_pim
