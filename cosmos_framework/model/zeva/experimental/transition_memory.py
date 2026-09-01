"""Experimental transition memory retained for non-paper ablations.

The memory stores only completed 16-control transitions.  It is deliberately
independent of the Cosmos model so that real-robot code can run it in shadow
mode before enabling the extra conditioning prefix.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TransitionMemorySchema:
    """Dimensions and temporal contract used by one memory session."""

    version: str = "arx_transition_memory_v1"
    task_contract: str = "arx_task7_model_a_5task"
    phase_dim: int = 128
    visual_key_dim: int = 128
    effect_dim: int = 128
    action_dim: int = 14
    action_horizon: int = 16
    capacity: int = 64
    top_k: int = 4
    cte_hash: str = ""
    vae_temporal_hash: str = ""

    @property
    def hash(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class TransitionRecord:
    """One causally completed replan transition."""

    task_cluster: str
    phase: torch.Tensor
    visual_key: torch.Tensor
    effect_post: torch.Tensor
    executed_action: torch.Tensor
    next_visual_key: torch.Tensor
    next_phase: torch.Tensor
    latent_index: int
    schema_hash: str
    valid: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

    def cpu(self) -> "TransitionRecord":
        """Detach tensors so the session never retains a training graph."""
        return TransitionRecord(
            task_cluster=self.task_cluster,
            phase=self.phase.detach().float().cpu().contiguous(),
            visual_key=self.visual_key.detach().float().cpu().contiguous(),
            effect_post=self.effect_post.detach().float().cpu().contiguous(),
            executed_action=self.executed_action.detach().float().cpu().contiguous(),
            next_visual_key=self.next_visual_key.detach().float().cpu().contiguous(),
            next_phase=self.next_phase.detach().float().cpu().contiguous(),
            latent_index=int(self.latent_index),
            schema_hash=self.schema_hash,
            valid=bool(self.valid),
            metadata=dict(self.metadata),
        )


class TransitionMemory:
    """Bounded FIFO memory scoped to one task session."""

    def __init__(self, schema: TransitionMemorySchema | None = None) -> None:
        self.schema = schema or TransitionMemorySchema()
        self._task_cluster: str | None = None
        self._entries: deque[TransitionRecord] = deque(maxlen=self.schema.capacity)

    @property
    def task_cluster(self) -> str | None:
        return self._task_cluster

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[TransitionRecord, ...]:
        """Return a stable, read-only snapshot for audit/finalization."""
        return tuple(self._entries)

    def reset(self, task_cluster: str | None = None) -> None:
        self._entries.clear()
        self._task_cluster = task_cluster

    def _validate(self, transition: TransitionRecord) -> None:
        if transition.schema_hash != self.schema.hash:
            raise ValueError(
                "transition memory schema mismatch: "
                f"got {transition.schema_hash}, expected {self.schema.hash}"
            )
        if transition.phase.shape != (self.schema.phase_dim,):
            raise ValueError(f"phase must be [{self.schema.phase_dim}], got {tuple(transition.phase.shape)}")
        if transition.visual_key.shape != (self.schema.visual_key_dim,):
            raise ValueError("visual_key has an invalid shape")
        if transition.effect_post.shape != (self.schema.effect_dim,):
            raise ValueError("effect_post has an invalid shape")
        if transition.executed_action.shape != (self.schema.action_horizon, self.schema.action_dim):
            raise ValueError("executed_action must contain exactly one completed 16-step window")
        if transition.next_visual_key.shape != (self.schema.visual_key_dim,):
            raise ValueError("next_visual_key has an invalid shape")
        if transition.next_phase.shape != (self.schema.phase_dim,):
            raise ValueError("next_phase has an invalid shape")
        if not transition.valid:
            raise ValueError("invalid or incomplete transitions must not enter transition memory")

    def append(self, transition: TransitionRecord) -> None:
        """Append one completed transition, resetting on a task change."""
        self._validate(transition)
        if self._task_cluster is None:
            self._task_cluster = transition.task_cluster
        if transition.task_cluster != self._task_cluster:
            raise ValueError(
                f"task mismatch: session={self._task_cluster!r}, transition={transition.task_cluster!r}; reset first"
            )
        self._entries.append(transition.cpu())

    def annotate_attempt_outcome(
        self,
        attempt_id: int,
        *,
        outcome: str,
        termination_reason: str,
        total_steps: int,
        final_progress: float,
    ) -> int:
        """Attach a terminal label to every transition from one attempt."""
        if outcome not in {"success", "failure"}:
            raise ValueError(f"unsupported attempt outcome: {outcome!r}")
        if termination_reason not in {"success", "env_done", "max_steps", "client_error", "safety"}:
            raise ValueError(f"unsupported termination reason: {termination_reason!r}")
        if attempt_id < 0 or total_steps < 0:
            raise ValueError("attempt_id and total_steps must be non-negative")
        if not 0.0 <= float(final_progress) <= 1.0:
            raise ValueError("final_progress must be in [0,1]")
        updated = 0
        for entry in self._entries:
            if int(entry.metadata.get("attempt_id", -1)) != attempt_id:
                continue
            entry.metadata.update(
                {
                    "terminal_outcome": outcome,
                    "termination_reason": termination_reason,
                    "total_steps": int(total_steps),
                    "final_progress": float(final_progress),
                    "progress_source": "success_only",
                }
            )
            updated += 1
        return updated

    def query(
        self,
        phase: torch.Tensor,
        visual_key: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> tuple[list[TransitionRecord], torch.Tensor]:
        """Return cosine-ranked history entries and scores."""
        if phase.shape != (self.schema.phase_dim,) or visual_key.shape != (self.schema.visual_key_dim,):
            raise ValueError("query phase/visual_key shape mismatch")
        if not self._entries:
            return [], torch.empty(0, dtype=torch.float32)
        k = min(int(top_k or self.schema.top_k), len(self._entries))
        q = F.normalize(torch.cat((phase.float(), visual_key.float())), dim=0)
        keys = torch.stack([
            F.normalize(torch.cat((entry.phase, entry.visual_key)), dim=0)
            for entry in self._entries
        ])
        scores = keys @ q
        values, indices = torch.topk(scores, k=k, largest=True, sorted=True)
        entries = [list(self._entries)[int(index)] for index in indices]
        return entries, values


class TransitionMemoryEncoder(nn.Module):
    """Encode top-k transitions into one 256-D causal context token."""

    def __init__(self, schema: TransitionMemorySchema | None = None, hidden_dim: int = 256) -> None:
        super().__init__()
        self.schema = schema or TransitionMemorySchema()
        self.hidden_dim = hidden_dim
        value_dim = (
            self.schema.phase_dim
            + self.schema.visual_key_dim
            + self.schema.effect_dim
            + self.schema.action_horizon * self.schema.action_dim
            + self.schema.visual_key_dim
            + self.schema.phase_dim
        )
        query_dim = self.schema.phase_dim + self.schema.visual_key_dim
        self.query_proj = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.key_proj = nn.Sequential(nn.Linear(query_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.value_proj = nn.Sequential(nn.Linear(value_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.bos_online = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.bos_online, std=0.02)

    def forward(
        self,
        phase: torch.Tensor,
        visual_key: torch.Tensor,
        entries: Iterable[TransitionRecord] | None = None,
    ) -> torch.Tensor:
        """Return ``[B, hidden_dim]`` or ``[hidden_dim]`` for one query."""
        if phase.ndim == 1:
            phase = phase.unsqueeze(0)
        if visual_key.ndim == 1:
            visual_key = visual_key.unsqueeze(0)
        if phase.shape[0] != 1 or visual_key.shape[0] != 1:
            raise ValueError("TransitionMemoryEncoder currently accepts one query at a time")
        device = phase.device
        query = torch.cat((phase.float(), visual_key.float()), dim=-1)
        q = self.query_proj(query)
        items = list(entries or [])[: self.schema.top_k]
        if not items:
            return self.output(q + self.bos_online.to(q)).to(phase.dtype)
        keys = torch.stack([torch.cat((item.phase, item.visual_key)) for item in items]).to(device=device)
        values = torch.stack([
            torch.cat((
                item.phase,
                item.visual_key,
                item.effect_post,
                item.executed_action.reshape(-1),
                item.next_visual_key,
                item.next_phase,
            ))
            for item in items
        ]).to(device=device)
        attention = torch.softmax(F.normalize(self.key_proj(keys), dim=-1) @ F.normalize(q, dim=-1).squeeze(0), dim=0)
        context = (attention.unsqueeze(-1) * self.value_proj(values)).sum(dim=0, keepdim=True)
        return self.output(q + context).to(phase.dtype)
