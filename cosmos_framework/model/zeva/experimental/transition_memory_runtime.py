"""Runtime glue for the experimental task-session transition memory."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .transition_memory import TransitionMemory, TransitionMemoryEncoder, TransitionMemorySchema, TransitionRecord


@dataclass
class TransitionMemoryReadout:
    context: torch.Tensor
    scores: torch.Tensor
    num_entries: int


class TransitionMemoryController:
    """Owns task reset, causal writes, and context reads for one policy server."""

    def __init__(
        self,
        schema: TransitionMemorySchema | None = None,
        *,
        encoder: TransitionMemoryEncoder | None = None,
        enabled: bool = False,
    ) -> None:
        self.schema = schema or TransitionMemorySchema()
        self.memory = TransitionMemory(self.schema)
        self.encoder = encoder or TransitionMemoryEncoder(self.schema)
        self.enabled = bool(enabled)
        self._replan_open = False

    def reset(self, task_cluster: str) -> None:
        self.memory.reset(task_cluster)
        self._replan_open = False

    def begin_replan(self) -> None:
        if self.memory.task_cluster is None:
            raise RuntimeError("reset(task_cluster) is required before begin_replan")
        if self._replan_open:
            raise RuntimeError("previous replan is still open")
        self._replan_open = True

    def discard_open_replan(self) -> bool:
        """Discard a terminal partial window without writing it to memory."""
        was_open = self._replan_open
        self._replan_open = False
        return was_open

    @torch.inference_mode()
    def read(self, phase: torch.Tensor, visual_key: torch.Tensor) -> TransitionMemoryReadout:
        """Read only transitions written before this replan."""
        if not self._replan_open:
            raise RuntimeError("begin_replan() must precede an transition-memory read")
        entries, scores = self.memory.query(phase.detach().cpu(), visual_key.detach().cpu())
        if self.enabled:
            context = self.encoder(phase, visual_key, entries)
        else:
            context = phase.new_zeros((1, self.encoder.hidden_dim))
        return TransitionMemoryReadout(context=context, scores=scores, num_entries=len(entries))

    def complete_replan(
        self,
        *,
        phase: torch.Tensor,
        visual_key: torch.Tensor,
        effect_post: torch.Tensor,
        executed_action: torch.Tensor,
        next_visual_key: torch.Tensor,
        next_phase: torch.Tensor,
        latent_index: int,
        completed: bool,
        metadata: dict[str, object] | None = None,
    ) -> bool:
        """Commit exactly one completed 16-step transition, or discard it."""
        if not self._replan_open:
            raise RuntimeError("begin_replan() must precede complete_replan()")
        self._replan_open = False
        if not completed:
            return False
        if executed_action.shape != (self.schema.action_horizon, self.schema.action_dim):
            raise ValueError("only a completed 16-step action window may enter transition memory")
        assert self.memory.task_cluster is not None
        self.memory.append(
            TransitionRecord(
                task_cluster=self.memory.task_cluster,
                phase=phase,
                visual_key=visual_key,
                effect_post=effect_post,
                executed_action=executed_action,
                next_visual_key=next_visual_key,
                next_phase=next_phase,
                latent_index=int(latent_index),
                schema_hash=self.schema.hash,
                metadata=dict(metadata or {}),
            )
        )
        return True
