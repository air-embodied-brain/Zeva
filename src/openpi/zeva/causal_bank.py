"""Frozen training-only causal bank shared by Zeva Stages 2 and 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


CAUSAL_BANK_SCHEMA = "zeva-robotwin-train-causal-bank-v2"
_LEGACY_CAUSAL_BANK_SCHEMA = "zeva-robotwin-train-causal-bank-v1"


@dataclass
class CausalBankBatch:
    phase_token: torch.Tensor
    brief_signals: torch.Tensor
    retrieved_signals: torch.Tensor
    brief_mask: torch.Tensor | None = None
    retrieved_mask: torch.Tensor | None = None


class RobotWinCausalBank:
    """Read-only phase-indexed ZTE keys and values from train95."""

    def __init__(self, payload: dict[str, Any], *, device: torch.device | str = "cpu"):
        if payload.get("schema") not in (CAUSAL_BANK_SCHEMA, _LEGACY_CAUSAL_BANK_SCHEMA):
            raise ValueError("Unsupported Zeva causal bank schema.")
        if payload.get("split") != "train95":
            raise ValueError("Stage 2 may only load a train95 causal bank.")
        self.manifest = payload["manifest"]
        self.task_names = tuple(payload["task_names"])
        self.count = torch.as_tensor(payload["count"], dtype=torch.long, device=device)
        if payload.get("schema") == CAUSAL_BANK_SCHEMA:
            self.task_prototype = F.normalize(
                torch.as_tensor(payload["task_prototype"], device=device), dim=-1
            )
        else:
            legacy_task_key = F.normalize(torch.as_tensor(payload["task_key"], device=device), dim=-1)
            legacy_weights = self.count.to(dtype=legacy_task_key.dtype).unsqueeze(-1)
            self.task_prototype = F.normalize(
                (legacy_task_key * legacy_weights).sum(dim=1)
                / legacy_weights.sum(dim=1).clamp_min(1.0),
                dim=-1,
            )
        self.phase_key = F.normalize(torch.as_tensor(payload["phase_key"], device=device), dim=-1)
        self.causal_value = F.normalize(torch.as_tensor(payload["causal_value"], device=device), dim=-1)
        expected = self.count.shape
        if self.phase_key.shape[:2] != expected or self.causal_value.shape[:2] != expected:
            raise ValueError("Causal bank count/key/value shapes disagree.")
        if self.task_prototype.shape[0] != expected[0] or len(self.task_names) != expected[0]:
            raise ValueError("Causal bank task table is inconsistent.")
        if not torch.all(self.count.sum(dim=1) > 0):
            raise ValueError("Every RoboTwin task must have at least one train95 causal entry.")

    @classmethod
    def load(cls, path: str | Path, *, device: torch.device | str = "cpu") -> RobotWinCausalBank:
        return cls(torch.load(path, map_location="cpu"), device=device)

    @property
    def phase_bins(self) -> int:
        return self.count.shape[1]

    def lookup(
        self,
        task_ids: torch.Tensor,
        progress: torch.Tensor,
        *,
        brief_size: int,
        retrieval_top_k: int,
    ) -> CausalBankBatch:
        task_ids = task_ids.to(device=self.phase_key.device, dtype=torch.long)
        progress = progress.to(device=self.phase_key.device, dtype=self.phase_key.dtype).clamp(0.0, 1.0)
        if torch.any(task_ids < 0) or torch.any(task_ids >= len(self.task_names)):
            raise ValueError("Stage 2 task id is outside the causal bank.")
        bin_ids = torch.round(progress * (self.phase_bins - 1)).to(torch.long)
        batch_ids = torch.arange(len(task_ids), device=task_ids.device)
        phase_table = self.phase_key[task_ids]
        value_table = self.causal_value[task_ids]
        valid = self.count[task_ids] > 0
        phase = phase_table[batch_ids, bin_ids]

        offsets = torch.arange(brief_size - 1, -1, -1, device=task_ids.device)
        brief_indices = (bin_ids[:, None] - offsets[None, :]).clamp_min(0)
        brief = torch.gather(
            value_table,
            1,
            brief_indices.unsqueeze(-1).expand(-1, -1, value_table.shape[-1]),
        )

        scores = torch.einsum("bpd,bd->bp", phase_table, phase)
        scores = scores.masked_fill(~valid, -torch.inf)
        top_k = min(retrieval_top_k, self.phase_bins)
        retrieved_indices = torch.topk(scores, k=top_k, dim=1).indices
        retrieved = torch.gather(
            value_table,
            1,
            retrieved_indices.unsqueeze(-1).expand(-1, -1, value_table.shape[-1]),
        )
        return CausalBankBatch(phase_token=phase, brief_signals=brief, retrieved_signals=retrieved)

    def retrieve(
        self,
        task_ids: torch.Tensor,
        phase_queries: torch.Tensor,
        *,
        brief_size: int,
        retrieval_top_k: int,
    ) -> CausalBankBatch:
        """Retrieve the nearest populated phase for an inferred task.

        Stage 2 has ground-truth task/progress labels and uses :meth:`lookup`.
        Deployment instead gets the task from the Stage 3 VLM retrieval head
        and locates progress by matching the live ZTE phase token against the
        frozen train95 phase table.
        """
        task_ids = task_ids.to(device=self.phase_key.device, dtype=torch.long)
        queries = F.normalize(
            phase_queries.to(device=self.phase_key.device, dtype=self.phase_key.dtype),
            dim=-1,
        )
        if queries.ndim != 2 or len(queries) != len(task_ids):
            raise ValueError("Phase queries must have shape [B, phase_dim].")
        if torch.any(task_ids < 0) or torch.any(task_ids >= len(self.task_names)):
            raise ValueError("Inferred task id is outside the causal bank.")

        phase_table = self.phase_key[task_ids]
        value_table = self.causal_value[task_ids]
        valid = self.count[task_ids] > 0
        scores = torch.einsum("bpd,bd->bp", phase_table, queries).masked_fill(~valid, -torch.inf)
        if torch.any(~torch.isfinite(scores).any(dim=1)):
            raise RuntimeError("An inferred task has no populated causal-bank phase.")
        bin_ids = scores.argmax(dim=1)
        batch_ids = torch.arange(len(task_ids), device=task_ids.device)
        phase = phase_table[batch_ids, bin_ids]

        offsets = torch.arange(brief_size - 1, -1, -1, device=task_ids.device)
        brief_indices = (bin_ids[:, None] - offsets[None, :]).clamp_min(0)
        brief_valid = torch.gather(valid, 1, brief_indices)
        # Empty predecessor bins are replaced by the selected populated bin.
        brief_indices = torch.where(brief_valid, brief_indices, bin_ids[:, None])
        brief = torch.gather(
            value_table,
            1,
            brief_indices.unsqueeze(-1).expand(-1, -1, value_table.shape[-1]),
        )

        top_k = min(retrieval_top_k, self.phase_bins)
        retrieved_indices = torch.topk(scores, k=top_k, dim=1).indices
        retrieved = torch.gather(
            value_table,
            1,
            retrieved_indices.unsqueeze(-1).expand(-1, -1, value_table.shape[-1]),
        )
        return CausalBankBatch(phase_token=phase, brief_signals=brief, retrieved_signals=retrieved)
