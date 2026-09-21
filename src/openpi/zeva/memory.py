from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass
class PersistentInteraction:
    phase: torch.Tensor
    signal: torch.Tensor
    count: int
    timestamp: int


class CausalMemoryManager:
    """Zeva's Brief Interaction Trace and Persistent Interaction Memory."""

    def __init__(
        self,
        *,
        brief_size: int = 8,
        persistent_size: int = 256,
        retrieval_top_k: int = 5,
        merge_phase_weight: float = 0.6,
        merge_signal_weight: float = 0.4,
        merge_threshold: float = 0.85,
        use_brief_memory: bool = True,
        use_persistent_memory: bool = True,
    ):
        if brief_size <= 0 or persistent_size <= 0 or retrieval_top_k <= 0:
            raise ValueError("Memory sizes and retrieval_top_k must be positive.")
        self.brief_size = brief_size
        self.persistent_size = persistent_size
        self.retrieval_top_k = retrieval_top_k
        self.merge_phase_weight = merge_phase_weight
        self.merge_signal_weight = merge_signal_weight
        self.merge_threshold = merge_threshold
        self.use_brief_memory = use_brief_memory
        self.use_persistent_memory = use_persistent_memory
        self._brief: list[torch.Tensor] = []
        self._persistent: list[PersistentInteraction] = []
        self._clock = 0

    def reset_attempt(self) -> None:
        self._brief.clear()

    def reset_episode(self) -> None:
        self._brief.clear()
        self._persistent.clear()
        self._clock = 0

    @staticmethod
    def _single_vector(value: torch.Tensor, name: str) -> torch.Tensor:
        value = value.detach()
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise ValueError(f"{name} must describe one transition, got {tuple(value.shape)}.")
        return F.normalize(value.to(dtype=torch.float32), dim=-1)

    def update(self, phase: torch.Tensor, signal: torch.Tensor) -> bool:
        """Update BIT and PIM; return True when a PIM entry was merged."""
        phase = self._single_vector(phase, "phase")
        signal = self._single_vector(signal, "signal")
        self._clock += 1
        if self.use_brief_memory:
            self._brief.append(signal)
            del self._brief[:-self.brief_size]

        if not self.use_persistent_memory:
            return False

        if not self._persistent:
            self._persistent.append(PersistentInteraction(phase.cpu(), signal.cpu(), 1, self._clock))
            return False

        phases = torch.stack([entry.phase.to(phase.device) for entry in self._persistent])
        signals = torch.stack([entry.signal.to(signal.device) for entry in self._persistent])
        phase_scores = torch.mv(phases, phase)
        signal_scores = torch.mv(signals, signal)
        scores = self.merge_phase_weight * phase_scores + self.merge_signal_weight * signal_scores
        best_score, best_index = torch.max(scores, dim=0)

        if float(best_score) >= self.merge_threshold:
            index = int(best_index)
            entry = self._persistent[index]
            count = entry.count + 1
            merged_phase = F.normalize((entry.phase * entry.count + phase.cpu()) / count, dim=-1)
            merged_signal = F.normalize((entry.signal * entry.count + signal.cpu()) / count, dim=-1)
            self._persistent[index] = PersistentInteraction(merged_phase, merged_signal, count, self._clock)
            return True

        self._persistent.append(PersistentInteraction(phase.cpu(), signal.cpu(), 1, self._clock))
        if len(self._persistent) > self.persistent_size:
            # Prefer keeping consolidated evidence; break count ties by recency.
            eviction_index = min(
                range(len(self._persistent)),
                key=lambda index: (self._persistent[index].count, self._persistent[index].timestamp),
            )
            self._persistent.pop(eviction_index)
        return False

    def brief_tensor(self, *, device: torch.device | str) -> torch.Tensor | None:
        if not self.use_brief_memory or not self._brief:
            return None
        return torch.stack([signal.to(device) for signal in self._brief]).unsqueeze(0)

    def retrieve(self, phase: torch.Tensor, *, device: torch.device | str) -> torch.Tensor | None:
        if not self.use_persistent_memory or not self._persistent:
            return None
        query = self._single_vector(phase, "phase").to(device)
        phases = torch.stack([entry.phase.to(device) for entry in self._persistent])
        scores = torch.mv(phases, query)
        top_k = min(self.retrieval_top_k, len(self._persistent))
        indices = torch.topk(scores, k=top_k).indices.tolist()
        return torch.stack([self._persistent[index].signal.to(device) for index in indices]).unsqueeze(0)

    def snapshot(self) -> dict[str, int]:
        return {
            "brief_size": len(self._brief),
            "persistent_size": len(self._persistent),
            "consolidated_observations": sum(entry.count for entry in self._persistent),
        }
