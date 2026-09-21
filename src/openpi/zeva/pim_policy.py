"""Persistent interaction memory variants for ZeVA CTE + BIT + EAP.

The within-episode policy treats BIT as short-term boundary state and PIM as a
causal long-term history of earlier BITs. The cross-attempt variant commits a
completed attempt's BIT trace and exposes it to the next attempt.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from safetensors.torch import load_model
import torch
from torch import nn
import torch.nn.functional as F

from openpi.zeva.cte_eap import SCHEMA, ZevaEffectActionPrior
from openpi.zeva.cte_eap_policy import ZevaCTEEAPPolicy, file_sha, load_cte
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS


PIM_POLICY_SCHEMA = "zeva-cte-eap-pim-v1"  # Legacy checkpoint schema.
CROSS_ATTEMPT_PIM_POLICY_SCHEMA = PIM_POLICY_SCHEMA
EPISODE_PIM_POLICY_SCHEMA = "zeva-cte-bit-episode-pim-eap-v1"


def directory_checkpoint_sha(path: str | Path) -> str:
    """Hash the immutable parent Stage2 files used to initialize PIM training."""
    root = Path(path)
    digest = hashlib.sha256()
    for name in ("model.safetensors", "zeva_adapter.pth", "COMPLETE"):
        source = root / name
        if not source.is_file():
            raise FileNotFoundError(f"Incomplete parent Stage2 checkpoint: {source}")
        digest.update(name.encode())
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


class AttemptPersistentMemory:
    """Attempt-local BIT trace plus episode-local cross-attempt PIM.

    Entries remain on the policy device and are detached from autograd. PIM is
    bounded by complete attempts rather than individual replans so eviction
    never leaves a partially retained attempt.
    """

    def __init__(self, *, max_attempts: int = 4, max_entries_per_attempt: int = 64):
        if max_attempts <= 0 or max_entries_per_attempt <= 0:
            raise ValueError("PIM capacities must be positive.")
        self.max_attempts = int(max_attempts)
        self.max_entries_per_attempt = int(max_entries_per_attempt)
        self._bit: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._attempts: list[tuple[torch.Tensor, torch.Tensor]] = []

    @staticmethod
    def _batch_one(value: torch.Tensor, name: str) -> torch.Tensor:
        value = value.detach()
        if value.ndim != 2 or value.shape[0] != 1:
            raise ValueError(f"Online {name} must be [1,D], got {tuple(value.shape)}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"Online {name} contains non-finite values.")
        return value[0].clone()

    def append_bit(self, phase: torch.Tensor, bit: torch.Tensor) -> None:
        self._bit.append((self._batch_one(phase, "phase"), self._batch_one(bit, "BIT")))

    @staticmethod
    def _uniform_indices(length: int, keep: int, device: torch.device) -> torch.Tensor:
        if length <= keep:
            return torch.arange(length, device=device)
        return torch.linspace(0, length - 1, keep, device=device).round().long().unique()

    def commit_attempt(self) -> bool:
        """Move the completed attempt's BIT trace into PIM."""
        if not self._bit:
            return False
        phases = torch.stack([item[0] for item in self._bit])
        bits = torch.stack([item[1] for item in self._bit])
        indices = self._uniform_indices(len(phases), self.max_entries_per_attempt, phases.device)
        self._attempts.append((phases[indices], bits[indices]))
        del self._attempts[:-self.max_attempts]
        self._bit.clear()
        return True

    def reset_attempt(self) -> None:
        # An explicit attempt reset is the commit boundary. Empty attempts do
        # not create synthetic PIM entries.
        self.commit_attempt()

    def reset_episode(self) -> None:
        self._bit.clear()
        self._attempts.clear()

    def entries(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._attempts:
            return None, None
        phases = torch.cat([item[0] for item in self._attempts]).unsqueeze(0)
        bits = torch.cat([item[1] for item in self._attempts]).unsqueeze(0)
        return phases, bits

    def snapshot(self) -> dict[str, int]:
        return {
            "bit_entries": len(self._bit),
            "pim_attempts": len(self._attempts),
            "pim_entries": sum(len(item[0]) for item in self._attempts),
        }


class EpisodePersistentMemory:
    """Bounded causal BIT history inside one episode.

    A decision reads only entries written by earlier decisions.  The current
    BIT is appended after action prediction, and an episode reset clears all
    entries.  No attempt or cross-episode state exists in this memory.
    """

    def __init__(self, *, max_entries: int = 64):
        if max_entries <= 0:
            raise ValueError("Episode PIM capacity must be positive.")
        self.max_entries = int(max_entries)
        self._entries: list[tuple[torch.Tensor, torch.Tensor]] = []

    def append_bit(self, phase: torch.Tensor, bit: torch.Tensor) -> None:
        phase_value = AttemptPersistentMemory._batch_one(phase, "phase")
        bit_value = AttemptPersistentMemory._batch_one(bit, "BIT")
        self._entries.append((phase_value, bit_value))
        del self._entries[:-self.max_entries]

    def reset_episode(self) -> None:
        self._entries.clear()

    def entries(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self._entries:
            return None, None
        phase = torch.stack([item[0] for item in self._entries]).unsqueeze(0)
        bit = torch.stack([item[1] for item in self._entries]).unsqueeze(0)
        return phase, bit

    def snapshot(self) -> dict[str, int]:
        return {"episode_pim_entries": len(self._entries)}


class ZevaPIMEAP(ZevaEffectActionPrior):
    """Existing EAP plus a phase-retrieved PIM token."""

    def __init__(self, dim: int = 256, prefix_dim: int = 2048, expert_dim: int = 1024):
        super().__init__(dim=dim, prefix_dim=prefix_dim, expert_dim=expert_dim)
        self.pim_phase = nn.Linear(dim, dim, bias=False)
        self.pim_bit = nn.Linear(dim, dim, bias=False)
        self.pim_query = nn.Linear(dim, dim, bias=False)
        self.pim_projector = nn.Linear(dim, prefix_dim)
        self.pim_to_global = nn.Linear(dim, dim)
        # PIM starts as a conservative addition to the already trained EAP.
        # The prefix path remains trainable on the first update; the EAP mean
        # path is initially unchanged and opens during Stage2 fine-tuning.
        nn.init.normal_(self.pim_projector.weight, std=0.002)
        nn.init.zeros_(self.pim_projector.bias)
        nn.init.zeros_(self.pim_to_global.weight)
        nn.init.zeros_(self.pim_to_global.bias)
        self._last_pim_context: torch.Tensor | None = None

    def _retrieve_pim(
        self,
        phase: torch.Tensor,
        pim_phase: torch.Tensor,
        pim_bit: torch.Tensor,
        pim_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if pim_phase.ndim != 3 or pim_bit.shape != pim_phase.shape:
            raise ValueError("PIM phase/BIT tensors must share shape [B,N,D].")
        if pim_phase.shape[0] != phase.shape[0]:
            raise ValueError("PIM batch size differs from the current phase batch.")
        valid = (
            torch.ones(pim_phase.shape[:2], dtype=torch.bool, device=pim_phase.device)
            if pim_mask is None
            else pim_mask.to(device=pim_phase.device, dtype=torch.bool)
        )
        if valid.shape != pim_phase.shape[:2] or not valid.any(dim=1).all():
            raise ValueError("Each PIM sample needs at least one valid historical BIT.")
        query = F.normalize(self.pim_query(phase), dim=-1)
        keys = F.normalize(self.pim_phase(pim_phase), dim=-1)
        scores = torch.einsum("bd,bnd->bn", query, keys).masked_fill(~valid, -torch.inf)
        weights = scores.softmax(dim=-1)
        return torch.einsum("bn,bnd->bd", weights, self.pim_bit(pim_bit))

    def activate(
        self,
        global_token: torch.Tensor,
        phase: torch.Tensor,
        effect: torch.Tensor,
        *,
        include_effect: bool = True,
        pim_phase: torch.Tensor | None = None,
        pim_bit: torch.Tensor | None = None,
        pim_mask: torch.Tensor | None = None,
        include_pim: bool = True,
    ):
        context = None
        if include_pim and pim_phase is not None and pim_bit is not None:
            context = self._retrieve_pim(phase, pim_phase, pim_bit, pim_mask)
            global_token = global_token + self.pim_to_global(context)
        prior = super().activate(global_token, phase, effect, include_effect=include_effect)
        if context is not None:
            tokens, residual = self._active
            self._active = (torch.cat([tokens, self.pim_projector(context).unsqueeze(1)], dim=1), residual)
        self._last_pim_context = context
        return prior

    def clear(self):
        super().clear()
        self._last_pim_context = None


class ZevaPIMPolicy(ZevaCTEEAPPolicy):
    """Stage2-only PIM extension initialized from a trained CTE+EAP policy."""

    POLICY_SCHEMA = CROSS_ATTEMPT_PIM_POLICY_SCHEMA
    ACTION_PRIOR_CLASS = ZevaPIMEAP

    def __init__(self, loader, cte, bank, retrieval, *, max_attempts: int = 4):
        self._max_attempts = max_attempts
        super().__init__(loader, cte, bank, retrieval)

    @classmethod
    def from_parent_handoff(
        cls,
        handoff,
        foundation_checkpoint,
        cte_checkpoint,
        artifacts,
        retrieval_checkpoint,
        parent_stage2_checkpoint,
        *,
        device="cuda",
    ):
        """Create a new PIM policy from the already validated Stage2 parent."""
        from openpi.zeva.robotwin_policy import RobotWinZevaPolicy

        bank = torch.load(artifacts, map_location="cpu", weights_only=False)
        if bank["cte_sha256"] != file_sha(cte_checkpoint):
            raise ValueError("CTE and task memory SHA differ.")
        cte = load_cte(cte_checkpoint, device)
        loader = RobotWinZevaPolicy.from_handoff(
            handoff,
            foundation_checkpoint=foundation_checkpoint,
            goal_embedding_checkpoint=Path(handoff) / "checkpoint/pretrained_model",
            install_injection_hooks=False,
            device=device,
        )
        retrieval = torch.load(retrieval_checkpoint, map_location="cpu", weights_only=False)
        policy = cls(loader, cte, bank, retrieval).to(device)
        expected_parent_identity = {
            "schema": SCHEMA,
            "cte_sha256": file_sha(cte_checkpoint),
            "artifacts_sha256": file_sha(artifacts),
            "retrieval_sha256": file_sha(retrieval_checkpoint),
            "foundation_sha256": file_sha(Path(foundation_checkpoint) / "model.safetensors"),
        }
        parent = Path(parent_stage2_checkpoint)
        adapter = torch.load(parent / "zeva_adapter.pth", map_location=device, weights_only=False)
        if adapter.get("identity") != expected_parent_identity:
            raise ValueError("Parent Stage2 is not the validated CTE+EAP lineage.")
        parent_state = adapter.get("eap")
        if parent_state is None:
            raise ValueError("Parent Stage2 has no EAP adapter state.")
        incompatible = policy.eap.load_state_dict(parent_state, strict=False)
        allowed = ("pim_phase.", "pim_bit.", "pim_query.", "pim_projector.", "pim_to_global.")
        if incompatible.unexpected_keys or any(not key.startswith(allowed) for key in incompatible.missing_keys):
            raise ValueError(f"Unexpected parent adapter mismatch: {incompatible}")
        load_model(policy.foundation, str(parent / "model.safetensors"), strict=True)
        policy.identity = {
            "schema": cls.POLICY_SCHEMA,
            "cte_sha256": file_sha(cte_checkpoint),
            "artifacts_sha256": file_sha(artifacts),
            "retrieval_sha256": file_sha(retrieval_checkpoint),
            "foundation_sha256": file_sha(Path(foundation_checkpoint) / "model.safetensors"),
            "parent_stage2_sha256": directory_checkpoint_sha(parent),
        }
        return policy.eval()

    @classmethod
    def load_trained(
        cls,
        handoff,
        foundation_checkpoint,
        cte_checkpoint,
        artifacts,
        retrieval_checkpoint,
        stage2_checkpoint,
        *,
        device="cuda",
    ):
        from openpi.zeva.robotwin_policy import RobotWinZevaPolicy

        bank = torch.load(artifacts, map_location="cpu", weights_only=False)
        cte = load_cte(cte_checkpoint, device)
        loader = RobotWinZevaPolicy.from_handoff(
            handoff,
            foundation_checkpoint=foundation_checkpoint,
            goal_embedding_checkpoint=Path(handoff) / "checkpoint/pretrained_model",
            install_injection_hooks=False,
            device=device,
        )
        retrieval = torch.load(retrieval_checkpoint, map_location="cpu", weights_only=False)
        policy = cls(loader, cte, bank, retrieval).to(device)
        folder = Path(stage2_checkpoint)
        adapter = torch.load(folder / "zeva_adapter.pth", map_location=device, weights_only=False)
        identity = adapter.get("identity", {})
        expected_core = {
            "schema": cls.POLICY_SCHEMA,
            "cte_sha256": file_sha(cte_checkpoint),
            "artifacts_sha256": file_sha(artifacts),
            "retrieval_sha256": file_sha(retrieval_checkpoint),
            "foundation_sha256": file_sha(Path(foundation_checkpoint) / "model.safetensors"),
        }
        if any(identity.get(key) != value for key, value in expected_core.items()):
            raise ValueError("PIM Stage2 lineage mismatch.")
        policy.eap.load_state_dict(adapter["eap"], strict=True)
        load_model(policy.foundation, str(folder / "model.safetensors"), strict=True)
        policy.identity = identity
        return policy.eval()

    def reset(self, *, scope="episode"):
        if scope not in {"attempt", "episode"}:
            raise ValueError("PIM policy reset scope must be 'attempt' or 'episode'.")
        if not hasattr(self, "pim"):
            self.pim = AttemptPersistentMemory(max_attempts=self._max_attempts)
        if scope == "attempt":
            self.pim.reset_attempt()
        else:
            self.pim.reset_episode()
        self._cache = None
        self._global_token = None
        self.memory.last_diagnostics = []
        self.eap.clear()
        self.foundation.reset()

    def forward(
        self,
        processed,
        phase,
        effect,
        task_language,
        *,
        pim_phase,
        pim_bit,
        pim_mask,
        include_pim=True,
    ):
        global_token = self.memory(task_language)
        prior = self.eap.activate(
            global_token,
            phase.detach(),
            effect.detach(),
            pim_phase=pim_phase.detach(),
            pim_bit=pim_bit.detach(),
            pim_mask=pim_mask,
            include_pim=include_pim,
        )
        try:
            output = self.foundation(processed)
            flow = output[0] if isinstance(output, tuple) else output
            nll = self.eap.prior_loss(prior, processed["action"])
            return flow.mean() + nll, {"flow": flow.mean().detach(), "nll": nll.detach()}
        finally:
            self.eap.clear()

    @torch.no_grad()
    def predict_action_chunk(self, processed, *, task, executed_actions=None, include_pim=True):
        if self.training:
            raise RuntimeError("Deployment must use policy.eval().")
        views = torch.stack(
            [
                F.interpolate(
                    processed[key].float(),
                    (224, 224),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).mul(2).sub(1)
                for key in ROBOTWIN_CAMERA_KEYS
            ],
            dim=1,
        )
        previous = None if executed_actions is None else self.action_normalizer.normalize(executed_actions)
        phase, bit, self._cache = self.cte.step(views, previous, self._cache)
        if self._global_token is None:
            self._global_token = self.memory(self.language(task))
        pim_phase, pim_bit = self.pim.entries()
        self.eap.activate(
            self._global_token,
            phase,
            bit,
            pim_phase=pim_phase,
            pim_bit=pim_bit,
            include_pim=include_pim,
        )
        try:
            actions = self.foundation.predict_action_chunk(processed)
        finally:
            self.eap.clear()
        self.pim.append_bit(phase, bit)
        return actions

    def retrieval_diagnostics(self):
        diagnostics = list(super().retrieval_diagnostics())
        if diagnostics:
            diagnostics[0] = {**diagnostics[0], **self.pim.snapshot()}
        return diagnostics


class ZevaEpisodePIMPolicy(ZevaPIMPolicy):
    """BIT short-term state plus causal, within-episode long-term PIM."""

    POLICY_SCHEMA = EPISODE_PIM_POLICY_SCHEMA

    def __init__(self, loader, cte, bank, retrieval, *, max_entries: int = 64):
        self._episode_pim_max_entries = int(max_entries)
        super().__init__(loader, cte, bank, retrieval, max_attempts=1)

    def reset(self, *, scope="episode"):
        if scope != "episode":
            raise ValueError("Episode PIM resets only at complete episode boundaries.")
        if not hasattr(self, "pim") or not isinstance(self.pim, EpisodePersistentMemory):
            self.pim = EpisodePersistentMemory(max_entries=self._episode_pim_max_entries)
        self.pim.reset_episode()
        self._cache = None
        self._global_token = None
        self.memory.last_diagnostics = []
        self.eap.clear()
        self.foundation.reset()

    @torch.no_grad()
    def predict_action_chunk(self, processed, *, task, executed_actions=None, include_pim=True):
        if self.training:
            raise RuntimeError("Deployment must use policy.eval().")
        views = torch.stack(
            [
                F.interpolate(
                    processed[key].float(), (224, 224), mode="bilinear",
                    align_corners=False, antialias=True,
                ).mul(2).sub(1)
                for key in ROBOTWIN_CAMERA_KEYS
            ],
            dim=1,
        )
        previous = None if executed_actions is None else self.action_normalizer.normalize(executed_actions)
        phase, bit, self._cache = self.cte.step(views, previous, self._cache)
        if self._global_token is None:
            self._global_token = self.memory(self.language(task))
        pim_phase, pim_bit = self.pim.entries()
        self.eap.activate(
            self._global_token, phase, bit,
            pim_phase=pim_phase, pim_bit=pim_bit, include_pim=include_pim,
        )
        try:
            actions = self.foundation.predict_action_chunk(processed)
        finally:
            self.eap.clear()
        # Causal order: the action at this boundary cannot read its own BIT.
        self.pim.append_bit(phase, bit)
        return actions


# Explicit names for the two supported PIM scopes.
ZevaCrossAttemptPIMPolicy = ZevaPIMPolicy
