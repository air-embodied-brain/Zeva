from __future__ import annotations

import torch

from cosmos_framework.model.zeva.causal_prompt import (
    CausalPromptConfig,
    CausalPromptEncoder,
    inject_causal_prompt,
)
from cosmos_framework.model.zeva.persistent_interaction_memory import (
    PersistentInteractionMemory,
    PersistentInteractionMemoryConfig,
    PIMMemoryEntry,
)
from cosmos_framework.model.zeva.phase_conditioned_retrieval import PhaseConditionedPIMRetrieval


def _unit(index: int, dim: int = 4) -> torch.Tensor:
    value = torch.zeros(dim)
    value[index] = 1.0
    return value


def test_pim_is_retained_across_attempts_and_cleared_across_episodes() -> None:
    cfg = PersistentInteractionMemoryConfig(
        phase_dim=4, effect_dim=4, capacity=8, top_k=2, merge_threshold=0.9
    )
    memory = PersistentInteractionMemory(cfg)
    memory.reset_episode("pick")
    memory.append_completed(
        task_cluster="pick", phase=_unit(0), effect=_unit(1), attempt_id=0
    )
    memory.begin_attempt(1)
    assert len(memory) == 1
    assert memory.entries[0].first_attempt_id == 0
    memory.reset_episode("pour")
    assert len(memory) == 0
    assert memory.attempt_id == 0


def test_pim_merges_by_weighted_phase_and_effect_similarity() -> None:
    cfg = PersistentInteractionMemoryConfig(
        phase_dim=4, effect_dim=4, capacity=8, top_k=2, merge_threshold=0.8
    )
    memory = PersistentInteractionMemory(cfg)
    memory.reset_episode("pick")
    _, merged = memory.append_completed(
        task_cluster="pick", phase=_unit(0), effect=_unit(1), attempt_id=0
    )
    assert not merged
    memory.begin_attempt(1)
    _, merged = memory.append_completed(
        task_cluster="pick", phase=_unit(0), effect=_unit(1), attempt_id=1
    )
    assert merged
    assert len(memory) == 1
    assert memory.entries[0].merge_count == 2
    _, merged = memory.append_completed(
        task_cluster="pick", phase=_unit(2), effect=_unit(3), attempt_id=1
    )
    assert not merged
    assert len(memory) == 2


def test_phase_conditioned_pim_retrieval_uses_current_phase() -> None:
    cfg = PersistentInteractionMemoryConfig(
        phase_dim=4, effect_dim=4, capacity=8, top_k=1, merge_threshold=0.99
    )
    memory = PersistentInteractionMemory(cfg)
    memory.reset_episode("pick")
    memory.append_completed(
        task_cluster="pick", phase=_unit(0), effect=_unit(3), attempt_id=0
    )
    memory.append_completed(
        task_cluster="pick", phase=_unit(1), effect=_unit(3), attempt_id=0
    )
    entries, scores = memory.query_phase(_unit(1))
    assert len(entries) == 1
    assert torch.equal(entries[0].phase, _unit(1))
    assert torch.allclose(scores, torch.ones(1))

    phases, effects, valid, wrapped_scores = PhaseConditionedPIMRetrieval(memory)(_unit(1))
    assert torch.equal(phases[0], _unit(1))
    assert torch.equal(effects[0], _unit(3))
    assert valid.tolist() == [True]
    assert torch.equal(wrapped_scores, scores)


def test_zero_gate_is_exact_released_prefix_bypass() -> None:
    base = torch.randn(3, 16)
    residual = torch.randn_like(base)
    valid = torch.tensor([[1, 0], [0, 0], [1, 1]], dtype=torch.bool)
    output = inject_causal_prompt(base, residual, torch.zeros(1), valid)
    assert torch.equal(output, base)
    opened = inject_causal_prompt(base, residual, torch.ones(1), valid)
    assert not torch.equal(opened[0], base[0])
    assert torch.equal(opened[1], base[1])
    assert torch.equal(inject_causal_prompt(base, residual, torch.zeros(1), valid), base)


def test_prompt_encoder_fuses_task_bit_and_pim_tensors() -> None:
    cfg = CausalPromptConfig(
        global_dim=8,
        phase_dim=4,
        effect_dim=4,
        brief_length=2,
        persistent_length=2,
        hidden_dim=8,
        num_heads=2,
    )
    encoder = CausalPromptEncoder(cfg)
    output = encoder(
        torch.randn(3, 8),
        torch.randn(3, 4),
        torch.randn(3, 2, 4),
        torch.tensor([[1, 1], [1, 0], [0, 0]], dtype=torch.bool),
        torch.randn(3, 2, 4),
        torch.randn(3, 2, 4),
        torch.tensor([[1, 0], [1, 1], [0, 0]], dtype=torch.bool),
    )
    assert output.shape == (3, 8)
    assert torch.isfinite(output).all()


def test_pim_entries_use_paper_type() -> None:
    memory = PersistentInteractionMemory(
        PersistentInteractionMemoryConfig(phase_dim=4, effect_dim=4)
    )
    memory.reset_episode("pick")
    memory.append_completed(
        task_cluster="pick", phase=_unit(0), effect=_unit(1), attempt_id=0
    )
    assert isinstance(memory.entries[0], PIMMemoryEntry)
