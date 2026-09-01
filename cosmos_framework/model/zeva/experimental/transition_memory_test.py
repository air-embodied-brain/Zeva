from __future__ import annotations

import pytest
import torch

from cosmos_framework.model.zeva.experimental.transition_memory import (
    TransitionMemory,
    TransitionMemoryEncoder,
    TransitionMemorySchema,
    TransitionRecord,
)
from cosmos_framework.model.zeva.experimental.transition_memory_runtime import TransitionMemoryController
from cosmos_framework.model.zeva.experimental.robocasa_transition_memory import make_robocasa_atomic5_schema


def make_transition(schema: TransitionMemorySchema, task: str, i: int, *, valid: bool = True) -> TransitionRecord:
    return TransitionRecord(
        task_cluster=task,
        phase=torch.full((schema.phase_dim,), float(i)),
        visual_key=torch.full((schema.visual_key_dim,), float(i + 1)),
        effect_post=torch.full((schema.effect_dim,), float(i + 2)),
        executed_action=torch.full((schema.action_horizon, schema.action_dim), float(i + 3)),
        next_visual_key=torch.full((schema.visual_key_dim,), float(i + 4)),
        next_phase=torch.full((schema.phase_dim,), float(i + 5)),
        latent_index=4 * i,
        schema_hash=schema.hash,
        valid=valid,
    )


def test_fifo_reset_and_task_isolation() -> None:
    schema = TransitionMemorySchema(capacity=2)
    memory = TransitionMemory(schema)
    memory.append(make_transition(schema, "pick", 0))
    memory.append(make_transition(schema, "pick", 1))
    memory.append(make_transition(schema, "pick", 2))
    assert len(memory) == 2
    with pytest.raises(ValueError, match="task mismatch"):
        memory.append(make_transition(schema, "pour", 3))
    memory.reset("pour")
    memory.append(make_transition(schema, "pour", 3))
    assert len(memory) == 1
    assert memory.task_cluster == "pour"


def test_schema_and_incomplete_transition_fail_fast() -> None:
    schema = TransitionMemorySchema()
    memory = TransitionMemory(schema)
    bad_hash = make_transition(schema, "pick", 0)
    bad_hash.schema_hash = "wrong"
    with pytest.raises(ValueError, match="schema mismatch"):
        memory.append(bad_hash)
    with pytest.raises(ValueError, match="invalid or incomplete"):
        memory.append(make_transition(schema, "pick", 0, valid=False))


def test_query_is_bounded_and_encoder_handles_empty_memory() -> None:
    schema = TransitionMemorySchema(capacity=64, top_k=4)
    memory = TransitionMemory(schema)
    for i in range(10):
        memory.append(make_transition(schema, "pick", i))
    entries, scores = memory.query(torch.zeros(schema.phase_dim), torch.zeros(schema.visual_key_dim))
    assert len(entries) == 4
    assert scores.shape == (4,)
    encoder = TransitionMemoryEncoder(schema)
    context = encoder(torch.zeros(schema.phase_dim), torch.zeros(schema.visual_key_dim), entries)
    empty_context = encoder(torch.zeros(schema.phase_dim), torch.zeros(schema.visual_key_dim), [])
    assert context.shape == (1, 256)
    assert empty_context.shape == (1, 256)
    assert torch.isfinite(context).all()


def test_controller_does_not_commit_incomplete_replan() -> None:
    schema = TransitionMemorySchema()
    controller = TransitionMemoryController(schema, enabled=True)
    controller.reset("pick")
    controller.begin_replan()
    readout = controller.read(torch.zeros(schema.phase_dim), torch.zeros(schema.visual_key_dim))
    assert readout.num_entries == 0
    assert not controller.complete_replan(
        phase=torch.zeros(schema.phase_dim),
        visual_key=torch.zeros(schema.visual_key_dim),
        effect_post=torch.zeros(schema.effect_dim),
        executed_action=torch.zeros(schema.action_horizon, schema.action_dim),
        next_visual_key=torch.zeros(schema.visual_key_dim),
        next_phase=torch.zeros(schema.phase_dim),
        latent_index=0,
        completed=False,
    )
    assert len(controller.memory) == 0


def test_robocasa_atomic5_contract_is_arm7_and_temporally_bounded() -> None:
    schema = make_robocasa_atomic5_schema(cte_hash="cte", vae_temporal_hash="vae")
    assert schema.action_dim == 7
    assert schema.action_horizon == 16
    assert schema.capacity == 64
    assert schema.top_k == 4
    # The provenance fields participate in the hash; a cache from another
    # A different CTE/VAE contract cannot be accepted by the runtime.
    assert schema.hash != make_robocasa_atomic5_schema(cte_hash="other", vae_temporal_hash="vae").hash


def test_robocasa_controller_writes_only_after_completed_window() -> None:
    schema = make_robocasa_atomic5_schema()
    controller = TransitionMemoryController(schema, enabled=True)
    controller.reset("OpenStandMixerHead")
    def zeros(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(shape)
    controller.begin_replan()
    controller.read(zeros((128,)), zeros((128,)))
    assert not controller.complete_replan(
        phase=zeros((128,)), visual_key=zeros((128,)), effect_post=zeros((128,)),
        executed_action=zeros((16, 7)), next_visual_key=zeros((128,)),
        next_phase=zeros((128,)), latent_index=0, completed=False,
    )
    assert len(controller.memory) == 0
    controller.begin_replan()
    controller.read(zeros((128,)), zeros((128,)))
    assert controller.complete_replan(
        phase=zeros((128,)), visual_key=zeros((128,)), effect_post=zeros((128,)),
        executed_action=zeros((16, 7)), next_visual_key=zeros((128,)),
        next_phase=zeros((128,)), latent_index=0, completed=True,
    )
    assert len(controller.memory) == 1


def test_attempt_outcome_annotation_is_scoped_and_auditable() -> None:
    schema = make_robocasa_atomic5_schema()
    memory = TransitionMemory(schema)
    first = make_transition(schema, "OpenStandMixerHead", 0)
    first.metadata = {"attempt_id": 0, "session_id": "mixer:195", "environment_seed": 195}
    second = make_transition(schema, "OpenStandMixerHead", 1)
    second.metadata = {"attempt_id": 1, "session_id": "mixer:195", "environment_seed": 195}
    memory.append(first)
    memory.append(second)
    updated = memory.annotate_attempt_outcome(
        0,
        outcome="failure",
        termination_reason="max_steps",
        total_steps=300,
        final_progress=0.0,
    )
    assert updated == 1
    assert memory.entries[0].metadata["terminal_outcome"] == "failure"
    assert "terminal_outcome" not in memory.entries[1].metadata


def test_discard_open_replan_never_commits_partial_window() -> None:
    schema = make_robocasa_atomic5_schema()
    controller = TransitionMemoryController(schema, enabled=False)
    controller.reset("OpenStandMixerHead")
    controller.begin_replan()
    assert controller.discard_open_replan()
    assert len(controller.memory) == 0
    assert not controller.discard_open_replan()


    expected = {(95, frame) for frame in range(85)}
    assert (95, 83) in expected
    assert (93, 83) not in expected
