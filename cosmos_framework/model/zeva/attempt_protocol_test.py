from __future__ import annotations

import pytest

from cosmos_framework.model.zeva.attempt_protocol import (
    EFFECT_ACTION_HORIZON,
    EFFECT_CTE_TRANSITIONS,
    EXECUTED_ACTION_HORIZON,
    PREDICTED_ACTION_HORIZON,
    REQUIRED_WARMUP_REQUESTS,
    CTE_TRANSITION_ACTION_HORIZON,
    AttemptSessionKey,
    default_session_id,
    inference_seed,
)


def test_zeva_temporal_contract_is_exact() -> None:
    assert PREDICTED_ACTION_HORIZON == 32
    assert EXECUTED_ACTION_HORIZON == 16
    assert EFFECT_ACTION_HORIZON == 16
    assert CTE_TRANSITION_ACTION_HORIZON == 4
    assert EFFECT_CTE_TRANSITIONS == 4
    assert CTE_TRANSITION_ACTION_HORIZON * EFFECT_CTE_TRANSITIONS == EFFECT_ACTION_HORIZON
    assert REQUIRED_WARMUP_REQUESTS == 1


def test_attempt_memory_survives_only_exact_task_seed_session() -> None:
    session = AttemptSessionKey(default_session_id("OpenStandMixerHead", 195), "OpenStandMixerHead", 195)
    session.assert_compatible(
        AttemptSessionKey(default_session_id("OpenStandMixerHead", 195), "OpenStandMixerHead", 195)
    )
    with pytest.raises(ValueError, match="attempt-session mismatch"):
        session.assert_compatible(
            AttemptSessionKey(default_session_id("OpenStandMixerHead", 195), "OpenStandMixerHead", 196)
        )
    with pytest.raises(ValueError, match="attempt-session mismatch"):
        session.assert_compatible(AttemptSessionKey(session.session_id, "TurnOnMicrowave", 195))


def test_paired_inference_seed_schedule_is_deterministic() -> None:
    seeds_a = [inference_seed(20260824, attempt, replan) for attempt in range(3) for replan in range(19)]
    seeds_b = [inference_seed(20260824, attempt, replan) for attempt in range(3) for replan in range(19)]
    assert seeds_a == seeds_b
    assert len(seeds_a) == len(set(seeds_a))
