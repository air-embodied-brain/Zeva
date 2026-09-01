"""Frozen repeated-attempt protocol for RoboCasa Atomic-5 evaluation.

This module intentionally has no simulator or model dependencies so clients,
servers, tests, and result auditors can share one exact protocol definition.
"""

from __future__ import annotations

from dataclasses import dataclass


PROTOCOL_VERSION = "zeva_robocasa_atomic5_v1"
PREDICTED_ACTION_HORIZON = 32
EXECUTED_ACTION_HORIZON = 16
EFFECT_ACTION_HORIZON = 16
CTE_TRANSITION_ACTION_HORIZON = 4
EFFECT_CTE_TRANSITIONS = 4
MAX_CONTROLS_PER_ATTEMPT = 300
REQUIRED_WARMUP_REQUESTS = 1


def inference_seed(base_seed: int, attempt_id: int, replan_index: int) -> int:
    """Return a reproducible per-request seed shared by paired conditions."""
    if base_seed < 0 or attempt_id < 0 or replan_index < 0:
        raise ValueError("base_seed, attempt_id, and replan_index must be non-negative")
    # A RoboCasa attempt has at most ceil(300/16)=19 replans.  Reserving 1,000
    # values per attempt makes the mapping readable and collision-free.
    if replan_index >= 1_000:
        raise ValueError("replan_index must be < 1000")
    value = int(base_seed) + 1_000 * int(attempt_id) + int(replan_index)
    if value >= 2**32:
        raise ValueError("derived inference seed exceeds uint32")
    return value


@dataclass(frozen=True)
class AttemptSessionKey:
    """Identity of memory that may survive an environment reset."""

    session_id: str
    task_cluster: str
    environment_seed: int

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if not self.task_cluster:
            raise ValueError("task_cluster must be non-empty")
        if self.environment_seed < 0:
            raise ValueError("environment_seed must be non-negative")

    def assert_compatible(self, other: "AttemptSessionKey") -> None:
        if self != other:
            raise ValueError(
                "attempt-session mismatch: memory may persist only for the exact "
                f"(session_id, task, seed); existing={self}, requested={other}"
            )


def default_session_id(task_cluster: str, environment_seed: int) -> str:
    if not task_cluster:
        raise ValueError("task_cluster must be non-empty")
    if environment_seed < 0:
        raise ValueError("environment_seed must be non-negative")
    return f"{PROTOCOL_VERSION}:{task_cluster}:seed={int(environment_seed)}"


def contract_manifest() -> dict[str, int | str]:
    return {
        "version": PROTOCOL_VERSION,
        "predicted_action_horizon": PREDICTED_ACTION_HORIZON,
        "executed_action_horizon": EXECUTED_ACTION_HORIZON,
        "effect_action_horizon": EFFECT_ACTION_HORIZON,
        "cte_transition_action_horizon": CTE_TRANSITION_ACTION_HORIZON,
        "effect_cte_transitions": EFFECT_CTE_TRANSITIONS,
        "max_controls_per_attempt": MAX_CONTROLS_PER_ATTEMPT,
        "discarded_warmup_requests": REQUIRED_WARMUP_REQUESTS,
    }
