"""RoboCasa Atomic-5 experimental transition-memory contract.

This module keeps the RoboCasa contract separate from the ARX defaults in
the generic transition-memory defaults. In particular, RoboCasa uses the
frozen Atomic-5 CTE,
7-dimensional arm actions and the left-agent/wrist 20-Hz temporal contract.
"""

from __future__ import annotations

from .transition_memory import TransitionMemorySchema


ROBOCASA_ATOMIC5_TASKS = (
    "OpenStandMixerHead",
    "TurnOnElectricKettle",
    "CloseToasterOvenDoor",
    "TurnOnMicrowave",
    "CoffeeSetupMug",
)


def make_robocasa_atomic5_schema(
    *,
    cte_hash: str = "",
    vae_temporal_hash: str = "",
    capacity: int = 64,
    top_k: int = 4,
) -> TransitionMemorySchema:
    """Return the Atomic-5 transition-memory schema.

    ``cte_hash`` and ``vae_temporal_hash`` are part of the schema hash.  A
    runtime/cache produced with a different CTE or temporal contract is
    therefore rejected instead of silently mixing representations.
    """

    return TransitionMemorySchema(
        version="robocasa_atomic5_transition_memory_v1",
        task_contract="robocasa365_atomic5_5task_left_wrist_arm7_20hz_cosmos32_exec16",
        phase_dim=128,
        visual_key_dim=128,
        effect_dim=128,
        action_dim=7,
        action_horizon=16,
        capacity=int(capacity),
        top_k=int(top_k),
        cte_hash=str(cte_hash),
        vae_temporal_hash=str(vae_temporal_hash),
    )
