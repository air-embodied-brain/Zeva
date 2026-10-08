"""Simulator-agnostic four-attempt control loop for RoboTwin evaluation.

Wire these callbacks to the RoboTwin scene and ZeVA model server. A failed
attempt recreates the *same* seed/instruction; the policy RNG is continuous.
For cross-attempt PIM, reset_policy("attempt") commits the completed BIT trace.
"""

from __future__ import annotations

from collections.abc import Callable


def run_episode(
    *,
    seed: int,
    instruction: str,
    episode_index: int,
    reset_scope: str,
    setup_scene: Callable[[int, str], None],
    close_scene: Callable[[], None],
    reset_policy: Callable[[str], None],
    rollout: Callable[[], tuple[bool, int, int]],
    start_video: Callable[[int], None],
    finish_video: Callable[[int, bool], None],
    finalize_trace: Callable[[bool, int, int], None] | None = None,
    max_attempts: int = 4,
) -> dict:
    """Return one `episode_results` row compatible with report_multiattempt.

    `rollout()` returns (success, executed_steps, step_limit). It should run
    H50 prediction/H15 execution until success or the fixed step limit.
    `finish_video` must save one distinct, labeled video per executed attempt.
    Set up the initial scene before calling this function; later attempts are
    recreated here with the same seed and instruction. Scene setup must fail
    rather than silently substitute another seed. Neither callback should
    reseed the model after the first policy reset.
    """
    if reset_scope not in {"episode", "attempt"}:
        raise ValueError("reset_scope must be episode or attempt")
    if not 1 <= max_attempts <= 4:
        raise ValueError("max_attempts must be between 1 and 4")
    attempts = []
    for number in range(1, max_attempts + 1):
        if number > 1:
            close_scene()
            setup_scene(seed, instruction)
        scope = "episode" if number == 1 else reset_scope
        start_video(number)
        reset_policy(scope)
        success, steps, step_limit = rollout()
        if not isinstance(success, bool) or not 0 <= steps <= step_limit or step_limit <= 0:
            raise ValueError("invalid rollout outcome or step count")
        finish_video(number, success)
        if finalize_trace is not None:
            finalize_trace(success, steps, step_limit)
        attempts.append(
            {
                "attempt": number,
                "success": success,
                "steps": steps,
                "step_limit": step_limit,
                "reset_scope": scope,
            }
        )
        if success:
            break
    return {
        "episode_index": episode_index,
        "seed": seed,
        "instruction": instruction,
        "success": attempts[-1]["success"],
        "steps": sum(a["steps"] for a in attempts),
        "step_limit": sum(a["step_limit"] for a in attempts),
        "attempts": attempts,
    }
