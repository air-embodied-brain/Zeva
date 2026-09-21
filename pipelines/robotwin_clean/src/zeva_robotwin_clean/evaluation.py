"""Published RoboTwin randomized evaluation contract and result aggregation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

ROBOTWIN_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


@dataclass(frozen=True)
class EvaluationSpec:
    tasks: tuple[str, ...] = ROBOTWIN_TASKS
    split: str = "demo_randomized"
    instruction_protocol: str = "seen"
    episodes_per_task: int = 100
    first_seed: int = 1000
    replan_horizon: int = 15

    def __post_init__(self) -> None:
        if len(self.tasks) != 50 or len(set(self.tasks)) != 50:
            raise ValueError("the published RoboTwin protocol requires 50 unique tasks")
        if self.split != "demo_randomized" or self.instruction_protocol != "seen":
            raise ValueError("the published protocol is demo_randomized with seen instructions")
        if self.episodes_per_task < 1 or self.replan_horizon < 1:
            raise ValueError("episodes_per_task and replan_horizon must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        return payload


def build_jobs(spec: EvaluationSpec) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "task": task,
            "episode": episode,
            "seed": spec.first_seed + episode,
            "split": spec.split,
            "instruction_protocol": spec.instruction_protocol,
            "replan_horizon": spec.replan_horizon,
        }
        for task in spec.tasks
        for episode in range(spec.episodes_per_task)
    )


def summarize_results(rows: Iterable[Mapping[str, Any]], spec: EvaluationSpec) -> dict[str, Any]:
    counts = {task: [0, 0] for task in spec.tasks}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        task = str(row["task"])
        episode = int(row["episode"])
        if task not in counts or not 0 <= episode < spec.episodes_per_task:
            raise ValueError("evaluation row is outside the published protocol")
        key = (task, episode)
        if key in seen:
            raise ValueError(f"duplicate evaluation row: {key}")
        expected_fields = {
            "seed": spec.first_seed + episode,
            "split": spec.split,
            "instruction_protocol": spec.instruction_protocol,
            "replan_horizon": spec.replan_horizon,
        }
        for name, expected in expected_fields.items():
            if row.get(name) != expected:
                raise ValueError(f"evaluation row has incorrect {name}: {row.get(name)!r}")
        seen.add(key)
        counts[task][1] += 1
        counts[task][0] += int(bool(row["success"]))
    expected = len(spec.tasks) * spec.episodes_per_task
    if len(seen) != expected:
        raise ValueError(f"incomplete evaluation: found {len(seen)} of {expected} episodes")
    per_task = {
        task: {"successes": success, "episodes": total, "success_rate": success / total}
        for task, (success, total) in counts.items()
    }
    macro = sum(value["success_rate"] for value in per_task.values()) / len(per_task)
    return {"protocol": spec.to_dict(), "per_task": per_task, "macro_success_rate": macro}
