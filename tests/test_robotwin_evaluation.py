from __future__ import annotations

import hashlib
from itertools import groupby
import json
from pathlib import Path

import pytest

from scripts.robotwin.evaluate_multiattempt import evaluate_pim
from scripts.robotwin.multiattempt_protocol import run_episode
from scripts.robotwin.report_multiattempt import report_pim


def test_released_manifest_is_the_frozen_ten_by_twenty() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "robotwin_multiattempt_original10x20.json"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ("1b9dbf74bd9d9b8871647a00d6557f84884685d4459065f004bd600a86b1679b")
    payload = json.loads(raw)
    assert payload["episodes_per_task"] == 20
    assert len(payload["tasks"]) == 10
    assert all(len(rows) == 20 for rows in payload["tasks"].values())
    assert all(set(row) == {"seed", "instruction"} for rows in payload["tasks"].values() for row in rows)


class FakeAdapter:
    def __init__(self) -> None:
        self.events = []
        self.attempt = 0
        self.closed = False

    def setup_scene(self, task: str, seed: int, instruction: str) -> None:
        self.events.append(("setup", task, seed, instruction))

    def close_scene(self) -> None:
        self.events.append(("close_scene",))

    def reset_policy(self, scope: str) -> None:
        self.attempt += 1
        self.events.append(("reset", scope))

    def begin_episode_trace(self, task: str, seed: int, episode_index: int, instruction: str) -> None:
        self.events.append(("begin_trace", task, seed, episode_index, instruction))

    def finalize_episode_trace(self, success: bool, steps: int, step_limit: int) -> None:  # noqa: FBT001
        self.events.append(("finalize_trace", success, steps, step_limit))

    def rollout(self, video_path: Path) -> tuple[bool, int, int]:
        video_path.write_bytes(b"video")
        return self.attempt == 2, 10, 20

    def close(self) -> None:
        self.closed = True


def test_runner_writes_auditable_four_attempt_result(tmp_path: Path) -> None:
    manifest = tmp_path / "frozen.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes_per_task": 1,
                "tasks": {"task_a": [{"seed": 1000, "instruction": "pick the object"}]},
            }
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    adapter = FakeAdapter()
    output = tmp_path / "pim"
    audit = evaluate_pim(
        manifest_path=manifest,
        expected_manifest_sha256=digest,
        output_root=output,
        adapter=adapter,
    )
    assert audit["video_audit"] == "passed"
    assert [row["solved"] for row in audit["cumulative_success_rate"]] == [0, 1, 1, 1]
    assert adapter.events.count(("setup", "task_a", 1000, "pick the object")) == 2
    assert ("reset", "attempt") in adapter.events
    assert len([event for event in adapter.events if event[0] == "begin_trace"]) == 2
    assert len([event for event in adapter.events if event[0] == "finalize_trace"]) == 2
    assert adapter.closed
    assert (output / "COMPLETE").is_file()
    assert len(list((output / "rollouts" / "results").glob("**/*.mp4"))) == 2
    with pytest.raises(FileExistsError):
        evaluate_pim(
            manifest_path=manifest,
            expected_manifest_sha256=digest,
            output_root=output,
            adapter=FakeAdapter(),
        )


def test_runner_rejects_changed_manifest_before_writing(tmp_path: Path) -> None:
    manifest = tmp_path / "frozen.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes_per_task": 1,
                "tasks": {"task_a": [{"seed": 1000, "instruction": "pick"}]},
            }
        )
    )
    with pytest.raises(ValueError, match="SHA256"):
        evaluate_pim(
            manifest_path=manifest,
            expected_manifest_sha256="0" * 64,
            output_root=tmp_path / "run",
            adapter=FakeAdapter(),
        )
    assert not (tmp_path / "run").exists()


def test_original_task_to_slot_rng_streams(tmp_path: Path) -> None:
    manifest = tmp_path / "frozen.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes_per_task": 1,
                "tasks": {name: [{"seed": 1000, "instruction": "pick"}] for name in ("task_c", "task_a", "task_b")},
            }
        )
    )
    adapters: dict[int, FakeAdapter] = {}

    def factory(config: dict) -> FakeAdapter:
        slot = config["evaluation_slot"]
        assert slot["count"] == 2
        assert slot["model_rng_seed"] == 20260907
        adapter = FakeAdapter()
        adapters[slot["index"]] = adapter
        return adapter

    output = tmp_path / "pim"
    audit = evaluate_pim(
        manifest_path=manifest,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        output_root=output,
        adapter_factory=factory,
        adapter_config={},
        slots=2,
    )
    assert audit["video_audit"] == "passed"
    assert json.loads((output / "protocol.json").read_text())["slots"] == 2
    assert [name for name, _ in groupby(event[1] for event in adapters[0].events if event[0] == "setup")] == [
        "task_a",
        "task_c",
    ]
    assert {event[1] for event in adapters[1].events if event[0] == "setup"} == {"task_b"}
    assert all(adapter.closed for adapter in adapters.values())


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _pim(root: Path) -> None:
    tasks = {
        task: [{"seed": 10 + index, "instruction": f"{task}-{index}"} for index in range(2)] for task in ("a", "b")
    }
    _write(root / "seed_manifest.json", {"episodes_per_task": 2, "tasks": tasks})
    paths = (("a", 1, 2), ("b", 4, None))
    attempts_total = successes = 0
    for task, first, second in paths:
        episodes = []
        for index, winning_attempt in enumerate((first, second)):
            attempts = []
            for number in range(1, (winning_attempt or 4) + 1):
                success = number == winning_attempt
                attempts.append(
                    {
                        "attempt": number,
                        "success": success,
                        "steps": 10,
                        "step_limit": 20,
                        "reset_scope": "episode" if number == 1 else "attempt",
                    }
                )
                video = (
                    root / "rollouts" / "results" / task / f"episode{index}_attempt-{number}_randomized-true_"
                    f"success-{str(success).lower()}.mp4"
                )
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"video")
            attempts_total += len(attempts)
            successes += bool(winning_attempt)
            episodes.append(
                {
                    "episode_index": index,
                    "seed": 10 + index,
                    "instruction": f"{task}-{index}",
                    "success": bool(winning_attempt),
                    "attempts": attempts,
                }
            )
        _write(
            root / "rollouts" / "progress" / f"{task}.json",
            {
                "complete": True,
                "identity": {
                    "task_name": task,
                    "max_policy_attempts": 4,
                    "attempt_reset_scope": "attempt",
                    "model_seed_policy": "continuous",
                    "instruction_type": "seen",
                    "execute_horizon": 15,
                    "fixed_seed_sequence": True,
                },
                "episode_results": episodes,
            },
        )
    _write(
        root / "rollouts" / "report.json",
        {
            "total_episodes": 4,
            "task_count": 2,
            "episodes_per_task": 2,
            "model_seed_policy": "continuous",
            "action_contract": "eef16_h50_execute_h15",
            "total_attempts_executed": attempts_total,
            "total_successes": successes,
        },
    )


def test_four_attempt_curve_and_video_audit(tmp_path: Path) -> None:
    pim = tmp_path / "pim"
    _pim(pim)
    result = report_pim(pim, metadata_only=False, expected_sha=None)
    assert set(result) == {"schema", "seed_manifest_sha256", "video_audit", "cumulative_success_rate"}
    assert result["video_audit"] == "passed"
    assert result["cumulative_success_rate"] == [
        {"attempt": 1, "solved": 1, "total": 4, "rate": 0.25},
        {"attempt": 2, "solved": 2, "total": 4, "rate": 0.5},
        {"attempt": 3, "solved": 2, "total": 4, "rate": 0.5},
        {"attempt": 4, "solved": 3, "total": 4, "rate": 0.75},
    ]


def test_mismatch_and_missing_video_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "pim"
    _pim(root)
    video = next((root / "rollouts" / "results").glob("**/*.mp4"))
    video.unlink()
    with pytest.raises(ValueError, match="videos"):
        report_pim(root, metadata_only=False, expected_sha=None)
    assert report_pim(root, metadata_only=True, expected_sha=None)["video_audit"] == "not_run_metadata_only"
    progress = root / "rollouts" / "progress" / "a.json"
    payload = json.loads(progress.read_text())
    payload["episode_results"][0]["attempts"][0]["success"] = False
    _write(progress, payload)
    with pytest.raises(ValueError, match="outcome mismatch"):
        report_pim(root, metadata_only=True, expected_sha=None)


def test_historical_pim_evidence_directory_is_readable(tmp_path: Path) -> None:
    root = tmp_path / "historical_pim"
    _pim(root)
    (root / "rollouts").rename(root / "baseline")
    result = report_pim(root, metadata_only=False, expected_sha=None)
    assert result["video_audit"] == "passed"
    assert [row["solved"] for row in result["cumulative_success_rate"]] == [1, 2, 2, 3]


def test_protocol_same_scene_and_stop_on_success() -> None:
    events = []
    outcomes = iter([(False, 20, 20), (True, 7, 20)])
    row = run_episode(
        seed=42,
        instruction="pick",
        episode_index=0,
        reset_scope="attempt",
        setup_scene=lambda seed, instruction: events.append(("setup", seed, instruction)),
        close_scene=lambda: events.append("close"),
        reset_policy=lambda scope: events.append(("reset", scope)),
        rollout=lambda: next(outcomes),
        start_video=lambda n: events.append(("start_video", n)),
        finish_video=lambda n, success: events.append(("finish_video", n, success)),
    )
    assert [a["reset_scope"] for a in row["attempts"]] == ["episode", "attempt"]
    assert row["success"] is True
    assert row["steps"] == 27
    assert ("setup", 42, "pick") in events
    assert events.count("close") == 1
    assert ("start_video", 3) not in events


def test_minimal_adapter_does_not_need_trace_export_hooks(tmp_path: Path) -> None:
    manifest = tmp_path / "frozen.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes_per_task": 1,
                "tasks": {"task_a": [{"seed": 1000, "instruction": "pick"}]},
            }
        )
    )
    adapter = FakeAdapter()
    adapter.begin_episode_trace = None
    adapter.finalize_episode_trace = None
    result = evaluate_pim(
        manifest_path=manifest,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        output_root=tmp_path / "pim",
        adapter=adapter,
    )
    assert [row["solved"] for row in result["cumulative_success_rate"]] == [0, 1, 1, 1]
