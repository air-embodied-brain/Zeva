#!/usr/bin/env python3
"""Run frozen cross-attempt PIM evaluation with a RoboTwin adapter.

The adapter factory is ``package.module:function`` and receives a JSON config.
It returns an object with ``setup_scene(task, seed, instruction)``,
``close_scene()``, ``reset_policy(scope)``, and ``rollout(video_path)`` methods.
Optional trace hooks are supported for external adapters.
``rollout`` writes a nonempty MP4 and returns ``(success, steps, step_limit)``.
The adapter owns simulator/model-server details; it must keep the model sampling
RNG continuous and must never substitute a different scene seed or instruction.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import importlib
import json
from pathlib import Path
import re
from typing import Any

from scripts.robotwin.multiattempt_protocol import run_episode
from scripts.robotwin.report_multiattempt import report_pim


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest(raw: bytes) -> tuple[dict[str, Any], str]:
    manifest = json.loads(raw)
    tasks = manifest.get("tasks")
    count = manifest.get("episodes_per_task")
    if not isinstance(tasks, dict) or not tasks or type(count) is not int or count < 1:
        raise ValueError("seed manifest needs tasks and a positive episodes_per_task")
    for task, rows in tasks.items():
        if (
            not isinstance(task, str)
            or re.fullmatch(r"[A-Za-z0-9_]+", task) is None
            or not isinstance(rows, list)
            or len(rows) != count
        ):
            raise ValueError("seed manifest has an invalid task or episode count")
        seeds = []
        for row in rows:
            if (
                not isinstance(row, dict)
                or type(row.get("seed")) is not int
                or not isinstance(row.get("instruction"), str)
                or not row["instruction"].strip()
            ):
                raise ValueError("each frozen episode needs an integer seed and seen instruction")
            seeds.append(row["seed"])
        if len(set(seeds)) != count:
            raise ValueError(f"duplicate frozen seed for {task}")
    return manifest, hashlib.sha256(raw).hexdigest()


def _load_factory(specification: str):
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("adapter must be package.module:function")
    factory = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(factory):
        raise TypeError("adapter factory is not callable")
    return factory


def _run_frozen_episode(
    adapter: Any,
    task: str,
    seed: int,
    instruction: str,
    index: int,
    scope: str,
    video_dir: Path,
) -> dict[str, Any]:
    pending: Path | None = None

    def start_video(number: int) -> None:
        nonlocal pending
        pending = video_dir / f"episode{index}_attempt-{number}.pending.mp4"
        if pending.exists():
            raise FileExistsError(pending)
        begin = getattr(adapter, "begin_episode_trace", None)
        if callable(begin):
            begin(task, seed, index, instruction)

    def rollout() -> tuple[bool, int, int]:
        if pending is None:
            raise RuntimeError("video path was not initialized")
        return adapter.rollout(pending)

    def finish_video(number: int, success: bool) -> None:  # noqa: FBT001
        if pending is None or not pending.is_file() or pending.stat().st_size == 0:
            raise RuntimeError("adapter did not write a nonempty attempt video")
        target = video_dir / (f"episode{index}_attempt-{number}_randomized-true_success-{str(success).lower()}.mp4")
        if target.exists():
            raise FileExistsError(target)
        pending.rename(target)

    try:
        adapter.setup_scene(task, seed, instruction)
        finalize = getattr(adapter, "finalize_episode_trace", None)
        return run_episode(
            seed=seed,
            instruction=instruction,
            episode_index=index,
            reset_scope=scope,
            setup_scene=lambda s, text: adapter.setup_scene(task, s, text),
            close_scene=adapter.close_scene,
            reset_policy=adapter.reset_policy,
            rollout=rollout,
            start_video=start_video,
            finish_video=finish_video,
            finalize_trace=finalize if callable(finalize) else None,
        )
    finally:
        adapter.close_scene()


def evaluate_pim(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    output_root: Path,
    adapter: Any = None,
    adapter_factory: Callable[[dict[str, Any]], Any] | None = None,
    adapter_config: dict[str, Any] | None = None,
    adapter_config_sha256: str | None = None,
    slots: int = 1,
    model_rng_seed: int = 20260907,
) -> dict[str, Any]:
    """Evaluate cross-attempt PIM; never resume or overwrite an existing output.

    A failed run remains on disk as evidence. Continuous model RNG cannot be
    reconstructed safely after a partial run, so retry into a *new* output.
    """
    if type(slots) is not int or slots < 1 or type(model_rng_seed) is not int:
        raise ValueError("slots must be positive and model_rng_seed must be an integer")
    if (adapter is None) == (adapter_factory is None):
        raise ValueError("provide exactly one adapter or adapter_factory")
    if adapter is not None and slots != 1:
        raise ValueError("multiple slots require an adapter_factory")
    if adapter_factory is not None and not isinstance(adapter_config, dict):
        raise ValueError("adapter_factory requires a JSON object config")
    raw = manifest_path.read_bytes()
    manifest, digest = _manifest(raw)
    if digest != expected_manifest_sha256:
        raise ValueError("frozen seed manifest SHA256 mismatch")
    scope = "attempt"
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(exist_ok=False)
    (output_root / "seed_manifest.json").write_bytes(raw)
    evidence = output_root / "rollouts"
    for name in ("progress", "results"):
        (evidence / name).mkdir(parents=True)
    _write_json(
        output_root / "protocol.json",
        {
            "policy": "cross_attempt_pim",
            "seed_manifest_sha256": digest,
            "adapter_config_sha256": adapter_config_sha256,
            "attempt_budget": 4,
            "attempt_reset_scope": scope,
            "model_seed_policy": "continuous",
            "model_rng_seed_per_slot": model_rng_seed,
            "slots": slots,
            "task_slot_assignment": "sorted-task-index modulo slots",
            "action_contract": "eef16_h50_execute_h15",
            "instruction_type": "seen",
            "stop_on_success": True,
        },
    )
    task_reports = []
    try:
        ordered_tasks = sorted(manifest["tasks"].items())
        for slot in range(slots):
            if adapter_factory is not None:
                slot_config = dict(adapter_config or {})
                slot_config["evaluation_slot"] = {
                    "index": slot,
                    "count": slots,
                    "model_rng_seed": model_rng_seed,
                    "model_seed_policy": "continuous",
                }
                slot_adapter = adapter_factory(slot_config)
            else:
                slot_adapter = adapter
            try:
                for index_in_task_list, (task, frozen_rows) in enumerate(ordered_tasks):
                    if index_in_task_list % slots != slot:
                        continue
                    task_reports.append(
                        _run_task(
                            slot_adapter,
                            task,
                            frozen_rows,
                            evidence,
                            scope,
                        )
                    )
            finally:
                close = getattr(slot_adapter, "close", None)
                if callable(close):
                    close()
        total = len(manifest["tasks"]) * manifest["episodes_per_task"]
        report = {
            "task_count": len(manifest["tasks"]),
            "episodes_per_task": manifest["episodes_per_task"],
            "total_episodes": total,
            "total_successes": sum(row["successes"] for row in task_reports),
            "total_attempts_executed": sum(row["attempt_videos"] for row in task_reports),
            "model_seed_policy": "continuous",
            "action_contract": "eef16_h50_execute_h15",
            "tasks": task_reports,
        }
        _write_json(evidence / "report.json", report)
        audit = report_pim(output_root, metadata_only=False, expected_sha=digest)
        _write_json(output_root / "audit.json", audit)
        _write_json(evidence / "state.json", {"state": "complete", "jobs": len(task_reports)})
        (output_root / "COMPLETE").write_text("PIM protocol and videos audited\n", encoding="utf-8")
        return audit
    except Exception as error:
        _write_json(
            evidence / "state.json",
            {
                "state": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def _run_task(
    adapter: Any,
    task: str,
    frozen_rows: list[dict[str, Any]],
    evidence: Path,
    scope: str,
) -> dict[str, Any]:
    episodes = []
    progress_path = evidence / "progress" / f"{task}.json"
    video_dir = evidence / "results" / task
    video_dir.mkdir()
    identity = {
        "task_name": task,
        "max_policy_attempts": 4,
        "attempt_reset_scope": scope,
        "model_seed_policy": "continuous",
        "instruction_type": "seen",
        "execute_horizon": 15,
        "fixed_seed_sequence": True,
    }
    for index, frozen in enumerate(frozen_rows):
        seed, instruction = frozen["seed"], frozen["instruction"]
        row = _run_frozen_episode(adapter, task, seed, instruction, index, scope, video_dir)
        episodes.append(row)
        _write_json(
            progress_path,
            {
                "identity": identity,
                "episode_results": episodes,
                "complete": False,
            },
        )
    successes = sum(row["success"] for row in episodes)
    videos = sum(len(row["attempts"]) for row in episodes)
    _write_json(
        progress_path,
        {
            "identity": identity,
            "episode_results": episodes,
            "complete": True,
        },
    )
    return {
        "task": task,
        "episodes": len(episodes),
        "successes": successes,
        "attempt_videos": videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--adapter", required=True, help="package.module:function factory")
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--slots", type=int, default=8, help="independent model RNG streams; original evaluation used eight"
    )
    parser.add_argument("--model-rng-seed", type=int, default=20260907)
    args = parser.parse_args()
    config_bytes = args.adapter_config.read_bytes()
    config = json.loads(config_bytes)
    audit = evaluate_pim(
        manifest_path=args.seed_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        output_root=args.output_root,
        adapter_factory=_load_factory(args.adapter),
        adapter_config=config,
        adapter_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        slots=args.slots,
        model_rng_seed=args.model_rng_seed,
    )
    print(json.dumps(audit["cumulative_success_rate"], indent=2))


if __name__ == "__main__":
    main()
