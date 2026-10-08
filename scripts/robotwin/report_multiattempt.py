#!/usr/bin/env python3
"""Audit PIM RoboTwin rollouts and report four cumulative success rates.

Each run root contains seed_manifest.json and rollouts/{report.json,
progress/<task>.json,results/<task>/*.mp4}. No model or simulator is needed to
recompute the metrics from completed rollouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def require(ok: bool, message: str) -> None:  # noqa: FBT001
    if not ok:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text())


def cumulative_success_rates(rows: list[dict], budget: int) -> list[dict]:
    total = len(rows)
    require(total > 0, "no completed episodes")
    rates = []
    for attempt in range(1, budget + 1):
        solved = sum(any(a["success"] for a in row["attempts"][:attempt]) for row in rows)
        rates.append(
            {
                "attempt": attempt,
                "solved": solved,
                "total": total,
                "rate": solved / total,
            }
        )
    return rates


def load_pim(root: Path, *, metadata_only: bool) -> list[dict]:
    # Older recorded PIM runs used the simulator's historical directory name.
    evidence = root / "rollouts"
    if not evidence.exists():
        evidence = root / "baseline"
    manifest_path = root / "seed_manifest.json"
    manifest = read_json(manifest_path)
    report = read_json(evidence / "report.json")
    task_seeds = manifest["tasks"]
    per_task = int(manifest["episodes_per_task"])
    budget = 4
    require(report.get("total_episodes") == len(task_seeds) * per_task, f"{root}: episode count")
    require(report.get("task_count") == len(task_seeds), f"{root}: task count")
    require(report.get("episodes_per_task") == per_task, f"{root}: episodes per task")
    require(report.get("model_seed_policy") == "continuous", f"{root}: expected continuous RNG")
    require(report.get("action_contract") == "eef16_h50_execute_h15", f"{root}: action contract")
    files = list((evidence / "progress").glob("*.json"))
    require({path.stem for path in files} == set(task_seeds), f"{root}: missing/extra tasks")
    rows: dict[tuple[str, int], dict] = {}
    expected_videos: set[Path] = set()
    for task in sorted(task_seeds):
        progress = read_json(evidence / "progress" / f"{task}.json")
        identity = progress["identity"]
        episodes = progress["episode_results"]
        require(progress.get("complete") is True and len(episodes) == per_task, f"{task}: incomplete")
        require(identity.get("task_name") == task, f"{task}: task identity")
        require(identity.get("max_policy_attempts") == budget, f"{task}: attempt budget")
        require(identity.get("attempt_reset_scope") == "attempt", f"{task}: reset scope")
        require(identity.get("model_seed_policy") == "continuous", f"{task}: RNG policy")
        require(identity.get("instruction_type") == "seen", f"{task}: instruction type")
        require(identity.get("execute_horizon") == 15, f"{task}: execution horizon")
        require(identity.get("fixed_seed_sequence") is True, f"{task}: seeds not frozen")
        expected = task_seeds[task]
        for episode in episodes:
            index = episode["episode_index"]
            require(type(index) is int and 0 <= index < per_task, f"{task}: episode index")
            key = task, index
            require(key not in rows, f"{task}: duplicate episode {index}")
            require(
                (episode["seed"], episode["instruction"]) == (expected[index]["seed"], expected[index]["instruction"]),
                f"{task}/{index}: seed or instruction changed",
            )
            attempts = episode["attempts"]
            require(1 <= len(attempts) <= budget, f"{task}/{index}: attempt count")
            for number, attempt in enumerate(attempts, 1):
                scope = "episode" if number == 1 else "attempt"
                require(
                    attempt["attempt"] == number and attempt["reset_scope"] == scope,
                    f"{task}/{index}: attempt/reset trace",
                )
                require(type(attempt["success"]) is bool, f"{task}/{index}: outcome type")
                require(attempt["steps"] >= 0 and attempt["step_limit"] > 0, f"{task}/{index}: invalid step budget")
                require(attempt["steps"] <= attempt["step_limit"], f"{task}/{index}: step overflow")
                video = (
                    evidence / "results" / task / f"episode{index}_attempt-{number}_randomized-true_"
                    f"success-{str(attempt['success']).lower()}.mp4"
                )
                expected_videos.add(video)
            require(not any(a["success"] for a in attempts[:-1]), f"{task}/{index}: ran after success")
            require(episode["success"] is attempts[-1]["success"], f"{task}/{index}: outcome mismatch")
            require(episode["success"] or len(attempts) == budget, f"{task}/{index}: stopped before budget")
            rows[key] = episode
    require(len(rows) == len(task_seeds) * per_task, f"{root}: missing episodes")
    require(report["total_attempts_executed"] == len(expected_videos), f"{root}: attempt count mismatch")
    successes = sum(row["success"] for row in rows.values())
    require(report["total_successes"] == successes, f"{root}: report success mismatch")
    if not metadata_only:
        actual_videos = set((evidence / "results").glob("**/*.mp4"))
        require(actual_videos == expected_videos, f"{root}: missing/extra/mislabeled attempt videos")
        require(all(path.stat().st_size > 0 for path in actual_videos), f"{root}: empty video")
    return cumulative_success_rates([rows[key] for key in sorted(rows)], budget)


def report_pim(root: Path, *, metadata_only: bool, expected_sha: str | None) -> dict:
    manifest_sha = hashlib.sha256((root / "seed_manifest.json").read_bytes()).hexdigest()
    if expected_sha:
        require(manifest_sha == expected_sha, "seed manifest SHA256 mismatch")
    rates = load_pim(root, metadata_only=metadata_only)
    return {
        "schema": "zeva-robotwin-pim-multiattempt-report-v1",
        "seed_manifest_sha256": manifest_sha,
        "video_audit": "not_run_metadata_only" if metadata_only else "passed",
        "cumulative_success_rate": rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--metadata-only", action="store_true", help="recompute from JSON without claiming to audit videos"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report_pim(args.run_root, metadata_only=args.metadata_only, expected_sha=args.expected_manifest_sha256)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
