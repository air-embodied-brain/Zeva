#!/usr/bin/env python3
"""Verify one fixed-seed PIM evolve case and its empty-memory repeat."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_attempts(root: Path) -> list[dict[str, Any]]:
    path = root / "episodes.jsonl"
    if not path.is_file():
        raise ValueError(f"missing {path}")
    attempts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not attempts:
        raise ValueError(f"no attempts in {path}")
    ids = [int(row["attempt_id"]) for row in attempts]
    if ids != list(range(len(attempts))):
        raise ValueError(f"attempt ids are not contiguous from zero: {ids}")
    return attempts


def _replay_count(row: dict[str, Any]) -> int:
    return sum(
        bool(query.get("replay_applied"))
        for query in row.get("pim_queries", [])
    )


def _validate_artifacts(root: Path, attempts: list[dict[str, Any]]) -> None:
    for row in attempts:
        artifact = root / str(row["artifact_npz"])
        if not artifact.is_file():
            raise ValueError(f"missing artifact {artifact}")
        expected = str(row["artifact_sha256"])
        actual = _sha256(artifact)
        if actual != expected:
            raise ValueError(f"artifact SHA mismatch: {artifact}: {actual} != {expected}")


def _summarize(
    root: Path,
    *,
    expected_task: str,
    expected_seed: int,
    stable_successes: int,
) -> dict[str, Any]:
    attempts = _load_attempts(root)
    _validate_artifacts(root, attempts)

    tasks = {str(row["task"]) for row in attempts}
    seeds = {int(row["environment_seed"]) for row in attempts}
    sessions = {str(row["session_id"]) for row in attempts}
    if tasks != {expected_task}:
        raise ValueError(f"task mismatch: {tasks} != {{{expected_task!r}}}")
    if seeds != {expected_seed}:
        raise ValueError(f"seed mismatch: {seeds} != {{{expected_seed}}}")
    if len(sessions) != 1:
        raise ValueError(f"expected one session, got {sessions}")

    successes = [bool(row["success"]) for row in attempts]
    try:
        first_success = successes.index(True)
    except ValueError as exc:
        raise ValueError("case has no successful attempt") from exc
    if first_success == 0:
        raise ValueError("first attempt succeeded; this does not demonstrate evolution")

    stable_end = first_success + 1 + stable_successes
    if len(successes) < stable_end:
        raise ValueError(
            f"need {stable_successes} attempts after first success; only "
            f"{len(successes) - first_success - 1} are present"
        )
    if not all(successes[first_success + 1 : stable_end]):
        raise ValueError("success is not stable after the first successful attempt")

    replay_counts = [_replay_count(row) for row in attempts]
    if not all(count > 0 for count in replay_counts[first_success + 1 : stable_end]):
        raise ValueError("a stable-success attempt did not apply PIM success replay")

    return {
        "root": str(root),
        "task": expected_task,
        "environment_seed": expected_seed,
        "attempts": len(attempts),
        "sequence": "".join("S" if value else "F" for value in successes),
        "first_success_attempt": first_success,
        "stable_successes_required": stable_successes,
        "replay_counts": replay_counts,
        "artifact_sha256_verified": len(attempts),
        "session_id": next(iter(sessions)),
        "diffusion_seed_base": attempts[0].get("diffusion_seed_base"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stable-successes", type=int, default=3)
    args = parser.parse_args()

    first = _summarize(
        args.first,
        expected_task=args.task,
        expected_seed=args.seed,
        stable_successes=args.stable_successes,
    )
    repeat = _summarize(
        args.repeat,
        expected_task=args.task,
        expected_seed=args.seed,
        stable_successes=args.stable_successes,
    )
    if first["sequence"] != repeat["sequence"]:
        raise ValueError(
            "empty-memory repeat did not reproduce the success sequence: "
            f"{first['sequence']} != {repeat['sequence']}"
        )
    if first["diffusion_seed_base"] != repeat["diffusion_seed_base"]:
        raise ValueError("diffusion seed base differs between first and repeat runs")

    print(json.dumps({"verified": True, "first": first, "repeat": repeat}, indent=2))


if __name__ == "__main__":
    main()
