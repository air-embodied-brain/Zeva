#!/usr/bin/env python3
"""Run the published RoboTwin randomized protocol through an evaluator hook."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import importlib
import json
import os
from pathlib import Path
from typing import Any

from zeva_robotwin_clean.evaluation import EvaluationSpec
from zeva_robotwin_clean.evaluation import build_jobs
from zeva_robotwin_clean.evaluation import summarize_results

Evaluator = Callable[[Mapping[str, Any], Mapping[str, Any]], bool | Mapping[str, Any]]


def _import_evaluator(specification: str) -> Evaluator:
    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("evaluator must use package.module:function syntax")
    evaluator = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(evaluator):
        raise TypeError(f"evaluator is not callable: {specification}")
    return evaluator


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("result JSONL rows must be objects")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--first-seed", type=int)
    parser.add_argument("--replan-horizon", type=int)
    parser.add_argument("--evaluator", default=os.environ.get("ZEVA_ROBOTWIN_EVALUATOR", ""))
    args = parser.parse_args()
    if not args.evaluator:
        raise RuntimeError("set --evaluator or ZEVA_ROBOTWIN_EVALUATOR")
    application_config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    spec = EvaluationSpec(
        episodes_per_task=(
            args.episodes_per_task
            if args.episodes_per_task is not None
            else int(application_config.get("episodes_per_task", 100))
        ),
        first_seed=(
            args.first_seed if args.first_seed is not None else int(application_config.get("first_seed", 1000))
        ),
        replan_horizon=(
            args.replan_horizon
            if args.replan_horizon is not None
            else int(application_config.get("replan_horizon", 15))
        ),
    )
    if application_config.get("split", spec.split) != spec.split:
        raise ValueError("release evaluation split must be demo_randomized")
    if application_config.get("instruction_protocol", spec.instruction_protocol) != spec.instruction_protocol:
        raise ValueError("release evaluation must use seen instructions")
    evaluator = _import_evaluator(args.evaluator)
    rows = _load_existing(args.output)
    completed = {(str(row["task"]), int(row["episode"])) for row in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        for job in build_jobs(spec):
            if (job["task"], job["episode"]) in completed:
                continue
            raw = evaluator(job, application_config)
            result = dict(raw) if isinstance(raw, Mapping) else {"success": bool(raw)}
            if "success" not in result:
                raise KeyError("the evaluator result must contain success")
            row = {**result, **job, "success": bool(result["success"])}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            rows.append(row)
    summary = summarize_results(rows, spec)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_name(f".{args.summary.name}.partial")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.summary)
    print(json.dumps({"macro_success_rate": summary["macro_success_rate"]}))


if __name__ == "__main__":
    main()
