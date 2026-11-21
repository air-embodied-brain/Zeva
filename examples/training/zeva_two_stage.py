# SPDX-License-Identifier: OpenMDW-1.1
"""Run CTE training, a configured Cosmos recipe, and the gain-memory replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import subprocess


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def run_command(argv, env):
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("Commands must be nonempty argument arrays")
    subprocess.run(argv, env=env, check=True)


def checkpoint_ready(path):
    marker = path/".metadata" if path.is_dir() else path
    return marker.is_file() and marker.stat().st_size > 0


def run_memory(evidence_path, run, cfg):
    from cosmos_framework.model.zeva.gain_memory import CHANNELS, fit_gain_head, replay_episodes, replay_summary

    data = read(evidence_path)
    train, stream, initial = data["fit"], data["stream"], data["initial_donors"]
    fit_keys = {r["episode"] for r in train}
    stream_keys = {r["episode"] for r in stream} | set(initial)
    fit_donors = {i["donor"] for r in train for d in r["decisions"] for i in d["candidates"]}
    if fit_keys & stream_keys or fit_donors & stream_keys:
        raise ValueError("Source fit and chronological replay must use disjoint episodes")
    heads = {channel: fit_gain_head(train, channel) for channel in CHANNELS}
    write(run/"selector.json", {channel: asdict(head) for channel, head in heads.items()})
    reports = {}
    for mode, capacity, rolling in (
        ("fixed", cfg.get("fixed_capacity", 4), False),
        ("rolling_1", 1, True), ("rolling_2", 2, True), ("rolling_4", 4, True),
    ):
        rows = replay_episodes(stream, heads, initial_donors=initial, capacity=capacity, rolling=rolling,
                               margins=cfg.get("harm_margins"), warmup=cfg.get("warmup", 8),
                               evidence_limit=cfg.get("evidence_limit", 16), max_harm=cfg.get("max_harm", .30))
        warmup_trials = data.get("warmup_trials", 0)
        if not isinstance(warmup_trials, int) or not 0 <= warmup_trials < len(rows):
            raise ValueError("Invalid chronological warmup trial count")
        heldout = rows[warmup_trials:]
        summary = {channel: replay_summary(heldout, channel) for channel in CHANNELS}
        reports[mode] = {"summary": summary, "episodes": heldout}
    write(run/"memory_replay.json", reports)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("all", "prepare", "stage1", "stage2", "memory"), default="all")
    args = parser.parse_args(argv)
    config = read(args.config)

    def path(key):
        return (args.config.resolve().parent / config[key]).resolve()

    run, manifest = path("output"), path("episode_manifest")
    run.mkdir(parents=True, exist_ok=True)
    cte_checkpoint = run/"cte.pt"
    cte_features = run/"cte_features"
    trained = run/"cte_train.json"
    ready = run/"cte_export.json"
    actor, evidence = path("policy_checkpoint"), path("paired_evidence")
    env = dict(os.environ, ZEVA_RUN_DIR=str(run), ZEVA_EPISODE_MANIFEST=str(manifest),
               ZEVA_CTE_CHECKPOINT=str(cte_checkpoint), ZEVA_CTE_FEATURES=str(cte_features),
               ZEVA_POLICY_CHECKPOINT=str(actor), ZEVA_PAIRED_EVIDENCE=str(evidence))
    phase = args.phase
    with (run/".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase in ("all", "prepare") and not manifest.is_file():
            run_command(config["prepare_command"], env)
        if phase in ("all", "prepare", "stage1") and not manifest.is_file():
            raise FileNotFoundError(manifest)
        if phase in ("all", "stage1") and not (
            ready.is_file() and cte_checkpoint.is_file() and cte_features.is_dir()
        ):
            from cosmos_framework.model.zeva.two_stage_cte import export_cte, train_cte

            settings = config.get("stage1", {})
            if not (trained.is_file() and cte_checkpoint.is_file()):
                train_cte(manifest, cte_checkpoint, settings)
                write(trained, {"status": "completed"})
            export_cte(manifest, cte_checkpoint, cte_features, settings.get("device", "cuda"))
            write(ready, {"status": "completed"})
        if phase in ("all", "stage2"):
            if not (ready.is_file() and cte_checkpoint.is_file() and cte_features.is_dir()):
                raise ValueError("Complete Stage 1 and its CTE export before Stage 2")
            if not checkpoint_ready(actor):
                run_command(config["policy_command"], env)
            if not checkpoint_ready(actor):
                raise FileNotFoundError(actor)
        if phase in ("all", "memory"):
            if not checkpoint_ready(actor):
                raise FileNotFoundError(actor)
            if not evidence.is_file():
                run_command(config["replay_command"], env)
            run_memory(evidence, run, config.get("memory", {}))


if __name__ == "__main__":
    main()
