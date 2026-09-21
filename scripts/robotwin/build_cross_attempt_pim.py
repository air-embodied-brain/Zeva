"""Build label-free, train-only cross-attempt BIT pairings for ZeVA PIM.

Pairing never uses task success labels.  A target decision receives three
predeclared candidate histories from a different episode of the same task:
nearest/farthest within the same scene condition and nearest across conditions.
Distance is measured by the frozen CTE's initial phase token.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro

from openpi.zeva.cte_eap_policy import file_sha


PIM_ARTIFACT_SCHEMA = "zeva-cross-attempt-bit-pairings-v1"


@dataclasses.dataclass
class Args:
    cte_artifacts: str
    output: str
    max_entries_per_attempt: int = 64


def record_parts(record_id: str) -> tuple[str, str, int]:
    condition, task, episode = record_id.rsplit(":", 2)
    return condition, task, int(episode)


def choose(query: torch.Tensor, candidates: list[str], cache: dict, *, nearest: bool) -> tuple[str, float]:
    if not candidates:
        raise ValueError("PIM pairing candidate set is empty.")
    keys = torch.stack([cache[key]["phase"][0].float() for key in candidates])
    scores = F.normalize(keys, dim=-1) @ F.normalize(query.float(), dim=-1)
    index = int(scores.argmax() if nearest else scores.argmin())
    return candidates[index], float(scores[index])


def build_pairings(artifact: dict) -> dict[str, dict]:
    train = artifact["splits"]["train"]
    validation = artifact["splits"]["validation"]
    metadata = {key: record_parts(key) for key in (*train.keys(), *validation.keys())}
    pairings: dict[str, dict] = {}
    for split, targets in (("train", train), ("validation", validation)):
        pairings[split] = {}
        for target_id, target in targets.items():
            condition, task, _ = metadata[target_id]
            same = [key for key in train if metadata[key][:2] == (condition, task) and key != target_id]
            cross = [key for key in train if metadata[key][1] == task and metadata[key][0] != condition]
            if not same or not cross:
                raise ValueError(f"Need same- and cross-condition train histories for {target_id}.")
            query = target["phase"][0]
            near_id, near_score = choose(query, same, train, nearest=True)
            far_id, far_score = choose(query, same, train, nearest=False)
            cross_id, cross_score = choose(query, cross, train, nearest=True)
            if target_id in {near_id, far_id, cross_id}:
                raise AssertionError("A PIM history must come from a different episode.")
            pairings[split][target_id] = {
                "matched": near_id,
                "same_condition_far": far_id,
                "cross_condition": cross_id,
                "cosine": {
                    "matched": near_score,
                    "same_condition_far": far_score,
                    "cross_condition": cross_score,
                },
            }
    return pairings


def main(args: Args) -> None:
    source = Path(args.cte_artifacts)
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing PIM artifact.")
    if args.max_entries_per_attempt <= 0:
        raise ValueError("PIM capacity must be positive.")
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    if artifact.get("bank_subset") != "train" or set(artifact.get("splits", {})) != {"train", "validation"}:
        raise ValueError("PIM requires complete train-only CTE artifacts.")
    overlap = set(artifact["splits"]["train"]) & set(artifact["splits"]["validation"])
    if overlap:
        raise ValueError("Train/validation episode overlap in CTE artifacts.")
    pairings = build_pairings(artifact)
    payload = {
        "schema": PIM_ARTIFACT_SCHEMA,
        "cte_artifacts_sha256": file_sha(source),
        "max_entries_per_attempt": args.max_entries_per_attempt,
        "pairing": "same-task; same-condition nearest/farthest and cross-condition nearest; initial-CTE-phase cosine",
        "label_free": True,
        "source_subset": "train-only",
        "pairings": pairings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(json.dumps({
        "output": str(output), "sha256": file_sha(output),
        "targets": {split: len(rows) for split, rows in pairings.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
