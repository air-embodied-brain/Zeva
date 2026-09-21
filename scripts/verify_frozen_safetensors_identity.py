#!/usr/bin/env python3
"""Prove two safetensors checkpoints have identical named tensor values."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("foundation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    different: list[str] = []
    with safe_open(args.candidate, framework="pt", device="cpu") as candidate, safe_open(
        args.foundation, framework="pt", device="cpu"
    ) as foundation:
        candidate_keys = set(candidate.keys())
        foundation_keys = set(foundation.keys())
        for key in sorted(candidate_keys & foundation_keys):
            left = candidate.get_tensor(key)
            right = foundation.get_tensor(key)
            if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
                different.append(key)
        identical = candidate_keys == foundation_keys and not different
        payload = {
            "schema": "zeva-frozen-foundation-tensor-identity-v1",
            "candidate": str(args.candidate.resolve()),
            "foundation": str(args.foundation.resolve()),
            "candidate_bytes": args.candidate.stat().st_size,
            "foundation_bytes": args.foundation.stat().st_size,
            "candidate_tensor_count": len(candidate_keys),
            "foundation_tensor_count": len(foundation_keys),
            "missing_from_candidate": sorted(foundation_keys - candidate_keys),
            "extra_in_candidate": sorted(candidate_keys - foundation_keys),
            "different_tensors": different,
            "tensor_values_bit_identical": identical,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"identical": identical, "output": str(args.output)}, indent=2))
    if not identical:
        raise RuntimeError(f"frozen PI tensor identity failed; see {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
