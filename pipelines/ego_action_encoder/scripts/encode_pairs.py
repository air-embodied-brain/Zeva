#!/usr/bin/env python3
"""Encode caller-sampled RGB frame pairs into action and environment tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from zeva_action_encoder.inference import iter_numpy_pair_batches, load_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="uint8 RGB pair NPY")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    pairs = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if len(pairs) == 0:
        raise ValueError("input contains no pairs")
    model, vision = load_encoder(args.checkpoint, device=args.device)
    task_shape = (len(pairs), model.config.num_task_tokens, model.config.latent_dim)
    environment_shape = (
        len(pairs),
        model.config.num_environment_tokens,
        model.config.latent_dim,
    )
    args.output.mkdir(parents=True)
    task_output = np.lib.format.open_memmap(
        args.output / "task_tokens.npy", mode="w+", dtype=np.float16, shape=task_shape
    )
    environment_output = np.lib.format.open_memmap(
        args.output / "environment_tokens.npy",
        mode="w+",
        dtype=np.float16,
        shape=environment_shape,
    )
    first = 0
    device = torch.device(args.device)
    for batch in iter_numpy_pair_batches(pairs, batch_size=args.batch_size):
        images = torch.from_numpy(batch).to(device=device, dtype=torch.float32).div_(255.0)
        with torch.inference_mode(), torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            start, future = vision.forward_pair(images[:, 0], images[:, 1])
            environment, task = model.encode_transition(start, future)
        last = first + len(batch)
        task_output[first:last] = task.float().cpu().numpy().astype(np.float16)
        environment_output[first:last] = environment.float().cpu().numpy().astype(np.float16)
        first = last
    task_output.flush()
    environment_output.flush()
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "zeva-action-encoder-output-v1",
                "count": len(pairs),
                "task_shape": list(task_shape),
                "environment_shape": list(environment_shape),
                "temporal_sampling": "caller supplied",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
