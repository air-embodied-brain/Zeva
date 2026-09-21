#!/usr/bin/env python3
"""Train the Zeva-Ego environment encoding stage on caller-provided data."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from dataclasses import fields
import json
import os
from pathlib import Path

import torch
from zeva_action_encoder.checkpoints import load_checkpoint_payload
from zeva_action_encoder.checkpoints import restore_resume_state
from zeva_action_encoder.checkpoints import save_training_checkpoint
from zeva_action_encoder.configs import EnvironmentEncodingLossConfig
from zeva_action_encoder.models import DinoV2Config
from zeva_action_encoder.models import EnvironmentEncoder
from zeva_action_encoder.models import EnvironmentEncoderConfig
from zeva_action_encoder.models import FrozenDinoV2
from zeva_action_encoder.training import environment_encoding_step
from zeva_action_encoder.training.runtime import build_optimizer
from zeva_action_encoder.training.runtime import build_warmup_cosine_scheduler
from zeva_action_encoder.training.runtime import cycle_loader
from zeva_action_encoder.training.runtime import finish_distributed
from zeva_action_encoder.training.runtime import import_dataset_factory
from zeva_action_encoder.training.runtime import initialize_distributed
from zeva_action_encoder.training.runtime import make_loader
from zeva_action_encoder.training.runtime import move_batch_to_device
from zeva_action_encoder.training.runtime import read_json
from zeva_action_encoder.training.runtime import seed_everything
from zeva_action_encoder.training.runtime import unwrap_model
from zeva_action_encoder.training.runtime import wrap_ddp


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-factory", default=os.environ.get("ZEVA_EGO_DATASET_FACTORY", ""))
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _dataclass_kwargs(cls: type, values: dict[str, object]) -> dict[str, object]:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(values.keys() - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    return values


def main() -> None:
    args = _arguments()
    if not args.dataset_factory:
        raise RuntimeError("set --dataset-factory or ZEVA_EGO_DATASET_FACTORY")
    config = read_json(args.config)
    runtime = config["runtime"]
    optimization = config["optimization"]
    context = initialize_distributed(str(runtime.get("device", "cuda")))
    try:
        expected_world_size = int(runtime.get("expected_world_size", context.world_size))
        if context.world_size != expected_world_size:
            raise ValueError(f"release recipe expects world size {expected_world_size}, got {context.world_size}")
        seed_everything(int(runtime["seed"]), rank=context.rank)
        model_config = EnvironmentEncoderConfig(**_dataclass_kwargs(EnvironmentEncoderConfig, config["model"]))
        loss_config = EnvironmentEncodingLossConfig(**_dataclass_kwargs(EnvironmentEncodingLossConfig, config["loss"]))
        vision_config = DinoV2Config(**_dataclass_kwargs(DinoV2Config, config.get("vision", {})))
        model = EnvironmentEncoder(model_config).to(context.device)
        model.configure_execution(
            attention_backend=str(runtime.get("attention_backend", "sdpa")),
            activation_checkpointing=bool(runtime.get("activation_checkpointing", True)),
        )
        allowed_sizes = tuple(tuple(value) for value in config.get("data", {}).get("allowed_image_sizes", ()))
        vision = (
            FrozenDinoV2(
                vision_config,
                allowed_image_sizes=allowed_sizes or None,
            )
            .to(context.device)
            .eval()
        )
        optimizer = build_optimizer(model, optimization)
        total_steps = int(runtime["steps"])
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=int(optimization.get("warmup_steps", 0)),
            total_steps=total_steps,
            minimum_ratio=float(optimization.get("minimum_learning_rate_ratio", 0.1)),
        )
        first_step = 0
        if args.resume is not None:
            payload = load_checkpoint_payload(args.resume, expected_stage="environment_encoding")
            if payload["model_config"] != asdict(model_config):
                raise ValueError("resume checkpoint model configuration differs from the run")
            first_step = restore_resume_state(payload, model=model, optimizer=optimizer, scheduler=scheduler)
        model = wrap_ddp(model, context)
        data_config = dict(config.get("data", {}))
        data_config["distributed_rank"] = context.rank
        data_config["distributed_world_size"] = context.world_size
        dataset = import_dataset_factory(args.dataset_factory)(data_config, "environment_encoding")
        loader, sampler = make_loader(
            dataset,
            batch_size=int(runtime["per_device_batch_size"]),
            workers=int(runtime.get("workers", 4)),
            context=context,
            seed=int(runtime["seed"]),
        )
        accumulation = int(runtime.get("gradient_accumulation_steps", 1))
        if accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        batches = cycle_loader(loader, sampler, consumed_batches=first_step * accumulation)
        precision = str(runtime.get("precision", "bfloat16"))
        autocast_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
        use_autocast = context.device.type == "cuda" and precision in {"bfloat16", "float16"}
        output = Path(runtime["output_dir"])
        for step in range(first_step + 1, total_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            value = 0.0
            for micro_step in range(accumulation):
                batch = move_batch_to_device(next(batches), context.device)
                synchronization = (
                    model.no_sync() if context.world_size > 1 and micro_step + 1 < accumulation else nullcontext()
                )
                with (
                    synchronization,
                    torch.autocast(
                        device_type=context.device.type,
                        dtype=autocast_dtype,
                        enabled=use_autocast,
                    ),
                ):
                    losses = environment_encoding_step(
                        model=model,
                        visual_encoder=vision,
                        batch=batch,
                        loss_config=loss_config,
                    )
                    loss = losses.total / accumulation
                loss.backward()
                value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            if context.is_primary and step % int(runtime.get("log_every_steps", 10)) == 0:
                print(json.dumps({"step": step, "loss": value, "lr": scheduler.get_last_lr()[0]}), flush=True)
            save_every = int(runtime.get("checkpoint_every_steps", 1000))
            if context.is_primary and (step % save_every == 0 or step == total_steps):
                save_training_checkpoint(
                    output / f"environment-encoding-step-{step:08d}.pt",
                    stage="environment_encoding",
                    step=step,
                    model=unwrap_model(model),
                    model_config=model_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
    finally:
        finish_distributed(context)


if __name__ == "__main__":
    main()
