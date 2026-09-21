"""Train ZeVA PIM as causal long-term memory within the current episode.

BIT remains the current-boundary short-term representation produced by the
frozen CTE.  For a decision at time ``t``, PIM may read only BITs from the same
episode at times ``[0, t)``. The parent policy is frozen bit-for-bit;
only the five PIM projection/retrieval modules are optimized.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import save_model
import torch
from torch.utils.data import DataLoader, Dataset
import tyro

from openpi.zeva.cte_eap_policy import file_sha
from openpi.zeva.pim_policy import EPISODE_PIM_POLICY_SCHEMA, ZevaEpisodePIMPolicy
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS, prepare_robotwin_pi_image
from openpi.zeva.robotwin_data import TorchCodecRoboTwinDataset


PIM_PREFIXES = ("pim_phase.", "pim_bit.", "pim_query.", "pim_projector.", "pim_to_global.")


@dataclasses.dataclass
class Args:
    dataset_root: str
    cte_checkpoint: str
    cte_artifacts: str
    retrieval_checkpoint: str
    foundation_checkpoint: str
    parent_stage2_checkpoint: str
    save_dir: str
    handoff_root: str
    steps: int = 2000
    batch_size: int = 32
    workers: int = 2
    seed: int = 20260920
    pim_lr: float = 5e-5
    max_entries: int = 64
    smoke_test: bool = False


def causal_episode_history(row: dict, timestep: int, capacity: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a bounded history containing strictly earlier same-episode BITs."""
    phase = torch.as_tensor(row["phase"], dtype=torch.float32)
    bit = torch.as_tensor(row["effect"], dtype=torch.float32)
    if phase.shape != bit.shape or phase.ndim != 2 or phase.shape[1] != 256:
        raise ValueError("Episode PIM source must contain aligned [T,256] phase/BIT traces.")
    if timestep <= 0 or timestep >= len(phase):
        raise ValueError("Episode PIM decisions require 0 < timestep < episode length.")
    phase, bit = phase[:timestep], bit[:timestep]
    if len(phase) > capacity:
        index = torch.linspace(0, len(phase) - 1, capacity).round().long().unique()
        phase, bit = phase[index], bit[index]
    padded_phase = torch.zeros(capacity, 256)
    padded_bit = torch.zeros(capacity, 256)
    mask = torch.zeros(capacity, dtype=torch.bool)
    padded_phase[: len(phase)] = phase
    padded_bit[: len(bit)] = bit
    mask[: len(phase)] = True
    return padded_phase, padded_bit, mask


class EpisodePIMDecisions(Dataset):
    """Train-only decisions paired with their causal history in the same episode."""

    def __init__(self, manifest: Path, cte: dict, *, capacity: int):
        if cte.get("adapter_sha256") != file_sha(manifest):
            raise ValueError("Dataset adapter differs from the frozen CTE artifact.")
        if capacity <= 0:
            raise ValueError("Episode PIM capacity must be positive.")
        self.source = TorchCodecRoboTwinDataset(manifest, "train")
        self.cache = cte["splits"]["train"]
        self.capacity = int(capacity)
        self.samples: list[tuple[int, str, int, int]] = []
        seen_records: set[str] = set()
        for source_index, record in enumerate(self.source.dataset._records):
            record_id = f"{record['key'][0]}:{record['key'][1]}:{record['episode_index']}"
            row = self.cache.get(record_id)
            if row is None:
                # The raw adapter contains all RoboTwin tasks, whereas this
                # experiment's frozen CTE artifact intentionally contains the
                # selected ten tasks only.
                continue
            seen_records.add(record_id)
            frames = torch.as_tensor(row["frames"]).tolist()
            for timestep, frame in enumerate(frames):
                if frame >= int(record["length"]):
                    raise ValueError("CTE trace extends beyond its source episode.")
                # t=0 has no long-term history and is the exact frozen Parent path.
                if timestep > 0:
                    self.samples.append((source_index, record_id, timestep, int(frame)))
        missing_source = set(self.cache) - seen_records
        if missing_source:
            raise ValueError(
                f"{len(missing_source)} frozen CTE train episodes are absent from raw data; "
                f"first={sorted(missing_source)[:3]}"
            )
        if not self.samples:
            raise ValueError("No causal within-episode PIM decisions were found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        source_index, record_id, timestep, frame = self.samples[index]
        dataset = self.source.dataset
        record = dataset._records[source_index]
        raw = dict(dataset[int(dataset._cumulative[source_index]) + frame])
        images = self.source.read_images(record, [frame])
        for key in ROBOTWIN_CAMERA_KEYS:
            raw[key] = prepare_robotwin_pi_image(images[key][0], name=key)
        phase, bit, mask = causal_episode_history(self.cache[record_id], timestep, self.capacity)
        return {
            "raw": raw,
            "phase": self.cache[record_id]["phase"][timestep],
            "effect": self.cache[record_id]["effect"][timestep],
            "pim_phase": phase,
            "pim_bit": bit,
            "pim_mask": mask,
            "record_id": record_id,
            "timestep": timestep,
        }


def _trainable_pim(policy: ZevaEpisodePIMPolicy) -> list[torch.nn.Parameter]:
    policy.requires_grad_(False)
    parameters = []
    for name, parameter in policy.eap.named_parameters():
        if name.startswith(PIM_PREFIXES):
            parameter.requires_grad_(True)
            parameters.append(parameter)
    if not parameters or any(parameter.requires_grad for parameter in policy.foundation.parameters()):
        raise RuntimeError("Only PIM modules may be trainable.")
    return parameters


def main(args: Args) -> None:
    accelerator = Accelerator(
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)],
    )
    global_batch = args.batch_size * accelerator.num_processes
    if args.smoke_test:
        if args.steps != 2 or global_batch not in {8, 256}:
            raise ValueError("Episode-PIM smoke requires two updates at global8 or global256.")
    elif args.steps != 2000 or global_batch != 256:
        raise ValueError("Episode-PIM is fixed at 2000 updates and global batch 256.")
    if args.max_entries != 64 or args.pim_lr != 5e-5:
        raise ValueError("This recipe uses PIM capacity 64 and learning rate 5e-5.")

    torch.manual_seed(args.seed + accelerator.process_index)
    torch.cuda.manual_seed_all(args.seed + accelerator.process_index)
    cte = torch.load(args.cte_artifacts, map_location="cpu", weights_only=False)
    policy = ZevaEpisodePIMPolicy.from_parent_handoff(
        args.handoff_root,
        args.foundation_checkpoint,
        args.cte_checkpoint,
        args.cte_artifacts,
        args.retrieval_checkpoint,
        args.parent_stage2_checkpoint,
        device=str(accelerator.device),
    )
    policy.foundation.model.gradient_checkpointing_enable()
    pim_parameters = _trainable_pim(policy)
    policy.train()
    dataset = EpisodePIMDecisions(Path(args.dataset_root) / "adapter.json", cte, capacity=args.max_entries)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(pim_parameters, lr=args.pim_lr, weight_decay=1e-4)

    output = Path(args.save_dir)
    nonempty = torch.tensor(int(output.exists() and any(output.iterdir())), device=accelerator.device)
    if accelerator.reduce(nonempty, reduction="sum").item():
        raise ValueError("Fresh Episode-PIM training must not overwrite another run.")
    accelerator.wait_for_everyone()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": EPISODE_PIM_POLICY_SCHEMA + "-training-v1",
        "setting_id": "within-episode",
        "args": dataclasses.asdict(args),
        "lineage": policy.identity,
        "global_batch": global_batch,
        "decision_count": len(dataset),
        "source_subset": "train-only",
        "history_contract": "same episode, strictly earlier BIT boundaries only",
        "cross_episode": False,
        "cross_attempt": False,
        "success_labels_used": False,
        "shared_parent_frozen": True,
        "trainable_modules": list(PIM_PREFIXES),
        "cte_artifacts_sha256": file_sha(args.cte_artifacts),
        "parent_stage2_checkpoint": args.parent_stage2_checkpoint,
        "recipe": "runtime-smoke" if args.smoke_test else "global256-fixed2000",
        "source_sha256": file_sha(__file__),
    }
    if accelerator.is_main_process:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    preprocess, language = policy.preprocessor, policy.language
    policy, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
    step = epoch = 0
    while step < args.steps:
        for item in loader:
            processed = preprocess(item["raw"])
            task_language = language(item["raw"]["task"])
            loss, metrics = policy(
                processed,
                item["phase"].float(),
                item["effect"].float(),
                task_language,
                pim_phase=item["pim_phase"].float(),
                pim_bit=item["pim_bit"].float(),
                pim_mask=item["pim_mask"],
                include_pim=True,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite Episode-PIM loss.")
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(pim_parameters, 1.0)
            if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
                raise RuntimeError(f"Invalid Episode-PIM gradient norm: {float(grad_norm)}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or args.smoke_test:
                accelerator.print(json.dumps({
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    **{key: float(value) for key, value in metrics.items()},
                }))
            if step % 500 == 0 or (args.smoke_test and step == args.steps):
                accelerator.wait_for_everyone()
                folder = output / f"{step:06d}"
                folder.mkdir(exist_ok=True)
                torch.save(
                    {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                    folder / f"rng_rank{accelerator.process_index}.pth",
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    model = accelerator.unwrap_model(policy)
                    save_model(model.foundation, str(folder / "model.safetensors"))
                    torch.save({"identity": model.identity, "eap": model.eap.state_dict()}, folder / "zeva_adapter.pth")
                    torch.save(
                        {"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "manifest": manifest},
                        folder / "training_state.pth",
                    )
                    (folder / "COMPLETE").write_text("complete frozen Parent, Episode-PIM, optimizer and rank RNG\n")
                accelerator.wait_for_everyone()
            if step >= args.steps:
                break
        epoch += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (output / "COMPLETED").write_text(
            "runtime smoke completed\n" if args.smoke_test else
            "2000 optimizer steps completed\n"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
