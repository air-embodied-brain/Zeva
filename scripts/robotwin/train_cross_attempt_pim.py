"""Stage2-only training for the ZeVA cross-attempt PIM setting.

The frozen CTE supplies the current BIT exactly as in the validated CTE+EAP
policy.  PIM is initialized from that policy and trained with label-free,
train-only previous-attempt pairings.  Global batch is fixed at 256.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin
from safetensors.torch import save_model
import torch
from torch.utils.data import DataLoader, Dataset
import tyro

from openpi.zeva.cte_eap_policy import file_sha
from openpi.zeva.pim_policy import CROSS_ATTEMPT_PIM_POLICY_SCHEMA, ZevaCrossAttemptPIMPolicy
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS, prepare_robotwin_pi_image
from scripts.robotwin.build_cross_attempt_pim import PIM_ARTIFACT_SCHEMA
from openpi.zeva.robotwin_data import TorchCodecRoboTwinDataset


@dataclasses.dataclass
class Args:
    dataset_root: str
    cte_checkpoint: str
    cte_artifacts: str
    pim_artifacts: str
    retrieval_checkpoint: str
    foundation_checkpoint: str
    parent_stage2_checkpoint: str
    save_dir: str
    handoff_root: str
    steps: int = 2000
    warmup_steps: int = 500
    batch_size: int = 8
    accumulation: int = 8
    workers: int = 4
    seed: int = 20260919
    foundation_lr: float = 1e-6
    eap_lr: float = 1e-5
    pim_lr: float = 5e-5
    smoke_test: bool = False


def _bounded_trace(row: dict, capacity: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase, bit = row["phase"].float(), row["effect"].float()
    if phase.shape != bit.shape or phase.ndim != 2 or phase.shape[1] != 256:
        raise ValueError("PIM source must contain aligned [T,256] phase/BIT traces.")
    if len(phase) > capacity:
        index = torch.linspace(0, len(phase) - 1, capacity).round().long()
        phase, bit = phase[index], bit[index]
    padded_phase = torch.zeros(capacity, 256)
    padded_bit = torch.zeros(capacity, 256)
    mask = torch.zeros(capacity, dtype=torch.bool)
    padded_phase[: len(phase)], padded_bit[: len(bit)], mask[: len(phase)] = phase, bit, True
    return padded_phase, padded_bit, mask


class CrossAttemptPIMDecisions(Dataset):
    def __init__(self, manifest: Path, cte: dict, pim: dict, split: str):
        if cte["adapter_sha256"] != file_sha(manifest):
            raise ValueError("Dataset adapter differs from the CTE cache.")
        if pim.get("schema") != PIM_ARTIFACT_SCHEMA or pim["cte_artifacts_sha256"] != file_sha_cached(cte):
            raise ValueError("PIM pairings do not belong to this CTE artifact.")
        self.source = TorchCodecRoboTwinDataset(manifest, split)
        self.cache = cte["splits"][split]
        self.train_cache = cte["splits"]["train"]
        self.pairings = pim["pairings"][split]
        self.capacity = int(pim["max_entries_per_attempt"])
        self.samples = []
        for index, record in enumerate(self.source.dataset._records):
            record_id = f"{record['key'][0]}:{record['key'][1]}:{record['episode_index']}"
            if record_id not in self.cache:
                continue
            if record_id not in self.pairings:
                raise ValueError(f"Missing PIM pairing for {record_id}.")
            for timestep, frame in enumerate(self.cache[record_id]["frames"].tolist()):
                if frame >= int(record["length"]):
                    raise ValueError("CTE cache extends beyond the source episode.")
                self.samples.append((index, record_id, timestep, frame))
        if not self.samples:
            raise ValueError("No PIM Stage2 decisions.")

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
        histories = {}
        for mode in ("matched", "same_condition_far", "cross_condition"):
            previous_id = self.pairings[record_id][mode]
            if previous_id not in self.train_cache or previous_id == record_id:
                raise ValueError("PIM source must be a distinct train-only attempt.")
            phase, bit, mask = _bounded_trace(self.train_cache[previous_id], self.capacity)
            histories[mode] = {"phase": phase, "bit": bit, "mask": mask}
        return {
            "raw": raw,
            "phase": self.cache[record_id]["phase"][timestep],
            "effect": self.cache[record_id]["effect"][timestep],
            "pim": histories,
        }


_CTE_ARTIFACT_SHA: dict[int, str] = {}


def file_sha_cached(payload: dict) -> str:
    # The loaded payload carries its source path only in this process; the
    # trainer fills this cache before dataset construction to avoid rehashing.
    value = _CTE_ARTIFACT_SHA.get(id(payload))
    if value is None:
        raise RuntimeError("CTE artifact SHA was not registered.")
    return value


def main(args: Args) -> None:
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=args.accumulation, sync_with_dataloader=False
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)],
    )
    global_batch = args.batch_size * args.accumulation * accelerator.num_processes
    if args.smoke_test:
        if args.steps != 2 or (global_batch > 8 and global_batch != 256):
            raise ValueError("PIM smoke uses two updates at global batch <=8 or 256.")
    elif global_batch != 256 or args.steps != 2000 or args.warmup_steps != 500:
        raise ValueError("This recipe uses global batch 256, 500 warmup steps, and 2000 total steps.")
    torch.manual_seed(args.seed + accelerator.process_index)
    torch.cuda.manual_seed_all(args.seed + accelerator.process_index)
    cte = torch.load(args.cte_artifacts, map_location="cpu", weights_only=False)
    cte_sha = file_sha(args.cte_artifacts)
    _CTE_ARTIFACT_SHA[id(cte)] = cte_sha
    pim = torch.load(args.pim_artifacts, map_location="cpu", weights_only=False)
    if pim.get("cte_artifacts_sha256") != cte_sha or not pim.get("label_free"):
        raise ValueError("PIM artifact provenance mismatch.")
    policy = ZevaCrossAttemptPIMPolicy.from_parent_handoff(
        args.handoff_root, args.foundation_checkpoint, args.cte_checkpoint,
        args.cte_artifacts, args.retrieval_checkpoint, args.parent_stage2_checkpoint,
        device=str(accelerator.device),
    )
    policy.foundation.model.gradient_checkpointing_enable()
    policy.train()
    dataset = CrossAttemptPIMDecisions(Path(args.dataset_root) / "adapter.json", cte, pim, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    pim_prefixes = ("pim_phase.", "pim_bit.", "pim_query.", "pim_projector.", "pim_to_global.")
    pim_params, eap_params = [], []
    for name, parameter in policy.eap.named_parameters():
        (pim_params if name.startswith(pim_prefixes) else eap_params).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": policy.foundation.parameters(), "lr": 0.0, "target_lr": args.foundation_lr},
        {"params": eap_params, "lr": 0.0, "target_lr": args.eap_lr},
        {"params": pim_params, "lr": args.pim_lr, "target_lr": args.pim_lr},
    ], weight_decay=1e-4)
    output = Path(args.save_dir)
    nonempty = torch.tensor(int(output.exists() and any(output.iterdir())), device=accelerator.device)
    if accelerator.reduce(nonempty, reduction="sum").item():
        raise ValueError("Fresh PIM Stage2 must not overwrite another run.")
    accelerator.wait_for_everyone()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": CROSS_ATTEMPT_PIM_POLICY_SCHEMA + "-training-v1",
        "setting_id": "cross-attempt",
        "args": dataclasses.asdict(args),
        "lineage": policy.identity, "global_batch": global_batch, "decision_count": len(dataset),
        "cte_artifacts_sha256": cte_sha, "pim_artifacts_sha256": file_sha(args.pim_artifacts),
        "sampling_schedule": ["matched"] * 5 + ["same_condition_far"] * 2
                             + ["cross_condition"] + ["none"] * 2,
        "recipe": "runtime-smoke" if args.smoke_test else "global256-warmup500-fixed2000",
        "source_sha256": file_sha(__file__),
    }
    if accelerator.is_main_process:
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    preprocess, language = policy.preprocessor, policy.language
    policy, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
    schedule = manifest["sampling_schedule"]
    warmup_schedule = ["matched"] * 5 + ["same_condition_far"] * 2 + ["cross_condition"]
    step = epoch = 0
    while step < args.steps:
        for item in loader:
            if step == args.warmup_steps and not args.smoke_test:
                for group in optimizer.param_groups:
                    group["lr"] = group["target_lr"]
            active_schedule = warmup_schedule if step < args.warmup_steps else schedule
            mode = active_schedule[step % len(active_schedule)]
            raw = item["raw"]
            processed = preprocess(raw)
            lang = language(raw["task"])
            if mode == "none":
                selected = item["pim"]["matched"]
                include_pim = False
            else:
                selected = item["pim"][mode]
                include_pim = True
            with accelerator.accumulate(policy):
                loss, metrics = policy(
                    processed, item["phase"].float(), item["effect"].float(), lang,
                    pim_phase=selected["phase"].float(), pim_bit=selected["bit"].float(),
                    pim_mask=selected["mask"], include_pim=include_pim,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PIM Stage2 loss.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if step < args.warmup_steps:
                        current = accelerator.unwrap_model(policy)
                        clipped = [parameter for name, parameter in current.eap.named_parameters()
                                   if name.startswith(pim_prefixes)]
                    else:
                        clipped = policy.parameters()
                    norm = accelerator.clip_grad_norm_(clipped, 1.0)
                    if not torch.isfinite(norm) or norm == 0:
                        current = accelerator.unwrap_model(policy)
                        gradients = {
                            name: (None if parameter.grad is None else float(parameter.grad.float().norm()))
                            for name, parameter in current.eap.named_parameters()
                            if name.startswith(pim_prefixes)
                        }
                        accelerator.print(json.dumps({"invalid_grad_norm": float(norm),
                                                       "pim_gradients": gradients}))
                        raise RuntimeError(f"Invalid PIM Stage2 gradient norm: {float(norm)}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                if step % 10 == 0 or args.smoke_test:
                    accelerator.print(json.dumps({"step": step, "epoch": epoch, "mode": mode,
                                                   "loss": float(loss), "grad_norm": float(norm),
                                                   **{key: float(value) for key, value in metrics.items()}}))
                if step % 500 == 0 or (args.smoke_test and step == args.steps):
                    accelerator.wait_for_everyone()
                    folder = output / f"{step:06d}"
                    folder.mkdir(exist_ok=True)
                    torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                               folder / f"rng_rank{accelerator.process_index}.pth")
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model = accelerator.unwrap_model(policy)
                        save_model(model.foundation, str(folder / "model.safetensors"))
                        torch.save({"identity": model.identity, "eap": model.eap.state_dict()},
                                   folder / "zeva_adapter.pth")
                        torch.save({"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                                    "manifest": manifest}, folder / "training_state.pth")
                        (folder / "COMPLETE").write_text("complete PI, EAP, PIM, optimizer and rank RNG\n")
                    accelerator.wait_for_everyone()
                if step >= args.steps:
                    break
        epoch += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        message = ("runtime smoke completed\n" if args.smoke_test else
                   "2000 optimizer steps completed\n")
        (output / "COMPLETED").write_text(message)


if __name__ == "__main__":
    main(tyro.cli(Args))
