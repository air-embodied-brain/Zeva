"""Train the ZeVA causal transition encoder and boundary interaction token."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tyro

from openpi.zeva.cte_eap import SCHEMA, ZevaCTE, ZevaCTEConfig, cte_loss
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS, RobotWinHandoff, MeanStdActionNormalizer
from openpi.zeva.robotwin_data import TorchCodecRoboTwinDataset


@dataclasses.dataclass
class Args:
    dataset_root: str
    save_dir: str
    handoff_root: str
    task_subset: str
    epochs: int = 80
    batch_size: int = 8
    workers: int = 4
    effect_weight: float = 0.2
    seed: int = 1000
    # Smoke produces explicitly non-promotable artifacts, never an epoch80 model.
    smoke_steps: int = 0
    resume: str | None = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_views(images, image_size=224):
    """Use independently resized views, not a resized three-camera panorama."""
    return torch.stack([
        F.interpolate(images[key].float().div(255), (image_size, image_size),
                      mode="bilinear", align_corners=False, antialias=True).mul(2).sub(1)
        for key in ROBOTWIN_CAMERA_KEYS
    ], dim=1)


class Episodes(Dataset):
    def __init__(self, manifest, subset, tasks):
        self.source = TorchCodecRoboTwinDataset(manifest, subset)
        self.tasks = tuple(tasks)
        self.indices = [i for i, r in enumerate(self.source.dataset._records)
                        if r["key"][1] in self.tasks and int(r["length"]) > 15]
        if not self.indices:
            raise ValueError(f"No {subset} episodes for the frozen task subset.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        record_index = self.indices[index]
        dataset = self.source.dataset
        record = dataset._records[record_index]
        starts = list(range(0, int(record["length"]) - 15, 15))
        frames = starts + [starts[-1] + 15]
        images = resize_views(self.source.read_images(record, frames))
        origin = int(dataset._cumulative[record_index])
        actions = torch.stack([dataset[origin + frame]["action"][:15] for frame in starts])
        return {"images": images, "actions": actions,
                "task_id": self.tasks.index(record["key"][1]), "record_index": record_index}


def collate(samples):
    n = max(len(s["actions"]) for s in samples)
    images = samples[0]["images"].new_zeros(len(samples), n + 1, *samples[0]["images"].shape[1:])
    actions = samples[0]["actions"].new_zeros(len(samples), n, 15, 16)
    valid = torch.zeros(len(samples), n, dtype=torch.bool)
    for i, s in enumerate(samples):
        length = len(s["actions"])
        images[i, :length + 1] = s["images"]
        actions[i, :length] = s["actions"]
        valid[i, :length] = True
    return {"images": images, "actions": actions, "valid": valid,
            "task_id": torch.tensor([s["task_id"] for s in samples]),
            "record_index": torch.tensor([s["record_index"] for s in samples])}


class PairedEpochSampler:
    """Full coverage, same-task pairs; no replacement or episode-index conditioning."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        groups = [[] for _ in self.dataset.tasks]
        for index, record_index in enumerate(self.dataset.indices):
            name = self.dataset.source.dataset._records[record_index]["key"][1]
            groups[self.dataset.tasks.index(name)].append(index)
        pairs, leftovers = [], []
        for group in groups:
            order = [group[i] for i in torch.randperm(len(group), generator=generator).tolist()]
            pairs.extend([order[i:i+2] for i in range(0, len(order)-1, 2)])
            if len(order) % 2:
                leftovers.append(order[-1])
        result = [i for p in torch.randperm(len(pairs), generator=generator).tolist() for i in pairs[p]]
        return iter(result + leftovers)


@torch.no_grad()
def validate(model, loader, normalizer, device):
    model.eval()
    sums = dict(action=0., action_zero=0., vision=0., vision_persistence=0., effect=0., effect_zero=0.)
    count = 0
    for batch in loader:
        images = batch["images"].to(device)
        actions = normalizer.normalize(batch["actions"].to(device))
        mask = batch["valid"].to(device)
        outputs = model(images, actions)
        rows = {
            "action": (outputs["pred_act"]-actions).square().mean((-1, -2)),
            "action_zero": actions.square().mean((-1, -2)),
            "vision": (outputs["pred_vis"]-outputs["target_vis"]).square().mean(-1),
            "vision_persistence": outputs["target_effect"].square().mean(-1),
            "effect": (outputs["pred_effect"]-outputs["target_effect"]).square().mean(-1),
            "effect_zero": outputs["target_effect"].square().mean(-1),
        }
        count += int(mask.sum())
        for key, value in rows.items():
            sums[key] += float(value[mask].sum())
    result = {key: value / max(count, 1) for key, value in sums.items()}
    result["transitions"] = count
    result["finite"] = count > 0 and all(math.isfinite(v) for v in result.values())
    result["action_gain"] = 1 - result["action"] / max(result["action_zero"], 1e-12)
    result["effect_gain"] = 1 - result["effect"] / max(result["effect_zero"], 1e-12)
    result["vision_gain"] = 1 - result["vision"] / max(result["vision_persistence"], 1e-12)
    result["stage1_gate"] = result["finite"] and min(result[k] for k in ("action_gain", "effect_gain", "vision_gain")) >= .05
    return result


def main(args: Args):
    if min(args.epochs, args.batch_size, args.workers + 1) <= 0:
        raise ValueError("epochs and batch_size must be positive; workers must be non-negative.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    handoff = RobotWinHandoff.from_root(args.handoff_root)
    normalizer = MeanStdActionNormalizer.from_stats_file(handoff.statistics)
    manifest = Path(args.dataset_root) / "adapter.json"
    tasks = json.loads(Path(args.task_subset).read_text())["task_names"]
    train, validation = [Episodes(manifest, subset, tasks) for subset in ("train", "validation")]
    def record_ids(ds):
        return {(*ds.source.dataset._records[i]["key"], int(ds.source.dataset._records[i]["episode_index"]))
                for i in ds.indices}
    if record_ids(train) & record_ids(validation):
        raise ValueError("Train/validation episode overlap.")
    sampler = PairedEpochSampler(train, args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.workers,
                            collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    config = ZevaCTEConfig()
    # Strict checkpoint load avoids a redundant pretrained download on resume.
    construction = dataclasses.replace(config, vision_pretrained=not bool(args.resume))
    model = ZevaCTE(construction).to(device)
    model.config = config
    optimizer = torch.optim.AdamW([
        {"params": model.vision_encoder.parameters(), "lr": 1e-5},
        {"params": [p for name, p in model.named_parameters() if p.requires_grad and "vision_encoder" not in name], "lr": 1e-4}
    ], weight_decay=1e-4)
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"schema": SCHEMA,
                "config": dataclasses.asdict(config), "args": dataclasses.asdict(args),
                "tasks": tasks, "train_episodes": len(train), "validation_episodes": len(validation),
                "train_keys": sorted(record_ids(train)), "validation_keys": sorted(record_ids(validation)),
                "adapter_sha256": sha(manifest), "statistics_sha256": sha(handoff.statistics),
                "runtime": {"torch":torch.__version__, "cuda":torch.version.cuda, "gpu":torch.cuda.get_device_name()},
                "source_sha256": {"trainer": sha(__file__), "model": sha(Path(__file__).parents[2]/"src/openpi/zeva/cte_eap.py")},
                "normalization": "baseline-mean-std", "decoder": "torchcodec",
                "recipe": "epoch80-with-held-out-validation",
                "promotable": not bool(args.smoke_steps)}
    start_epoch, step = 0, 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["schema"] != SCHEMA or checkpoint["manifest"]["config"] != identity["config"]:
            raise ValueError("Incompatible Stage1 schema/config.")
        for key in ("tasks", "train_keys", "validation_keys", "adapter_sha256", "statistics_sha256", "source_sha256"):
            if checkpoint["manifest"][key] != identity[key]:
                raise ValueError(f"Exact resume identity mismatch: {key}")
        if not checkpoint["manifest"]["promotable"]:
            raise ValueError("A smoke checkpoint cannot resume full training.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["rng_cpu"])
        torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        start_epoch, step = checkpoint["epoch"], checkpoint["step"]
    elif any(output.iterdir()):
        raise ValueError("Fresh training requires an empty output directory.")
    (output / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
    for epoch in range(start_epoch, args.epochs):
        sampler.epoch = epoch
        model.train()
        last_end = time.perf_counter()
        for batch in train_loader:
            batch_ready = time.perf_counter()
            actions = normalizer.normalize(batch["actions"].to(device, non_blocking=True))
            predictions = model(batch["images"].to(device, non_blocking=True), actions)
            loss, metrics = cte_loss(predictions, actions, batch["valid"].to(device), batch["task_id"].to(device), effect_weight=args.effect_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CTE loss.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            if float(norm) == 0:
                raise RuntimeError("CTE backward produced zero gradients.")
            optimizer.step()
            model.update_ema()
            step += 1
            if step % 10 == 0 or args.smoke_steps:
                torch.cuda.synchronize()
                print(json.dumps({"epoch": epoch+1, "step": step, "loss": float(loss), "grad_norm": float(norm),
                                  "data_seconds":batch_ready-last_end, "update_seconds":time.perf_counter()-batch_ready,
                                  **{k: float(v) for k, v in metrics.items()}}), flush=True)
            last_end = time.perf_counter()
            if args.smoke_steps and step >= args.smoke_steps:
                print("SMOKE_COMPLETED; no promotable checkpoint", flush=True)
                return
        metrics = validate(model, val_loader, normalizer, device) if (epoch+1) % 5 == 0 else None
        checkpoint = {"schema": SCHEMA, "manifest": identity, "config": dataclasses.asdict(config),
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch+1, "step": step, "validation": metrics,
                      "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all()}
        path = output / f"cte_epoch_{epoch+1:03d}.pth"
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        print(json.dumps({"completed_epoch": epoch+1, "step": step, "validation": metrics}), flush=True)
    (output / "COMPLETED").write_text("epoch80 completed\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
