"""Train the RoboTwin foundation policy with ZeVA CTE, BIT, and EAP."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, GradientAccumulationPlugin
from safetensors.torch import save_model
import torch
from torch.utils.data import Dataset, DataLoader
import tyro

from openpi.zeva.cte_eap_policy import ZevaCTEEAPPolicy, file_sha
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS, prepare_robotwin_pi_image
from openpi.zeva.robotwin_data import TorchCodecRoboTwinDataset


@dataclasses.dataclass
class Args:
    dataset_root: str
    cte_checkpoint: str
    artifacts: str
    retrieval_checkpoint: str
    foundation_checkpoint: str
    save_dir: str
    handoff_root: str
    steps: int = 5000
    batch_size: int = 8
    accumulation: int = 4
    workers: int = 4
    seed: int = 1000


class Decisions(Dataset):
    def __init__(self, manifest, artifact, split):
        if artifact["adapter_sha256"] != file_sha(manifest):
            raise ValueError("Dataset adapter differs from the new CTE cache.")
        self.source = TorchCodecRoboTwinDataset(manifest, split)
        self.cache = artifact["splits"][split]
        self.samples = []
        for i, record in enumerate(self.source.dataset._records):
            record_id = f"{record['key'][0]}:{record['key'][1]}:{record['episode_index']}"
            if record_id in self.cache:
                for t, frame in enumerate(self.cache[record_id]["frames"].tolist()):
                    if frame >= int(record["length"]):
                        raise ValueError("CTE cache extends beyond the actual episode.")
                    self.samples.append((i, record_id, t, frame))
        if not self.samples:
            raise ValueError("No Stage2 decisions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        i, record_id, t, frame = self.samples[index]
        dataset = self.source.dataset
        record = dataset._records[i]
        raw = dict(dataset[int(dataset._cumulative[i])+frame])
        images = self.source.read_images(record, [frame])
        for key in ROBOTWIN_CAMERA_KEYS:
            raw[key] = prepare_robotwin_pi_image(images[key][0], name=key)
        return {"raw": raw, "phase": self.cache[record_id]["phase"][t],
                "effect": self.cache[record_id]["effect"][t]}


def main(args):
    accelerator = Accelerator(mixed_precision="bf16",
                              gradient_accumulation_plugin=GradientAccumulationPlugin(
                                  num_steps=args.accumulation, sync_with_dataloader=False),
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True,
                                                                           broadcast_buffers=False)])
    global_batch = args.batch_size * args.accumulation * accelerator.num_processes
    if global_batch != 256 or args.steps != 5000:
        raise ValueError("This recipe uses global batch 256 and 5000 optimizer steps.")
    torch.manual_seed(args.seed + accelerator.process_index)
    torch.cuda.manual_seed_all(args.seed + accelerator.process_index)
    policy = ZevaCTEEAPPolicy.from_handoff(
        args.handoff_root,
        args.foundation_checkpoint,
        args.cte_checkpoint,
        args.artifacts,
        args.retrieval_checkpoint,
        device=str(accelerator.device),
    )
    policy.foundation.model.gradient_checkpointing_enable()
    policy.train()
    artifact = torch.load(args.artifacts, map_location="cpu", weights_only=False)
    dataset = Decisions(Path(args.dataset_root)/"adapter.json", artifact, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    optimizer = torch.optim.AdamW([
        {"params": policy.foundation.parameters(), "lr": 5e-6},
        {"params": policy.eap.parameters(), "lr": 5e-5},
    ], weight_decay=1e-4)
    output = Path(args.save_dir)
    # Check once, before any rank writes a manifest; avoid a first-launch race.
    nonempty = torch.tensor(int(output.exists() and any(output.iterdir())), device=accelerator.device)
    nonempty = accelerator.reduce(nonempty, reduction="sum")
    if nonempty.item():
        raise ValueError("Fresh Stage2 must not overwrite or implicitly resume another run.")
    accelerator.wait_for_everyone()
    output.mkdir(parents=True, exist_ok=True)
    identity = {"args": dataclasses.asdict(args), "lineage": policy.identity, "global_batch": global_batch,
                "decision_count": len(dataset), "source_sha256": file_sha(__file__),
                "recipe": "global256-fixed5000"}
    if accelerator.is_main_process:
        (output/"manifest.json").write_text(json.dumps(identity, indent=2)+"\n")
    preprocess, language = policy.preprocessor, policy.language
    policy, optimizer, loader = accelerator.prepare(policy, optimizer, loader)
    step, epoch = 0, 0
    while step < args.steps:
        for item in loader:
            raw = item["raw"]
            lang = language(raw["task"])
            processed = preprocess(raw)
            with accelerator.accumulate(policy):
                loss, metrics = policy(processed, item["phase"].float(), item["effect"].float(), lang)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Stage2 loss.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(policy.parameters(), 1.)
                    if not torch.isfinite(norm) or norm == 0:
                        raise RuntimeError("Invalid Stage2 gradient norm.")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                if step % 10 == 0:
                    accelerator.print(json.dumps({"step":step, "epoch":epoch, "loss":float(loss),
                                                  **{k:float(v) for k,v in metrics.items()}}))
                if step % 500 == 0:
                    accelerator.wait_for_everyone()
                    folder = output/f"{step:06d}"
                    folder.mkdir(exist_ok=True)
                    torch.save({"cpu":torch.get_rng_state(), "cuda":torch.cuda.get_rng_state_all()},
                               folder/f"rng_rank{accelerator.process_index}.pth")
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model = accelerator.unwrap_model(policy)
                        save_model(model.foundation, str(folder/"model.safetensors"))
                        torch.save({"identity": model.identity, "eap": model.eap.state_dict()}, folder/"zeva_adapter.pth")
                        torch.save({"optimizer":optimizer.state_dict(), "step":step, "epoch":epoch,
                                    "manifest":identity}, folder/"training_state.pth")
                        (folder/"COMPLETE").write_text("complete model, EAP, optimizer, and rank RNG\n")
                    accelerator.wait_for_everyone()
                if step >= args.steps:
                    break
        epoch += 1
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        (output/"COMPLETED").write_text("5000 optimizer steps completed\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
