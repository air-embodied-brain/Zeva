"""Export new CTE train-only memory and deployment-identical recurrent H15 cache."""
from __future__ import annotations

import dataclasses
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import tyro

from openpi.zeva.cte_eap import SCHEMA
from openpi.zeva.cte_eap_policy import load_cte
from openpi.zeva.robotwin_contract import RobotWinHandoff, MeanStdActionNormalizer
from scripts.robotwin.train_cte import Episodes, sha


@dataclasses.dataclass
class Args:
    checkpoint: str
    dataset_root: str
    output: str
    task_subset: str
    handoff_root: str


@torch.no_grad()
def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(args.output)
    if output.exists():
        raise ValueError("Do not overwrite a prior CTE artifact.")
    cte = load_cte(args.checkpoint)
    tasks = json.loads(Path(args.task_subset).read_text())["task_names"]
    normalizer = MeanStdActionNormalizer.from_stats_file(RobotWinHandoff.from_root(args.handoff_root).statistics)
    keys, values, task_ids, bank_record_ids = [], [], [], []
    splits = {}
    for split in ("train", "validation"):
        dataset = Episodes(Path(args.dataset_root)/"adapter.json", split, tasks)
        records = {}
        for index in range(len(dataset)):
            sample = dataset[index]
            images = sample["images"].unsqueeze(0).cuda()
            actions = normalizer.normalize(sample["actions"].unsqueeze(0).cuda())
            phases, effects, cache = [], [], None
            for t in range(images.shape[1]):
                phase, effect, cache = cte.step(images[:, t], None if t == 0 else actions[:, t-1], cache)
                phases.append(phase[0]); effects.append(effect[0])
            phase = torch.stack(phases)
            # Pool exactly the same valid decision states as Stage1 (not terminal).
            if split == "train":
                keys.append(F.normalize(cte.task_head(phase[:-1]).mean(0), dim=-1).cpu())
                values.append(F.normalize(phase[:-1].mean(0), dim=-1).cpu())
                task_ids.append(sample["task_id"])
            record = dataset.source.dataset._records[sample["record_index"]]
            record_id = f"{record['key'][0]}:{record['key'][1]}:{record['episode_index']}"
            if split == "train":
                bank_record_ids.append(record_id)
            records[record_id] = {"phase": phase.cpu(), "effect": torch.stack(effects).cpu(),
                                  "frames": torch.arange(len(phase))*15}
        splits[split] = records
    if set(splits["train"]) & set(splits["validation"]):
        raise ValueError("Train/validation trajectory overlap.")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA+"-artifacts", "cte_sha256": sha(args.checkpoint),
               "adapter_sha256": sha(Path(args.dataset_root)/"adapter.json"),
               "tasks": tasks, "bank_subset": "train", "bank_record_ids": bank_record_ids,
               "keys": torch.stack(keys), "values": torch.stack(values), "task_ids": torch.tensor(task_ids),
               "splits": splits, "execution_horizon": 15, "policy_horizon": 50,
               "source_sha256": sha(__file__)}
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(json.dumps({"output": str(output), "sha256":sha(output), "memory_entries": len(keys),
                      "episodes": {k:len(v) for k,v in splits.items()}}), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
