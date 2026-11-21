# SPDX-License-Identifier: OpenMDW-1.1
import json
import sys

import torch

from examples.training.zeva_two_stage import main


def test_two_stage_entrypoint_with_synthetic_adapters(tmp_path):
    torch.set_num_threads(1)
    ids = ("train_a", "train_b", "val")
    for name in ids:
        torch.save({"frames": torch.zeros(5, 3, 16, 16, dtype=torch.uint8),
                    "actions": torch.randn(4, 2, 2)}, tmp_path/f"{name}.pt")
    manifest = tmp_path/"episodes.json"
    manifest.write_text(json.dumps({"episodes": [
        {"id": name, "split": "validation" if name == "val" else "train", "file": f"{name}.pt"} for name in ids
    ]}))
    adapter = tmp_path/"adapter.py"
    adapter.write_text("""
import json, os, sys
from pathlib import Path
actor = Path(os.environ["ZEVA_POLICY_CHECKPOINT"])
if sys.argv[1] == "policy":
    assert (Path(os.environ["ZEVA_CTE_FEATURES"]) / "train_a.pt").is_file()
    actor.write_text("synthetic checkpoint")
else:
    assert actor.is_file()
    def episode(key, donors):
        return {"episode": key, "decisions": [{"candidates": [
            {"donor": name, "features": [float(i)], "distance": float(i),
             "gain": {"action": .25, "effect": .25}, "random_gain": {"action": 0., "effect": 0.}}
            for i, name in enumerate(donors)
        ]}]}
    payload = {"fit": [episode("train_a", ["train_b"])], "initial_donors": ["seed"],
               "stream": [episode("warmup", ["seed"]), episode("query", ["warmup"])], "warmup_trials": 1}
    Path(os.environ["ZEVA_PAIRED_EVIDENCE"]).write_text(json.dumps(payload))
""")
    cfg = tmp_path/"config.json"
    cfg.write_text(json.dumps({
        "output": "run", "episode_manifest": "episodes.json", "policy_checkpoint": "run/policy.txt",
        "paired_evidence": "run/evidence.json",
        "policy_command": [sys.executable, str(adapter), "policy"],
        "replay_command": [sys.executable, str(adapter), "evidence"],
        "stage1": {"device": "cpu", "epochs": 1, "batch_size": 2,
                   "model": {"action_dim": 2, "transition_steps": 2, "effect_window_transitions": 2,
                             "hidden_dim": 8, "phase_dim": 4, "effect_dim": 4,
                             "retrieval_dim": 4, "num_layers": 1, "num_heads": 2}},
        "memory": {"warmup": 1},
    }))
    main(["--config", str(cfg)])
    output = tmp_path/"run"
    features = torch.load(output/"cte_features"/"train_a.pt", weights_only=True)
    assert features["phase"].shape == (5, 4)
    assert features["effect_end"].tolist() == [2, 4]
    results = json.loads((output/"memory_replay.json").read_text())
    assert results["rolling_2"]["episodes"][0]["channels"]["action"]["mode"] == "nearest"
    assert results["rolling_2"]["summary"]["action"]["gain"] == .25
    main(["--config", str(cfg), "--phase", "memory"])
