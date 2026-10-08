"""CTE, task-memory, and EAP training/inference composition for RoboTwin."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.zeva.cte_eap import SCHEMA
from openpi.zeva.cte_eap import ZevaCTE
from openpi.zeva.cte_eap import ZevaCTEConfig
from openpi.zeva.cte_eap import ZevaEffectActionPrior
from openpi.zeva.retrieval import CausalRetrievalHead
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS

CTE_SCHEMAS = {SCHEMA, "zeva-behavior-effect-cte-v1"}  # Earlier saved checkpoints.


def canonical_cte_state_dict(state):
    """Rename earlier Mamba block keys without altering any tensor values."""
    names = {
        "mamba_v": "temporal.0",
        "mamba_a": "temporal.1",
        "mamba_b": "temporal.2",
        "norm_v": "temporal_norm.0",
        "norm_a": "temporal_norm.1",
        "norm_b": "temporal_norm.2",
        "attn_va": "vision_from_action",
        "attn_av": "action_from_vision",
        "attn_b": "interaction_from_transition",
        "norms": "attention_norm",
    }
    result = {}
    for key, tensor in state.items():
        parts = key.split(".")
        canonical_key = key
        if len(parts) >= 4 and parts[0] == "blocks" and parts[2] in names:
            canonical_key = ".".join([*parts[:2], names[parts[2]], *parts[3:]])
        if canonical_key in result:
            raise ValueError(f"Duplicate CTE key after checkpoint conversion: {canonical_key}")
        result[canonical_key] = tensor
    return result


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cte_checkpoint(payload, *, exploratory_epoch40=False):
    """Enforce the selected Stage-1 endpoint before Stage-2 uses its weights."""
    if exploratory_epoch40:
        expected_epoch, expected_step = 40, 26160
    else:
        expected_epoch, expected_step = 80, None
    if (
        payload.get("epoch") != expected_epoch
        or (expected_step is not None and payload.get("step") != expected_step)
        or not payload.get("manifest", {}).get("promotable")
        or not (payload.get("validation") or {}).get("stage1_gate")
    ):
        route = "explicit epoch-40 experiment" if exploratory_epoch40 else "epoch-80 recipe"
        raise ValueError(f"CTE checkpoint does not satisfy the {route} validation contract.")


def load_cte(path, device="cuda", *, require_ready=True, exploratory_epoch40=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") not in CTE_SCHEMAS:
        raise ValueError("New CTE+effect cannot load a legacy ZTE or ego encoder checkpoint.")
    if exploratory_epoch40 and not require_ready:
        raise ValueError("The explicit epoch-40 route cannot bypass validation.")
    if require_ready:
        validate_cte_checkpoint(payload, exploratory_epoch40=exploratory_epoch40)
    config = ZevaCTEConfig(**payload["config"])
    model = ZevaCTE(replace(config, vision_pretrained=False))
    model.config = config
    model.load_state_dict(canonical_cte_state_dict(payload["model_state_dict"]), strict=True)
    return model.to(device).requires_grad_(requires_grad=False).eval()


class TaskLanguageMemory(nn.Module):
    """Frozen language classifier selecting task-scoped CTE keys and values.

    A language-predicted task maps to its mean CTE key. No episode identifier
    or ground-truth task label is supplied at inference.
    """

    def __init__(self, bank, retrieval):
        super().__init__()
        if (
            bank.get("schema") not in {name + "-artifacts" for name in CTE_SCHEMAS}
            or bank.get("bank_subset") != "train"
        ):
            raise ValueError("Require a train-only CTE memory artifact.")
        if retrieval.get("source_feature") != "task_language":
            raise ValueError("Only task-language retrieval is authorized.")
        state = retrieval["model_state_dict"]
        self.head = CausalRetrievalHead(
            input_dim=state["network.0.weight"].shape[1],
            hidden_dim=state["network.0.weight"].shape[0],
            output_dim=state["network.4.weight"].shape[0],
            dropout=0.0,
        )
        self.head.load_state_dict(state, strict=True)
        self.task_names = tuple(retrieval["task_names"])
        new_tasks = tuple(bank["tasks"])
        if len(set(new_tasks)) != len(new_tasks) or not set(new_tasks).issubset(self.task_names):
            raise ValueError("New memory tasks must be unique and represented by the frozen language classifier.")
        entries = len(bank["keys"])
        if (
            bank["keys"].shape != (entries, 128)
            or bank["values"].shape != (entries, 256)
            or bank["task_ids"].shape != (entries,)
        ):
            raise ValueError("Memory must contain aligned [N,128] keys, [N,256] values and [N] task ids.")
        if not torch.isfinite(bank["keys"]).all() or not torch.isfinite(bank["values"]).all():
            raise ValueError("Non-finite CTE memory.")
        if (bank["task_ids"] < 0).any() or (bank["task_ids"] >= len(new_tasks)).any():
            raise ValueError("Memory task id outside its declared task table.")
        if any(int((bank["task_ids"] == i).sum()) == 0 for i in range(len(new_tasks))):
            raise ValueError("Every declared memory task needs train-only entries.")
        self.register_buffer("language_prototypes", F.normalize(retrieval["task_prototypes"].float(), dim=-1))
        self.register_buffer("keys", bank["keys"].float())
        self.register_buffer("values", bank["values"].float())
        self.register_buffer("entry_task", bank["task_ids"].long())
        mapping = torch.tensor([new_tasks.index(n) if n in new_tasks else -1 for n in self.task_names])
        self.register_buffer("task_mapping", mapping)
        prototypes = torch.stack(
            [F.normalize(self.keys[self.entry_task == i].mean(0), dim=-1) for i in range(len(new_tasks))]
        )
        self.register_buffer("new_task_keys", prototypes)
        self.requires_grad_(requires_grad=False).eval()
        self.last_diagnostics = []

    def forward(self, task_language):
        language_scores = self.head(task_language) @ self.language_prototypes.T
        # Restrict the classifier to the fixed, predeclared memory task scope.
        # This uses no per-example task label and prevents a 50-task classifier
        # from crashing a valid ten-task deployment on an out-of-scope argmax.
        language_scores = language_scores.masked_fill(self.task_mapping[None] < 0, -torch.inf)
        confidence, predicted = language_scores.max(-1)
        self.last_diagnostics = [
            {"task": self.task_names[int(i)], "score": float(s)}
            for i, s in zip(predicted.detach().cpu(), confidence.detach().cpu(), strict=True)
        ]
        task = self.task_mapping[predicted]
        if (task < 0).any():
            raise ValueError("Language retrieved a task outside the frozen ten-task memory.")
        query = self.new_task_keys[task]
        # Top-5 cosine/softmax aggregation is restricted to the
        # language-retrieved task, never to the current trajectory.
        scores = query @ F.normalize(self.keys, dim=-1).T
        scores = scores.masked_fill(self.entry_task[None] != task[:, None], -torch.inf)
        values, indices = scores.topk(min(5, scores.shape[1]), dim=-1)
        return (values.softmax(-1).unsqueeze(-1) * self.values[indices]).sum(1)


class ZevaCTEEAPPolicy(nn.Module):
    POLICY_SCHEMA = SCHEMA
    ACTION_PRIOR_CLASS = ZevaEffectActionPrior

    def __init__(self, loader, cte, bank, retrieval):
        super().__init__()
        self.foundation = loader.foundation
        self.preprocessor, self.postprocessor = loader.preprocessor, loader.postprocessor
        self.action_normalizer = loader.action_normalizer
        self.tokenizer = loader._task_only_tokenizer  # noqa: SLF001
        self.register_buffer("language_table", loader.frozen_goal_embedding_table, persistent=False)
        self.cte = cte
        self.memory = TaskLanguageMemory(bank, retrieval)
        self.eap = self.ACTION_PRIOR_CLASS(dim=cte.config.d_model)
        self.eap.install(self.foundation.model)
        self.foundation.requires_grad_(requires_grad=False)
        self.reset()

    @classmethod
    def from_handoff(
        cls,
        handoff,
        foundation_checkpoint,
        cte_checkpoint,
        artifacts,
        retrieval_checkpoint,
        *,
        device="cuda",
        stage2_checkpoint=None,
        exploratory_epoch40=False,
    ):
        from safetensors.torch import load_model  # noqa: PLC0415

        from openpi.zeva.robotwin_policy import RobotWinZevaPolicy  # noqa: PLC0415

        bank = torch.load(artifacts, map_location="cpu", weights_only=False)
        if bank["cte_sha256"] != file_sha(cte_checkpoint):
            raise ValueError("New CTE and memory/live cache SHA differ.")
        cte = load_cte(cte_checkpoint, device, exploratory_epoch40=exploratory_epoch40)
        loader = RobotWinZevaPolicy.from_handoff(
            handoff,
            foundation_checkpoint=foundation_checkpoint,
            goal_embedding_checkpoint=Path(handoff) / "checkpoint/pretrained_model",
            install_injection_hooks=False,
            device=device,
        )
        retrieval = torch.load(retrieval_checkpoint, map_location="cpu", weights_only=False)
        policy = cls(loader, cte, bank, retrieval).to(device)
        policy.identity = {
            "schema": cls.POLICY_SCHEMA,
            "cte_sha256": file_sha(cte_checkpoint),
            "artifacts_sha256": file_sha(artifacts),
            "retrieval_sha256": file_sha(retrieval_checkpoint),
            "foundation_sha256": file_sha(Path(foundation_checkpoint) / "model.safetensors"),
        }
        if stage2_checkpoint:
            folder = Path(stage2_checkpoint)
            adapter = torch.load(folder / "zeva_adapter.pth", map_location=device, weights_only=False)
            if adapter["identity"] != policy.identity:
                raise ValueError("Stage2 lineage mismatch.")
            policy.eap.load_state_dict(adapter["eap"], strict=True)
            load_model(policy.foundation, str(folder / "model.safetensors"), strict=True)
        policy.foundation.requires_grad_(requires_grad=True)
        return policy.eval()

    def train(self, mode=True):  # noqa: FBT002 - torch.nn.Module signature
        super().train(mode)
        self.cte.eval()
        self.memory.eval()
        return self

    @torch.no_grad()
    def language(self, tasks):
        if isinstance(tasks, str):
            tasks = [tasks]
        prompts = [t.strip().replace("_", " ").replace("\n", " ") + "\n" for t in tasks]
        encoded = self.tokenizer(prompts, max_length=200, padding="max_length", truncation=True, return_tensors="pt")
        embeddings = F.embedding(encoded["input_ids"].to(self.language_table.device), self.language_table).float()
        mask = encoded["attention_mask"].to(embeddings.device).unsqueeze(-1)
        return (embeddings * mask).sum(1) / mask.sum(1).clamp_min(1)

    def forward(self, processed, phase, effect, task_language):
        global_token = self.memory(task_language)
        prior = self.eap.activate(global_token, phase.detach(), effect.detach())
        try:
            output = self.foundation(processed)
            flow = output[0] if isinstance(output, tuple) else output
            prior_loss = self.eap.prior_loss(prior, processed["action"])
            loss = flow.mean() + prior_loss
            return loss, {"flow": flow.mean().detach(), "nll": prior_loss.detach()}
        finally:
            self.eap.clear()

    def reset(self, *, scope="episode"):
        if scope != "episode":
            raise ValueError("This CTE cache must reset at complete episode boundaries.")
        self._cache = None
        self._global_token = None
        self.memory.last_diagnostics = []
        self.eap.clear()
        self.foundation.reset()

    def retrieval_diagnostics(self):
        return self.memory.last_diagnostics

    @torch.inference_mode()
    def extract_vlm_features(self, batch):
        # Reuse the baseline trace extractor with inactive EAP, not a second
        # encoder or a different tokenizer/state/image preparation path.
        from openpi.zeva.robotwin_policy import RobotWinZevaPolicy  # noqa: PLC0415

        if self.eap._active is not None:  # noqa: SLF001
            raise RuntimeError("Trace extraction must run outside a conditioned forward.")
        return RobotWinZevaPolicy.extract_vlm_features(self, batch)

    @torch.no_grad()
    def infer_chunk(self, raw, *, executed_actions=None):
        processed = self.preprocessor(raw)
        if executed_actions is not None:
            executed_actions = torch.as_tensor(
                executed_actions, dtype=torch.float32, device=processed[ROBOTWIN_CAMERA_KEYS[0]].device
            )
            if executed_actions.ndim == 2:
                executed_actions = executed_actions.unsqueeze(0)
        normalized = self.predict_action_chunk(processed, task=raw["task"], executed_actions=executed_actions)
        return self.postprocessor(normalized)

    @torch.no_grad()
    def predict_action_chunk(self, processed, *, task, executed_actions=None):
        if self.training:
            raise RuntimeError("Deployment must use policy.eval().")
        views = torch.stack(
            [
                F.interpolate(processed[key].float(), (224, 224), mode="bilinear", align_corners=False, antialias=True)
                .mul(2)
                .sub(1)
                for key in ROBOTWIN_CAMERA_KEYS
            ],
            dim=1,
        )
        previous = None if executed_actions is None else self.action_normalizer.normalize(executed_actions)
        phase, effect, self._cache = self.cte.step(views, previous, self._cache)
        if self._global_token is None:
            self._global_token = self.memory(self.language(task))
        self.eap.activate(self._global_token, phase, effect)
        try:
            return self.foundation.predict_action_chunk(processed)
        finally:
            self.eap.clear()
