# SPDX-License-Identifier: OpenMDW-1.1
"""Train and export the repository's CTE from an embodiment-neutral episode manifest."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .causal_transition_encoder import CausalTransitionEncoder, CausalTransitionEncoderConfig
from .cte_losses import CTELossConfig, causal_transition_encoder_loss


class Episodes(Dataset):
    def __init__(self, root, rows, config):
        self.root, self.rows, self.config = root, rows, config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        episode = torch.load(self.root / row["file"], map_location="cpu", weights_only=True)
        frames = episode["frames"].float()
        if episode["frames"].dtype == torch.uint8:
            frames /= 255.
        actions = episode["actions"].float()
        shape = (len(frames)-1, self.config.transition_steps, self.config.action_dim)
        if frames.ndim != 4 or frames.shape[1] != self.config.image_channels or len(frames) < 2 or actions.shape != shape:
            raise ValueError("Episode does not match the configured CTE input shape")
        if not torch.isfinite(actions).all() or not (torch.isfinite(frames) & (frames >= 0) & (frames <= 1)).all():
            raise ValueError("Expected finite actions and observations in [0, 1]")
        return frames, actions, int(row.get("semantic_id", 0))


def collate(rows):
    frames, actions, labels = zip(*rows, strict=True)
    length = max(map(len, frames))
    video = frames[0].new_zeros(len(rows), length, *frames[0].shape[1:])
    action = actions[0].new_zeros(len(rows), length-1, *actions[0].shape[1:])
    valid = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, (x, a) in enumerate(zip(frames, actions, strict=True)):
        video[i, :len(x)] = x
        action[i, :len(a)] = a
        valid[i, :len(x)] = True
    return video, action, valid, torch.tensor(labels)


def manifest_rows(manifest):
    rows = json.loads(manifest.read_text())["episodes"]
    identifiers = [row["id"] for row in rows]
    if len(set(identifiers)) != len(rows) or any(
        row["split"] not in {"train", "validation", "holdout"} or
        not row["id"].replace("_", "").replace("-", "").isalnum()
        for row in rows
    ):
        raise ValueError("Expected unique safe episode identifiers and declared splits")
    return rows


def _pass(model, batches, device, loss_config, optimizer=None):
    total = 0.
    model.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for video, action, valid, semantic in batches:
            video, action, valid, semantic = (value.to(device) for value in (video, action, valid, semantic))
            transitions = (valid[:, :-1] & valid[:, 1:]).unsqueeze(-1).expand_as(action[..., 0])
            pred = model(video, action, valid_mask=valid, transition_valid=transitions)
            loss = causal_transition_encoder_loss(pred, action, valid, semantic, loss_config)["total"]
            if not torch.isfinite(loss):
                raise ValueError("Non-finite Stage 1 loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                model.update_ema_target()
            total += float(loss.detach())
    return total / len(batches)


def _save(path, artifact):
    temporary = path.with_suffix(".partial.pt")
    torch.save(artifact, temporary)
    temporary.replace(path)


def train_cte(manifest, output, settings):
    rows = manifest_rows(manifest)
    train, val = ([row for row in rows if row["split"] == split] for split in ("train", "validation"))
    if not train or not val:
        raise ValueError("Stage 1 requires separate training and validation episodes")
    config = CausalTransitionEncoderConfig(**settings.get("model", {}))
    loss_config = CTELossConfig(**settings.get("loss", {}))
    epochs = settings.get("epochs", 80)
    batch = settings.get("batch_size", 8)
    workers = settings.get("workers", 0)
    device = settings.get("device", "cuda")
    if min(epochs, batch) < 1:
        raise ValueError("Stage 1 epochs and batch size must be positive")
    torch.manual_seed(settings.get("seed", 1000))
    model = CausalTransitionEncoder(config).to(device)
    loaders = [DataLoader(Episodes(manifest.parent, subset, config), batch_size=batch, shuffle=shuffle,
                          num_workers=workers, collate_fn=collate)
               for subset, shuffle in ((train, True), (val, False))]
    visual = list(model.visual_encoder.parameters())
    other = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("visual_encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": visual, "lr": settings.get("vision_lr", 1e-5)},
         {"params": other, "lr": settings.get("lr", 1e-4)}],
        weight_decay=settings.get("weight_decay", 1e-4),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1, epochs+1):
        loss = _pass(model, loaders[0], device, loss_config, optimizer)
        score = _pass(model, loaders[1], device, loss_config)
        print(json.dumps({"epoch": epoch, "train_loss": loss, "validation_loss": score}), flush=True)
        if score < best:
            best = score
            _save(output, {"model": model.state_dict(), "config": config.to_dict(),
                           "epoch": epoch, "validation_loss": score})


@torch.inference_mode()
def export_cte(manifest, checkpoint, directory, device):
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = CausalTransitionEncoderConfig(**artifact["config"])
    model = CausalTransitionEncoder(config).to(device).eval()
    model.load_state_dict(artifact["model"])
    rows = manifest_rows(manifest)
    dataset = Episodes(manifest.parent, rows, config)
    directory.mkdir(parents=True, exist_ok=True)
    for row, (frames, actions, _) in zip(rows, dataset, strict=True):
        encoded = model(frames[None].to(device), actions[None].to(device))
        frame_fields = ("retrieval", "phase", "visual_key", "next_action", "next_vision")
        pack = {name: encoded[name][0].cpu() for name in frame_fields}
        effect_valid = encoded["effect_complete"][0]
        effect_end = config.effect_window_transitions * torch.arange(1, len(effect_valid)+1, device=device)
        pack.update(effect_end=effect_end[effect_valid].cpu(), observed_effect=encoded["effect_post"][0, effect_valid].cpu(),
                    executed_action_effect_forecast=encoded["effect_pre"][0, effect_valid].cpu())
        _save(directory / (row["id"]+".pt"), pack)
