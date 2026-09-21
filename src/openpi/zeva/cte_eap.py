"""ZeVA causal transition encoder (CTE) and effect action prior (EAP).

Source attribution is kept separately in ``THIRD_PARTY_NOTICES.md``; public
model concepts and checkpoint contracts use ZeVA terminology.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as F


SCHEMA = "zeva-cte-bit-eap-v1"


@dataclass
class ZevaCTEConfig:
    d_model: int = 256
    n_layers: int = 4
    dropout: float = 0.1
    ema_decay: float = 0.99
    vision_pretrained: bool = True
    action_dim: int = 16
    executed_horizon: int = 15
    policy_horizon: int = 50
    num_views: int = 3
    image_size: int = 224


class ZevaCTE(nn.Module):
    """Vision/previous-action/interaction streams with a forward effect head.

    Inputs are three separately resized views in [-1,1]. The action stream
    consumes the previous *ordered H15 chunk*, never the current expert action.
    The effect token predicts the next visual feature difference from the same
    causal representation used at deployment. Future images are loss targets only.
    """

    def __init__(self, config: ZevaCTEConfig):
        super().__init__()
        from openpi.zeva.tri_stream import TriStreamBlock
        from openpi.zeva.vision import LightweightVisionEncoder

        self.config = config
        d = config.d_model
        self.vision_encoder = LightweightVisionEncoder(
            d, pretrained=config.vision_pretrained, num_views=config.num_views
        )
        self.target_vision_encoder = copy.deepcopy(self.vision_encoder).requires_grad_(False)
        self.action_proj = nn.Linear(config.action_dim * config.executed_horizon, d)
        self.action_sos = nn.Parameter(torch.randn(1, 1, d))
        self.cte_token = nn.Parameter(torch.randn(1, 1, d))
        self.norm_input = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([TriStreamBlock(config, block_idx=i) for i in range(config.n_layers)])
        self.norm_final = nn.LayerNorm(d)
        self.action_predictor = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                             nn.Linear(d, config.executed_horizon * config.action_dim))
        self.vision_predictor = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.task_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 128))
        self.progress_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 128))
        self.effect_predictor = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.logit_scale = nn.Parameter(torch.ones(()) * 4.0)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Batch statistics across trajectory frames would leak future observations.
        # Keep pretrained running statistics while allowing visual weights to learn.
        for module in self.vision_encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        self.target_vision_encoder.eval()
        return self

    @torch.no_grad()
    def update_ema(self):
        for online, target in zip(self.vision_encoder.parameters(), self.target_vision_encoder.parameters(), strict=True):
            target.lerp_(online, 1 - self.config.ema_decay)
        for online, target in zip(self.vision_encoder.buffers(), self.target_vision_encoder.buffers(), strict=True):
            target.copy_(online)

    def _visual(self, images, *, target=False):
        if images.ndim != 6 or images.shape[2:4] != (self.config.num_views, 3):
            raise ValueError("CTE expects [B,T,V,3,H,W] images, individually resized, in [-1,1].")
        encoder = self.target_vision_encoder if target else self.vision_encoder
        return encoder(images.flatten(0, 1)).reshape(*images.shape[:2], -1)

    def _streams(self, visual, shifted_actions, cache=None):
        v, a = self.norm_input(visual), self.norm_input(shifted_actions)
        b = self.cte_token.expand_as(v)
        for block in self.blocks:
            v, a, b = block(v, a, b, inference_params=cache)
        return v, a, self.norm_final(b)

    def forward(self, images, actions):
        """images: T+1 observations; actions: T executed, normalized H15 chunks."""
        batch, length = actions.shape[:2]
        if actions.shape[2:] != (self.config.executed_horizon, self.config.action_dim):
            raise ValueError("CTE requires ordered H15 × EEF16 executed actions.")
        if images.shape[:2] != (batch, length + 1):
            raise ValueError("Exactly one more observation than executed chunks is required.")
        visual = self._visual(images[:, :-1])
        with torch.no_grad():
            target = self._visual(images, target=True)
        embedded = self.action_proj(actions.flatten(2))
        shifted = torch.cat([self.action_sos.expand(batch, 1, -1), embedded[:, :-1]], dim=1)
        v, a, z = self._streams(visual, shifted)
        return {
            "pred_act": self.action_predictor(a).reshape_as(actions),
            "pred_vis": self.vision_predictor(v), "target_vis": target[:, 1:],
            "pred_effect": self.effect_predictor(z),
            "target_effect": (target[:, 1:] - target[:, :-1]).detach(),
            "z_seq": z, "z_global": self.task_head(z),
            "z_local": F.normalize(self.progress_head(z), dim=-1),
            "logit_scale": self.logit_scale,
        }

    @torch.no_grad()
    def step(self, image, previous_actions=None, cache=None):
        """One real replanning boundary; reset by passing cache=None and no action."""
        from mamba_ssm.utils.generation import InferenceParams

        if self.training:
            raise RuntimeError("Online CTE must be in eval mode.")
        if cache is None:
            if previous_actions is not None:
                raise ValueError("Initial boundary must use SOS, not an unobserved action.")
            cache = InferenceParams(max_seqlen=1_000_000, max_batch_size=image.shape[0])
            action = self.action_sos.expand(image.shape[0], 1, -1)
        else:
            expected = (image.shape[0], self.config.executed_horizon, self.config.action_dim)
            if previous_actions is None or previous_actions.shape != expected:
                raise ValueError("Subsequent boundaries require the actual previous normalized H15.")
            action = self.action_proj(previous_actions.flatten(1)).unsqueeze(1)
        _, _, z = self._streams(self._visual(image.unsqueeze(1)), action, cache)
        cache.seqlen_offset += 1
        return z[:, 0], self.effect_predictor(z[:, 0]), cache


def cte_loss(outputs, actions, valid_mask, task_ids, *, effect_weight=0.2):
    """CTE representation objectives plus effect MSE."""
    mask = valid_mask.bool()
    if not mask.any():
        raise ValueError("An all-padding CTE batch has no training objective.")
    # Average over executed time and sum over the real action coordinates.
    action = (outputs["pred_act"] - actions).square().sum(-1).mean(-1)[mask].mean()
    vision = (outputs["pred_vis"] - outputs["target_vis"]).square().sum(-1)[mask].mean()
    effect = (outputs["pred_effect"] - outputs["target_effect"]).square().sum(-1)[mask].mean()
    weight = mask.unsqueeze(-1)
    pooled = F.normalize((outputs["z_global"] * weight).sum(1) / weight.sum(1).clamp_min(1), dim=-1)
    episode_valid = mask.any(1)
    pooled, labels = pooled[episode_valid], task_ids[episode_valid]
    scale = outputs["logit_scale"].exp()
    logits = pooled @ pooled.T * scale
    positives = labels[:, None].eq(labels[None, :])
    positives.fill_diagonal_(False)
    # Include the self logit in the denominator.
    log_prob = logits - logits.logsumexp(-1, keepdim=True)
    global_loss = -(log_prob * positives).sum() / positives.sum().clamp_min(1)
    local = outputs["z_local"][mask]
    phase = F.cross_entropy(local @ local.T * scale, torch.arange(len(local), device=local.device))
    total = 0.1 * action + 0.2 * vision + 2.0 * global_loss + phase + effect_weight * effect
    return total, {"action": action, "vision": vision, "global": global_loss, "phase": phase, "effect": effect}


class ZevaActionPrior(nn.Module):
    """ZeVA action-prior network for the H50/EEF16 policy contract."""
    def __init__(self, dim=256, horizon=50, action_dim=16):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.temporal_waypoint = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, dim * 8))
        self.pos_emb = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm_global, self.norm_local = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dist_head = nn.Sequential(nn.Linear(dim, 512), nn.GELU(), nn.Linear(512, horizon * action_dim * 2))
        nn.init.normal_(self.dist_head[-1].weight, std=0.01)
        nn.init.zeros_(self.dist_head[-1].bias)

    def forward(self, global_token, phase):
        waypoints = self.temporal_waypoint(global_token).reshape(len(phase), 8, -1) + self.pos_emb
        kv = self.norm_global(waypoints)
        context, _ = self.attention(self.norm_local(phase).unsqueeze(1), kv, kv)
        params = self.dist_head(context[:, 0]).reshape(len(phase), self.horizon, self.action_dim, 2)
        return torch.distributions.Normal(params[..., 0], params[..., 1].clamp(-5, 2).exp())


class ZevaEffectActionPrior(nn.Module):
    """Global+effect prefix tokens and Gaussian-mean action-embedding residual.

    No context gate, output correction, teacher hinge or extra causal bank pool.
    One prior dropout mask is sampled per training forward, shared across all
    diffusion evaluations in that forward. Inference uses a fixed 0.5 scale.
    """
    def __init__(self, dim=256, prefix_dim=2048, expert_dim=1024):
        super().__init__()
        self.global_projector = nn.Linear(dim, prefix_dim)
        self.effect_projector = nn.Linear(dim, prefix_dim)
        self.action_prior = ZevaActionPrior(dim)
        self.prior_emb_proj = nn.Linear(16, expert_dim)
        nn.init.zeros_(self.prior_emb_proj.weight)
        nn.init.zeros_(self.prior_emb_proj.bias)
        self._active = None

    def activate(self, global_token, phase, effect, *, include_effect=True):
        if self._active is not None:
            raise RuntimeError("Nested EAP activation would corrupt the current diffusion forward.")
        prior = self.action_prior(global_token, phase)
        tokens = [self.global_projector(global_token)]
        if include_effect:
            tokens.append(self.effect_projector(effect))
        residual = self.prior_emb_proj(prior.loc)
        multiplier = (torch.rand((len(phase), 1, 1), device=phase.device) < 0.6).to(residual) if self.training else 0.5
        self._active = (torch.stack(tokens, dim=1), residual * multiplier)
        return prior

    def clear(self):
        self._active = None

    def prefix(self, embeddings, pad, attention):
        if self._active is None:
            return embeddings, pad, attention
        tokens = self._active[0].to(embeddings)
        valid = torch.ones(tokens.shape[:2], dtype=pad.dtype, device=pad.device)
        att = torch.zeros(tokens.shape[:2], dtype=attention.dtype, device=attention.device)
        return torch.cat([tokens, embeddings], 1), torch.cat([valid, pad], 1), torch.cat([att, attention], 1)

    def suffix(self, embeddings, pad, attention, adarms):
        if self._active is not None:
            residual = self._active[1].to(embeddings)
            if embeddings.shape != residual.shape:
                raise ValueError("Expected PI0.5 H50 action suffix; refusing a silently misaligned residual.")
            embeddings = embeddings + residual
        return embeddings, pad, attention, adarms

    def install(self, core):
        """Install on the released LeRobot PI0.5, not the old ZeVA conditioner."""
        if getattr(core, "_zeva_eap_installed", False):
            raise ValueError("EAP hooks are already installed.")
        prefix, suffix, owner = core.embed_prefix, core.embed_suffix, self

        def wrapped_prefix(_core, *args, **kwargs):
            return owner.prefix(*prefix(*args, **kwargs))

        def wrapped_suffix(_core, *args, **kwargs):
            return owner.suffix(*suffix(*args, **kwargs))

        core.embed_prefix = MethodType(wrapped_prefix, core)
        core.embed_suffix = MethodType(wrapped_suffix, core)
        core._zeva_eap_installed = True

    @staticmethod
    def prior_loss(prior, normalized_actions):
        if normalized_actions.shape[:2] != prior.loc.shape[:2]:
            raise ValueError("Gaussian NLL requires the full H50 target, not an H15 surrogate.")
        return -0.01 * prior.log_prob(normalized_actions[..., :16]).sum(-1).mean()
