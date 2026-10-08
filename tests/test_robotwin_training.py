"""Training objectives and gradient routes, without model/data downloads."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors.torch")

from openpi.zeva.cte_eap import ZevaCTE  # noqa: E402
from openpi.zeva.cte_eap import ZevaCTEConfig  # noqa: E402
from openpi.zeva.cte_eap import ZevaEffectActionPrior  # noqa: E402
from openpi.zeva.cte_eap import cte_loss  # noqa: E402
from openpi.zeva.cte_eap_policy import ZevaCTEEAPPolicy  # noqa: E402
from openpi.zeva.cte_eap_policy import validate_cte_checkpoint  # noqa: E402
from openpi.zeva.pim_policy import EpisodePersistentMemory  # noqa: E402
from openpi.zeva.pim_policy import ZevaEpisodePIMPolicy  # noqa: E402
from openpi.zeva.pim_policy import ZevaPIMEAP  # noqa: E402
from openpi.zeva.pim_policy import ZevaPIMPolicy  # noqa: E402


class TinyVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 4)
        self.register_buffer("running", torch.zeros(1))

    def forward(self, images):
        return self.proj(images.mean((-1, -2)).flatten(1))


class TinyStreams(torch.nn.Module):
    def forward(self, vision, action, interaction, inference_params=None):
        return vision, action, interaction + vision + action


def tiny_cte():
    # Exercise the actual CTE forward/loss with tiny encoders, not a substitute
    # for a Mamba training or rollout reproducibility check.
    model = ZevaCTE.__new__(ZevaCTE)
    torch.nn.Module.__init__(model)
    model.config = ZevaCTEConfig(d_model=4, action_dim=3, executed_horizon=2, num_views=1)
    model.vision_encoder = TinyVision()
    model.target_vision_encoder = copy.deepcopy(model.vision_encoder).requires_grad_(requires_grad=False)
    model.action_proj = torch.nn.Linear(6, 4)
    model.action_sos = torch.nn.Parameter(torch.randn(1, 1, 4))
    model.cte_token = torch.nn.Parameter(torch.randn(1, 1, 4))
    model.norm_input = torch.nn.LayerNorm(4)
    model.norm_final = torch.nn.LayerNorm(4)
    model.blocks = torch.nn.ModuleList([TinyStreams()])
    model.action_predictor = torch.nn.Linear(4, 6)
    model.vision_predictor = torch.nn.Linear(4, 4)
    model.effect_predictor = torch.nn.Linear(4, 4)
    model.task_head = torch.nn.Linear(4, 4)
    model.progress_head = torch.nn.Linear(4, 4)
    model.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
    return model


def test_cte_forward_loss_and_causal_gradient_route():
    torch.manual_seed(7)
    model = tiny_cte()
    images = torch.randn(2, 4, 1, 3, 2, 2, requires_grad=True)
    actions = torch.randn(2, 3, 2, 3, requires_grad=True)
    outputs = model(images, actions)
    assert outputs["pred_act"].shape == actions.shape
    assert not outputs["target_vis"].requires_grad
    assert not outputs["target_effect"].requires_grad
    changed = actions.detach().clone()
    changed[:, -1] += 100
    changed_outputs = model(images.detach(), changed)
    for key in ("pred_act", "pred_vis", "pred_effect", "z_seq"):
        torch.testing.assert_close(outputs[key], changed_outputs[key])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    loss, metrics = cte_loss(outputs, actions, mask, torch.zeros(2, dtype=torch.long))
    assert torch.isfinite(loss)
    assert set(metrics) == {"action", "vision", "global", "phase", "effect"}
    loss.backward()
    for head in (model.action_predictor, model.vision_predictor, model.effect_predictor, model.task_head):
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
    assert model.vision_encoder.proj.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.target_vision_encoder.parameters())
    assert images.grad[:, -1].count_nonzero() == 0


def test_cte_ema_updates_targets_and_copies_buffers():
    model = tiny_cte()
    before = model.target_vision_encoder.proj.weight.detach().clone()
    with torch.no_grad():
        model.vision_encoder.proj.weight.add_(2)
        model.vision_encoder.running.fill_(3)
    model.update_ema()
    torch.testing.assert_close(model.target_vision_encoder.proj.weight, before + 2 * (1 - model.config.ema_decay))
    torch.testing.assert_close(model.target_vision_encoder.running, model.vision_encoder.running)
    assert not model.target_vision_encoder.proj.weight.requires_grad


def test_cte_loss_rejects_all_padding():
    with pytest.raises(ValueError, match="all-padding"):
        cte_loss({}, torch.zeros(1, 2, 2, 3), torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, dtype=torch.long))


class TinyFoundation(torch.nn.Module):
    def __init__(self, eap, *, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        # Do not register EAP twice as a foundation submodule.
        object.__setattr__(self, "eap", eap)
        self.fail = fail

    def forward(self, processed):
        if self.fail:
            raise RuntimeError("synthetic foundation failure")
        tokens, residual = self.eap._active  # noqa: SLF001
        return self.weight.square() + tokens.square().mean() + residual.square().mean()


def tiny_policy(cls, *, fail=False):
    policy = cls.__new__(cls)
    torch.nn.Module.__init__(policy)
    policy.cte = torch.nn.Linear(4, 4).requires_grad_(requires_grad=False)
    policy.memory = torch.nn.Identity()
    eap_cls = ZevaEffectActionPrior if cls is ZevaCTEEAPPolicy else ZevaPIMEAP
    policy.eap = eap_cls(dim=4, prefix_dim=6, expert_dim=5)
    policy.foundation = TinyFoundation(policy.eap, fail=fail)
    return policy.train()


@pytest.mark.parametrize("cls", [ZevaCTEEAPPolicy, ZevaPIMPolicy, ZevaEpisodePIMPolicy])
def test_policy_training_forward_backpropagates_and_clears_conditioning(cls):
    torch.manual_seed(11)
    policy = tiny_policy(cls)
    phase = torch.randn(2, 4, requires_grad=True)
    effect = torch.randn(2, 4, requires_grad=True)
    history = torch.randn(2, 3, 4, requires_grad=True)
    options = (
        {}
        if cls is ZevaCTEEAPPolicy
        else {
            "pim_phase": history,
            "pim_bit": history,
            "pim_mask": torch.ones(2, 3, dtype=torch.bool),
        }
    )
    loss, metrics = policy({"action": torch.randn(2, 50, 16)}, phase, effect, torch.randn(2, 4), **options)
    assert torch.isfinite(loss)
    assert set(metrics) == {"flow", "nll"}
    assert policy.eap._active is None  # noqa: SLF001
    loss.backward()
    assert policy.foundation.weight.grad.abs() > 0
    assert policy.eap.action_prior.dist_head[0].weight.grad is not None
    if cls is not ZevaCTEEAPPolicy:
        assert policy.eap.pim_projector.weight.grad.abs().sum() > 0
    assert phase.grad is None
    assert effect.grad is None
    assert history.grad is None
    assert not policy.cte.training


@pytest.mark.parametrize("cls", [ZevaCTEEAPPolicy, ZevaPIMPolicy])
def test_training_forward_clears_conditioning_on_failure(cls):
    policy = tiny_policy(cls, fail=True)
    options = (
        {}
        if cls is ZevaCTEEAPPolicy
        else {
            "pim_phase": torch.ones(1, 1, 4),
            "pim_bit": torch.ones(1, 1, 4),
            "pim_mask": torch.ones(1, 1, dtype=torch.bool),
        }
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        policy({"action": torch.zeros(1, 50, 16)}, torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 4), **options)
    assert policy.eap._active is None  # noqa: SLF001


def test_cte_checkpoint_validation_keeps_both_training_routes():
    payload = {"epoch": 40, "step": 26160, "manifest": {"promotable": True}, "validation": {"stage1_gate": True}}
    validate_cte_checkpoint(payload, exploratory_epoch40=True)
    with pytest.raises(ValueError, match="epoch-80"):
        validate_cte_checkpoint(payload)
    payload["epoch"] = 80
    validate_cte_checkpoint(payload)


def test_within_episode_memory_remains_bounded_and_resettable():
    memory = EpisodePersistentMemory(max_entries=2)
    for value in range(3):
        token = torch.full((1, 4), float(value))
        memory.append_bit(token, token)
    phase, _ = memory.entries()
    assert phase[0, :, 0].tolist() == [1, 2]
    memory.reset_episode()
    assert memory.entries() == (None, None)
