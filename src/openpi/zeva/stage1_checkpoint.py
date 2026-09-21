"""Schema-aware Stage1 loading; never reinterpret v2 weights as a v5 encoder."""

from dataclasses import asdict, fields, replace

from openpi.zeva.config import ZevaConfig


V2_SCHEMA = "zeva-robotwin-zte-stage1-v2-checkpoint"
LEGACY_SCHEMAS = frozenset({
    "zeva-robotwin-zte-stage1-checkpoint-v4",
    "zeva-robotwin-zte-stage1-checkpoint-v5",
})


def _declared_config(checkpoint: dict):
    schema = checkpoint.get("schema")
    if schema == V2_SCHEMA:
        from openpi.zeva.transition_encoder_v2 import TransitionEncoderV2Config

        return TransitionEncoderV2Config(**checkpoint["zte_config"])
    if schema in LEGACY_SCHEMAS:
        return ZevaConfig(**checkpoint["zte_config"])
    raise ValueError(f"Unsupported Stage1 checkpoint schema: {schema!r}")


def stage1_policy_config(checkpoint: dict) -> ZevaConfig:
    """Shared adapter dimensions, not a replacement for the encoder's full config.

    The constructor flag disables redundant pretrained downloads. The original
    declared encoder configuration remains in the checkpoint and on the loaded
    encoder as ``stage1_declared_config``.
    """
    declared = asdict(_declared_config(checkpoint))
    shared = {field.name for field in fields(ZevaConfig)}
    return ZevaConfig(**{key: value for key, value in declared.items()
                         if key in shared and key != "vision_pretrained"},
                      vision_pretrained=False)


def stage1_transition_horizon(checkpoint: dict) -> int:
    """Read explicit execution horizon and reject contradictory metadata."""
    config = _declared_config(checkpoint)
    manifest = checkpoint.get("manifest", {})
    candidates = [manifest.get("causal_transition_horizon"),
                  manifest.get("contract", {}).get("executed_horizon")]
    if checkpoint["schema"] == V2_SCHEMA:
        # Deployment must not silently default an absent execution contract.
        candidates.append(checkpoint["zte_config"].get("executed_action_steps"))
    values = [value for value in candidates if value is not None]
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Stage1 checkpoint requires an explicit positive transition horizon.")
    if len(set(values)) != 1:
        raise ValueError(f"Conflicting Stage1 execution horizons: {values}")
    horizon = values[0]
    if horizon > config.action_horizon:
        raise ValueError("Stage1 execution horizon exceeds the policy action horizon.")
    policy_horizon = manifest.get("contract", {}).get("policy_horizon")
    if policy_horizon is not None and policy_horizon != config.action_horizon:
        raise ValueError("Stage1 policy horizon conflicts with encoder configuration.")
    if checkpoint["schema"] == V2_SCHEMA and horizon != config.executed_action_steps:
        raise ValueError("Stage1 execution horizon conflicts with v2 encoder configuration.")
    return horizon


def load_stage1_encoder(checkpoint: dict, device="cpu"):
    """Strict-load the declared architecture without fetching pretrained weights.

    This helper does not freeze parameters: training versus inference ownership
    belongs to the caller. Production Mamba availability is enforced by the
    encoder itself; there is no implicit CPU/fallback architecture conversion.
    """
    declared = _declared_config(checkpoint)
    stage1_transition_horizon(checkpoint)
    construction = replace(declared, vision_pretrained=False)
    if checkpoint["schema"] == V2_SCHEMA:
        from openpi.zeva.transition_encoder_v2 import CausalTransitionEncoderV2

        encoder = CausalTransitionEncoderV2(construction)
    else:
        from openpi.zeva.transition_encoder import CausalTransitionEncoder

        encoder = CausalTransitionEncoder(construction)
    encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
    encoder.stage1_schema = checkpoint["schema"]
    encoder.stage1_declared_config = asdict(declared)
    return encoder.to(device)
