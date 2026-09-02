# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WebSocket inference server for the fixed-base RoboCasa365 Zeva policy with PIM.

The server uses OpenPI's WebsocketPolicyServer and speaks its msgpack+NumPy protocol:

- on connection, it sends an empty metadata dict;
- each client message is an observation dict;
- each response is a dict with ``action`` and, when enabled, ``video``.

Example:

  PYTHONPATH=. python -m cosmos_framework.scripts.action_policy_server_robocasa365 \
    --checkpoint-path /path/to/iter_000010000 \
    --experiment action_policy_robocasa365_atomic5_nano \
    --domain-name robocasa-panda-omron \
    --port 8000
"""

# Initialize the script runtime before any cosmos-framework imports, mirroring
# the LIBERO server.  The shared helpers below (single-rank distributed init,
# local IP discovery, frozen-config EMA disable) live in
# ``action_policy_server_utils`` so this module doesn't have to import from
# the sibling LIBERO server just to share runtime utilities.
from cosmos_framework.inference.common.init import init_script

init_script()

import hashlib
import json
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch import nn
import tyro

from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
    pose_rel_to_abs,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.model.zeva import (
    CausalPromptConfig,
    CausalPromptEncoder,
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    StaticTaskContextRetrievalConfig,
    StaticTaskContextRetrievalHead,
    PersistentInteractionMemory,
    PersistentInteractionMemoryConfig,
    normalize_cte_state_dict,
    retrieve_static_task_context,
)
from cosmos_framework.model.zeva.experimental.robocasa_transition_memory import make_robocasa_atomic5_schema
from cosmos_framework.model.zeva.experimental.transition_memory import (
    TransitionMemoryEncoder,
    TransitionMemorySchema,
    TransitionRecord,
)
from cosmos_framework.model.zeva.experimental.transition_memory_runtime import TransitionMemoryController
from cosmos_framework.model.zeva.attempt_protocol import (
    EXECUTED_ACTION_HORIZON,
    PREDICTED_ACTION_HORIZON,
    AttemptSessionKey,
    contract_manifest,
)
from cosmos_framework.inference.args import OmniSetupArgs, OmniSetupOverrides
from cosmos_framework.inference.common.args import ConfigFileType, ConfigOverrides, tyro_cli
from cosmos_framework.inference.common.config import deserialize_config, deserialize_config_dict, load_config
from cosmos_framework.inference.common.init import init_output_dir
from cosmos_framework.inference.inference import OmniInference
from cosmos_framework.scripts.action_policy_server_utils import (
    DEFAULT_FALLBACK_OUTPUT_DIR,
    disable_runtime_ema_for_frozen_config,
    get_local_ip,
    maybe_init_distributed,
)
from cosmos_framework.scripts.action_policy_server_robocasa365_zeva import extract_batch, format_action_prompt
from cosmos_framework.utils import log
from cosmos_framework.utils.checkpoint_db import CheckpointDirHf

_DEFAULT_DROID_POLICY_CHECKPOINT = "nvidia/Cosmos3-Nano-Policy-DROID"
_DEFAULT_CONDITIONING_FPS = 20.0
_DEFAULT_ACTION_CHUNK_SIZE = 32
_DEFAULT_IMAGE_HEIGHT = 256
_DEFAULT_IMAGE_WIDTH = 512
_DEFAULT_ACTION_DIM = 7
_DEFAULT_ROBOLAB_OUTPUT_DIR = DEFAULT_FALLBACK_OUTPUT_DIR / "robolab"
_CONCAT_VIEW_DESCRIPTION = "The left panel is the left agent view and the right panel is the wrist view."
_DEFAULT_HF_REVISION = "main"
_ROBOLAB_POLICY_HF_REPOSITORIES = {
    "Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
    "nvidia/Cosmos3-Nano-Policy-DROID": "nvidia/Cosmos3-Nano-Policy-DROID",
    "Cosmos3-Edge-Policy-DROID": "nvidia/Cosmos3-Edge-Policy-DROID",
    "nvidia/Cosmos3-Edge-Policy-DROID": "nvidia/Cosmos3-Edge-Policy-DROID",
}

ActionSpace = Literal["joint_pos", "midtrain"]


def _load_checkpoint_metadata(checkpoint_path: str) -> dict[str, Any] | None:
    if "://" in checkpoint_path:
        return None
    checkpoint_dir = Path(checkpoint_path).expanduser().absolute()
    if (checkpoint_dir / "model").is_dir():
        checkpoint_dir = checkpoint_dir / "model"
    metadata_path = checkpoint_dir / "checkpoint.json"
    if not metadata_path.exists():
        return None
    return deserialize_config_dict(metadata_path)


def _load_training_config_from_metadata(metadata: dict[str, Any]) -> Any | None:
    config_file = metadata.get("config_file")
    if not isinstance(config_file, str) or not config_file:
        return None
    config_overrides = ConfigOverrides(
        config_file=config_file,
        experiment=str(metadata.get("experiment", "")),
        experiment_overrides=list(metadata.get("experiment_overrides", [])),
    )
    config_args = config_overrides.build_config()
    if config_args.config_file_type == ConfigFileType.MODULE:
        return load_config(config_args.config_file, config_args.experiment, overrides=config_args.experiment_overrides)
    return deserialize_config(Path(config_args.config_file))


def _load_training_config(setup_args: OmniSetupArgs, checkpoint_path: str) -> Any | None:
    metadata = _load_checkpoint_metadata(checkpoint_path)
    if metadata is not None:
        try:
            config = _load_training_config_from_metadata(metadata)
        except Exception as exc:
            log.warning(f"[robolab-policy-server] could not load checkpoint metadata config for transforms: {exc}")
            config = None
        if config is not None:
            return config

    if setup_args.config_file_type == ConfigFileType.MODULE and not setup_args.experiment:
        return None

    try:
        return setup_args.load_config()
    except Exception as exc:
        log.warning(f"[robolab-policy-server] could not load training config for transforms: {exc}")
        return None


def _resolve_checkpoint_path(checkpoint_path: str, *, hf_revision: str) -> str:
    if Path(checkpoint_path).expanduser().exists():
        return checkpoint_path

    repository = _ROBOLAB_POLICY_HF_REPOSITORIES.get(checkpoint_path)
    if repository is None:
        return checkpoint_path

    log.info(
        f"[robolab-policy-server] downloading consolidated checkpoint from Hugging Face: "
        f"repository={repository!r} revision={hf_revision!r}"
    )
    return CheckpointDirHf(repository=repository, revision=hf_revision).download()


def _validate_checkpoint(checkpoint_path: str, *, allow_dcp_checkpoint: bool) -> None:
    if checkpoint_path in OmniSetupOverrides.CHECKPOINTS:
        return
    if "://" in checkpoint_path:
        if allow_dcp_checkpoint:
            return
        raise ValueError(
            "RoboLab OSS serving expects a consolidated local safetensors checkpoint directory, not a DCP path. "
            "Run cosmos_framework.scripts.export_model first and pass the exported model directory, or pass "
            "--allow-dcp-checkpoint to opt into direct DCP loading."
        )

    checkpoint_dir = Path(checkpoint_path).expanduser().absolute()
    if (checkpoint_dir / "model").is_dir():
        checkpoint_dir = checkpoint_dir / "model"
    if any(checkpoint_dir.glob("*.distcp")):
        if allow_dcp_checkpoint:
            return
        raise ValueError(
            "RoboLab OSS serving expects a consolidated safetensors checkpoint, but found a DCP checkpoint. "
            "Run cosmos_framework.scripts.export_model first and pass the exported model directory, or pass "
            "--allow-dcp-checkpoint to opt into direct DCP loading."
        )
    has_config = (checkpoint_dir / "config.json").exists()
    has_consolidated_safetensors = any(checkpoint_dir.glob("*.safetensors"))
    has_diffusers_safetensors_index = (checkpoint_dir / "model.safetensors.index.json").exists()
    if not checkpoint_dir.is_dir() or not has_config or not (
        has_consolidated_safetensors or has_diffusers_safetensors_index
    ):
        raise ValueError(f"Invalid safetensors checkpoint directory: {checkpoint_dir}")


def _ensure_rgb_uint8_image(value: Any, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{key!r} must have shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _ensure_2d_float_array(value: Any, key: str, width: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{key!r} must have shape [T,D] or [D], got {array.shape}")
    if width is not None and array.shape[-1] != width:
        raise ValueError(f"{key!r} must have width {width}, got {array.shape[-1]}")
    return np.ascontiguousarray(array)


def _ensure_gripper_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[-1] != 1:
        raise ValueError(f"'observation/gripper_position' must have shape [T,1], [T], or scalar, got {array.shape}")
    return np.ascontiguousarray(array)


def _resize_rgb_uint8(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()  # [1,3,H,W]
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)  # [1,3,H2,W2]
    return resized.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # [H2,W2,3]


def _compose_roboarena_views(obs: dict[str, Any]) -> np.ndarray | None:
    required_keys = (
        "observation/wrist_image_left",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
    )
    if not all(key in obs for key in required_keys):
        return None
    wrist = _ensure_rgb_uint8_image(obs["observation/wrist_image_left"], "observation/wrist_image_left")
    left_raw = _ensure_rgb_uint8_image(obs["observation/exterior_image_1_left"], "observation/exterior_image_1_left")
    right_raw = _ensure_rgb_uint8_image(obs["observation/exterior_image_2_left"], "observation/exterior_image_2_left")
    half_h, half_w = wrist.shape[0] // 2, wrist.shape[1] // 2
    left = _resize_rgb_uint8(left_raw, (half_h, half_w))
    right = _resize_rgb_uint8(right_raw, (half_h, half_w))
    return np.concatenate([wrist, np.concatenate([left, right], axis=1)], axis=0)


def _extract_observation_image(obs: dict[str, Any]) -> np.ndarray:
    if "observation/image" in obs:
        return _ensure_rgb_uint8_image(obs["observation/image"], "observation/image")
    image = _compose_roboarena_views(obs)
    if image is not None:
        return image
    raise ValueError("Observation must contain 'observation/image' or RoBoArena wrist/exterior image keys")


def _build_data_batch_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    data_batch: dict[str, Any] = {}
    for key, value in sample.items():
        if key in IterativeJointDataLoader._MULTI_ITEM_KEYS:
            data_batch[key] = [[value]]
        elif isinstance(value, torch.Tensor):
            data_batch[key] = [value.unsqueeze(0)]  # value: [...], batch item: [1,...]
        else:
            data_batch[key] = [value]
    return data_batch


def _load_openpi_websocket_policy_server() -> type[Any]:
    try:
        from openpi_server.websocket_policy_server import WebsocketPolicyServer
    except ModuleNotFoundError:
        try:
            from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RoboLab WebSocket serving uses OpenPI's WebsocketPolicyServer. Install it with "
                "`uv sync --all-extras --group=cu130-train --group=policy-server`, "
                "or install the full Physical-Intelligence/openpi package."
            ) from exc
    return WebsocketPolicyServer


class _DummyDataset(torch.utils.data.IterableDataset):
    def __iter__(self) -> Any:
        return iter(())


@dataclass(frozen=True)
class RobolabPolicyConfig:
    checkpoint_path: str
    domain_name: str
    decode_video: bool
    seed: int
    deterministic_seed: bool
    guidance: float
    num_steps: int
    shift: float
    conditioning_fps: float
    resolution: str | None
    action_chunk_size: int
    action_dim: int
    image_height: int = _DEFAULT_IMAGE_HEIGHT
    image_width: int = _DEFAULT_IMAGE_WIDTH
    action_space: ActionSpace = "joint_pos"
    use_state: bool = True
    history_length: int = 1
    proprio_dim: int = 9


class RobolabServerArgs(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    checkpoint_path: str = _DEFAULT_DROID_POLICY_CHECKPOINT
    """Consolidated local safetensors checkpoint directory, registered checkpoint name, or DCP path with --allow-dcp-checkpoint."""
    hf_revision: str = _DEFAULT_HF_REVISION
    """Hugging Face revision used when --checkpoint-path is a supported public RoboLab policy repository."""
    allow_dcp_checkpoint: bool = False
    """If set, allow direct DCP/S3 checkpoint loading instead of requiring a consolidated safetensors export."""
    experiment: str | None = None
    """Experiment name for DCP checkpoints using module configs, e.g. droid_lerobot_8b_policy."""
    experiment_overrides: list[str] = pydantic.Field(default_factory=list)
    """Hydra experiment overrides forwarded to OmniSetup for DCP checkpoint loading."""
    credential_path: str | None = None
    """Optional checkpoint object-store credential path for DCP/S3 loading."""

    port: int = 8000
    """WebSocket port to bind."""
    host: str = "0.0.0.0"
    """WebSocket host to bind."""
    domain_name: str = "robocasa-panda-omron"
    """Action domain name passed to get_domain_id()."""
    decode_video: bool = False
    """If set, decode and return the predicted rollout video as a uint8 NumPy array."""
    guardrails: bool = False
    """Enable text/video guardrails. Robot policy inference keeps these disabled by default."""

    output_dir: Path | None = None
    """Output directory for OmniInference. Defaults to /tmp/cosmos3_action_server/robolab."""
    sampler: Literal["unipc", "edm"] = "unipc"
    """Diffusion sampler used by OmniInference."""

    seed: int = 0
    """Base generation seed used to initialize the request RNG."""
    deterministic_seed: bool = False
    """Use the same seed for every request. The official server advances the RNG by default."""
    guidance: float = 3.0
    """Guidance scale for denoising."""
    num_steps: int = 4
    """Number of denoising steps."""
    shift: float = 5.0
    """UniPC sampler shift."""

    resolution: str | None = "480"
    """Action transform resolution. The default matches the released DROID RoboLab policy."""
    conditioning_fps: float | None = _DEFAULT_CONDITIONING_FPS
    """Conditioning FPS. The default matches the released DROID RoboLab policy."""
    action_chunk_size: int | None = _DEFAULT_ACTION_CHUNK_SIZE
    """Number of action steps to predict. The default matches the released DROID RoboLab policy."""
    action_dim: int | None = _DEFAULT_ACTION_DIM
    """Raw action dimension. The default matches the released DROID RoboLab policy."""
    image_height: int = _DEFAULT_IMAGE_HEIGHT
    """Input observation image height. The default matches the released DROID RoboLab policy."""
    image_width: int = _DEFAULT_IMAGE_WIDTH
    """Input observation image width. The default matches the released DROID RoboLab policy."""
    action_space: ActionSpace = "joint_pos"
    """RoboLab action representation to serve."""
    use_state: bool = False
    """Include action state in the policy input."""
    history_length: int = 0
    """State/history action rows to trim from the generated action output."""
    format_prompt_as_json: bool | None = None
    """Serve prompts as structured JSON (matching training ``format_prompt_as_json``)."""
    proprio_dim: int = 9
    """Independent fixed-base proprio input: relative EEF pose (7) plus gripper qpos (2)."""
    task_context_bank: Path | None = None
    """Task-context bank used by Zeva."""
    cte_checkpoint: Path | None = None
    """Frozen CTE checkpoint used to derive phase and causal-effect features."""
    task_context_instruction: str | None = None
    """Optional task key used to select a task-context prototype."""
    static_task_context_checkpoint: Path | None = None
    """Static task-context retrieval head, mutually exclusive with the task-ID oracle."""
    static_task_context_top_k: int = 5
    """Number of retrieved entries used to form the task-context token."""
    bit_mode: Literal["normal", "zero", "shuffled"] = "normal"
    """Evaluation ablation for completed causal effects; never use outside controlled diagnostics."""
    disable_policy_injection: bool = False
    """Diagnostic only: zero the policy adapter and task-context projector."""
    transition_memory_encoder_checkpoint: Path | None = None
    """Standalone history-only RoboCasa Atomic-5 transition-memory encoder checkpoint."""
    transition_memory_enabled: bool = False
    """Condition the action branch on the retrieved online context."""
    transition_memory_shadow: bool = False
    """Compute and log transition memory while injecting a zero context (no action change)."""
    transition_memory_mode: Literal["normal", "empty", "shuffled", "wrong_task"] = "normal"
    """Controlled runtime ablation for transition-memory retrieval."""
    transition_memory_support_records: Path | None = None
    """Offline RoboCasa transition-record file used only by the wrong-task ablation."""
    transition_memory_support_task: str | None = None
    """Task cluster whose offline transitions are deliberately supplied as wrong-task support."""
    pim_enabled: bool = False
    """Enable phase-conditioned Persistent Interaction Memory."""
    pim_adapter_delta: Path | None = None
    """Optional trained PIM-only delta loaded on top of the released Stage-2 checkpoint."""
    pim_shadow: bool = False
    """Build/query PIM but send empty PIM tensors to the model."""
    pim_capacity: int = 64
    """Maximum number of merged phase/effect prototypes in one episode."""
    pim_top_k: int = 4
    """Number of phase-matched PIM entries supplied to the model."""
    pim_merge_threshold: float = 0.85
    """Minimum weighted phase/effect cosine score for schema merging."""
    pim_phase_merge_weight: float = 0.5
    """Phase contribution to PIM merge similarity."""
    pim_effect_merge_weight: float = 0.5
    """Effect contribution to PIM merge similarity."""
    pim_executed_action_horizon: int = EXECUTED_ACTION_HORIZON
    """Executed controls required before committing one PIM effect."""
    pim_success_replay_enabled: bool = False
    """Retain a successful phase/action trace for later attempts."""
    pim_success_replay_phase_threshold: float = 0.80
    """Minimum current/locked phase cosine similarity for guarded replay."""
    pim_success_replay_blend: float = 1.0
    """Blend of locked successful actions into the generated action chunk."""


class RobolabPolicyService:
    def __init__(self, args: RobolabServerArgs) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for OmniMoTModel inference in this repo.")
        resolved_checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path, hf_revision=args.hf_revision)
        args = args.model_copy(update={"checkpoint_path": resolved_checkpoint_path})
        _validate_checkpoint(args.checkpoint_path, allow_dcp_checkpoint=args.allow_dcp_checkpoint)
        maybe_init_distributed()

        setup_args = self._build_setup_args(args)
        log.info(
            f"[robolab-policy-server] loading model: checkpoint_path={setup_args.checkpoint_path!r} "
            f"config_file={setup_args.config_file!r} experiment={setup_args.experiment!r}"
        )
        pipe = OmniInference.create(setup_args)
        self.pipe: OmniInference = pipe
        self.model = pipe.model
        self.model.eval()
        self._load_pim_adapter_delta(args.pim_adapter_delta, args.checkpoint_path)
        assert isinstance(pipe.setup_args, OmniSetupArgs)
        self.setup_args: OmniSetupArgs = pipe.setup_args
        self._model_has_online_prefix = getattr(self.model.net, "behavior_online_projector", None) is not None
        self._model_has_pim = getattr(self.model.net, "behavior_pim_encoder", None) is not None
        training_config = _load_training_config(self.setup_args, args.checkpoint_path)
        self._transform, inferred = self._build_transform(training_config, args)
        self.cfg = RobolabPolicyConfig(
            checkpoint_path=self.setup_args.checkpoint_path,
            domain_name=args.domain_name,
            decode_video=bool(args.decode_video),
            seed=int(args.seed),
            deterministic_seed=bool(args.deterministic_seed),
            guidance=float(args.guidance),
            num_steps=int(args.num_steps),
            shift=float(args.shift),
            conditioning_fps=float(
                args.conditioning_fps or inferred.get("conditioning_fps") or _DEFAULT_CONDITIONING_FPS
            ),
            resolution=args.resolution or inferred.get("resolution"),
            action_chunk_size=int(
                args.action_chunk_size or inferred.get("action_chunk_size") or _DEFAULT_ACTION_CHUNK_SIZE
            ),
            action_dim=int(args.action_dim or (8 if args.action_space == "joint_pos" else 10)),
            image_height=int(args.image_height),
            image_width=int(args.image_width),
            action_space=args.action_space,
            use_state=bool(args.use_state),
            history_length=int(args.history_length),
            proprio_dim=int(args.proprio_dim),
        )
        if self.cfg.history_length < (1 if self.cfg.use_state else 0):
            raise ValueError("--history-length must be >= 1 when --use-state is true")
        if self.cfg.image_height <= 0 or self.cfg.image_width <= 0:
            raise ValueError("--image-height and --image-width must be positive")

        self._init_zeva(args)
        self._init_transition_memory(args)
        self._init_pim(args)
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.cfg.seed)
        log.info(
            f"[robolab-policy-server] ready domain={self.cfg.domain_name!r} resolution={self.cfg.resolution!r} "
            f"action_space={self.cfg.action_space} action_dim={self.cfg.action_dim} "
            f"chunk={self.cfg.action_chunk_size} history={self.cfg.history_length} use_state={self.cfg.use_state} "
            f"image={self.cfg.image_height}x{self.cfg.image_width} fps={self.cfg.conditioning_fps} "
            f"guidance={self.cfg.guidance} num_steps={self.cfg.num_steps} shift={self.cfg.shift} "
            f"seed={self.cfg.seed} deterministic_seed={self.cfg.deterministic_seed}"
        )

    def _load_pim_adapter_delta(self, checkpoint: Path | None, base_checkpoint: str) -> None:
        """Overlay the small trained PIM adapter without renaming model keys."""
        if checkpoint is None:
            return
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "cosmos_behavior_pim_adapter_delta_v1":
            raise ValueError(f"Unsupported PIM adapter delta format in {checkpoint}")
        expected_base_hash = payload.get("base_metadata_sha256")
        base_path = Path(base_checkpoint).expanduser()
        if (base_path / "model").is_dir():
            base_path = base_path / "model"
        metadata_path = base_path / ".metadata"
        if metadata_path.is_file() and expected_base_hash:
            actual_base_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            if actual_base_hash != expected_base_hash:
                raise ValueError(
                    "PIM adapter delta was exported from a different Stage-2 checkpoint: "
                    f"expected metadata {expected_base_hash}, got {actual_base_hash}"
                )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError(f"PIM adapter delta has no state_dict: {checkpoint}")
        net = self.model.net
        if getattr(net, "behavior_pim_encoder", None) is None:
            reference = next(net.parameters())
            global_projector = getattr(net, "behavior_global_projector", None)
            if global_projector is None:
                raise ValueError("PIM adapter delta requires the released Zeva Stage-2 model")
            global_dim = int(global_projector.in_features)
            pim_dim = 256
            net.behavior_pim_encoder = CausalPromptEncoder(
                CausalPromptConfig(
                    global_dim=global_dim,
                    phase_dim=128,
                    effect_dim=128,
                    brief_length=4,
                    persistent_length=4,
                    hidden_dim=pim_dim,
                    num_heads=4,
                )
            ).to(device=reference.device, dtype=reference.dtype)
            net.behavior_pim_projector = nn.Linear(pim_dim, net.hidden_size, bias=True).to(
                device=reference.device, dtype=reference.dtype
            )
            net.behavior_pim_gate = nn.Parameter(
                torch.empty(1, device=reference.device, dtype=reference.dtype)
            )
            net.behavior_pim_gate_init = 0.0
            net.behavior_pim_force_bypass = False
            behavior_cfg = self.model.config.behavior_stage2
            behavior_cfg.pim_memory_enabled = True
            behavior_cfg.pim_persistent_length = 4
            behavior_cfg.pim_context_dim = pim_dim
            net.config.behavior_stage2_config.update(
                pim_memory_enabled=True,
                pim_persistent_length=4,
                pim_context_dim=pim_dim,
                pim_force_bypass=False,
            )
        model_keys = set(self.model.state_dict())
        unexpected = sorted(set(state_dict) - model_keys)
        if unexpected:
            raise ValueError(
                "PIM adapter delta does not match the configured model; "
                f"unexpected keys: {unexpected[:5]}"
            )
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                "PIM adapter delta produced unexpected model keys: "
                f"{incompatible.unexpected_keys[:5]}"
            )
        log.info(
            "[robolab-policy-server] loaded PIM adapter delta "
            f"path={checkpoint} tensors={len(state_dict)} format={payload['format']}"
        )

    def _init_transition_memory(self, args: RobolabServerArgs) -> None:
        """Load the experimental task-scoped transition-memory controller."""
        self._transition_enabled = bool(args.transition_memory_enabled or args.transition_memory_shadow)
        self._transition_shadow = bool(args.transition_memory_shadow)
        self._transition_mode = args.transition_memory_mode
        self._transition_controller: Any | None = None
        self._transition_replan_open = False
        self._transition_pending_start: dict[str, torch.Tensor] | None = None
        self._transition_session_id: str | None = None
        self._transition_session_key: AttemptSessionKey | None = None
        self._transition_attempt_id: int | None = None
        self._transition_wrong_task_entries: list[TransitionRecord] = []
        self._transition_support_task: str | None = None
        if not self._transition_enabled:
            return
        if not self._zeva_enabled:
            raise ValueError("Experimental transition memory requires CTE/task-context features")
        if args.transition_memory_encoder_checkpoint is None:
            raise ValueError("--transition-memory-encoder-checkpoint is required for transition memory")
        if not args.transition_memory_encoder_checkpoint.is_file():
            raise FileNotFoundError(args.transition_memory_encoder_checkpoint)
        payload = torch.load(args.transition_memory_encoder_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "robocasa_atomic5_transition_memory_encoder_v1":
            raise ValueError("Unsupported RoboCasa transition-memory encoder format")
        raw_schema = dict(payload.get("schema", {}))
        schema = TransitionMemorySchema(**{k: raw_schema[k] for k in TransitionMemorySchema.__dataclass_fields__ if k in raw_schema})
        expected = make_robocasa_atomic5_schema(
            cte_hash=schema.cte_hash,
            vae_temporal_hash=schema.vae_temporal_hash,
            capacity=schema.capacity,
            top_k=schema.top_k,
        )
        if schema.hash != expected.hash or raw_schema.get("hash") != schema.hash:
            raise ValueError("Transition-memory encoder schema/hash is not RoboCasa Atomic-5 compatible")
        if (
            schema.action_dim != 7
            or schema.action_horizon != EXECUTED_ACTION_HORIZON
            or schema.task_contract != expected.task_contract
        ):
            raise ValueError("Transition-memory encoder has the wrong RoboCasa action/temporal contract")
        encoder = TransitionMemoryEncoder(schema=schema, hidden_dim=256)
        encoder.load_state_dict(payload["model"], strict=True)
        device = next(self.model.parameters()).device
        self._transition_controller = TransitionMemoryController(
            schema=schema, encoder=encoder.to(device), enabled=bool(args.transition_memory_enabled)
        )
        self._transition_controller.encoder.eval()
        if self._transition_mode == "wrong_task":
            if args.transition_memory_support_records is None or args.transition_memory_support_task is None:
                raise ValueError(
                    "wrong_task mode requires --transition-memory-support-records and "
                    "--transition-memory-support-task"
                )
            support_payload = torch.load(args.transition_memory_support_records, map_location="cpu", weights_only=False)
            self._transition_support_task = str(args.transition_memory_support_task)
            support_schema = str(support_payload.get("schema", {}).get("hash", ""))
            if support_schema != schema.hash:
                raise ValueError("wrong-task support records have a schema hash mismatch")
            for raw in support_payload.get("transitions", []):
                if raw.get("task_cluster") != args.transition_memory_support_task:
                    continue
                self._transition_wrong_task_entries.append(
                    TransitionRecord(
                        task_cluster=str(raw["task_cluster"]),
                        phase=raw["phase"].float().cpu(),
                        visual_key=raw["visual_key"].float().cpu(),
                        effect_post=raw["effect_post"].float().cpu(),
                        executed_action=raw["executed_action"].float().cpu(),
                        next_visual_key=raw["next_visual_key"].float().cpu(),
                        next_phase=raw["next_phase"].float().cpu(),
                        latent_index=int(raw.get("latent_index", 0)),
                        schema_hash=schema.hash,
                    )
                )
            if not self._transition_wrong_task_entries:
                raise ValueError(f"No transitions found for wrong-task support {args.transition_memory_support_task!r}")
        log.info(
            "[robolab-policy-server] RoboCasa transition memory ready "
            f"mode={self._transition_mode} conditioning={bool(args.transition_memory_enabled)} "
            f"shadow={self._transition_shadow} schema_hash={schema.hash}"
        )

    def _init_pim(self, args: RobolabServerArgs) -> None:
        """Create the episode-scoped PIM without touching released model weights."""
        self._pim_active = bool(args.pim_enabled or args.pim_shadow)
        self._pim_conditioning = bool(args.pim_enabled)
        self._pim_shadow = bool(args.pim_shadow)
        self._pim_executed_action_horizon = int(args.pim_executed_action_horizon)
        if self._pim_executed_action_horizon <= 0:
            raise ValueError("--pim-executed-action-horizon must be positive")
        self._pim_success_replay_enabled = bool(args.pim_success_replay_enabled)
        self._pim_success_replay_phase_threshold = float(args.pim_success_replay_phase_threshold)
        self._pim_success_replay_blend = float(args.pim_success_replay_blend)
        if not -1.0 <= self._pim_success_replay_phase_threshold <= 1.0:
            raise ValueError("--pim-success-replay-phase-threshold must be in [-1,1]")
        if not 0.0 <= self._pim_success_replay_blend <= 1.0:
            raise ValueError("--pim-success-replay-blend must be in [0,1]")
        self._pim: PersistentInteractionMemory | None = None
        self._pim_replan_open = False
        self._pim_pending_start: dict[str, Any] | None = None
        self._pim_session_id: str | None = None
        self._pim_session_key: AttemptSessionKey | None = None
        self._pim_attempt_id: int | None = None
        self._pim_attempt_chunks: list[dict[str, Any]] = []
        self._pim_success_trace: list[dict[str, Any]] = []
        self._pim_success_attempt_id: int | None = None
        if not self._pim_active:
            return
        if self._transition_enabled:
            raise ValueError("PIM and transition-memory conditioning are mutually exclusive")
        if not self._zeva_enabled:
            raise ValueError("PIM requires CTE/task-context features")
        if self._pim_conditioning and not self._model_has_pim:
            raise ValueError("--pim-enabled requires a checkpoint trained with pim_memory_enabled=true")
        model_pim = getattr(self.model.net, "behavior_pim_encoder", None)
        model_gate = getattr(self.model.net, "behavior_pim_gate", None)
        self._pim_model_hard_bypass = bool(
            getattr(self.model.net, "behavior_pim_force_bypass", False)
        )
        if self._pim_conditioning and self._pim_model_hard_bypass:
            raise ValueError(
                "--pim-enabled is incompatible with "
                "model.config.behavior_stage2.pim_force_bypass=true"
            )
        if self._pim_conditioning and (
            model_gate is None or bool(torch.count_nonzero(model_gate.detach()).item() == 0)
        ):
            raise ValueError(
                "--pim-enabled requires a trained non-zero PIM gate; use the PIM inference "
                "experiment so checkpoint loading does not skip adapter weights"
            )
        if model_pim is not None and int(args.pim_top_k) != int(model_pim.config.persistent_length):
            raise ValueError(
                f"--pim-top-k={args.pim_top_k} must match checkpoint persistent_length="
                f"{model_pim.config.persistent_length}"
            )
        config = PersistentInteractionMemoryConfig(
            phase_dim=128,
            effect_dim=128,
            capacity=int(args.pim_capacity),
            top_k=int(args.pim_top_k),
            merge_threshold=float(args.pim_merge_threshold),
            phase_merge_weight=float(args.pim_phase_merge_weight),
            effect_merge_weight=float(args.pim_effect_merge_weight),
        )
        self._pim = PersistentInteractionMemory(config)
        log.info(
            "[robolab-policy-server] PIM ready "
            f"conditioning={self._pim_conditioning} shadow={self._pim_shadow} "
            f"model_hard_bypass={self._pim_model_hard_bypass} "
            f"capacity={config.capacity} top_k={config.top_k} "
            f"merge_threshold={config.merge_threshold} "
            f"success_replay={self._pim_success_replay_enabled} "
            f"replay_phase_threshold={self._pim_success_replay_phase_threshold} "
            f"replay_blend={self._pim_success_replay_blend}"
        )

    def _build_setup_args(self, args: RobolabServerArgs) -> OmniSetupArgs:
        setup_overrides: dict[str, Any] = {
            "checkpoint_path": args.checkpoint_path,
            "output_dir": args.output_dir or _DEFAULT_ROBOLAB_OUTPUT_DIR,
            "sampler": args.sampler,
            "guardrails": args.guardrails,
        }
        if args.experiment is not None:
            setup_overrides["experiment"] = args.experiment
        if args.experiment_overrides:
            setup_overrides["experiment_overrides"] = list(args.experiment_overrides)
        if args.credential_path is not None:
            setup_overrides["credential_path"] = args.credential_path
        overrides = OmniSetupOverrides.model_validate(setup_overrides)
        setup_args = overrides.build_setup()
        init_output_dir(setup_args.output_dir)
        return disable_runtime_ema_for_frozen_config(setup_args)

    def _build_transform(self, training_config: Any | None, args: RobolabServerArgs) -> tuple[Any, dict[str, Any]]:
        inferred: dict[str, Any] = {}
        model_max_action_dim = getattr(getattr(self.model, "config", None), "max_action_dim", None)
        max_action_dim = int(model_max_action_dim) if isinstance(model_max_action_dim, int) else 64

        try:
            action_dataset_config = training_config.dataloader_train.dataloader.datasets["robocasa365"].dataset
        except (AttributeError, KeyError, TypeError):
            action_dataset_config = None

        if action_dataset_config is None:
            log.warning(
                "[robolab-policy-server] no training action dataset config found; using default "
                f"ActionTransformPipeline (format_prompt_as_json={bool(args.format_prompt_as_json)})"
            )
            return (
                ActionTransformPipeline(
                    max_action_dim=max_action_dim,
                    cfg_dropout_rate=0.0,
                    format_prompt_as_json=bool(args.format_prompt_as_json),
                ),
                inferred,
            )

        chunk_length = getattr(action_dataset_config, "chunk_length", None)
        if isinstance(chunk_length, int):
            inferred["action_chunk_size"] = chunk_length
        fps = getattr(action_dataset_config, "fps", None)
        if isinstance(fps, (int, float)):
            inferred["conditioning_fps"] = float(fps)
        configured_resolution = getattr(action_dataset_config, "resolution", None)
        if configured_resolution is not None:
            inferred["resolution"] = str(configured_resolution)

        format_json = getattr(action_dataset_config, "format_prompt_as_json", True)
        if args.format_prompt_as_json is not None:
            format_json = bool(args.format_prompt_as_json)
        transform = ActionTransformPipeline(
            tokenizer_config=getattr(action_dataset_config, "tokenizer_config", None),
            cfg_dropout_rate=0.0,
            max_action_dim=max_action_dim,
            append_viewpoint_info=bool(getattr(action_dataset_config, "append_viewpoint_info", True)),
            append_duration_fps_timestamps=bool(
                getattr(action_dataset_config, "append_duration_fps_timestamps", True)
            ),
            append_resolution_info=bool(getattr(action_dataset_config, "append_resolution_info", True)),
            append_idle_frames=bool(getattr(action_dataset_config, "append_idle_frames", False)),
            format_prompt_as_json=bool(format_json),
        )
        return transform, inferred

    def _next_seed(self) -> int:
        if self.cfg.deterministic_seed:
            return self.cfg.seed
        return int(self._rng.integers(0, 2**31))

    def _init_zeva(self, args: RobolabServerArgs) -> None:
        """Load the CTE and task-context retrieval components."""
        artifacts = (
            args.task_context_bank,
            args.cte_checkpoint,
            args.task_context_instruction,
            args.static_task_context_checkpoint,
        )
        if not any(item is not None for item in artifacts):
            self._zeva_enabled = False
            return
        if args.task_context_bank is None or args.cte_checkpoint is None:
            raise ValueError(
                "Zeva serving requires --task-context-bank and --cte-checkpoint"
            )
        if (args.task_context_instruction is None) == (args.static_task_context_checkpoint is None):
            raise ValueError(
                "Provide exactly one of --task-context-instruction or --static-task-context-checkpoint"
            )
        if args.static_task_context_top_k < 1:
            raise ValueError("--static-task-context-top-k must be positive")
        self._bit_mode = args.bit_mode
        assert args.task_context_bank is not None
        assert args.cte_checkpoint is not None
        if not args.task_context_bank.is_file() or not args.cte_checkpoint.is_file():
            raise FileNotFoundError("Zeva artifacts must be regular files")
        if getattr(self.model.net, "behavior_pbd", None) is None:
            raise ValueError("Loaded policy has no Zeva policy-injection prior; refusing oracle-effect evaluation")
        if args.disable_policy_injection:
            for name in ("behavior_adapter", "behavior_global_projector"):
                module = getattr(self.model.net, name, None)
                if module is None:
                    raise ValueError(f"Loaded policy has no {name} for zero-residual ablation")
                for parameter in module.parameters():
                    parameter.data.zero_()
            log.info("[robolab-policy-server] zeroed Zeva policy injection for ablation")

        device = next(self.model.parameters()).device
        bank = torch.load(args.task_context_bank, map_location="cpu", weights_only=True)
        entries = bank.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Task-context bank has no entries: {args.task_context_bank}")
        self._static_task_context_enabled = args.static_task_context_checkpoint is not None
        if self._static_task_context_enabled:
            assert args.static_task_context_checkpoint is not None
            if not args.static_task_context_checkpoint.is_file():
                raise FileNotFoundError(f"Static task-context checkpoint not found: {args.static_task_context_checkpoint}")
            retrieval_payload = torch.load(args.static_task_context_checkpoint, map_location="cpu", weights_only=True)
            supported_formats = {
                "zeva_retrieval_head_v1",
                "cosmos_" + "behavior_retrieval_head_v1",
            }
            if retrieval_payload.get("format") not in supported_formats:
                raise ValueError("Unsupported static task-context checkpoint")
            self._static_task_context_head = StaticTaskContextRetrievalHead(
                StaticTaskContextRetrievalConfig(**retrieval_payload["config"])
            ).to(device)
            self._static_task_context_head.load_state_dict(retrieval_payload["model"])
            self._static_task_context_head.eval()
            self._static_task_context_keys = torch.stack([entry["retrieval_key"].float() for entry in entries]).to(device)
            self._static_task_context_values = torch.stack([entry["behavior_value"].float() for entry in entries]).to(device)
            if self._static_task_context_keys.shape[-1] != self._static_task_context_head.config.output_dim:
                raise ValueError("Static task-context output dimension does not match bank retrieval keys")
            self._static_task_context_top_k = min(args.static_task_context_top_k, len(entries))
        else:
            assert args.task_context_instruction is not None
            matching = [
                entry["behavior_value"].float()
                for entry in entries
                if entry.get("instruction") == args.task_context_instruction
            ]
            if not matching:
                available = sorted({str(entry.get("instruction", "")) for entry in entries})
                raise ValueError(f"No memory-bank entries for {args.task_context_instruction!r}; available={available}")
            self._task_context = torch.stack(matching).mean(dim=0, keepdim=True).to(device=device)
        payload = torch.load(args.cte_checkpoint, map_location="cpu", weights_only=False)
        cte_model_config = dict(payload["model_config"])
        cte_state = payload["model"]
        if any(key.endswith("mixer.weight_ih_l0") for key in cte_state):
            cte_model_config["use_mamba"] = False
        elif any(key.endswith("mixer.A_log") for key in cte_state):
            cte_model_config["use_mamba"] = True
        self._cte = CausalTransitionEncoder(CausalTransitionEncoderConfig(**cte_model_config)).to(device)
        self._cte.load_state_dict(normalize_cte_state_dict(cte_state))
        self._cte.eval()
        if self._cte.cfg.action_dim != self.cfg.action_dim:
            raise ValueError("CTE action dimension does not match policy action dimension")
        self._zeva_enabled = True
        if self._static_task_context_enabled:
            log.info(
                "[robolab-policy-server] static task-context retrieval enabled "
                f"bank_entries={len(entries)} top_k={self._static_task_context_top_k}"
            )
        else:
            log.info(
                "[robolab-policy-server] task-context prototype enabled "
                f"task={args.task_context_instruction!r} bank_entries={len(matching)}"
            )

    def _encode_cte_frame(self, image: np.ndarray) -> torch.Tensor:
        """Map one observed concatenated RGB image to the CTE's Wan latent."""
        device = next(self.model.parameters()).device
        frame = torch.from_numpy(np.ascontiguousarray(image)).to(device=device, dtype=torch.float32)
        frame = frame.permute(2, 0, 1).unsqueeze(0)
        frame = F.interpolate(frame, size=(480, 832), mode="bilinear", align_corners=False)
        latent = self.model.encode(frame.unsqueeze(2).div_(127.5).sub_(1.0))
        if latent.ndim != 5 or latent.shape[:3] != (1, self._cte.cfg.image_channels, 1):
            raise RuntimeError(f"Unexpected Wan latent for CTE: {tuple(latent.shape)}")
        return latent[0, :, 0].float().contiguous()

    def _task_context_from_initial_observation(self, image: np.ndarray, prompt: str) -> torch.Tensor:
        """Retrieve static task context from the initial observation and instruction."""
        if not self._static_task_context_enabled:
            return self._task_context
        latent = self._encode_cte_frame(image).unsqueeze(0)
        readout = extract_batch(self.model, latent, [format_action_prompt(prompt)]).to(
            device=latent.device, dtype=torch.float32
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            query = self._static_task_context_head(readout)
        values, _, _ = retrieve_static_task_context(
            query,
            self._static_task_context_keys,
            self._static_task_context_values,
            top_k=self._static_task_context_top_k,
        )
        return values.float()

    def _causal_interaction_features(
        self, obs: dict[str, Any], image: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Return task-cluster global plus causal phase/four-effect history.

        ``cte_boundary_images`` contains observations at raw offsets
        0,4,8,...; ``cte_transition_actions`` contains the corresponding
        executed [4,arm7] transitions.  Thus an effect token becomes readable
        only after four completed transitions (16 raw controls).
        """
        if not self._zeva_enabled:
            raise RuntimeError("Zeva causal-interaction features are not configured")
        frames_rgb = np.asarray(obs.get("cte_boundary_images", np.expand_dims(image, axis=0)))
        transitions = np.asarray(
            obs.get("cte_transition_actions", np.empty((0, 4, self.cfg.action_dim), dtype=np.float32)),
            dtype=np.float32,
        )
        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError(f"cte_boundary_images must be [T,H,W,3], got {frames_rgb.shape}")
        if transitions.shape != (frames_rgb.shape[0] - 1, 4, self.cfg.action_dim):
            raise ValueError(
                "cte_transition_actions must be [T-1,4,arm7], got "
                f"{transitions.shape} for T={frames_rgb.shape[0]}"
            )
        latents = torch.stack([self._encode_cte_frame(_ensure_rgb_uint8_image(frame, "cte_boundary_images")) for frame in frames_rgb])
        frames = latents.unsqueeze(0)
        actions = torch.from_numpy(np.ascontiguousarray(transitions)).to(device=frames.device, dtype=torch.float32).unsqueeze(0)
        valid = torch.ones((1, frames.shape[1]), dtype=torch.bool, device=frames.device)
        transition_valid = torch.ones(actions.shape[:-1], dtype=torch.bool, device=frames.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = self._cte(frames, actions, valid, transition_valid)
        completed = encoded["effect_post"][0][encoded["effect_complete"][0]].float()
        history = torch.zeros((1, 4, self._cte.cfg.effect_dim), dtype=torch.float32, device=frames.device)
        history_valid = torch.zeros((1, 4), dtype=torch.bool, device=frames.device)
        take = min(4, completed.shape[0])
        if take:
            history[0, -take:] = completed[-take:]
            history_valid[0, -take:] = True
        if self._bit_mode == "zero":
            # Preserve temporal availability while removing effect content.
            history.zero_()
        elif self._bit_mode == "shuffled" and take > 1:
            # Reverse only completed slots and keep their right-aligned causal positions.
            history[0, -take:] = history[0, -take:].flip(dims=(0,))
        global_feature = self._task_context_from_initial_observation(frames_rgb[0], str(obs["prompt"]))
        current_phase = encoded["phase"][:, -1].float()
        current_visual_key = encoded["visual_key"][:, -1].float()
        complete_indices = torch.nonzero(encoded["effect_complete"][0], as_tuple=False).flatten()
        if complete_indices.numel() > 0:
            current_effect_post = encoded["effect_post"][:, complete_indices[-1]].float()
            current_effect_valid = True
        else:
            current_effect_post = current_phase.new_zeros((1, self._cte.cfg.effect_dim))
            current_effect_valid = False
        return (
            global_feature,
            current_phase,
            history,
            history_valid,
            current_visual_key,
            current_effect_post,
            current_effect_valid,
        )

    def _transition_context(
        self,
        obs: dict[str, Any],
        phase: torch.Tensor,
        visual_key: torch.Tensor,
        effect_post: torch.Tensor,
        effect_valid: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]] | None:
        """Read/commit one causal transition-memory step.

        ``pim_executed_action`` is accepted only on the *next* request and
        must contain exactly the 16 controls executed after the previous
        request.  Therefore an incomplete/safety-aborted rollout cannot enter
        the memory.  The current request itself is never used as support.
        """
        controller = self._transition_controller
        if controller is None:
            return None
        task_cluster = str(obs.get("pim_task_cluster") or obs.get("prompt") or "")
        if not task_cluster:
            raise ValueError("pim_task_cluster/prompt is required for transition memory")
        session_id = str(obs.get("pim_session_id") or task_cluster)
        environment_seed = int(obs.get("pim_environment_seed", -1))
        attempt_id = int(obs.get("pim_attempt_id", -1))
        if environment_seed < 0 or attempt_id < 0:
            raise ValueError("pim_environment_seed and pim_attempt_id are required and must be non-negative")
        requested_session = AttemptSessionKey(session_id, task_cluster, environment_seed)
        reset_memory = bool(obs.get("pim_memory_reset", False))
        reset_replan = bool(obs.get("pim_replan_reset", False))
        if reset_memory or self._transition_session_id != session_id:
            if attempt_id != 0:
                raise ValueError("a new attempt-session must start at pim_attempt_id=0")
            controller.reset(task_cluster)
            self._transition_replan_open = False
            self._transition_pending_start = None
            self._transition_session_id = session_id
            self._transition_session_key = requested_session
            self._transition_attempt_id = attempt_id
        elif controller.memory.task_cluster != task_cluster:
            # A task change without an explicit reset is unsafe; fail-fast
            # rather than mixing support across Atomic-5 clusters.
            raise ValueError(
                f"transition memory task changed from {controller.memory.task_cluster!r} to {task_cluster!r}; "
                "send pim_memory_reset=true"
            )
        else:
            assert self._transition_session_key is not None
            self._transition_session_key.assert_compatible(requested_session)
            assert self._transition_attempt_id is not None
            if attempt_id < self._transition_attempt_id or attempt_id > self._transition_attempt_id + 1:
                raise ValueError(
                    f"pim_attempt_id must stay constant within an attempt or increment by one; "
                    f"previous={self._transition_attempt_id}, requested={attempt_id}"
                )
            if attempt_id == self._transition_attempt_id + 1 and not reset_replan:
                raise ValueError("a new attempt must send pim_replan_reset=true")
            self._transition_attempt_id = attempt_id

        if self._transition_replan_open:
            if reset_replan:
                completed = False
                executed = torch.zeros((controller.schema.action_horizon, controller.schema.action_dim))
            else:
                executed_np = np.asarray(obs.get("pim_executed_action", np.empty((0, 7))), dtype=np.float32)
                completed = bool(obs.get("pim_transition_complete", False))
                if completed:
                    if executed_np.shape != (controller.schema.action_horizon, controller.schema.action_dim):
                        raise ValueError(
                            "pim_transition_complete requires exactly [16,7] executed actions, "
                            f"got {executed_np.shape}"
                        )
                    if not np.isfinite(executed_np).all() or self._transition_pending_start is None or not effect_valid:
                        completed = False
                    executed = torch.from_numpy(np.ascontiguousarray(executed_np))
                else:
                    executed = torch.zeros((controller.schema.action_horizon, controller.schema.action_dim))
            pending = self._transition_pending_start
            if pending is None:
                completed = False
                pending = {"phase": phase.detach(), "visual_key": visual_key.detach()}
            controller.complete_replan(
                phase=pending["phase"],
                visual_key=pending["visual_key"],
                effect_post=effect_post[0].detach(),
                executed_action=executed,
                next_visual_key=visual_key[0].detach(),
                next_phase=phase[0].detach(),
                latent_index=int(obs.get("pim_latent_index", 0)),
                completed=completed,
                metadata=(pending.get("metadata", {}) if isinstance(pending, dict) else {}),
            )
            self._transition_replan_open = False
            self._transition_pending_start = None

        controller.begin_replan()
        phase_cpu = phase[0].detach().float().cpu()
        visual_cpu = visual_key[0].detach().float().cpu()
        if self._transition_mode == "wrong_task":
            query = F.normalize(torch.cat((phase_cpu, visual_cpu)), dim=0)
            keys = torch.stack(
                [F.normalize(torch.cat((entry.phase, entry.visual_key)), dim=0) for entry in self._transition_wrong_task_entries]
            )
            all_scores = keys @ query
            k = min(controller.schema.top_k, len(self._transition_wrong_task_entries))
            scores, indices = torch.topk(all_scores, k=k, largest=True, sorted=True)
            entries = [self._transition_wrong_task_entries[int(index)] for index in indices]
        else:
            entries, scores = controller.memory.query(phase_cpu, visual_cpu)
        if self._transition_mode == "empty":
            entries = []
            scores = torch.empty(0, dtype=torch.float32)
        elif self._transition_mode == "shuffled":
            entries = list(reversed(entries))
        context = controller.encoder(phase, visual_key, entries)
        self._transition_replan_open = True
        self._transition_pending_start = {
            "phase": phase[0].detach().float().cpu(),
            "visual_key": visual_key[0].detach().float().cpu(),
            "metadata": {
                "session_id": session_id,
                "task_cluster": task_cluster,
                "environment_seed": environment_seed,
                "attempt_id": attempt_id,
                "replan_index": int(obs.get("pim_replan_index", 0)),
            },
        }
        info = {
            "session_id": session_id,
            "task_cluster": task_cluster,
            "environment_seed": environment_seed,
            "attempt_id": attempt_id,
            "num_entries": len(entries),
            "topk_latent_indices": [int(entry.latent_index) for entry in entries],
            "scores": [float(score) for score in scores.tolist()],
            "mode": self._transition_mode,
            "support_task": self._transition_support_task,
            "conditioning": bool(controller.enabled and not self._transition_shadow),
        }
        log.info(f"[robolab-policy-server] transition_memory {info}")
        return context, info

    def _empty_pim_tensors(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = 4
        if self._pim is not None:
            length = self._pim.config.top_k
        elif self._model_has_pim:
            length = int(self.model.net.behavior_pim_encoder.config.persistent_length)
        return (
            reference.new_zeros((1, length, 128), dtype=torch.float32),
            reference.new_zeros((1, length, 128), dtype=torch.float32),
            torch.zeros((1, length), dtype=torch.bool, device=reference.device),
        )

    def _pim_context(
        self,
        obs: dict[str, Any],
        phase: torch.Tensor,
        effect_post: torch.Tensor,
        effect_valid: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], np.ndarray | None] | None:
        """Commit the previous completed interaction, then retrieve by current phase."""
        pim = self._pim
        if pim is None:
            return None
        task_cluster = str(obs.get("pim_task_cluster") or obs.get("prompt") or "")
        session_id = str(obs.get("pim_session_id") or task_cluster)
        environment_seed = int(obs.get("pim_environment_seed", -1))
        attempt_id = int(obs.get("pim_attempt_id", -1))
        if not task_cluster or environment_seed < 0 or attempt_id < 0:
            raise ValueError("PIM requires task_cluster, environment_seed, and attempt_id")
        requested_session = AttemptSessionKey(session_id, task_cluster, environment_seed)
        reset_memory = bool(obs.get("pim_memory_reset", False))
        reset_replan = bool(obs.get("pim_replan_reset", False))
        if reset_memory or self._pim_session_id != session_id:
            pim.reset_episode(task_cluster, attempt_id=attempt_id)
            self._pim_replan_open = False
            self._pim_pending_start = None
            self._pim_session_id = session_id
            self._pim_session_key = requested_session
            self._pim_attempt_id = attempt_id
            self._pim_attempt_chunks = []
            self._pim_success_trace = []
            self._pim_success_attempt_id = None
        else:
            assert self._pim_session_key is not None and self._pim_attempt_id is not None
            self._pim_session_key.assert_compatible(requested_session)
            if attempt_id < self._pim_attempt_id or attempt_id > self._pim_attempt_id + 1:
                raise ValueError(
                    f"PIM attempt_id must stay constant or increment by one: "
                    f"active={self._pim_attempt_id}, requested={attempt_id}"
                )
            new_attempt = attempt_id == self._pim_attempt_id + 1
            if new_attempt and not reset_replan:
                raise ValueError("A new PIM attempt must send pim_replan_reset=true")
            pim.begin_attempt(attempt_id)
            self._pim_attempt_id = attempt_id
            if new_attempt:
                self._pim_attempt_chunks = []

        committed = False
        merged = False
        if self._pim_replan_open:
            if not reset_replan:
                completed = bool(obs.get("pim_transition_complete", False))
                executed = np.asarray(obs.get("pim_executed_action", np.empty((0, 7))), dtype=np.float32)
                expected_shape = (self._pim_executed_action_horizon, 7)
                if completed and executed.shape != expected_shape:
                    raise ValueError(
                        "PIM completed transition requires "
                        f"[{self._pim_executed_action_horizon},7] actions, "
                        f"got {executed.shape}"
                    )
                pending = self._pim_pending_start
                if completed and pending is not None and np.isfinite(executed).all():
                    self._pim_attempt_chunks.append(
                        {
                            "phase": pending["phase"].detach().float().cpu().contiguous(),
                            "action": torch.from_numpy(executed.copy()).float().contiguous(),
                            "replan_index": int(pending.get("metadata", {}).get("replan_index", 0)),
                            "attempt_id": attempt_id,
                        }
                    )
                if completed and pending is not None and effect_valid and np.isfinite(executed).all():
                    _, merged = pim.append_completed(
                        task_cluster=task_cluster,
                        phase=pending["phase"],
                        effect=effect_post[0],
                        attempt_id=attempt_id,
                        metadata=dict(pending.get("metadata", {})),
                    )
                    committed = True
            self._pim_replan_open = False
            self._pim_pending_start = None

        current_phase = phase[0].detach().float().cpu()
        pim_phase, pim_effect, pim_valid, scores = pim.query_tensors(current_phase)
        replan_index = int(obs.get("pim_replan_index", 0))
        replay_action: np.ndarray | None = None
        replay_phase_similarity: float | None = None
        if (
            self._pim_success_replay_enabled
            and self._pim_conditioning
            and not self._pim_shadow
            and self._pim_success_attempt_id is not None
            and attempt_id > self._pim_success_attempt_id
            and 0 <= replan_index < len(self._pim_success_trace)
        ):
            locked = self._pim_success_trace[replan_index]
            replay_phase_similarity = float(
                F.cosine_similarity(
                    current_phase.unsqueeze(0),
                    locked["phase"].unsqueeze(0),
                    dim=-1,
                )[0]
            )
            if replay_phase_similarity >= self._pim_success_replay_phase_threshold:
                replay_action = locked["action"].detach().float().cpu().numpy().copy()
        self._pim_replan_open = True
        self._pim_pending_start = {
            "phase": phase[0].detach().float().cpu(),
            "metadata": {
                "session_id": session_id,
                "task_cluster": task_cluster,
                "environment_seed": environment_seed,
                "attempt_id": attempt_id,
                "replan_index": replan_index,
            },
        }
        info = {
            "session_id": session_id,
            "task_cluster": task_cluster,
            "environment_seed": environment_seed,
            "attempt_id": attempt_id,
            "entries_total": len(pim),
            "num_entries": len(pim),
            "retrieved": int(pim_valid.sum()),
            "scores": [float(value) for value in scores[pim_valid].tolist()],
            "committed": committed,
            "merged": merged,
            "conditioning": bool(self._pim_conditioning and not self._pim_shadow),
            "success_replay_enabled": self._pim_success_replay_enabled,
            "success_trace_attempt_id": self._pim_success_attempt_id,
            "success_trace_chunks": len(self._pim_success_trace),
            "replay_candidate": replay_action is not None,
            "replay_phase_similarity": replay_phase_similarity,
        }
        log.info(f"[robolab-policy-server] pim {info}")
        return (
            pim_phase.unsqueeze(0).to(device=phase.device),
            pim_effect.unsqueeze(0).to(device=phase.device),
            pim_valid.unsqueeze(0).to(device=phase.device),
            info,
            replay_action,
        )

    def _finalize_pim_attempt(self, obs: dict[str, Any]) -> dict[str, Any]:
        pim = self._pim
        if pim is None:
            raise ValueError("PIM finalization requested while PIM is disabled")
        session_id = str(obs.get("pim_session_id") or "")
        task_cluster = str(obs.get("pim_task_cluster") or "")
        environment_seed = int(obs.get("pim_environment_seed", -1))
        attempt_id = int(obs.get("pim_attempt_id", -1))
        requested = AttemptSessionKey(session_id, task_cluster, environment_seed)
        if self._pim_session_key is None or self._pim_attempt_id is None:
            raise ValueError("cannot finalize PIM before a session has started")
        self._pim_session_key.assert_compatible(requested)
        if attempt_id != self._pim_attempt_id:
            raise ValueError("PIM finalize attempt_id does not match the active attempt")
        committed = False
        merged = False
        executed = np.asarray(obs.get("pim_executed_action", np.empty((0, 7))), dtype=np.float32)
        pending = self._pim_pending_start
        if (
            self._pim_replan_open
            and pending is not None
            and executed.ndim == 2
            and executed.shape[1:] == (7,)
            and 0 < executed.shape[0] <= self._pim_executed_action_horizon
            and np.isfinite(executed).all()
        ):
            self._pim_attempt_chunks.append(
                {
                    "phase": pending["phase"].detach().float().cpu().contiguous(),
                    "action": torch.from_numpy(executed.copy()).float().contiguous(),
                    "replan_index": int(pending.get("metadata", {}).get("replan_index", 0)),
                    "attempt_id": attempt_id,
                }
            )
        if self._pim_replan_open and bool(obs.get("pim_transition_complete", False)):
            expected_shape = (self._pim_executed_action_horizon, 7)
            if executed.shape != expected_shape:
                raise ValueError(
                    "terminal PIM transition requires exactly "
                    f"[{self._pim_executed_action_horizon},7] executed actions"
                )
            if pending is not None and np.isfinite(executed).all():
                image = _extract_observation_image(obs)
                features = self._causal_interaction_features(obs, image)
                current_effect_post, current_effect_valid = features[-2], features[-1]
                if current_effect_valid:
                    _, merged = pim.append_completed(
                        task_cluster=task_cluster,
                        phase=pending["phase"],
                        effect=current_effect_post[0],
                        attempt_id=attempt_id,
                        metadata=dict(pending.get("metadata", {})),
                    )
                    committed = True
        terminal_outcome = str(obs.get("pim_terminal_outcome") or "")
        locked_success_trace = False
        if (
            self._pim_success_replay_enabled
            and terminal_outcome == "success"
            and self._pim_success_attempt_id is None
            and self._pim_attempt_chunks
        ):
            ordered = sorted(self._pim_attempt_chunks, key=lambda item: int(item["replan_index"]))
            expected_indices = list(range(len(ordered)))
            actual_indices = [int(item["replan_index"]) for item in ordered]
            if actual_indices != expected_indices:
                raise ValueError(
                    "successful PIM replay trace must contain contiguous replan indices: "
                    f"expected={expected_indices}, actual={actual_indices}"
                )
            self._pim_success_trace = [
                {
                    "phase": item["phase"].detach().float().cpu().contiguous(),
                    "action": item["action"].detach().float().cpu().contiguous(),
                    "replan_index": int(item["replan_index"]),
                    "attempt_id": attempt_id,
                }
                for item in ordered
            ]
            self._pim_success_attempt_id = attempt_id
            locked_success_trace = True
        self._pim_replan_open = False
        self._pim_pending_start = None
        info = {
            "session_id": session_id,
            "task_cluster": task_cluster,
            "attempt_id": attempt_id,
            "committed_terminal_transition": committed,
            "merged": merged,
            "pim_entries_total": len(pim),
            "conditioning": bool(self._pim_conditioning and not self._pim_shadow),
            "terminal_outcome": terminal_outcome,
            "success_trace_locked": locked_success_trace,
            "success_trace_attempt_id": self._pim_success_attempt_id,
            "success_trace_chunks": len(self._pim_success_trace),
        }
        log.info(f"[robolab-policy-server] pim_finalize {info}")
        return info

    def _restore_pim_session(self, obs: dict[str, Any]) -> dict[str, Any]:
        pim = self._pim
        if pim is None:
            raise ValueError("PIM restore requested while PIM is disabled")
        session_id = str(obs.get("pim_session_id") or "")
        task_cluster = str(obs.get("pim_task_cluster") or "")
        environment_seed = int(obs.get("pim_environment_seed", -1))
        last_attempt_id = int(obs.get("pim_last_attempt_id", -1))
        phase = np.asarray(obs.get("restore_phase"), dtype=np.float32)
        effect = np.asarray(obs.get("restore_effect_post"), dtype=np.float32)
        n = phase.shape[0] if phase.ndim == 2 else -1
        effect_valid = np.asarray(
            obs.get("restore_effect_valid", np.ones(max(n, 0), dtype=np.bool_)), dtype=np.bool_
        )
        attempt_ids = np.asarray(obs.get("restore_attempt_ids"), dtype=np.int64)
        success_phase = np.asarray(
            obs.get("restore_success_trace_phase", np.empty((0, 128))), dtype=np.float32
        )
        success_action = np.asarray(
            obs.get(
                "restore_success_trace_action",
                np.empty((0, self._pim_executed_action_horizon, 7)),
            ),
            dtype=np.float32,
        )
        success_action_count = np.asarray(
            obs.get("restore_success_trace_action_count", np.empty((0,))), dtype=np.int64
        )
        success_attempt_id = int(obs.get("restore_success_trace_attempt_id", -1))
        success_chunks = success_phase.shape[0] if success_phase.ndim == 2 else -1
        if (
            not session_id
            or not task_cluster
            or environment_seed < 0
            or last_attempt_id < 0
            or phase.shape != (n, 128)
            or effect.shape != (n, 128)
            or effect_valid.shape != (n,)
            or attempt_ids.shape != (n,)
            or success_phase.shape != (success_chunks, 128)
            or success_action.shape != (success_chunks, self._pim_executed_action_horizon, 7)
            or success_action_count.shape != (success_chunks,)
        ):
            raise ValueError("invalid PIM restore payload")
        if n and (
            attempt_ids.min() < 0
            or attempt_ids.max() > last_attempt_id
            or np.any(np.diff(attempt_ids) < 0)
        ):
            raise ValueError("PIM restore attempt IDs must be ordered within the declared history")
        if success_chunks:
            if (
                success_attempt_id < 0
                or success_attempt_id > last_attempt_id
                or np.any(success_action_count <= 0)
                or np.any(success_action_count > self._pim_executed_action_horizon)
                or not np.isfinite(success_phase).all()
                or not np.isfinite(success_action).all()
            ):
                raise ValueError("invalid successful PIM replay trace in restore payload")
        elif success_attempt_id != -1:
            raise ValueError("successful PIM replay attempt id requires a non-empty trace")
        pim.reset_episode(task_cluster)
        for index in range(n):
            target_attempt = int(attempt_ids[index])
            while pim.attempt_id is not None and pim.attempt_id < target_attempt:
                pim.begin_attempt(pim.attempt_id + 1)
            if bool(effect_valid[index]):
                pim.append_completed(
                    task_cluster=task_cluster,
                    phase=torch.from_numpy(np.ascontiguousarray(phase[index])),
                    effect=torch.from_numpy(np.ascontiguousarray(effect[index])),
                    attempt_id=target_attempt,
                    metadata={"restored": True},
                )
        while pim.attempt_id is not None and pim.attempt_id < last_attempt_id:
            pim.begin_attempt(pim.attempt_id + 1)
        self._pim_session_id = session_id
        self._pim_session_key = AttemptSessionKey(session_id, task_cluster, environment_seed)
        self._pim_attempt_id = last_attempt_id
        self._pim_replan_open = False
        self._pim_pending_start = None
        self._pim_attempt_chunks = []
        self._pim_success_trace = [
            {
                "phase": torch.from_numpy(np.ascontiguousarray(success_phase[index])).float(),
                "action": torch.from_numpy(
                    np.ascontiguousarray(success_action[index, : success_action_count[index]])
                ).float(),
                "replan_index": index,
                "attempt_id": success_attempt_id,
            }
            for index in range(success_chunks)
        ]
        self._pim_success_attempt_id = success_attempt_id if success_chunks else None
        info = {
            "session_id": session_id,
            "task_cluster": task_cluster,
            "last_attempt_id": last_attempt_id,
            "restored_transitions": n,
            "restored_valid_effects": int(effect_valid.sum()),
            "pim_entries_total": len(pim),
            "conditioning": bool(self._pim_conditioning and not self._pim_shadow),
            "success_trace_attempt_id": self._pim_success_attempt_id,
            "success_trace_chunks": len(self._pim_success_trace),
        }
        log.info(f"[robolab-policy-server] pim_restore {info}")
        return info

    def _finalize_transition_attempt(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Commit a final full window, discard a partial one, then label the attempt."""
        controller = self._transition_controller
        if controller is None:
            raise ValueError("pim_finalize_attempt requires transition memory or shadow mode")
        session_id = str(obs.get("pim_session_id") or "")
        task_cluster = str(obs.get("pim_task_cluster") or "")
        environment_seed = int(obs.get("pim_environment_seed", -1))
        attempt_id = int(obs.get("pim_attempt_id", -1))
        if not session_id or not task_cluster or environment_seed < 0 or attempt_id < 0:
            raise ValueError("finalize requires session_id, task_cluster, environment_seed, and attempt_id")
        requested_session = AttemptSessionKey(session_id, task_cluster, environment_seed)
        if self._transition_session_key is None or self._transition_attempt_id is None:
            raise ValueError("cannot finalize before a transition-memory session has started")
        self._transition_session_key.assert_compatible(requested_session)
        if attempt_id != self._transition_attempt_id:
            raise ValueError(
                f"finalize attempt_id mismatch: active={self._transition_attempt_id}, requested={attempt_id}"
            )

        committed_terminal_transition = False
        if self._transition_replan_open:
            completed = bool(obs.get("pim_transition_complete", False))
            executed_np = np.asarray(obs.get("pim_executed_action", np.empty((0, 7))), dtype=np.float32)
            pending = self._transition_pending_start
            if completed:
                if executed_np.shape != (controller.schema.action_horizon, controller.schema.action_dim):
                    raise ValueError(
                        "terminal pim_transition_complete requires exactly [16,7] actions, "
                        f"got {executed_np.shape}"
                    )
                if not np.isfinite(executed_np).all() or pending is None:
                    completed = False
            if completed:
                (
                    _,
                    phase,
                    _,
                    _,
                    visual_key,
                    current_effect_post,
                    current_effect_valid,
                ) = self._causal_interaction_features(obs, _extract_observation_image(obs))
                completed = bool(current_effect_valid)
            if completed:
                assert pending is not None
                committed_terminal_transition = controller.complete_replan(
                    phase=pending["phase"],
                    visual_key=pending["visual_key"],
                    effect_post=current_effect_post[0].detach(),
                    executed_action=torch.from_numpy(np.ascontiguousarray(executed_np)),
                    next_visual_key=visual_key[0].detach(),
                    next_phase=phase[0].detach(),
                    latent_index=int(obs.get("pim_latent_index", 0)),
                    completed=True,
                    metadata=pending.get("metadata", {}),
                )
            else:
                controller.discard_open_replan()
            self._transition_replan_open = False
            self._transition_pending_start = None

        outcome = str(obs.get("pim_terminal_outcome") or "")
        termination_reason = str(obs.get("pim_termination_reason") or "")
        total_steps = int(obs.get("pim_total_steps", -1))
        final_progress = float(obs.get("pim_final_progress", -1.0))
        annotated = controller.memory.annotate_attempt_outcome(
            attempt_id,
            outcome=outcome,
            termination_reason=termination_reason,
            total_steps=total_steps,
            final_progress=final_progress,
        )
        info = {
            "session_id": session_id,
            "task_cluster": task_cluster,
            "environment_seed": environment_seed,
            "attempt_id": attempt_id,
            "terminal_outcome": outcome,
            "termination_reason": termination_reason,
            "total_steps": total_steps,
            "final_progress": final_progress,
            "progress_source": "success_only",
            "annotated_transitions": annotated,
            "committed_terminal_transition": committed_terminal_transition,
            "memory_entries_total": len(controller.memory),
            "conditioning": bool(controller.enabled and not self._transition_shadow),
        }
        log.info(f"[robolab-policy-server] pim_finalize {info}")
        return info

    def _restore_transition_session(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Restore causally completed transitions from attempt artifacts."""
        controller = self._transition_controller
        if controller is None:
            raise ValueError("pim_restore_session requires transition memory or shadow mode")
        session_id = str(obs.get("pim_session_id") or "")
        task_cluster = str(obs.get("pim_task_cluster") or "")
        environment_seed = int(obs.get("pim_environment_seed", -1))
        last_attempt_id = int(obs.get("pim_last_attempt_id", -1))
        if not session_id or not task_cluster or environment_seed < 0 or last_attempt_id < 0:
            raise ValueError("restore requires session/task/seed and a non-negative last_attempt_id")

        phase = np.asarray(obs.get("restore_phase"), dtype=np.float32)
        visual_key = np.asarray(obs.get("restore_visual_key"), dtype=np.float32)
        effect_post = np.asarray(obs.get("restore_effect_post"), dtype=np.float32)
        executed_action = np.asarray(obs.get("restore_executed_action"), dtype=np.float32)
        next_visual_key = np.asarray(obs.get("restore_next_visual_key"), dtype=np.float32)
        next_phase = np.asarray(obs.get("restore_next_phase"), dtype=np.float32)
        latent_index = np.asarray(obs.get("restore_latent_index"), dtype=np.int64)
        attempt_ids = np.asarray(obs.get("restore_attempt_ids"), dtype=np.int64)
        replan_indices = np.asarray(obs.get("restore_replan_indices"), dtype=np.int64)
        outcomes = list(obs.get("restore_outcomes", []))
        reasons = list(obs.get("restore_termination_reasons", []))
        total_steps = np.asarray(obs.get("restore_total_steps"), dtype=np.int64)
        final_progress = np.asarray(obs.get("restore_final_progress"), dtype=np.float32)
        n = phase.shape[0] if phase.ndim == 2 else -1
        expected_shapes = {
            "phase": (n, controller.schema.phase_dim),
            "visual_key": (n, controller.schema.visual_key_dim),
            "effect_post": (n, controller.schema.effect_dim),
            "executed_action": (n, controller.schema.action_horizon, controller.schema.action_dim),
            "next_visual_key": (n, controller.schema.visual_key_dim),
            "next_phase": (n, controller.schema.phase_dim),
            "latent_index": (n,),
            "attempt_ids": (n,),
            "replan_indices": (n,),
            "total_steps": (n,),
            "final_progress": (n,),
        }
        actual = {
            "phase": phase.shape,
            "visual_key": visual_key.shape,
            "effect_post": effect_post.shape,
            "executed_action": executed_action.shape,
            "next_visual_key": next_visual_key.shape,
            "next_phase": next_phase.shape,
            "latent_index": latent_index.shape,
            "attempt_ids": attempt_ids.shape,
            "replan_indices": replan_indices.shape,
            "total_steps": total_steps.shape,
            "final_progress": final_progress.shape,
        }
        bad = {key: (actual[key], shape) for key, shape in expected_shapes.items() if actual[key] != shape}
        if n < 0 or bad or len(outcomes) != n or len(reasons) != n:
            raise ValueError(f"invalid online restore payload: n={n}, bad_shapes={bad}")
        if n > controller.schema.capacity:
            raise ValueError(f"restore payload exceeds memory capacity {controller.schema.capacity}")
        if n and (attempt_ids.min() < 0 or attempt_ids.max() > last_attempt_id):
            raise ValueError("restore transition attempt IDs exceed the declared session history")

        controller.reset(task_cluster)
        for index in range(n):
            outcome = str(outcomes[index])
            reason = str(reasons[index])
            if outcome not in {"success", "failure"}:
                raise ValueError(f"invalid restored outcome {outcome!r}")
            controller.memory.append(
                TransitionRecord(
                    task_cluster=task_cluster,
                    phase=torch.from_numpy(np.ascontiguousarray(phase[index])),
                    visual_key=torch.from_numpy(np.ascontiguousarray(visual_key[index])),
                    effect_post=torch.from_numpy(np.ascontiguousarray(effect_post[index])),
                    executed_action=torch.from_numpy(np.ascontiguousarray(executed_action[index])),
                    next_visual_key=torch.from_numpy(np.ascontiguousarray(next_visual_key[index])),
                    next_phase=torch.from_numpy(np.ascontiguousarray(next_phase[index])),
                    latent_index=int(latent_index[index]),
                    schema_hash=controller.schema.hash,
                    metadata={
                        "session_id": session_id,
                        "task_cluster": task_cluster,
                        "environment_seed": environment_seed,
                        "attempt_id": int(attempt_ids[index]),
                        "replan_index": int(replan_indices[index]),
                        "terminal_outcome": outcome,
                        "termination_reason": reason,
                        "total_steps": int(total_steps[index]),
                        "final_progress": float(final_progress[index]),
                        "progress_source": "success_only",
                        "restored_from_client_artifact": True,
                    },
                )
            )
        self._transition_session_id = session_id
        self._transition_session_key = AttemptSessionKey(session_id, task_cluster, environment_seed)
        self._transition_attempt_id = last_attempt_id
        self._transition_replan_open = False
        self._transition_pending_start = None
        info = {
            "session_id": session_id,
            "task_cluster": task_cluster,
            "environment_seed": environment_seed,
            "last_attempt_id": last_attempt_id,
            "restored_transitions": n,
            "memory_entries_total": len(controller.memory),
            "conditioning": bool(controller.enabled and not self._transition_shadow),
        }
        log.info(f"[robolab-policy-server] pim_restore {info}")
        return info

    def _build_sample(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt = obs.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("'prompt' must be a string")

        image = _extract_observation_image(obs)
        image_h = self.cfg.image_height
        image_w = self.cfg.image_width
        if image.shape[:2] != (image_h, image_w):
            image = _resize_rgb_uint8(image, (image_h, image_w))
        t_frames = self.cfg.action_chunk_size + 1
        video = torch.zeros((3, t_frames, image_h, image_w), dtype=torch.uint8)  # [3,T,H,W]
        video[:, 0] = torch.from_numpy(image.copy()).permute(2, 0, 1)  # [3,H,W]

        use_state_rows = 1 if self.cfg.use_state else 0
        action = torch.zeros(
            (self.cfg.action_chunk_size + use_state_rows, self.cfg.action_dim),
            dtype=torch.float32,
        )  # [T,D]
        history_action: torch.Tensor | None = None
        num_history_rows = self.cfg.history_length - use_state_rows
        gripper_position: np.ndarray | None = None
        if self.cfg.use_state or num_history_rows > 0:
            gripper_position = 1.0 - _ensure_gripper_array(obs["observation/gripper_position"])

        if self.cfg.action_space == "joint_pos" and (self.cfg.use_state or num_history_rows > 0):
            joint_position = _ensure_2d_float_array(obs["observation/joint_position"], "observation/joint_position", 7)
            if self.cfg.use_state:
                assert gripper_position is not None
                action[0] = torch.from_numpy(np.concatenate((joint_position[-1], gripper_position[-1])))  # [D]
            if num_history_rows > 0:
                assert gripper_position is not None
                if len(joint_position) < num_history_rows + 1:
                    raise ValueError("Not enough joint_position rows for requested history_length")
                history_np = np.concatenate(
                    (joint_position[-num_history_rows - 1 : -1], gripper_position[-num_history_rows - 1 : -1]),
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()  # [H,D]

        if self.cfg.action_space == "midtrain" and (self.cfg.use_state or num_history_rows > 0):
            eef_pos = _ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = _ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            if self.cfg.use_state:
                assert gripper_position is not None
                rot6d = convert_rotation(eef_quat[-1], "quat_xyzw", "rot6d")
                action[0] = torch.from_numpy(np.concatenate((eef_pos[-1], rot6d, gripper_position[-1])))  # [D]
            if num_history_rows > 0:
                assert gripper_position is not None
                if len(eef_pos) < num_history_rows + 1 or len(eef_quat) < num_history_rows + 1:
                    raise ValueError("Not enough eef_pos/eef_quat rows for requested history_length")
                poses_abs = build_abs_pose_from_components(eef_pos, eef_quat, "quat_xyzw")
                poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
                history_np = np.concatenate(
                    [poses_rel[-num_history_rows:], gripper_position[-num_history_rows:]],
                    axis=-1,
                )
                history_action = torch.from_numpy(history_np).float()  # [H,D]

        sample: dict[str, Any] = {
            "ai_caption": prompt,
            "video": video,
            "action": action,
            "conditioning_fps": torch.tensor(self.cfg.conditioning_fps, dtype=torch.long),  # []
            "mode": "wam",
            "domain_id": torch.tensor(get_domain_id(self.cfg.domain_name), dtype=torch.long),  # []
            "viewpoint": "concat_view",
            "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        }
        proprio = np.asarray(obs.get("observation/proprio"), dtype=np.float32)
        if proprio.ndim > 1:
            proprio = proprio[-1]
        if proprio.shape != (self.cfg.proprio_dim,):
            raise ValueError(
                f"'observation/proprio' must have shape [{self.cfg.proprio_dim}], got {proprio.shape}"
            )
        sample["proprio"] = torch.from_numpy(proprio.copy())
        if history_action is not None:
            sample["history_action"] = history_action
        # The selected arm7 channels have min/max statistics of -1/+1, so the
        # configured min-max action normalization is numerically the identity.
        # Build the same model-space sample used by training.
        sample = self._transform(sample, self.cfg.resolution)
        if isinstance(sample.get("ai_caption"), dict):
            sample["ai_caption"] = json.dumps(sample["ai_caption"])
        return sample

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if bool(obs.get("pim_restore_session", False)):
            with self._lock:
                restore_info = (
                    self._restore_pim_session(obs) if self._pim_active else self._restore_transition_session(obs)
                )
            zeva_contract = contract_manifest()
            zeva_contract["pim_conditioning"] = bool(restore_info["conditioning"])
            zeva_contract["memory_backend"] = "pim" if self._pim_active else "transition_memory"
            return {"pim_restore": restore_info, "zeva_contract": zeva_contract}

        if bool(obs.get("pim_finalize_attempt", False)):
            with self._lock:
                with torch.inference_mode():
                    finalize_info = (
                        self._finalize_pim_attempt(obs) if self._pim_active else self._finalize_transition_attempt(obs)
                    )
            zeva_contract = contract_manifest()
            zeva_contract["pim_conditioning"] = bool(finalize_info["conditioning"])
            zeva_contract["memory_backend"] = "pim" if self._pim_active else "transition_memory"
            return {"pim_finalize": finalize_info, "zeva_contract": zeva_contract}

        sample = self._build_sample(obs)
        requested_seed = obs.get("inference_seed")
        seed = self._next_seed() if requested_seed is None else int(requested_seed)
        if seed < 0 or seed >= 2**32:
            raise ValueError("inference_seed must be in uint32 range")
        transition_info: dict[str, Any] | None = None
        pim_info: dict[str, Any] | None = None
        pim_replay_action: np.ndarray | None = None
        cte_features: dict[str, Any] | None = None

        with self._lock:
            with torch.inference_mode():
                if self._zeva_enabled:
                    (
                        global_feature,
                        phase,
                        effect,
                        effect_valid,
                        visual_key,
                        current_effect_post,
                        current_effect_valid,
                    ) = self._causal_interaction_features(
                        obs, _extract_observation_image(obs)
                    )
                    sample["behavior_global"] = global_feature[0].cpu()
                    sample["behavior_phase"] = phase[0].cpu()
                    sample["behavior_effect"] = effect[0].cpu()
                    sample["behavior_effect_valid"] = effect_valid[0].cpu()
                    cte_features = {
                        "global": global_feature[0].detach().float().cpu().numpy(),
                        "phase": phase[0].detach().float().cpu().numpy(),
                        "effect_history": effect[0].detach().float().cpu().numpy(),
                        "visual_key": visual_key[0].detach().float().cpu().numpy(),
                        "latest_effect_post": current_effect_post[0].detach().float().cpu().numpy(),
                        "latest_effect_valid": bool(current_effect_valid),
                        "effect_history_valid": effect_valid[0].detach().cpu().numpy(),
                    }
                    transition_readout = self._transition_context(
                        obs,
                        phase,
                        visual_key,
                        current_effect_post,
                        current_effect_valid,
                    )
                    if transition_readout is not None:
                        transition_context, transition_info = transition_readout
                        if self._transition_shadow:
                            transition_context = torch.zeros_like(transition_context)
                        if self._model_has_online_prefix:
                            sample["behavior_online_context"] = transition_context[0].detach().cpu()
                    elif self._model_has_online_prefix:
                        # Policies without a transition-memory readout receive
                        # an exact zero context.
                        sample["behavior_online_context"] = torch.zeros((256,), dtype=torch.float32)
                    pim_diagnostic_skip_lifecycle = bool(
                        obs.get("pim_diagnostic_skip_lifecycle", False)
                    )
                    pim_readout = (
                        None
                        if pim_diagnostic_skip_lifecycle
                        else self._pim_context(
                            obs,
                            phase,
                            current_effect_post,
                            current_effect_valid,
                        )
                    )
                    if pim_readout is not None:
                        pim_phase, pim_effect, pim_valid, pim_info, pim_replay_action = pim_readout
                        if self._pim_shadow:
                            pim_valid = torch.zeros_like(pim_valid)
                        if self._model_has_pim:
                            sample["behavior_pim_phase"] = pim_phase[0].detach().cpu()
                            sample["behavior_pim_effect"] = pim_effect[0].detach().cpu()
                            sample["behavior_pim_valid"] = pim_valid[0].detach().cpu()
                    elif self._model_has_pim:
                        pim_phase, pim_effect, pim_valid = self._empty_pim_tensors(phase)
                        sample["behavior_pim_phase"] = pim_phase[0].cpu()
                        sample["behavior_pim_effect"] = pim_effect[0].cpu()
                        sample["behavior_pim_valid"] = pim_valid[0].cpu()
                elif self._model_has_online_prefix or self._model_has_pim:
                    raise RuntimeError("Loaded PIM policy requires CTE features")
                data_batch = _build_data_batch_from_sample(sample)
                log.info(f"[robolab-policy-server] prompt={data_batch['ai_caption'][0]!r} seed={seed}")
                samples = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=self.cfg.guidance,
                    seed=[seed],
                    num_steps=self.cfg.num_steps,
                    shift=self.cfg.shift,
                )

        action = samples["action"][0][:, : self.cfg.action_dim]  # [T,D]
        action = action[self.cfg.history_length :]  # [T2,D]
        action_np = action.detach().cpu().numpy()  # [T2,D]
        if self.cfg.action_dim == 7 and action_np.shape[0] != PREDICTED_ACTION_HORIZON:
            raise RuntimeError(
                f"Zeva RoboCasa server must predict {PREDICTED_ACTION_HORIZON} actions, "
                f"got {action_np.shape}"
            )
        if pim_replay_action is not None:
            if (
                pim_replay_action.ndim != 2
                or pim_replay_action.shape[1] != self.cfg.action_dim
                or not 0 < pim_replay_action.shape[0] <= action_np.shape[0]
                or not np.isfinite(pim_replay_action).all()
            ):
                raise ValueError(f"invalid PIM successful replay action shape {pim_replay_action.shape}")
            replay_count = int(pim_replay_action.shape[0])
            blend = self._pim_success_replay_blend
            action_np[:replay_count] = (
                (1.0 - blend) * action_np[:replay_count] + blend * pim_replay_action
            )
            if pim_info is not None:
                pim_info["replay_applied"] = True
                pim_info["replay_action_count"] = replay_count
                pim_info["replay_blend"] = blend
            log.info(
                "[robolab-policy-server] pim_success_replay "
                f"attempt={pim_info.get('attempt_id') if pim_info else None} "
                f"count={replay_count} blend={blend} "
                f"phase_similarity={pim_info.get('replay_phase_similarity') if pim_info else None}"
            )
        elif pim_info is not None:
            pim_info["replay_applied"] = False
        # RoboCasa365 arm7 already stores gripper_close in the simulator convention.
        log.info(
            "[robolab-policy-server] action_raw "
            f"shape={action_np.shape} min={action_np.min():.4f} max={action_np.max():.4f} "
            f"mean_abs={np.abs(action_np).mean():.4f}"
        )

        if self.cfg.action_space == "midtrain":
            eef_pos = _ensure_2d_float_array(obs["observation/eef_pos"], "observation/eef_pos", 3)
            eef_quat = _ensure_2d_float_array(obs["observation/eef_quat"], "observation/eef_quat", 4)
            initial_pose = np.eye(4, dtype=np.float32)
            initial_pose[:3, :3] = convert_rotation(eef_quat[-1], "quat_xyzw", "matrix")
            initial_pose[:3, 3] = eef_pos[-1]
            abs_pose = pose_rel_to_abs(
                action_np[:, :9],
                rotation_format="rot6d",
                pose_convention="backward_framewise",
                initial_pose=initial_pose,
            )
            position = abs_pose[1:, :3, 3]
            quat_xyzw = convert_rotation(abs_pose[1:, :3, :3], "matrix", "quat_xyzw")
            action_np = np.concatenate([position, quat_xyzw, action_np[:, 9:]], axis=-1)

        zeva_contract = contract_manifest()
        zeva_contract.update(
            {
                "inference_seed": seed,
                "pim_conditioning": bool(
                    (
                        self._transition_controller is not None
                        and self._transition_controller.enabled
                        and not self._transition_shadow
                    )
                    or (self._pim_conditioning and not self._pim_shadow)
                ),
                "memory_backend": (
                    "pim" if self._pim_active else ("transition_memory" if self._transition_enabled else "none")
                ),
                "pim_model_hard_bypass": bool(
                    self._pim_active and getattr(self, "_pim_model_hard_bypass", False)
                ),
                "pim_diagnostic_skip_lifecycle": bool(
                    obs.get("pim_diagnostic_skip_lifecycle", False)
                ),
            }
        )
        outputs: dict[str, Any] = {"action": action_np, "zeva_contract": zeva_contract}
        if transition_info is not None:
            outputs["transition_memory"] = transition_info
        if pim_info is not None:
            outputs["pim"] = pim_info
        if cte_features is not None:
            outputs["cte_features"] = cte_features
        if self.cfg.decode_video:
            pred_vision_latent = samples["vision"][0]  # [C,T,H,W]
            video = self.model.decode(pred_vision_latent)  # [1,C,T,H,W]
            video = ((video[0].clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 3, 0)  # [T,H,W,3]
            outputs["video"] = video.detach().cpu().numpy()
        return outputs


def serve(args: RobolabServerArgs) -> None:
    hostname = socket.gethostname()
    log.info(f"[robolab-policy-server] starting host={hostname} bind={args.host}:{int(args.port)}")
    service = RobolabPolicyService(args)
    local_ip = get_local_ip()
    log.info(f"[robolab-policy-server] Server accessible at: ws://{local_ip}:{int(args.port)}/")
    log.info(f"[robolab-policy-server] Health check: http://{local_ip}:{int(args.port)}/healthz")
    server_cls = _load_openpi_websocket_policy_server()
    server_cls(policy=service, host=args.host, port=int(args.port), metadata={}).serve_forever()


def main() -> None:
    cascade_subcommand_args = getattr(
        tyro.conf,
        "CascadeSubcommandArgs",
        tyro.conf.ConsolidateSubcommandArgs,
    )
    args = tyro_cli(
        RobolabServerArgs,
        description=__doc__,
        config=(
            tyro.conf.OmitArgPrefixes,
            cascade_subcommand_args,
            tyro.conf.OmitSubcommandPrefixes,
        ),
    )
    serve(args)


if __name__ == "__main__":
    main()
