# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WebSocket inference server for the fixed-base RoboCasa365 Zeva policy.

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
import tyro

from cosmos_framework.data.generator.action.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    convert_rotation,
    pose_abs_to_rel,
    pose_rel_to_abs,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.zeva import (
    CausalTransitionEncoder,
    CausalTransitionEncoderConfig,
    StaticTaskContextRetrievalConfig,
    StaticTaskContextRetrievalHead,
    normalize_cte_state_dict,
    retrieve_static_task_context,
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
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
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


def format_action_prompt(instruction: str) -> str:
    """Format a RoboCasa instruction exactly as the static retrieval head saw it."""
    sample = {
        "ai_caption": instruction.strip(),
        "video": torch.empty((3, 33, 1, 1), dtype=torch.uint8),
        "conditioning_fps": torch.tensor(20.0),
        "image_size": torch.tensor([480, 832, 480, 832]),
        "viewpoint": "concat_view",
        "additional_view_description": _CONCAT_VIEW_DESCRIPTION,
        "mode": "wam",
    }
    formatted = ActionPromptJsonFormatter(caption_key="ai_caption")(sample)["ai_caption"]
    return json.dumps(formatted) if isinstance(formatted, dict) else str(formatted)


@torch.inference_mode()
def extract_batch(model, latent_batch: torch.Tensor, prompts: list[str]) -> torch.Tensor:
    """Extract clean initial-image policy readouts used by static task-context retrieval."""
    batch_size = len(prompts)
    plans = [
        SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0])
        for _ in prompts
    ]
    token_ids = model._tokenize_captions(
        prompts,
        use_system_prompt=bool(model.vlm_config.use_system_prompt),
        system_prompt=None,
        is_video=False,
    )
    clean = GenerationDataClean(
        batch_size=batch_size,
        is_image_batch=True,
        x0_tokens_vision=[latent_batch[i].unsqueeze(0).unsqueeze(2) for i in range(batch_size)],
        fps_vision=torch.full((batch_size,), 20.0),
    )
    packed = model._pack_input_sequence(
        plans,
        token_ids,
        clean,
        torch.zeros(batch_size, dtype=torch.float32),
        include_end_of_generation_token=model._derive_include_end_of_generation_token(),
    )
    if packed.behavior_indexes.numel() != 0:
        raise RuntimeError("Clean task-context extraction unexpectedly contains Zeva prefix tokens")
    packed.to_cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model.net(packed_seq=packed)
    hidden = output["last_hidden_state"]
    indexes = packed.vision.sequence_indexes
    readouts = []
    sample_start = 0
    for sample_length in packed.sample_lens:
        sample_end = sample_start + sample_length
        sample_indexes = indexes[(indexes >= sample_start) & (indexes < sample_end)]
        readouts.append(hidden.index_select(0, sample_indexes).mean(dim=0))
        sample_start = sample_end
    return torch.stack(readouts).float().cpu()


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
    """Override structured JSON prompting. The Atomic-5 service defaults this to true."""
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
        assert isinstance(pipe.setup_args, OmniSetupArgs)
        self.setup_args: OmniSetupArgs = pipe.setup_args

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
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(self.cfg.seed)
        resolved_domain_id = get_domain_id(self.cfg.domain_name)
        log.info(
            f"[robolab-policy-server] ready domain={self.cfg.domain_name!r} domain_id={resolved_domain_id} "
            f"resolution={self.cfg.resolution!r} "
            f"action_space={self.cfg.action_space} action_dim={self.cfg.action_dim} "
            f"chunk={self.cfg.action_chunk_size} history={self.cfg.history_length} use_state={self.cfg.use_state} "
            f"image={self.cfg.image_height}x{self.cfg.image_width} fps={self.cfg.conditioning_fps} "
            f"guidance={self.cfg.guidance} num_steps={self.cfg.num_steps} shift={self.cfg.shift} "
            f"prompt_json={inferred.get('format_prompt_as_json')} seed={self.cfg.seed} "
            f"deterministic_seed={self.cfg.deterministic_seed}"
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
            format_json = True if args.format_prompt_as_json is None else bool(args.format_prompt_as_json)
            inferred["format_prompt_as_json"] = format_json
            log.warning(
                "[robolab-policy-server] no training action dataset config found; using default "
                f"ActionTransformPipeline (format_prompt_as_json={format_json})"
            )
            return (
                ActionTransformPipeline(
                    max_action_dim=max_action_dim,
                    cfg_dropout_rate=0.0,
                    format_prompt_as_json=format_json,
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

        format_json = bool(getattr(action_dataset_config, "format_prompt_as_json", True))
        if args.format_prompt_as_json is not None:
            format_json = bool(args.format_prompt_as_json)
        inferred["format_prompt_as_json"] = format_json
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
        # Early GRU checkpoints were exported with ``use_mamba=True`` in their
        # metadata even though their state dict contains nn.GRU parameters.
        # Infer the serialized architecture from its keys so installing
        # mamba_ssm later cannot silently change the model being constructed.
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

    def _causal_interaction_features(self, obs: dict[str, Any], image: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return global_feature, encoded["phase"][:, -1].float(), history, history_valid

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
        sample = self._build_sample(obs)
        requested_seed = obs.get("inference_seed")
        seed = self._next_seed() if requested_seed is None else int(requested_seed)
        if seed < 0 or seed >= 2**32:
            raise ValueError("inference_seed must be in uint32 range")
        cte_features: dict[str, Any] | None = None

        with self._lock:
            with torch.inference_mode():
                if self._zeva_enabled:
                    global_feature, phase, effect, effect_valid = self._causal_interaction_features(
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
                        "effect_history_valid": effect_valid[0].detach().cpu().numpy(),
                    }
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

        outputs: dict[str, Any] = {"action": action_np}
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
