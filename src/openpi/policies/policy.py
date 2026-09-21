from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.zeva.memory import CausalMemoryManager

BasePolicy: TypeAlias = _base_policy.BasePolicy
class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        action_norm_stats: Any | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        print("Input transforms:", transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self.replan_steps = 5
        self._action_norm_stats = action_norm_stats

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
        
        self._causal_memory = None
        if self._is_pytorch_model and getattr(self._model, "use_zeva", False):
            config = self._model.config
            self._causal_memory = CausalMemoryManager(
                brief_size=config.brief_memory_size,
                persistent_size=config.persistent_memory_size,
                retrieval_top_k=config.causal_retrieval_top_k,
                merge_phase_weight=config.causal_merge_phase_weight,
                merge_signal_weight=config.causal_merge_signal_weight,
                merge_threshold=config.causal_merge_threshold,
                use_brief_memory=config.use_brief_memory,
                use_persistent_memory=config.use_persistent_memory,
            )
        self.reset(scope="episode")
        
    def reset(self, scope: str = "episode"):
        """Reset Zeva state at attempt or fixed-episode boundaries.

        BIT and Mamba recurrence are attempt-local. PIM survives an attempt
        reset and is cleared only for a new fixed episode.
        """
        if scope not in {"attempt", "episode"}:
            raise ValueError(f"Unknown reset scope: {scope!r}.")
        existing_schema = None
        if hasattr(self, "_inference_state") and scope == "attempt":
            existing_schema = self._inference_state.get("task_schema")
        if self._causal_memory is not None:
            if scope == "episode":
                self._causal_memory.reset_episode()
            else:
                self._causal_memory.reset_attempt()
        self._inference_state = {
            "task_schema": existing_schema,
            "phase_token": None,
            "causal_context": None,
            "previous_image": None,
            "pending_normalized_actions": None,
            "cte_inference_params": None,
        }

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        runtime = {
            key: inputs.pop(key, None)
            for key in (
                "executed_actions",
                "executed_actions_normalized",
                "executed_steps",
                "episode_id",
                "attempt_id",
                "observe_only",
            )
        }
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        task_schema = None
        phase_token = None
        causal_context = None
        use_zeva = self._is_pytorch_model and getattr(self._model, "use_zeva", False)
        if use_zeva:
            current_image = observation.images["base_0_rgb"]
            causal_action_dim = self._model.config.causal_action_dim
            with torch.no_grad():
                task_schema = self._inference_state.get("task_schema")
                if task_schema is None:
                    vlm_features = self._model.extract_vlm_features(observation)
                    task_schema = self._model.retrieve_task_schema(vlm_features)
                    self._inference_state["task_schema"] = task_schema

                previous_image = self._inference_state.get("previous_image")
                pending_actions = self._inference_state.get("pending_normalized_actions")
                if previous_image is not None and pending_actions is not None:
                    executed_normalized = runtime["executed_actions_normalized"]
                    if executed_normalized is not None:
                        executed_normalized = torch.as_tensor(
                            executed_normalized, dtype=torch.float32, device=self._pytorch_device
                        )
                        if executed_normalized.ndim == 2:
                            executed_normalized = executed_normalized.unsqueeze(0)
                    elif runtime["executed_actions"] is not None and self._action_norm_stats is not None:
                        executed_raw = torch.as_tensor(
                            runtime["executed_actions"], dtype=torch.float32, device=self._pytorch_device
                        )
                        if executed_raw.ndim == 2:
                            executed_raw = executed_raw.unsqueeze(0)
                        if self._model.config.causal_action_normalization == "mean_std":
                            mean = torch.as_tensor(
                                self._action_norm_stats.mean[:causal_action_dim],
                                dtype=torch.float32,
                                device=self._pytorch_device,
                            )
                            std = torch.as_tensor(
                                self._action_norm_stats.std[:causal_action_dim],
                                dtype=torch.float32,
                                device=self._pytorch_device,
                            )
                            executed_normalized = (executed_raw[..., :causal_action_dim] - mean) / (std + 1e-8)
                        else:
                            q01 = torch.as_tensor(
                                self._action_norm_stats.q01[:causal_action_dim],
                                dtype=torch.float32,
                                device=self._pytorch_device,
                            )
                            q99 = torch.as_tensor(
                                self._action_norm_stats.q99[:causal_action_dim],
                                dtype=torch.float32,
                                device=self._pytorch_device,
                            )
                            executed_normalized = (
                                (executed_raw[..., :causal_action_dim] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
                            )
                    else:
                        executed_steps = runtime["executed_steps"]
                        if executed_steps is None and runtime["executed_actions"] is not None:
                            executed_steps = len(runtime["executed_actions"])
                        executed_steps = int(executed_steps or min(self.replan_steps, pending_actions.shape[1]))
                        executed_normalized = pending_actions[:, :executed_steps]

                    causal_encoding, inference_params = self._model.encode_causal_transition(
                        previous_image,
                        executed_normalized,
                        current_image,
                        inference_params=self._inference_state.get("cte_inference_params"),
                    )
                    self._inference_state["cte_inference_params"] = inference_params
                    phase_token = causal_encoding.phase_token
                    self._causal_memory.update(phase_token, causal_encoding.causal_signal)
                else:
                    phase_token = self._model.initialize_causal_phase(current_image)

                brief = self._causal_memory.brief_tensor(device=self._pytorch_device)
                retrieved = self._causal_memory.retrieve(phase_token, device=self._pytorch_device)
                causal_context = self._model.build_causal_context(task_schema, phase_token, brief, retrieved)
                self._inference_state["phase_token"] = phase_token
                self._inference_state["causal_context"] = causal_context
                self._inference_state["previous_image"] = current_image.detach().clone()

            if runtime["observe_only"]:
                return {
                    "actions": np.empty((0, causal_action_dim), dtype=np.float32),
                    "zeva_memory": self._causal_memory.snapshot(),
                }
        
        start_time = time.monotonic()
        model_kwargs = sample_kwargs
        if use_zeva:
            model_kwargs = {
                **sample_kwargs,
                "task_schema": task_schema,
                "phase_token": phase_token,
                "causal_context": causal_context,
            }
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **model_kwargs),
        }
        model_time = time.monotonic() - start_time
        
        if use_zeva:
            self._inference_state["pending_normalized_actions"] = (
                outputs["actions"][..., :causal_action_dim].detach().clone()
            )
        
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if use_zeva:
            outputs["zeva_memory"] = self._causal_memory.snapshot()
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def reset(self, scope: str = "episode") -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset(scope=scope)
