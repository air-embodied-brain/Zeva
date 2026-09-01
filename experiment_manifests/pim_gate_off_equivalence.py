#!/usr/bin/env python3
"""Compare one identical RoboCasa observation against base and gate-off PIM servers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

import robocasa  # noqa: F401
import robocasa.wrappers.gym_wrapper  # noqa: F401


TASK_PROMPTS = {
    "OpenStandMixerHead": "Open the stand mixer head.",
    "TurnOnElectricKettle": "Press down the lever to turn on the electric kettle.",
    "CloseToasterOvenDoor": "Close the toaster oven door.",
    "TurnOnMicrowave": "Press the start button on the microwave.",
    "CoffeeSetupMug": "Pick the mug from the counter and place it under the coffee machine dispenser.",
}


def obs_value(obs: dict, *keys: str) -> np.ndarray:
    for key in keys:
        if key in obs:
            return np.asarray(obs[key])
    raise KeyError(f"Missing observation; tried {keys}. Available: {sorted(obs)}")


def prepare_policy_observation(obs: dict, prompt: str) -> tuple[dict, np.ndarray]:
    left = obs_value(obs, "video.robot0_agentview_left", "robot0_agentview_left_image")
    wrist = obs_value(obs, "video.robot0_eye_in_hand", "robot0_eye_in_hand_image")
    image = np.concatenate([left, wrist], axis=1).astype(np.uint8, copy=False)
    if "state.end_effector_position_relative" in obs:
        proprio = np.concatenate(
            [
                np.asarray(obs["state.end_effector_position_relative"], dtype=np.float32),
                np.asarray(obs["state.end_effector_rotation_relative"], dtype=np.float32),
                np.asarray(obs["state.gripper_qpos"], dtype=np.float32),
            ]
        )
    else:
        proprio = np.concatenate(
            [
                np.asarray(obs["robot0_base_to_eef_pos"], dtype=np.float32),
                np.asarray(obs["robot0_base_to_eef_quat"], dtype=np.float32),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            ]
        )
    if image.shape != (256, 512, 3) or proprio.shape != (9,):
        raise ValueError(f"Unexpected observation shapes: image={image.shape}, proprio={proprio.shape}")
    return {
        "prompt": prompt,
        "observation/image": image,
        "observation/proprio": proprio,
    }, left


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-host", default=None)
    parser.add_argument("--pim-host", default=None)
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--pim-port", type=int, required=True)
    parser.add_argument("--task", default="TurnOnMicrowave", choices=sorted(TASK_PROMPTS))
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--repeat-each", type=int, default=1)
    parser.add_argument("--base-skip-pim-lifecycle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    env = gym.make(
        f"robocasa/{args.task}",
        split="target",
        seed=args.seed,
        camera_widths=256,
        camera_heights=256,
        enable_render=True,
        render_gpu_device_id=args.render_gpu_device_id,
        robots="PandaOmron",
        randomize_cameras=False,
        translucent_robot=False,
    )
    try:
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        request, _ = prepare_policy_observation(obs, TASK_PROMPTS[args.task])
        request["cte_boundary_images"] = np.expand_dims(
            request["observation/image"].copy(), axis=0
        )
        request["cte_transition_actions"] = np.empty((0, 4, 7), dtype=np.float32)
        request["cte_reset"] = True
        request["inference_seed"] = 0
        request.update(
            {
                "pim_session_id": f"gateoff-equivalence:{args.task}:{args.seed}",
                "pim_task_cluster": args.task,
                "pim_environment_seed": args.seed,
                "pim_attempt_id": 0,
                "pim_memory_reset": True,
                "pim_replan_reset": False,
                "pim_latent_index": 0,
                "pim_replan_index": 0,
                "pim_executed_action": np.empty((0, 7), dtype=np.float32),
                "pim_transition_complete": False,
            }
        )

        base_host = args.base_host or args.host
        pim_host = args.pim_host or args.host
        base_request = dict(request)
        pim_request = dict(request)
        if args.base_skip_pim_lifecycle:
            base_request["pim_diagnostic_skip_lifecycle"] = True
            pim_request["pim_diagnostic_skip_lifecycle"] = False
        if args.repeat_each < 1:
            raise ValueError("--repeat-each must be positive")
        base_client = WebsocketClientPolicy(base_host, args.base_port)
        pim_client = WebsocketClientPolicy(pim_host, args.pim_port)
        base_outputs = [base_client.infer(base_request) for _ in range(args.repeat_each)]
        pim_outputs = [pim_client.infer(pim_request) for _ in range(args.repeat_each)]
        base_runs = [np.asarray(value["action"], dtype=np.float32) for value in base_outputs]
        pim_runs = [np.asarray(value["action"], dtype=np.float32) for value in pim_outputs]
        base = base_runs[0]
        pim = pim_runs[0]
    finally:
        env.close()

    if base.shape != (32, 7) or pim.shape != (32, 7):
        raise ValueError(f"Unexpected action shapes: base={base.shape}, pim={pim.shape}")
    delta = np.abs(base - pim)
    base_repeat_delta = max(
        (float(np.abs(base - value).max()) for value in base_runs[1:]), default=0.0
    )
    pim_repeat_delta = max(
        (float(np.abs(pim - value).max()) for value in pim_runs[1:]), default=0.0
    )
    feature_comparison = {}
    base_features = base_outputs[0].get("cte_features", {})
    pim_features = pim_outputs[0].get("cte_features", {})
    for name in sorted(set(base_features) & set(pim_features)):
        base_value = np.asarray(base_features[name])
        pim_value = np.asarray(pim_features[name])
        if base_value.shape != pim_value.shape or base_value.dtype == np.bool_:
            feature_comparison[name] = {
                "base_shape": list(base_value.shape),
                "pim_shape": list(pim_value.shape),
                "exact_equal": bool(np.array_equal(base_value, pim_value)),
            }
            continue
        feature_delta = np.abs(base_value.astype(np.float32) - pim_value.astype(np.float32))
        feature_comparison[name] = {
            "shape": list(base_value.shape),
            "exact_equal": bool(np.array_equal(base_value, pim_value)),
            "max_abs_delta": float(feature_delta.max(initial=0.0)),
            "mean_abs_delta": float(feature_delta.mean()) if feature_delta.size else 0.0,
        }
    report = {
        "format": "pim_gate_off_equivalence_v1",
        "task": args.task,
        "environment_seed": args.seed,
        "inference_seed": 0,
        "observation_image_sha256": array_hash(request["observation/image"]),
        "observation_proprio_sha256": array_hash(request["observation/proprio"]),
        "base_endpoint": f"{base_host}:{args.base_port}",
        "pim_endpoint": f"{pim_host}:{args.pim_port}",
        "base_skip_pim_lifecycle": args.base_skip_pim_lifecycle,
        "base_zeva_contract": base_outputs[0].get("zeva_contract"),
        "pim_zeva_contract": pim_outputs[0].get("zeva_contract"),
        "pim_metadata_present": isinstance(pim_outputs[0].get("pim"), dict),
        "base_shape": list(base.shape),
        "pim_shape": list(pim.shape),
        "base_sha256": array_hash(base),
        "pim_sha256": array_hash(pim),
        "exact_equal": bool(np.array_equal(base, pim)),
        "allclose_atol_1e_6": bool(np.allclose(base, pim, rtol=0.0, atol=1e-6)),
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "repeat_each": args.repeat_each,
        "base_repeat_max_abs_delta": base_repeat_delta,
        "pim_repeat_max_abs_delta": pim_repeat_delta,
        "cte_feature_comparison": feature_comparison,
        "finite": bool(np.isfinite(base).all() and np.isfinite(pim).all()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["allclose_atol_1e_6"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
