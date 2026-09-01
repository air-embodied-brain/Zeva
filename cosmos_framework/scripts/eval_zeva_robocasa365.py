#!/usr/bin/env python3
"""Closed-loop RoboCasa365 evaluator for the released Atomic-5 policy."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

import robocasa  # noqa: F401
import robocasa.wrappers.gym_wrapper  # noqa: F401
from robocasa.utils.env_utils import convert_action


TASK_PROMPTS = {
    "OpenStandMixerHead": "Open the stand mixer head.",
    "TurnOnElectricKettle": "Press down the lever to turn on the electric kettle.",
    "CloseToasterOvenDoor": "Close the toaster oven door.",
    "TurnOnMicrowave": "Press the start button on the microwave.",
    "CoffeeSetupMug": "Pick the mug from the counter and place it under the coffee machine dispenser.",
}


def _obs_value(obs: dict, *keys: str) -> np.ndarray:
    for key in keys:
        if key in obs:
            return np.asarray(obs[key])
    raise KeyError(f"Missing observation; tried {keys}. Available: {sorted(obs)}")


def prepare_policy_observation(obs: dict, prompt: str) -> tuple[dict, np.ndarray]:
    left = _obs_value(obs, "video.robot0_agentview_left", "robot0_agentview_left_image")
    wrist = _obs_value(obs, "video.robot0_eye_in_hand", "robot0_eye_in_hand_image")
    if left.shape != wrist.shape or left.ndim != 3 or left.shape[-1] != 3:
        raise ValueError(f"Expected matching RGB views, got left={left.shape}, wrist={wrist.shape}")
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
    if proprio.shape != (9,):
        raise ValueError(f"Expected arm-only 9D proprio, got {proprio.shape}")
    return {
        "prompt": prompt,
        "observation/image": image,
        "observation/proprio": proprio,
    }, left


def env_step(env, arm_action: np.ndarray):
    arm_action = np.asarray(arm_action, dtype=np.float32)
    if arm_action.shape != (7,) or not np.isfinite(arm_action).all():
        raise ValueError(
            f"Invalid arm7 action: shape={arm_action.shape}, finite={np.isfinite(arm_action).all()}"
        )
    simulator_action = np.zeros(12, dtype=np.float32)
    simulator_action[:7] = np.clip(arm_action, -1.0, 1.0)
    result = env.step(convert_action(simulator_action))
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = result
    success = bool(info.get("success", False) or reward > 0)
    return obs, success, done


def save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Inconsistent video frame shape {frame.shape}")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--pim-formal32",
        action="store_true",
        help=(
            "Send the PIM lifecycle fields while preserving the released "
            "32-control formal evaluation protocol."
        ),
    )
    parser.add_argument(
        "--open-loop-steps",
        type=int,
        default=32,
        help="Execute the complete 32-action training chunk; shorter values invalidate comparison.",
    )
    parser.add_argument("--split", default="target")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--render-gpu-device-id",
        type=int,
        default=-1,
        help="MuJoCo EGL device index visible to this evaluator (-1: automatic).",
    )
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if args.pim_formal32 and args.open_loop_steps != 32:
        raise ValueError("--pim-formal32 requires --open-loop-steps=32")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = WebsocketClientPolicy(args.host, args.port)
    results = []

    for episode_idx in range(args.episodes):
        episode_seed = args.seed + episode_idx
        env = gym.make(
            f"robocasa/{args.task}",
            split=args.split,
            seed=episode_seed,
            camera_widths=256,
            camera_heights=256,
            enable_render=True,
            render_gpu_device_id=args.render_gpu_device_id,
            robots="PandaOmron",
            randomize_cameras=False,
            translucent_robot=False,
        )
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        frames: list[np.ndarray] = []
        action_queue: deque[np.ndarray] = deque()

        initial_policy_obs, _ = prepare_policy_observation(obs, TASK_PROMPTS[args.task])
        cte_boundary_images: list[np.ndarray] = [initial_policy_obs["observation/image"].copy()]
        cte_transitions: list[np.ndarray] = []
        pending_transition_actions: list[np.ndarray] = []
        actions_since_replan: list[np.ndarray] = []
        pim_query_info: list[dict] = []
        query_times = []
        success = False
        done = False
        steps = 0

        while steps < args.max_steps and not success and not done:
            policy_obs, left_frame = prepare_policy_observation(obs, TASK_PROMPTS[args.task])
            policy_obs["cte_boundary_images"] = np.stack(cte_boundary_images, axis=0)
            policy_obs["cte_transition_actions"] = (
                np.stack(cte_transitions, axis=0)
                if cte_transitions
                else np.empty((0, 4, 7), dtype=np.float32)
            )
            policy_obs["cte_reset"] = steps == 0
            if not args.no_video:
                frames.append(left_frame.copy())
            if not action_queue:
                if args.pim_formal32:
                    policy_obs.update(
                        {
                            "pim_session_id": f"formal32:{args.task}:{episode_seed}",
                            "pim_task_cluster": args.task,
                            "pim_environment_seed": episode_seed,
                            "pim_attempt_id": 0,
                            "pim_memory_reset": steps == 0,
                            "pim_replan_reset": False,
                            "pim_latent_index": steps,
                            "pim_replan_index": len(query_times),
                            "pim_executed_action": np.asarray(
                                actions_since_replan, dtype=np.float32
                            ).reshape(-1, 7),
                            "pim_transition_complete": len(actions_since_replan) == 32,
                        }
                    )
                start = time.monotonic()
                output = client.infer(policy_obs)
                query_times.append(time.monotonic() - start)
                if args.pim_formal32:
                    contract = output.get("zeva_contract")
                    if not isinstance(contract, dict):
                        raise ValueError("PIM server did not return zeva_contract")
                    if contract.get("memory_backend") != "pim" or not bool(
                        contract.get("pim_conditioning", False)
                    ):
                        raise ValueError(f"PIM conditioning is not active: {contract}")
                    info = output.get("pim")
                    if not isinstance(info, dict):
                        raise ValueError("PIM server did not return retrieval metadata")
                    pim_query_info.append(dict(info))
                actions = np.asarray(output["action"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"Policy returned action shape {actions.shape}, expected [T,7]")
                action_queue.extend(actions[: args.open_loop_steps])
                actions_since_replan = []
            executed_action = action_queue.popleft()
            obs, success, done = env_step(env, executed_action)
            actions_since_replan.append(executed_action.astype(np.float32, copy=True))
            pending_transition_actions.append(executed_action.astype(np.float32, copy=True))
            if len(pending_transition_actions) == 4:
                boundary_policy_obs, _ = prepare_policy_observation(obs, TASK_PROMPTS[args.task])
                cte_boundary_images.append(boundary_policy_obs["observation/image"].copy())
                cte_transitions.append(np.stack(pending_transition_actions, axis=0))
                pending_transition_actions.clear()
            steps += 1

        if args.pim_formal32:
            final_policy_obs, _ = prepare_policy_observation(obs, TASK_PROMPTS[args.task])
            final_policy_obs["cte_boundary_images"] = np.stack(cte_boundary_images, axis=0)
            final_policy_obs["cte_transition_actions"] = (
                np.stack(cte_transitions, axis=0)
                if cte_transitions
                else np.empty((0, 4, 7), dtype=np.float32)
            )
            final_policy_obs.update(
                {
                    "pim_finalize_attempt": True,
                    "pim_session_id": f"formal32:{args.task}:{episode_seed}",
                    "pim_task_cluster": args.task,
                    "pim_environment_seed": episode_seed,
                    "pim_attempt_id": 0,
                    "pim_latent_index": steps,
                    "pim_executed_action": np.asarray(
                        actions_since_replan, dtype=np.float32
                    ).reshape(-1, 7),
                    "pim_transition_complete": len(actions_since_replan) == 32,
                    "pim_terminal_outcome": "success" if success else "failure",
                    "pim_termination_reason": "success" if success else (
                        "env_done" if done else "max_steps"
                    ),
                    "pim_total_steps": steps,
                    "pim_final_progress": 1.0 if success else 0.0,
                }
            )
            finalize_output = client.infer(final_policy_obs)
            if not isinstance(finalize_output.get("pim_finalize"), dict):
                raise ValueError("PIM server did not acknowledge attempt finalization")

        env.close()
        record = {
            "task": args.task,
            "episode": episode_idx,
            "seed": episode_seed,
            "success": success,
            "steps": steps,
            "queries": len(query_times),
            "mean_query_seconds": float(np.mean(query_times)) if query_times else None,
            "pim_formal32": bool(args.pim_formal32),
            "pim_entries_last": (
                int(pim_query_info[-1].get("entries_total", 0)) if pim_query_info else 0
            ),
        }
        results.append(record)
        with (args.output_dir / "episodes.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        if not args.no_video:
            save_video(
                frames,
                args.output_dir
                / f"ep_{episode_idx:03d}_seed_{episode_seed}_{'success' if success else 'fail'}.mp4",
            )
        print(json.dumps(record), flush=True)

    summary = {
        "task": args.task,
        "episodes": len(results),
        "successes": sum(item["success"] for item in results),
        "success_rate": float(np.mean([item["success"] for item in results])),
        "mean_steps": float(np.mean([item["steps"] for item in results])),
        "mean_query_seconds": float(
            np.mean(
                [
                    item["mean_query_seconds"]
                    for item in results
                    if item["mean_query_seconds"] is not None
                ]
            )
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
