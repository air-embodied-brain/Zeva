#!/usr/bin/env python3
"""Closed-loop RoboCasa365 evaluation client for the Zeva Atomic-5 policy with PIM."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import cv2
import gymnasium as gym
import msgpack_numpy
import numpy as np
import websockets.sync.client
from openpi_client.websocket_client_policy import WebsocketClientPolicy

import robocasa  # noqa: F401
# Gymnasium registration for the Panda-Omron adapter resides in this module
# rather than in RoboCasa's package initializer.
import robocasa.wrappers.gym_wrapper  # noqa: F401
from robocasa.utils.env_utils import convert_action

try:
    from cosmos_framework.model.zeva.attempt_protocol import (
        EXECUTED_ACTION_HORIZON,
        MAX_CONTROLS_PER_ATTEMPT,
        PREDICTED_ACTION_HORIZON,
        PROTOCOL_VERSION,
        REQUIRED_WARMUP_REQUESTS,
        contract_manifest,
        default_session_id,
        inference_seed,
    )
except ModuleNotFoundError:
    # Allow the evaluator and contract file to be copied together to a
    # simulator-only machine that does not contain the full Cosmos source.
    from attempt_protocol import (  # type: ignore[no-redef]
        EXECUTED_ACTION_HORIZON,
        MAX_CONTROLS_PER_ATTEMPT,
        PREDICTED_ACTION_HORIZON,
        PROTOCOL_VERSION,
        REQUIRED_WARMUP_REQUESTS,
        contract_manifest,
        default_session_id,
        inference_seed,
    )

try:
    from cosmos_framework.model.zeva.attempt_artifacts import (
        ATTEMPT_RECORD_VERSION,
        CHUNK_RECORD_VERSION,
        sha256_file,
        validate_attempt_record,
        write_manifest,
    )
except ModuleNotFoundError:
    from attempt_artifacts import (  # type: ignore[no-redef]
        ATTEMPT_RECORD_VERSION,
        CHUNK_RECORD_VERSION,
        sha256_file,
        validate_attempt_record,
        write_manifest,
    )


TASK_PROMPTS = {
    "OpenStandMixerHead": "Open the stand mixer head.",
    "TurnOnElectricKettle": "Press down the lever to turn on the electric kettle.",
    "CloseToasterOvenDoor": "Close the toaster oven door.",
    "TurnOnMicrowave": "Press the start button on the microwave.",
    "CoffeeSetupMug": "Pick the mug from the counter and place it under the coffee machine dispenser.",
}


class _LongInferenceWebsocketClientPolicy(WebsocketClientPolicy):
    """OpenPI client without a keepalive deadline during long GPU inference.

    A normal request takes only a few seconds, but a temporarily contended GPU
    can exceed websockets' 20-second keepalive timeout. The request itself is
    synchronous and already bounded by the evaluator process, so disabling
    protocol pings prevents a completed model outcome from being mislabeled as
    an infrastructure failure.
    """

    def _wait_for_server(self):  # type: ignore[no-untyped-def, override]
        logging.info("Waiting for server at %s...", self._uri)
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        connection = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            ping_interval=None,
        )
        metadata = msgpack_numpy.unpackb(connection.recv())
        return connection, metadata


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
        raise ValueError(f"Invalid arm7 action: shape={arm_action.shape}, finite={np.isfinite(arm_action).all()}")
    raw = np.zeros(12, dtype=np.float32)
    raw[:7] = np.clip(arm_action, -1.0, 1.0)
    result = env.step(convert_action(raw))
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = result
    success = bool(info.get("success", False) or reward > 0)
    return obs, success, done, dict(info)


def task_predicate_progress(env, task: str) -> tuple[float, dict[str, float | bool | str]]:
    """Return a transparent task-state progress signal for Phase-2 supervision."""
    core = env.unwrapped.env
    if task == "TurnOnMicrowave":
        turned_on = bool(core.microwave.get_state()["turned_on"])
        button_id = core.sim.model.geom_name2id(
            f"{core.microwave.naming_prefix}start_button"
        )
        button_pos = core.sim.data.geom_xpos[button_id]
        gripper_pos = core.sim.data.site_xpos[core.robots[0].eef_site_id["right"]]
        distance = float(np.linalg.norm(gripper_pos - button_pos))
        reach = float(np.exp(-distance / 0.15))
        # Before activation, proximity supplies only the first half of progress.
        # After activation, moving >15cm away completes RoboCasa's success predicate.
        progress = (0.5 * reach) if not turned_on else (0.75 + 0.25 * min(distance / 0.15, 1.0))
        return float(np.clip(progress, 0.0, 1.0)), {
            "schema": "turn_on_microwave_predicate_progress_v1",
            "turned_on": turned_on,
            "gripper_button_distance": distance,
            "reach_score": reach,
        }
    if task == "OpenStandMixerHead":
        head = float(core.stand_mixer.get_state(core)["head"])
        return float(np.clip(head, 0.0, 1.0)), {
            "schema": "open_stand_mixer_head_predicate_progress_v1",
            "head_joint": head,
        }
    # Phase-2 training currently admits only tasks with an explicit dense adapter.
    return 1.0 if bool(core._check_success()) else 0.0, {
        "schema": "terminal_predicate_only_v1",
    }


def _feature_array(items: list[np.ndarray], width: int) -> np.ndarray:
    if not items:
        return np.empty((0, width), dtype=np.float32)
    result = np.asarray(items, dtype=np.float32)
    if result.shape != (len(items), width):
        raise ValueError(f"invalid behavior feature shape: expected [N,{width}], got {result.shape}")
    return result


def restore_session_from_artifacts(
    client: WebsocketClientPolicy,
    output_dir: Path,
    records: list[dict],
    *,
    session_id: str,
    task_cluster: str,
    environment_seed: int,
) -> dict:
    """Restore completed history so the local manifest remains the session source of truth."""
    restored: dict[str, list] = {
        "phase": [], "visual_key": [], "effect_post": [], "effect_valid": [], "executed_action": [],
        "next_visual_key": [], "next_phase": [], "latent_index": [], "attempt_ids": [],
        "replan_indices": [], "outcomes": [], "termination_reasons": [],
        "total_steps": [], "final_progress": [],
    }
    success_trace_phase = np.empty((0, 128), dtype=np.float32)
    success_trace_action = np.empty((0, EXECUTED_ACTION_HORIZON, 7), dtype=np.float32)
    success_trace_action_count = np.empty((0,), dtype=np.int64)
    success_trace_attempt_id = -1
    for record in records:
        artifact = np.load(output_dir / record["artifact_npz"])
        complete = [chunk for chunk in record["chunk_records"] if chunk["completed_16"]]
        available = min(
            int(artifact["phase"].shape[0]),
            int(artifact["visual_key"].shape[0]),
            int(artifact["latest_effect_post"].shape[0]),
            int(artifact["latest_effect_valid"].shape[0]),
        ) - 1
        if len(complete) > available:
            terminal_without_post = (
                len(complete) == available + 1
                and bool(record["success"])
                and bool(complete[-1].get("terminal_chunk"))
            )
            if not terminal_without_post:
                raise ValueError(
                    "cannot restore a complete chunk without its post-transition behavior features"
                )
            # A success exactly on a 16-control boundary has no next
            # observation. It is not a valid transition-memory entry, but it
            # remains part of the separately restored successful replay trace.
            complete = complete[:available]
        n = len(complete)
        actions = artifact["executed_actions"][: n * EXECUTED_ACTION_HORIZON].reshape(n, 16, 7)
        restored["phase"].extend(artifact["phase"][:n])
        restored["visual_key"].extend(artifact["visual_key"][:n])
        restored["effect_post"].extend(artifact["latest_effect_post"][1 : n + 1])
        restored["effect_valid"].extend(artifact["latest_effect_valid"][1 : n + 1])
        restored["executed_action"].extend(actions)
        restored["next_visual_key"].extend(artifact["visual_key"][1 : n + 1])
        restored["next_phase"].extend(artifact["phase"][1 : n + 1])
        restored["latent_index"].extend(chunk["end_step"] for chunk in complete)
        restored["attempt_ids"].extend([record["attempt_id"]] * n)
        restored["replan_indices"].extend(chunk["replan_index"] for chunk in complete)
        restored["outcomes"].extend([record["terminal_outcome"]] * n)
        restored["termination_reasons"].extend([record["termination_reason"]] * n)
        restored["total_steps"].extend([record["total_steps"]] * n)
        restored["final_progress"].extend([record["final_progress"]] * n)
        if bool(record["success"]) and success_trace_attempt_id < 0:
            query_steps = np.asarray(artifact["query_steps"], dtype=np.int64)
            executed_actions = np.asarray(artifact["executed_actions"], dtype=np.float32)
            phases = np.asarray(artifact["phase"], dtype=np.float32)
            if phases.shape[0] < query_steps.shape[0]:
                raise ValueError("successful artifact is missing a phase for a replay chunk")
            padded_actions = np.zeros(
                (query_steps.shape[0], EXECUTED_ACTION_HORIZON, 7), dtype=np.float32
            )
            action_counts = []
            for replan_index, start in enumerate(query_steps):
                end = (
                    int(query_steps[replan_index + 1])
                    if replan_index + 1 < query_steps.shape[0]
                    else int(record["total_steps"])
                )
                count = end - int(start)
                if not 0 < count <= EXECUTED_ACTION_HORIZON:
                    raise ValueError(f"invalid successful replay chunk length {count}")
                padded_actions[replan_index, :count] = executed_actions[int(start) : end]
                action_counts.append(count)
            success_trace_phase = phases[: query_steps.shape[0]].copy()
            success_trace_action = padded_actions
            success_trace_action_count = np.asarray(action_counts, dtype=np.int64)
            success_trace_attempt_id = int(record["attempt_id"])
    n = len(restored["phase"])
    request = {
        "pim_restore_session": True,
        "pim_session_id": session_id,
        "pim_task_cluster": task_cluster,
        "pim_environment_seed": environment_seed,
        "pim_last_attempt_id": int(records[-1]["attempt_id"]),
        "restore_phase": np.asarray(restored["phase"], dtype=np.float32).reshape(n, 128),
        "restore_visual_key": np.asarray(restored["visual_key"], dtype=np.float32).reshape(n, 128),
        "restore_effect_post": np.asarray(restored["effect_post"], dtype=np.float32).reshape(n, 128),
        "restore_effect_valid": np.asarray(restored["effect_valid"], dtype=np.bool_).reshape(n),
        "restore_executed_action": np.asarray(restored["executed_action"], dtype=np.float32).reshape(n, 16, 7),
        "restore_next_visual_key": np.asarray(restored["next_visual_key"], dtype=np.float32).reshape(n, 128),
        "restore_next_phase": np.asarray(restored["next_phase"], dtype=np.float32).reshape(n, 128),
        "restore_latent_index": np.asarray(restored["latent_index"], dtype=np.int64),
        "restore_attempt_ids": np.asarray(restored["attempt_ids"], dtype=np.int64),
        "restore_replan_indices": np.asarray(restored["replan_indices"], dtype=np.int64),
        "restore_outcomes": restored["outcomes"],
        "restore_termination_reasons": restored["termination_reasons"],
        "restore_total_steps": np.asarray(restored["total_steps"], dtype=np.int64),
        "restore_final_progress": np.asarray(restored["final_progress"], dtype=np.float32),
        "restore_success_trace_phase": success_trace_phase,
        "restore_success_trace_action": success_trace_action,
        "restore_success_trace_action_count": success_trace_action_count,
        "restore_success_trace_attempt_id": success_trace_attempt_id,
    }
    output = client.infer(request).get("pim_restore")
    if not isinstance(output, dict) or int(output.get("restored_transitions", -1)) != n:
        raise ValueError("policy server did not acknowledge the exact restored transition count")
    return output


def save_video(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
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
    parser.add_argument(
        "--attempt-start",
        type=int,
        default=0,
        help="Resume a fixed-seed session at this attempt ID; existing episodes.jsonl is audited first.",
    )
    parser.add_argument("--seed", type=int, default=195)
    parser.add_argument("--max-steps", type=int, default=MAX_CONTROLS_PER_ATTEMPT)
    parser.add_argument(
        "--open-loop-steps",
        type=int,
        default=EXECUTED_ACTION_HORIZON,
        help="Controls executed from each 32-step policy chunk; 16 is the PIM contract.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=("fixed", "increment"),
        default="fixed",
        help="fixed repeats one environment seed across attempts; increment is an independent-seed batch.",
    )
    parser.add_argument(
        "--diffusion-seed",
        type=int,
        default=20260824,
        help="Base for the explicit attempt/replan inference-seed schedule used by paired conditions.",
    )
    parser.add_argument(
        "--expect-pim-conditioning",
        dest="expect_pim_conditioning",
        choices=("off", "on", "any"),
        default="off",
        help="Require the server to report the expected PIM conditioning state.",
    )
    parser.add_argument(
        "--pim-session-id",
        dest="pim_session_id",
        default=None,
        help="Session key shared across repeated attempts of the same task; defaults to the task name.",
    )
    parser.add_argument(
        "--task-cluster",
        dest="task_cluster",
        default=None,
        help="Atomic-5 task cluster key; defaults to --task.",
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
    if args.open_loop_steps != EXECUTED_ACTION_HORIZON:
        raise ValueError(
            f"Zeva requires --open-loop-steps={EXECUTED_ACTION_HORIZON}; "
            "all PIM comparisons must use the same controller horizon"
        )
    if args.max_steps != MAX_CONTROLS_PER_ATTEMPT:
        raise ValueError(f"Zeva requires --max-steps={MAX_CONTROLS_PER_ATTEMPT}")
    if args.attempt_start < 0:
        raise ValueError("--attempt-start must be non-negative")
    if args.attempt_start and args.seed_mode != "fixed":
        raise ValueError("--attempt-start is supported only for fixed-seed repeated attempts")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = _LongInferenceWebsocketClientPolicy(args.host, args.port)
    results = []
    task_cluster = args.task_cluster or args.task
    base_session_id = args.pim_session_id or default_session_id(task_cluster, args.seed)

    episodes_path = args.output_dir / "episodes.jsonl"
    if args.attempt_start:
        if not episodes_path.is_file():
            raise ValueError("resume requires an existing episodes.jsonl")
        for line in episodes_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                validate_attempt_record(record)
                results.append(record)
        expected_ids = list(range(args.attempt_start))
        actual_ids = [int(item["attempt_id"]) for item in results]
        if actual_ids != expected_ids:
            raise ValueError(f"resume attempt history mismatch: expected={expected_ids}, actual={actual_ids}")
        if any(
            item["task"] != args.task
            or int(item["environment_seed"]) != args.seed
            or item["session_id"] != base_session_id
            for item in results
        ):
            raise ValueError("resume history does not match task/seed/session")
        restore_info = restore_session_from_artifacts(
            client,
            args.output_dir,
            results,
            session_id=base_session_id,
            task_cluster=task_cluster,
            environment_seed=args.seed,
        )
        print("RESTORE " + json.dumps(restore_info), flush=True)

    # The first request after server startup may initialize/compile kernels.
    # It is a separate, discarded session so this one-time effect cannot
    # contaminate attempt 0 or enter its memory.
    for warmup_index in range(REQUIRED_WARMUP_REQUESTS if args.attempt_start == 0 else 0):
        warmup_env = gym.make(
            f"robocasa/{args.task}",
            split=args.split,
            seed=args.seed,
            camera_widths=256,
            camera_heights=256,
            enable_render=True,
            render_gpu_device_id=args.render_gpu_device_id,
            robots="PandaOmron",
            randomize_cameras=False,
            translucent_robot=False,
        )
        warmup_reset = warmup_env.reset()
        warmup_obs = warmup_reset[0] if isinstance(warmup_reset, tuple) else warmup_reset
        warmup_policy_obs, _ = prepare_policy_observation(warmup_obs, TASK_PROMPTS[args.task])
        warmup_policy_obs["cte_boundary_images"] = np.expand_dims(
            warmup_policy_obs["observation/image"].copy(), axis=0
        )
        warmup_policy_obs["cte_transition_actions"] = np.empty((0, 4, 7), dtype=np.float32)
        warmup_policy_obs["cte_reset"] = True
        warmup_policy_obs["pim_session_id"] = f"{base_session_id}:discarded-warmup"
        warmup_policy_obs["pim_task_cluster"] = task_cluster
        warmup_policy_obs["pim_environment_seed"] = args.seed
        warmup_policy_obs["pim_attempt_id"] = 0
        warmup_policy_obs["pim_memory_reset"] = True
        warmup_policy_obs["pim_replan_reset"] = False
        warmup_policy_obs["pim_latent_index"] = 0
        warmup_policy_obs["pim_replan_index"] = 0
        warmup_policy_obs["pim_executed_action"] = np.empty((0, 7), dtype=np.float32)
        warmup_policy_obs["pim_transition_complete"] = False
        warmup_policy_obs["inference_seed"] = inference_seed(args.diffusion_seed + 1_000_000, 0, warmup_index)
        warmup_output = client.infer(warmup_policy_obs)
        warmup_action = np.asarray(warmup_output["action"], dtype=np.float32)
        if warmup_action.shape != (PREDICTED_ACTION_HORIZON, 7):
            raise ValueError(f"Warm-up returned invalid action shape {warmup_action.shape}")
        warmup_env.close()

    for episode_idx in range(args.attempt_start, args.attempt_start + args.episodes):
        episode_seed = args.seed if args.seed_mode == "fixed" else args.seed + episode_idx
        attempt_id = episode_idx if args.seed_mode == "fixed" else 0
        pim_session_id = (
            base_session_id
            if args.seed_mode == "fixed"
            else default_session_id(task_cluster, episode_seed)
        )
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
        # Match the official RoboCasa365 RLDX harness: the seed is supplied to
        # gym.make(), whose constructor performs an initial reset, and rollout
        # then calls reset() without rewinding the environment RNG a second time.
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        frames: list[np.ndarray] = []
        action_queue: deque[np.ndarray] = deque()
        # The CTE is defined at Wan's four-control cadence. Retain only
        # observations at raw offsets 0,4,8,... and the *executed* arm7
        # controls between them.  The policy server derives a first effect
        # only after four such transitions (16 executed controls).
        initial_policy_obs, _ = prepare_policy_observation(obs, TASK_PROMPTS[args.task])
        cte_boundary_images: list[np.ndarray] = [initial_policy_obs["observation/image"].copy()]
        cte_transitions: list[np.ndarray] = []
        pending_transition_actions: list[np.ndarray] = []
        actions_since_replan: list[np.ndarray] = []
        query_times = []
        query_steps: list[int] = []
        query_seeds: list[int] = []
        predicted_actions: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []
        phase_features: list[np.ndarray] = []
        visual_key_features: list[np.ndarray] = []
        effect_post_features: list[np.ndarray] = []
        effect_valid_values: list[bool] = []
        effect_history_valid_values: list[np.ndarray] = []
        predicate_progress_values: list[float] = []
        predicate_progress_components: list[dict] = []
        pim_queries = []
        observed_pim_conditioning: set[bool] = set()
        clipped_values = 0
        nonfinite_actions = 0
        success = False
        done = False
        final_info: dict = {}

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
            policy_obs["pim_session_id"] = pim_session_id
            policy_obs["pim_task_cluster"] = task_cluster
            policy_obs["pim_environment_seed"] = episode_seed
            policy_obs["pim_attempt_id"] = attempt_id
            policy_obs["pim_memory_reset"] = steps == 0 and (
                episode_idx == 0 or args.seed_mode == "increment"
            )
            policy_obs["pim_replan_reset"] = (
                steps == 0 and episode_idx > 0 and args.seed_mode == "fixed"
            )
            policy_obs["pim_latent_index"] = steps
            policy_obs["pim_replan_index"] = len(query_times)
            if not args.no_video:
                frames.append(left_frame.copy())
            if not action_queue:
                predicate_progress, predicate_components = task_predicate_progress(env, args.task)
                predicate_progress_values.append(predicate_progress)
                predicate_progress_components.append(predicate_components)
                if actions_since_replan:
                    policy_obs["pim_executed_action"] = np.asarray(actions_since_replan, dtype=np.float32)
                    policy_obs["pim_transition_complete"] = len(actions_since_replan) == 16
                else:
                    policy_obs["pim_executed_action"] = np.empty((0, 7), dtype=np.float32)
                    policy_obs["pim_transition_complete"] = False
                request_seed = inference_seed(args.diffusion_seed, attempt_id, len(query_times))
                policy_obs["inference_seed"] = request_seed
                start = time.monotonic()
                output = client.infer(policy_obs)
                query_times.append(time.monotonic() - start)
                query_steps.append(steps)
                query_seeds.append(request_seed)
                if isinstance(output.get("pim"), dict):
                    pim_queries.append({"backend": "pim", **dict(output["pim"])})
                elif isinstance(output.get("transition_memory"), dict):
                    pim_queries.append({"backend": "transition_memory", **dict(output["transition_memory"])})
                else:
                    pim_queries.append({})
                actions = np.asarray(output["action"], dtype=np.float32)
                if actions.shape != (PREDICTED_ACTION_HORIZON, 7):
                    raise ValueError(
                        f"Policy returned action shape {actions.shape}, expected "
                        f"[{PREDICTED_ACTION_HORIZON},7]"
                    )
                server_contract = output.get("zeva_contract")
                if not isinstance(server_contract, dict) or server_contract.get("version") != PROTOCOL_VERSION:
                    raise ValueError("Policy server did not acknowledge the Zeva protocol")
                if int(server_contract.get("inference_seed", -1)) != request_seed:
                    raise ValueError("Policy server did not use the requested paired inference seed")
                pim_conditioning = bool(server_contract.get("pim_conditioning", False))
                observed_pim_conditioning.add(pim_conditioning)
                expected_conditioning = {"off": False, "on": True}.get(args.expect_pim_conditioning)
                if expected_conditioning is not None and pim_conditioning != expected_conditioning:
                    raise ValueError(
                        "Policy server memory conditioning mismatch: "
                        f"expected={expected_conditioning}, actual={pim_conditioning}"
                    )
                nonfinite_actions += int((~np.isfinite(actions)).sum())
                clipped_values += int((np.abs(actions) > 1.0).sum())
                predicted_actions.append(actions.copy())
                features = output.get("cte_features")
                if not isinstance(features, dict):
                    raise ValueError("PIM evaluation requires CTE features from the policy server")
                phase_features.append(np.asarray(features["phase"], dtype=np.float32))
                visual_key_features.append(np.asarray(features["visual_key"], dtype=np.float32))
                effect_post_features.append(np.asarray(features["latest_effect_post"], dtype=np.float32))
                effect_valid_values.append(bool(features["latest_effect_valid"]))
                effect_history_valid_values.append(
                    np.asarray(features["effect_history_valid"], dtype=np.bool_)
                )
                action_queue.extend(actions[: args.open_loop_steps])
                actions_since_replan = []
            executed_action = action_queue.popleft()
            obs, success, done, final_info = env_step(env, executed_action)
            actions_since_replan.append(executed_action.astype(np.float32, copy=True))
            executed_actions.append(executed_action.astype(np.float32, copy=True))
            pending_transition_actions.append(executed_action.astype(np.float32, copy=True))
            if len(pending_transition_actions) == 4:
                boundary_policy_obs, _ = prepare_policy_observation(obs, TASK_PROMPTS[args.task])
                cte_boundary_images.append(boundary_policy_obs["observation/image"].copy())
                cte_transitions.append(np.stack(pending_transition_actions, axis=0))
                pending_transition_actions.clear()
            steps += 1

        terminal_outcome = "success" if success else "failure"
        termination_reason = "success" if success else ("env_done" if done else "max_steps")
        final_progress = 1.0 if success else 0.0
        terminal_predicate_progress, terminal_predicate_components = task_predicate_progress(env, args.task)
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
                "pim_session_id": pim_session_id,
                "pim_task_cluster": task_cluster,
                "pim_environment_seed": episode_seed,
                "pim_attempt_id": attempt_id,
                "pim_latent_index": steps,
                "pim_executed_action": np.asarray(actions_since_replan, dtype=np.float32),
                "pim_transition_complete": len(actions_since_replan) == EXECUTED_ACTION_HORIZON,
                "pim_terminal_outcome": terminal_outcome,
                "pim_termination_reason": termination_reason,
                "pim_total_steps": steps,
                "pim_final_progress": final_progress,
            }
        )
        finalize_output = client.infer(final_policy_obs)
        finalize_info = finalize_output.get("pim_finalize")
        if not isinstance(finalize_info, dict):
            raise ValueError("PIM server did not acknowledge attempt finalization")
        if int(finalize_info.get("attempt_id", -1)) != attempt_id:
            raise ValueError("PIM attempt finalization mismatch")

        chunk_records = []
        for replan_index, start_step in enumerate(query_steps):
            end_step = query_steps[replan_index + 1] if replan_index + 1 < len(query_steps) else steps
            executed_count = end_step - start_step
            chunk_records.append(
                {
                    "schema_version": CHUNK_RECORD_VERSION,
                    "replan_index": replan_index,
                    "start_step": start_step,
                    "end_step": end_step,
                    "executed_count": executed_count,
                    "completed_16": executed_count == EXECUTED_ACTION_HORIZON,
                    "inference_seed": query_seeds[replan_index],
                    "memory_entries_before_query": int(
                        pim_queries[replan_index].get(
                            "entries_total", pim_queries[replan_index].get("num_entries", 0)
                        )
                    ),
                    "predicate_progress_before": predicate_progress_values[replan_index],
                    "predicate_progress_after": (
                        predicate_progress_values[replan_index + 1]
                        if replan_index + 1 < len(predicate_progress_values)
                        else terminal_predicate_progress
                    ),
                    "terminal_chunk": replan_index == len(query_steps) - 1,
                }
            )
            chunk_records[-1]["predicate_progress_delta"] = (
                chunk_records[-1]["predicate_progress_after"]
                - chunk_records[-1]["predicate_progress_before"]
            )

        artifact_path = args.output_dir / f"attempt_{attempt_id:03d}_seed_{episode_seed}.npz"
        np.savez_compressed(
            artifact_path,
            cte_boundary_images=np.stack(cte_boundary_images, axis=0),
            cte_transition_actions=(
                np.stack(cte_transitions, axis=0)
                if cte_transitions
                else np.empty((0, 4, 7), dtype=np.float32)
            ),
            executed_actions=np.asarray(executed_actions, dtype=np.float32),
            predicted_actions=np.asarray(predicted_actions, dtype=np.float32),
            query_steps=np.asarray(query_steps, dtype=np.int32),
            inference_seeds=np.asarray(query_seeds, dtype=np.uint32),
            phase=_feature_array(phase_features, 128),
            visual_key=_feature_array(visual_key_features, 128),
            latest_effect_post=_feature_array(effect_post_features, 128),
            latest_effect_valid=np.asarray(effect_valid_values, dtype=np.bool_),
            effect_history_valid=np.asarray(effect_history_valid_values, dtype=np.bool_),
            memory_entries_before_query=np.asarray(
                [item["memory_entries_before_query"] for item in chunk_records], dtype=np.int32
            ),
            predicate_progress_before=np.asarray(predicate_progress_values, dtype=np.float32),
            predicate_progress_after=np.asarray(
                [item["predicate_progress_after"] for item in chunk_records], dtype=np.float32
            ),
        )
        env.close()
        record = {
            "schema_version": ATTEMPT_RECORD_VERSION,
            "session_id": pim_session_id,
            "task": args.task,
            "episode": episode_idx,
            "attempt_id": attempt_id,
            "environment_seed": episode_seed,
            "seed": episode_seed,
            "success": success,
            "terminal_outcome": terminal_outcome,
            "termination_reason": termination_reason,
            "total_steps": steps,
            "steps": steps,
            "final_progress": final_progress,
            "progress_source": "success_only",
            "predicate_progress_source": predicate_progress_components[0]["schema"],
            "terminal_predicate_progress": terminal_predicate_progress,
            "terminal_predicate_components": terminal_predicate_components,
            "predicate_progress_components": predicate_progress_components,
            "queries": len(query_times),
            "mean_query_seconds": float(np.mean(query_times)) if query_times else None,
            "pim_session_id": pim_session_id,
            "seed_mode": args.seed_mode,
            "diffusion_seed_base": args.diffusion_seed,
            "protocol": contract_manifest(),
            "server_pim_conditioning": sorted(observed_pim_conditioning),
            "pim_queries": pim_queries,
            "pim_finalize": finalize_info,
            "chunk_records": chunk_records,
            "artifact_npz": artifact_path.name,
            "artifact_sha256": sha256_file(artifact_path),
            "final_env_info_keys": sorted(final_info),
            "action_values_clipped": clipped_values,
            "action_nonfinite_values": nonfinite_actions,
        }
        validate_attempt_record(record)
        results.append(record)
        with episodes_path.open("a") as file:
            file.write(json.dumps(record) + "\n")
        if not args.no_video:
            save_video(
                frames,
                args.output_dir / f"ep_{episode_idx:03d}_seed_{episode_seed}_{'success' if success else 'fail'}.mp4",
            )
        print(json.dumps(record), flush=True)

    summary = {
        "schema_version": ATTEMPT_RECORD_VERSION,
        "task": args.task,
        "episodes": len(results),
        "successes": sum(item["success"] for item in results),
        "success_rate": float(np.mean([item["success"] for item in results])),
        "mean_steps": float(np.mean([item["steps"] for item in results])),
        "seed_mode": args.seed_mode,
        "environment_seeds": sorted({item["seed"] for item in results}),
        "diffusion_seed_base": args.diffusion_seed,
        "protocol": contract_manifest(),
        "expected_pim_conditioning": args.expect_pim_conditioning,
        "observed_pim_conditioning": sorted(
            {value for item in results for value in item["server_pim_conditioning"]}
        ),
        "mean_query_seconds": float(
            np.mean([item["mean_query_seconds"] for item in results if item["mean_query_seconds"] is not None])
        ),
    }
    write_manifest(args.output_dir / "attempts_manifest.json", results)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
