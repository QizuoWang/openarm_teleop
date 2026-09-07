#!/usr/bin/env python3
"""Guarded local ACT preview and deployment controller for pedestal OpenArm."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client
from pathlib import Path
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
import openarm_driver

from dora_hikrobot_rgb_camera.main import HikrobotCamera

from representation_adapter import resolve_checkpoint


ROOT = Path(__file__).resolve().parents[2]
LOCAL_LEROBOT_PYTHON = Path(os.environ.get("OPENARM_POLICY_PYTHON", str(ROOT.parent / ".venv/bin/python")))
ROBOT_CONFIG = Path(os.environ.get("OPENARM_ROBOT_CONFIG", str(ROOT / "openarm_pedestal_vr.yaml")))
CHECKPOINT = (
    ROOT
    / "deployments/act/act-tshirt-fold-rad-v1-step100000-lerobot062-local"
)
SOCKET = Path("/tmp/openarm-local-act-policy.socket")
AUTHKEY = b"openarm-local-act-v1"
IFF_UP = 0x1
CAMERA_SERIALS = {
    "observation.images.wrist_right": os.environ.get("OPENARM_CAMERA_RIGHT_SERIAL", "00DA8573792"),
    "observation.images.wrist_left": os.environ.get("OPENARM_CAMERA_LEFT_SERIAL", "00DB1265540"),
    "observation.images.ceiling": os.environ.get("OPENARM_CAMERA_BASE_SERIAL", "00DA8573794"),
}
JOINT_NAMES = (
    *(f"right_joint_{index}" for index in range(1, 8)),
    "right_gripper",
    *(f"left_joint_{index}" for index in range(1, 8)),
    "left_gripper",
)


def start_policy(checkpoint: Path, num_actions: int, deployment_manifest: Path | None = None):
    adapter = resolve_checkpoint(checkpoint, deployment_manifest)
    if not 1 <= num_actions <= adapter.chunk_size:
        raise ValueError(f"--num-actions must be in [1,{adapter.chunk_size}]")
    # Never attach to a still-running server for a different checkpoint.
    socket_path = Path("/tmp") / f"openarm-policy-{uuid.uuid4().hex}.socket"
    command = [
        adapter.policy_python or str(LOCAL_LEROBOT_PYTHON),
        str(ROOT / "deployments/local_act/policy_server.py"),
        "--checkpoint",
        str(checkpoint),
        "--socket",
        str(socket_path),
        "--num-actions",
        str(num_actions),
    ]
    if deployment_manifest is not None:
        command.extend(["--deployment-manifest", str(deployment_manifest.resolve())])
    process = subprocess.Popen(command, cwd=ROOT)
    startup_timeout = 180.0 if adapter.policy_type == "smolvla" else 90.0
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"policy server exited with code {process.returncode}")
        try:
            return process, Client(str(socket_path), family="AF_UNIX", authkey=AUTHKEY)
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.1)
    process.terminate()
    raise TimeoutError(f"policy server did not become ready in {startup_timeout:.0f} seconds")


def require_can_interfaces_up(config) -> None:
    failures = []
    for arm_side in ("right_arm", "left_arm"):
        interface = config.get_can_interface(arm_side)
        flags_path = Path("/sys/class/net") / interface / "flags"
        try:
            flags = int(flags_path.read_text(encoding="ascii").strip(), 16)
        except (FileNotFoundError, OSError, ValueError) as exc:
            failures.append(f"{interface} unavailable ({exc})")
            continue
        if not flags & IFF_UP:
            failures.append(f"{interface} is DOWN")
    if failures:
        detail = "; ".join(failures)
        raise RuntimeError(
            f"CAN preflight failed: {detail}. Configure both buses before deployment: "
            "sudo openarm-can-cli -i can0 can_configure && "
            "sudo openarm-can-cli -i can1 can_configure"
        )


def open_cameras():
    return {
        key: HikrobotCamera(
            device_index=0,
            serial_number=serial,
            working_mode=2,
            image_mode=8,
        )
        for key, serial in CAMERA_SERIALS.items()
    }


def capture_images(cameras, pool: ThreadPoolExecutor):
    futures = {key: pool.submit(camera.fetch_bgr, 3000) for key, camera in cameras.items()}
    images = {}
    for key, future in futures.items():
        bgr = future.result()
        if bgr is None:
            raise RuntimeError(f"camera returned no frame: {key}")
        if bgr.shape[:2] != (360, 640):
            bgr = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
        images[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return images


def request_actions(connection, state, images):
    connection.send({"state": state, "images": images})
    response = connection.recv()
    if not response.get("ok"):
        raise RuntimeError(f"policy inference failed: {response.get('error')}")
    return np.asarray(response["actions"], dtype=np.float32), float(response["latency_ms"])


def check_chunk(
    actions,
    current,
    config,
    first_arm_delta_limit: float,
    *,
    enforce_first_delta: bool = True,
):
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
        raise RuntimeError(f"invalid policy chunk: shape={actions.shape}")
    arm_axes = [*range(7), *range(8, 15)]
    limits = np.concatenate(
        [config.get_joint_limits("right_arm"), config.get_joint_limits("left_arm")],
        axis=0,
    )
    below = limits[:, 0] - actions
    above = actions - limits[:, 1]
    violation = np.maximum(below, above)
    arm_violation = violation[:, arm_axes]
    maximum_violation = float(max(np.max(arm_violation), 0.0))
    if maximum_violation > 0.02:
        step, local_axis = np.unravel_index(
            np.argmax(arm_violation), arm_violation.shape
        )
        axis = arm_axes[local_axis]
        raise RuntimeError(
            "policy arm target exceeds pedestal joint limits: "
            f"step={step} axis={JOINT_NAMES[axis]} target={actions[step, axis]:.4f} "
            f"limits=[{limits[axis, 0]:.4f},{limits[axis, 1]:.4f}] "
            f"violation={maximum_violation:.4f} rad"
        )

    # Legacy demonstrations record upstream logical gripper commands: right uses
    # [-1, 0] and left uses [0, 1]. The POS_FORCE driver then saturates those
    # commands to the smaller measured hardware travel in the pedestal limits.
    # Reproduce that collection boundary explicitly while keeping strict limit
    # rejection for all fourteen arm joints. Degree-model outputs have already
    # been converted to physical radian gripper targets by the policy adapter;
    # these lie within the same sign domains and need no additional scaling.
    actions = actions.copy()
    actions[:, 7] = np.clip(actions[:, 7], -1.0, 0.0)
    actions[:, 15] = np.clip(actions[:, 15], 0.0, 1.0)
    actions = np.clip(actions, limits[:, 0], limits[:, 1])
    first_axis_delta = np.abs(actions[0] - current)
    first_delta = float(np.max(first_axis_delta[arm_axes]))
    if enforce_first_delta and first_delta > first_arm_delta_limit:
        largest_axes = sorted(
            arm_axes, key=lambda axis: first_axis_delta[axis], reverse=True
        )[:3]
        details = ", ".join(
            f"{JOINT_NAMES[axis]} measured={current[axis]:.3f} "
            f"target={actions[0, axis]:.3f} delta={first_axis_delta[axis]:.3f}"
            for axis in largest_axes
        )
        raise RuntimeError(
            f"first policy target is {first_delta:.3f} rad from measured state; "
            f"limit is {first_arm_delta_limit:.3f} rad; largest deltas: {details}"
        )
    return actions, first_delta


def fetch_bimanual_state(right, left, pool: ThreadPoolExecutor):
    right_future = pool.submit(right.fetch_state, True)
    left_future = pool.submit(left.fetch_state, True)
    right_state = right_future.result()
    left_state = left_future.result()
    return np.concatenate([right_state["qpos"], left_state["qpos"]]).astype(np.float32)


def send_bimanual(right, left, target, pool: ThreadPoolExecutor):
    right_future = pool.submit(right.send_position, target[:8])
    left_future = pool.submit(left.send_position, target[8:])
    right_future.result()
    left_future.result()


def align_to_first_target(right, left, current, target, pool, hz=50.0):
    arm_axes = [*range(7), *range(8, 15)]
    duration = max(2.0, float(np.max(np.abs(target[arm_axes] - current[arm_axes]))) / 0.15)
    steps = max(2, int(duration * hz))
    print(f"[motion] aligning to first target over {duration:.1f}s", flush=True)
    for alpha in np.linspace(0.0, 1.0, steps):
        send_bimanual(right, left, current + alpha * (target - current), pool)
        time.sleep(1.0 / hz)


def immediately_disable(arm) -> None:
    try:
        arm.openarm.disable_all()
        arm.started = False
    except Exception as exc:
        print(f"[warning] immediate disable failed: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "run"))
    parser.add_argument("--confirm-hardware", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--deployment-manifest", type=Path)
    parser.add_argument("--num-actions", type=int, default=85)
    parser.add_argument("--max-policy-steps", type=int, default=1800)
    parser.add_argument("--first-arm-delta-limit", type=float, default=0.5)
    args = parser.parse_args()
    if not args.confirm_hardware:
        parser.error("hardware access requires --confirm-hardware")
    if args.mode == "run" and not sys.stdin.isatty():
        parser.error("run mode requires an interactive terminal")

    config = openarm_driver.Config(ROBOT_CONFIG)
    require_can_interfaces_up(config)
    policy_process = None
    connection = None
    cameras = {}
    right = left = None
    motion_authorized = False
    with ThreadPoolExecutor(max_workers=8) as pool:
        try:
            policy_process, connection = start_policy(args.checkpoint, args.num_actions, args.deployment_manifest)
            cameras = open_cameras()
            right = openarm_driver.SingleArmDriver("right_arm", config)
            left = openarm_driver.SingleArmDriver("left_arm", config)
            current = fetch_bimanual_state(right, left, pool)
            images = capture_images(cameras, pool)
            actions, latency_ms = request_actions(connection, current, images)
            actions, first_delta = check_chunk(
                actions,
                current,
                config,
                args.first_arm_delta_limit,
                enforce_first_delta=False,
            )
            print(
                f"[preview] chunk={actions.shape} inference={latency_ms:.1f}ms "
                f"range=[{actions.min():.3f},{actions.max():.3f}] "
                f"first_arm_delta={first_delta:.3f}rad",
                flush=True,
            )
            if first_delta > args.first_arm_delta_limit:
                print(
                    "[preview] NOTE: the current unpowered pose is outside the "
                    "motion-start delta guard; run mode will check again after "
                    "moving to the configured pedestal start pose",
                    flush=True,
                )
            if args.mode == "preview":
                print("[preview] PASS: motors were never enabled", flush=True)
                return

            answer = input(
                "Clear the workspace, keep your hand beside the emergency stop "
                "(do not press it unless aborting), then type MOVE to enable motion: "
            )
            if answer.strip() != "MOVE":
                print("[motion] cancelled; motors were never enabled", flush=True)
                return

            motion_authorized = True
            right.start()
            left.start()
            current = fetch_bimanual_state(right, left, pool)
            start_config = config.get_start_config()
            if start_config.get("moves"):
                final_start = start_config["moves"][-1]["position"]
                expected = np.concatenate(
                    [final_start["right_arm"], final_start["left_arm"]]
                ).astype(np.float32)
                arm_axes = [*range(7), *range(8, 15)]
                start_error = float(
                    np.max(np.abs(current[arm_axes] - expected[arm_axes]))
                )
                if start_error > 0.2:
                    raise RuntimeError(
                        "startup motion did not reach the configured pose: "
                        f"maximum arm-joint error={start_error:.3f} rad"
                    )
            images = capture_images(cameras, pool)
            actions, latency_ms = request_actions(connection, current, images)
            actions, _ = check_chunk(actions, current, config, args.first_arm_delta_limit)
            align_to_first_target(right, left, current, actions[0], pool)

            period = 1.0 / 30.0
            executed = 0
            previous = actions[0].copy()
            while executed < args.max_policy_steps:
                current = fetch_bimanual_state(right, left, pool)
                images = capture_images(cameras, pool)
                actions, latency_ms = request_actions(connection, current, images)
                actions, _ = check_chunk(actions, current, config, args.first_arm_delta_limit)
                print(
                    f"[motion] step={executed} inference={latency_ms:.1f}ms",
                    flush=True,
                )
                velocity = np.tile(
                    np.asarray(config.get_joint_velocity_limits(), dtype=np.float32), 2
                )
                deadline = time.monotonic()
                for target in actions:
                    deadline += period
                    limited = previous + np.clip(
                        target - previous, -velocity * period, velocity * period
                    )
                    send_bimanual(right, left, limited, pool)
                    previous = limited
                    executed += 1
                    if executed >= args.max_policy_steps:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            print("\n[motion] interrupted; disabling torque immediately", flush=True)
        finally:
            if motion_authorized and right is not None:
                immediately_disable(right)
            if motion_authorized and left is not None:
                immediately_disable(left)
            for camera in cameras.values():
                camera.close()
            if connection is not None:
                try:
                    connection.send({"command": "stop"})
                except (BrokenPipeError, EOFError):
                    pass
                connection.close()
            if policy_process is not None:
                try:
                    policy_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    policy_process.terminate()


if __name__ == "__main__":
    main()
