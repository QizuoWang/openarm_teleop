#!/usr/bin/env python3
"""Local ACT real-robot evaluation UI and raw rollout recorder."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import openarm_driver
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from controller import (
    CHECKPOINT,
    JOINT_NAMES,
    ROOT,
    ROBOT_CONFIG,
    check_chunk,
    fetch_bimanual_state,
    immediately_disable,
    open_cameras,
    request_actions,
    require_can_interfaces_up,
    send_bimanual,
    start_policy,
)
from representation_adapter import resolve_checkpoint


CAMERA_KEYS = {
    "wrist_right": "observation.images.wrist_right",
    "wrist_left": "observation.images.wrist_left",
    "ceiling": "observation.images.ceiling",
}
FAILURE_REASONS = (
    "grasp_failure",
    "incomplete_fold",
    "unsafe_motion",
    "hardware_or_sensor",
    "bad_initial_setup",
    "timeout",
    "other",
)


def atomic_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    temporary.replace(path)


def configured_start_position(config) -> np.ndarray:
    moves = config.get_start_config().get("moves", [])
    if not moves:
        raise RuntimeError("pedestal configuration has no startup pose")
    position = moves[-1]["position"]
    return np.asarray(position["right_arm"] + position["left_arm"], dtype=np.float32)


def require_start_position(current: np.ndarray, expected: np.ndarray) -> None:
    arm_axes = [*range(7), *range(8, 15)]
    error = np.abs(current - expected)
    worst_axis = max(arm_axes, key=lambda axis: error[axis])
    if error[worst_axis] > 0.2:
        raise RuntimeError(
            "startup motion did not reach the configured pose: "
            f"{JOINT_NAMES[worst_axis]} measured={current[worst_axis]:.3f} "
            f"expected={expected[worst_axis]:.3f} "
            f"error={error[worst_axis]:.3f} rad"
        )


def save_rejection_diagnostic(
    dataset_root: Path,
    current: np.ndarray,
    actions: np.ndarray,
    frames: dict[str, CameraFrame],
    error: Exception,
) -> Path:
    timestamp_ns = time.time_ns()
    output = dataset_root.parent / "diagnostics" / str(timestamp_ns)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "measured_state.npy", current)
    np.save(output / "policy_actions.npy", actions)
    for name, feature_key in CAMERA_KEYS.items():
        (output / f"{name}.jpeg").write_bytes(frames[feature_key].jpeg)
    first_delta = np.abs(actions[0] - current)
    atomic_yaml(
        output / "diagnostic.yaml",
        {
            "timestamp_ns": timestamp_ns,
            "error": str(error),
            "measured_state": {
                name: float(value) for name, value in zip(JOINT_NAMES, current)
            },
            "first_policy_target": {
                name: float(value) for name, value in zip(JOINT_NAMES, actions[0])
            },
            "first_target_delta": {
                name: float(value) for name, value in zip(JOINT_NAMES, first_delta)
            },
            "camera_timestamps_ns": {
                name: frames[feature_key].timestamp_ns
                for name, feature_key in CAMERA_KEYS.items()
            },
        },
    )
    return output


@dataclass(frozen=True)
class CameraFrame:
    timestamp_ns: int
    rgb: np.ndarray
    jpeg: bytes


class CameraHub:
    def __init__(self):
        self.cameras = open_cameras()
        self.frames: dict[str, CameraFrame] = {}
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for feature_key, camera in self.cameras.items():
            thread = threading.Thread(
                target=self._capture_loop,
                args=(feature_key, camera),
                name=f"camera-{feature_key.rsplit('.', 1)[-1]}",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)
        self.snapshot(timeout_s=10.0)

    def _capture_loop(self, feature_key, camera) -> None:
        while not self.stop_event.is_set():
            bgr = camera.fetch_bgr(3000)
            if bgr is None:
                continue
            if bgr.shape[:2] != (360, 640):
                bgr = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                continue
            frame = CameraFrame(
                timestamp_ns=time.time_ns(),
                rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                jpeg=encoded.tobytes(),
            )
            with self.condition:
                self.frames[feature_key] = frame
                self.condition.notify_all()

    def snapshot(self, timeout_s: float = 2.0) -> dict[str, CameraFrame]:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while len(self.frames) < len(CAMERA_KEYS):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = sorted(set(CAMERA_KEYS.values()) - set(self.frames))
                    raise RuntimeError(f"camera frames unavailable: {missing}")
                self.condition.wait(remaining)
            return dict(self.frames)

    def close(self) -> None:
        self.stop_event.set()
        for thread in self.threads:
            thread.join(timeout=4.0)
        for camera in self.cameras.values():
            camera.close()


class EpisodeBuffer:
    def __init__(self, pending_root: Path, episode_id: int, session: dict):
        self.episode_id = episode_id
        self.session = session
        self.started_at_ns = time.time_ns()
        self.ended_at_ns = self.started_at_ns
        self.root = pending_root / f"episode-{episode_id}-{self.started_at_ns}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.observations = {"right": [], "left": []}
        self.actions = {"right": [], "left": []}
        self.camera_timestamps = {name: set() for name in CAMERA_KEYS}
        for name in CAMERA_KEYS:
            (self.root / "cameras" / name).mkdir(parents=True, exist_ok=True)

    def append(self, timestamp_ns, right_state, left_state, action, frames) -> None:
        self.ended_at_ns = timestamp_ns
        for side, state in (("right", right_state), ("left", left_state)):
            self.observations[side].append(
                {
                    "timestamp": timestamp_ns,
                    "qpos": np.asarray(state["qpos"], dtype=np.float32).tolist(),
                    "qvel": np.asarray(state["qvel"], dtype=np.float32).tolist(),
                    "qtorque": np.asarray(state["qtorque"], dtype=np.float32).tolist(),
                }
            )
        self.actions["right"].append(
            {"timestamp": timestamp_ns, "qpos": action[:8].astype(np.float32).tolist()}
        )
        self.actions["left"].append(
            {"timestamp": timestamp_ns, "qpos": action[8:].astype(np.float32).tolist()}
        )
        for name, feature_key in CAMERA_KEYS.items():
            frame = frames[feature_key]
            path = self.root / "cameras" / name / f"{frame.timestamp_ns}.jpeg"
            path.write_bytes(frames[feature_key].jpeg)
            self.camera_timestamps[name].add(frame.timestamp_ns)

    @staticmethod
    def _write_actions(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "timestamp": pa.array(
                    [row["timestamp"] for row in rows], type=pa.timestamp("ns")
                ),
                "qpos": pa.array(
                    [row["qpos"] for row in rows], type=pa.list_(pa.float32())
                ),
            }
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, temporary)
        temporary.replace(path)

    @staticmethod
    def _write_observations(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "timestamp": pa.array(
                    [row["timestamp"] for row in rows], type=pa.timestamp("ns")
                ),
                "qpos": pa.array(
                    [row["qpos"] for row in rows], type=pa.list_(pa.float32())
                ),
                "qvel": pa.array(
                    [row["qvel"] for row in rows], type=pa.list_(pa.float32())
                ),
                "qtorque": pa.array(
                    [row["qtorque"] for row in rows], type=pa.list_(pa.float32())
                ),
            }
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(table, temporary)
        temporary.replace(path)

    def _manifest(
        self,
        *,
        status: str,
        success: bool | None,
        failure_reason: str | None,
        interruption_reason: str | None = None,
    ) -> dict:
        stream_counts = {
            "arm_right_action": len(self.actions["right"]),
            "arm_left_action": len(self.actions["left"]),
            "arm_right_observation": len(self.observations["right"]),
            "arm_left_observation": len(self.observations["left"]),
            **{
                f"camera_{name}": len(timestamps)
                for name, timestamps in sorted(self.camera_timestamps.items())
            },
        }
        manifest = {
            "id": str(self.episode_id),
            "status": status,
            "success": success,
            "failure_reason": failure_reason,
            "task_index": 0,
            "started_at_ns": self.started_at_ns,
            "ended_at_ns": self.ended_at_ns,
            "session": self.session,
            "stream_counts": stream_counts,
            "qc_errors": [],
        }
        if interruption_reason is not None:
            manifest["interruption_reason"] = interruption_reason
        return manifest

    def checkpoint(
        self,
        *,
        status: str,
        interruption_reason: str | None = None,
    ) -> dict:
        """Persist an in-progress episode without accepting or rejecting it."""
        if not self.actions["right"]:
            raise RuntimeError("cannot checkpoint an evaluation episode with zero policy steps")
        for side in ("right", "left"):
            self._write_actions(
                self.root / "action" / "arms" / side / "state.parquet",
                self.actions[side],
            )
            self._write_observations(
                self.root / "obs" / "arms" / side / "state.parquet",
                self.observations[side],
            )
        manifest = self._manifest(
            status=status,
            success=None,
            failure_reason=None,
            interruption_reason=interruption_reason,
        )
        atomic_yaml(self.root / "episode.yaml", manifest)
        return manifest

    def commit(self, dataset_root: Path, success: bool, failure_reason: str | None) -> Path:
        if not self.actions["right"]:
            raise RuntimeError("cannot commit an evaluation episode with zero policy steps")
        for side in ("right", "left"):
            self._write_actions(
                self.root / "action" / "arms" / side / "state.parquet",
                self.actions[side],
            )
            self._write_observations(
                self.root / "obs" / "arms" / side / "state.parquet",
                self.observations[side],
            )
        manifest = self._manifest(
            status="accepted" if success else "failed",
            success=success,
            failure_reason=failure_reason,
        )
        atomic_yaml(self.root / "episode.yaml", manifest)
        destination = dataset_root / "episodes" / str(self.episode_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"episode destination already exists: {destination}")
        self.root.replace(destination)
        metadata_path = dataset_root / "metadata.yaml"
        metadata = {}
        if metadata_path.exists():
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        episodes = [
            item for item in metadata.get("episodes", []) if str(item.get("id")) != str(self.episode_id)
        ]
        episodes.append(manifest)
        metadata["episodes"] = sorted(episodes, key=lambda item: int(item["id"]))
        atomic_yaml(metadata_path, metadata)
        return destination


class EvaluationRuntime:
    def __init__(
        self,
        *,
        config,
        right,
        left,
        pool,
        camera_hub,
        policy_connection,
        dataset_root,
        session,
        num_actions,
        max_policy_steps,
        first_arm_delta_limit,
    ):
        self.config = config
        self.right = right
        self.left = left
        self.pool = pool
        self.camera_hub = camera_hub
        self.policy_connection = policy_connection
        self.dataset_root = dataset_root
        self.session = session
        self.num_actions = num_actions
        self.max_policy_steps = max_policy_steps
        self.first_arm_delta_limit = first_arm_delta_limit
        self.lock = threading.Lock()
        self.phase = "ready"
        self.message = "Robot is at the configured start pose. Flatten the T-shirt, then start."
        self.executed_steps = 0
        self.current_episode = None
        self.pending_episode: EpisodeBuffer | None = None
        self.active_episode: EpisodeBuffer | None = None
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.shutdown_requested = False
        self.last_error: str | None = None
        self.last_diagnostic: str | None = None

    def status(self) -> dict:
        with self.lock:
            return {
                "phase": self.phase,
                "message": self.message,
                "executed_steps": self.executed_steps,
                "current_episode": self.current_episode,
                "pending_label": self.pending_episode is not None,
                "dataset_root": str(self.dataset_root),
                "session_id": self.session["collection_session_id"],
                "policy_type": self.session.get("policy_type", "act"),
                "policy_checkpoint": self.session["policy_checkpoint"],
                "policy_task": self.session.get("policy_task"),
                "last_error": self.last_error,
                "last_diagnostic": self.last_diagnostic,
            }

    def _next_episode_id(self) -> int:
        episodes_root = self.dataset_root / "episodes"
        ids = [
            int(path.name)
            for path in episodes_root.glob("*")
            if path.is_dir() and path.name.isdigit()
        ]
        return max(ids, default=-1) + 1

    def start_rollout(self) -> None:
        with self.lock:
            if self.phase != "ready" or self.pending_episode is not None:
                raise RuntimeError(f"cannot start rollout while phase is {self.phase}")
            episode_id = self._next_episode_id()
            self.phase = "recording"
            self.message = "Policy is running and the evaluation episode is recording."
            self.executed_steps = 0
            self.current_episode = episode_id
            self.last_error = None
            self.last_diagnostic = None
            self.shutdown_requested = False
            self.stop_event.clear()
            self.worker = threading.Thread(
                target=self._rollout_loop,
                args=(episode_id,),
                name=f"evaluation-episode-{episode_id}",
                daemon=True,
            )
            self.worker.start()

    def _fetch_states(self):
        right_future = self.pool.submit(self.right.fetch_state, True)
        left_future = self.pool.submit(self.left.fetch_state, True)
        return right_future.result(), left_future.result()

    def _rollout_loop(self, episode_id: int) -> None:
        pending_root = self.dataset_root / ".pending"
        pending_root.mkdir(parents=True, exist_ok=True)
        episode = EpisodeBuffer(pending_root, episode_id, self.session)
        with self.lock:
            self.active_episode = episode
        period = 1.0 / 30.0
        previous = None
        try:
            while self.executed_steps < self.max_policy_steps and not self.stop_event.is_set():
                right_state, left_state = self._fetch_states()
                current = np.concatenate(
                    [right_state["qpos"], left_state["qpos"]]
                ).astype(np.float32)
                frames = self.camera_hub.snapshot()
                images = {
                    feature_key: frame.rgb for feature_key, frame in frames.items()
                }
                actions, latency_ms = request_actions(
                    self.policy_connection, current, images
                )
                try:
                    actions, _ = check_chunk(
                        actions,
                        current,
                        self.config,
                        self.first_arm_delta_limit,
                    )
                except RuntimeError as exc:
                    diagnostic = save_rejection_diagnostic(
                        self.dataset_root, current, actions, frames, exc
                    )
                    with self.lock:
                        self.last_error = str(exc)
                        self.last_diagnostic = str(diagnostic)
                    raise RuntimeError(f"{exc}; diagnostic saved at {diagnostic}") from exc
                if previous is None:
                    previous = current.copy()
                velocity = np.tile(
                    np.asarray(self.config.get_joint_velocity_limits(), dtype=np.float32),
                    2,
                )
                deadline = time.monotonic()
                for target in actions:
                    if self.stop_event.is_set() or self.executed_steps >= self.max_policy_steps:
                        break
                    deadline += period
                    limited = previous + np.clip(
                        target - previous, -velocity * period, velocity * period
                    )
                    send_bimanual(self.right, self.left, limited, self.pool)
                    right_state, left_state = self._fetch_states()
                    frames = self.camera_hub.snapshot()
                    timestamp_ns = time.time_ns()
                    episode.append(
                        timestamp_ns,
                        right_state,
                        left_state,
                        limited,
                        frames,
                    )
                    previous = limited
                    with self.lock:
                        self.executed_steps += 1
                        self.message = (
                            f"Recording episode {episode_id}: step {self.executed_steps}, "
                            f"last inference {latency_ms:.1f} ms"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                if episode.actions["right"]:
                    episode.checkpoint(status="recording")
            if episode.actions["right"]:
                with self.lock:
                    interrupted = self.shutdown_requested
                episode.checkpoint(
                    status="interrupted" if interrupted else "awaiting_label",
                    interruption_reason=("evaluation_server_shutdown" if interrupted else None),
                )
            with self.lock:
                self.active_episode = None
                if episode.actions["right"]:
                    self.pending_episode = episode
                    if interrupted:
                        self.message = f"Interrupted rollout saved at {episode.root}."
                    else:
                        self.phase = "awaiting_label"
                        self.message = (
                            "Rollout stopped and checkpointed. "
                            "Mark it Success or Failure before resetting."
                        )
                else:
                    shutil.rmtree(episode.root, ignore_errors=True)
                    self.phase = "ready"
                    self.message = "Rollout stopped before any policy step was recorded."
                    self.current_episode = None
        except Exception as exc:
            print(f"[evaluation-error] {exc}", flush=True)
            if episode.actions["right"]:
                try:
                    episode.checkpoint(
                        status="interrupted",
                        interruption_reason=f"runtime_error: {exc}",
                    )
                except Exception as checkpoint_exc:
                    print(f"[evaluation-save-error] {checkpoint_exc}", flush=True)
            with self.lock:
                self.active_episode = None
                self.last_error = str(exc)
                if episode.actions["right"]:
                    self.pending_episode = episode
                    self.phase = "awaiting_label"
                    self.message = f"Policy stopped with error: {exc}. Label the partial rollout."
                else:
                    shutil.rmtree(episode.root, ignore_errors=True)
                    self.phase = "ready"
                    self.current_episode = None
                    self.message = f"Policy did not move: {exc}"

    def stop_rollout(self) -> None:
        with self.lock:
            if self.phase != "recording":
                raise RuntimeError("no rollout is currently recording")
            self.stop_event.set()
            self.message = "Stopping after the current command..."

    def label(self, success: bool, failure_reason: str | None) -> Path:
        with self.lock:
            if self.phase != "awaiting_label" or self.pending_episode is None:
                raise RuntimeError("there is no completed rollout awaiting a label")
            if not success and failure_reason not in FAILURE_REASONS:
                raise RuntimeError("select a valid failure reason")
            episode = self.pending_episode
            destination = episode.commit(
                self.dataset_root,
                success=success,
                failure_reason=None if success else failure_reason,
            )
            self.pending_episode = None
            self.phase = "needs_reset"
            self.message = f"Saved episode {episode.episode_id} to {destination}. Return arms to start."
            return destination

    def reset_to_start(self) -> None:
        with self.lock:
            if self.phase != "needs_reset":
                raise RuntimeError(f"cannot reset while phase is {self.phase}")
            self.phase = "resetting"
            self.message = "Returning both arms to the configured start pose..."
            self.worker = threading.Thread(
                target=self._reset_worker, name="evaluation-reset", daemon=True
            )
            self.worker.start()

    def _reset_worker(self) -> None:
        try:
            right_future = self.pool.submit(self.right.move_to_start_position)
            left_future = self.pool.submit(self.left.move_to_start_position)
            right_future.result()
            left_future.result()
            current = fetch_bimanual_state(self.right, self.left, self.pool)
            require_start_position(current, configured_start_position(self.config))
            with self.lock:
                self.phase = "ready"
                self.current_episode = None
                self.executed_steps = 0
                self.message = "Start pose reached. Reset the T-shirt, then start another rollout."
        except Exception as exc:
            with self.lock:
                self.phase = "disabled"
                self.message = f"Reset failed and motors were disabled: {exc}"
            self.disable_motors()

    def disable_motors(self) -> None:
        self.request_shutdown()
        immediately_disable(self.right)
        immediately_disable(self.left)
        with self.lock:
            self.phase = "disabled"
            self.message = "Motor torque is disabled. Restart the program to continue."

    def request_shutdown(self) -> None:
        with self.lock:
            self.shutdown_requested = True
            self.stop_event.set()

    def checkpoint_for_shutdown(self, timeout_s: float = 15.0) -> Path | None:
        worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout_s)
        if worker is not None and worker.is_alive():
            print(
                "[evaluation-save-warning] rollout worker did not stop before the "
                "checkpoint timeout; the most recent periodic checkpoint was preserved",
                flush=True,
            )
            with self.lock:
                root = self.active_episode.root if self.active_episode is not None else None
            return root if root is not None and (root / "episode.yaml").exists() else None
        with self.lock:
            episode = self.pending_episode or self.active_episode
        if episode is None or not episode.actions["right"]:
            return None
        episode.checkpoint(
            status="interrupted",
            interruption_reason="evaluation_server_shutdown",
        )
        return episode.root


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenArm Policy Evaluation</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#071019;color:#eef6fc}
header{padding:14px 18px;background:#0d1a25;position:sticky;top:0;z-index:2;display:flex;gap:18px;align-items:center}
h1{font-size:19px;margin:0}.badge{padding:5px 10px;border-radius:999px;background:#24394a;text-transform:uppercase;font-weight:700}
#message{color:#b9cedd}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:8px}
figure{margin:0;background:#101e2a;border:1px solid #294354}img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#000}
figcaption{padding:8px 10px;color:#a9c0d0}.panel{margin:8px;padding:14px;background:#101e2a;border:1px solid #294354;border-radius:8px}
button,select{font:inherit;padding:10px 14px;margin:4px;background:#1c3b50;color:#fff;border:1px solid #42677e;border-radius:6px}
button:disabled{opacity:.35}.success{background:#17623f}.failure{background:#7a382d}.danger{background:#a32121;font-weight:800}
.meta{font-family:ui-monospace,monospace;color:#9fb6c7;white-space:pre-wrap}strong{color:#fff}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1 id="policy-title">OpenArm Policy Evaluation</h1><span id="phase" class="badge">loading</span><span id="message">Connecting…</span></header>
<main class="grid">
<figure><img src="/stream/wrist_right.mjpg"><figcaption>Right wrist</figcaption></figure>
<figure><img src="/stream/ceiling.mjpg"><figcaption>Ceiling</figcaption></figure>
<figure><img src="/stream/wrist_left.mjpg"><figcaption>Left wrist</figcaption></figure>
</main>
<section class="panel">
  <button id="start">Start &amp; Record</button>
  <button id="stop">Stop Policy</button>
  <button id="success" class="success">Success</button>
  <select id="reason">FAILURE_OPTIONS</select>
  <button id="failure" class="failure">Failure</button>
  <button id="reset">Return Arms to Start</button>
  <button id="disable" class="danger">DISABLE MOTORS</button>
  <p><strong>Physical E-stop:</strong> use it for emergencies. The red browser button depends on the PC and network stack.</p>
  <div id="meta" class="meta"></div>
</section>
<script>
const token="CSRF_TOKEN";
const phase=document.getElementById('phase'),message=document.getElementById('message'),meta=document.getElementById('meta');
const start=document.getElementById('start'),stop=document.getElementById('stop'),success=document.getElementById('success'),failure=document.getElementById('failure'),reset=document.getElementById('reset'),disable=document.getElementById('disable'),reason=document.getElementById('reason');
async function post(path,body={}){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:JSON.stringify(body)});const x=await r.json();if(!r.ok)throw new Error(x.error||r.statusText);return x}
async function update(){try{const s=await(await fetch('/api/status')).json();
 phase.textContent=s.phase;message.textContent=s.message;
 document.getElementById('policy-title').textContent=`OpenArm ${s.policy_type.toUpperCase()} Evaluation`;
 start.disabled=s.phase!=='ready';stop.disabled=s.phase!=='recording';success.disabled=s.phase!=='awaiting_label';failure.disabled=s.phase!=='awaiting_label';reset.disabled=s.phase!=='needs_reset';disable.disabled=s.phase==='disabled';
 meta.textContent=`policy: ${s.policy_checkpoint}\ntask: ${s.policy_task??'-'}\nsession: ${s.session_id}\nepisode: ${s.current_episode??'-'}\nsteps: ${s.executed_steps}\ndataset: ${s.dataset_root}\nlast error: ${s.last_error??'-'}\ndiagnostic: ${s.last_diagnostic??'-'}`;
}catch(e){message.textContent=e.message}setTimeout(update,500)}
start.onclick=()=>post('/api/start').catch(e=>alert(e.message));
stop.onclick=()=>post('/api/stop').catch(e=>alert(e.message));
success.onclick=()=>confirm('Save this rollout as successful?')&&post('/api/label',{success:true}).catch(e=>alert(e.message));
failure.onclick=()=>confirm('Save this rollout as failed?')&&post('/api/label',{success:false,failure_reason:reason.value}).catch(e=>alert(e.message));
reset.onclick=()=>confirm('Workspace clear and safe to return both arms to start?')&&post('/api/reset').catch(e=>alert(e.message));
disable.onclick=()=>confirm('Disable motor torque now?')&&post('/api/disable').catch(e=>alert(e.message));update();
</script></body></html>"""


def make_handler(runtime: EvaluationRuntime, camera_hub: CameraHub, csrf_token: str):
    options = "".join(f'<option value="{reason}">{reason}</option>' for reason in FAILURE_REASONS)
    page = HTML.replace("FAILURE_OPTIONS", options).replace("CSRF_TOKEN", csrf_token).encode()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status=HTTPStatus.OK):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif path == "/api/status":
                self._json(runtime.status())
            elif path.startswith("/stream/") and path.endswith(".mjpg"):
                name = path.removeprefix("/stream/").removesuffix(".mjpg")
                self._stream(name)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _stream(self, name):
            feature_key = CAMERA_KEYS.get(name)
            if feature_key is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_timestamp = None
            try:
                while True:
                    frame = camera_hub.snapshot()[feature_key]
                    if frame.timestamp_ns == last_timestamp:
                        time.sleep(0.02)
                        continue
                    last_timestamp = frame.timestamp_ns
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame.jpeg)}\r\n\r\n".encode())
                    self.wfile.write(frame.jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if self.headers.get("X-CSRF-Token") != csrf_token:
                self._json({"error": "invalid session token"}, HTTPStatus.FORBIDDEN)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/start":
                    runtime.start_rollout()
                elif path == "/api/stop":
                    runtime.stop_rollout()
                elif path == "/api/label":
                    runtime.label(bool(body.get("success")), body.get("failure_reason"))
                elif path == "/api/reset":
                    runtime.reset_to_start()
                elif path == "/api/disable":
                    runtime.disable_motors()
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                self._json({"ok": True})
            except (RuntimeError, ValueError, OSError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.CONFLICT)

        def log_message(self, format, *args):
            if not self.path.startswith("/api/status"):
                print(f"evaluation-ui: {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-hardware", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--deployment-manifest", type=Path)
    parser.add_argument("--session-id")
    parser.add_argument("--garment-id", default="tshirt-evaluation")
    parser.add_argument("--operator-id", default="operator")
    parser.add_argument("--num-actions", type=int, default=85)
    parser.add_argument("--max-policy-steps", type=int, default=1800)
    parser.add_argument("--first-arm-delta-limit", type=float, default=0.5)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    if not args.confirm_hardware:
        parser.error("hardware access requires --confirm-hardware")
    if not __import__("sys").stdin.isatty():
        parser.error("evaluation mode requires an interactive terminal")
    adapter = resolve_checkpoint(args.checkpoint, args.deployment_manifest)
    if not 1 <= args.num_actions <= adapter.chunk_size:
        parser.error(f"--num-actions must be in [1,{adapter.chunk_size}]")
    session_id = args.session_id or datetime.now().strftime(f"{adapter.policy_type}-eval-%Y%m%d-%H%M%S")
    if not all(character.isalnum() or character in "._-" for character in session_id):
        parser.error("--session-id may contain only letters, numbers, '.', '_' and '-'")
    config = openarm_driver.Config(ROBOT_CONFIG)
    require_can_interfaces_up(config)
    dataset_root = ROOT / "evaluation_data" / "sessions" / session_id / "dataset"
    if dataset_root.exists():
        parser.error(f"evaluation session already exists: {dataset_root}")
    dataset_root.mkdir(parents=True)
    session = {
        "collection_session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root.relative_to(ROOT)),
        "dataset_split": "evaluation",
        "garment_id": args.garment_id,
        "initial_pose_id": "flat-horizontal",
        "operator_id": args.operator_id,
        "policy_checkpoint": str(args.checkpoint.resolve()),
        "policy_type": adapter.policy_type,
        "policy_task": adapter.task,
        "deployment_manifest": None if args.deployment_manifest is None else str(args.deployment_manifest.resolve()),
        "policy_num_actions": args.num_actions,
        "policy_representation": adapter.representation,
        "recorded_joint_order": "right_then_left",
        "recorded_joint_units": "radians",
        "recorded_action_grippers": "executed_physical_radians",
    }
    atomic_yaml(dataset_root.parent / "session.yaml", session)
    atomic_yaml(dataset_root / "metadata.yaml", {"episodes": []})

    policy_process = None
    connection = None
    camera_hub = None
    right = left = None
    server = None
    runtime = None
    motion_authorized = False
    with ThreadPoolExecutor(max_workers=8) as pool:
        try:
            policy_process, connection = start_policy(args.checkpoint, args.num_actions, args.deployment_manifest)
            camera_hub = CameraHub()
            camera_hub.start()
            right = openarm_driver.SingleArmDriver("right_arm", config)
            left = openarm_driver.SingleArmDriver("left_arm", config)
            answer = input(
                "Clear the workspace, keep your hand beside the emergency stop "
                "(do not press it unless aborting), then type MOVE to home the arms: "
            )
            if answer.strip() != "MOVE":
                print("[evaluation] cancelled; motors were never enabled")
                return
            motion_authorized = True
            right.start()
            left.start()
            current = fetch_bimanual_state(right, left, pool)
            require_start_position(current, configured_start_position(config))
            runtime = EvaluationRuntime(
                config=config,
                right=right,
                left=left,
                pool=pool,
                camera_hub=camera_hub,
                policy_connection=connection,
                dataset_root=dataset_root,
                session=session,
                num_actions=args.num_actions,
                max_policy_steps=args.max_policy_steps,
                first_arm_delta_limit=args.first_arm_delta_limit,
            )
            token = secrets.token_urlsafe(24)
            server = ThreadingHTTPServer(
                ("127.0.0.1", args.port), make_handler(runtime, camera_hub, token)
            )
            server.daemon_threads = True
            print(f"Evaluation UI: http://127.0.0.1:{args.port}")
            print(f"Evaluation dataset: {dataset_root}")
            print("Press Ctrl+C to disable torque and stop the server")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n[evaluation] interrupted; disabling torque immediately")
        finally:
            if runtime is not None:
                runtime.request_shutdown()
            if server is not None:
                server.server_close()
            if motion_authorized and right is not None:
                immediately_disable(right)
            if motion_authorized and left is not None:
                immediately_disable(left)
            if runtime is not None:
                checkpoint_path = runtime.checkpoint_for_shutdown()
                if checkpoint_path is not None:
                    print(f"[evaluation] interrupted rollout saved at {checkpoint_path}")
            if camera_hub is not None:
                camera_hub.close()
            if connection is not None:
                try:
                    connection.send({"command": "stop"})
                except (BrokenPipeError, EOFError):
                    pass
                connection.close()
            if policy_process is not None:
                try:
                    policy_process.wait(timeout=5)
                except Exception:
                    policy_process.terminate()


if __name__ == "__main__":
    main()
