# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License 2.0.

"""Monitor-first control UI for safe, resumable OpenArm data collection."""

import argparse
import asyncio
import collections
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
import dataclasses
import datetime
import json
import os
import pathlib
import shutil
import time

import dora
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.templating import Jinja2Templates
import pyarrow as pa
import uvicorn
import yaml


base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=f"{base_dir}/templates")

node = None
tasks = []
session_metadata = {}
dataset_directory = pathlib.Path("folding_data/dataset")
minimum_free_bytes = 50 * 1024**3
auto_open = False
port = 8000


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if auto_open:
        await asyncio.create_subprocess_exec("xdg-open", f"http://127.0.0.1:{port}")
    yield


app = FastAPI(lifespan=_lifespan)


FAILURE_REASONS = (
    "bad_demonstration",
    "grasp_failure",
    "incomplete_fold",
    "unsafe_motion",
    "hardware_or_sensor",
)

REQUIRED_CAMERAS = (
    "camera_wrist_right",
    "camera_wrist_left",
    "camera_ceiling",
)
CAMERA_INPUTS = REQUIRED_CAMERAS + ("camera_head_left", "camera_head_right")
ARM_STATUS_INPUTS = ("arm_status_right", "arm_status_left")
VR_RECEIVE_TIMES_INPUTS = ("vr_receive_times", "vr_recv_ts")

CAMERA_TIMESTAMP_WINDOW = 60
CAMERA_STALE_AFTER_S = 1.0
CAMERA_MIN_FPS = 20.0
VR_TIMESTAMP_WINDOW = 120
VR_STALE_AFTER_S = 1.0
VR_MIN_FPS = 50.0
LONG_PRESS_S = 1.5


@dataclasses.dataclass
class State:
    """Current collection and operator state."""

    collecting: bool = False
    running: bool = True
    failure_menu: bool = False
    failure_reason_index: int = 0
    episode_number: int = 0
    task_index: int = 0
    task_title: str = ""
    arm_status_right: str = "stopped"
    arm_status_left: str = "stopped"
    recorder_state: str = "unknown"
    recorder_message: str = "Waiting for recorder"
    last_finalized_episode: int | None = None
    accepted_count: int = 0
    failed_count: int = 0
    quarantined_count: int = 0
    park_enabled: bool = False


@dataclasses.dataclass
class CameraStats:
    """Rolling health for one camera stream."""

    fps: float = 0.0
    jitter_ms: float = 0.0


@dataclasses.dataclass
class VrStreamStats:
    """Rolling health for the Quest packet stream."""

    fps: float = 0.0
    jitter_ms: float = 0.0


state = State()
camera_stats = {name: CameraStats() for name in CAMERA_INPUTS}
camera_timestamps = {
    name: collections.deque(maxlen=CAMERA_TIMESTAMP_WINDOW) for name in CAMERA_INPUTS
}
vr_stats = VrStreamStats()
vr_timestamps = collections.deque(maxlen=VR_TIMESTAMP_WINDOW)

_state_changed = asyncio.Condition()
state_version = 0


def _event_ts_to_seconds(timestamp) -> float:
    if isinstance(timestamp, datetime.datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        return timestamp.timestamp()
    if isinstance(timestamp, (int, float)):
        return float(timestamp) / 1e9
    return time.time()


def _update_rate(series, timestamp_s):
    if series and timestamp_s - series[-1] > CAMERA_STALE_AFTER_S:
        series.clear()
    series.append(timestamp_s)
    if len(series) < 2:
        return 0.0, 0.0
    span = series[-1] - series[0]
    if span <= 0:
        return 0.0, 0.0
    diffs = [series[index] - series[index - 1] for index in range(1, len(series))]
    return (len(series) - 1) / span, (max(diffs) - min(diffs)) * 1e3


def _update_camera_stats(event_id: str, timestamp_s: float) -> None:
    fps, jitter_ms = _update_rate(camera_timestamps[event_id], timestamp_s)
    camera_stats[event_id].fps = fps
    camera_stats[event_id].jitter_ms = jitter_ms


def _update_vr_stats(timestamp_s: float) -> None:
    fps, jitter_ms = _update_rate(vr_timestamps, timestamp_s)
    vr_stats.fps = fps
    vr_stats.jitter_ms = jitter_ms


async def _notify_state_changed() -> None:
    global state_version
    async with _state_changed:
        state_version += 1
        _state_changed.notify_all()


def _load_yaml(path: pathlib.Path | None, default=None):
    if path is None or not path.exists():
        return {} if default is None else default
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or ({} if default is None else default)


def _load_dataset_progress() -> None:
    """Restore counters and choose the next never-used episode number."""
    metadata = _load_yaml(dataset_directory / "metadata.yaml")
    episode_records = metadata.get("episodes", [])
    ids = []
    session_id = session_metadata.get("collection_session_id")
    for episode in episode_records:
        try:
            ids.append(int(episode["id"]))
        except (KeyError, TypeError, ValueError):
            continue
        episode_session = episode.get("session", {})
        if session_id and episode_session.get("collection_session_id") != session_id:
            continue
        if episode.get("success"):
            state.accepted_count += 1
        else:
            state.failed_count += 1

    episodes_directory = dataset_directory / "episodes"
    if episodes_directory.exists():
        for path in episodes_directory.iterdir():
            if path.is_dir() and path.name.isdigit():
                ids.append(int(path.name))

    quarantine = dataset_directory / "quarantine"
    if quarantine.exists():
        state.quarantined_count = sum(1 for path in quarantine.iterdir() if path.is_dir())

    state.episode_number = max(ids, default=-1) + 1


def _disk_status():
    probe = dataset_directory
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "ok": usage.free >= minimum_free_bytes,
        "free_bytes": usage.free,
        "required_bytes": minimum_free_bytes,
    }


def _preflight(require_arms=False):
    now = time.time()
    checks = {}
    for camera in REQUIRED_CAMERAS:
        timestamps = camera_timestamps[camera]
        live = bool(timestamps) and now - timestamps[-1] <= CAMERA_STALE_AFTER_S
        checks[camera] = live and camera_stats[camera].fps >= CAMERA_MIN_FPS

    vr_live = bool(vr_timestamps) and now - vr_timestamps[-1] <= VR_STALE_AFTER_S
    checks["vr"] = vr_live and vr_stats.fps >= VR_MIN_FPS
    checks["recorder"] = state.recorder_state in {"ready", "finalized", "cancelled"} or (
        state.collecting and state.recorder_state == "recording"
    )
    checks["disk"] = _disk_status()["ok"]
    if require_arms:
        checks["arm_left"] = state.arm_status_left == "aligned"
        checks["arm_right"] = state.arm_status_right == "aligned"
    reasons = [name for name, healthy in checks.items() if not healthy]
    return not reasons, checks, reasons


def _snapshot():
    data_ready, data_checks, data_reasons = _preflight(require_arms=False)
    collection_ready, collection_checks, collection_reasons = _preflight(
        require_arms=True
    )
    return {
        "state": dataclasses.asdict(state),
        "failure_reasons": FAILURE_REASONS,
        "selected_failure_reason": FAILURE_REASONS[state.failure_reason_index],
        "session": session_metadata,
        "data_ready": data_ready,
        "data_checks": data_checks,
        "data_reasons": data_reasons,
        "collection_ready": collection_ready,
        "collection_checks": collection_checks,
        "collection_reasons": collection_reasons,
        "disk": _disk_status(),
    }


def _set_message(message: str) -> None:
    state.recorder_message = message


def _command_start():
    ready, _, reasons = _preflight(require_arms=True)
    if not ready:
        _set_message("Start blocked: " + ", ".join(reasons))
        return False
    node.send_output(
        "command",
        pa.array(["start"]),
        {
            "episode_number": state.episode_number,
            "task_index": state.task_index,
            "session_json": json.dumps(session_metadata, sort_keys=True),
        },
    )
    state.collecting = True
    state.failure_menu = False
    state.recorder_state = "recording"
    _set_message(f"Recording episode {state.episode_number}")
    return True


def _command_success():
    ready, _, reasons = _preflight(require_arms=True)
    if not ready:
        _set_message("Accept blocked: " + ", ".join(reasons))
        return False
    node.send_output("command", pa.array(["success"]))
    state.collecting = False
    state.failure_menu = False
    state.recorder_state = "finalizing"
    state.episode_number += 1
    _set_message("Finalizing accepted episode")
    return True


def _command_fail(reason: str):
    node.send_output(
        "command", pa.array(["fail"]), {"failure_reason": reason}
    )
    state.collecting = False
    state.failure_menu = False
    state.recorder_state = "finalizing"
    state.episode_number += 1
    _set_message(f"Finalizing failed episode: {reason}")


def _command_cancel():
    if not state.collecting:
        _set_message("Cancel ignored: no active episode")
        return
    node.send_output("command", pa.array(["cancel"]))
    state.collecting = False
    state.failure_menu = False
    state.recorder_state = "finalizing"
    state.episode_number += 1
    _set_message("Moving cancelled episode to quarantine")


def _command_quit():
    node.send_output("command", pa.array(["quit"]))
    state.running = False


def _command_arm_start():
    ready, _, reasons = _preflight(require_arms=False)
    if not ready:
        _set_message("Arm enable blocked: " + ", ".join(reasons))
        return False
    node.send_output("arm_command", pa.array(["start"]))
    _set_message("Arm enable requested; align both arms before collection")
    return True


def _command_arm_stop():
    node.send_output("arm_command", pa.array(["stop"]))
    _set_message("Arm commands paused")


def _command_park():
    if not state.park_enabled:
        _set_message("Park is locked until a real-hardware trajectory is approved")
        return False
    _set_message("Park requested")
    return True


@app.get("/", response_class=HTMLResponse)
def _root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="root.html",
        context={"snapshot": _snapshot(), "state_version": state_version},
    )


@app.post("/start")
def _start(request: Request):
    _command_start()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/success")
def _success(request: Request):
    _command_success()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/fail/{reason}")
def _fail(request: Request, reason: str):
    if reason not in FAILURE_REASONS:
        reason = "hardware_or_sensor"
    _command_fail(reason)
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/cancel")
def _cancel(request: Request):
    _command_cancel()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/quit")
def _quit(request: Request):
    _command_quit()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/arm/start")
def _arm_start(request: Request):
    _command_arm_start()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/arm/stop")
def _arm_stop(request: Request):
    _command_arm_stop()
    return RedirectResponse(request.url_for("_root"), 303)


@app.post("/park")
def _park(request: Request):
    _command_park()
    return RedirectResponse(request.url_for("_root"), 303)


@app.get("/events", response_class=EventSourceResponse)
async def _events(request: Request) -> AsyncIterable[ServerSentEvent]:
    try:
        last_version = int(request.query_params.get("since"))
    except (TypeError, ValueError):
        last_version = state_version
    while state.running:
        async with _state_changed:
            await _state_changed.wait_for(
                lambda: state_version != last_version or not state.running
            )
        if not state.running:
            break
        last_version = state_version
        yield ServerSentEvent(data=_snapshot(), id=str(state_version))


@app.get("/stats", response_class=EventSourceResponse)
async def _stats() -> AsyncIterable[ServerSentEvent]:
    while state.running:
        now = time.time()
        cameras = {}
        for name, stats in camera_stats.items():
            timestamps = camera_timestamps[name]
            if not timestamps or now - timestamps[-1] > CAMERA_STALE_AFTER_S:
                cameras[name] = {"fps": 0.0, "jitter_ms": 0.0}
            else:
                cameras[name] = dataclasses.asdict(stats)
        vr = (
            {"fps": 0.0, "jitter_ms": 0.0}
            if not vr_timestamps or now - vr_timestamps[-1] > VR_STALE_AFTER_S
            else dataclasses.asdict(vr_stats)
        )
        yield ServerSentEvent(data={"cameras": cameras, "vr": vr})
        await asyncio.sleep(0.5)


def _handle_recorder_status(value: str) -> None:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        state.recorder_state = "error"
        _set_message(f"Invalid recorder status: {value}")
        return
    recorder_state = payload.get("state", "unknown")
    state.recorder_state = recorder_state
    if recorder_state == "finalized":
        episode_number = payload.get("episode_number")
        state.last_finalized_episode = episode_number
        if payload.get("success"):
            state.accepted_count += 1
            _set_message(f"Episode {episode_number} accepted and saved")
        else:
            state.failed_count += 1
            reason = payload.get("failure_reason", "unclassified")
            _set_message(f"Episode {episode_number} saved as failure: {reason}")
    elif recorder_state == "cancelled":
        state.quarantined_count += 1
        _set_message(f"Episode {payload.get('episode_number')} quarantined")
    elif recorder_state == "error":
        _set_message(payload.get("message", "Recorder error"))
    elif recorder_state == "ready":
        _set_message("Recorder ready; collection remains arm-gated")


async def _handle_button_event(event_id, pressed, button_state):
    now = time.monotonic()
    previous = button_state.setdefault(
        event_id, {"pressed": False, "started": 0.0, "long_fired": False}
    )
    rising = pressed and not previous["pressed"]
    falling = not pressed and previous["pressed"]

    if rising:
        previous.update(pressed=True, started=now, long_fired=False)
        if event_id == "button_a":
            if state.failure_menu:
                _command_fail(FAILURE_REASONS[state.failure_reason_index])
            elif state.collecting:
                _command_success()
            else:
                _command_start()
        elif event_id == "button_b" and state.collecting:
            state.failure_menu = True
            _command_arm_stop()
            _set_message("Choose a failure reason with the left joystick; press A")
        elif event_id == "button_x":
            _command_arm_stop()

    if (
        pressed
        and not previous["long_fired"]
        and now - previous["started"] >= LONG_PRESS_S
    ):
        previous["long_fired"] = True
        if event_id == "button_b" and not state.collecting:
            _command_quit()
        elif event_id == "button_x" and not state.collecting:
            _command_arm_start()
        elif event_id == "button_y":
            if state.collecting:
                _command_cancel()
            else:
                _command_park()

    if falling:
        previous["pressed"] = False
    await _notify_state_changed()


async def _main_dora(server):
    """Process Dora inputs without authorizing arm motion at startup."""
    last_joystick_change = 0.0
    button_state = {}
    while state.running:
        if node.is_empty():
            await asyncio.sleep(0.001)
            continue
        event = node.next()
        if event["type"] == "STOP":
            state.running = False
            continue
        if event["type"] != "INPUT":
            continue

        event_id = event["id"]
        if event_id in CAMERA_INPUTS:
            _update_camera_stats(
                event_id,
                _event_ts_to_seconds(
                    event["metadata"].get(
                        "capture_timestamp_ns", event["metadata"].get("timestamp")
                    )
                ),
            )
            continue
        if event_id in ARM_STATUS_INPUTS:
            value = event["value"][0].as_py()
            if getattr(state, event_id) != value:
                setattr(state, event_id, value)
                await _notify_state_changed()
            continue
        if event_id in VR_RECEIVE_TIMES_INPUTS:
            for timestamp_ns in event["value"].to_pylist():
                _update_vr_stats(float(timestamp_ns) / 1e9)
            continue
        if event_id == "recorder_status":
            _handle_recorder_status(event["value"][0].as_py())
            await _notify_state_changed()
            continue
        if event_id == "joystick_y_left" and state.failure_menu:
            joystick = float(event["value"][0].as_py())
            now = time.monotonic()
            if abs(joystick) >= 0.7 and now - last_joystick_change >= 0.35:
                direction = 1 if joystick < 0 else -1
                state.failure_reason_index = (
                    state.failure_reason_index + direction
                ) % len(FAILURE_REASONS)
                last_joystick_change = now
                _set_message(
                    f"Failure reason: {FAILURE_REASONS[state.failure_reason_index]}"
                )
                await _notify_state_changed()
            continue
        if event_id in {"button_a", "button_b", "button_x", "button_y"}:
            await _handle_button_event(
                event_id, bool(event["value"][0].as_py()), button_state
            )

    server.should_exit = True


async def _main_async():
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="info"))
    uvicorn_task = asyncio.create_task(server.serve())
    dora_task = asyncio.create_task(_main_dora(server))
    await uvicorn_task
    state.running = False
    await dora_task


def main():
    """Run the collection dashboard and controller state machine."""
    global auto_open, dataset_directory, minimum_free_bytes, node, port
    global session_metadata, tasks

    parser = argparse.ArgumentParser(description="Control OpenArm folding collection")
    parser.add_argument(
        "--metadata-file", default=os.getenv("METADATA_FILE"), type=pathlib.Path
    )
    parser.add_argument(
        "--session-file", default=os.getenv("SESSION_FILE"), type=pathlib.Path
    )
    parser.add_argument(
        "--dataset-directory",
        default=os.getenv("DATASET_DIRECTORY", "folding_data/dataset"),
        type=pathlib.Path,
    )
    parser.add_argument(
        "--minimum-free-gib",
        default=float(os.getenv("MINIMUM_FREE_GIB", "50")),
        type=float,
    )
    parser.add_argument(
        "--auto-open",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("AUTO_OPEN", "") == "yes",
    )
    parser.add_argument("--port", default=int(os.getenv("PORT", "8000")), type=int)
    args = parser.parse_args()

    auto_open = args.auto_open
    port = args.port
    dataset_directory = args.dataset_directory
    minimum_free_bytes = int(args.minimum_free_gib * 1024**3)
    metadata = _load_yaml(args.metadata_file)
    session_metadata = _load_yaml(args.session_file)
    tasks = metadata.get("tasks", [])
    if not tasks:
        raise ValueError("metadata file must define at least one task")
    state.task_title = tasks[state.task_index]["prompt"]
    # No real park consumer or approved trajectory exists in this repository.
    # Keep this locked even if a session file is edited by hand.
    state.park_enabled = False
    _load_dataset_progress()

    node = dora.Node()
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
