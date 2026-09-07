"""Deterministically align raw OpenArm episodes and export LeRobotDataset v3."""

import argparse
from bisect import bisect_left
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .raw_dataset import CAMERA_NAMES, EpisodeRecord, assigned_split, load_records
from .representation import (
    LEROBOT_OPENARM_REPRESENTATION,
    REPRESENTATIONS,
    SAFE_GRIPPER_OPEN_DEG,
    get_spec,
    transform_action,
    transform_state,
)


FPS = 30
FRAME_PERIOD_NS = round(1e9 / FPS)
DEFAULT_MAX_CAMERA_SKEW_NS = 25_000_000
class AlignmentError(RuntimeError):
    """Raised when a raw episode cannot form a complete causal timeline."""


@dataclass
class KinematicSeries:
    timestamps: np.ndarray
    fields: dict[str, np.ndarray]


@dataclass
class AlignedEpisode:
    record: EpisodeRecord
    ticks: np.ndarray
    camera_paths: dict[str, list[Path]]
    state: np.ndarray
    action: np.ndarray
    velocity: np.ndarray | None
    effort: np.ndarray | None
    maximum_camera_skew_ns: int


def _read_kinematics(path: Path) -> KinematicSeries:
    if not path.exists():
        raise AlignmentError(f"missing kinematic stream: {path}")
    table = pq.read_table(path)
    if "timestamp" not in table.column_names:
        raise AlignmentError(f"missing timestamp column: {path}")
    timestamps = table["timestamp"].cast(pa.int64()).to_numpy(zero_copy_only=False)
    order = np.argsort(timestamps, kind="stable")
    fields = {}
    for name in ("qpos", "qvel", "qtorque"):
        if name not in table.column_names:
            continue
        values = np.asarray(table[name].to_pylist(), dtype=np.float32)
        fields[name] = values[order]
    if "qpos" not in fields:
        raise AlignmentError(f"missing qpos column: {path}")
    return KinematicSeries(timestamps=timestamps[order], fields=fields)


def _camera_index(path: Path) -> tuple[list[int], list[Path]]:
    images = sorted((*path.glob("*.jpeg"), *path.glob("*.jpg")))
    indexed = []
    for image in images:
        try:
            indexed.append((int(image.stem), image))
        except ValueError:
            continue
    indexed.sort(key=lambda item: item[0])
    return [item[0] for item in indexed], [item[1] for item in indexed]


def _nearest_paths(ticks, timestamps, paths, maximum_skew_ns):
    selected = []
    largest_skew = 0
    for tick in ticks:
        index = bisect_left(timestamps, int(tick))
        candidates = []
        if index < len(timestamps):
            candidates.append(index)
        if index > 0:
            candidates.append(index - 1)
        if not candidates:
            raise AlignmentError("camera stream is empty")
        nearest = min(candidates, key=lambda candidate: abs(timestamps[candidate] - tick))
        skew = abs(timestamps[nearest] - int(tick))
        if skew > maximum_skew_ns:
            raise AlignmentError(
                f"camera skew {skew / 1e6:.1f} ms exceeds {maximum_skew_ns / 1e6:.1f} ms"
            )
        largest_skew = max(largest_skew, skew)
        selected.append(paths[nearest])
    return selected, largest_skew


def _causal_sample(series: KinematicSeries, ticks, field="qpos"):
    values = series.fields[field]
    indexes = np.searchsorted(series.timestamps, ticks, side="right") - 1
    if np.any(indexes < 0):
        raise AlignmentError("action timeline requires a future command")
    return values[indexes]


def _interpolate(series: KinematicSeries, ticks, field="qpos"):
    values = series.fields[field]
    right = np.searchsorted(series.timestamps, ticks, side="right")
    left = right - 1
    if np.any(left < 0) or np.any(right >= len(series.timestamps)):
        raise AlignmentError(f"{field} cannot bracket the complete timeline")
    left_t = series.timestamps[left].astype(np.float64)
    right_t = series.timestamps[right].astype(np.float64)
    denominator = np.maximum(right_t - left_t, 1.0)
    alpha = ((ticks.astype(np.float64) - left_t) / denominator)[:, None]
    return (values[left] + alpha * (values[right] - values[left])).astype(np.float32)


def align_episode(
    record: EpisodeRecord,
    maximum_camera_skew_ns=DEFAULT_MAX_CAMERA_SKEW_NS,
) -> AlignedEpisode:
    right_action = _read_kinematics(record.path / "action/arms/right/state.parquet")
    left_action = _read_kinematics(record.path / "action/arms/left/state.parquet")
    right_observation = _read_kinematics(record.path / "obs/arms/right/state.parquet")
    left_observation = _read_kinematics(record.path / "obs/arms/left/state.parquet")

    camera_indexes = {
        name: _camera_index(record.path / "cameras" / name) for name in CAMERA_NAMES
    }
    if any(not timestamps for timestamps, _ in camera_indexes.values()):
        raise AlignmentError("one or more required camera streams are empty")

    starts = [
        right_action.timestamps[0],
        left_action.timestamps[0],
        right_observation.timestamps[0],
        left_observation.timestamps[0],
        *(timestamps[0] for timestamps, _ in camera_indexes.values()),
    ]
    ends = [
        right_action.timestamps[-1],
        left_action.timestamps[-1],
        right_observation.timestamps[-1],
        left_observation.timestamps[-1],
        *(timestamps[-1] for timestamps, _ in camera_indexes.values()),
    ]
    overlap_start = int(max(starts))
    overlap_end = int(min(ends))
    first_tick = ((overlap_start + FRAME_PERIOD_NS - 1) // FRAME_PERIOD_NS) * FRAME_PERIOD_NS
    if overlap_end <= first_tick:
        raise AlignmentError("streams have no usable common time interval")
    ticks = np.arange(first_tick, overlap_end, FRAME_PERIOD_NS, dtype=np.int64)
    if len(ticks) < FPS:
        raise AlignmentError("aligned episode is shorter than one second")

    selected_cameras = {}
    maximum_skew = 0
    for name, (timestamps, paths) in camera_indexes.items():
        selected, camera_skew = _nearest_paths(
            ticks, timestamps, paths, maximum_camera_skew_ns
        )
        selected_cameras[name] = selected
        maximum_skew = max(maximum_skew, camera_skew)

    action = np.concatenate(
        [_causal_sample(right_action, ticks), _causal_sample(left_action, ticks)], axis=1
    ).astype(np.float32)
    state = np.concatenate(
        [_interpolate(right_observation, ticks), _interpolate(left_observation, ticks)],
        axis=1,
    ).astype(np.float32)

    velocity = None
    if "qvel" in right_observation.fields and "qvel" in left_observation.fields:
        velocity = np.concatenate(
            [
                _interpolate(right_observation, ticks, "qvel"),
                _interpolate(left_observation, ticks, "qvel"),
            ],
            axis=1,
        ).astype(np.float32)
    effort = None
    if "qtorque" in right_observation.fields and "qtorque" in left_observation.fields:
        effort = np.concatenate(
            [
                _interpolate(right_observation, ticks, "qtorque"),
                _interpolate(left_observation, ticks, "qtorque"),
            ],
            axis=1,
        ).astype(np.float32)

    return AlignedEpisode(
        record=record,
        ticks=ticks,
        camera_paths=selected_cameras,
        state=state,
        action=action,
        velocity=velocity,
        effort=effort,
        maximum_camera_skew_ns=maximum_skew,
    )


def _feature_spec(first: AlignedEpisode, representation: str):
    spec = get_spec(representation)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(spec.joint_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": list(spec.joint_names),
        },
    }
    if spec.include_telemetry and first.velocity is not None:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": list(spec.joint_names),
        }
    if spec.include_telemetry and first.effort is not None:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": list(spec.joint_names),
        }
    for raw_camera, output_camera in spec.camera_map.items():
        with Image.open(first.camera_paths[raw_camera][0]) as image:
            width, height = image.size
        features[f"observation.images.{output_camera}"] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def convert(
    raw_root: Path,
    output_root: Path,
    repo_id: str,
    dataset_split: str,
    maximum_camera_skew_ms: float,
    representation: str = LEROBOT_OPENARM_REPRESENTATION,
):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    spec = get_spec(representation)
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    raw_metadata = {}
    metadata_path = raw_root / "metadata.yaml"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as source:
            raw_metadata = yaml.safe_load(source) or {}
    task = raw_metadata.get("tasks", [{}])[0].get(
        "prompt", "Fold the T-shirt into a compact rectangle."
    )

    records = load_records(raw_root)
    splits = assigned_split(records)
    selected = [
        record
        for record in records
        if record.success and splits.get(record.episode_id) == dataset_split
    ]
    if not selected:
        raise ValueError(f"no accepted episodes assigned to split {dataset_split!r}")

    maximum_camera_skew_ns = round(maximum_camera_skew_ms * 1e6)
    aligned = []
    converted = []
    for index, record in enumerate(selected, start=1):
        episode = align_episode(record, maximum_camera_skew_ns=maximum_camera_skew_ns)
        aligned.append(episode)
        converted.append(
            (
                episode,
                transform_state(episode.state, representation),
                transform_action(episode.action, representation),
            )
        )
        print(
            f"[align {index}/{len(selected)}] raw episode {record.episode_id}: "
            f"{len(episode.ticks)} frames, max camera skew "
            f"{episode.maximum_camera_skew_ns / 1e6:.2f} ms",
            flush=True,
        )
    features = _feature_spec(aligned[0], representation)
    include_velocity = spec.include_telemetry and all(
        episode.velocity is not None for episode in aligned
    )
    include_effort = spec.include_telemetry and all(
        episode.effort is not None for episode in aligned
    )
    if not include_velocity:
        features.pop("observation.velocity", None)
    if not include_effort:
        features.pop("observation.effort", None)

    conversion_started_ns = time.time_ns()
    staging_root = output_root.with_name(
        f".{output_root.name}.staging-{conversion_started_ns}"
    )
    dataset = None
    finalized = False
    reports = []
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=staging_root,
            fps=FPS,
            robot_type=spec.robot_type,
            features=features,
            use_videos=True,
            image_writer_threads=3,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=23),
        )
        for episode, state, action in converted:
            for frame_index in range(len(episode.ticks)):
                frame = {
                    "task": task,
                    "observation.state": state[frame_index],
                    "action": action[frame_index],
                }
                if include_velocity:
                    frame["observation.velocity"] = episode.velocity[frame_index]
                if include_effort:
                    frame["observation.effort"] = episode.effort[frame_index]
                for raw_camera, output_camera in spec.camera_map.items():
                    with Image.open(
                        episode.camera_paths[raw_camera][frame_index]
                    ) as image:
                        frame[f"observation.images.{output_camera}"] = np.asarray(
                            image.convert("RGB"), dtype=np.uint8
                        )
                dataset.add_frame(frame)
            dataset.save_episode()
            reports.append(
                {
                    "raw_episode_id": episode.record.episode_id,
                    "garment_id": episode.record.garment_id,
                    "frames": len(episode.ticks),
                    "duration_s": len(episode.ticks) / FPS,
                    "maximum_camera_skew_ms": episode.maximum_camera_skew_ns / 1e6,
                }
            )
            print(
                f"[encode {len(reports)}/{len(converted)}] raw episode "
                f"{episode.record.episode_id}: {len(episode.ticks)} frames",
                flush=True,
            )
        dataset.finalize()
        finalized = True

        manifest = {
            "format": "LeRobotDataset-v3",
            "representation": representation,
            "repo_id": repo_id,
            "robot_type": spec.robot_type,
            "split": dataset_split,
            "fps": FPS,
            "source_representation": {
                "joint_order": "right_then_left",
                "arm_joint_units": "radians",
                "right_gripper_action_domain": [-1.0, 0.0],
                "left_gripper_action_domain": [0.0, 1.0],
                "gripper_state_units": "radians",
                "camera_names": list(CAMERA_NAMES),
            },
            "target_representation": {
                "joint_order": spec.joint_order,
                "joint_units": spec.joint_units,
                "gripper_units": spec.gripper_units,
                "joint_names": list(spec.joint_names),
                "camera_map": spec.camera_map,
                "safe_gripper_open_degrees": SAFE_GRIPPER_OPEN_DEG,
                "safe_gripper_closed_degrees": 0.0,
            },
            "action_semantics": (
                "latest causal absolute arm target with safe calibrated gripper target"
                if representation == LEROBOT_OPENARM_REPRESENTATION
                else "latest causal absolute bimanual IK target"
            ),
            "state_semantics": "linearly interpolated measured bimanual joint state",
            "camera_semantics": "nearest original JPEG within strict skew tolerance",
            "maximum_camera_skew_ms": maximum_camera_skew_ms,
            "conversion_started_ns": conversion_started_ns,
            "conversion_finished_ns": time.time_ns(),
            "source": str(raw_root),
            "episodes": reports,
        }
        with (staging_root / "openarm_conversion_manifest.json").open(
            "w", encoding="utf-8"
        ) as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
        staging_root.replace(output_root)
    except Exception:
        if dataset is not None and not finalized:
            try:
                dataset.finalize()
            except Exception:
                pass
        if staging_root.exists():
            failed_root = output_root.with_name(
                f".{output_root.name}.failed-{time.time_ns()}"
            )
            staging_root.replace(failed_root)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("folding_data/dataset"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/openarm-tshirt-fold")
    parser.add_argument(
        "--split", choices=("train", "validation", "evaluation"), default="train"
    )
    parser.add_argument("--maximum-camera-skew-ms", type=float, default=30.0)
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default=LEROBOT_OPENARM_REPRESENTATION,
    )
    args = parser.parse_args()
    convert(
        raw_root=args.raw_root,
        output_root=args.output_root,
        repo_id=args.repo_id,
        dataset_split=args.split,
        maximum_camera_skew_ms=args.maximum_camera_skew_ms,
        representation=args.representation,
    )


if __name__ == "__main__":
    main()
