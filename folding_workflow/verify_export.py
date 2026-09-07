"""Verify a LeRobot 0.6 OpenArm export against its reviewed raw source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .convert import FPS, align_episode
from .raw_dataset import assigned_split, load_records
from .representation import (
    LEROBOT_CAMERA_MAP,
    LEROBOT_JOINT_NAMES,
    LEROBOT_OPENARM_REPRESENTATION,
    transform_action,
    transform_state,
)


REQUIRED_SYSTEM_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
}
MODEL_FEATURES = {
    "observation.state",
    "action",
    *(f"observation.images.{name}" for name in LEROBOT_CAMERA_MAP.values()),
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_data(root: Path) -> pa.Table:
    files = sorted(root.glob("data/**/*.parquet"))
    _check(bool(files), f"no data parquet files below {root}")
    table = pa.concat_tables([pq.read_table(path) for path in files])
    return table.sort_by([("index", "ascending")])


def _matrix(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=np.float32)


def _assert_numeric_equal(label: str, actual: np.ndarray, expected: np.ndarray) -> float:
    _check(actual.shape == expected.shape, f"{label} shape {actual.shape} != {expected.shape}")
    maximum = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    _check(np.allclose(actual, expected, rtol=1e-6, atol=1e-5), f"{label} max error {maximum}")
    return maximum


def _verify_stats(stats: dict, name: str, values: np.ndarray) -> None:
    feature_stats = stats.get(name)
    _check(isinstance(feature_stats, dict), f"missing statistics for {name}")
    for key in ("min", "max", "mean", "std", "count", "q01", "q99"):
        _check(key in feature_stats, f"{name} statistics missing {key}")
    expected = {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values.astype(np.float64), axis=0),
        "std": np.std(values.astype(np.float64), axis=0),
    }
    for key, target in expected.items():
        actual = np.asarray(feature_stats[key], dtype=np.float64)
        _check(
            np.allclose(actual, target, rtol=2e-4, atol=2e-4),
            f"{name} statistics {key} do not match exported rows",
        )
    count = np.asarray(feature_stats["count"]).reshape(-1)
    _check(len(count) == 1 and int(count[0]) == len(values), f"bad {name} count")


def verify_export(
    raw_root: Path,
    dataset_root: Path,
    expected_split: str,
    expected_episodes: int | None = None,
    expected_frames: int | None = None,
) -> dict:
    """Run offline structural, numeric, statistical, and video checks."""
    from lerobot.datasets import LeRobotDataset

    raw_root = raw_root.resolve()
    dataset_root = dataset_root.resolve()
    manifest = _load_json(dataset_root / "openarm_conversion_manifest.json")
    info = _load_json(dataset_root / "meta/info.json")
    stats = _load_json(dataset_root / "meta/stats.json")

    _check(manifest.get("representation") == LEROBOT_OPENARM_REPRESENTATION, "wrong representation")
    _check(manifest.get("split") == expected_split, "wrong manifest split")
    _check(info.get("codebase_version") == "v3.0", "wrong dataset codebase version")
    _check(info.get("robot_type") == "bi_openarm_follower", "wrong robot_type")
    _check(info.get("fps") == FPS, f"dataset FPS is not {FPS}")

    features = info.get("features", {})
    expected_feature_keys = MODEL_FEATURES | REQUIRED_SYSTEM_FEATURES
    _check(set(features) == expected_feature_keys, f"unexpected features: {sorted(set(features) ^ expected_feature_keys)}")
    for name in ("observation.state", "action"):
        feature = features[name]
        _check(feature.get("shape") == [16], f"bad {name} shape")
        _check(feature.get("names") == list(LEROBOT_JOINT_NAMES), f"bad {name} names/order")
    for camera_name in LEROBOT_CAMERA_MAP.values():
        feature = features[f"observation.images.{camera_name}"]
        _check(feature.get("dtype") == "video", f"{camera_name} is not video")
        _check(feature.get("shape") == [360, 640, 3], f"bad {camera_name} shape")

    records = load_records(raw_root)
    split_assignments = assigned_split(records)
    selected = [
        record
        for record in records
        if record.success and split_assignments.get(record.episode_id) == expected_split
    ]
    reports = manifest.get("episodes", [])
    raw_ids = [record.episode_id for record in selected]
    manifest_ids = [int(report["raw_episode_id"]) for report in reports]
    _check(manifest_ids == raw_ids, "manifest episode IDs do not match deterministic raw split")
    _check(info.get("total_episodes") == len(selected), "info episode count mismatch")
    if expected_episodes is not None:
        _check(len(selected) == expected_episodes, f"expected {expected_episodes} episodes, got {len(selected)}")

    table = _read_data(dataset_root)
    state = _matrix(table, "observation.state")
    action = _matrix(table, "action")
    total_frames = len(table)
    _check(info.get("total_frames") == total_frames, "info frame count mismatch")
    _check(sum(int(report["frames"]) for report in reports) == total_frames, "manifest frame count mismatch")
    if expected_frames is not None:
        _check(total_frames == expected_frames, f"expected {expected_frames} frames, got {total_frames}")

    indexes = table["index"].to_numpy(zero_copy_only=False)
    _check(np.array_equal(indexes, np.arange(total_frames)), "global indexes are not contiguous")
    exported_episode = table["episode_index"].to_numpy(zero_copy_only=False)
    exported_frame = table["frame_index"].to_numpy(zero_copy_only=False)
    exported_timestamp = table["timestamp"].to_numpy(zero_copy_only=False)

    state_error = 0.0
    action_error = 0.0
    offset = 0
    maximum_skew_ns = round(float(manifest["maximum_camera_skew_ms"]) * 1e6)
    for output_episode, (record, report) in enumerate(zip(selected, reports, strict=True)):
        aligned = align_episode(record, maximum_camera_skew_ns=maximum_skew_ns)
        frames = len(aligned.ticks)
        _check(frames == int(report["frames"]), f"episode {record.episode_id} frame count changed")
        episode_slice = slice(offset, offset + frames)
        state_error = max(
            state_error,
            _assert_numeric_equal(
                f"episode {record.episode_id} state",
                state[episode_slice],
                transform_state(aligned.state, LEROBOT_OPENARM_REPRESENTATION),
            ),
        )
        action_error = max(
            action_error,
            _assert_numeric_equal(
                f"episode {record.episode_id} action",
                action[episode_slice],
                transform_action(aligned.action, LEROBOT_OPENARM_REPRESENTATION),
            ),
        )
        _check(np.all(exported_episode[episode_slice] == output_episode), "episode indexes are not contiguous")
        _check(np.array_equal(exported_frame[episode_slice], np.arange(frames)), "frame indexes are not contiguous")
        _check(
            np.allclose(exported_timestamp[episode_slice], np.arange(frames) / FPS, atol=1e-5),
            f"episode {record.episode_id} timestamps are not {FPS} Hz",
        )
        offset += frames
    _check(offset == total_frames, "not all exported rows were verified")

    _verify_stats(stats, "observation.state", state)
    _verify_stats(stats, "action", action)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dataset = LeRobotDataset(repo_id=manifest["repo_id"], root=dataset_root)
    _check(len(dataset) == total_frames, "LeRobotDataset length mismatch")
    sample_indexes = sorted({0, total_frames // 2, total_frames - 1})
    decoded = []
    for sample_index in sample_indexes:
        item = dataset[sample_index]
        for camera_name in LEROBOT_CAMERA_MAP.values():
            key = f"observation.images.{camera_name}"
            image = item[key]
            _check(tuple(image.shape) == (3, 360, 640), f"bad decoded shape for {key}")
            _check(bool(np.isfinite(image.numpy()).all()), f"non-finite decoded pixels in {key}")
            decoded.append({"index": sample_index, "camera": camera_name})

    result = {
        "format": "openarm-export-verification-v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "dataset_root": str(dataset_root),
        "raw_root": str(raw_root),
        "split": expected_split,
        "episodes": len(selected),
        "frames": total_frames,
        "maximum_state_error": state_error,
        "maximum_action_error": action_error,
        "decoded_video_samples": decoded,
    }
    output = dataset_root / "openarm_export_verification.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "evaluation"), required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--expected-frames", type=int)
    args = parser.parse_args()
    result = verify_export(
        raw_root=args.raw_root,
        dataset_root=args.dataset_root,
        expected_split=args.split,
        expected_episodes=args.expected_episodes,
        expected_frames=args.expected_frames,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
