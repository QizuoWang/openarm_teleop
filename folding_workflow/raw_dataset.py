"""Read and summarize the lossless OpenArm episode format."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics

import yaml


CAMERA_NAMES = ("wrist_right", "wrist_left", "ceiling")
ARM_STREAMS = (
    "action/arms/right/state.parquet",
    "action/arms/left/state.parquet",
    "obs/arms/right/state.parquet",
    "obs/arms/left/state.parquet",
)


@dataclass(frozen=True)
class EpisodeRecord:
    """Metadata and path for one committed raw episode."""

    episode_id: int
    path: Path
    success: bool
    failure_reason: str | None
    task_index: int
    started_at_ns: int
    ended_at_ns: int
    session: dict
    stream_counts: dict

    @property
    def duration_s(self) -> float | None:
        if self.ended_at_ns <= self.started_at_ns:
            return None
        return (self.ended_at_ns - self.started_at_ns) / 1e9

    @property
    def garment_id(self) -> str:
        return str(self.session.get("garment_id", "unknown"))


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def load_records(raw_root: Path) -> list[EpisodeRecord]:
    """Load committed episode records, preferring per-episode manifests."""
    metadata = load_yaml(raw_root / "metadata.yaml")
    metadata_by_id = {
        str(item.get("id")): item for item in metadata.get("episodes", [])
    }
    episodes_root = raw_root / "episodes"
    if not episodes_root.exists():
        return []

    records = []
    for episode_path in sorted(
        (path for path in episodes_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    ):
        manifest = load_yaml(episode_path / "episode.yaml")
        source = {**metadata_by_id.get(episode_path.name, {}), **manifest}
        records.append(
            EpisodeRecord(
                episode_id=int(episode_path.name),
                path=episode_path,
                success=bool(source.get("success", False)),
                failure_reason=source.get("failure_reason"),
                task_index=int(source.get("task_index", 0)),
                started_at_ns=int(source.get("started_at_ns", 0) or 0),
                ended_at_ns=int(source.get("ended_at_ns", 0) or 0),
                session=source.get("session", {}) or {},
                stream_counts=source.get("stream_counts", {}) or {},
            )
        )
    return records


def assigned_split(records: list[EpisodeRecord]) -> dict[int, str]:
    """Create a deterministic complete-episode 90/10 train/validation split."""
    assignments = {}
    by_garment = defaultdict(list)
    for record in records:
        requested_split = record.session.get("dataset_split", "train")
        if requested_split in {"evaluation", "heldout", "test"}:
            assignments[record.episode_id] = "evaluation"
        elif record.success:
            by_garment[record.garment_id].append(record)
        else:
            assignments[record.episode_id] = "failure"

    total_learning = sum(len(group) for group in by_garment.values())
    target_validation = round(total_learning * 0.10)
    quotas = {
        garment_id: int(len(group) * 0.10)
        for garment_id, group in by_garment.items()
    }
    remainders = sorted(
        by_garment,
        key=lambda garment_id: (
            -(len(by_garment[garment_id]) * 0.10 - quotas[garment_id]),
            garment_id,
        ),
    )
    remaining = target_validation - sum(quotas.values())
    for garment_id in remainders[:remaining]:
        quotas[garment_id] += 1

    for garment_id, garment_records in by_garment.items():
        garment_records.sort(key=lambda record: record.episode_id)
        quota = quotas[garment_id]
        validation_indexes = {
            max(0, round((index + 1) * len(garment_records) / (quota + 1)) - 1)
            for index in range(quota)
        }
        for index, record in enumerate(garment_records):
            assignments[record.episode_id] = "validation" if index in validation_indexes else "train"
    return assignments


def raw_manifest_digest(raw_root: Path, records: list[EpisodeRecord]) -> str:
    """Hash stable episode metadata without reading large image payloads."""
    digest = hashlib.sha256()
    metadata_path = raw_root / "metadata.yaml"
    if metadata_path.exists():
        digest.update(metadata_path.read_bytes())
    for record in records:
        digest.update(str(record.episode_id).encode())
        digest.update(json.dumps(record.session, sort_keys=True).encode())
        digest.update(json.dumps(record.stream_counts, sort_keys=True).encode())
        episode_manifest = record.path / "episode.yaml"
        if episode_manifest.exists():
            digest.update(episode_manifest.read_bytes())
    return digest.hexdigest()


def summarize(raw_root: Path) -> dict:
    records = load_records(raw_root)
    splits = assigned_split(records)
    durations = [record.duration_s for record in records if record.duration_s is not None]
    successes = [record for record in records if record.success]
    failures = [record for record in records if not record.success]
    failure_reasons = Counter(record.failure_reason or "unclassified" for record in failures)
    garment_counts = Counter(record.garment_id for record in successes)
    split_counts = Counter(splits.values())
    session_records = defaultdict(list)
    for record in records:
        session_records[
            str(record.session.get("collection_session_id", "unknown"))
        ].append(record)
    session_summaries = {}
    for session_id, grouped_records in sorted(session_records.items()):
        session_failures = sum(not record.success for record in grouped_records)
        session_rate = session_failures / len(grouped_records)
        session_summaries[session_id] = {
            "episodes": len(grouped_records),
            "accepted": len(grouped_records) - session_failures,
            "failed": session_failures,
            "rejection_rate": session_rate,
            "batch_gate_pass": session_rate <= 0.10,
            "garment_ids": sorted({record.garment_id for record in grouped_records}),
        }

    duration_summary = {}
    if durations:
        ordered = sorted(durations)
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        duration_summary = {
            "minimum_s": min(durations),
            "median_s": statistics.median(durations),
            "p95_s": ordered[p95_index],
            "maximum_s": max(durations),
        }

    total = len(records)
    rejection_rate = (len(failures) / total) if total else 0.0
    return {
        "raw_root": str(raw_root.resolve()),
        "raw_manifest_sha256": raw_manifest_digest(raw_root, records),
        "episodes": total,
        "accepted": len(successes),
        "failed": len(failures),
        "rejection_rate": rejection_rate,
        "batch_gate": {
            "pass": rejection_rate <= 0.10,
            "reason": None if rejection_rate <= 0.10 else "rejection_rate_above_10_percent",
        },
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "accepted_by_garment": dict(sorted(garment_counts.items())),
        "sessions": session_summaries,
        "split_counts": dict(sorted(split_counts.items())),
        "durations": duration_summary,
        "episode_splits": {str(key): value for key, value in sorted(splits.items())},
    }


def representative_images(record: EpisodeRecord, camera="ceiling") -> list[Path]:
    images = sorted((record.path / "cameras" / camera).glob("*.jpeg"))
    if not images:
        images = sorted((record.path / "cameras" / camera).glob("*.jpg"))
    if not images:
        return []
    indexes = sorted({0, len(images) // 2, len(images) - 1})
    return [images[index] for index in indexes]
