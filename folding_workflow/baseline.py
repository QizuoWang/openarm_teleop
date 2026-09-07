"""Compact integrity manifests and workflow guards for frozen datasets."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable


FROZEN_MARKER = ".openarm-fold-frozen.json"
IGNORED_NAMES = {FROZEN_MARKER, ".DS_Store"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        if "__pycache__" in candidate.parts:
            continue
        if candidate.name in IGNORED_NAMES or candidate.suffix in IGNORED_SUFFIXES:
            continue
        yield candidate


def hash_path(path: Path) -> dict:
    """Hash file content plus relative paths into one compact tree digest."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    base = path if path.is_dir() else path.parent
    for candidate in _iter_files(path):
        if candidate.is_symlink():
            raise ValueError(f"baseline artifact contains a symlink: {candidate}")
        relative = candidate.relative_to(base).as_posix()
        size = candidate.stat().st_size
        file_digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.digest())
        count += 1
        total_bytes += size
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "files": count,
        "bytes": total_bytes,
    }


def frozen_marker(raw_root: Path) -> Path:
    return raw_root.resolve() / FROZEN_MARKER


def require_mutable(raw_root: Path, operation: str) -> None:
    marker = frozen_marker(raw_root)
    if not marker.exists():
        return
    try:
        detail = json.loads(marker.read_text(encoding="utf-8"))
        baseline = detail.get("baseline", "unknown")
    except (json.JSONDecodeError, OSError):
        baseline = "unknown"
    raise SystemExit(
        f"Dataset is frozen by baseline {baseline}: {raw_root.resolve()}. "
        f"Refusing to {operation}. Conversion and review remain read-only."
    )


def freeze(
    name: str,
    raw_root: Path,
    manifest_path: Path,
    artifacts: dict[str, Path],
    metadata: dict,
) -> dict:
    manifest_path = manifest_path.resolve()
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite baseline manifest: {manifest_path}")
    marker = frozen_marker(raw_root)
    if marker.exists():
        raise FileExistsError(f"dataset already has a frozen marker: {marker}")

    frozen_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(
        marker,
        {
            "baseline": name,
            "frozen_at": frozen_at,
            "manifest": str(manifest_path),
        },
    )
    try:
        hashed = {label: hash_path(path) for label, path in artifacts.items()}
        document = {
            "format": "openarm-fold-baseline-v1",
            "name": name,
            "frozen_at": frozen_at,
            "artifacts": hashed,
            "metadata": metadata,
        }
        _atomic_json(manifest_path, document)
    except Exception:
        marker.unlink(missing_ok=True)
        raise
    return document


def verify(manifest_path: Path) -> tuple[bool, list[dict]]:
    manifest_path = manifest_path.resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for label, expected in document.get("artifacts", {}).items():
        try:
            actual = hash_path(Path(expected["path"]))
            ok = all(actual[key] == expected[key] for key in ("sha256", "files", "bytes"))
            results.append({"artifact": label, "ok": ok, "expected": expected, "actual": actual})
        except Exception as error:
            results.append({"artifact": label, "ok": False, "expected": expected, "error": repr(error)})
    return all(item["ok"] for item in results), results
