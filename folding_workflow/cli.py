"""Single-command operator workflow for OpenArm T-shirt folding."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from .raw_dataset import load_records, representative_images, summarize


ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = ROOT / ".openarm-fold"
SESSION_FILE = STATE_ROOT / "session.yaml"
RAW_ROOT = ROOT / "folding_data" / "dataset"
REPORT_ROOT = ROOT / "reports" / "folding"
TRAINING_PYTHON = ROOT / ".venv-lerobot" / "bin" / "python"
TRAINING_COMMAND = ROOT / ".venv-lerobot" / "bin" / "lerobot-train"
PROFILE_PATH = ROOT / "profiles" / "act-fold.yaml"


def _atomic_yaml(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        yaml.safe_dump(value, output, sort_keys=False, allow_unicode=True)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _read_yaml(path: Path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as source:
        return yaml.safe_load(source) or {}


def _format_bytes(value):
    return f"{value / 1024**3:.1f} GiB"


def _sha256(path: Path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_preflight(args):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    for path in (
        ROOT / "dataflow-vr.yaml",
        ROOT / "metadata.yaml",
        ROOT / "gripper_calibration.json",
        ROOT / "openarm_pedestal_vr.yaml",
    ):
        add(path.name, path.exists(), path)
    add("dora", shutil.which("dora") is not None, shutil.which("dora") or "not found")
    add("session", SESSION_FILE.exists(), SESSION_FILE)
    disk = shutil.disk_usage(ROOT)
    add("disk", disk.free >= 50 * 1024**3, f"{_format_bytes(disk.free)} free")

    if args.training:
        add("training environment", TRAINING_COMMAND.exists(), TRAINING_COMMAND)
        add("NVIDIA device", Path("/dev/nvidia0").exists(), "/dev/nvidia0")
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi:
            nvidia = subprocess.run(
                [nvidia_smi, "-L"],
                capture_output=True,
                text=True,
                check=False,
            )
            detail = (nvidia.stdout or nvidia.stderr).strip()
            add("CUDA driver", nvidia.returncode == 0, detail)
        else:
            add("CUDA driver", False, "nvidia-smi not found")

    result = {"ok": all(check["ok"] for check in checks), "checks": checks}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for check in checks:
            marker = "OK" if check["ok"] else "BLOCK"
            print(f"[{marker:5}] {check['name']}: {check['detail']}")
    return 0 if result["ok"] else 2


def command_collect(args):
    existing = _read_yaml(SESSION_FILE)
    if args.resume:
        if not existing:
            raise SystemExit(f"No session to resume at {SESSION_FILE}")
        session = existing
    else:
        missing = [
            name
            for name, value in (
                ("--session-id", args.session_id),
                ("--garment-id", args.garment_id),
                ("--operator-id", args.operator_id),
            )
            if not value
        ]
        if missing:
            raise SystemExit("New collection session requires " + ", ".join(missing))
        session = {
            "collection_session_id": args.session_id,
            "operator_id": args.operator_id,
            "dataset_split": args.dataset_split,
            "garment_id": args.garment_id,
            "garment": {
                "size": args.size,
                "material": args.material,
                "color": args.color,
                "notes": args.notes,
            },
            "initial_pose_id": args.initial_pose_id,
            "target_accepted_episodes": args.target,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "park": {"approved": False, "profile": None},
        }
        _atomic_yaml(SESSION_FILE, session)

    print(f"Session: {session['collection_session_id']}")
    print(f"Garment: {session['garment_id']} ({session['dataset_split']})")
    print("Monitor: http://127.0.0.1:8000")
    print("Arms start paused. Passing preflight does not authorize motion.")
    if args.prepare_only:
        return 0
    command = ["dora", "run", "dataflow-vr.yaml", "--uv"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _review_html(raw_root: Path):
    records = load_records(raw_root)
    cards = []
    for record in records:
        images = representative_images(record)
        image_html = "".join(
            f'<img src="{html.escape(os.path.relpath(path, REPORT_ROOT))}" alt="episode {record.episode_id}">'
            for path in images
        )
        duration = "unknown" if record.duration_s is None else f"{record.duration_s:.1f}s"
        cards.append(
            f"""
            <article class="{'accepted' if record.success else 'failed'}">
              <h2>Episode {record.episode_id}</h2>
              <p>{'accepted' if record.success else html.escape(record.failure_reason or 'failed')} ·
                 {html.escape(record.garment_id)} · {duration}</p>
              <div>{image_html or '<em>No ceiling frames</em>'}</div>
            </article>
            """
        )
    return f"""<!doctype html>
    <meta charset="utf-8"><title>OpenArm folding review</title>
    <style>
      body{{font-family:system-ui;background:#071019;color:#edf7ff;margin:24px}}
      article{{border:1px solid #294052;border-left:8px solid #ff5c70;padding:14px;margin:14px 0;background:#101d29}}
      article.accepted{{border-left-color:#39e58c}} img{{width:31%;margin-right:1%;vertical-align:top}}
      p{{color:#91a6b7}}
    </style>
    <h1>OpenArm T-shirt Folding Review</h1>{''.join(cards)}"""


def command_review(args):
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    output = REPORT_ROOT / "episodes.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(_review_html(args.raw_root), encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 0


def command_report(args):
    report = summarize(args.raw_root)
    durations = report.get("durations", {})
    if report["accepted"] >= 25 and durations.get("median_s"):
        candidate = round(durations["median_s"] * 30 / 20)
        report["act_chunk_profile"] = {
            "status": "candidate_after_25_episode_gate",
            "candidate_steps": min(100, max(30, candidate)),
            "heuristic": "approximately five percent of the median episode at 30 Hz",
            "requires_explicit_profile_update": True,
        }
        report["soft_duration_warning_s"] = durations.get("p95_s")
    else:
        report["act_chunk_profile"] = {
            "status": "deferred_until_25_accepted_episodes"
        }
    output = args.output or REPORT_ROOT / "dataset-report.json"
    _atomic_json(output, report)
    print(output)
    if not args.checkpoint:
        return 0
    if not args.validation_root:
        raise SystemExit("--checkpoint requires --validation-root")
    if not TRAINING_PYTHON.exists():
        raise SystemExit(f"Pinned training Python not found: {TRAINING_PYTHON}")
    policy_output = REPORT_ROOT / "offline-policy-report.json"
    command = [
        str(TRAINING_PYTHON),
        "-m",
        "folding_workflow.evaluate",
        "--checkpoint",
        str(args.checkpoint),
        "--dataset-root",
        str(args.validation_root),
        "--repo-id",
        args.validation_repo_id,
        "--output",
        str(policy_output),
        "--batch-size",
        str(args.batch_size),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def command_convert(args):
    python = args.python or TRAINING_PYTHON
    if not python.exists():
        raise SystemExit(
            f"Pinned training Python not found: {python}. Follow TRAINING.md first."
        )
    command = [
        str(python),
        "-m",
        "folding_workflow.convert",
        "--raw-root",
        str(args.raw_root),
        "--output-root",
        str(args.output_root),
        "--repo-id",
        args.repo_id,
        "--split",
        args.dataset_split,
        "--maximum-camera-skew-ms",
        str(args.maximum_camera_skew_ms),
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def command_train(args):
    readiness = argparse.Namespace(training=True, json=False)
    if command_preflight(readiness) != 0:
        raise SystemExit("Training blocked by preflight")
    profile = _read_yaml(PROFILE_PATH)
    chunk_size = args.chunk_size or profile.get("policy", {}).get("chunk_size")
    if not chunk_size:
        raise SystemExit(
            "ACT chunk size is intentionally unset until the first 25 accepted episodes are profiled"
        )
    output_dir = args.output_dir or ROOT / "outputs" / "act-tshirt-fold"
    command = [
        str(TRAINING_COMMAND),
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={args.dataset_root}",
        "--policy.type=act",
        "--policy.device=cuda",
        f"--policy.chunk_size={chunk_size}",
        f"--policy.n_action_steps={chunk_size}",
        f"--steps={profile.get('steps', 100000)}",
        f"--batch_size={profile.get('batch_size', 8)}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        "--wandb.enable=false",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision or None,
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_conversion_manifest_sha256": _sha256(
            args.dataset_root / "openarm_conversion_manifest.json"
        ),
        "requirements_lock_sha256": _sha256(ROOT / "requirements-lerobot.lock.txt"),
        "repo_id": args.repo_id,
        "profile": profile,
        "resolved_chunk_size": chunk_size,
        "command": command,
        "deployment_authorized": False,
    }
    _atomic_json(output_dir / "openarm_training_manifest.json", run_manifest)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    preflight = subcommands.add_parser("preflight", help="Read-only readiness checks")
    preflight.add_argument("--training", action="store_true")
    preflight.add_argument("--json", action="store_true")
    preflight.set_defaults(handler=command_preflight)

    collect = subcommands.add_parser("collect", help="Create/resume a session and run Dora")
    collect.add_argument("--resume", action="store_true")
    collect.add_argument("--prepare-only", action="store_true")
    collect.add_argument("--session-id")
    collect.add_argument("--operator-id")
    collect.add_argument("--garment-id")
    collect.add_argument("--dataset-split", choices=("train", "evaluation"), default="train")
    collect.add_argument("--size", default="unknown")
    collect.add_argument("--material", default="unknown")
    collect.add_argument("--color", default="unknown")
    collect.add_argument("--notes", default="")
    collect.add_argument("--initial-pose-id", default="flat-a")
    collect.add_argument("--target", type=int, default=25)
    collect.set_defaults(handler=command_collect)

    review = subcommands.add_parser("review", help="Build a batch review page")
    review.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    review.set_defaults(handler=command_review)

    report = subcommands.add_parser("report", help="Write QC, split, and duration report")
    report.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    report.add_argument("--output", type=Path)
    report.add_argument("--checkpoint", type=Path)
    report.add_argument("--validation-root", type=Path)
    report.add_argument(
        "--validation-repo-id", default="local/openarm-tshirt-fold-validation"
    )
    report.add_argument("--batch-size", type=int, default=8)
    report.set_defaults(handler=command_report)

    convert = subcommands.add_parser("convert", help="Export one LeRobot v3 split")
    convert.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    convert.add_argument("--output-root", type=Path, required=True)
    convert.add_argument("--repo-id", default="local/openarm-tshirt-fold")
    convert.add_argument("--dataset-split", choices=("train", "validation"), default="train")
    convert.add_argument("--maximum-camera-skew-ms", type=float, default=25.0)
    convert.add_argument("--python", type=Path)
    convert.set_defaults(handler=command_convert)

    train = subcommands.add_parser("train", help="Launch pinned ACT training")
    train.add_argument("--dataset-root", type=Path, required=True)
    train.add_argument("--repo-id", default="local/openarm-tshirt-fold")
    train.add_argument("--chunk-size", type=int)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--job-name", default="act-openarm-tshirt-fold")
    train.set_defaults(handler=command_train)
    return parser


def main():
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
