"""Single-command operator workflow for OpenArm T-shirt folding."""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

import yaml

from .baseline import freeze as freeze_baseline
from .baseline import require_mutable, verify as verify_baseline
from .raw_dataset import load_records, representative_images, summarize
from .representation import (
    LEGACY_REPRESENTATION,
    LEROBOT_OPENARM_REPRESENTATION,
    REPRESENTATIONS,
)
from .visualize import build_episode_player


ROOT = Path(__file__).resolve().parent.parent
STATE_ROOT = ROOT / ".openarm-fold"
SESSION_FILE = STATE_ROOT / "session.yaml"
RAW_ROOT = ROOT / "folding_data" / "dataset"
SESSION_DATA_ROOT = ROOT / "folding_data" / "sessions"
RUNTIME_DATAFLOW = ROOT / ".openarm-fold-session-dataflow.yaml"
REPORT_ROOT = ROOT / "reports" / "folding"
WORKSPACE_ROOT = Path(os.environ.get("OPENARM_WORKSPACE", str(ROOT.parent)))
LEROBOT_ROOT = Path(os.environ.get("OPENARM_LEROBOT_ROOT", str(WORKSPACE_ROOT / "lerobot")))
TRAINING_PYTHON = Path(os.environ.get("OPENARM_POLICY_PYTHON", str(WORKSPACE_ROOT / ".venv/bin/python")))
TRAINING_COMMAND = TRAINING_PYTHON.parent / "lerobot-train"
PINNED_LEROBOT_VERSION = "0.6.2"
PINNED_LEROBOT_REVISION = "fbb811fca92504439792b97d216f0d00c2268382"
PROFILE_PATH = ROOT / "profiles" / "act-fold.yaml"
BASELINE_NAME = "act-tshirt-fold-rad-v1"
BASELINE_MANIFEST = ROOT / "baselines" / BASELINE_NAME / "manifest.json"
BASELINE_CHECKPOINT = (
    ROOT / "deployments" / "act" / "act-tshirt-fold-rad-v1-step100000-lerobot062-local"
)
FAILURE_REASONS = (
    "bad_demonstration",
    "grasp_failure",
    "incomplete_fold",
    "unsafe_motion",
    "hardware_or_sensor",
    "bad_initial_setup",
    "timeout",
    "other",
)


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


def _new_session_dataset_root(session_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", session_id):
        raise SystemExit(
            "--session-id must contain only letters, numbers, '.', '_' or '-', "
            "and must start with a letter or number"
        )
    return SESSION_DATA_ROOT / session_id / "dataset"


def _session_dataset_root(session: dict) -> Path:
    configured = session.get("dataset_root")
    if not configured:
        return RAW_ROOT
    path = Path(configured)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(SESSION_DATA_ROOT.resolve()):
        raise SystemExit(f"Session dataset root is outside {SESSION_DATA_ROOT}: {path}")
    return path


def _selected_raw_root(raw_root: Path | None) -> Path:
    if raw_root is not None:
        return raw_root if raw_root.is_absolute() else ROOT / raw_root
    return _session_dataset_root(_read_yaml(SESSION_FILE))


def _write_runtime_dataflow(dataset_root: Path) -> Path:
    descriptor = _read_yaml(ROOT / "dataflow-vr.yaml")
    nodes = {node.get("id"): node for node in descriptor.get("nodes", [])}
    try:
        ui_environment = nodes["ui"].setdefault("env", {})
        recorder_environment = nodes["recorder"].setdefault("env", {})
    except KeyError as error:
        raise SystemExit(f"Missing node in dataflow-vr.yaml: {error.args[0]}") from error
    ui_environment["DATASET_DIRECTORY"] = str(dataset_root)
    recorder_environment["DIRECTORY"] = str(dataset_root.parent)
    recorder_environment["NAME"] = dataset_root.name
    sdk = os.environ.get("HIKROBOT_MV3D_LIB")
    for node in nodes.values():
        environment = node.get("env", {})
        if sdk and "HIKROBOT_MV3D_LIB" in environment:
            environment["HIKROBOT_MV3D_LIB"] = sdk
            environment["LD_LIBRARY_PATH"] = sdk + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    for node_id, variable in (
        ("camera-wrist-right", "OPENARM_CAMERA_RIGHT_SERIAL"),
        ("camera-wrist-left", "OPENARM_CAMERA_LEFT_SERIAL"),
        ("camera-ceiling", "OPENARM_CAMERA_BASE_SERIAL"),
    ):
        if os.environ.get(variable) and node_id in nodes:
            nodes[node_id].setdefault("env", {})["SERIAL_NUMBER"] = os.environ[variable]
    _atomic_yaml(RUNTIME_DATAFLOW, descriptor)
    return RUNTIME_DATAFLOW


def _sha256(path: Path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout or result.stderr).strip()


def _lerobot_runtime_info() -> tuple[bool, dict | str]:
    if not TRAINING_PYTHON.exists():
        return False, f"missing Python: {TRAINING_PYTHON}"
    script = (
        "import json, lerobot; "
        "print(json.dumps({'version': getattr(lerobot, '__version__', None), "
        "'module': str(lerobot.__file__)}))"
    )
    result = subprocess.run(
        [str(TRAINING_PYTHON), "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()
    try:
        return True, json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, f"unexpected runtime response: {result.stdout.strip()}"


def _default_repo_id(dataset_split: str, representation: str) -> str:
    if representation == LEROBOT_OPENARM_REPRESENTATION:
        return f"local/openarm-tshirt-fold-lerobot06-{dataset_split}"
    if dataset_split == "train":
        return "local/openarm-tshirt-fold"
    return f"local/openarm-tshirt-fold-{dataset_split}"


def _dataset_manifest(dataset_root: Path) -> dict:
    return _read_yaml(dataset_root / "openarm_conversion_manifest.json")


def command_preflight(args):
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    if not args.training:
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
        runtime_ok, runtime = _lerobot_runtime_info()
        add("LeRobot import", runtime_ok, runtime)
        if runtime_ok:
            version = runtime.get("version")
            module_path = Path(runtime.get("module", "")).resolve()
            add(
                "LeRobot version",
                version == PINNED_LEROBOT_VERSION,
                f"{version} (expected {PINNED_LEROBOT_VERSION})",
            )
            add(
                "LeRobot source",
                module_path.is_relative_to(LEROBOT_ROOT.resolve()),
                module_path,
            )
        revision_status, revision = _git_output(LEROBOT_ROOT, "rev-parse", "HEAD")
        add(
            "LeRobot revision",
            revision_status == 0 and revision == PINNED_LEROBOT_REVISION,
            f"{revision or 'unknown'} (expected {PINNED_LEROBOT_REVISION})",
        )
        worktree_status, worktree = _git_output(
            LEROBOT_ROOT, "status", "--porcelain", "--untracked-files=no"
        )
        add(
            "LeRobot worktree",
            worktree_status == 0 and not worktree,
            "clean" if worktree_status == 0 and not worktree else worktree or "git error",
        )
        add("LeRobot uv.lock", (LEROBOT_ROOT / "uv.lock").exists(), LEROBOT_ROOT / "uv.lock")
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
        dataset_root = _new_session_dataset_root(args.session_id)
        if dataset_root.exists():
            raise SystemExit(
                f"Session dataset already exists: {dataset_root}. "
                "Use --resume or choose a new --session-id."
            )
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
            "dataset_root": str(dataset_root.relative_to(ROOT)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "park": {"approved": False, "profile": None},
        }
        _atomic_yaml(SESSION_FILE, session)

    dataset_root = _session_dataset_root(session)
    if args.resume:
        require_mutable(dataset_root, "resume collection")

    print(f"Session: {session['collection_session_id']}")
    print(f"Garment: {session['garment_id']} ({session['dataset_split']})")
    print(f"Dataset: {dataset_root}")
    print("Monitor: http://127.0.0.1:8000")
    print("Arms start paused. Passing preflight does not authorize motion.")
    if args.prepare_only:
        return 0
    runtime_dataflow = _write_runtime_dataflow(dataset_root)
    command = ["dora", "run", str(runtime_dataflow), "--uv"]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _review_html(
    raw_root: Path,
    player_links: dict[int, str],
    *,
    csrf_token: str | None = None,
    message: str | None = None,
):
    records = load_records(raw_root)
    cards = []
    for record in records:
        images = representative_images(record)
        image_html = "".join(
            f'<img src="{html.escape(os.path.relpath(path, REPORT_ROOT))}" alt="episode {record.episode_id}">'
            for path in images
        )
        duration = "unknown" if record.duration_s is None else f"{record.duration_s:.1f}s"
        player_link = player_links.get(record.episode_id)
        video_html = (
            f'<a class="video" href="{html.escape(player_link)}" target="_blank">'
            "Open synchronized video</a>"
            if player_link
            else '<span class="unavailable">No camera video available</span>'
        )
        if csrf_token is None:
            relabel_html = ""
        else:
            options = "".join(
                f'<option value="{reason}">{reason}</option>'
                for reason in FAILURE_REASONS
            )
            relabel_html = f"""
              <form class="relabel" action="/relabel" method="post"
                    onsubmit="return confirm('Change the label for episode {record.episode_id}?')">
                <input type="hidden" name="csrf_token" value="{html.escape(csrf_token)}">
                <input type="hidden" name="episode" value="{record.episode_id}">
                <select name="failure_reason" aria-label="Failure reason for episode {record.episode_id}">
                  {options}
                </select>
                <button name="status" value="failed">Mark failed</button>
                <button class="accept" name="status" value="accepted">Mark accepted</button>
              </form>
            """
        cards.append(
            f"""
            <article id="episode-{record.episode_id}" class="{'accepted' if record.success else 'failed'}">
              <h2>Episode {record.episode_id}</h2>
              <p>{'accepted' if record.success else html.escape(record.failure_reason or 'failed')} ·
                 {html.escape(record.garment_id)} · {duration}</p>
              <p>{video_html}</p>
              {relabel_html}
              <div>{image_html or '<em>No ceiling frames</em>'}</div>
            </article>
            """
        )
    notice = f'<p class="notice">{html.escape(message)}</p>' if message else ""
    instructions = (
        "<p>Open a synchronized video, choose a failure reason, and confirm the label change. "
        "The collector must be stopped.</p>"
        if csrf_token is not None
        else """
        <p>Review the full synchronized video before changing a label. Stop the collector first, then run:</p>
        <code>./openarm-fold relabel --episode ID --status failed --failure-reason bad_demonstration</code>
        <code>./openarm-fold relabel --episode ID --status accepted</code>
        """
    )
    return f"""<!doctype html>
    <meta charset="utf-8"><title>OpenArm folding review</title>
    <style>
      body{{font-family:system-ui;background:#071019;color:#edf7ff;margin:24px}}
      article{{border:1px solid #294052;border-left:8px solid #ff5c70;padding:14px;margin:14px 0;background:#101d29}}
      article.accepted{{border-left-color:#39e58c}} img{{width:31%;margin-right:1%;vertical-align:top}}
      p{{color:#91a6b7}} a.video{{display:inline-block;padding:8px 12px;border-radius:7px;background:#183044;color:#edf7ff;text-decoration:none}}
      code{{display:block;margin:7px 0;padding:9px;background:#071019;color:#45d7ff;white-space:pre-wrap}}
      .unavailable{{color:#ff5c70}}
      .notice{{padding:12px;border-left:5px solid #45d7ff;background:#101d29;color:#edf7ff}}
      form.relabel{{display:flex;gap:8px;align-items:center;margin:10px 0 15px}}
      select,button{{padding:8px 11px;border:1px solid #36556d;border-radius:7px;background:#183044;color:#edf7ff}}
      button{{background:#5c1a28;border-color:#ff5c70;cursor:pointer}}
      button.accept{{background:#0a6840;border-color:#39e58c}}
    </style>
    <h1>OpenArm T-shirt Folding Review</h1>
    {notice}{instructions}
    {''.join(cards)}"""


def command_review(args):
    raw_root = _selected_raw_root(args.raw_root)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    player_links = {}
    records = load_records(raw_root)
    for record in records:
        try:
            player, _, _ = build_episode_player(
                raw_root, REPORT_ROOT, record.episode_id
            )
        except ValueError:
            continue
        player_links[record.episode_id] = os.path.relpath(player, REPORT_ROOT)
    output = REPORT_ROOT / "episodes.html"
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(_review_html(raw_root, player_links), encoding="utf-8")
    os.replace(temporary, output)
    print(f"Built synchronized players for {len(player_links)} of {len(records)} episodes")
    print(output)
    if args.open and not args.serve:
        raise SystemExit("--open requires --serve for the interactive review tool")
    if args.serve:
        from .review_server import serve_review

        serve_review(
            root=ROOT,
            raw_root=raw_root,
            report_root=REPORT_ROOT,
            player_links=player_links,
            render_page=_review_html,
            relabel_episode=_relabel_episode,
            port=args.port,
            open_browser=args.open,
        )
    return 0


def _collection_ui_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", 8000)) == 0


def _relabel_episode(
    raw_root: Path,
    episode_number: int,
    status: str,
    failure_reason: str | None,
) -> str:
    require_mutable(raw_root, "relabel episodes")
    if _collection_ui_running():
        raise SystemExit(
            "Collection UI is still running on port 8000. Quit collection before "
            "relabeling so the recorder cannot overwrite the corrected metadata."
        )

    episode_manifest_path = raw_root / "episodes" / str(episode_number) / "episode.yaml"
    metadata_path = raw_root / "metadata.yaml"
    if not episode_manifest_path.exists():
        raise SystemExit(f"Episode {episode_number} does not exist at {episode_manifest_path}")
    if not metadata_path.exists():
        raise SystemExit(f"Dataset metadata does not exist at {metadata_path}")

    if status not in {"accepted", "failed"}:
        raise SystemExit(f"Unsupported status: {status}")
    if failure_reason is not None and failure_reason not in FAILURE_REASONS:
        raise SystemExit(f"Unsupported failure reason: {failure_reason}")
    if status == "accepted":
        if failure_reason is not None:
            raise SystemExit("--failure-reason cannot be used with --status accepted")
        success = True
        failure_reason = None
    else:
        if failure_reason is None:
            raise SystemExit("--status failed requires --failure-reason")
        success = False

    episode_manifest = _read_yaml(episode_manifest_path)
    metadata = _read_yaml(metadata_path)
    matching_records = [
        record
        for record in metadata.get("episodes", [])
        if str(record.get("id")) == str(episode_number)
    ]
    if len(matching_records) != 1:
        raise SystemExit(
            f"Expected one metadata record for episode {episode_number}, found "
            f"{len(matching_records)}; no files changed"
        )

    previous_status = episode_manifest.get("status", "unknown")
    previous_reason = episode_manifest.get("failure_reason")
    for record in (episode_manifest, matching_records[0]):
        record["status"] = status
        record["success"] = success
        record["failure_reason"] = failure_reason

    _atomic_yaml(episode_manifest_path, episode_manifest)
    _atomic_yaml(metadata_path, metadata)
    previous = previous_status
    if previous_reason:
        previous += f" ({previous_reason})"
    current = status
    if failure_reason:
        current += f" ({failure_reason})"
    return f"Episode {episode_number}: {previous} -> {current}"


def command_relabel(args):
    raw_root = _selected_raw_root(args.raw_root)
    message = _relabel_episode(
        raw_root, args.episode, args.status, args.failure_reason
    )
    print(message)
    print("Raw camera and robot files were preserved")
    return 0


def command_visualize(args):
    raw_root = _selected_raw_root(args.raw_root)
    try:
        output, frame_count, fps = build_episode_player(
            raw_root, REPORT_ROOT, args.episode
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Episode {args.episode}: {frame_count} synchronized frames at {fps:.2f} FPS")
    print(output)
    if args.open:
        opener = shutil.which("xdg-open")
        if opener is None:
            raise SystemExit(f"xdg-open not found; open {output} in a browser")
        subprocess.Popen(
            [opener, str(output)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    return 0


def command_report(args):
    raw_root = _selected_raw_root(args.raw_root)
    report = summarize(raw_root)
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
    raw_root = _selected_raw_root(args.raw_root)
    python = args.python or TRAINING_PYTHON
    if not python.exists():
        raise SystemExit(
            f"Pinned training Python not found: {python}. Follow TRAINING.md first."
        )
    repo_id = args.repo_id or _default_repo_id(args.dataset_split, args.representation)
    command = [
        str(python),
        "-m",
        "folding_workflow.convert",
        "--raw-root",
        str(raw_root),
        "--output-root",
        str(args.output_root),
        "--repo-id",
        repo_id,
        "--split",
        args.dataset_split,
        "--maximum-camera-skew-ms",
        str(args.maximum_camera_skew_ms),
        "--representation",
        args.representation,
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def command_verify_export(args):
    raw_root = _selected_raw_root(args.raw_root)
    command = [
        str(TRAINING_PYTHON),
        "-m",
        "folding_workflow.verify_export",
        "--raw-root",
        str(raw_root),
        "--dataset-root",
        str(args.dataset_root),
        "--split",
        args.dataset_split,
    ]
    if args.expected_episodes is not None:
        command.extend(["--expected-episodes", str(args.expected_episodes)])
    if args.expected_frames is not None:
        command.extend(["--expected-frames", str(args.expected_frames)])
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def command_freeze_baseline(args):
    raw_root = _selected_raw_root(args.raw_root).resolve()
    manifest_path = (args.manifest or BASELINE_MANIFEST).resolve()
    artifacts = {
        "reviewed_raw_dataset": raw_root,
        "legacy_lerobot_train": (ROOT / "derived/lerobot/openarm-tshirt-fold-train").resolve(),
        "legacy_lerobot_validation": (
            ROOT / "derived/lerobot/openarm-tshirt-fold-validation"
        ).resolve(),
        "migrated_act_100k": BASELINE_CHECKPOINT.resolve(),
        "local_act_deployment": (ROOT / "deployments/local_act").resolve(),
        "pedestal_configuration": (ROOT / "openarm_pedestal_vr.yaml").resolve(),
        "gripper_configuration": (ROOT / "gripper_calibration.json").resolve(),
    }
    missing = [f"{label}: {path}" for label, path in artifacts.items() if not path.exists()]
    if missing:
        raise SystemExit("Cannot freeze baseline; missing artifacts:\n" + "\n".join(missing))
    _, repository_revision = _git_output(ROOT, "rev-parse", "HEAD")
    _, lerobot_revision = _git_output(LEROBOT_ROOT, "rev-parse", "HEAD")
    runtime_ok, runtime = _lerobot_runtime_info()
    document = freeze_baseline(
        name=args.name,
        raw_root=raw_root,
        manifest_path=manifest_path,
        artifacts=artifacts,
        metadata={
            "repository_revision": repository_revision or None,
            "lerobot_revision": lerobot_revision or None,
            "lerobot_runtime": runtime if runtime_ok else {"error": runtime},
            "note": "Application-level guard; source artifacts are not duplicated.",
        },
    )
    print(f"Frozen baseline: {document['name']}")
    print(manifest_path)
    return 0


def command_verify_baseline(args):
    manifest_path = (args.manifest or BASELINE_MANIFEST).resolve()
    ok, results = verify_baseline(manifest_path)
    for result in results:
        marker = "OK" if result["ok"] else "MISMATCH"
        detail = result.get("actual", {}).get("sha256") or result.get("error")
        print(f"[{marker:8}] {result['artifact']}: {detail}")
    return 0 if ok else 2


def command_train(args):
    readiness = argparse.Namespace(training=True, json=False)
    if command_preflight(readiness) != 0:
        raise SystemExit("Training blocked by preflight")
    profile = _read_yaml(PROFILE_PATH)
    dataset_manifest = _dataset_manifest(args.dataset_root)
    representation = dataset_manifest.get("representation")
    expected_representation = profile.get("dataset", {}).get("representation")
    if representation != expected_representation and not args.allow_legacy_representation:
        raise SystemExit(
            f"Dataset representation is {representation or 'unspecified'}, expected "
            f"{expected_representation}. Pass --allow-legacy-representation only to "
            "reproduce a historical run."
        )
    chunk_size = args.chunk_size or profile.get("policy", {}).get("chunk_size")
    if not chunk_size:
        raise SystemExit(
            "ACT chunk size is intentionally unset until the first 25 accepted episodes are profiled"
        )
    output_dir = args.output_dir or ROOT / "outputs" / "act-tshirt-fold-lerobot06"
    if output_dir.exists():
        raise SystemExit(
            f"Training output already exists: {output_dir}. "
            "Choose a new --output-dir; existing runs are never overwritten."
        )
    repo_id = args.repo_id or dataset_manifest.get("repo_id")
    if not repo_id:
        raise SystemExit("Dataset repo_id is missing; pass --repo-id explicitly")
    command = [
        str(TRAINING_COMMAND),
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={args.dataset_root}",
        "--policy.type=act",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--policy.chunk_size={chunk_size}",
        f"--policy.n_action_steps={chunk_size}",
        f"--steps={profile.get('steps', 100000)}",
        f"--batch_size={profile.get('batch_size', 8)}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        "--wandb.enable=false",
    ]
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
        "lerobot_root": str(LEROBOT_ROOT),
        "lerobot_revision": PINNED_LEROBOT_REVISION,
        "lerobot_version": PINNED_LEROBOT_VERSION,
        "lerobot_uv_lock_sha256": _sha256(LEROBOT_ROOT / "uv.lock"),
        "training_python": str(TRAINING_PYTHON),
        "dataset_representation": representation,
        "repo_id": repo_id,
        "profile": profile,
        "resolved_chunk_size": chunk_size,
        "command": command,
        "deployment_authorized": False,
    }
    staged_manifest = output_dir.with_name(
        f".{output_dir.name}.openarm_training_manifest.json"
    )
    _atomic_json(staged_manifest, run_manifest)
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False)
    finally:
        if output_dir.exists() and staged_manifest.exists():
            os.replace(staged_manifest, output_dir / "openarm_training_manifest.json")
    return completed.returncode


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    preflight = subcommands.add_parser("preflight", help="Read-only readiness checks")
    preflight.add_argument("--training", action="store_true")
    preflight.add_argument("--json", action="store_true")
    preflight.set_defaults(handler=command_preflight)

    freeze_parser = subcommands.add_parser(
        "freeze-baseline", help="Hash and guard the reviewed ACT/radian baseline"
    )
    freeze_parser.add_argument("--name", default=BASELINE_NAME)
    freeze_parser.add_argument("--raw-root", type=Path)
    freeze_parser.add_argument("--manifest", type=Path)
    freeze_parser.set_defaults(handler=command_freeze_baseline)

    verify_parser = subcommands.add_parser(
        "verify-baseline", help="Recompute and compare frozen baseline tree hashes"
    )
    verify_parser.add_argument("--manifest", type=Path)
    verify_parser.set_defaults(handler=command_verify_baseline)

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

    review = subcommands.add_parser(
        "review", help="Build a batch review page with synchronized video players"
    )
    review.add_argument("--raw-root", type=Path)
    review.add_argument(
        "--serve", action="store_true", help="Serve interactive review on localhost"
    )
    review.add_argument(
        "--open", action="store_true", help="Open the interactive review in a browser"
    )
    review.add_argument("--port", type=int, default=8001)
    review.set_defaults(handler=command_review)

    relabel = subcommands.add_parser(
        "relabel", help="Change one finalized episode label without deleting raw data"
    )
    relabel.add_argument("--episode", type=int, required=True)
    relabel.add_argument("--raw-root", type=Path)
    relabel.add_argument("--status", choices=("accepted", "failed"), required=True)
    relabel.add_argument("--failure-reason", choices=FAILURE_REASONS)
    relabel.set_defaults(handler=command_relabel)

    visualize = subcommands.add_parser(
        "visualize", help="Play one episode as synchronized three-camera video"
    )
    visualize.add_argument("--episode", type=int, required=True)
    visualize.add_argument("--raw-root", type=Path)
    visualize.add_argument("--open", action="store_true", help="Open the player in a browser")
    visualize.set_defaults(handler=command_visualize)

    report = subcommands.add_parser("report", help="Write QC, split, and duration report")
    report.add_argument("--raw-root", type=Path)
    report.add_argument("--output", type=Path)
    report.add_argument("--checkpoint", type=Path)
    report.add_argument("--validation-root", type=Path)
    report.add_argument(
        "--validation-repo-id",
        default="local/openarm-tshirt-fold-lerobot06-validation",
    )
    report.add_argument("--batch-size", type=int, default=8)
    report.set_defaults(handler=command_report)

    convert = subcommands.add_parser("convert", help="Export one LeRobot v3 split")
    convert.add_argument("--raw-root", type=Path)
    convert.add_argument("--output-root", type=Path, required=True)
    convert.add_argument("--repo-id")
    convert.add_argument(
        "--dataset-split",
        choices=("train", "validation", "evaluation"),
        default="train",
    )
    convert.add_argument("--maximum-camera-skew-ms", type=float, default=30.0)
    convert.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default=LEROBOT_OPENARM_REPRESENTATION,
    )
    convert.add_argument("--python", type=Path)
    convert.set_defaults(handler=command_convert)

    verify_export = subcommands.add_parser(
        "verify-export", help="Compare one LeRobot export with its reviewed raw source"
    )
    verify_export.add_argument("--raw-root", type=Path)
    verify_export.add_argument("--dataset-root", type=Path, required=True)
    verify_export.add_argument(
        "--dataset-split",
        choices=("train", "validation", "evaluation"),
        required=True,
    )
    verify_export.add_argument("--expected-episodes", type=int)
    verify_export.add_argument("--expected-frames", type=int)
    verify_export.set_defaults(handler=command_verify_export)

    train = subcommands.add_parser("train", help="Launch pinned ACT training")
    train.add_argument("--dataset-root", type=Path, required=True)
    train.add_argument("--repo-id")
    train.add_argument("--chunk-size", type=int)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--job-name", default="act-openarm-tshirt-fold-lerobot06")
    train.add_argument("--allow-legacy-representation", action="store_true")
    train.set_defaults(handler=command_train)
    return parser


def main():
    args = build_parser().parse_args()
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
