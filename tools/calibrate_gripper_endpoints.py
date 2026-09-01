#!/usr/bin/env python3
"""Interactively record physical OpenArm gripper endpoints without commanding motion."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import time

import numpy as np
import openarm_driver


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "openarm_pedestal_vr.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "gripper_calibration.json"


def _sample_endpoint(
    arm: openarm_driver.SingleArmDriver,
    *,
    samples: int,
    interval: float,
    max_spread: float,
) -> tuple[float, list[float]]:
    readings = []
    for _ in range(samples):
        readings.append(float(arm.fetch_position(refresh=True)[-1]))
        time.sleep(interval)
    spread = float(np.ptp(readings))
    if spread > max_spread:
        raise RuntimeError(
            f"endpoint was not stable: spread={spread:.6f} rad, "
            f"allowed={max_spread:.6f} rad"
        )
    return float(np.median(readings)), readings


def _load_document(path: pathlib.Path) -> dict:
    if not path.exists():
        return {"version": 1}
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"invalid calibration document: {path}")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record manually positioned gripper open/closed encoder endpoints. "
            "This tool never enables a motor and never sends a motion command."
        )
    )
    parser.add_argument("--side", choices=("right", "left"), required=True)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    parser.add_argument("--max-spread", type=float, default=0.02)
    parser.add_argument("--min-span", type=float, default=0.05)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    arm_name = f"{args.side}_arm"
    arm = openarm_driver.SingleArmDriver(
        arm_name, openarm_driver.Config(config_path)
    )
    gripper_motor = arm.openarm.get_gripper().get_motors()[0]
    if gripper_motor.is_enabled():
        raise RuntimeError(
            f"{arm_name} gripper motor is enabled. Stop the Dora dataflow and "
            "disable the motor before manual calibration."
        )

    print(f"Side: {args.side}")
    print(f"Driver config: {config_path}")
    print("No motor will be enabled and no position command will be sent.")
    input("Manually open the gripper to its safe mechanical maximum, hold it, then press Enter: ")
    open_value, open_samples = _sample_endpoint(
        arm,
        samples=args.samples,
        interval=args.sample_interval,
        max_spread=args.max_spread,
    )
    print(
        f"OPEN: median={open_value:.6f} rad, "
        f"range=[{min(open_samples):.6f}, {max(open_samples):.6f}]"
    )

    input("Manually close the empty gripper to its safe mechanical limit, hold it, then press Enter: ")
    closed_value, closed_samples = _sample_endpoint(
        arm,
        samples=args.samples,
        interval=args.sample_interval,
        max_spread=args.max_spread,
    )
    print(
        f"CLOSED: median={closed_value:.6f} rad, "
        f"range=[{min(closed_samples):.6f}, {max(closed_samples):.6f}]"
    )

    span = abs(closed_value - open_value)
    if span < args.min_span:
        raise RuntimeError(
            f"measured span is too small: {span:.6f} rad; "
            f"minimum={args.min_span:.6f} rad"
        )

    confirmation = input(
        f"Write {args.side} endpoints open={open_value:.6f}, "
        f"closed={closed_value:.6f}? Type APPLY to confirm: "
    )
    if confirmation != "APPLY":
        print("Cancelled; no calibration file was changed.")
        return 1

    document = _load_document(output_path)
    document["version"] = 1
    document[args.side] = {"open": open_value, "closed": closed_value}
    metadata = document.setdefault("metadata", {})
    metadata[args.side] = {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(config_path),
        "samples": args.samples,
        "open_spread": float(np.ptp(open_samples)),
        "closed_spread": float(np.ptp(closed_samples)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = output_path.with_name(f"{output_path.name}.{stamp}.bak")
        shutil.copy2(output_path, backup)
        print(f"Backup: {backup}")
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Calibration saved: {output_path}")
    print("Restart dataflow-vr.yaml to load the measured endpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
