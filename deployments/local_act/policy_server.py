#!/usr/bin/env python3
"""Local LeRobot ACT/SmolVLA inference server; never opens robot hardware."""

from __future__ import annotations

import argparse
from multiprocessing.connection import Listener
from pathlib import Path
import time

import numpy as np
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedConfig

from representation_adapter import resolve_checkpoint


AUTHKEY = b"openarm-local-act-v1"


def load_policy(checkpoint: Path, device: str, adapter):
    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.device = device
    if config.type == "act":
        config.pretrained_backbone_weights = None
    elif config.type == "smolvla":
        # Full local checkpoint includes trained VLM weights. Build from the
        # local architecture/processor assets, then strictly restore weights;
        # never download or substitute the base VLM at robot startup.
        config.load_vlm_weights = False
        config.vlm_model_name = adapter.vlm_assets
    policy = get_policy_class(config.type).from_pretrained(
        str(checkpoint), config=config, strict=config.type == "smolvla"
    ).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, preprocessor, postprocessor


def infer(policy, preprocessor, postprocessor, request, num_actions: int, adapter):
    state = np.asarray(request["state"], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("state must be a finite 16-D vector")
    observation = {"observation.state": torch.from_numpy(adapter.model_state(state))}
    if adapter.task is not None:
        observation["task"] = adapter.task
    for source_key, key in adapter.camera_map.items():
        image = np.asarray(request["images"][source_key], dtype=np.uint8)
        if image.shape != (360, 640, 3):
            raise ValueError(f"{key} must have shape (360,640,3), got {image.shape}")
        observation[key] = (
            torch.from_numpy(np.array(image, copy=True))
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
        )

    started = time.perf_counter()
    with torch.inference_mode():
        raw = policy.predict_action_chunk(preprocessor(observation))
        raw = raw[:, :num_actions, :]
        actions = torch.stack(
            [postprocessor(raw[:, index, :]) for index in range(raw.shape[1])],
            dim=1,
        ).squeeze(0)
    if actions.shape != (num_actions, 16):
        raise ValueError(f"unexpected policy output shape: {tuple(actions.shape)}")
    result = actions.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("policy output contains NaN or Inf")
    result = adapter.controller_actions(result)
    return result, (time.perf_counter() - started) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--num-actions", type=int, default=85)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    adapter = resolve_checkpoint(args.checkpoint, args.deployment_manifest)
    if not 1 <= args.num_actions <= adapter.chunk_size:
        parser.error(f"--num-actions must be in [1,{adapter.chunk_size}]")
    print(
        f"[policy] checkpoint={args.checkpoint.resolve()} "
        f"type={adapter.policy_type} representation={adapter.representation}; "
        f"controller=right-first/radians task={adapter.task!r}",
        flush=True,
    )

    policy, preprocessor, postprocessor = load_policy(args.checkpoint, args.device, adapter)
    args.socket.unlink(missing_ok=True)
    listener = Listener(str(args.socket), family="AF_UNIX", authkey=AUTHKEY)
    print(f"[policy] ready on {args.socket}", flush=True)
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    if request.get("command") == "stop":
                        return
                    try:
                        actions, latency_ms = infer(
                            policy,
                            preprocessor,
                            postprocessor,
                            request,
                            args.num_actions,
                            adapter,
                        )
                        connection.send(
                            {"ok": True, "actions": actions, "latency_ms": latency_ms}
                        )
                    except Exception as exc:
                        connection.send({"ok": False, "error": repr(exc)})
            except EOFError:
                pass
            finally:
                connection.close()
    finally:
        listener.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
