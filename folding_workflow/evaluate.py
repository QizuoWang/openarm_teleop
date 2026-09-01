"""Compute teacher-forced ACT loss on a complete-episode validation dataset."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


def evaluate(checkpoint: Path, dataset_root: Path, repo_id: str, output: Path, batch_size: int):
    policy = ACTPolicy.from_pretrained(str(checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    policy.eval()

    delta_timestamps = {
        "action": [index / 30 for index in policy.config.action_delta_indices]
    }
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    weighted_loss = 0.0
    samples = 0
    started_ns = time.time_ns()
    with torch.no_grad():
        for batch in loader:
            processed = preprocessor(batch)
            loss, _ = policy.forward(processed)
            batch_samples = int(processed["action"].shape[0])
            weighted_loss += float(loss.item()) * batch_samples
            samples += batch_samples

    result = {
        "evaluation_type": "teacher_forced_validation_loss",
        "checkpoint": str(checkpoint.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "repo_id": repo_id,
        "samples": samples,
        "mean_loss": weighted_loss / samples if samples else None,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "limitations": [
            "This is teacher-forced offline loss, not closed-loop policy performance.",
            "Real folding acceptance still requires the separately safety-gated 10+10 rollout protocol.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/openarm-tshirt-fold-validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        output=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
