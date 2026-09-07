"""Compute teacher-forced ACT loss on a complete-episode validation dataset."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    absolute_error_sum = 0.0
    valid_values = 0
    samples = 0
    started_ns = time.time_ns()
    with torch.no_grad():
        for batch in loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device, non_blocking=device.type == "cuda")
            predicted = policy.predict_action_chunk(batch)
            target = batch["action"]
            valid = ~batch["action_is_pad"].unsqueeze(-1).expand_as(target)
            absolute_error_sum += (predicted - target).abs()[valid].sum().item()
            valid_values += int(valid.sum().item())
            batch_samples = int(batch["action"].shape[0])
            samples += batch_samples

    result = {
        "evaluation_type": "deterministic_validation_action_chunk_mae",
        "checkpoint": str(checkpoint.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "repo_id": repo_id,
        "samples": samples,
        "mean_absolute_action_error": (
            absolute_error_sum / valid_values if valid_values else None
        ),
        "valid_action_values": valid_values,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "limitations": [
            "This is deterministic offline action error, not closed-loop policy performance.",
            "The metric mixes the configured physical units of all 16 action channels.",
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
