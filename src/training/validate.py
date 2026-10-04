"""Held-out flow-matching loss is a diagnostic, not a manipulation success metric."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common import write_json
from src.inference.policy import VLAPolicy


def main():
    from lerobot.datasets import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="datasets/splits/validation")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("outputs/validation.json"))
    args = parser.parse_args()
    torch.manual_seed(42)
    policy = VLAPolicy(args.checkpoint)
    cfg = policy.model.config
    # Use the held-out dataset's images/actions, but the TRAINING checkpoint's normalizer.
    ds = LeRobotDataset(
        "local/vlarobot_pick_place_validation",
        root=args.root,
        video_backend="pyav",
        delta_timestamps={"action": [i / 10 for i in range(cfg.chunk_size)]},
    )
    indices = np.linspace(0, len(ds) - 1, min(args.samples, len(ds)), dtype=int)
    losses = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in indices:
            batch = policy.pre(ds[int(index)])
            loss, _ = policy.model(batch)
            losses.append(float(loss))
    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.root,
        "samples": len(losses),
        "mean_flow_matching_loss": float(np.mean(losses)),
        "std_flow_matching_loss": float(np.std(losses)),
        "seed": 42,
        "normalization_source": "training checkpoint",
        "losses": losses,
        "interpretation": "Noise-prediction/flow loss on held-out demonstrations; not closed-loop success.",
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
