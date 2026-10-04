"""Measure immediate action error and input sensitivity on recorded observations.

These are offline diagnostics, not simulator success or proof of language grounding.
Counterfactuals use the same flow-noise seed as the original observation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common import IMAGE_KEY, write_json
from src.inference.policy import VLAPolicy


def main():
    from lerobot.datasets import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", default="datasets/splits/train")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("outputs/action_diagnostics.json"))
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("samples must be positive")
    ds = LeRobotDataset("local/vlarobot_pick_place", root=args.root, video_backend="pyav")
    policy = VLAPolicy(args.checkpoint)
    indices = np.linspace(0, len(ds) - 1, min(args.samples, len(ds)), dtype=int)
    rows = []

    def predict(inputs, seed):
        policy.reset(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = policy.model.predict_action_chunk(policy.pre(inputs))
            return policy.post(chunk[:, 0]).squeeze(0).cpu().float().numpy()

    for index in indices:
        sample = ds[int(index)]
        inputs = {
            IMAGE_KEY: sample[IMAGE_KEY],
            "observation.state": sample["observation.state"],
            "task": sample["task"],
        }
        seed = 42 + int(index)
        original = predict(inputs, seed)
        blank_image = predict({**inputs, IMAGE_KEY: torch.zeros_like(sample[IMAGE_KEY])}, seed)
        no_language = predict({**inputs, "task": ""}, seed)
        target = sample["action"].numpy()
        state = sample["observation.state"].numpy()
        rows.append(
            {
                "frame_index": int(index),
                "episode_index": int(sample["episode_index"]),
                "task": sample["task"],
                "flow_noise_seed": seed,
                "expert_action": target.tolist(),
                "predicted_action": original.tolist(),
                "xyz_error_m": float(np.linalg.norm(original[:3] - target[:3])),
                "gripper_absolute_error": float(abs(original[6] - target[6])),
                "state_hold_xyz_error_m": float(np.linalg.norm(state[:3] - target[:3])),
                "blank_image_xyz_change_m": float(np.linalg.norm(blank_image[:3] - original[:3])),
                "empty_language_xyz_change_m": float(np.linalg.norm(no_language[:3] - original[:3])),
            }
        )
    measures = [key for key in rows[0] if key.endswith("_m") or key == "gripper_absolute_error"]
    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.root,
        "samples": len(rows),
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in measures},
        "median": {key: float(np.median([row[key] for row in rows])) for key in measures},
        "interpretation": (
            "First predicted action versus expert target on recorded states. Blank-image and empty-language "
            "changes test sensitivity only; they are out-of-distribution inputs and do not prove correct "
            "object localization or language grounding. No actions are executed in this diagnostic."
        ),
        "rows": rows,
    }
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
