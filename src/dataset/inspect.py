"""Inspect a local LeRobot dataset without starting Isaac Sim."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.common import config, write_json


def inspect_dataset(root):
    from lerobot.datasets import LeRobotDataset

    cfg = config("dataset")
    ds = LeRobotDataset(cfg["repo_id"], root=root, video_backend="pyav")
    rows = [json.loads(line) for line in (Path(root) / "meta/collection.jsonl").read_text().splitlines()]
    sample = ds[0]
    result = {
        "root": str(root),
        "episodes": ds.num_episodes,
        "frames": ds.num_frames,
        "fps": ds.fps,
        "features": ds.features,
        "sample_shapes": {k: list(v.shape) for k, v in sample.items() if hasattr(v, "shape")},
        "instructions": sorted({r["instruction"] for r in rows}),
        "source_destination_pairs": sorted({f"{r['source']} → {r['destination']}" for r in rows}),
        "episode_duration_s": {
            "min": min(r["frames"] for r in rows) / ds.fps,
            "max": max(r["frames"] for r in rows) / ds.fps,
            "mean": float(np.mean([r["frames"] for r in rows])) / ds.fps,
        },
        "unique_seeds": len({r["seed"] for r in rows}),
        "all_saved_episodes_successful": all(r["success"] for r in rows),
    }
    if result["unique_seeds"] != ds.num_episodes:
        raise ValueError("Duplicate seeds detected")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(config("dataset")["root"]))
    args = parser.parse_args()
    report = inspect_dataset(args.root)
    write_json(args.root / "meta/inspection.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
