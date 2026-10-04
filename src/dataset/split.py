"""Materialize disjoint episode splits with LeRobot's official split utility."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.common import config, json_default, write_json


def main():
    from lerobot.datasets import LeRobotDataset, split_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(config("dataset")["root"]))
    parser.add_argument("--output", type=Path, default=Path("datasets/splits"))
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; choose a new output directory.")
    ds = LeRobotDataset(config("dataset")["repo_id"], root=args.root, video_backend="pyav")
    rows = [json.loads(line) for line in (args.root / "meta/collection.jsonl").read_text().splitlines()]
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")
    rng = np.random.default_rng(args.seed)
    # Stratify by canonical pair; keep at least one training example for each pair.
    groups = {}
    for row in rows:
        groups.setdefault((row["source"], row["destination"]), []).append(row["episode_index"])
    candidates = []
    for indices in groups.values():
        shuffled = rng.permutation(indices).tolist()
        candidates.extend(shuffled[1:])
    target = max(1, round(ds.num_episodes * args.validation_fraction))
    if len(candidates) < target:
        raise ValueError(
            "Too few demonstrations to hold out validation while retaining every task in training."
        )
    validation = sorted(rng.choice(candidates, size=target, replace=False).tolist())
    train = sorted(set(range(ds.num_episodes)) - set(validation))
    indices = {"train": train, "validation": validation}
    split_dataset(ds, indices, output_dir=args.output)
    for name, selected in indices.items():
        with (args.output / name / "meta/collection.jsonl").open("w") as stream:
            for new_index, old_index in enumerate(selected):
                row = {
                    **rows[old_index],
                    "original_episode_index": old_index,
                    "episode_index": new_index,
                    "split": name,
                }
                stream.write(json.dumps(row, default=json_default) + "\n")
    manifest = {
        "source": str(args.root),
        "seed": args.seed,
        "episodes": indices,
        "train_seeds": [rows[i]["seed"] for i in train],
        "validation_seeds": [rows[i]["seed"] for i in validation],
    }
    assert not set(manifest["train_seeds"]) & set(manifest["validation_seeds"])
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
