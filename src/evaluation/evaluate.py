"""Paired generalization evaluations with exact per-episode outcomes and Wilson CIs."""

import argparse
import csv
import json
from pathlib import Path

from src.common import config, json_default, write_json
from src.evaluation.metrics import summarize
from src.evaluation.rollout import rollout
from src.inference.policy import VLAPolicy
from src.simulation.client import SimulationClient
from src.tasks import Task, sample_task


def save_results(output, rows, checkpoint):
    # Persist after every episode, including failures; interrupted jobs retain evidence.
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "initial_configuration": json.dumps(row["initial_configuration"], default=json_default),
                }
            )
    groups = {
        condition: summarize([r for r in rows if r["condition"] == condition])
        for condition in dict.fromkeys(r["condition"] for r in rows)
    }
    write_json(output / "summary.json", {"checkpoint": str(checkpoint), "conditions": groups})
    table = ["| Condition | Successes / episodes | Success rate | Wilson 95% CI |", "|---|---:|---:|---:|"]
    for name, stats in groups.items():
        table.append(
            f"| {name} | {stats['successes']} / {stats['episodes']} | {stats['success_rate']:.1%} | "
            f"{stats['wilson_95_low']:.1%}–{stats['wilson_95_high']:.1%} |"
        )
    (output / "report.md").write_text(
        f"# Measured evaluation\n\nCheckpoint: `{checkpoint}`\n\n" + "\n".join(table) + "\n"
    )
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="evaluation")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--videos", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    cfg = config(args.config)
    checkpoint = args.checkpoint or cfg["checkpoint"]
    output = args.output or Path(cfg["output_dir"])
    if (output / "results.csv").exists():
        raise FileExistsError(f"{output}/results.csv already exists. Select --output for a new experiment.")
    conditions = args.conditions or cfg["conditions"]
    allowed = {"train_replay", "position", "language", "composition", "lighting", "camera", "scale"}
    if set(conditions) - allowed:
        raise ValueError(f"Unknown conditions: {set(conditions) - allowed}")
    episodes = args.episodes or cfg["episodes_per_condition"]
    videos = args.videos if args.videos is not None else cfg["videos_per_condition"]
    if episodes < 1 or videos < 0:
        raise ValueError("episodes must be positive and videos nonnegative")
    video_dir = Path(cfg["video_dir"]) / output.name
    train_rows = []
    if "train_replay" in conditions:
        train_rows = [
            json.loads(line)
            for line in Path(config("training")["dataset_root"], "meta/collection.jsonl")
            .read_text()
            .splitlines()
        ]
        if episodes > len(train_rows):
            raise ValueError("Do not repeat training seeds to inflate the train_replay sample size.")
    policy = VLAPolicy(checkpoint, n_action_steps=cfg["n_action_steps"])
    rows = []
    with SimulationClient(gui=args.gui, log_path=output / "simulation.log") as sim:
        for condition in conditions:
            for episode in range(episodes):
                seed = cfg["seed_start"] + episode  # paired scene seeds across perturbation conditions
                task = sample_task(episode, seed, condition)
                if condition == "train_replay":
                    meta = train_rows[episode]
                    seed = meta["seed"]
                    task = Task(
                        **{key: meta[key] for key in ("source", "destination", "instruction", "template_id")}
                    )
                video = video_dir / f"{condition}_{episode:04d}.mp4" if episode < videos else None
                row, trace = rollout(
                    sim,
                    policy,
                    task,
                    seed,
                    condition,
                    max_steps=args.max_steps or cfg["max_steps"],
                    video_path=video,
                    episode=episode,
                )
                rows.append(row)
                write_json(output / "traces" / f"{condition}_{episode:04d}.json", trace)
                save_results(output, rows, checkpoint)
                print(
                    f"{condition} {episode + 1}/{episodes}: {'SUCCESS' if row['success'] else row['failure_category']} "
                    f"({row['episode_length']} steps)",
                    flush=True,
                )
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
