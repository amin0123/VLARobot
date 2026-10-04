"""Paired generalization evaluations with exact per-episode outcomes and Wilson CIs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.common import config, expert_fingerprint, json_default, write_json
from src.evaluation.metrics import summarize
from src.evaluation.rollout import rollout
from src.inference.policy import VLAPolicy
from src.simulation.client import SimulationClient
from src.tasks import Task, evaluation_task


def save_results(output, rows, checkpoint):
    # Persist after every episode, including failures; interrupted jobs retain evidence.
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "initial_configuration": json.dumps(row["initial_configuration"], default=json_default),
                }
            )
    temporary.replace(output / "results.csv")
    write_json(output / "results.json", rows)
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
    parser.add_argument(
        "--training-root", type=Path, help="Override checkpoint dataset location for train replay"
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--videos", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--instruction-sampling", choices=("seen", "templates"))
    args = parser.parse_args()
    cfg = config(args.config)
    if args.seed_start is not None:
        cfg["seed_start"] = args.seed_start
    if args.n_action_steps is not None:
        cfg["n_action_steps"] = args.n_action_steps
    checkpoint = args.checkpoint or cfg["checkpoint"]
    instruction_sampling = args.instruction_sampling or cfg.get("instruction_sampling", "seen")
    training_config_path = Path(checkpoint, "train_config.json")
    checkpoint_training_root = (
        json.loads(training_config_path.read_text())["dataset"]["root"]
        if training_config_path.is_file()
        else config("training")["dataset_root"]
    )
    training_root = args.training_root or Path(checkpoint_training_root)
    output = args.output or Path(cfg["output_dir"])
    if (output / "results.csv").exists():
        raise FileExistsError(f"{output}/results.csv already exists. Select --output for a new experiment.")
    conditions = args.conditions or cfg["conditions"]
    allowed = {"train_replay", "position", "language", "composition", "lighting", "camera", "scale"}
    if set(conditions) - allowed:
        raise ValueError(f"Unknown conditions: {set(conditions) - allowed}")
    if len(conditions) != len(set(conditions)):
        raise ValueError("Each condition must appear once; repeated seeds are not independent trials.")
    episodes = args.episodes if args.episodes is not None else cfg["episodes_per_condition"]
    videos = args.videos if args.videos is not None else cfg["videos_per_condition"]
    max_steps = args.max_steps if args.max_steps is not None else cfg["max_steps"]
    if episodes < 1 or videos < 0 or max_steps < 1:
        raise ValueError("episodes/max-steps must be positive and videos nonnegative")
    with Path(checkpoint, "model.safetensors").open("rb") as weights:
        checkpoint_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
    write_json(
        output / "run_config.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "training_root": str(training_root),
            "instruction_sampling": instruction_sampling,
            "protocol_version": 2,
            "conditions": conditions,
            "episodes_per_condition": episodes,
            "seed_start": cfg["seed_start"],
            "n_action_steps": cfg["n_action_steps"],
            "max_steps": min(max_steps, config()["max_steps"]),
            "videos_per_condition": videos,
            "environment": config(),
            "environment_controller_fingerprint": expert_fingerprint(),
            "initial_configuration_note": (
                "Results contain settled object poses. Camera/light/scale perturbations are reproduced "
                "from each recorded seed and condition using the environment configuration here."
            ),
        },
    )
    video_dir = Path(cfg["video_dir"]) / output.name
    train_rows = []
    if "train_replay" in conditions or instruction_sampling == "seen":
        train_rows = [
            json.loads(line) for line in (training_root / "meta/collection.jsonl").read_text().splitlines()
        ]
        if "train_replay" in conditions and episodes > len(train_rows):
            raise ValueError("Do not repeat training seeds to inflate the train_replay sample size.")
    policy = VLAPolicy(checkpoint, n_action_steps=cfg["n_action_steps"])
    rows = []
    with SimulationClient(gui=args.gui, log_path=output / "simulation.log") as sim:
        for condition in conditions:
            retained_outcomes = set()
            for episode in range(episodes):
                seed = cfg["seed_start"] + episode  # paired scene seeds across perturbation conditions
                task = evaluation_task(episode, seed, condition, train_rows, instruction_sampling)
                if condition == "train_replay":
                    meta = train_rows[episode]
                    seed = meta["seed"]
                    task = Task(
                        **{key: meta[key] for key in ("source", "destination", "instruction", "template_id")}
                    )
                # Capture candidates so the first success and first failure can
                # be retained even when they occur after the initial examples.
                video = (
                    video_dir / f"{condition}_{episode:04d}.mp4"
                    if videos and (episode < videos or len(retained_outcomes) < 2)
                    else None
                )
                row, trace = rollout(
                    sim,
                    policy,
                    task,
                    seed,
                    condition,
                    max_steps=max_steps,
                    video_path=video,
                    episode=episode,
                )
                if video:
                    if episode >= videos and row["success"] in retained_outcomes:
                        video.unlink()
                        row["video"] = ""
                    else:
                        retained_outcomes.add(row["success"])
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
