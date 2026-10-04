"""Run a natural-language command through the learned policy in closed loop."""

import argparse
from pathlib import Path

from src.common import config, write_json
from src.evaluation.rollout import rollout
from src.inference.policy import VLAPolicy
from src.simulation.client import SimulationClient
from src.tasks import task_from_instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--checkpoint", default=config("evaluation")["checkpoint"])
    parser.add_argument(
        "--source", help="Optional metric label for a command outside the controlled templates"
    )
    parser.add_argument("--destination", help="Optional metric label; never sent to the policy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=200000)
    parser.add_argument("--n-action-steps", type=int, default=5)
    parser.add_argument("--video", type=Path, default=Path("outputs/videos/interactive.mp4"))
    args = parser.parse_args()
    task = task_from_instruction(args.instruction, args.source, args.destination)
    policy = VLAPolicy(args.checkpoint, n_action_steps=args.n_action_steps)
    with SimulationClient(gui=not args.headless) as sim:
        row, trace = rollout(sim, policy, task, args.seed, video_path=args.video)
    write_json(args.video.with_suffix(".json"), {"result": row, **trace})
    print(
        f"{'SUCCESS' if row['success'] else 'FAILURE: ' + row['failure_category']} | "
        f"{row['episode_length']} steps | video: {args.video}"
    )


if __name__ == "__main__":
    main()
