"""Validate contact-based teacher over randomized, reproducible episodes."""

import argparse
import time
from pathlib import Path

from src.common import IMAGE_KEY, config, expert_fingerprint, write_json
from src.evaluation.metrics import classify_failure, summarize
from src.simulation.client import SimulationClient
from src.tasks import sample_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("outputs/expert"))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    from src.video import VideoRecorder

    c = config()
    rows = []
    with SimulationClient(gui=args.gui, log_path=args.output / "simulation.log") as sim:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            task = sample_task(episode, seed)
            render = episode < args.videos or args.gui
            obs, info = sim.reset(seed, task, render=render, episode=episode)
            trace, actions = [], []
            video = (
                VideoRecorder(args.output / f"episode_{episode:04d}.mp4", c["control_hz"])
                if episode < args.videos
                else None
            )
            start = time.monotonic()
            try:
                for step in range(c["max_steps"]):
                    state = sim.call("privileged")
                    trace.append(state)
                    action = sim.call("expert_action")
                    actions.append(action)
                    if video:
                        video.add(
                            obs[IMAGE_KEY], task.instruction, f"EXPERT | Episode {episode} | Step {step}"
                        )
                    obs, info = sim.step(action, render=render)
                    if args.debug and step % 10 == 0:
                        print(
                            {
                                "step": step,
                                "tcp": state["tcp"],
                                "cube": state["objects"][task.source],
                                "action": action.tolist(),
                                "ik": info["ik_success"],
                            },
                            flush=True,
                        )
                    if info["terminated"] or info["truncated"]:
                        break
                trace.append(sim.call("privileged"))
                if video:
                    video.add(
                        obs[IMAGE_KEY],
                        task.instruction,
                        "EXPERT | " + ("SUCCESS" if info["success"] else "FAILURE"),
                    )
            finally:
                if video:
                    video.close()
            row = {
                "episode": episode,
                "seed": seed,
                **task.to_dict(),
                "success": info["success"],
                "steps": info["step"],
                "sim_seconds": info["sim_seconds"],
                "wall_seconds": time.monotonic() - start,
                "failure_category": "" if info["success"] else classify_failure(trace, task.to_dict(), c),
                "initial_configuration": info["initial_configuration"],
            }
            rows.append(row)
            write_json(args.output / "episodes.json", rows)
            write_json(
                args.output / "summary.json", {**summarize(rows), "expert_fingerprint": expert_fingerprint()}
            )
            if not row["success"] or args.debug:
                write_json(
                    args.output / f"trace_{episode:04d}.json",
                    {"task": task.to_dict(), "trace": trace, "actions": actions},
                )
            print(
                f"Expert {episode + 1}/{args.episodes}: {'SUCCESS' if row['success'] else row['failure_category']} "
                f"in {row['steps']} steps; cumulative {sum(r['success'] for r in rows)}/{len(rows)}",
                flush=True,
            )
    print(summarize(rows))
    if args.episodes >= 100 and summarize(rows)["success_rate"] <= 0.95:
        raise SystemExit("Expert gate failed: improve the controller before collecting demonstrations.")


if __name__ == "__main__":
    main()
