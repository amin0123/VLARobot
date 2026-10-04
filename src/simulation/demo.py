"""Launch a rendered scene and run the transparent scripted teacher."""

import argparse
from pathlib import Path

from PIL import Image

from src.common import IMAGE_KEY, config
from src.simulation.client import SimulationClient
from src.tasks import task_from_instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--instruction", default="Put the red cube in the blue box.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--snapshot", default="outputs/scene.png")
    args = parser.parse_args()
    task = task_from_instruction(args.instruction)
    with SimulationClient(gui=not args.headless) as sim:
        obs, info = sim.reset(args.seed, task)
        path = Path(args.snapshot)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(obs[IMAGE_KEY]).save(path)
        print(
            f"Rendered {obs[IMAGE_KEY].shape}; state {obs['observation.state'].shape}. Saved {path}",
            flush=True,
        )
        for step in range(min(args.steps, config()["max_steps"])):
            action = sim.call("expert_action")
            obs, info = sim.step(action)
            if step % 10 == 0:
                print(f"step={step} success={info['success']} ik={info['ik_success']}", flush=True)
            if info["terminated"] or info["truncated"]:
                break
        Image.fromarray(obs[IMAGE_KEY]).save(path.with_stem(path.stem + "_final"))
        print(f"Expert outcome: {info['success']}; simulated duration {info['sim_seconds']} s")


if __name__ == "__main__":
    main()
