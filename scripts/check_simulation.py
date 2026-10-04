"""Real simulator checks for observation synchronization and rendering side effects."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import IMAGE_KEY, write_json
from src.simulation.client import SimulationClient
from src.tasks import task_from_instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task = task_from_instruction("Put the red cube in the blue box.")
    with SimulationClient(gui=args.gui, log_path=args.output_dir / "simulation-check.log") as sim:
        obs, _ = sim.reset(7, task)
        assert obs[IMAGE_KEY].shape == (256, 256, 3)
        assert obs[IMAGE_KEY].dtype == np.uint8
        assert obs["observation.state"].shape == (14,)
        assert np.isfinite(obs["observation.state"]).all()
        assert obs[IMAGE_KEY].std() > 10, "Blank or nearly constant RGB frame"
        Image.fromarray(obs[IMAGE_KEY]).save(args.output_dir / "scene.png")
        again = sim.call("observe", render=True)
        capture_drift = float(np.max(np.abs(again["observation.state"] - obs["observation.state"])))
        assert capture_drift < 1e-5, f"Rendering advanced robot state: {capture_drift}"
        initial_rgb = obs[IMAGE_KEY].astype(float)
        actions, states = [], []
        for _ in range(12):
            action = sim.call("expert_action")
            obs, _ = sim.step(action, render=False)
            actions.append(action)
            states.append(obs["observation.state"])
        sim.reset(7, task)
        errors = []
        for action, expected in zip(actions, states, strict=True):
            obs, _ = sim.step(action, render=True)
            errors.append(float(np.max(np.abs(obs["observation.state"] - expected))))
        Image.fromarray(obs[IMAGE_KEY]).save(args.output_dir / "scene_after_actions.png")
        visual_change = float(np.abs(initial_rgb - obs[IMAGE_KEY]).mean())
        assert max(errors) < 1e-4, f"Rendering changed physics trajectory: {max(errors)}"
        assert visual_change > 0.5, "Image did not change after physical movement"
        report = {
            "passed": True,
            "gui": args.gui,
            "image_shape": list(obs[IMAGE_KEY].shape),
            "state_shape": list(obs["observation.state"].shape),
            "capture_state_drift": capture_drift,
            "render_vs_no_render_max_state_error": max(errors),
            "image_mean_absolute_change": visual_change,
            "asset_path": sim.ready["asset_path"],
        }
        write_json(args.output_dir / "simulation_check.json", report)
        print(report)


if __name__ == "__main__":
    main()
