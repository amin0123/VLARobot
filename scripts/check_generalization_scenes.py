"""Check test-scene rendering and physical solvability with the labeled teacher.

This is an environment integration check, never a learned-policy success score.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import IMAGE_KEY, config, write_json
from src.simulation.client import SimulationClient
from src.tasks import sample_task


def main():
    output = ROOT / "outputs/checks/generalization_scenes"
    output.mkdir(parents=True, exist_ok=True)
    cfg = config()
    rows = []
    baseline_rgb = None
    cases = [
        (condition, 350000)
        for condition in ("position", "language", "composition", "lighting", "camera", "scale")
    ] + [("scale", 350001), ("position", 350000), ("scale", 350002), ("position", 350000)]
    with SimulationClient(log_path=output / "simulation.log") as sim:
        for episode, (condition, seed) in enumerate(cases):
            case_id = f"{episode:02d}_{condition}"
            task = sample_task(0, seed, condition)
            obs, info = sim.reset(seed, task, condition=condition, episode=episode)
            initial = sim.call("privileged")
            rgb = obs[IMAGE_KEY]
            assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
            assert float(rgb.std()) > 5, "Camera is blank or nearly constant"
            assert obs["observation.state"].shape == (14,)
            assert all(
                abs(position[2] - initial["cube_size"] / 2) < 0.0003
                for position in initial["objects"].values()
            ), "Settled body height does not match the requested collision scale"
            if baseline_rgb is None:
                baseline_rgb = rgb.copy()
            Image.fromarray(rgb).save(output / f"{case_id}_initial.png")
            # Rendering was checked above. The privileged teacher needs no images
            # during motion, so skip unnecessary captures in this integration check.
            for _ in range(cfg["max_steps"]):
                _, info = sim.step(sim.call("expert_action"), render=False)
                if info["terminated"] or info["truncated"]:
                    break
            final = sim.call("observe", render=True)
            Image.fromarray(final[IMAGE_KEY]).save(output / f"{case_id}_final.png")
            row = {
                "condition": condition,
                "case_id": case_id,
                "seed": seed,
                "controller": "scripted_expert",
                "task": task.to_dict(),
                "success": info["success"],
                "steps": info["step"],
                "initial_configuration": initial,
                "initial_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "rgb_mean_absolute_difference_from_position": float(
                    np.abs(rgb.astype(float) - baseline_rgb).mean()
                ),
            }
            rows.append(row)
            write_json(output / "results.json", rows)
            print(
                f"EXPERT SCENE CHECK {condition}: success={row['success']}, steps={row['steps']}", flush=True
            )
    if not all(row["success"] for row in rows):
        raise RuntimeError("A test scene failed the teacher smoke check; inspect the saved images/logs.")
    print("All scene integration checks passed. These are expert results, not VLA performance.")


if __name__ == "__main__":
    main()
