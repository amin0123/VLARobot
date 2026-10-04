"""Optional terminal jogger for understanding the shared Cartesian action space."""

import numpy as np

from src.simulation.client import SimulationClient
from src.tasks import task_from_instruction


def main():
    print("Terminal jog: w/s +x/-x, a/d +y/-y, r/f +z/-z, o/c open/close. Enter sends; q quits.")
    task = task_from_instruction("Put the red cube in the blue box.")
    with SimulationClient(gui=True) as sim:
        obs, _ = sim.reset(7, task)
        opening = 1.0
        for command in iter(lambda: input("jog> ").strip().lower(), "q"):
            action = np.array([*obs["observation.state"][:6], opening], dtype=np.float32)
            for key in command:
                if key in "wsadrf":
                    axis, sign = {
                        "w": (0, 1),
                        "s": (0, -1),
                        "a": (1, 1),
                        "d": (1, -1),
                        "r": (2, 1),
                        "f": (2, -1),
                    }[key]
                    action[axis] += sign * 0.015
                elif key in "oc":
                    opening = float(key == "o")
                    action[6] = opening
            obs, info = sim.step(action)
            print(
                f"TCP {obs['observation.state'][:3]}, grip={obs['observation.state'][6]:.3f}, success={info['success']}"
            )


if __name__ == "__main__":
    main()
