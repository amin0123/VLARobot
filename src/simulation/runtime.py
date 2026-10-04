"""Discover an existing standalone Isaac Sim installation; never install one."""

import os
from pathlib import Path


def simulator_python():
    explicit = os.environ.get("VLAROBOT_SIM_PYTHON")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"VLAROBOT_SIM_PYTHON does not exist: {path}")
        return path
    roots = [os.environ.get("ISAAC_SIM_PATH"), os.environ.get("ISAAC_PATH"), str(Path.home() / "isaacsim")]
    for root in roots:
        if root and (path := Path(root).expanduser() / "python.sh").is_file():
            return path
    raise FileNotFoundError(
        "Existing Isaac Sim not found. Set ISAAC_SIM_PATH to its installation directory "
        "(containing python.sh). This project does not install Isaac Sim or ROS."
    )


if __name__ == "__main__":
    print(simulator_python())
