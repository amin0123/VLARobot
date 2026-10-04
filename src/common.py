"""Small configuration and serialization helpers shared by both environments."""

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("red", "green", "blue", "yellow")
RGB = {
    "red": (0.85, 0.035, 0.025),
    "green": (0.035, 0.65, 0.055),
    "blue": (0.035, 0.12, 0.85),
    "yellow": (0.95, 0.72, 0.025),
}
IMAGE_KEY = "observation.images.camera1"
STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "tcp_roll", "tcp_pitch", "tcp_yaw", "gripper_width"] + [
    f"panda_joint{i}" for i in range(1, 8)
]
ACTION_NAMES = [
    "tcp_target_x",
    "tcp_target_y",
    "tcp_target_z",
    "target_roll",
    "target_pitch",
    "target_yaw",
    "gripper_open",
]


def config(name="environment"):
    path = Path(name)
    if not path.is_file():
        path = ROOT / "configs" / f"{name}.yaml"
    return yaml.safe_load(path.read_text())


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=json_default) + "\n")
    temporary.replace(path)


def expert_fingerprint():
    """Invalidate the demonstration gate if physics/control/sampling changes."""
    import hashlib

    digest = hashlib.sha256()
    for name in (
        "configs/environment.yaml",
        "src/simulation/environment.py",
        "src/simulation/objects.py",
        "src/simulation/robot.py",
        "src/control/scripted_policy.py",
        "src/control/ik_controller.py",
        "src/evaluation/metrics.py",
    ):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()
