"""Scene sampling is separate from rendering and can be tested without Isaac Sim."""

import numpy as np

from src.common import COLORS


def sample_layout(cfg, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    centers = np.asarray(list(cfg["container_centers"].values()))
    side = cfg["cube_size"] * scale
    half_bin = cfg["bin_inner_width"] / 2 + cfg["bin_wall_thickness"]
    chosen = []
    positions = {}
    for color in COLORS:
        for _ in range(10000):
            xy = rng.uniform(
                [cfg["cube_spawn_x"][0], cfg["cube_spawn_y"][0]],
                [cfg["cube_spawn_x"][1], cfg["cube_spawn_y"][1]],
            )
            if any(np.linalg.norm(xy - p) < cfg["minimum_cube_distance"] for p in chosen):
                continue
            if np.any(np.all(np.abs(centers - xy) < half_bin + side / 2 + 0.012, axis=1)):
                continue
            chosen.append(xy)
            positions[color] = np.array([*xy, cfg["table_height"] + side / 2 + 0.002])
            break
        else:
            raise RuntimeError("Could not sample a non-overlapping scene; check workspace constraints.")
    return positions


def cube_corners(position, quaternion_wxyz, side):
    from scipy.spatial.transform import Rotation

    q = np.asarray(quaternion_wxyz)
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    offsets = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * side / 2
    return offsets @ rotation.T + np.asarray(position)
