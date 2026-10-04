"""Physical success and conservative failure diagnostics; never policy inputs."""

from math import sqrt

import numpy as np

from src.simulation.objects import cube_corners


def wilson_interval(successes, total, z=1.959963984540054):
    if total < 1:
        return (None, None)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def contained(position, orientation, side, destination_xy, cfg):
    corners = cube_corners(position, orientation, side)
    half = cfg["bin_inner_width"] / 2
    xy_inside = np.all(np.abs(corners[:, :2] - destination_xy) <= half - 0.001)
    floor_top = cfg["table_height"] + cfg["bin_floor_thickness"]
    # PhysX contact offsets can leave the resting body slightly above the floor.
    resting_height = floor_top - 0.003 <= corners[:, 2].min() <= floor_top + 0.006
    below_rim = corners[:, 2].max() < floor_top + cfg["bin_wall_height"] + 0.002
    return bool(xy_inside and resting_height and below_rim)


def summarize(rows):
    n = len(rows)
    k = sum(bool(r["success"]) for r in rows)
    low, high = wilson_interval(k, n)
    failures = {}
    for row in rows:
        if not row["success"]:
            category = row.get("failure_category", "unclassified")
            failures[category] = failures.get(category, 0) + 1
    return {
        "episodes": n,
        "successes": k,
        "success_rate": k / n if n else None,
        "wilson_95_low": low,
        "wilson_95_high": high,
        "failure_categories": failures,
    }


def classify_failure(trace, task, cfg):
    if not trace:
        return "unclassified"
    target = task["source"]
    final = trace[-1]
    if any(t.get("invalid_action", False) for t in trace):
        return "invalid_action"
    # Infer observable events; these do not prove language misunderstanding.
    lifted = {c for t in trace for c, p in t["objects"].items() if p[2] > 0.09}
    if lifted and target not in lifted:
        return "selected_wrong_object"
    if target in lifted:
        for color, xy in cfg["container_centers"].items():
            if (
                color != task["destination"]
                and np.linalg.norm(np.array(final["objects"][target])[:2] - xy) < 0.05
            ):
                return "selected_wrong_destination"
        if final["objects"][target][2] < 0.05:
            return "dropped_or_misplaced_object"
        return "placement_timeout"
    nearest = min(np.linalg.norm(np.array(t["tcp"]) - t["objects"][target]) for t in trace)
    if nearest < 0.055:
        return "failed_grasp"
    return "failed_to_approach_target"
