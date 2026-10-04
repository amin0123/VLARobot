"""Collect synchronized pre-action observations with the official LeRobot writer."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.common import (
    ACTION_NAMES,
    IMAGE_KEY,
    STATE_NAMES,
    config,
    expert_fingerprint,
    json_default,
    write_json,
)
from src.simulation.client import SimulationClient
from src.tasks import sample_task


def features(image_size):
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }


def validate_expert_gate(path, cfg):
    if not Path(path).is_file():
        raise RuntimeError("Run python -m src.control.validate_expert --episodes 100 before collecting.")
    gate = json.loads(Path(path).read_text())
    if (
        gate["episodes"] < cfg["minimum_expert_episodes"]
        or gate["success_rate"] <= cfg["minimum_expert_success"]
    ):
        raise RuntimeError("Expert must exceed 95% success over at least 100 randomized episodes.")
    if gate.get("expert_fingerprint") != expert_fingerprint():
        raise RuntimeError("Expert/environment changed since validation. Repeat the 100-episode gate.")
    return gate


def main():
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--config", default="dataset")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    cfg, env_cfg = config(args.config), config()
    episodes = args.episodes or cfg["episodes"]
    root = args.root or Path(cfg["root"])
    seed_start = args.seed_start if args.seed_start is not None else cfg["seed_start"]
    gate = validate_expert_gate(cfg["expert_gate"], cfg)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists. Select a new --root or use --resume.")
    writer_kwargs = {
        "repo_id": cfg["repo_id"],
        "root": root,
        "rgb_encoder": RGBEncoderConfig(vcodec="h264", pix_fmt="yuv420p", crf=18),
        "image_writer_threads": 4,
        "encoder_threads": 2,
    }
    if args.resume:
        dataset = LeRobotDataset.resume(**writer_kwargs)
    else:
        dataset = LeRobotDataset.create(
            fps=env_cfg["control_hz"],
            features=features(env_cfg["image_size"]),
            robot_type="franka_panda_cartesian",
            **writer_kwargs,
        )
    metadata_path = root / "meta/collection.jsonl"
    previous = (
        [json.loads(line) for line in metadata_path.read_text().splitlines()]
        if metadata_path.exists()
        else []
    )
    if len(previous) != dataset.num_episodes:
        raise RuntimeError(
            "Collection sidecar differs from dataset episode count; inspect the interrupted write before resuming."
        )
    used_seeds = {r["seed"] for r in previous}
    completed = dataset.num_episodes
    started = time.monotonic()
    write_json(
        root / "meta/project.json",
        {
            "environment": env_cfg,
            "collection_config": cfg,
            "expert_gate": gate,
            "observation_timing": "observation before action",
            "action_names": ACTION_NAMES,
            "state_names": STATE_NAMES,
            "seed_start": seed_start,
            "split": cfg["split"],
        },
    )
    try:
        with SimulationClient(gui=args.gui, log_path=root / "simulation.log") as sim:
            with metadata_path.open("a") as metadata:
                for attempt in range(episodes * cfg["max_attempt_factor"]):
                    if completed >= episodes:
                        break
                    seed = seed_start + attempt
                    if seed in used_seeds:
                        continue
                    task = sample_task(completed, seed, cfg["split"])
                    obs, info = sim.reset(seed, task, episode=completed)
                    for _ in range(env_cfg["max_steps"]):
                        action = sim.call("expert_action")
                        next_obs, info = sim.step(action)
                        dataset.add_frame(
                            {
                                **obs,
                                "action": np.asarray(action, dtype=np.float32),
                                "task": task.instruction,
                                "next.done": np.array([info["terminated"] or info["truncated"]]),
                                "next.success": np.array([info["success"]]),
                            }
                        )
                        obs = next_obs
                        if info["terminated"] or info["truncated"]:
                            break
                    if info["success"]:
                        dataset.save_episode()
                        row = {
                            "episode_index": completed,
                            "seed": seed,
                            **task.to_dict(),
                            "success": True,
                            "frames": info["step"],
                            "split": cfg["split"],
                            "initial_configuration": info["initial_configuration"],
                        }
                        metadata.write(json.dumps(row, default=json_default) + "\n")
                        metadata.flush()
                        completed += 1
                        print(
                            f"Saved {completed}/{episodes}: {task.source} → {task.destination}, "
                            f"seed={seed}, frames={info['step']}",
                            flush=True,
                        )
                    else:
                        dataset.clear_episode_buffer()
                        print(f"Discarded failed demonstration seed={seed}", flush=True)
    finally:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer()
        dataset.finalize()
    write_json(
        root / "meta/collection_summary.json",
        {
            "episodes": completed,
            "requested": episodes,
            "wall_seconds": time.monotonic() - started,
            "expert_fingerprint": expert_fingerprint(),
        },
    )
    if completed < episodes:
        raise RuntimeError(
            f"Collected {completed}/{episodes}; investigate expert or increase attempt budget."
        )


if __name__ == "__main__":
    main()
