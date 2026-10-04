"""Replay recorded RGB and actions, export video and a temporal contact sheet."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.common import IMAGE_KEY, config
from src.video import VideoRecorder


def main():
    from lerobot.datasets import LeRobotDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(config("dataset")["root"]))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_preview"))
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()
    ds = LeRobotDataset(
        config("dataset")["repo_id"], root=args.root, episodes=[args.episode], video_backend="pyav"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / f"episode_{args.episode:04d}.mp4"
    selected = set(np.linspace(0, len(ds) - 1, 12, dtype=int).tolist())
    panels = []
    with VideoRecorder(video_path, ds.fps) as video:
        for index in range(len(ds)):
            sample = ds[index]
            rgb = (sample[IMAGE_KEY].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            action = sample["action"].tolist()
            video.add(
                rgb,
                sample["task"],
                f"DEMO {args.episode} | frame {index} | xyz {np.round(action[:3], 3)} | grip {action[6]:.1f}",
            )
            if index in selected:
                panel = Image.new("RGB", (256, 280), "white")
                panel.paste(Image.fromarray(rgb).resize((256, 256)), (0, 0))
                ImageDraw.Draw(panel).text((6, 260), f"frame {index}, t={index / ds.fps:.1f}s", fill="black")
                panels.append(panel)
    sheet = Image.new("RGB", (4 * 256, 3 * 280), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 4) * 256, (i // 4) * 280))
    sheet.save(video_path.with_suffix(".jpg"))
    print(f"Saved {video_path} and {video_path.with_suffix('.jpg')}")
    if args.play:
        import subprocess

        subprocess.run(["ffplay", "-autoexit", str(video_path)], check=True)


if __name__ == "__main__":
    main()
