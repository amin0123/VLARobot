"""Summarize recorded outcomes and create contact sheets from actual rollout videos."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from src.common import write_json


def contact_sheet(video_path, output):
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        count = stream.frames
        if count <= 0:
            raise ValueError(f"Video has no frame count: {video_path}")
        selected = set(np.linspace(0, count - 1, 12, dtype=int).tolist())
        panels = []
        for index, frame in enumerate(container.decode(stream)):
            if index in selected:
                rgb = Image.fromarray(frame.to_ndarray(format="rgb24"))
                rgb.thumbnail((256, 304))
                panel = Image.new("RGB", (256, 324), "white")
                panel.paste(rgb, (0, 0))
                ImageDraw.Draw(panel).text((5, 308), f"frame {index}", fill="black")
                panels.append(panel)
    sheet = Image.new("RGB", (1024, 972), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 4) * 256, (i // 4) * 324))
    sheet.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--sheets", type=int, default=4)
    args = parser.parse_args()
    with (args.root / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter(row["failure_category"] for row in rows if row["success"] != "True")
    summaries = []
    sheet_dir = args.root / "contact_sheets"
    sheet_dir.mkdir(exist_ok=True)
    for row in rows:
        name = f"{row['condition']}_{int(row['episode']):04d}"
        data = json.loads((args.root / "traces" / f"{name}.json").read_text())
        target = row["source"]
        trace = data["trace"]
        positions = np.array([t["objects"][target] for t in trace])
        tcp = np.array([t["tcp"] for t in trace])
        summaries.append(
            {
                "condition": row["condition"],
                "episode": int(row["episode"]),
                "success": row["success"] == "True",
                "failure_category": row["failure_category"],
                # Our teacher's initial approach target is directly above the
                # cube. This is a post-hoc BC diagnostic, never a policy input.
                "initial_action_xy_error_m": float(
                    np.linalg.norm(np.asarray(data["actions"][0])[:2] - positions[0, :2])
                ),
                "nearest_tcp_to_target_m": float(np.linalg.norm(tcp - positions, axis=1).min()),
                "target_max_lift_m": float(positions[:, 2].max() - positions[0, 2]),
                "target_final_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])),
                "tcp_path_length_m": float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum()),
            }
        )
    initial_action_by_condition = {}
    for condition in dict.fromkeys(row["condition"] for row in summaries):
        values = [r["initial_action_xy_error_m"] for r in summaries if r["condition"] == condition]
        initial_action_by_condition[condition] = {
            "episodes": len(values),
            "mean_xy_error_m": float(np.mean(values)),
            "median_xy_error_m": float(np.median(values)),
        }
    write_json(
        args.root / "failure_analysis.json",
        {
            "episodes": len(rows),
            "failure_counts": dict(counts),
            "initial_action_by_condition": initial_action_by_condition,
            "episode_diagnostics": summaries,
            "interpretation": "Observable outcome heuristics; these do not prove language misunderstanding or collision causes.",
        },
    )
    # Prefer failures for inspection, then include successful examples if available.
    videos = sorted((row for row in rows if row["video"]), key=lambda r: r["success"] == "True")
    for row in videos[: args.sheets]:
        name = f"{row['condition']}_{int(row['episode']):04d}.jpg"
        contact_sheet(Path(row["video"]), sheet_dir / name)
    lines = ["| Failure category | Episodes |", "|---|---:|"]
    lines += [f"| {category} | {count} |" for category, count in counts.most_common()]
    (args.root / "failure_analysis.md").write_text(
        "# Observed failure categories\n\n"
        + "\n".join(lines)
        + "\n\nLabels are conservative outcome heuristics. Inspect video and action traces before assigning a cause.\n"
    )
    print(
        json.dumps(
            {"episodes": len(rows), "failure_counts": dict(counts), "contact_sheets": str(sheet_dir)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
